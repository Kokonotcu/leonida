// Opt-in PresentX slot routing and a windowless first-frame capture (docs/d3d12_vulkan.md section 4.5).
//
// The queue proxy (d3d12_xs_device.cpp) forwards only a slot named by LEONIDA_PRESENTX_SLOT, _SUSPEND_SLOT or
// _RESUME_SLOT here. Without LEONIDA_CAPTURE a routed present is accepted and dropped: there is no window and no
// swapchain, so this path must not touch the planes at all. With LEONIDA_CAPTURE the plane array is validated and each
// plane's first resource is copied through the queue that presented and written as a PPM.
#include "presentx_capture.h"

#include "d3d12_object_tags.h"
#include "xbox_present_types.h"

#include <windows.h>

#include <atomic>
#include <cerrno>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace rt::d3d12 {

namespace {

// Returns 0 when `name` is unset or empty. A value that is not a queue slot in [19, 127] is logged once and treated
// as unset: the proxy never routes a public ID3D12CommandQueue method and never indexes outside its own table.
unsigned env_queue_slot(const char* name) {
    char value[32] = {};
    const DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    if (length == 0) return 0;
    if (length >= sizeof(value)) {
        std::fprintf(stderr, "[rt] presentx: %s is too long for a queue slot; not routed\n", name);
        return 0;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long slot = std::strtoul(value, &end, 0);
    if (errno != 0 || end == value || *end != '\0' || slot < 19 || slot > 127) {
        std::fprintf(stderr, "[rt] presentx: %s=%s is not a queue slot in 19..127; not routed\n", name, value);
        return 0;
    }
    return static_cast<unsigned>(slot);
}

std::string env_string(const char* name) {
    const DWORD length = GetEnvironmentVariableA(name, nullptr, 0);
    if (length == 0) return {};
    std::string value(length, '\0');
    const DWORD written = GetEnvironmentVariableA(name, value.data(), length);
    if (written == 0 || written >= length) return {};
    value.resize(written);
    return value;
}

// A directory (and its parents) by CreateDirectoryA, without shell32. A path that already exists is success.
bool ensure_directory(const std::string& directory) {
    if (directory.empty()) return false;
    if (CreateDirectoryA(directory.c_str(), nullptr)) return true;
    const DWORD error = GetLastError();
    if (error == ERROR_ALREADY_EXISTS) return true;
    const std::size_t slash = directory.find_last_of("\\/");
    if (slash == std::string::npos || slash == 0) return false;
    if (!ensure_directory(directory.substr(0, slash))) return false;
    return CreateDirectoryA(directory.c_str(), nullptr) != FALSE || GetLastError() == ERROR_ALREADY_EXISTS;
}

// Every byte in [address, address + bytes) is readable, committed memory: no PAGE_NOACCESS, no guard page, no hole.
bool readable_range(const void* address, std::size_t bytes) {
    if (!address) return false;
    if (bytes == 0) return true;
    const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(address);
    if (bytes > UINTPTR_MAX - start) return false;
    const std::uintptr_t end = start + bytes;
    std::uintptr_t at = start;
    while (at < end) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<const void*>(at), &info, sizeof(info)) == 0) return false;
        if (info.State != MEM_COMMIT) return false;
        if ((info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) != 0) return false;
        const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(info.BaseAddress);
        if (info.RegionSize > UINTPTR_MAX - base) return false;
        const std::uintptr_t region_end = base + info.RegionSize;
        if (region_end <= at) return false;
        at = region_end;
    }
    return true;
}

bool executable_pointer(const void* address) {
    if (!readable_range(address, sizeof(void*))) return false;
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(address, &info, sizeof(info)) == 0) return false;
    const DWORD protection = info.Protect & 0xff;
    return protection == PAGE_EXECUTE || protection == PAGE_EXECUTE_READ || protection == PAGE_EXECUTE_READWRITE ||
           protection == PAGE_EXECUTE_WRITECOPY;
}

// Every accepted format is one 32-bit texel: BGRA8/RGBA8 as four bytes, HDR10 as a packed R10G10B10A2 word.
constexpr UINT kBytesPerTexel = 4;

enum class PlaneFormat { Bgra8, Rgba8, Rgb10a2 };

const char* format_name(PlaneFormat format) {
    switch (format) {
        case PlaneFormat::Bgra8: return "BGRA";
        case PlaneFormat::Rgba8: return "RGBA";
        case PlaneFormat::Rgb10a2: return "R10G10B10A2";
    }
    return "unknown";
}

// The caller's structures are its own; this side reads only the fields it names, after the range check.
struct CapturePlane {
    ID3D12Resource* resource = nullptr;
    D3D12_RESOURCE_DESC desc{};
    PlaneFormat format = PlaneFormat::Bgra8;
};

// A mis-routed slot can call this path at frame rate with plausible-looking arguments. Each distinct refusal line is
// printed at most once per 250 ms; the same line from repeated mis-routed calls is counted and dropped. Distinct
// failures (null plane array, foreign object, no ALLOW_DISPLAY, ...) still get their first line, so the throttle does
// not hide a diagnosis. Only refusal lines are throttled, never the "wrote ..." success line.
constexpr ULONGLONG kRefusalMinIntervalMs = 250;
constexpr std::size_t kRefusalEntries = 32;

struct RefusalEntry {
    std::uint64_t hash = 0;
    ULONGLONG last_ms = 0;
    bool used = false;
};

void log_refusal(const char* format, ...) {
    char line[384] = {};
    va_list args;
    va_start(args, format);
    std::vsnprintf(line, sizeof(line), format, args);
    va_end(args);

    std::uint64_t hash = 1469598103934665603ull;  // FNV-1a
    for (const char* at = line; *at; ++at) {
        hash ^= static_cast<unsigned char>(*at);
        hash *= 1099511628211ull;
    }

    static std::mutex mutex;
    static RefusalEntry entries[kRefusalEntries];
    static unsigned suppressed = 0;
    unsigned dropped = 0;
    bool emit = false;
    {
        std::lock_guard lock(mutex);
        const ULONGLONG now = GetTickCount64();
        RefusalEntry& entry = entries[hash % kRefusalEntries];
        if (entry.used && entry.hash == hash && now - entry.last_ms < kRefusalMinIntervalMs) {
            ++suppressed;
        } else {
            entry.used = true;
            entry.hash = hash;
            entry.last_ms = now;
            dropped = suppressed;
            suppressed = 0;
            emit = true;
        }
    }
    if (!emit) return;
    if (dropped) std::fprintf(stderr, "[rt] presentx: %u refusal line(s) suppressed\n", dropped);
    std::fprintf(stderr, "%s\n", line);
}

void refuse(const char* why) {
    log_refusal("[rt] presentx: PresentX refused: %s; E_INVALIDARG", why);
}

void refuse_plane(UINT plane_index, const char* why) {
    log_refusal("[rt] presentx: PresentX plane %u refused: %s; E_INVALIDARG", plane_index, why);
}

// A resource is what the proxy can pass on safely: readable COM memory whose QueryInterface, GetDesc (ID3D12Resource
// slot 10) and GetHeapProperties (slot 14) are executable, and which answers the layer's private Resource tag. The tag
// check is what makes the object ours: an executable vtable alone can belong to another object, and calling its
// GetDesc would leave `desc` uninitialised. A host runtime resource does not answer kLeonidaResourceTag and is
// refused. "Committed" is not queryable through ID3D12Resource; ALLOW_DISPLAY is the part the console's PresentX
// requires and the part checked.
bool check_resource(ID3D12Resource* resource, UINT plane_index, CapturePlane& out) {
    if (!resource) {
        refuse_plane(plane_index, "null resource");
        return false;
    }
    if (!readable_range(resource, sizeof(void*))) {
        refuse_plane(plane_index, "resource pointer is not readable committed memory");
        return false;
    }
    void* const* vtable = *reinterpret_cast<void* const* const*>(resource);
    if (!readable_range(vtable, 15 * sizeof(void*))) {
        refuse_plane(plane_index, "resource vtable is not readable");
        return false;
    }
    if (!executable_pointer(vtable[0]) || !executable_pointer(vtable[10]) ||
        !executable_pointer(vtable[14])) {
        refuse_plane(plane_index, "resource vtable has no QueryInterface, GetDesc or GetHeapProperties");
        return false;
    }

    // The layer's Object::QueryInterface answers its private kTag with the implementation itself (one inheritance
    // chain; d3d12_com_object.h). This is the same recognition Object::from<T> uses. No ID3D12Resource method is
    // called unless it succeeds.
    void* implementation = nullptr;
    const HRESULT tag_result = resource->QueryInterface(kLeonidaResourceTag, &implementation);
    if (FAILED(tag_result) || !implementation) {
        refuse_plane(plane_index, "resource is not a Leonida layer ID3D12Resource");
        return false;
    }
    static_cast<IUnknown*>(implementation)->Release();  // the caller holds its own reference

    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D) {
        refuse_plane(plane_index, "resource is not a 2D texture");
        return false;
    }
    if (desc.SampleDesc.Count != 1) {
        refuse_plane(plane_index, "resource is multisampled");
        return false;
    }
    if (desc.DepthOrArraySize != 1 || desc.MipLevels < 1) {
        refuse_plane(plane_index, "resource is not a single-slice texture");
        return false;
    }
    if (desc.Width == 0 || desc.Height == 0 || desc.Width > 32768 || desc.Height > 32768) {
        refuse_plane(plane_index, "resource size is outside 1..32768");
        return false;
    }

    PlaneFormat format = PlaneFormat::Bgra8;
    switch (desc.Format) {
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: format = PlaneFormat::Bgra8; break;
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: format = PlaneFormat::Rgba8; break;
        // HDR10 presents R10G10B10A2_UNORM (PQ through the plane ColorSpace). dxgiformat.h has no _SRGB variant.
        case DXGI_FORMAT_R10G10B10A2_UNORM: format = PlaneFormat::Rgb10a2; break;
        default:
            refuse_plane(plane_index, "resource is not BGRA8, RGBA8 or R10G10B10A2_UNORM");
            return false;
    }

    D3D12_HEAP_PROPERTIES properties{};
    D3D12_HEAP_FLAGS heap_flags = D3D12_HEAP_FLAG_NONE;
    if (FAILED(resource->GetHeapProperties(&properties, &heap_flags))) {
        refuse_plane(plane_index, "GetHeapProperties failed");
        return false;
    }
    if ((heap_flags & D3D12_HEAP_FLAG_ALLOW_DISPLAY) == 0) {
        refuse_plane(plane_index, "resource has no D3D12_HEAP_FLAG_ALLOW_DISPLAY");
        return false;
    }

    out.resource = resource;
    out.desc = desc;
    out.format = format;
    return true;
}

std::uint8_t unorm10_to_8(std::uint32_t value) {
    return static_cast<std::uint8_t>((value * 255u + 511u) / 1023u);
}

bool write_ppm(const std::string& path, const void* pixels, UINT width, UINT height, UINT row_pitch,
               PlaneFormat format) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << "P6\n" << width << ' ' << height << "\n255\n";
    std::vector<std::uint8_t> row(static_cast<std::size_t>(width) * 3);
    const auto* bytes = static_cast<const std::uint8_t*>(pixels);
    for (UINT y = 0; y < height; ++y) {
        const std::uint8_t* in = bytes + static_cast<std::size_t>(y) * row_pitch;
        for (UINT x = 0; x < width; ++x) {
            const std::uint8_t* texel = in + static_cast<std::size_t>(x) * kBytesPerTexel;
            std::uint8_t* rgb = row.data() + static_cast<std::size_t>(x) * 3;
            if (format == PlaneFormat::Rgb10a2) {
                // Little-endian packed word: R in bits 0..9, G in 10..19, B in 20..29. The PPM stays 8-bit
                // without a colour space, so an HDR10 (PQ) plane looks dark in it; that is expected.
                std::uint32_t word = 0;
                std::memcpy(&word, texel, sizeof(word));
                rgb[0] = unorm10_to_8(word & 0x3ffu);
                rgb[1] = unorm10_to_8((word >> 10) & 0x3ffu);
                rgb[2] = unorm10_to_8((word >> 20) & 0x3ffu);
            } else {
                rgb[0] = format == PlaneFormat::Bgra8 ? texel[2] : texel[0];
                rgb[1] = texel[1];
                rgb[2] = format == PlaneFormat::Bgra8 ? texel[0] : texel[2];
            }
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    return out.good();
}

std::string ppm_path(const std::string& directory, std::uint64_t present, UINT plane) {
    const char separator = !directory.empty() && (directory.back() == '\\' || directory.back() == '/') ? '\0' : '\\';
    std::string path = directory;
    if (separator) path.push_back(separator);
    path += "present-" + std::to_string(present) + "-plane-" + std::to_string(plane) + ".ppm";
    return path;
}

// One committed readback buffer, one command list on the presenting queue, a fence, a Map and the PPM. Called with
// the capture mutex held, after every pointer has been validated.
HRESULT capture_plane(ID3D12Device* device, ID3D12CommandQueue* queue, const CapturePlane& plane,
                      const std::string& path) {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0, total_bytes = 0;
    device->GetCopyableFootprints(&plane.desc, 0, 1, 0, &footprint, &rows, &row_bytes, &total_bytes);
    if (rows == 0 || row_bytes == 0 || total_bytes == 0 || total_bytes > (512ull << 20)) {
        log_refusal("[rt] presentx: PresentX refused: copy footprint is empty or too large; E_INVALIDARG");
        return E_INVALIDARG;
    }
    const UINT64 row_pitch = footprint.Footprint.RowPitch;
    // All accepted formats are 4 bytes per texel, so GetCopyableFootprints must give a row that holds Width * 4.
    const UINT64 last_row = static_cast<UINT64>(plane.desc.Height - 1) * row_pitch + row_bytes;
    if (rows < plane.desc.Height || row_pitch < plane.desc.Width * kBytesPerTexel || row_bytes > row_pitch ||
        footprint.Offset > total_bytes || last_row > total_bytes - footprint.Offset) {
        log_refusal("[rt] presentx: PresentX refused: readback footprint does not hold the texture "
                    "(rows=%u height=%u row_pitch=%llu row_bytes=%llu width=%llu offset=%llu total=%llu); "
                    "E_INVALIDARG",
                    rows, static_cast<unsigned>(plane.desc.Height), static_cast<unsigned long long>(row_pitch),
                    static_cast<unsigned long long>(row_bytes), static_cast<unsigned long long>(plane.desc.Width),
                    static_cast<unsigned long long>(footprint.Offset),
                    static_cast<unsigned long long>(total_bytes));
        return E_INVALIDARG;
    }

    D3D12_HEAP_PROPERTIES readback_heap{};
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC readback_desc{};
    readback_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    readback_desc.Width = total_bytes;
    readback_desc.Height = 1;
    readback_desc.DepthOrArraySize = 1;
    readback_desc.MipLevels = 1;
    readback_desc.Format = DXGI_FORMAT_UNKNOWN;
    readback_desc.SampleDesc.Count = 1;
    readback_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    ID3D12Resource* readback = nullptr;
    HRESULT hr = device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                 IID_PPV_ARGS(&readback));
    if (FAILED(hr) || !readback) {
        log_refusal("[rt] presentx: PresentX refused: readback buffer creation failed (0x%08lX)",
                    static_cast<unsigned long>(hr));
        if (readback) readback->Release();
        return E_FAIL;
    }

    const D3D12_COMMAND_QUEUE_DESC queue_desc = queue->GetDesc();
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* fence = nullptr;
    HRESULT result = E_FAIL;
    do {
        hr = device->CreateCommandAllocator(queue_desc.Type, IID_PPV_ARGS(&allocator));
        if (FAILED(hr)) break;
        hr = device->CreateCommandList(0, queue_desc.Type, allocator, nullptr, IID_PPV_ARGS(&list));
        if (FAILED(hr)) break;
        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (FAILED(hr)) break;

        D3D12_RESOURCE_BARRIER to_copy{};
        to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        to_copy.Transition = {plane.resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                              D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_SOURCE};
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION destination{};
        destination.pResource = readback;
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = footprint;
        D3D12_TEXTURE_COPY_LOCATION source{};
        source.pResource = plane.resource;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = 0;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        D3D12_RESOURCE_BARRIER to_present = to_copy;
        to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
        list->ResourceBarrier(1, &to_present);
        hr = list->Close();
        if (FAILED(hr)) break;

        ID3D12CommandList* lists[] = {list};
        queue->ExecuteCommandLists(1, lists);
        hr = queue->Signal(fence, 1);
        if (FAILED(hr)) break;
        const ULONGLONG deadline = GetTickCount64() + 30000;
        while (fence->GetCompletedValue() < 1) {
            if (GetTickCount64() > deadline) {
                log_refusal("[rt] presentx: PresentX refused: the readback did not complete in 30 s; E_FAIL");
                hr = E_FAIL;
                break;
            }
            Sleep(1);
        }
        if (FAILED(hr)) break;

        D3D12_RANGE read_range{0, static_cast<SIZE_T>(total_bytes)};
        void* mapped = nullptr;
        hr = readback->Map(0, &read_range, &mapped);
        if (FAILED(hr) || !mapped) break;
        const bool wrote = write_ppm(path, mapped, static_cast<UINT>(plane.desc.Width),
                                     static_cast<UINT>(plane.desc.Height), footprint.Footprint.RowPitch,
                                     plane.format);
        readback->Unmap(0, nullptr);
        if (!wrote) {
            log_refusal("[rt] presentx: PresentX refused: cannot write %s; E_FAIL", path.c_str());
            break;
        }
        std::fprintf(stderr, "[rt] presentx: wrote %s (%ux%u, %s)\n", path.c_str(),
                     static_cast<unsigned>(plane.desc.Width), static_cast<unsigned>(plane.desc.Height),
                     format_name(plane.format));
        result = S_OK;
    } while (false);

    if (fence) fence->Release();
    if (list) list->Release();
    if (allocator) allocator->Release();
    readback->Release();
    return result;
}

std::mutex g_capture_mutex;               // one GPU readback and one present-<n>-* set at a time
std::atomic<std::uint64_t> g_present{0};  // routed PresentX calls, blank ones included (0-based)

const std::string& capture_directory() {
    static const std::string directory = env_string("LEONIDA_CAPTURE");
    return directory;
}

}  // namespace

const PresentXRouteConfig& presentx_route_config() {
    static const PresentXRouteConfig config = [] {
        PresentXRouteConfig route;
        route.present_slot = env_queue_slot("LEONIDA_PRESENTX_SLOT");
        route.present_slot_alt = env_queue_slot("LEONIDA_PRESENTX_SLOT_ALT");
        route.suspend_slot = env_queue_slot("LEONIDA_PRESENTX_SUSPEND_SLOT");
        route.resume_slot = env_queue_slot("LEONIDA_PRESENTX_RESUME_SLOT");
        // The public HLE reports PresentX at both 30 and 32. A capture run defaults to those two; without a capture
        // the routing stays opt-in, exactly as before.
        if (presentx_capture_enabled()) {
            if (!route.present_slot) route.present_slot = 30;
            if (!route.present_slot_alt) route.present_slot_alt = 32;
        }
        if (route.present_slot || route.present_slot_alt || route.suspend_slot || route.resume_slot) {
            std::fprintf(stderr,
                         "[rt] presentx: routed queue slots: PresentX=%u (alt %u, public HLE 30/32) SuspendX=%u "
                         "ResumeX=%u (%s)\n",
                         route.present_slot, route.present_slot_alt, route.suspend_slot, route.resume_slot,
                         presentx_capture_enabled() ? "windowless capture" : "no LEONIDA_CAPTURE: presents dropped");
        }
        return route;
    }();
    return config;
}

bool presentx_capture_enabled() { return !capture_directory().empty(); }

HRESULT presentx_capture_present(ID3D12Device* device, ID3D12CommandQueue* queue, UINT plane_count,
                                 const XboxPresentPlaneParameters* planes) {
    if (!presentx_capture_enabled()) return S_OK;  // routed, accepted, dropped: no window and no swapchain
    try {
        const std::uint64_t present = g_present.fetch_add(1, std::memory_order_relaxed);
        if (plane_count > 2) {
            refuse("plane count above 2");
            return E_INVALIDARG;
        }
        if (plane_count && !planes) {
            refuse("null plane array");
            return E_INVALIDARG;
        }
        if (plane_count && !readable_range(planes, sizeof(*planes) * plane_count)) {
            refuse("plane array is not readable committed memory");
            return E_INVALIDARG;
        }
        if (plane_count == 0) return S_OK;  // a blank present writes nothing
        if (!device || !queue) {
            refuse("no device or queue behind the routed call");
            return E_INVALIDARG;
        }

        CapturePlane sources[2];
        for (UINT i = 0; i < plane_count; ++i) {
            const XboxPresentPlaneParameters& plane = planes[i];
            if (plane.ResourceCount == 0) continue;
            if (!plane.ppResources) {
                log_refusal("[rt] presentx: PresentX plane %u refused: ResourceCount is %u but ppResources "
                            "is null; E_INVALIDARG", i, plane.ResourceCount);
                return E_INVALIDARG;
            }
            if (!readable_range(plane.ppResources, sizeof(ID3D12Resource*))) {
                refuse_plane(i, "ppResources is not readable committed memory");
                return E_INVALIDARG;
            }
            if (!check_resource(plane.ppResources[0], i, sources[i])) return E_INVALIDARG;
        }

        const std::string directory = capture_directory();
        if (!ensure_directory(directory)) {
            log_refusal("[rt] presentx: PresentX refused: cannot create capture directory %s; E_FAIL",
                        directory.c_str());
            return E_FAIL;
        }

        std::lock_guard serialise(g_capture_mutex);
        for (UINT i = 0; i < plane_count; ++i) {
            if (!sources[i].resource) continue;
            const HRESULT hr = capture_plane(device, queue, sources[i], ppm_path(directory, present, i));
            if (FAILED(hr)) return hr;
        }
        return S_OK;
    } catch (...) {
        log_refusal("[rt] presentx: PresentX capture failed with an exception; E_FAIL");
        return E_FAIL;
    }
}

}  // namespace rt::d3d12
