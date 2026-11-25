// Synthetic probe for the opt-in PresentX slot routing and the windowless first-frame capture (docs section 4.5).
//
// Modes:
//   capture    LEONIDA_PRESENTX_SLOT=40 and LEONIDA_CAPTURE=<temp dir>: known BGRA8 and HDR10 (R10G10B10A2_UNORM)
//              ALLOW_DISPLAY 2D textures are uploaded and each is presented through the proxy's queue slot 40; both
//              PPMs must hold the known pixels. A following 0-plane PresentX must return S_OK and write nothing.
//   malformed  The same environment, but the routed slot is called with a null plane array, a null ppResources, a
//              null resource, 3 planes, unreadable (PAGE_NOACCESS) plane/ppResources/resource pointers, a texture
//              without ALLOW_DISPLAY, a buffer and a foreign object with a readable/executable vtable. Each must
//              return E_INVALIDARG, write no file and not fault; the foreign object's GetDesc must not be called.
//   inert      No environment: slot 40 must still return E_NOTIMPL (the old refusal) and no file appears.
//
// Each mode is a separate process: the route and capture environment is read once. Needs a Vulkan 1.3 device; skips
// (77) when there is none.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/xbox_present_types.h"

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using rt::d3d12::XboxPresentPlaneParameters;
using QueueSlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t);
constexpr unsigned kPresentxSlot = 40;
constexpr UINT kWidth = 8, kHeight = 8;

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

std::uint64_t call_slot_at(void* queue, unsigned slot, std::uint64_t a1, std::uint64_t a2, std::uint64_t a3) {
    void** vtable = *reinterpret_cast<void***>(queue);
    return reinterpret_cast<QueueSlotFn>(vtable[slot])(queue, a1, a2, a3);
}

std::uint64_t call_slot(void* queue, std::uint64_t a1, std::uint64_t a2, std::uint64_t a3) {
    return call_slot_at(queue, kPresentxSlot, a1, a2, a3);
}

HRESULT as_hr(std::uint64_t value) { return static_cast<HRESULT>(static_cast<std::uint32_t>(value)); }

void expected_texel(UINT x, UINT y, std::uint8_t* rgb) {
    rgb[0] = static_cast<std::uint8_t>(10 + x * 10);
    rgb[1] = static_cast<std::uint8_t>(20 + y * 10);
    rgb[2] = static_cast<std::uint8_t>(30 + x + y);
}

// HDR10 plane: four quadrants with known packed values. Red is 0x3FF, green 0x3FF << 10, blue 0x3FF << 20, and the
// bottom-right is R=512, G=511, B=513 to exercise per-channel rounding. Expected PPM bytes are hard-coded:
// 1023 -> 255, 0 -> 0, 512 -> 128, 511 -> 127, 513 -> 128.
std::uint32_t expected_hdr_word(UINT x, UINT y) {
    if (y < kHeight / 2) return x < kWidth / 2 ? 0x3ffu : (0x3ffu << 10);
    if (x < kWidth / 2) return 0x3ffu << 20;
    return 512u | (511u << 10) | (513u << 20);
}

void expected_hdr_texel(UINT x, UINT y, std::uint8_t* rgb) {
    if (y < kHeight / 2) {
        if (x < kWidth / 2) {
            rgb[0] = 255; rgb[1] = 0; rgb[2] = 0;
        } else {
            rgb[0] = 0; rgb[1] = 255; rgb[2] = 0;
        }
    } else if (x < kWidth / 2) {
        rgb[0] = 0; rgb[1] = 0; rgb[2] = 255;
    } else {
        rgb[0] = 128; rgb[1] = 127; rgb[2] = 128;
    }
}

// A readable object with a readable, executable vtable that is NOT a Leonida layer resource. The capture guard must
// recognise it through the private tag QueryInterface and refuse before calling GetDesc/GetHeapProperties.
struct ForeignResource {
    void** vtable;
};
unsigned g_foreign_qi_calls = 0, g_foreign_desc_calls = 0, g_foreign_heap_calls = 0;

void WINAPI foreign_noop(void*) {}

HRESULT WINAPI foreign_query_interface(void*, const IID*, void** out) {
    ++g_foreign_qi_calls;
    if (out) *out = nullptr;
    return E_NOINTERFACE;
}

D3D12_RESOURCE_DESC WINAPI foreign_get_desc(void*) {
    ++g_foreign_desc_calls;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = kWidth;
    desc.Height = kHeight;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    return desc;
}

HRESULT WINAPI foreign_get_heap_properties(void*, D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS*) {
    ++g_foreign_heap_calls;
    return S_OK;
}

void* foreign_vtable[15];

void init_foreign_vtable() {
    for (void*& entry : foreign_vtable) entry = reinterpret_cast<void*>(&foreign_noop);
    foreign_vtable[0] = reinterpret_cast<void*>(&foreign_query_interface);
    foreign_vtable[10] = reinterpret_cast<void*>(&foreign_get_desc);
    foreign_vtable[14] = reinterpret_cast<void*>(&foreign_get_heap_properties);
}

struct Device {
    ID3D12Device* inner = nullptr;
    ID3D12Device* proxy = nullptr;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Resource> texture;
    Ref<ID3D12Resource> upload;
    Ref<ID3D12Resource> hdr_texture;
    Ref<ID3D12Resource> hdr_upload;
    Ref<ID3D12Resource> non_display;
    Ref<ID3D12Resource> buffer;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    std::uint64_t signalled = 0;

    ~Device() {
        // Release the children before the device: a layer object's destructor uses the device it was made on.
        fence.reset();
        list.reset();
        allocator.reset();
        buffer.reset();
        non_display.reset();
        hdr_upload.reset();
        hdr_texture.reset();
        upload.reset();
        texture.reset();
        queue.reset();
        if (proxy) proxy->Release();
        if (inner) inner->Release();
    }
};

bool create_device(Device& out, vktest::Device& device) {
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = device.instance_proc;
    info.instance = device.instance;
    info.physical = device.physical;
    info.device = device.device;
    info.queue = device.queue;
    info.queue_family = device.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&out.inner)))) return false;
    out.inner->AddRef();  // the backend's own reference; create_device_proxy takes the other
    if (FAILED(rt::d3d12_xs::create_device_proxy(out.inner, __uuidof(ID3D12Device),
                                                  reinterpret_cast<void**>(&out.proxy))))
        return false;
    rt::d3d12::set_device_outer(out.inner, out.proxy);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (FAILED(out.proxy->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&out.queue.p)))) return false;
    if (FAILED(out.proxy->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                  IID_PPV_ARGS(&out.allocator.p))))
        return false;
    if (FAILED(out.proxy->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, out.allocator.p, nullptr,
                                            IID_PPV_ARGS(&out.list.p))))
        return false;
    if (FAILED(out.list->Close())) return false;  // D3D12 resets only closed lists
    if (FAILED(out.proxy->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&out.fence.p)))) return false;
    return true;
}

bool run_list(Device& d) {
    if (FAILED(d.list->Close())) return false;
    ID3D12CommandList* lists[] = {d.list.p};
    d.queue->ExecuteCommandLists(1, lists);
    ++d.signalled;
    return SUCCEEDED(d.queue->Signal(d.fence.p, d.signalled)) &&
           SUCCEEDED(d.fence->SetEventOnCompletion(d.signalled, nullptr));
}

// Creates a BGRA8 display texture in PRESENT and uploads the known pattern into it on the test's queue.
bool make_texture(Device& d) {
    D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = kWidth;
    desc.Height = kHeight;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (FAILED(d.proxy->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                 D3D12_RESOURCE_STATE_PRESENT, nullptr,
                                                 IID_PPV_ARGS(&d.texture.p))))
        return false;

    // Guard cases: a 2D texture without ALLOW_DISPLAY and a buffer. Both must be refused without a file.
    D3D12_RESOURCE_DESC non_display_desc = desc;
    non_display_desc.Flags = D3D12_RESOURCE_FLAG_NONE;
    if (FAILED(d.proxy->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &non_display_desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&d.non_display.p))))
        return false;
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = 256;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.Format = DXGI_FORMAT_UNKNOWN;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(d.proxy->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
                                                 D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                 IID_PPV_ARGS(&d.buffer.p))))
        return false;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total_bytes = 0;
    d.proxy->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);
    if (total_bytes == 0 || footprint.Footprint.RowPitch < kWidth * 4) return false;

    D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC upload_desc{};
    upload_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    upload_desc.Width = total_bytes;
    upload_desc.Height = 1;
    upload_desc.DepthOrArraySize = 1;
    upload_desc.MipLevels = 1;
    upload_desc.Format = DXGI_FORMAT_UNKNOWN;
    upload_desc.SampleDesc.Count = 1;
    upload_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(d.proxy->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                 IID_PPV_ARGS(&d.upload.p))))
        return false;

    void* mapped = nullptr;
    if (FAILED(d.upload->Map(0, nullptr, &mapped)) || !mapped) return false;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    for (UINT y = 0; y < kHeight; ++y) {
        auto* row = bytes + static_cast<std::size_t>(y) * footprint.Footprint.RowPitch;
        for (UINT x = 0; x < kWidth; ++x) {
            std::uint8_t rgb[3];
            expected_texel(x, y, rgb);
            row[x * 4 + 0] = rgb[2];  // B
            row[x * 4 + 1] = rgb[1];  // G
            row[x * 4 + 2] = rgb[0];  // R
            row[x * 4 + 3] = 255;     // A
        }
    }
    d.upload->Unmap(0, nullptr);

    if (FAILED(d.allocator->Reset()) || FAILED(d.list->Reset(d.allocator.p, nullptr))) return false;
    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition = {d.texture.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                          D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_COPY_DEST};
    d.list->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = d.texture.p;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = d.upload.p;
    source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source.PlacedFootprint = footprint;
    d.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    D3D12_RESOURCE_BARRIER to_present = to_copy;
    to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    d.list->ResourceBarrier(1, &to_present);
    if (!run_list(d)) return false;

    // Second case: the same 8x8 footprint as HDR10 (R10G10B10A2_UNORM, also 4 bytes per texel), uploaded through a
    // row-major buffer with the known packed words.
    D3D12_RESOURCE_DESC hdr_desc = desc;
    hdr_desc.Format = DXGI_FORMAT_R10G10B10A2_UNORM;
    if (FAILED(d.proxy->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &hdr_desc,
                                                 D3D12_RESOURCE_STATE_PRESENT, nullptr,
                                                 IID_PPV_ARGS(&d.hdr_texture.p))))
        return false;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT hdr_footprint{};
    UINT64 hdr_total_bytes = 0;
    d.proxy->GetCopyableFootprints(&hdr_desc, 0, 1, 0, &hdr_footprint, nullptr, nullptr, &hdr_total_bytes);
    if (hdr_total_bytes == 0 || hdr_footprint.Footprint.RowPitch < kWidth * 4) return false;

    D3D12_RESOURCE_DESC hdr_upload_desc = upload_desc;
    hdr_upload_desc.Width = hdr_total_bytes;
    if (FAILED(d.proxy->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &hdr_upload_desc,
                                                 D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                 IID_PPV_ARGS(&d.hdr_upload.p))))
        return false;

    mapped = nullptr;
    if (FAILED(d.hdr_upload->Map(0, nullptr, &mapped)) || !mapped) return false;
    bytes = static_cast<std::uint8_t*>(mapped);
    for (UINT y = 0; y < kHeight; ++y) {
        auto* row = bytes + static_cast<std::size_t>(y) * hdr_footprint.Footprint.RowPitch;
        for (UINT x = 0; x < kWidth; ++x) {
            const std::uint32_t word = expected_hdr_word(x, y);
            std::memcpy(row + x * 4, &word, sizeof(word));  // little-endian R10G10B10A2 texel
        }
    }
    d.hdr_upload->Unmap(0, nullptr);

    if (FAILED(d.allocator->Reset()) || FAILED(d.list->Reset(d.allocator.p, nullptr))) return false;
    to_copy.Transition.pResource = d.hdr_texture.p;
    to_copy.Transition.StateBefore = D3D12_RESOURCE_STATE_PRESENT;
    to_copy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
    d.list->ResourceBarrier(1, &to_copy);
    destination.pResource = d.hdr_texture.p;
    source.pResource = d.hdr_upload.p;
    source.PlacedFootprint = hdr_footprint;
    d.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    to_present.Transition.pResource = d.hdr_texture.p;
    to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    d.list->ResourceBarrier(1, &to_present);
    return run_list(d);
}

bool check_ppm(const std::filesystem::path& path, void (*expected)(UINT, UINT, std::uint8_t*)) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string header = "P6\n8 8\n255\n";
    if (bytes.size() != header.size() + kWidth * kHeight * 3) {
        std::fprintf(stderr, "%s: %zu bytes, expected %zu\n", path.string().c_str(), bytes.size(),
                     header.size() + kWidth * kHeight * 3);
        return false;
    }
    if (!std::equal(header.begin(), header.end(), bytes.begin())) {
        std::fprintf(stderr, "%s: bad PPM header\n", path.string().c_str());
        return false;
    }
    const std::uint8_t* pixels = bytes.data() + header.size();
    for (UINT y = 0; y < kHeight; ++y) {
        for (UINT x = 0; x < kWidth; ++x) {
            std::uint8_t want[3];
            expected(x, y, want);
            const std::uint8_t* actual = pixels + (static_cast<std::size_t>(y) * kWidth + x) * 3;
            if (actual[0] != want[0] || actual[1] != want[1] || actual[2] != want[2]) {
                std::fprintf(stderr, "%s: texel (%u,%u) is (%u,%u,%u), expected (%u,%u,%u)\n",
                             path.string().c_str(), x, y, actual[0], actual[1], actual[2], want[0], want[1],
                             want[2]);
                return false;
            }
        }
    }
    return true;
}

std::size_t files_in(const std::filesystem::path& directory) {
    std::size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        std::fprintf(stderr, "capture file: %s\n", entry.path().string().c_str());
        ++count;
    }
    return count;
}

void set_route_environment(const std::filesystem::path& directory) {
    SetEnvironmentVariableA("LEONIDA_PRESENTX_SLOT", "40");
    SetEnvironmentVariableA("LEONIDA_PRESENTX_SUSPEND_SLOT", "41");
    SetEnvironmentVariableA("LEONIDA_PRESENTX_RESUME_SLOT", "42");
    SetEnvironmentVariableA("LEONIDA_CAPTURE", directory.string().c_str());
}

void clear_route_environment() {
    SetEnvironmentVariableA("LEONIDA_PRESENTX_SLOT", nullptr);
    SetEnvironmentVariableA("LEONIDA_PRESENTX_SUSPEND_SLOT", nullptr);
    SetEnvironmentVariableA("LEONIDA_PRESENTX_RESUME_SLOT", nullptr);
    SetEnvironmentVariableA("LEONIDA_CAPTURE", nullptr);
}

std::filesystem::path make_capture_directory(const char* name) {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / (std::string("leonida_presentx_") + name);
    std::error_code error;
    std::filesystem::remove_all(directory, error);
    std::filesystem::create_directories(directory, error);
    return directory;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc == 2 ? argv[1] : "";
    if (mode != "capture" && mode != "malformed" && mode != "inert") {
        std::fprintf(stderr, "usage: presentx_capture_probe capture|malformed|inert\n");
        return 2;
    }
    if (mode == "inert") clear_route_environment();

    const std::filesystem::path directory = make_capture_directory(mode.c_str());
    if (mode != "inert") set_route_environment(directory);

    vktest::Device vk;  // declared before the layer device: the layer is released first
    Device d;
    const vktest::DeviceStatus status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) {
        std::printf("presentx_capture_probe: no Vulkan device: skipped\n");
        return 77;
    }
    EXPECT(status == vktest::DeviceStatus::ready);
    EXPECT(create_device(d, vk));
    if (!make_texture(d)) {
        std::fprintf(stderr, "texture upload failed\n");
        return 1;
    }

    XboxPresentPlaneParameters plane{};
    plane.Token = 1;
    plane.ResourceCount = 1;
    plane.ppResources = &d.texture.p;
    ID3D12Resource* null_resource = nullptr;
    XboxPresentPlaneParameters null_resource_plane = plane;
    null_resource_plane.ppResources = &null_resource;

    if (mode == "capture") {
        EXPECT(as_hr(call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&plane), 0)) == S_OK);
        EXPECT(files_in(directory) == 1);
        EXPECT(check_ppm(directory / "present-0-plane-0.ppm", expected_texel));

        XboxPresentPlaneParameters hdr_plane{};
        hdr_plane.Token = 2;
        hdr_plane.ResourceCount = 1;
        hdr_plane.ppResources = &d.hdr_texture.p;
        EXPECT(as_hr(call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&hdr_plane), 0)) == S_OK);
        EXPECT(files_in(directory) == 2);
        EXPECT(check_ppm(directory / "present-1-plane-0.ppm", expected_hdr_texel));

        // A blank PresentX writes nothing and is accepted. The optional SuspendX/ResumeX slots are routed too:
        // with capture configured they are accepted without touching a window or the capture path.
        EXPECT(as_hr(call_slot(d.queue.p, 0, 0, 0)) == S_OK);
        EXPECT(as_hr(call_slot_at(d.queue.p, 41, 0, 0, 0)) == S_OK);
        EXPECT(as_hr(call_slot_at(d.queue.p, 42, 0, 0, 0)) == S_OK);
        // The alternate PresentX slot: with LEONIDA_CAPTURE set and no explicit ALT, the public HLE's 32 is routed.
        EXPECT(as_hr(call_slot_at(d.queue.p, 32, 0, 0, 0)) == S_OK);
        EXPECT(files_in(directory) == 2);
        std::printf("presentx_capture_probe: capture 2/2 presents, 2 PPMs (8x8 BGRA, 8x8 R10G10B10A2) checked, "
                    "blank/alt/SuspendX/ResumeX S_OK\n");
        return 0;
    }

    if (mode == "malformed") {
        XboxPresentPlaneParameters null_pp = plane;
        null_pp.ppResources = nullptr;
        ID3D12Resource* fake = static_cast<ID3D12Resource*>(
            VirtualAlloc(nullptr, 4096, MEM_COMMIT, PAGE_NOACCESS));
        EXPECT(fake != nullptr);
        XboxPresentPlaneParameters bad_pointer = plane;
        bad_pointer.ppResources = &fake;
        XboxPresentPlaneParameters guarded_array = plane;
        guarded_array.ppResources = reinterpret_cast<ID3D12Resource* const*>(fake);
        XboxPresentPlaneParameters non_display_plane = plane;
        non_display_plane.ppResources = &d.non_display.p;
        XboxPresentPlaneParameters buffer_plane = plane;
        buffer_plane.ppResources = &d.buffer.p;
        init_foreign_vtable();
        ForeignResource foreign{foreign_vtable};
        ID3D12Resource* foreign_resource = reinterpret_cast<ID3D12Resource*>(&foreign);
        XboxPresentPlaneParameters foreign_plane = plane;
        foreign_plane.ppResources = &foreign_resource;

        int refused = 0;
        const std::uint64_t cases[] = {
            call_slot(d.queue.p, 1, 0, 0),                                                   // null plane array
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&null_pp), 0),           // null ppResources
            call_slot(d.queue.p, 3, reinterpret_cast<std::uint64_t>(&plane), 0),             // 3 planes
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&null_resource_plane), 0),  // null resource
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(fake), 0),               // unreadable planes
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&guarded_array), 0),     // unreadable ppResources
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&bad_pointer), 0),       // unreadable resource
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&non_display_plane), 0),  // no ALLOW_DISPLAY
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&buffer_plane), 0),      // not a 2D texture
            call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&foreign_plane), 0),     // foreign vtable
        };
        for (std::uint64_t result : cases) {
            if (as_hr(result) == E_INVALIDARG) ++refused;
        }
        EXPECT(refused == 10);
        EXPECT(g_foreign_qi_calls > 0);
        EXPECT(g_foreign_desc_calls == 0);
        EXPECT(g_foreign_heap_calls == 0);
        // The same foreign object 40 more times: the guard keeps returning E_INVALIDARG and never calls GetDesc, and
        // the logger suppresses the repeated line (one line per 250 ms per distinct message).
        for (int i = 0; i < 40; ++i) {
            EXPECT(as_hr(call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&foreign_plane), 0)) ==
                   E_INVALIDARG);
        }
        EXPECT(g_foreign_qi_calls == 41);
        EXPECT(g_foreign_desc_calls == 0);
        EXPECT(g_foreign_heap_calls == 0);
        EXPECT(files_in(directory) == 0);
        VirtualFree(fake, 0, MEM_RELEASE);
        std::printf("presentx_capture_probe: malformed 10/10 E_INVALIDARG, foreign GetDesc not called, no files, "
                    "no crash\n");
        return 0;
    }

    // inert: the configured slot is unset, so nothing is captured. merge: the answer is upstream's for an unmodelled
    // console-only queue slot past the public chain, S_OK (a refusal wedged the present loop; d3d12_xs_device.cpp,
    // "unmodelled console queue methods"), where this tree used to refuse it with E_NOTIMPL.
    EXPECT(as_hr(call_slot(d.queue.p, 1, reinterpret_cast<std::uint64_t>(&plane), 0)) == S_OK);
    EXPECT(files_in(directory) == 0);
    std::printf("presentx_capture_probe: inert 1/1 S_OK, no files\n");
    return 0;
}
