// XboxBvhBuilder.dll served by the layer's own acceleration-structure builder (RT-5, boot stop 18).
//
// The package's XboxBvhBuilder.dll builds the game's ray-tracing structures with 52 console compute programs that
// its constructor creates as pipelines on the title's thread. Translated, they take 25-60 s to compile on a cold
// driver cache (31.2 s for one of them on an RTX 4080, 59.5 s for all 52 on an RX 9070 XT, measured in #7231 and
// #7049), longer than the title's startup watchdog (about 30 s), so a first boot ends at the builder. The layer
// already builds XDXR BVH4 structures itself (BuildRaytracingAccelerationStructure on the Vulkan device, D3D-178,
// src/xdxr_build.*), in the node format the game's translated traversal shaders read, and its one pipeline compiles
// in well under a second. So this file serves the DLL's exports with the layer's builder and the DLL is never
// loaded: the loader binds an import to a shim before a bundled module (pe_loader.cpp, Loader::resolve).
//
// The exports, as the executable imports them (7 of the DLL's 12, measured statically on 1.0.2.1: names and call
// counts only). Member functions: `this` comes first (rcx), then the arguments, as for a free function on x64.
//   ??0  XboxBvhBuilder(ID3D12Device11*, bool)                                      1 call site
//   ??1  ~XboxBvhBuilder()                                                         1
//   Set64BitMortonCodes(bool)                                                       8
//   GetRaytracingAccelerationStructurePrebuildInfo(const INPUTS*, const XDXR_STREAM_DESC*, PREBUILD_INFO*)  3
//   BuildRaytracingAccelerationStructureBatched2X(GCL6*, UINT, const BATCHED_DESC2*)  6
//   CopyRaytracingAccelerationStructure(GCL4*, UINT64 dest, UINT64 source, COPY_MODE)  2
//   RehydrateRaytracingAccelerationStructureBatched2X(GCL8*, UINT, const BUILD_DESC*, const UINT64*) -> HRESULT  4
// The Xbox descriptions are the measured layouts of src/xdxr_descs.h (docs/xdxr.md).
//
// What differs from the console builder, on purpose:
//   - No programs are created: the constructor only remembers the device.
//   - The XDXR stream records (docs/xdxr.md §3) tune the console builder's own heuristics. The layer's builder has
//     none of those settings, so a non-empty stream is logged once and otherwise ignored; the structure is still a
//     valid BVH of the same geometry. Set64BitMortonCodes is recorded and ignored for the same reason.
//   - The DLL throws its validation errors as C++ exceptions (docs/xdxr.md, "Errors are thrown, not returned"). A
//     shim must not throw into the title, so an element the DLL would reject is logged and skipped instead. The one
//     rule a correct title meets every time is kept: the offline-statistics postbuild type (0x10) is never written.
//   - Rehydration: RehydrateRaytracingAccelerationStructureBatched2X rebuilds each structure from the source
//     geometry its description carries (BuildRaytracingAccelerationStructure) (docs/bvh.md §4a). The leaf frame
//     compressionParams gives is not applied yet: the leaves are built from the geometry as the description has it.
//     The GPU cannot read offline dehydrated console streams directly (docs/bvh.md §4/§4a),
//     so source-only streams without geometry are refused with E_NOTIMPL. The title passes Scratch = 0, so the shim
//     supplies scratch: one buffer per call, sized via GetRaytracingAccelerationStructurePrebuildInfo, in which each
//     element has a region of its own (rehydrate_scratch).
//
// The builder object belongs to the title, which allocates it with a size this file doesn't know, so nothing is
// written into it: what the shim keeps (the device, the Morton option) lives in a table keyed by its address.
//
// LEONIDA_BVH_BUILDER=console (or our older LEONIDA_DLL_BVH=1) turns the shim off, so the imports bind to the bundled
// DLL as before (for work on the console programs themselves, #5680 and #7318). The variable is read once, when the
// shim registers.
//
// merge: kept upstream's whole-DLL shim (no DLL constructor compiling 52 translated programs, tests) with our build
// path inside it (leonida/work 7e9db802c, which served only the build/copy/size exports): the title's TYPELESS index
// formats read as UINT, bottom levels our XDXR builder cannot read left out instead of failing the whole list at
// replay, the counters, and the single-build export. In our runs the translated DLL chain hung the GPU (a
// bvh/RadixAlloc dispatch, a Windows TDR, then the title's GPU watchdog fatal); with this path 528 BLAS and 16 TLAS
// were built, none left out. Design after five-star-recomp (docs/raytracing.md, GPL-3.0); the code here is ours.
#include <windows.h>

#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef LEONIDA_VULKAN_DEVICE
#include "d3d12_com.h"   // rt::d3d12::hold_until_reset
#endif
#include "d3d12_xs_device.h"
#include "shims.h"
#include "xmem.h"
#include "../src/title_texture_memory.h"
#include "../src/xdxr_descs.h"

namespace {

namespace xdxr = leonida::xdxr;

struct BuilderState {
    ID3D12Device5* device = nullptr;   // a reference held for the builder's lifetime; null if it has no DXR methods
    bool morton64 = false;
    // Rehydration scratch made for a list that cannot hold it (not the layer's own, rehydrate_scratch): kept until the
    // builder is destroyed, since the shim never learns when the GPU is done with it.
    std::vector<ID3D12Resource*> kept;
};

void release_state(BuilderState& state) {
    for (ID3D12Resource* scratch : state.kept) scratch->Release();
    state.kept.clear();
    if (state.device) state.device->Release();
    state.device = nullptr;
}

std::mutex g_mutex;
std::unordered_map<const void*, BuilderState> g_builders;

void log_once(std::atomic<bool>& flag, const char* text) {
    if (!flag.exchange(true)) std::fprintf(stderr, "[rt] XboxBvhBuilder shim: %s\n", text);
}

// The PC device behind what the title passed: the real device behind a d3d12_xs device proxy, or the pointer
// itself when it is already a device (a test, or a backend that hands out unwrapped devices).
ID3D12Device5* device5_of(void* device) {
    if (!device) return nullptr;
    IUnknown* base = rt::d3d12_xs::device_proxy_real_device(device);
    if (!base) base = static_cast<IUnknown*>(device);
    ID3D12Device5* out = nullptr;
    if (FAILED(base->QueryInterface(IID_PPV_ARGS(&out)))) return nullptr;
    return out;
}

// The command list a title passes, as the backend's own list: a d3d12_xs proxy is unwrapped first, so the PC
// interfaces below are asked of the real list and never of a console-shaped proxy (whose vtable is not the PC one;
// docs/d3d12_command_objects.md §3.5a). The console list proxy is asked first (command_list_proxy_real_list), then
// any other proxy real_object_behind_proxy follows.
IUnknown* real_list(void* list) {
    if (!list) return nullptr;
    if (ID3D12GraphicsCommandList* pc = rt::d3d12_xs::command_list_proxy_real_list(list)) return pc;
    IUnknown* real = rt::d3d12_xs::real_object_behind_proxy(list);
    return real ? real : static_cast<IUnknown*>(list);
}

ID3D12GraphicsCommandList4* list4_of(void* list) {
    IUnknown* real = real_list(list);
    if (!real) return nullptr;
    ID3D12GraphicsCommandList4* out = nullptr;
    if (FAILED(real->QueryInterface(IID_PPV_ARGS(&out)))) return nullptr;
    return out;
}

// ---- our build path (leonida/work 7e9db802c) ----

std::atomic<std::uint64_t> g_blas{0}, g_tlas{0}, g_skipped{0}, g_copies{0};

void note(std::atomic<std::uint64_t>& counter, const char* what) {
    const std::uint64_t n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & (n - 1)) == 0)
        std::fprintf(stderr, "[rt] xbox_bvh: %llu %s (runtime XDXR builder; %llu BLAS, %llu TLAS, %llu left out)\n",
                     static_cast<unsigned long long>(n), what,
                     static_cast<unsigned long long>(g_blas.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(g_tlas.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(g_skipped.load(std::memory_order_relaxed)));
}

const D3D12_RAYTRACING_GEOMETRY_DESC* geometry(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& in,
                                               UINT i) {
    if (in.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS)
        return in.ppGeometryDescs ? in.ppGeometryDescs[i] : nullptr;
    return in.pGeometryDescs ? &in.pGeometryDescs[i] : nullptr;
}

// Whether the runtime's builder reads this bottom-level input (xdxr_inputs: R32G32B32/R32G32 float positions, no,
// R16 or R32 indices, triangles only). Anything else is left out here: a refused build fails the whole list at
// replay, and an empty structure only loses what the rays would hit.
bool readable(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& in) {
    if (in.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL) return true;
    static std::atomic<int> told{0};
    const auto refuse = [&](const char* why, UINT i, const D3D12_RAYTRACING_GEOMETRY_DESC* gd) {
        if (told.fetch_add(1, std::memory_order_relaxed) < 8)
            std::fprintf(stderr, "[rt] xbox_bvh: BLAS not built: %s (geometry %u of %u, layout %d, type %d, vertex "
                                 "format %d, index format %d, %u vertices, %u indices)\n",
                         why, i, in.NumDescs, static_cast<int>(in.DescsLayout), gd ? static_cast<int>(gd->Type) : -1,
                         gd ? static_cast<int>(gd->Triangles.VertexFormat) : -1,
                         gd ? static_cast<int>(gd->Triangles.IndexFormat) : -1, gd ? gd->Triangles.VertexCount : 0,
                         gd ? gd->Triangles.IndexCount : 0);
        return false;
    };
    for (UINT i = 0; i < in.NumDescs; ++i) {
        const D3D12_RAYTRACING_GEOMETRY_DESC* gd = geometry(in, i);
        if (!gd) return refuse("no geometry desc", i, gd);
        if (gd->Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) return refuse("not triangles", i, gd);
        const DXGI_FORMAT v = gd->Triangles.VertexFormat, x = gd->Triangles.IndexFormat;
        if (v != DXGI_FORMAT_R32G32B32_FLOAT && v != DXGI_FORMAT_R32G32_FLOAT)
            return refuse("vertex format (R32G32B32/R32G32 float only)", i, gd);
        if (x != DXGI_FORMAT_UNKNOWN && x != DXGI_FORMAT_R16_UINT && x != DXGI_FORMAT_R32_UINT)
            return refuse("index format", i, gd);
    }
    return true;
}

// The title names 16- and 32-bit indices by their TYPELESS formats (DXGI 53 and 39; seen on most of its BLAS builds),
// which D3D12 on the PC does not take for an index buffer: they are read as R16_UINT and R32_UINT.
DXGI_FORMAT index_format(DXGI_FORMAT f) {
    if (f == DXGI_FORMAT_R16_TYPELESS) return DXGI_FORMAT_R16_UINT;
    if (f == DXGI_FORMAT_R32_TYPELESS) return DXGI_FORMAT_R32_UINT;
    return f;
}

// What each build writes, [dest, dest + our layout) (the first few logged), and any build whose range reaches another
// structure's start: ours larger than what the title gave it, as the rehydrated ones were (2026-11-26). Measured on the
// retail boot after the rehydrate fix: 146 BLAS, 245 rehydrated and 5 TLAS builds, none overlapping.
std::mutex g_range_mutex;
std::map<UINT64, UINT64> g_written;  // dest -> bytes, under g_range_mutex
void note_written_range(const char* what, UINT64 dest, UINT64 bytes, UINT flags) {
    if (!dest || !bytes) return;
    static std::atomic<unsigned> s_told{0}, s_overlaps{0};
    if (s_told.fetch_add(1, std::memory_order_relaxed) < 16)
        std::fprintf(stderr, "[rt] bvh-write: %s %#llx..%#llx (%llu bytes, flags %#x)\n", what,
                     static_cast<unsigned long long>(dest), static_cast<unsigned long long>(dest + bytes),
                     static_cast<unsigned long long>(bytes), flags);
    std::lock_guard<std::mutex> lock(g_range_mutex);
    if (g_written.size() >= (1u << 16)) g_written.clear();
    // Another structure starting inside ours, or ours starting inside an earlier one.
    auto next = g_written.upper_bound(dest);
    UINT64 other = 0, other_bytes = 0;
    if (next != g_written.end() && next->first < dest + bytes) {
        other = next->first;
        other_bytes = next->second;
    } else if (next != g_written.begin()) {
        auto prev = std::prev(next);
        if (prev->first != dest && prev->first + prev->second > dest) {
            other = prev->first;
            other_bytes = prev->second;
        }
    }
    if (other && s_overlaps.fetch_add(1, std::memory_order_relaxed) < 30)
        std::fprintf(stderr, "[rt] bvh-overlap: %s %#llx + %llu reaches the structure at %#llx + %llu\n", what,
                     static_cast<unsigned long long>(dest), static_cast<unsigned long long>(bytes),
                     static_cast<unsigned long long>(other), static_cast<unsigned long long>(other_bytes));
    g_written[dest] = bytes;
}

// One build on the PC list, with the title's typeless index formats made plain and unreadable bottom levels left out.
void build_one(ID3D12GraphicsCommandList4* list, D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc,
               UINT postbuild_count, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* postbuild,
               ID3D12Device5* device = nullptr) {
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geometries;  // the inputs with plain index formats, as an array
    if (desc.Inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        bool typeless = false;
        for (UINT i = 0; i < desc.Inputs.NumDescs; ++i)
            if (const auto* gd = geometry(desc.Inputs, i);
                gd && gd->Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES &&
                index_format(gd->Triangles.IndexFormat) != gd->Triangles.IndexFormat)
                typeless = true;
        if (typeless) {
            geometries.reserve(desc.Inputs.NumDescs);
            for (UINT i = 0; i < desc.Inputs.NumDescs; ++i) {
                const auto* gd = geometry(desc.Inputs, i);
                geometries.push_back(gd ? *gd : D3D12_RAYTRACING_GEOMETRY_DESC{});
                if (gd && gd->Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES)
                    geometries.back().Triangles.IndexFormat = index_format(gd->Triangles.IndexFormat);
            }
            desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            desc.Inputs.pGeometryDescs = geometries.data();
        }
    }
    if (!readable(desc.Inputs)) {
        note(g_skipped, "builds left out");
        return;
    }
    if (device) {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild{};
        device->GetRaytracingAccelerationStructurePrebuildInfo(&desc.Inputs, &prebuild);
        note_written_range(desc.Inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL ? "tlas" : "blas",
                           desc.DestAccelerationStructureData, prebuild.ResultDataMaxSizeInBytes,
                           static_cast<UINT>(desc.Inputs.Flags));
    }
    list->BuildRaytracingAccelerationStructure(&desc, postbuild_count, postbuild);
    if (desc.Inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL)
        note(g_tlas, "TLAS builds");
    else
        note(g_blas, "BLAS builds");
}

// ??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z: XboxBvhBuilder(ID3D12Device11* device, bool).
// Returns `this`, as an MSVC constructor does. The bool's meaning is not measured; it is not needed here.
void* builder_construct(void* self, void* device, bool) {
    ID3D12Device5* device5 = device5_of(device);
    if (!device5) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "the device has no ID3D12Device5 (DXR) interface; builds will be skipped");
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    BuilderState& state = g_builders[self];
    release_state(state);   // the same storage constructed twice without a destructor
    state = BuilderState{device5, false, {}};
    return self;
}

// ??1XboxBvhBuilder@@QEAA@XZ: ~XboxBvhBuilder().
void builder_destruct(void* self) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it == g_builders.end()) return;
    release_state(it->second);
    g_builders.erase(it);
}

// ?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z: a console-builder option (64-bit Morton keys for its sort).
void builder_set_64bit_morton_codes(void* self, bool enable) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it != g_builders.end()) it->second.morton64 = enable;
}

// The builder's device with a reference of the caller's own, taken under the lock, so a destructor on another
// thread can't release it during the call. The caller releases it.
ID3D12Device5* builder_device(void* self) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it == g_builders.end() || !it->second.device) return nullptr;
    it->second.device->AddRef();
    return it->second.device;
}

// Scratch for one RehydrateRaytracingAccelerationStructureBatched2X call, where the title passes
// ScratchAccelerationStructureData = 0 (measured on the retail title): a new buffer of `bytes`, in which each element
// that needs scratch has a region of its own. It is never shared with another call: two calls on one list (the title
// puts no barrier between them for scratch it does not know of) or lists recorded on several threads never write the
// same bytes. The list holds the buffer until it is Reset or released (rt::d3d12::hold_until_reset), and the layer
// destroys it after the submission that used it has retired; a list that cannot hold it leaves it to the builder,
// which keeps it until its destruction. Returns the buffer's GPU address, 0 when it could not be made.
UINT64 rehydrate_scratch(void* self, ID3D12Device5* device, ID3D12GraphicsCommandList4* list, UINT64 bytes) {
    if (!device || !bytes) return 0;
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC res_desc{};
    res_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    res_desc.Alignment = 0;
    res_desc.Width = bytes;
    res_desc.Height = 1;
    res_desc.DepthOrArraySize = 1;
    res_desc.MipLevels = 1;
    res_desc.Format = DXGI_FORMAT_UNKNOWN;
    res_desc.SampleDesc.Count = 1;
    res_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    res_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

    ID3D12Resource* res = nullptr;
    HRESULT hr = device->CreateCommittedResource(
        &heap_props, D3D12_HEAP_FLAG_NONE, &res_desc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&res));
    if (FAILED(hr) || !res) return 0;
    const UINT64 va = res->GetGPUVirtualAddress();
    bool held = false;
#ifdef LEONIDA_VULKAN_DEVICE
    held = rt::d3d12::hold_until_reset(list, res);
#else
    (void)list;
#endif
    if (held) {
        res->Release();   // the list's reference keeps it from here
        return va;
    }
    static std::atomic<bool> s_logged{false};
    log_once(s_logged, "RehydrateRaytracingAccelerationStructureBatched2X: the list cannot hold its scratch; the "
                       "builder keeps it until it is destroyed");
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it == g_builders.end()) {   // destroyed during the call: nothing is recorded with this scratch
        res->Release();
        return 0;
    }
    it->second.kept.push_back(res);
    return va;
}

bool has_geometry(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& inputs) {
    return inputs.NumDescs > 0 && (inputs.pGeometryDescs || inputs.InstanceDescs);
}

void note_stream(const xdxr::StreamDesc& stream) {
    if (stream.size_in_bytes == 0 || !stream.stream) return;
    static std::atomic<bool> s_logged{false};
    log_once(s_logged, "XDXR stream records (console builder settings) are ignored by the layer's builder");
}

// ?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_
// STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z
// The sizes the layer's own builder needs, which are the sizes the layer's build will use.
void builder_prebuild_info(void* self, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* inputs,
                           const xdxr::StreamDesc* stream, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO* info) {
    if (!info) return;
    *info = {};
    if (!inputs) return;
    if (stream) note_stream(*stream);
    ID3D12Device5* device = builder_device(self);
    if (!device) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "GetRaytracingAccelerationStructurePrebuildInfo on a builder without a DXR device; sizes are 0");
        return;
    }
    device->GetRaytracingAccelerationStructurePrebuildInfo(inputs, info);
    device->Release();
}

// ?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_
// BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z
// Each element is a public build description followed by its postbuild descriptions and a stream (xdxr_descs.h), so
// it is one BuildRaytracingAccelerationStructure on the list.
void builder_build_batched2x(void* self, void* list, UINT count, const xdxr::BatchedDesc2* descs) {
    static_assert(sizeof(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC) == 0x30, "the BATCHED_DESC2 prefix");
    static_assert(sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC) == sizeof(xdxr::PostbuildInfoDesc),
                  "postbuild descriptions are the public ones");
    if (count == 0) return;
    if (!descs) return;
    ID3D12GraphicsCommandList4* list4 = list4_of(list);
    if (!list4) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "BuildRaytracingAccelerationStructureBatched2X: the list has no ID3D12GraphicsCommandList4");
        return;
    }
    ID3D12Device5* device = builder_device(self);
    for (UINT i = 0; i < count; ++i) {
        const xdxr::BatchedDesc2& element = descs[i];
        note_stream(element.stream);
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build;
        std::memcpy(&build, &element, sizeof(build));
        // The postbuild descriptions, without the offline-statistics type the DLL refuses (docs/xdxr.md §4).
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC kept[16];
        UINT n_kept = 0;
        bool skipped_statistics = false;
        const auto* post = reinterpret_cast<const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*>(
            element.postbuild_info_descs);
        for (UINT p = 0; post && p < element.num_postbuild_info_descs; ++p) {
            if (static_cast<uint32_t>(post[p].InfoType) == xdxr::kPostbuildInfoOfflineStatistics) {
                skipped_statistics = true;
                continue;
            }
            if (n_kept == 16) {
                static std::atomic<bool> s_logged{false};
                log_once(s_logged, "more than 16 postbuild descriptions in one element; the rest are dropped");
                break;
            }
            kept[n_kept++] = post[p];
        }
        if (skipped_statistics) {
            // The DLL throws E_INVALIDARG for this element before any work; the shim skips it the same way.
            static std::atomic<bool> s_logged{false};
            log_once(s_logged, "an element asked for offline statistics (0x10), which the DLL refuses; element skipped");
            continue;
        }
        build_one(list4, build, n_kept, n_kept ? kept : nullptr, device);
    }
    if (device) device->Release();
    list4->Release();
}

// ?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBU
// D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z
// (ours; the executable does not import it on 1.0.2.1, served in case another build does).
void builder_build(void*, void* list, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc,
                   UINT postbuild_count, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* postbuild) {
    if (!desc) return;
    ID3D12GraphicsCommandList4* list4 = list4_of(list);
    if (!list4) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "BuildRaytracingAccelerationStructure: the list has no ID3D12GraphicsCommandList4");
        return;
    }
    build_one(list4, *desc, postbuild_count, postbuild);
    list4->Release();
}

// ?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_
// ACCELERATION_STRUCTURE_COPY_MODE@@@Z
void builder_copy(void*, void* list, UINT64 dest, UINT64 source, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE mode) {
    ID3D12GraphicsCommandList4* list4 = list4_of(list);
    if (!list4) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "CopyRaytracingAccelerationStructure: the list has no ID3D12GraphicsCommandList4");
        return;
    }
    list4->CopyRaytracingAccelerationStructure(dest, source, mode);
    list4->Release();
    note(g_copies, "structure copies");
}

// The room the title gave a rehydrated structure: the console's own size of it, 64 * (totalSizeIn64BNodes + 1) bytes
// (docs/bvh.md §3.1, §4a), from the dehydrated stream's XboxBvh4Header (flags, then totalSizeIn64BNodes at +4). The
// retail title names the stream by its header (0x3e8265100: the payload's +0x100) and rehydrates in place: the
// destination is the same memory's GPU-side alias (0x7e8265100, 16 GiB up), so the room is what the stream's
// allocation holds. A source given by its payload start (header at +0x100) is read there instead.
// 0 when the stream is not readable here or the field is out of range. Read only after a VirtualQuery says the
// bytes are there (no fault: the process's exception handlers take any access violation as the title's). Pages the
// CPU may not read at the title's address are read through the page's backing in its section
// (rt::xmem::g_window_pages).
thread_local void* t_window_backing = nullptr;  // a plain pointer: no TLS destructor (no thread_local with one here)
void note_window_backing(std::uintptr_t, void* backing, void*, std::uint64_t) { t_window_backing = backing; }
const std::uint8_t* title_bytes(std::uintptr_t at, std::size_t size) {
    if (leonida::texture_memory_readable(at, size)) return reinterpret_cast<const std::uint8_t*>(at);
    constexpr std::uintptr_t kPage = 0x10000;
    const std::uintptr_t page = at & ~(kPage - 1);
    if (at + size > page + kPage) return nullptr;  // across two window pages: not needed for a 4-byte field
    const auto window_pages = rt::xmem::g_window_pages.load(std::memory_order_acquire);
    if (!window_pages) return nullptr;
    t_window_backing = nullptr;
    window_pages(page, page + kPage, &note_window_backing);
    if (!t_window_backing) return nullptr;
    const auto host = reinterpret_cast<std::uintptr_t>(t_window_backing) + (at - page);
    return leonida::texture_memory_readable(host, size) ? reinterpret_cast<const std::uint8_t*>(host) : nullptr;
}
UINT64 header_room(UINT64 header, std::uint32_t* flags_out) {
    if (!header || header > UINT64_MAX - 8) return 0;
    const std::uint8_t* words = title_bytes(static_cast<std::uintptr_t>(header), 8);
    if (!words) return 0;
    const std::uint32_t flags = reinterpret_cast<const volatile std::uint32_t*>(words)[0];
    const std::uint32_t nodes = reinterpret_cast<const volatile std::uint32_t*>(words)[1];
    if (nodes == 0 || nodes >= (1u << 26)) return 0;  // past 4 GiB: not a header
    if (flags_out) *flags_out = flags;
    return 64ull * (UINT64(nodes) + 1);
}
UINT64 rehydrated_room(UINT64 stream, std::uint32_t* flags = nullptr) {
    if (const UINT64 room = header_room(stream, flags)) return room;
    return stream <= UINT64_MAX - 0x100 ? header_room(stream + 0x100, flags) : 0;
}

// ?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_
// BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z -> HRESULT.
// Rebuilds each acceleration structure from geometry on the command list.
//
// The title rehydrates in place: the destination is the stream's own memory (its GPU alias), only as large as the
// console's structure (rehydrated_room). Our builder keeps one triangle a leaf where the console pairs two, and lays a
// structure out for the most nodes a tree can have: the retail streams' structures were 1.9-2.3 times their console
// size (33,088 bytes against our 69,696). Our builds write the title's pages in place, and past the room lies the
// title's next allocation: BOX16 nodes with +-65504 bounds written over title structures, read back as pointers,
// killed every boot 105-215 s in (2026-11-25/26). Even compacted ours does not fit, and a structure cut at the room
// sends traversal into the bytes after it (the GPU stopped, the title's GPU watchdog fired: 2026-11-26 03:58).
// So an element whose structure does not fit is built from its first triangle only, a valid structure that fits:
// that geometry is missing from ray-traced effects until the builder writes the console's paired leaves. The layer
// is told the room (note_acceleration_structure_room), which bounds any copy the title makes into it. An element
// whose room cannot be read (no stream, or one the CPU cannot read) is built in place as before.
HRESULT builder_rehydrate_batched2x(void* self, void* list, UINT count,
                                    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* descs,
                                    const UINT64* source_data) {
    if (count == 0) return S_OK;
    if (!descs) return E_INVALIDARG;
    if (!list) return E_INVALIDARG;
    ID3D12GraphicsCommandList4* list4 = list4_of(list);
    if (!list4) {
        static std::atomic<bool> s_logged{false};
        log_once(s_logged, "RehydrateRaytracingAccelerationStructureBatched2X: the list has no ID3D12GraphicsCommandList4");
        return E_NOINTERFACE;
    }
    ID3D12Device5* device = builder_device(self);
    // First pass: the descriptions as they are recorded, and for each element that builds without scratch of the
    // title's a region of its own in one scratch buffer for the batch, at D3D12's acceleration-structure alignment
    // (256 bytes, which the layer checks scratch addresses against). The XDXR recorder leaves the dependency between
    // builds to the caller's barrier (src/xdxr_build.h, dispatch()) and the title puts none between the elements of
    // a batch, so elements must not share scratch bytes. The first source-only element ends the batch (below).
    constexpr UINT64 kAlign = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
    constexpr UINT64 kNoRegion = ~0ull;
    std::vector<D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC> builds;
    std::vector<UINT64> regions;
    std::vector<UINT64> rooms;  // per element: the destination's room (rehydrated_room)
    // Per element: the one-triangle geometry built instead of one whose structure does not fit its room. Sized once,
    // so the pointers the descriptions keep stay valid.
    std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> singles(count);
    builds.reserve(count);
    regions.reserve(count);
    rooms.reserve(count);
    UINT64 scratch_bytes = 0;
    bool source_only = false;
    static std::atomic<std::uint64_t> s_whole{0}, s_single{0}, s_unfit{0}, s_no_room{0};
    for (UINT i = 0; i < count; ++i) {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build = descs[i];
        const UINT64 src = source_data ? source_data[i] : build.SourceAccelerationStructureData;
        std::uint32_t stream_flags = 0;
        const UINT64 room = rehydrated_room(src, &stream_flags);
        // A rehydrate is a fresh build from geometry. The title sets SourceAccelerationStructureData to the
        // dehydrated stream (measured on the retail title), and the layer reads that address for an update
        // (d3d12_com.cpp, BuildRaytracingAccelerationStructure), so with PERFORM_UPDATE kept it would refit from
        // the stream's bytes. Both are cleared.
        if (build.Inputs.Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE) {
            static std::atomic<bool> s_update_logged{false};
            log_once(s_update_logged,
                     "RehydrateRaytracingAccelerationStructureBatched2X: an element asks for PERFORM_UPDATE; it is "
                     "built afresh from its geometry instead (the source is the dehydrated stream)");
            build.Inputs.Flags &= ~D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
        }
        build.SourceAccelerationStructureData = 0;
        UINT64 region = kNoRegion;
        if (has_geometry(build.Inputs)) {
            if (!room && s_no_room.fetch_add(1, std::memory_order_relaxed) < 8) {
                // No readable stream (none given, or one the CPU cannot read here): built in place as before, the
                // destination taken to be as large as our layout (what the prebuild info told the title).
                MEMORY_BASIC_INFORMATION region{};
                VirtualQuery(reinterpret_cast<const void*>(static_cast<std::uintptr_t>(src + 0x104)), &region,
                             sizeof(region));
                std::fprintf(stderr, "[rt] XboxBvhBuilder shim: a rehydrated structure's console size could not be "
                                     "read from its stream at %#llx (state %#lx protect %#lx type %#lx; window pages "
                                     "%s); built in place at %#llx\n",
                             static_cast<unsigned long long>(src), region.State, region.Protect, region.Type,
                             rt::xmem::g_window_pages.load() ? "known" : "none",
                             static_cast<unsigned long long>(build.DestAccelerationStructureData));
            }
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO prebuild{};
            if (device) device->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs, &prebuild);
            if (room && prebuild.ResultDataMaxSizeInBytes > room) {
                {
                    static std::atomic<unsigned> s_rooms_told{0};
                    if (s_rooms_told.fetch_add(1, std::memory_order_relaxed) < 4)
                        std::fprintf(stderr, "[rt] XboxBvhBuilder shim: rehydrate at %#llx (stream flags %#x): console "
                                             "size %llu bytes, ours %llu: built from its first triangle only\n",
                                     static_cast<unsigned long long>(build.DestAccelerationStructureData),
                                     stream_flags, static_cast<unsigned long long>(room),
                                     static_cast<unsigned long long>(prebuild.ResultDataMaxSizeInBytes));
                }
                // The first triangle of the first geometry that has one, as a geometry of its own.
                const D3D12_RAYTRACING_GEOMETRY_DESC* first = nullptr;
                for (UINT g = 0; g < build.Inputs.NumDescs && !first; ++g) {
                    const D3D12_RAYTRACING_GEOMETRY_DESC* gd =
                        build.Inputs.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS
                            ? (build.Inputs.ppGeometryDescs ? build.Inputs.ppGeometryDescs[g] : nullptr)
                            : &build.Inputs.pGeometryDescs[g];
                    if (!gd || gd->Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) continue;
                    const bool indexed = gd->Triangles.IndexFormat != DXGI_FORMAT_UNKNOWN;
                    if ((indexed ? gd->Triangles.IndexCount : gd->Triangles.VertexCount) >= 3) first = gd;
                }
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO single{};
                if (first && device) {
                    singles[i] = *first;
                    if (singles[i].Triangles.IndexFormat != DXGI_FORMAT_UNKNOWN)
                        singles[i].Triangles.IndexCount = 3;
                    else
                        singles[i].Triangles.VertexCount = 3;
                    build.Inputs.NumDescs = 1;
                    build.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
                    build.Inputs.pGeometryDescs = &singles[i];
                    device->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs, &single);
                }
                if (!first || !single.ResultDataMaxSizeInBytes || single.ResultDataMaxSizeInBytes > room) {
                    // Nothing that fits: no build. The destination keeps the stream's bytes.
                    s_unfit.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                prebuild = single;
                s_single.fetch_add(1, std::memory_order_relaxed);
            } else {
                s_whole.fetch_add(1, std::memory_order_relaxed);
            }
            if (!build.ScratchAccelerationStructureData && device) {
                const UINT64 size = prebuild.ScratchDataSizeInBytes ? prebuild.ScratchDataSizeInBytes : 1;
                region = scratch_bytes;
                scratch_bytes += (size + kAlign - 1) / kAlign * kAlign;
            }
        } else if (src) {
            source_only = true;
            break;
        }
        builds.push_back(build);
        regions.push_back(region);
        rooms.push_back(room);
    }
    const UINT64 scratch = scratch_bytes ? rehydrate_scratch(self, device, list4, scratch_bytes) : 0;

    for (std::size_t i = 0; i < builds.size(); ++i) {
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC& build = builds[i];
        if (has_geometry(build.Inputs)) {
            // The title passes ScratchAccelerationStructureData = 0 (measured on the retail title).
            // Supply a scratch region to prevent d3d12_com.cpp from poisoning the command list.
            if (regions[i] != kNoRegion && scratch) build.ScratchAccelerationStructureData = scratch + regions[i];
            if (!build.ScratchAccelerationStructureData) {
                static std::atomic<bool> s_scratch_logged{false};
                log_once(s_scratch_logged,
                         "RehydrateRaytracingAccelerationStructureBatched2X: ScratchAccelerationStructureData is 0 "
                         "and could not be allocated; skipping build to avoid command list rejection");
                continue;
            }
#ifdef LEONIDA_VULKAN_DEVICE
            if (rooms[i])
                rt::d3d12::note_acceleration_structure_room(device, build.DestAccelerationStructureData, rooms[i]);
#endif
            if (device) {
                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO written{};
                device->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs, &written);
                note_written_range("rehydrated blas", build.DestAccelerationStructureData,
                                   written.ResultDataMaxSizeInBytes, static_cast<UINT>(build.Inputs.Flags));
            }
            list4->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
        } else {
            static std::atomic<bool> s_empty_logged{false};
            log_once(s_empty_logged,
                     "RehydrateRaytracingAccelerationStructureBatched2X: element has NumDescs == 0 with no geometry; "
                     "skipping empty structure");
        }
    }
    {
        static std::atomic<std::uint64_t> s_told{0};
        const std::uint64_t n = s_told.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((n & (n - 1)) == 0)
            std::fprintf(stderr, "[rt] XboxBvhBuilder shim: %llu rehydrate call(s): %llu structures fit their room, "
                                 "%llu built from one triangle (too large for it), %llu not built (nothing fits), %llu "
                                 "with no readable room (built in place)\n",
                         static_cast<unsigned long long>(n),
                         static_cast<unsigned long long>(s_whole.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(s_single.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(s_unfit.load(std::memory_order_relaxed)),
                         static_cast<unsigned long long>(s_no_room.load(std::memory_order_relaxed)));
    }
    if (source_only) {
        // docs/bvh.md §4/§4a: the rehydrate source is a dehydrated console stream (element chunks,
        // level table, box overrides), not a D3D12 serialized blob. On the XDXR path, a DESERIALIZE
        // copy is replayed as a byte clone, so the destination would hold dehydrated bytes the GPU
        // traversal shaders cannot read. Rehydration requires geometry reconstruction (docs/bvh.md §4a).
        // Refusing source-only rehydration without geometry.
        static std::atomic<bool> s_source_only_logged{false};
        log_once(s_source_only_logged,
                 "RehydrateRaytracingAccelerationStructureBatched2X: source-only dehydrated stream without "
                 "geometry cannot be deserialized as a D3D12 BLAS (docs/bvh.md §4/§4a); refusing element");
    }
    if (device) device->Release();
    list4->Release();
    return source_only ? E_NOTIMPL : S_OK;
}

constexpr const char kDll[] = "xboxbvhbuilder.dll";

const rt::shims::Entry kEntries[] = {
    {kDll, "??0XboxBvhBuilder@@QEAA@PEAUID3D12Device11@@_N@Z", reinterpret_cast<void*>(&builder_construct)},
    {kDll, "??1XboxBvhBuilder@@QEAA@XZ", reinterpret_cast<void*>(&builder_destruct)},
    {kDll, "?Set64BitMortonCodes@XboxBvhBuilder@@QEAAX_N@Z", reinterpret_cast<void*>(&builder_set_64bit_morton_codes)},
    {kDll,
     "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_"
     "STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z",
     reinterpret_cast<void*>(&builder_prebuild_info)},
    {kDll,
     "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12"
     "XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
     reinterpret_cast<void*>(&builder_build_batched2x)},
    {kDll,
     "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_"
     "RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
     reinterpret_cast<void*>(&builder_copy)},
    {kDll,
     "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_"
     "RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
     reinterpret_cast<void*>(&builder_build)},
    {kDll,
     "?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBU"
     "D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z",
     reinterpret_cast<void*>(&builder_rehydrate_batched2x)},
};

bool shim_enabled() {
    char value[16] = {};
    size_t length = 0;
    if (getenv_s(&length, value, "LEONIDA_DLL_BVH") == 0 && length > 1 && value[0] == '1') return false;  // ours
    length = 0;
    if (getenv_s(&length, value, "LEONIDA_BVH_BUILDER") != 0 || length <= 1) return true;
    return _stricmp(value, "console") != 0;
}

// Registers the table unless LEONIDA_BVH_BUILDER=console asks for the bundled DLL.
struct Registration {
    Registration() {
        if (!shim_enabled()) return;
        static rt::shims::Table table(kEntries, sizeof(kEntries) / sizeof(kEntries[0]));
    }
} g_registration;

}  // namespace

namespace rt::shims {
// For tests: the builders constructed and not destroyed.
size_t xbox_bvh_builder_live_count() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_builders.size();
}
}  // namespace rt::shims
