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
//   - Rehydration is not implemented yet: RehydrateRaytracingAccelerationStructureBatched2X logs once and returns
//     E_NOTIMPL. (Planned: rebuild each structure from the source geometry its description carries.)
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
#include <mutex>
#include <unordered_map>
#include <vector>

#include "d3d12_xs_device.h"
#include "shims.h"
#include "../src/xdxr_descs.h"

namespace {

namespace xdxr = leonida::xdxr;

struct BuilderState {
    ID3D12Device5* device = nullptr;   // a reference held for the builder's lifetime; null if it has no DXR methods
    bool morton64 = false;
};

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

// One build on the PC list, with the title's typeless index formats made plain and unreadable bottom levels left out.
void build_one(ID3D12GraphicsCommandList4* list, D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc,
               UINT postbuild_count, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* postbuild) {
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
    if (state.device) state.device->Release();   // the same storage constructed twice without a destructor
    state = BuilderState{device5, false};
    return self;
}

// ??1XboxBvhBuilder@@QEAA@XZ: ~XboxBvhBuilder().
void builder_destruct(void* self) {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_builders.find(self);
    if (it == g_builders.end()) return;
    if (it->second.device) it->second.device->Release();
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
void builder_build_batched2x(void*, void* list, UINT count, const xdxr::BatchedDesc2* descs) {
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
        build_one(list4, build, n_kept, n_kept ? kept : nullptr);
    }
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

// ?RehydrateRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAJPEAUID3D12GraphicsCommandList8@@IPEBUD3D12_
// BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@PEB_K@Z -> HRESULT. Not implemented yet (see the header comment).
HRESULT builder_rehydrate_batched2x(void*, void*, UINT, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*,
                                    const UINT64*) {
    static std::atomic<bool> s_logged{false};
    log_once(s_logged, "RehydrateRaytracingAccelerationStructureBatched2X is not implemented yet; E_NOTIMPL");
    return E_NOTIMPL;
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
