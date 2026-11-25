// XboxBvhBuilder.dll's build entry points, served by the runtime's own XDXR BVH4 builder (src/xdxr_build.h).
//
// The package's XboxBvhBuilder.dll builds acceleration structures in AMD's layout on the GPU: its batched build records
// a chain of console compute programs (Morton codes, radix sort, PLOC-style tree build) on the title's command list.
// Translated, that chain hung the GPU in the first loading screen (a bvh/RadixAlloc dispatch, then a Windows TDR,
// then the title's GPU watchdog fatal, 2026-11-24/25). The layer already turns a standard
// BuildRaytracingAccelerationStructure into an XDXR BVH4 build with its own compute shader (the layout the title's
// translated ray-tracing shaders walk), so the DLL's build, copy and sizing exports are answered by that path and the
// rest of the DLL (constructor, rehydration, validation) stays the DLL's. Design after five-star-recomp
// (github.com/ThatRecompDev/five-star-recomp, docs/raytracing.md, GPL-3.0), which also replaces the DLL's GPU builds
// with its own builder and keeps its CPU rehydration; the code here is ours.
//
// The title sizes its structures with the DLL's GetRaytracingAccelerationStructurePrebuildInfo, so that export answers
// with this builder's XdxrLayout (the same sizes ID3D12Device5::GetRaytracingAccelerationStructurePrebuildInfo gives),
// and what is built fits what the title allocated. LEONIDA_DLL_BVH=1 keeps the DLL's own GPU builder.
#include <windows.h>
#include <d3d12.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../src/xdxr_build.h"
#include "../src/xdxr_descs.h"
#include "d3d12_xs_device.h"
#include "shims.h"

namespace {

using leonida::xdxr::BatchedDesc2;

std::atomic<std::uint64_t> g_blas{0}, g_tlas{0}, g_skipped{0}, g_copies{0}, g_prebuilds{0};

void note(std::atomic<std::uint64_t>& counter, const char* what) {
    const std::uint64_t n = counter.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & (n - 1)) == 0)
        std::fprintf(stderr, "[rt] xbox_bvh: %llu %s (runtime XDXR builder; %llu BLAS, %llu TLAS, %llu left out)\n",
                     static_cast<unsigned long long>(n), what,
                     static_cast<unsigned long long>(g_blas.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(g_tlas.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(g_skipped.load(std::memory_order_relaxed)));
}

// The PC list behind the console list proxy the title passes, as ID3D12GraphicsCommandList4 (a reference the caller
// releases), or null.
ID3D12GraphicsCommandList4* pc_list(void* list) {
    ID3D12GraphicsCommandList* real = rt::d3d12_xs::command_list_proxy_real_list(list);
    if (!real) real = static_cast<ID3D12GraphicsCommandList*>(list);  // not a proxy: a PC list already
    ID3D12GraphicsCommandList4* list4 = nullptr;
    if (!real || FAILED(real->QueryInterface(IID_PPV_ARGS(&list4)))) {
        static std::atomic<bool> told{false};
        if (!told.exchange(true))
            std::fprintf(stderr, "[rt] xbox_bvh: a build on list %p that is not a D3D12 list: left out\n", list);
        return nullptr;
    }
    return list4;
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

// ?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBU
// D3D12XBOX_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z: `count` records of 0x50 bytes (xdxr_descs.h).
void batched_build(void* /*self*/, void* list, UINT count, const BatchedDesc2* descs) {
    if (!count || !descs) return;
    ID3D12GraphicsCommandList4* const list4 = pc_list(list);
    if (!list4) return;
    for (UINT i = 0; i < count; ++i) {
        const BatchedDesc2& b = descs[i];
        if (b.stream.size_in_bytes) {
            static std::atomic<bool> told{false};
            if (!told.exchange(true))
                std::fprintf(stderr, "[rt] xbox_bvh: a batched build carries an XDXR stream (%llu bytes): built from "
                                     "its inputs alone\n",
                             static_cast<unsigned long long>(b.stream.size_in_bytes));
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC desc{};
        desc.DestAccelerationStructureData = b.dest_acceleration_structure_data;
        std::memcpy(&desc.Inputs, &b.inputs, sizeof(desc.Inputs));
        desc.SourceAccelerationStructureData = b.source_acceleration_structure_data;
        desc.ScratchAccelerationStructureData = b.scratch_acceleration_structure_data;
        build_one(list4, desc, b.num_postbuild_info_descs,
                  reinterpret_cast<const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*>(
                      b.postbuild_info_descs));
    }
    list4->Release();
}

// ?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBU
// D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z
void single_build(void* /*self*/, void* list, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc,
                  UINT postbuild_count, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* postbuild) {
    if (!desc) return;
    ID3D12GraphicsCommandList4* const list4 = pc_list(list);
    if (!list4) return;
    build_one(list4, *desc, postbuild_count, postbuild);
    list4->Release();
}

// ?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4
// D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE@@@Z: the structures are this builder's, so the copy is too.
void copy_structure(void* /*self*/, void* list, D3D12_GPU_VIRTUAL_ADDRESS dest, D3D12_GPU_VIRTUAL_ADDRESS source,
                    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE mode) {
    ID3D12GraphicsCommandList4* const list4 = pc_list(list);
    if (!list4) return;
    list4->CopyRaytracingAccelerationStructure(dest, source, mode);
    list4->Release();
    note(g_copies, "structure copies");
}

// ?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBU
// D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAU
// D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z: this builder's sizes (the device's XDXR prebuild rule:
// triangles or instances, BOX16 for MINIMIZE_MEMORY bottom levels).
void prebuild_info(void* /*self*/, const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* in,
                   const void* /*stream*/, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO* out) {
    if (!out) return;
    *out = {};
    if (!in) return;
    const bool tlas = in->Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    std::uint64_t elements = 0;
    if (tlas) {
        elements = in->NumDescs;
    } else {
        for (UINT i = 0; i < in->NumDescs; ++i) {
            const D3D12_RAYTRACING_GEOMETRY_DESC* gd = geometry(*in, i);
            if (!gd || gd->Type != D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) continue;
            elements += (gd->Triangles.IndexFormat == DXGI_FORMAT_UNKNOWN ? gd->Triangles.VertexCount
                                                                           : gd->Triangles.IndexCount) / 3;
        }
    }
    const bool box16 = !tlas && (in->Flags & D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY);
    leonida::XdxrLayout layout;
    if (!layout.compute(elements > 0xffffffffu ? 0xffffffffu : static_cast<std::uint32_t>(elements), tlas, box16))
        return;
    out->ResultDataMaxSizeInBytes = layout.size;
    out->ScratchDataSizeInBytes = layout.scratch;
    out->UpdateScratchDataSizeInBytes = layout.scratch;
    note(g_prebuilds, "size queries");
}

bool dll_builder_kept() {
    char value[4] = {};
    std::size_t length = 0;
    return getenv_s(&length, value, "LEONIDA_DLL_BVH") == 0 && length > 1 && value[0] == '1';
}

constexpr const char* kDll = "xboxbvhbuilder.dll";

const ::rt::shims::Entry kEntries[] = {
    {kDll,
     "?BuildRaytracingAccelerationStructureBatched2X@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList6@@IPEBUD3D12XBOX_"
     "BUILD_RAYTRACING_ACCELERATION_STRUCTURE_BATCHED_DESC2@@@Z",
     reinterpret_cast<void*>(&batched_build)},
    {kDll,
     "?BuildRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@PEBUD3D12_BUILD_"
     "RAYTRACING_ACCELERATION_STRUCTURE_DESC@@IPEBUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC@@@Z",
     reinterpret_cast<void*>(&single_build)},
    {kDll,
     "?CopyRaytracingAccelerationStructure@XboxBvhBuilder@@QEAAXPEAUID3D12GraphicsCommandList4@@_K1W4D3D12_RAYTRACING_"
     "ACCELERATION_STRUCTURE_COPY_MODE@@@Z",
     reinterpret_cast<void*>(&copy_structure)},
    {kDll,
     "?GetRaytracingAccelerationStructurePrebuildInfo@XboxBvhBuilder@@QEAAXPEBUD3D12_BUILD_RAYTRACING_ACCELERATION_"
     "STRUCTURE_INPUTS@@PEBUD3D12XBOX_XDXR_STREAM_DESC@@PEAUD3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO@@@Z",
     reinterpret_cast<void*>(&prebuild_info)},
};

// Registered unless LEONIDA_DLL_BVH=1. A static table registers at startup, before the loader resolves the title's
// imports; an empty table registers nothing.
const ::rt::shims::Table kTable(kEntries, dll_builder_kept() ? 0 : sizeof(kEntries) / sizeof(kEntries[0]));

}  // namespace
