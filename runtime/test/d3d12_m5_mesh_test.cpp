// Milestone M5 (docs/d3d12_vulkan.md section 6), the mesh part: translated mesh shaders through the D3D12 API on the
// proxy, drawn by DispatchMesh and by ExecuteIndirect with DISPATCH_MESH command signatures, with every pixel compared
// with a CPU reference (error 0). The shaders are translator output (hand-written RDNA2 in test/shaders): ms_m5.asm,
// an NGG wave64 mesh shader of 128 threads written with the corpus's idioms (the group index from v5, the thread id
// from s3 and mbcnt, GS_ALLOC_REQ, one vertex and one primitive export per thread), and ps_m5_mesh.asm. Each mesh group
// draws one quad into cell (first cell + linear group index) of an 8x8 grid on a 64x64 target, in a colour from root
// constants; the mesh PSOs are built from pipeline state streams.
//
// (a) DispatchMesh: a 4x2x2 grid (all three counts reach the shader: 16 cells, so every group got its own index), a
//     3x1x1 grid and a 1x1x1 grid at other first cells and colours, and an empty grid that draws nothing.
// (b) ExecuteIndirect with DISPATCH_MESH signatures, with additive blending, so a cell's colour counts the records
//     whose group count reached it, so each record's group total shows in the pixels (the split of a total into X,
//     Y and Z, and the order groups are numbered in, are not visible: a group sees only its linear index):
//       I1  5 CPU-written records, stride 12, at argument offset 36, one of them empty;
//       I2  4 records, stride 24, copied by the GPU into a default buffer (COPY_DEST -> INDIRECT_ARGUMENT) at offset
//           48, with a count buffer holding 3 (so the 4th record is not drawn);
//       I3  a count of 0: nothing;
//       I4  a count of 7 against MaxCommandCount 2: the first two records only;
//       I5  20 records of 1 to 4 groups, stride 24 with zeroed padding (a wrong stride would read empty records),
//           without a count buffer: one call of 20, or calls of at most maxDrawIndirectCount records, or 20 calls
//           without multiDrawIndirect.
//     Without drawIndirectCount, I2..I4 are left out, and a count-buffer call must fail as unsupported.
//     --single-calls and --no-count-buffers turn multiDrawIndirect and drawIndirectCount off on the layer's device
//     (registered as d3d12_m5_mesh_fallbacks), so those paths run on every GPU that runs this test (64-wide mesh
//     subgroups), whatever its features; --max-draw-indirect-count N lowers the
//     layer's limit (d3d12_m5_mesh_chunks runs N = 3, so I5 is split into calls of 3 records at stride 24). The
//     one-call-per-record split itself only shows as a validation error on a device without multiDrawIndirect.
// (c) Refusals: a DISPATCH_MESH signature whose stride is below 12 is E_INVALIDARG; one with a root constant before
//     DISPATCH_MESH is E_NOTIMPL (multi-argument signatures are not expanded); one on a device without
//     VK_EXT_mesh_shader is E_NOTIMPL; a DISPATCH_MESH signature run with a VS PSO, and a DRAW signature with the mesh
//     PSO, fail at execution (ReplayResult::invalid_draw).
// Needs VK_EXT_mesh_shader (exit 77 without it). No game data. Set LEONIDA_VK_VALIDATION=1 to run under
// VK_LAYER_KHRONOS_validation (0 errors required).
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

#ifndef LEONIDA_TEST_M5_MESH_DIR
#error "LEONIDA_TEST_M5_MESH_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slot 9: handle in RAX

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// One pipeline state stream subobject as d3dx12.h declares it (CD3DX12_PIPELINE_STATE_STREAM_SUBOBJECT).
template <D3D12_PIPELINE_STATE_SUBOBJECT_TYPE Type, typename Inner>
struct alignas(void*) Subobject {
    D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type = Type;
    Inner inner;
};
using RootSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, ID3D12RootSignature*>;
using MsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, D3D12_SHADER_BYTECODE>;
using VsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, D3D12_SHADER_BYTECODE>;
using PsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, D3D12_SHADER_BYTECODE>;
using TopologySubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, D3D12_PRIMITIVE_TOPOLOGY_TYPE>;
using TargetsSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, D3D12_RT_FORMAT_ARRAY>;
using BlendSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, D3D12_BLEND_DESC>;
using DepthStencilSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, D3D12_DEPTH_STENCIL_DESC>;
using RasterizerSubobject = Subobject<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, D3D12_RASTERIZER_DESC>;

struct Stream {
    std::vector<std::uint8_t> bytes;
    template <typename S>
    Stream& add(const S& subobject) {
        static_assert(sizeof(S) % sizeof(void*) == 0);
        const std::size_t at = bytes.size();
        bytes.resize(at + sizeof(S));
        std::memcpy(bytes.data() + at, &subobject, sizeof(S));
        return *this;
    }
};

using Pixel = std::array<std::uint8_t, 4>;

struct Image {
    std::vector<Pixel> pixels;
    explicit Image(Pixel fill) : pixels(64 * 64, fill) {}
    Pixel& at(UINT x, UINT y) { return pixels[std::size_t(y) * 64 + x]; }
    // Cell k of the 8x8 grid (cells past 63 are off the target).
    template <typename F>
    void cell(unsigned k, F&& f) {
        if (k >= 64) return;
        for (UINT y = 0; y < 8; ++y)
            for (UINT x = 0; x < 8; ++x) f(at((k & 7) * 8 + x, (k >> 3) * 8 + y));
    }
};

unsigned compare(const char* what, const std::uint8_t* got, UINT pitch, Image& want) {
    unsigned mismatches = 0;
    for (UINT y = 0; y < 64; ++y)
        for (UINT x = 0; x < 64; ++x) {
            const std::uint8_t* g = got + std::size_t(y) * pitch + x * 4;
            const Pixel& w = want.at(x, y);
            if (std::memcmp(g, w.data(), 4) == 0) continue;
            if (mismatches++ < 6)
                std::fprintf(stderr, "%s pixel (%u, %u): got %u,%u,%u,%u want %u,%u,%u,%u\n", what, x, y, g[0], g[1],
                             g[2], g[3], w[0], w[1], w[2], w[3]);
        }
    return mismatches;
}

// The root constants of ms_m5.asm: s0 = the first cell, s1, s2 = red and green (k / 255).
struct Constants {
    UINT first;
    float r, g;
};
Constants constants(UINT first, unsigned r, unsigned g) { return {first, float(r) / 255.0f, float(g) / 255.0f}; }

}  // namespace

int main(int argc, char** argv) {
    bool single_calls = false, no_count_buffers = false;
    std::uint32_t max_draw_count = 0;  // 0: the device's maxDrawIndirectCount
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--single-calls") == 0) single_calls = true;
        else if (std::strcmp(argv[i], "--no-count-buffers") == 0) no_count_buffers = true;
        else if (std::strcmp(argv[i], "--max-draw-indirect-count") == 0 && i + 1 < argc)
            max_draw_count = std::uint32_t(std::strtoul(argv[++i], nullptr, 10));
        else {
            std::fprintf(stderr, "usage: d3d12_m5_mesh_test [--single-calls] [--no-count-buffers] "
                                 "[--max-draw-indirect-count N]\n");
            return 2;
        }
    }
    vktest::Device d;
    d.request_mesh_shader = true;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;

    const std::filesystem::path root_dir = LEONIDA_TEST_M5_MESH_DIR;
    leonida::ShaderLibrary library;
    EXPECT(library.load((root_dir / "spirv").string()));
    const auto container_of = [&](const char* name, const char* stage, std::vector<std::uint8_t>& out) {
        const auto program = read_file(root_dir / "fssm" / (std::string(name) + ".fssm"));
        if (program.size() < 4 || !library.find(stage, leonida::sha256_hex(program))) return false;
        out = shader_container(program);
        return true;
    };
    std::vector<std::uint8_t> ms_bytes, ps_bytes, vs_rect, ps_flat;
    EXPECT(container_of("ms_m5", "ms", ms_bytes) && container_of("ps_m5_mesh", "ps", ps_bytes) &&
           container_of("vs_m3", "vs", vs_rect) && container_of("ps_m5_flat", "ps", ps_flat));
    const D3D12_SHADER_BYTECODE ms{ms_bytes.data(), ms_bytes.size()}, ps{ps_bytes.data(), ps_bytes.size()};

    const auto device_info = [&](bool mesh) {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_device_address = true;
        info.multi_draw_indirect = d.multi_draw_indirect && !single_calls;
        info.draw_indirect_count = d.draw_indirect_count && !no_count_buffers;
        info.max_draw_indirect_count =
            max_draw_count ? std::min(max_draw_count, d.max_draw_indirect_count) : d.max_draw_indirect_count;
        info.draw_indirect_first_instance = d.draw_indirect_first_instance;
        info.mesh_shader = mesh;
        info.subgroup = d.subgroup;  // so the wave64 mesh stage asks for 64 where the device can require it
        info.shader_dir = (root_dir / "spirv").string();
        return info;
    };
    const auto make_device = [&](bool mesh, Ref<ID3D12Device>& device) {
        ID3D12Device* inner = nullptr;
        if (FAILED(rt::d3d12::create_vulkan_device(device_info(mesh), IID_PPV_ARGS(&inner)))) return false;
        if (FAILED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                     reinterpret_cast<void**>(&device.p))))
            return false;
        rt::d3d12::set_device_outer(inner, device.p);
        return true;
    };
    const auto mesh_signature = [](ID3D12Device* device, UINT stride, Ref<ID3D12CommandSignature>& out) {
        D3D12_INDIRECT_ARGUMENT_DESC arg{};
        arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
        const D3D12_COMMAND_SIGNATURE_DESC desc{stride, 1, &arg, 0};
        out.reset();
        return device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&out.p));
    };

    // (c) Without the extension, a DISPATCH_MESH signature is not implemented.
    {
        Ref<ID3D12Device> device;
        Ref<ID3D12CommandSignature> sig;
        EXPECT(make_device(false, device));
        EXPECT(mesh_signature(device.p, sizeof(D3D12_DISPATCH_MESH_ARGUMENTS), sig) == E_NOTIMPL && !sig.p);
    }
    if (!d.mesh_shader) {
        std::printf("no VK_EXT_mesh_shader on this device: only the refusal without it was checked\n");
        return 77;
    }
    // ms_m5 is wave64 (not built with --wave64-emulate): it needs 64-wide subgroups in the mesh stage. Decided from
    // the library entry and the device's caps, as the layer decides it, so a device that cannot run it skips before
    // any mesh PSO is created; the PSO's own request is checked against this below.
    const leonida::ShaderEntry* ms_entry =
        library.find("ms", leonida::sha256_hex(read_file(root_dir / "fssm" / "ms_m5.fssm")));
    EXPECT(ms_entry);
    const leonida::SubgroupRequest ms_request = leonida::decide_subgroup_size(*ms_entry, d.subgroup);
    if (ms_request.outcome == leonida::SubgroupOutcome::unknown_device ||
        ms_request.outcome == leonida::SubgroupOutcome::unknown_stage) {
        std::fprintf(stderr, "the mesh stage's subgroup size was not decided (outcome %d)\n", int(ms_request.outcome));
        return 1;
    }
    if (ms_request.outcome == leonida::SubgroupOutcome::unavailable) {
        std::printf("SKIP: the wave64 mesh shader cannot run 64-wide here (subgroup sizes %u..%u, default %u, "
                    "requirable stages 0x%x)\n", d.subgroup.min_size, d.subgroup.max_size, d.subgroup.default_size,
                    unsigned(d.subgroup.required_stages));
        return 77;
    }
    const bool counts = d.draw_indirect_count && !no_count_buffers;
    if (!d.multi_draw_indirect || single_calls) std::printf("note: no multiDrawIndirect: the layer issues one call per record\n");
    if (!counts) std::printf("note: no drawIndirectCount: the count-buffer calls (I2, I3, I4) are not run\n");

    unsigned direct_mismatches = 0, indirect_mismatches = 0, records_drawn = 0;
    const char* subgroup = "?";
    {
        Ref<ID3D12Device> device;
        EXPECT(make_device(true, device));
        if (rt::d3d12::device_push_constants_need_fallback(device.p)) {
            std::printf("SKIP: the PSOs need 144 push-constant bytes and this GPU has %u\n",
                        rt::d3d12::device_max_push_constants_size(device.p));
            return 77;
        }
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        UINT64 fence_value = 0;
        // Runs one list and waits; the replay result of the submission.
        const auto run = [&](ID3D12GraphicsCommandList* list) {
            ID3D12CommandList* lists[] = {list};
            queue->ExecuteCommandLists(1, lists);
            const auto result = rt::d3d12::last_execute_result(queue.p);
            ++fence_value;
            if (FAILED(queue->Signal(fence.p, fence_value)) || FAILED(fence->SetEventOnCompletion(fence_value, nullptr)))
                return leonida::ReplayResult::unsupported;
            return result;
        };
        const auto new_buffer = [&](const D3D12_HEAP_PROPERTIES& heap, UINT64 bytes, D3D12_RESOURCE_STATES state,
                                    Ref<ID3D12Resource>& out) {
            const auto desc = buffer_desc(bytes);
            return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                             IID_PPV_ARGS(&out.p)));
        };

        // [0] three 32-bit constants: s0..s2.
        Ref<ID3D12RootSignature> root;
        {
            D3D12_ROOT_PARAMETER param{};
            param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            param.Constants = {0, 0, 3};
            D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
            root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
            root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
            const auto blob = serialize_root_signature(root_desc);
            EXPECT(!blob.empty() &&
                   SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
        }

        // The PSOs: the mesh one opaque and additive, from streams; a VS PSO (vs_m3 + ps_m5_flat) for (c).
        D3D12_RT_FORMAT_ARRAY formats{};
        formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        formats.NumRenderTargets = 1;
        D3D12_RASTERIZER_DESC raster{};
        raster.FillMode = D3D12_FILL_MODE_SOLID;
        raster.CullMode = D3D12_CULL_MODE_NONE;
        raster.DepthClipEnable = TRUE;
        const D3D12_DEPTH_STENCIL_DESC no_depth{FALSE, D3D12_DEPTH_WRITE_MASK_ZERO, D3D12_COMPARISON_FUNC_ALWAYS};
        D3D12_BLEND_DESC opaque{};
        opaque.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        D3D12_BLEND_DESC additive = opaque;
        additive.RenderTarget[0].BlendEnable = TRUE;
        additive.RenderTarget[0].SrcBlend = additive.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
        additive.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
        additive.RenderTarget[0].SrcBlendAlpha = additive.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ONE;
        additive.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
        Ref<ID3D12Device2> device2;
        EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device2.p))));
        const auto stream_pso = [&](Stream& stream, Ref<ID3D12PipelineState>& out) {
            const D3D12_PIPELINE_STATE_STREAM_DESC desc{stream.bytes.size(), stream.bytes.data()};
            return SUCCEEDED(device2->CreatePipelineState(&desc, IID_PPV_ARGS(&out.p)));
        };
        Ref<ID3D12PipelineState> mesh_opaque, mesh_additive, vs_pso;
        {
            Stream s;
            s.add(RootSubobject{.inner = root.p}).add(MsSubobject{.inner = ms}).add(PsSubobject{.inner = ps});
            s.add(TargetsSubobject{.inner = formats}).add(RasterizerSubobject{.inner = raster});
            s.add(DepthStencilSubobject{.inner = no_depth});
            Stream a = s;
            s.add(BlendSubobject{.inner = opaque});
            a.add(BlendSubobject{.inner = additive});
            EXPECT(stream_pso(s, mesh_opaque) && stream_pso(a, mesh_additive));
            // The PSO's mesh-stage request must be the one decided above.
            const auto* requests = rt::d3d12::pipeline_subgroup_requests(mesh_opaque.p);
            EXPECT(requests);
            bool mesh_request = false;
            for (const auto& request : *requests) {
                if (request.stage != VK_SHADER_STAGE_MESH_BIT_EXT) continue;
                mesh_request = true;
                if (request.outcome != ms_request.outcome) {
                    std::fprintf(stderr, "the mesh PSO's subgroup outcome %d is not the predicted %d\n",
                                 int(request.outcome), int(ms_request.outcome));
                    return 1;
                }
                switch (request.outcome) {
                case leonida::SubgroupOutcome::required: subgroup = "required subgroup size 64"; break;
                case leonida::SubgroupOutcome::device_fixed: subgroup = "the device's only subgroup size, 64"; break;
                case leonida::SubgroupOutcome::device_default: subgroup = "the default subgroup size, 64"; break;
                case leonida::SubgroupOutcome::emulated: subgroup = "wave64 emulation"; break;
                default: return 1;  // the unavailable and unknown outcomes returned before the PSOs
                }
            }
            EXPECT(mesh_request);
            Stream v;
            v.add(RootSubobject{.inner = root.p}).add(VsSubobject{.inner = {vs_rect.data(), vs_rect.size()}});
            v.add(PsSubobject{.inner = {ps_flat.data(), ps_flat.size()}});
            v.add(TopologySubobject{.inner = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE});
            v.add(TargetsSubobject{.inner = formats}).add(RasterizerSubobject{.inner = raster});
            v.add(DepthStencilSubobject{.inner = no_depth}).add(BlendSubobject{.inner = opaque});
            EXPECT(stream_pso(v, vs_pso));
        }

        // The target, its RTV and readback.
        Ref<ID3D12Resource> target, readback;
        D3D12_RESOURCE_DESC rt_desc{};
        rt_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rt_desc.Width = rt_desc.Height = 64;
        rt_desc.DepthOrArraySize = 1;
        rt_desc.MipLevels = 1;
        rt_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rt_desc.SampleDesc.Count = 1;
        rt_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &rt_desc,
                                                         D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
                                                         IID_PPV_ARGS(&target.p))));
        constexpr UINT pitch = 256;
        EXPECT(new_buffer(readback_heap, pitch * 64, D3D12_RESOURCE_STATE_COPY_DEST, readback));
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv{SIZE_T(reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p))};
        device->CreateRenderTargetView(target.p, nullptr, rtv);

        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12GraphicsCommandList6> list6;
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(list->Close()));
        EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list6.p))));
        // Reset, COPY_SOURCE -> RENDER_TARGET, a clear, the pipeline and the target state.
        const auto begin = [&](ID3D12PipelineState* pso, const FLOAT (&clear)[4]) {
            if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, pso))) return false;
            const auto barrier = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(rtv, clear, 0, nullptr);
            list->SetGraphicsRootSignature(root.p);
            const D3D12_VIEWPORT viewport{0, 0, 64, 64, 0, 1};
            const D3D12_RECT scissor{0, 0, 64, 64};
            list->RSSetViewports(1, &viewport);
            list->RSSetScissorRects(1, &scissor);
            list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
            return true;
        };
        const auto end = [&]() {
            const auto barrier = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = readback.p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            dst.PlacedFootprint = {0, {DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 1, pitch}};
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = target.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            return SUCCEEDED(list->Close());
        };
        const auto set_constants = [&](const Constants& c) { list->SetGraphicsRoot32BitConstants(0, 3, &c, 0); };
        const auto check = [&](const char* what, Image& want) {
            std::uint8_t* mapped = nullptr;
            const D3D12_RANGE range{0, pitch * 64};
            if (FAILED(readback->Map(0, &range, reinterpret_cast<void**>(&mapped)))) return 64u * 64u;
            const unsigned bad = compare(what, mapped, pitch, want);
            readback->Unmap(0, nullptr);
            return bad;
        };

        // ===== (a) DispatchMesh =====
        {
            // 128/255 exactly, so the UNORM store cannot round either way (0.5 is 127.5, stored as 127 on some GPUs).
            const FLOAT grey[4] = {128.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f, 1.0f};
            Image want({128, 128, 128, 255});
            EXPECT(begin(mesh_opaque.p, grey));
            set_constants(constants(0, 200, 40));
            list6->DispatchMesh(4, 2, 2);
            for (unsigned k = 0; k < 16; ++k) want.cell(k, [](Pixel& p) { p = {200, 40, 0, 255}; });
            set_constants(constants(20, 60, 220));
            list6->DispatchMesh(3, 1, 1);
            for (unsigned k = 20; k < 23; ++k) want.cell(k, [](Pixel& p) { p = {60, 220, 0, 255}; });
            set_constants(constants(40, 255, 255));
            list6->DispatchMesh(0, 1, 1);  // empty: nothing
            set_constants(constants(63, 10, 250));
            list6->DispatchMesh(1, 1, 1);
            want.cell(63, [](Pixel& p) { p = {10, 250, 0, 255}; });
            EXPECT(end());
            EXPECT(run(list.p) == leonida::ReplayResult::success);
            direct_mismatches = check("DispatchMesh", want);
        }

        // ===== (b) ExecuteIndirect with DISPATCH_MESH =====
        {
            Ref<ID3D12CommandSignature> sig12, sig24, refused;
            EXPECT(SUCCEEDED(mesh_signature(device.p, 12, sig12)) && SUCCEEDED(mesh_signature(device.p, 24, sig24)));
            EXPECT(mesh_signature(device.p, 8, refused) == E_INVALIDARG && !refused.p);  // (c) stride below a record
            {   // (c) a root constant before DISPATCH_MESH: not expanded, so refused rather than misread
                D3D12_INDIRECT_ARGUMENT_DESC args[2] = {};
                args[0].Type = D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT;
                args[0].Constant = {0, 0, 1};
                args[1].Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
                const D3D12_COMMAND_SIGNATURE_DESC desc{16, 2, args, 0};
                EXPECT(device->CreateCommandSignature(&desc, root.p, IID_PPV_ARGS(&refused.p)) == E_NOTIMPL &&
                       !refused.p);
            }

            // The CPU-written records (I1, I4, I5) in an upload buffer; I2's records and the count words staged in a
            // second one and copied into default buffers by the GPU.
            Ref<ID3D12Resource> args_cpu, staging, args_gpu, count_gpu;
            EXPECT(new_buffer(upload_heap, 4096, D3D12_RESOURCE_STATE_GENERIC_READ, args_cpu));
            EXPECT(new_buffer(upload_heap, 4096, D3D12_RESOURCE_STATE_GENERIC_READ, staging));
            EXPECT(new_buffer(default_heap, 1024, D3D12_RESOURCE_STATE_COPY_DEST, args_gpu));
            EXPECT(new_buffer(default_heap, 256, D3D12_RESOURCE_STATE_COPY_DEST, count_gpu));
            std::uint8_t* cpu = nullptr;
            std::uint8_t* stage = nullptr;
            EXPECT(SUCCEEDED(args_cpu->Map(0, nullptr, reinterpret_cast<void**>(&cpu))));
            EXPECT(SUCCEEDED(staging->Map(0, nullptr, reinterpret_cast<void**>(&stage))));
            std::memset(cpu, 0xA5, 4096);  // bytes outside the records are never read
            std::memset(stage, 0xA5, 4096);
            const auto put = [](std::uint8_t* at, UINT x, UINT y, UINT z) {
                const D3D12_DISPATCH_MESH_ARGUMENTS record{x, y, z};
                std::memcpy(at, &record, sizeof(record));
            };
            // Each call draws from its first cell with an increment of colour per record; cell first + j gets the
            // increment once for each drawn record of more than j groups.
            const FLOAT black[4] = {0, 0, 0, 1};
            Image want({0, 0, 0, 255});
            const auto expect_records = [&](UINT first, unsigned dr, unsigned dg, const std::vector<unsigned>& groups) {
                for (const unsigned n : groups) {
                    for (unsigned j = 0; j < n; ++j)
                        want.cell(first + j, [&](Pixel& p) {
                            p[0] = std::uint8_t(std::min(255u, p[0] + dr));
                            p[1] = std::uint8_t(std::min(255u, p[1] + dg));
                        });
                    ++records_drawn;
                }
            };
            // I1: offset 36, stride 12: 3, 0, 2, 4 and 1 groups (as 3x1x1, 0x5x1, 1x2x1, 2x1x2, 1x1x1).
            put(cpu + 36, 3, 1, 1);
            put(cpu + 48, 0, 5, 1);
            put(cpu + 60, 1, 2, 1);
            put(cpu + 72, 2, 1, 2);
            put(cpu + 84, 1, 1, 1);
            expect_records(0, 32, 0, {3, 0, 2, 4, 1});
            // I4: offset 256, stride 12, three records (1, 2 and 4 groups) under a count of 7 and MaxCommandCount 2.
            put(cpu + 256, 1, 1, 1);
            put(cpu + 268, 2, 1, 1);
            put(cpu + 280, 4, 1, 1);
            if (counts) expect_records(40, 0, 64, {1, 2});
            // I5: offset 512, stride 24, 20 records of 1 + r % 4 groups (as 1x1, then 2x1x1, 1x3x1, 2x1x2), each
            // followed by 12 zero bytes: read at stride 12, half the records would be empty.
            std::memset(cpu + 512, 0, 20 * 24);
            std::vector<unsigned> i5;
            for (unsigned r = 0; r < 20; ++r) {
                static constexpr UINT shapes[4][3] = {{1, 1, 1}, {2, 1, 1}, {1, 3, 1}, {2, 1, 2}};
                put(cpu + 512 + 24 * r, shapes[r % 4][0], shapes[r % 4][1], shapes[r % 4][2]);
                i5.push_back(1 + r % 4);
            }
            expect_records(48, 8, 8, i5);
            // I2 (staged at 0, copied to args_gpu at 48): stride 24, 4, 3, 5 and 8 groups, the count 3 drops the 8.
            put(stage + 0, 2, 2, 1);
            put(stage + 24, 1, 1, 3);
            put(stage + 48, 5, 1, 1);
            put(stage + 72, 8, 1, 1);
            if (counts) expect_records(16, 0, 32, {4, 3, 5});
            // The count words, staged at 2048: 3 (I2, copied to count_gpu at 16), 0 (I3, at 20), 7 (I4, at 24).
            const std::uint32_t words[3] = {3, 0, 7};
            std::memcpy(stage + 2048, words, sizeof(words));

            EXPECT(begin(mesh_additive.p, black));
            list->CopyBufferRegion(args_gpu.p, 48, staging.p, 0, 96);
            list->CopyBufferRegion(count_gpu.p, 16, staging.p, 2048, 12);
            const D3D12_RESOURCE_BARRIER to_args[2] = {
                transition(args_gpu.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
                transition(count_gpu.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)};
            list->ResourceBarrier(2, to_args);
            set_constants(constants(0, 32, 0));
            list->ExecuteIndirect(sig12.p, 5, args_cpu.p, 36, nullptr, 0);
            if (counts) {
                set_constants(constants(16, 0, 32));
                list->ExecuteIndirect(sig24.p, 4, args_gpu.p, 48, count_gpu.p, 16);
                set_constants(constants(32, 255, 255));
                list->ExecuteIndirect(sig24.p, 4, args_gpu.p, 48, count_gpu.p, 20);  // I3: a count of 0
                set_constants(constants(40, 0, 64));
                list->ExecuteIndirect(sig12.p, 2, args_cpu.p, 256, count_gpu.p, 24);
            }
            set_constants(constants(48, 8, 8));
            list->ExecuteIndirect(sig24.p, 20, args_cpu.p, 512, nullptr, 0);
            EXPECT(end());
            const auto result = run(list.p);
            if (result != leonida::ReplayResult::success)
                std::fprintf(stderr, "indirect mesh replay: %d\n", int(result));
            EXPECT(result == leonida::ReplayResult::success);
            indirect_mismatches = check("ExecuteIndirect DISPATCH_MESH", want);

            // Without drawIndirectCount, a count-buffer call cannot be replayed.
            if (!counts) {
                EXPECT(begin(mesh_opaque.p, black));
                list->ExecuteIndirect(sig12.p, 2, args_cpu.p, 256, count_gpu.p, 24);
                EXPECT(end());
                EXPECT(run(list.p) == leonida::ReplayResult::unsupported);
            }

            // ===== (c) a draw of the wrong kind fails at execution =====
            Ref<ID3D12CommandSignature> sig_draw;
            {
                D3D12_INDIRECT_ARGUMENT_DESC arg{};
                arg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
                const D3D12_COMMAND_SIGNATURE_DESC desc{sizeof(D3D12_DRAW_ARGUMENTS), 1, &arg, 0};
                EXPECT(SUCCEEDED(device->CreateCommandSignature(&desc, nullptr, IID_PPV_ARGS(&sig_draw.p))));
            }
            EXPECT(begin(vs_pso.p, black));  // a DISPATCH_MESH signature with a VS PSO
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->ExecuteIndirect(sig12.p, 1, args_cpu.p, 36, nullptr, 0);
            EXPECT(end());
            EXPECT(run(list.p) == leonida::ReplayResult::invalid_draw);
            EXPECT(begin(mesh_opaque.p, black));  // a DRAW signature with the mesh PSO
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list->ExecuteIndirect(sig_draw.p, 1, args_cpu.p, 512, nullptr, 0);
            EXPECT(end());
            EXPECT(run(list.p) == leonida::ReplayResult::invalid_draw);
            args_cpu->Unmap(0, nullptr);
            staging->Unmap(0, nullptr);
        }
    }

    std::printf("mesh tier 1, translated wave64 mesh shader (128 threads, %s): DispatchMesh 4x2x2, 3x1x1, 1x1x1 and an empty "
                "grid, %u mismatched pixels; ExecuteIndirect DISPATCH_MESH, %u records drawn%s, %u mismatched pixels; "
                "wrong-kind indirect draws fail at execution\n",
                subgroup, direct_mismatches, records_drawn, counts ? " (count buffers included)" : " (no count buffers)",
                indirect_mismatches);
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    return direct_mismatches == 0 && indirect_mismatches == 0 ? 0 : 1;
}
