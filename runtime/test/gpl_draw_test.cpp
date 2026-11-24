// docs/aot_pipelines.md step 3: graphics PSOs linked from VK_EXT_graphics_pipeline_library parts draw the same pixels as
// whole pipelines, and after `--prebuild-shaders` a new device makes every shader part from the pipeline cache.
//   1. LEONIDA_GPL=0: three PSOs drawn as whole pipelines (the reference).
//   2. LEONIDA_GPL=1, an empty cache directory: prebuild_shader_binaries() makes the shared shader parts of every
//      translated shader into the pipeline cache and saves it.
//   3. LEONIDA_GPL=1: a new device loads that cache, links the same PSOs and draws: the same pixels, and every shared
//      part came from the cache (VkPipelineCreationFeedback: 0 compiled).
// The PSOs differ in what the libraries make dynamic: depth test and write, cull mode, depth bias and blending, so a
// state the linked pipeline failed to set would change the pixels. 77 (skip) without a Vulkan device, or one without
// graphicsPipelineLibrary. Shader objects stay off (LEONIDA_SHADER_OBJECTS=0): they would take the PSOs instead.
#include "d3d12_test_shaders.h"
#include "shader_fixture_path.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#if !defined(LEONIDA_TEST_VS_DEPTH_SPV) || !defined(LEONIDA_TEST_FS_SOLID_SPV)
#error "LEONIDA_TEST_VS_DEPTH_SPV and LEONIDA_TEST_FS_SOLID_SPV must name the test shaders"
#endif

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

using namespace d3d12test;

namespace {

template<typename T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

constexpr UINT kSize = 32;

struct Shaders {
    std::vector<std::uint8_t> vs, ps;
};

struct Run {
    std::vector<std::uint8_t> pixels;
    bool linked = false;  // the device linked its PSOs from pipeline libraries
    rt::d3d12::PipelineLibraryStats stats;
};

// One device: three draws into a colour target and a depth buffer, read back.
//   draw 1: depth 0.5 at the centre sloping 0.25 per unit of x, LESS with writes, no blending, no culling;
//   draw 2: flat depth 0.5, LESS without writes, additive blending, depth bias pushing it 0.0625 nearer: it lands where
//           draw 1 is farther than 0.4375 (the right side) and adds its colour there;
//   draw 3: as draw 1 but culling front faces: the triangle is front facing, so nothing is drawn.
bool draw_once(const Shaders& shaders, Run& run) {
    Ref<ID3D12Device> device;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))));
    const rt::d3d12::VulkanDeviceInfo* info = rt::d3d12::vulkan_device_info(device.p);
    CHECK(info);
    run.linked = info->graphics_pipeline_library;

    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, 2};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(root_desc);
    CHECK(!blob.empty());
    Ref<ID3D12RootSignature> root;
    CHECK(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));

    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.p;
    desc.VS = {shaders.vs.data(), shaders.vs.size()};
    desc.PS = {shaders.ps.data(), shaders.ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = 0xffffffff;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.DepthStencilState.DepthEnable = TRUE;
    desc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    desc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> first, second, culled;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&first.p))));
    D3D12_GRAPHICS_PIPELINE_STATE_DESC blend = desc;
    blend.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
    blend.RasterizerState.DepthBias = -(1 << 20);  // D32_FLOAT: 2^20 * 2^(exponent of 0.5 - 23) = 0.0625 nearer
    auto& rt0 = blend.BlendState.RenderTarget[0];
    rt0.BlendEnable = TRUE;
    rt0.SrcBlend = rt0.DestBlend = rt0.SrcBlendAlpha = rt0.DestBlendAlpha = D3D12_BLEND_ONE;
    rt0.BlendOp = rt0.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&blend, IID_PPV_ARGS(&second.p))));
    D3D12_GRAPHICS_PIPELINE_STATE_DESC cull = desc;
    cull.RasterizerState.CullMode = D3D12_CULL_MODE_FRONT;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&cull, IID_PPV_ARGS(&culled.p))));

    D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC tex{};
    tex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    tex.Width = kSize;
    tex.Height = kSize;
    tex.DepthOrArraySize = 1;
    tex.MipLevels = 1;
    tex.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    tex.SampleDesc.Count = 1;
    tex.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> target, depth;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                    IID_PPV_ARGS(&target.p))));
    D3D12_RESOURCE_DESC depth_desc = tex;
    depth_desc.Format = DXGI_FORMAT_D32_FLOAT;
    depth_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &depth_desc,
                                                    D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                                    IID_PPV_ARGS(&depth.p))));
    const UINT64 row = 256, bytes = row * kSize;
    Ref<ID3D12Resource> readback;
    const auto buffer = buffer_desc(bytes);
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback.p))));
    D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1};
    Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))));
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);
    device->CreateDepthStencilView(depth.p, nullptr, dsv);

    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, first.p,
                                              IID_PPV_ARGS(&list.p))));
    const float clear[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    list->ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    list->OMSetRenderTargets(1, &rtv, FALSE, &dsv);
    list->SetGraphicsRootSignature(root.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VIEWPORT viewport{0, 0, float(kSize), float(kSize), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kSize), LONG(kSize)};
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
    const auto constants = [&](float centre, float slope) {
        UINT values[2];
        std::memcpy(&values[0], &centre, 4);
        std::memcpy(&values[1], &slope, 4);
        list->SetGraphicsRoot32BitConstants(0, 2, values, 0);
    };
    constants(0.5f, 0.25f);
    list->DrawInstanced(3, 1, 0, 0);
    list->SetPipelineState(second.p);
    constants(0.5f, 0.0f);
    list->DrawInstanced(3, 1, 0, 0);
    list->SetPipelineState(culled.p);
    constants(0.0f, 0.0f);
    list->DrawInstanced(3, 1, 0, 0);
    const auto barrier = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &barrier);
    D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, kSize, kSize, 1, UINT(row)};
    D3D12_TEXTURE_COPY_LOCATION src{target.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    const HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, done)));
    CHECK(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
    CloseHandle(done);
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    run.pixels.assign(static_cast<std::uint8_t*>(mapped), static_cast<std::uint8_t*>(mapped) + bytes);
    readback->Unmap(0, nullptr);
    run.stats = rt::d3d12::pipeline_library_stats(device.p);
    return true;
}

bool write_library(const std::filesystem::path& dir, Shaders& shaders) {
    const auto vs_program = fssm_program({0x56530003, 0xbf810000});
    const auto ps_program = fssm_program({0x50530003, 0xbf810000});
    shaders.vs = shader_container(vs_program);
    shaders.ps = shader_container(ps_program);
    std::filesystem::create_directories(dir);
    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_DEPTH_SPV), dir / "vs_depth.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_FS_SOLID_SPV), dir / "fs_solid.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::ofstream index(dir / "index.json", std::ios::binary);
    index << "{\"vs_depth\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
             " \"fs_solid\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    return bool(index);
}

const std::uint8_t* at(const std::vector<std::uint8_t>& p, UINT x, UINT y) { return &p[y * 256 + x * 4]; }

}  // namespace

int main() {
    const auto scratch = std::filesystem::temp_directory_path() / "leonida_gpl_draw_test";
    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
    Shaders shaders;
    if (!write_library(scratch / "lib", shaders)) return 1;
    std::filesystem::create_directories(scratch / "cache");
    _wputenv_s(L"LEONIDA_SHADERS", (scratch / "lib").wstring().c_str());
    _wputenv_s(L"LEONIDA_PIPELINE_CACHE_DIR", (scratch / "cache").wstring().c_str());
    _wputenv_s(L"LEONIDA_PIPELINE_CACHE_PERSIST", L"1");
    _wputenv_s(L"LEONIDA_PSO_RECORD", L"0");
    _wputenv_s(L"LEONIDA_SHADER_OBJECTS", L"0");

    // 1. Whole pipelines.
    Run whole, linked;
    _wputenv_s(L"LEONIDA_GPL", L"0");
    if (!draw_once(shaders, whole)) return 77;  // no usable device
    if (whole.linked) {
        std::fprintf(stderr, "LEONIDA_GPL=0 still linked from pipeline libraries\n");
        return 1;
    }
    // The cache the whole pipelines left must not answer the libraries: start from an empty one.
    std::filesystem::remove_all(scratch / "cache", ec);
    std::filesystem::create_directories(scratch / "cache");

    // 2. The prebuild, on a device with pipeline libraries.
    _wputenv_s(L"LEONIDA_GPL", L"1");
    if (rt::d3d12::prebuild_shader_binaries(1) != 0) {
        // 5: the device has neither shader objects (kept off here) nor pipeline libraries.
        std::puts("SKIP: this device has no graphicsPipelineLibrary");
        return 77;
    }

    // 3. A new device: linked PSOs from the cache the prebuild saved.
    if (!draw_once(shaders, linked)) return 1;
    if (!linked.linked) {
        std::puts("SKIP: this device has no graphicsPipelineLibrary");
        return 77;
    }

    // The reference is not trivial: the right side has draw 2 added, the left side does not, and draw 3 drew nothing.
    const std::uint8_t* left = at(whole.pixels, 2, kSize / 2);
    const std::uint8_t* right = at(whole.pixels, kSize - 3, kSize / 2);
    if (std::memcmp(left, right, 4) == 0 || left[0] == 0) {
        std::fprintf(stderr, "the reference draw does not show the depth test: left %u,%u,%u right %u,%u,%u\n", left[0],
                     left[1], left[2], right[0], right[1], right[2]);
        return 1;
    }
    for (UINT y = 0; y < kSize; ++y)
        if (std::memcmp(at(whole.pixels, 0, y), at(linked.pixels, 0, y), kSize * 4) != 0) {
            std::fprintf(stderr, "row %u differs between the whole and the linked pipelines\n", y);
            for (UINT x = 0; x < kSize; ++x) {
                const std::uint8_t* w = at(whole.pixels, x, y);
                const std::uint8_t* l = at(linked.pixels, x, y);
                if (std::memcmp(w, l, 4) != 0) {
                    std::fprintf(stderr, "  x %u: whole %u,%u,%u,%u linked %u,%u,%u,%u\n", x, w[0], w[1], w[2], w[3],
                                 l[0], l[1], l[2], l[3]);
                    break;
                }
            }
            return 1;
        }
    const rt::d3d12::PipelineLibraryStats& s = linked.stats;
    std::printf("linked PSOs: %u shared shader parts made (%u from the pipeline cache), %u links (%u from the cache)\n",
                s.parts_made, s.parts_cached, s.links, s.links_cached);
    if (s.links < 3 || s.parts_made == 0) {
        std::fprintf(stderr, "the PSOs were not linked from shared parts\n");
        return 1;
    }
    if (s.parts_cached != s.parts_made) {
        std::fprintf(stderr, "%u shared shader parts were compiled after the prebuild\n", s.parts_made - s.parts_cached);
        return 1;
    }
    std::filesystem::remove_all(scratch, ec);
    std::puts("pipeline libraries: the same pixels as whole pipelines, every shader part from the prebuilt cache");
    return 0;
}
