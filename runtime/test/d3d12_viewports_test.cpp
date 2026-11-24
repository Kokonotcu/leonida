// M4 (cascaded shadow maps): more than viewport and scissor 0, through the D3D12 API on the runtime's own device, with
// translator-output shaders: test/shaders/vs_vpindex.asm writes SV_ViewportArrayIndex = SV_InstanceID (the misc
// vector at pos1) and covers its whole viewport, ps_m5_flat.asm fills it with a root-constant colour.
// A 64x32 target cleared to black, two viewports and two scissors:
//   viewport 0 = the left half, scissor 0 = the whole target;
//   viewport 1 = the right half, scissor 1 = the top half of the target;
//   one DrawInstanced of 2 instances: instance i draws through viewport i.
// Expected: the left half magenta, the right half magenta above row 16 and black below. Before M4 the layer kept only
// the first viewport and scissor (pipelines had one, RSSetViewports stubbed the rest), so the right half stayed black.
// A second draw sets one viewport (the left half) after the two: back to a single viewport, it covers the left half
// only, so instance 1's index past the bound count is never what the test relies on. 77: no Vulkan device, or no
// multiViewport / shaderOutputViewportIndex (said so).
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../src/shader_library.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifndef LEONIDA_TEST_VIEWPORTS_DIR
#error "LEONIDA_TEST_VIEWPORTS_DIR must name the folder build.py wrote (spirv/index.json, fssm/*.fssm)"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;

namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

constexpr UINT kWidth = 64, kHeight = 32, kRowPitch = 256;
constexpr std::uint32_t kColour = 0xFFFF00FFu;  // R 1, G 0, B 1, A 1 as RGBA8 bytes read little-endian
constexpr std::uint32_t kGreen = 0xFF00FF00u;   // R 0, G 1, B 0

}  // namespace

int main() {
    const std::filesystem::path root_dir = LEONIDA_TEST_VIEWPORTS_DIR;
    _wputenv_s(L"LEONIDA_SHADERS", (root_dir / "spirv").wstring().c_str());
    _wputenv_s(L"LEONIDA_PSO_RECORD", L"0");
    _wputenv_s(L"LEONIDA_PIPELINE_CACHE_PERSIST", L"0");
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) {
        std::puts("SKIP: no Vulkan device for the runtime");
        return 77;
    }
    const rt::d3d12::VulkanDeviceInfo* info = rt::d3d12::vulkan_device_info(device.p);
    EXPECT(info);
    if (info->max_viewports < 2 || !info->shader_output_viewport_index) {
        std::printf("SKIP: the GPU has %u viewports, shaderOutputViewportIndex %d\n", info->max_viewports,
                    int(info->shader_output_viewport_index));
        return 77;
    }

    const auto vs = shader_container(read_file(root_dir / "fssm" / "vs_vpindex.fssm"));
    const auto ps = shader_container(read_file(root_dir / "fssm" / "ps_m5_flat.fssm"));
    EXPECT(vs.size() > 64 && ps.size() > 64);

    // [0] eight constants: s8..s15; ps_m5_flat reads the colour from s13..s15.
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    param.Constants = {0, 0, 8};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(root_desc);
    EXPECT(!blob.empty());
    Ref<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = root.p;
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    desc.SampleMask = UINT_MAX;
    desc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.RasterizerState.DepthClipEnable = TRUE;
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> pso;
    EXPECT(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC target_desc{};
    target_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    target_desc.Width = kWidth;
    target_desc.Height = kHeight;
    target_desc.DepthOrArraySize = 1;
    target_desc.MipLevels = 1;
    target_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    target_desc.SampleDesc.Count = 1;
    target_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    Ref<ID3D12Resource> targets[2], readback;
    for (auto& target : targets)
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&target.p))));
    const auto readback_desc = buffer_desc(2 * kRowPitch * kHeight);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtv[2] = {rtv_heap->GetCPUDescriptorHandleForHeapStart()};
    rtv[1].ptr = rtv[0].ptr + rtv_step;
    for (int i = 0; i < 2; ++i) device->CreateRenderTargetView(targets[i].p, nullptr, rtv[i]);

    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                               IID_PPV_ARGS(&list.p))));
    const float black[4] = {0, 0, 0, 1};
    for (const auto& handle : rtv) list->ClearRenderTargetView(handle, black, 0, nullptr);
    list->SetGraphicsRootSignature(root.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
    // Target 0: two viewports and two scissors, instance i through viewport i.
    const float magenta[8] = {0, 0, 0, 0, 0, 1.0f, 0.0f, 1.0f};
    list->SetGraphicsRoot32BitConstants(0, 8, magenta, 0);
    list->OMSetRenderTargets(1, &rtv[0], FALSE, nullptr);
    const D3D12_VIEWPORT viewports[2] = {{0, 0, 32, 32, 0, 1}, {32, 0, 32, 32, 0, 1}};
    const D3D12_RECT scissors[2] = {{0, 0, kWidth, kHeight}, {0, 0, kWidth, kHeight / 2}};
    list->RSSetViewports(2, viewports);
    list->RSSetScissorRects(2, scissors);
    list->DrawInstanced(4, 2, 0, 0);
    // Target 1: back to one viewport and scissor (the left half), one instance, green: the draw state after a
    // multi-viewport one is a single viewport again.
    const float green[8] = {0, 0, 0, 0, 0, 0.0f, 1.0f, 0.0f};
    list->SetGraphicsRoot32BitConstants(0, 8, green, 0);
    list->OMSetRenderTargets(1, &rtv[1], FALSE, nullptr);
    list->RSSetViewports(1, viewports);
    list->RSSetScissorRects(1, scissors);
    list->DrawInstanced(4, 1, 0, 0);
    for (int i = 0; i < 2; ++i) {
        const D3D12_RESOURCE_BARRIER to_copy =
            transition(targets[i].p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = {UINT64(i) * kRowPitch * kHeight,
                               {DXGI_FORMAT_R8G8B8A8_UNORM, kWidth, kHeight, 1, kRowPitch}};
        D3D12_TEXTURE_COPY_LOCATION src{targets[i].p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    EXPECT(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    const HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, done)));
    EXPECT(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
    CloseHandle(done);

    void* mapped = nullptr;
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    unsigned mismatches = 0;
    for (UINT t = 0; t < 2; ++t)
        for (UINT y = 0; y < kHeight; ++y)
            for (UINT x = 0; x < kWidth; ++x) {
                std::uint32_t got = 0;
                std::memcpy(&got, bytes + (t * kHeight + y) * kRowPitch + x * 4, 4);
                const bool covered = x < 32 || (t == 0 && y < kHeight / 2);
                const std::uint32_t want = !covered ? 0xFF000000u : t == 0 ? kColour : kGreen;
                if (got != want && mismatches++ < 8)
                    std::fprintf(stderr, "target %u pixel (%u, %u): got %08x want %08x\n", t, x, y, got, want);
            }
    readback->Unmap(0, nullptr);
    EXPECT(mismatches == 0);
    std::printf("viewports: SV_ViewportArrayIndex picked viewport 1 and scissor 1 (the right half, its top), viewport 0 "
                "the left half; a later single viewport covers the left half only\n");
    return 0;
}
