// docs/aot_pipelines.md step 2: a draw through shader objects (LEONIDA_SHADER_OBJECTS=1) writes the same pixels as the
// same draw through a graphics pipeline, on the runtime's own device. Then `--prebuild-shaders`' path: the shaders'
// driver binaries are stored, and a third device makes its shader objects from them and draws the same pixels again.
// 77 (skip) without a Vulkan device, or one without VK_EXT_shader_object.
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

#if !defined(LEONIDA_TEST_VS_FULLSCREEN_SPV) || !defined(LEONIDA_TEST_PS_SPV)
#error "LEONIDA_TEST_VS_FULLSCREEN_SPV and LEONIDA_TEST_PS_SPV must name the test shaders"
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

// One device (shader objects per LEONIDA_SHADER_OBJECTS as set by the caller): clear, one full-screen draw, read back.
bool draw_once(const Shaders& shaders, std::vector<std::uint8_t>& pixels, bool& used_shader_objects) {
    Ref<ID3D12Device> device;
    CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))));
    const rt::d3d12::VulkanDeviceInfo* info = rt::d3d12::vulkan_device_info(device.p);
    CHECK(info);
    used_shader_objects = info->shader_objects;

    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {0, nullptr, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
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
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    Ref<ID3D12PipelineState> pso;
    CHECK(SUCCEEDED(device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso.p))));

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
    Ref<ID3D12Resource> target;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &tex,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                    IID_PPV_ARGS(&target.p))));
    const UINT64 row = 256, bytes = row * kSize;
    Ref<ID3D12Resource> readback;
    const auto buffer = buffer_desc(bytes);
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback.p))));
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);

    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    Ref<ID3D12GraphicsCommandList> list;
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                              IID_PPV_ARGS(&list.p))));
    const float clear[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    list->ClearRenderTargetView(rtv, clear, 0, nullptr);
    list->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    list->SetGraphicsRootSignature(root.p);
    list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    const D3D12_VIEWPORT viewport{0, 0, float(kSize), float(kSize), 0, 1};
    const D3D12_RECT scissor{0, 0, LONG(kSize), LONG(kSize / 2)};  // only the top half: the rest stays the clear
    list->RSSetViewports(1, &viewport);
    list->RSSetScissorRects(1, &scissor);
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
    Ref<ID3D12Fence> fence;
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    const HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, done)));
    CHECK(WaitForSingleObject(done, 10000) == WAIT_OBJECT_0);
    CloseHandle(done);
    void* mapped = nullptr;
    CHECK(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    pixels.assign(static_cast<std::uint8_t*>(mapped), static_cast<std::uint8_t*>(mapped) + bytes);
    readback->Unmap(0, nullptr);
    return true;
}

bool write_library(const std::filesystem::path& dir, Shaders& shaders) {
    const auto vs_program = fssm_program({0x56530002, 0xbf810000});
    const auto ps_program = fssm_program({0x50530001, 0xbf810000});
    shaders.vs = shader_container(vs_program);
    shaders.ps = shader_container(ps_program);
    std::filesystem::create_directories(dir);
    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_VS_FULLSCREEN_SPV), dir / "vs_fullscreen.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_PS_SPV), dir / "ps_draw.spv",
                               std::filesystem::copy_options::overwrite_existing);
    std::ofstream index(dir / "index.json", std::ios::binary);
    index << "{\"vs_fullscreen\": {\"stage\": \"vs\", \"fssm_sha256\": \"" << leonida::sha256_hex(vs_program)
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"input_signature\": []},\n"
             " \"ps_draw\": {\"stage\": \"ps\", \"fssm_sha256\": \"" << leonida::sha256_hex(ps_program)
          << "\", \"bindings\": [], \"push_constant_bytes\": 144, \"outputs\": [0]}}";
    return bool(index);
}

}  // namespace

int main() {
    const auto scratch = std::filesystem::temp_directory_path() / "leonida_shader_object_draw_test";
    std::error_code ec;
    std::filesystem::remove_all(scratch, ec);
    Shaders shaders;
    if (!write_library(scratch / "lib", shaders)) return 1;
    _wputenv_s(L"LEONIDA_SHADERS", (scratch / "lib").wstring().c_str());
    _wputenv_s(L"LEONIDA_SHADER_BINARIES", (scratch / "bin").wstring().c_str());
    _wputenv_s(L"LEONIDA_PSO_RECORD", L"0");
    _wputenv_s(L"LEONIDA_PIPELINE_CACHE_PERSIST", L"0");

    std::vector<std::uint8_t> pipeline_pixels, object_pixels, binary_pixels;
    bool used = false;
    _wputenv_s(L"LEONIDA_SHADER_OBJECTS", L"0");
    if (!draw_once(shaders, pipeline_pixels, used)) return 77;  // no usable device
    if (used) return 1;
    _wputenv_s(L"LEONIDA_SHADER_OBJECTS", L"1");
    if (!draw_once(shaders, object_pixels, used)) return 1;
    if (!used) {
        std::puts("SKIP: this device has no VK_EXT_shader_object");
        return 77;
    }
    // The draw covered the top half and nothing else.
    const auto at = [&](const std::vector<std::uint8_t>& p, UINT x, UINT y) { return &p[y * 256 + x * 4]; };
    if (std::memcmp(at(pipeline_pixels, 3, 3), at(pipeline_pixels, 3, kSize - 3), 4) == 0) {
        std::fprintf(stderr, "the pipeline draw left the top half as the clear colour\n");
        return 1;
    }
    for (UINT y = 0; y < kSize; ++y)
        if (std::memcmp(at(pipeline_pixels, 0, y), at(object_pixels, 0, y), kSize * 4) != 0) {
            std::fprintf(stderr, "row %u differs between the pipeline and shader-object draws\n", y);
            return 1;
        }

    // Stored binaries: --prebuild-shaders writes them, a new device makes its shader objects from them.
    if (rt::d3d12::prebuild_shader_binaries(1) != 0) return 1;
    if (!std::filesystem::exists(scratch / "bin" / "vs" / "vs_fullscreen.bin") ||
        !std::filesystem::exists(scratch / "bin" / "ps" / "ps_draw.bin")) {
        std::fprintf(stderr, "no binaries stored\n");
        return 1;
    }
    if (!draw_once(shaders, binary_pixels, used) || !used) return 1;
    if (binary_pixels != pipeline_pixels) {
        std::fprintf(stderr, "the draw from stored binaries differs\n");
        return 1;
    }
    std::filesystem::remove_all(scratch, ec);
    std::puts("shader objects: the same pixels as the pipeline, from SPIR-V and from stored driver binaries");
    return 0;
}
