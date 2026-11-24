// LEONIDA_TITLE_HEAP_SYNC, on the Vulkan device through the Xbox-ABI proxy. Descriptor heaps get their backing from a
// title allocator (the __XMemAllocationHooks__ path of slot 14), and the test, as console code may, copies a 32-byte
// T# straight into the title's view heap instead of calling CreateShaderResourceView there. A compute dispatch then
// samples that slot (test/glsl/cs_set0_sample.comp, as d3d12_set0_compute_test), twice: with texture A's T# in the
// slot, then, after a second memcpy, texture B's.
//   - Off (the upstream model): the bytes stay in title memory, the GPU reads the real heap, which never got them:
//     neither run shows its texture.
//   - On: title memory is the heap; each ExecuteCommandLists copies what changed into the real heap first: run 1
//     shows A and run 2 shows B. The sampler (made with CreateSampler into the title sampler heap) reaches the GPU
//     the same way, so its wrap is checked too.
// Skips (77) without a Vulkan GPU with the bindless set. No game data. LEONIDA_VK_VALIDATION=1: 0 errors required.
#include "shader_fixture_path.h"
#include "test_temp_dir.h"
#include "vk_test_device.h"
#include "d3d12_test_shaders.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../src/shader_library.h"

#include <malloc.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifndef LEONIDA_TEST_CS_SET0_SAMPLE_SPV
#error "LEONIDA_TEST_CS_SET0_SAMPLE_SPV must name the compiled test shader"
#endif

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

using namespace d3d12test;
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: handle in RAX

namespace {

using Texel = std::array<std::uint8_t, 4>;
constexpr UINT texture_size = 8, texels = texture_size * texture_size, run_bytes = texels * 4;

Texel gradient(UINT x, UINT y) {
    return {std::uint8_t(x * 36), std::uint8_t(y * 36), std::uint8_t(255 - (x + y) * 16), 255};
}
Texel reflected(UINT x, UINT y) { return gradient(texture_size - 1 - x, texture_size - 1 - y); }

// The title's allocator: ordinary host memory.
void* title_alloc(std::size_t bytes, std::uint64_t) { return _aligned_malloc(bytes, 4096); }
void title_free(void* p, std::uint64_t) { _aligned_free(p); }

// The shader samples one tile to the right in U with a wrapping sampler: texel (x, y) of the run shows (x, y).
unsigned mismatches(const std::uint8_t* run, Texel (*want)(UINT, UINT)) {
    unsigned bad = 0;
    for (UINT i = 0; i < texels; ++i)
        if (std::memcmp(run + i * 4, want(i % texture_size, i / texture_size).data(), 4) != 0) ++bad;
    return bad;
}

// Both runs, with the switch as given. `results`: run 1 then run 2.
int run(const vktest::Device& d, const std::filesystem::path& library, const std::vector<std::uint8_t>& container,
        bool sync, std::vector<std::uint8_t>& results) {
    rt::d3d12_xs::set_title_heap_sync(sync ? 1 : 0);
    rt::d3d12_xs::set_descriptor_heap_allocator(&title_alloc, &title_free);
    results.assign(2 * run_bytes, 0);
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;
    info.bindless = d.bindless;
    info.bindless_partially_bound = d.partially_bound;
    info.image_view_min_lod = d.image_view_min_lod;
    info.shader_dir = library.string();
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);
    {
        D3D12_DESCRIPTOR_RANGE view_range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 3, 0, 0, 0};
        D3D12_DESCRIPTOR_RANGE sampler_range{D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0, 0};
        D3D12_ROOT_PARAMETER params[4] = {};
        params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants = {0, 0, 2};
        params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        params[1].Descriptor = {0, 0};
        params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[2].DescriptorTable = {1, &view_range};
        params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[3].DescriptorTable = {1, &sampler_range};
        D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
        root_desc.Desc_1_0 = {4, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
        const auto root_blob = serialize_root_signature(root_desc);
        EXPECT(!root_blob.empty());
        Ref<ID3D12RootSignature> root;
        EXPECT(SUCCEEDED(device->CreateRootSignature(0, root_blob.data(), root_blob.size(), IID_PPV_ARGS(&root.p))));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
        pso_desc.pRootSignature = root.p;
        pso_desc.CS = {container.data(), container.size()};
        Ref<ID3D12PipelineState> pso;
        EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        D3D12_RESOURCE_DESC texture_desc{};
        texture_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture_desc.Width = texture_desc.Height = texture_size;
        texture_desc.DepthOrArraySize = 1;
        texture_desc.MipLevels = 1;
        texture_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture_desc.SampleDesc.Count = 1;
        Ref<ID3D12Resource> textures[2], upload, output, readback;
        for (auto& texture : textures)
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &texture_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&texture.p))));
        constexpr UINT64 b_offset = 2048;
        const auto upload_desc = buffer_desc(4096);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                         D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                         IID_PPV_ARGS(&upload.p))));
        const auto output_desc = buffer_desc(run_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &output_desc,
                                                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                         IID_PPV_ARGS(&output.p))));
        const auto readback_desc = buffer_desc(run_bytes);
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&readback.p))));
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
        device->GetCopyableFootprints(&texture_desc, 0, 1, 0, &footprint, nullptr, nullptr, nullptr);
        std::uint8_t* up = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
        for (UINT y = 0; y < texture_size; ++y)
            for (UINT x = 0; x < texture_size; ++x) {
                std::memcpy(up + y * footprint.Footprint.RowPitch + x * 4, gradient(x, y).data(), 4);
                std::memcpy(up + b_offset + y * footprint.Footprint.RowPitch + x * 4, reflected(x, y).data(), 4);
            }

        // Title-backed heaps: slot 9 gives the title's address, slot 10 the GPU handle of the real heap.
        const D3D12_DESCRIPTOR_HEAP_DESC view_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC sampler_heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2,
                                                           D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        Ref<ID3D12DescriptorHeap> view_heap, sampler_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&view_heap_desc, IID_PPV_ARGS(&view_heap.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&sampler_heap_desc, IID_PPV_ARGS(&sampler_heap.p))));
        const std::uint64_t view_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 9))(view_heap.p);
        const std::uint64_t view_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(view_heap.p, 10))(view_heap.p);
        const std::uint64_t sampler_cpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 9))(sampler_heap.p);
        const std::uint64_t sampler_gpu = reinterpret_cast<XboxHeapHandleFn>(slot(sampler_heap.p, 10))(sampler_heap.p);
        EXPECT(view_cpu && sampler_cpu && view_gpu >= 0x10000 && sampler_gpu >= 0x10000);

        // The T#s, made outside the heap (CreateShaderResourceView on the device itself, into host memory): the bytes
        // console code would copy around. The sampler goes into the title sampler heap through the proxy.
        alignas(32) std::uint8_t tsharp[2][32] = {};
        for (UINT i = 0; i < 2; ++i) {
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D = {0, 1, 0, 0.0f};
            inner->CreateShaderResourceView(textures[i].p, &srv,
                                            D3D12_CPU_DESCRIPTOR_HANDLE{reinterpret_cast<SIZE_T>(tsharp[i])});
        }
        D3D12_SAMPLER_DESC sampler_desc{};
        sampler_desc.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
        sampler_desc.AddressU = sampler_desc.AddressV = sampler_desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler_desc.MaxAnisotropy = 1;
        sampler_desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
        sampler_desc.MaxLOD = D3D12_FLOAT32_MAX;
        device->CreateSampler(&sampler_desc, D3D12_CPU_DESCRIPTOR_HANDLE{SIZE_T(sampler_cpu)});

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

        for (UINT i = 0; i < 2; ++i) {
            D3D12_TEXTURE_COPY_LOCATION dst{};
            dst.pResource = textures[i].p;
            dst.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION src{};
            src.pResource = upload.p;
            src.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            src.PlacedFootprint = footprint;
            src.PlacedFootprint.Offset = i ? b_offset : 0;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        D3D12_RESOURCE_BARRIER sampled[2] = {
            transition(textures[0].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE),
            transition(textures[1].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
        list->ResourceBarrier(2, sampled);
        EXPECT(list->Close() == S_OK);
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);

        // One list for both runs: the dispatch over slot 0, then the copy to the readback buffer.
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        ID3D12DescriptorHeap* heaps[] = {view_heap.p, sampler_heap.p};
        list->SetDescriptorHeaps(2, heaps);
        list->SetComputeRootSignature(root.p);
        list->SetPipelineState(pso.p);
        list->SetComputeRootUnorderedAccessView(1, output->GetGPUVirtualAddress());
        list->SetComputeRootDescriptorTable(2, D3D12_GPU_DESCRIPTOR_HANDLE{view_gpu});
        list->SetComputeRootDescriptorTable(3, D3D12_GPU_DESCRIPTOR_HANDLE{sampler_gpu});
        const std::uint32_t offset_and_first[2] = {0, 0};
        list->SetComputeRoot32BitConstants(0, 2, offset_and_first, 0);
        list->Dispatch(1, 1, 1);
        D3D12_RESOURCE_BARRIER to_copy = transition(output.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &to_copy);
        list->CopyBufferRegion(readback.p, 0, output.p, 0, run_bytes);
        D3D12_RESOURCE_BARRIER back = transition(output.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1, &back);
        EXPECT(list->Close() == S_OK);

        for (UINT r = 0; r < 2; ++r) {
            // What console code does: the 32 bytes straight into the title's heap memory.
            std::memcpy(reinterpret_cast<void*>(view_cpu), tsharp[r], sizeof(tsharp[r]));
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 2 + r)));
            EXPECT(SUCCEEDED(fence->SetEventOnCompletion(2 + r, nullptr)));
            void* mapped = nullptr;
            EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
            std::memcpy(results.data() + r * run_bytes, mapped, run_bytes);
            readback->Unmap(0, nullptr);
        }
        const auto stats = rt::d3d12_xs::title_heap_sync_stats();
        std::printf("%s: %llu syncs, %llu chunks copied\n", sync ? "sync on" : "sync off",
                    static_cast<unsigned long long>(stats.syncs), static_cast<unsigned long long>(stats.chunks_copied));
    }
    rt::d3d12_xs::set_descriptor_heap_allocator(nullptr, nullptr);
    rt::d3d12_xs::set_title_heap_sync(-1);
    return 0;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto device_status = d.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;
    if (!d.bindless) {
        std::printf("SKIP: the device lacks the descriptor indexing features of the bindless set 0\n");
        return 77;
    }
    const auto program = fssm_program({0x43530003, 0xbf810000});
    const auto container = shader_container(program);
    const testfiles::ScratchDir scratch("leonida_title_heap_sync_test");
    const auto& library = scratch.path();
    EXPECT(testfiles::copy_file(vktest::shader_fixture_path(LEONIDA_TEST_CS_SET0_SAMPLE_SPV),
                                library / "cs_set0_sample.spv"));
    {
        std::ofstream index(library / "index.json", std::ios::binary);
        index << "{\"cs_set0_sample\": {\"stage\": \"cs\", \"fssm_sha256\": \"" << leonida::sha256_hex(program)
              << "\", \"bindings\": [0, 1, 2], \"push_constant_bytes\": 144, \"local_size\": [64, 1, 1], "
                 "\"uses_code_base\": false, \"wave64\": false}}";
    }
    {
        // The PSO needs 144 push-constant bytes.
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        Ref<ID3D12Device> probe;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&probe.p))));
        if (rt::d3d12::device_push_constants_need_fallback(probe.p)) {
            std::printf("d3d12_title_heap_sync_test: skipped, 144 push-constant bytes needed\n");
            return 77;
        }
    }

    std::vector<std::uint8_t> off, on;
    EXPECT(run(d, library, container, false, off) == 0);
    EXPECT(run(d, library, container, true, on) == 0);
    const unsigned off_a = mismatches(off.data(), gradient), off_b = mismatches(off.data() + run_bytes, reflected);
    const unsigned on_a = mismatches(on.data(), gradient), on_b = mismatches(on.data() + run_bytes, reflected);
    std::printf("texels wrong of %u: sync off %u (A) and %u (B); sync on %u (A) and %u (B)\n", texels, off_a, off_b,
                on_a, on_b);
    EXPECT(off_a > 0 && off_b > 0);   // the upstream model: the copied bytes never reach the GPU
    EXPECT(on_a == 0 && on_b == 0);   // title memory canonical: both runs see what the title wrote
    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("d3d12_title_heap_sync_test: OK\n");
    return 0;
}
