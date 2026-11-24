// M5 (volumetric fog and clouds): RWTexture3D UAVs, through the D3D12 API on the runtime's own device, with
// translator-output compute shaders (test/shaders/cs_uav3d_store.asm and cs_uav3d_load.asm) and every texel checked.
// An 8x8x8 R32_UINT volume:
//   1. a UAV of the whole volume (D3D12_UAV_DIMENSION_TEXTURE3D, WSize -1): tag 1 + (z << 8) + (y << 4) + x everywhere;
//   2. a UAV of slices 2..4 (FirstWSlice 2, WSize 3): tag 2 + (z << 8) + ... at view z 0..2, so volume slices 2..4,
//      through a sliced storage view (VK_EXT_image_sliced_view_of_3d);
//   3. the same W-range UAV read with image_load (the translator adds its first slice: docs/design.md, "W-range 3D
//      UAVs") into a raw buffer: the view-relative values of step 2.
// Before 3D UAVs, CreateUnorderedAccessView refused TEXTURE3D and left a null slot: every write vanished. Steps 2 and 3
// are skipped (and said so) on a device without imageSlicedViewOf3D. 77: no Vulkan device.
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

#ifndef LEONIDA_TEST_UAV3D_DIR
#error "LEONIDA_TEST_UAV3D_DIR must name the folder build.py wrote (spirv/index.json, fssm/cs_uav3d_*.fssm)"
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

constexpr UINT kSize = 8;  // 8x8x8
constexpr std::uint32_t kTag1 = 0x01000000, kTag2 = 0x02000000;
constexpr UINT kFirstW = 2, kSizeW = 3;

std::uint32_t value(std::uint32_t tag, UINT x, UINT y, UINT z) { return tag + (z << 8) + (y << 4) + x; }

}  // namespace

int main() {
    const std::filesystem::path root_dir = LEONIDA_TEST_UAV3D_DIR;
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
    const bool sliced = info->image_sliced_view_of_3d;

    const auto store_program = read_file(root_dir / "fssm" / "cs_uav3d_store.fssm");
    const auto load_program = read_file(root_dir / "fssm" / "cs_uav3d_load.fssm");
    EXPECT(!store_program.empty() && !load_program.empty());
    const auto store_cs = shader_container(store_program), load_cs = shader_container(load_program);

    // [0] two constants, [1] a table of two UAVs (the volume's, a raw buffer's).
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 2, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[2] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, 2};
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable = {1, &range};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    root_desc.Desc_1_0 = {2, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(root_desc);
    EXPECT(!blob.empty());
    Ref<ID3D12RootSignature> root;
    EXPECT(SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p))));
    Ref<ID3D12PipelineState> store_pso, load_pso;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.CS = {store_cs.data(), store_cs.size()};
    EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&store_pso.p))));
    pso_desc.CS = {load_cs.data(), load_cs.size()};
    EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&load_pso.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC volume_desc{};
    volume_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    volume_desc.Width = volume_desc.Height = kSize;
    volume_desc.DepthOrArraySize = kSize;
    volume_desc.MipLevels = 1;
    volume_desc.Format = DXGI_FORMAT_R32_UINT;
    volume_desc.SampleDesc.Count = 1;
    volume_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    Ref<ID3D12Resource> volume, loads, readback;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &volume_desc,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&volume.p))));
    constexpr UINT load_count = 64 * kSizeW;
    const auto loads_desc = buffer_desc(load_count * 4, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &loads_desc,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&loads.p))));
    constexpr UINT row_pitch = 256, slice_pitch = row_pitch * kSize, volume_bytes = slice_pitch * kSize;
    const auto readback_desc = buffer_desc(volume_bytes + load_count * 4);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback.p))));

    // Slots 0, 1: the whole volume and the buffer; slots 2, 3: the W range and the buffer.
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                         D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    Ref<ID3D12DescriptorHeap> heap;
    EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    const UINT step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    EXPECT(step == 32);  // the shaders read slot 1 at +0x20
    const D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_GPU_DESCRIPTOR_HANDLE gpu = heap->GetGPUDescriptorHandleForHeapStart();
    const auto cpu_at = [&](UINT i) { return D3D12_CPU_DESCRIPTOR_HANDLE{cpu.ptr + SIZE_T(i) * step}; };
    D3D12_UNORDERED_ACCESS_VIEW_DESC whole{DXGI_FORMAT_R32_UINT, D3D12_UAV_DIMENSION_TEXTURE3D};
    whole.Texture3D = {0, 0, UINT(-1)};
    D3D12_UNORDERED_ACCESS_VIEW_DESC part = whole;
    part.Texture3D = {0, kFirstW, kSizeW};
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw{DXGI_FORMAT_R32_TYPELESS, D3D12_UAV_DIMENSION_BUFFER};
    raw.Buffer = {0, load_count, 0, 0, D3D12_BUFFER_UAV_FLAG_RAW};
    device->CreateUnorderedAccessView(volume.p, nullptr, &whole, cpu_at(0));
    device->CreateUnorderedAccessView(loads.p, nullptr, &raw, cpu_at(1));
    device->CreateUnorderedAccessView(volume.p, nullptr, &part, cpu_at(2));
    device->CreateUnorderedAccessView(loads.p, nullptr, &raw, cpu_at(3));
    // The whole-volume UAV slot holds a T# now; before 3D UAVs it stayed zero (a null descriptor).
    std::uint32_t slot0[8] = {};
    std::memcpy(slot0, reinterpret_cast<const void*>(cpu_at(0).ptr), sizeof(slot0));
    EXPECT(slot0[3] != 0);

    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, store_pso.p,
                                               IID_PPV_ARGS(&list.p))));
    ID3D12DescriptorHeap* heaps[] = {heap.p};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(root.p);
    D3D12_RESOURCE_BARRIER uav{};
    uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uav.UAV.pResource = volume.p;
    const UINT tag1[2] = {kTag1, 0}, tag2[2] = {kTag2, 0};
    list->SetComputeRoot32BitConstants(0, 2, tag1, 0);
    list->SetComputeRootDescriptorTable(1, gpu);
    list->Dispatch(kSize * kSize * kSize / 64, 1, 1);
    if (sliced) {
        list->ResourceBarrier(1, &uav);
        list->SetComputeRoot32BitConstants(0, 2, tag2, 0);
        list->SetComputeRootDescriptorTable(1, D3D12_GPU_DESCRIPTOR_HANDLE{gpu.ptr + 2 * step});
        list->Dispatch(kSizeW, 1, 1);
        list->ResourceBarrier(1, &uav);
        list->SetPipelineState(load_pso.p);
        list->Dispatch(kSizeW, 1, 1);
    }
    const D3D12_RESOURCE_BARRIER to_copy[] = {
        transition(volume.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE),
        transition(loads.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
    list->ResourceBarrier(2, to_copy);
    D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
    dst.PlacedFootprint = {0, {DXGI_FORMAT_R32_UINT, kSize, kSize, kSize, row_pitch}};
    D3D12_TEXTURE_COPY_LOCATION src{volume.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    list->CopyBufferRegion(readback.p, volume_bytes, loads.p, 0, load_count * 4);
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
    for (UINT z = 0; z < kSize; ++z)
        for (UINT y = 0; y < kSize; ++y)
            for (UINT x = 0; x < kSize; ++x) {
                std::uint32_t got = 0;
                std::memcpy(&got, bytes + z * slice_pitch + y * row_pitch + x * 4, 4);
                const bool in_range = sliced && z >= kFirstW && z < kFirstW + kSizeW;
                const std::uint32_t want = in_range ? value(kTag2, x, y, z - kFirstW) : value(kTag1, x, y, z);
                if (got != want && mismatches++ < 6)
                    std::fprintf(stderr, "texel (%u, %u, %u): got %08x want %08x\n", x, y, z, got, want);
            }
    if (sliced)
        for (UINT g = 0; g < load_count; ++g) {
            std::uint32_t got = 0;
            std::memcpy(&got, bytes + volume_bytes + g * 4, 4);
            const std::uint32_t want = value(kTag2, g & 7, (g >> 3) & 7, g >> 6);
            if (got != want && mismatches++ < 12)
                std::fprintf(stderr, "load %u (view z %u): got %08x want %08x\n", g, g >> 6, got, want);
        }
    readback->Unmap(0, nullptr);
    EXPECT(mismatches == 0);
    if (sliced)
        std::printf("3D UAVs: %u texels written through a whole-volume UAV, %u through a W-range UAV (slices %u..%u, a "
                    "sliced storage view) and read back through it with image_load: all match\n",
                    kSize * kSize * kSize, 64 * kSizeW, kFirstW, kFirstW + kSizeW - 1);
    else
        std::printf("3D UAVs: %u texels written through a whole-volume UAV match; W-range UAVs SKIPPED (no "
                    "imageSlicedViewOf3D)\n",
                    kSize * kSize * kSize);
    return 0;
}
