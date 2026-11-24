// M10: compute root arguments past the 32 user-data dwords go to the console's spill array, through the D3D12 API on
// the runtime's own device, with a translator-output shader (test/shaders/cs_spill.asm) and every word checked.
// Root signature [0] 27 constants (s4..s30), [1] a root UAV, [2] 3 constants: [1] does not fit in s31, so [1] and [2]
// are packed behind the 40-bit pointer in s[0:1] (docs/d3d12_vulkan.md section 7; RootArgumentLayout).
//   - Two dispatches with different spilled arguments (another UAV address and other constants) in one list: each
//     reads its own array, so the arrays are a snapshot per dispatch, as root arguments are.
//   - Binding the same signature for graphics is refused (Close fails): the graphics spill pointer is unmeasured.
//   - A signature whose spill would hold a descriptor table is still created but unusable for compute: how many bytes
//     a table takes there is unmeasured, so binding it invalidates the list rather than guess.
// Before M10 the compute layout refused the signature, so SetComputeRootSignature invalidated the list and Close
// failed. 77: no Vulkan device.
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

#ifndef LEONIDA_TEST_SPILL_DIR
#error "LEONIDA_TEST_SPILL_DIR must name the folder build.py wrote (spirv/index.json, fssm/cs_spill.fssm)"
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

constexpr UINT kInline = 27, kSpilled = 3, kThreads = 128;

struct Args {
    std::uint32_t s4, s30, c0, c1, c2;
    UINT64 offset;  // of the UAV range in the output buffer
};

std::uint32_t expected(const Args& a, std::uint32_t g) { return (((g + a.s4) ^ a.c0) + a.c1 - a.c2) + a.s30; }

// Ref has no copy semantics, so the signature comes back through `root`.
bool make_root(ID3D12Device* device, D3D12_ROOT_PARAMETER_TYPE middle, UINT inline_count, Ref<ID3D12RootSignature>& root) {
    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER params[3] = {};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants = {0, 0, inline_count};
    params[1].ParameterType = middle;
    if (middle == D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE)
        params[1].DescriptorTable = {1, &range};
    else
        params[1].Descriptor = {0, 0};
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants = {1, 0, kSpilled};
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};
    desc.Version = D3D_ROOT_SIGNATURE_VERSION_1_0;
    desc.Desc_1_0 = {3, params, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    const auto blob = serialize_root_signature(desc);
    return !blob.empty() && SUCCEEDED(device->CreateRootSignature(0, blob.data(), blob.size(), IID_PPV_ARGS(&root.p)));
}

}  // namespace

int main() {
    const std::filesystem::path root_dir = LEONIDA_TEST_SPILL_DIR;
    _wputenv_s(L"LEONIDA_SHADERS", (root_dir / "spirv").wstring().c_str());
    _wputenv_s(L"LEONIDA_PSO_RECORD", L"0");
    _wputenv_s(L"LEONIDA_PIPELINE_CACHE_PERSIST", L"0");
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) {
        std::puts("SKIP: no Vulkan device for the runtime");
        return 77;
    }
    const auto program = read_file(root_dir / "fssm" / "cs_spill.fssm");
    EXPECT(!program.empty());
    const auto cs = shader_container(program);

    Ref<ID3D12RootSignature> root;
    EXPECT(make_root(device.p, D3D12_ROOT_PARAMETER_TYPE_UAV, kInline, root));
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc{};
    pso_desc.pRootSignature = root.p;
    pso_desc.CS = {cs.data(), cs.size()};
    Ref<ID3D12PipelineState> pso;
    EXPECT(SUCCEEDED(device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso.p))));

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    constexpr UINT64 bytes = 2 * 4096;
    const auto out_desc = buffer_desc(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    Ref<ID3D12Resource> out, readback;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &out_desc,
                                                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
                                                     IID_PPV_ARGS(&out.p))));
    const auto readback_desc = buffer_desc(bytes);
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback.p))));

    D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
    Ref<ID3D12CommandQueue> queue;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    Ref<ID3D12GraphicsCommandList> list;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, pso.p,
                                               IID_PPV_ARGS(&list.p))));

    // Graphics: the same signature is refused there, loudly (Close fails), never bound with arguments dropped.
    list->SetGraphicsRootSignature(root.p);
    EXPECT(list->Close() == E_INVALIDARG);
    EXPECT(SUCCEEDED(list->Reset(allocator.p, pso.p)));

    const Args runs[2] = {{0x100, 0x7000, 0x55aa55aa, 0x01020304, 0x00000111, 0},
                          {0x2000, 0x31, 0x0f0f0f0f, 0x10203040, 0x00002222, 4096}};
    list->SetComputeRootSignature(root.p);
    for (const Args& a : runs) {
        std::uint32_t inline_values[kInline] = {};
        for (UINT i = 0; i < kInline; ++i) inline_values[i] = 0xdead0000u + i;  // only s4 and s30 are read
        inline_values[0] = a.s4;
        inline_values[kInline - 1] = a.s30;
        list->SetComputeRoot32BitConstants(0, kInline, inline_values, 0);
        list->SetComputeRootUnorderedAccessView(1, out->GetGPUVirtualAddress() + a.offset);
        const std::uint32_t spilled[kSpilled] = {a.c0, a.c1, a.c2};
        list->SetComputeRoot32BitConstants(2, kSpilled, spilled, 0);
        list->Dispatch(kThreads / 64, 1, 1);
    }
    const D3D12_RESOURCE_BARRIER to_copy =
        transition(out.p, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    list->CopyBufferRegion(readback.p, 0, out.p, 0, bytes);
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
    const auto* bytes_out = static_cast<const std::uint8_t*>(mapped);
    unsigned mismatches = 0;
    for (const Args& a : runs)
        for (std::uint32_t g = 0; g < kThreads; ++g) {
            std::uint32_t got = 0;
            std::memcpy(&got, bytes_out + a.offset + 4 * g, 4);
            if (got != expected(a, g) && mismatches++ < 8)
                std::fprintf(stderr, "run at +%llu, g %u: got %08x want %08x\n", (unsigned long long)a.offset, g, got,
                             expected(a, g));
        }
    readback->Unmap(0, nullptr);
    EXPECT(mismatches == 0);

    // A descriptor table in the spill (28 constants fill s4..s31 first): created, as on the console, but not usable
    // for compute here.
    Ref<ID3D12RootSignature> table_root;
    EXPECT(make_root(device.p, D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE, 28, table_root));
    EXPECT(SUCCEEDED(allocator->Reset()));
    EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
    list->SetComputeRootSignature(table_root.p);
    EXPECT(list->Close() == E_INVALIDARG);

    std::printf("root spill: 2 dispatches read a root UAV and 3 constants from their own spill arrays behind s[0:1] "
                "(%u words match); graphics and a spilled table are refused\n",
                2 * kThreads);
    return 0;
}
