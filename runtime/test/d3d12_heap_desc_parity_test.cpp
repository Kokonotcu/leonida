// Host + WARP parity test for ID3D12Device::CreateHeap (heap_desc_refusal and null-output S_FALSE capability probe)
// and ID3D12Device::CopyDescriptors / CopyDescriptorsSimple descriptor-heap-type validation.
// No shaders or game data. Skips (77) without a Vulkan GPU. Set LEONIDA_VK_VALIDATION=1 to run under validation.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition)                                                            \
    do {                                                                             \
        if (!(condition)) {                                                          \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);             \
            return 1;                                                                \
        }                                                                            \
    } while (0)

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

D3D12_HEAP_DESC make_heap(D3D12_HEAP_TYPE type, D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE,
                          UINT64 size = 64 * 1024, UINT64 alignment = 0,
                          D3D12_CPU_PAGE_PROPERTY cpu_page = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                          D3D12_MEMORY_POOL pool = D3D12_MEMORY_POOL_UNKNOWN) {
    D3D12_HEAP_DESC desc{};
    desc.SizeInBytes = size;
    desc.Properties.Type = type;
    desc.Properties.CPUPageProperty = cpu_page;
    desc.Properties.MemoryPoolPreference = pool;
    desc.Alignment = alignment;
    desc.Flags = flags;
    return desc;
}

struct HeapCase {
    const char* name;
    D3D12_HEAP_DESC desc;
    bool null_out = false;
    bool bad_riid = false;
    HRESULT expected = S_OK;
};

HRESULT run_heap_case(ID3D12Device* dev, const HeapCase& c) {
    void* out = nullptr;
    const IID iid = c.bad_riid ? __uuidof(ID3D12Fence) : __uuidof(ID3D12Heap);
    const HRESULT hr = dev->CreateHeap(&c.desc, iid, c.null_out ? nullptr : &out);
    if (out) static_cast<IUnknown*>(out)->Release();
    return hr;
}

bool create_host_devices(Ref<ID3D12Device>& hw, Ref<ID3D12Device>& warp) {
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create_dev =
        d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
              : nullptr;
    if (!create_dev) return false;
    create_dev(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hw.p));

    Ref<IDXGIFactory4> factory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) && factory.p) {
        Ref<IDXGIAdapter> adapter;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) && adapter.p) {
            create_dev(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p));
        }
    }
    return hw.p != nullptr || warp.p != nullptr;
}

int test_copy_descriptors(ID3D12Device* dev) {
    const UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    EXPECT(inc > 0);

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 4;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;

    Ref<ID3D12DescriptorHeap> src_heap, dst_heap;
    EXPECT(SUCCEEDED(dev->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&src_heap.p))));
    EXPECT(SUCCEEDED(dev->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&dst_heap.p))));

    const D3D12_CPU_DESCRIPTOR_HANDLE src_base = src_heap->GetCPUDescriptorHandleForHeapStart();
    const D3D12_CPU_DESCRIPTOR_HANDLE dst_base = dst_heap->GetCPUDescriptorHandleForHeapStart();

    // Populate source slots 0 and 1 with distinct CBVs, and leave destination slots zeroed.
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv0{0x10000000ull, 256};
    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv1{0x20000000ull, 512};
    dev->CreateConstantBufferView(&cbv0, {src_base.ptr + 0 * inc});
    dev->CreateConstantBufferView(&cbv1, {src_base.ptr + 1 * inc});
    dev->CreateConstantBufferView(nullptr, {dst_base.ptr + 0 * inc});
    dev->CreateConstantBufferView(nullptr, {dst_base.ptr + 1 * inc});
    dev->CreateConstantBufferView(nullptr, {dst_base.ptr + 2 * inc});
    dev->CreateConstantBufferView(nullptr, {dst_base.ptr + 3 * inc});

    std::vector<std::uint8_t> src0(inc), src1(inc), zero(inc, 0);
    std::memcpy(src0.data(), reinterpret_cast<const void*>(src_base.ptr + 0 * inc), inc);
    std::memcpy(src1.data(), reinterpret_cast<const void*>(src_base.ptr + 1 * inc), inc);
    EXPECT(std::memcmp(src0.data(), zero.data(), inc) != 0);
    EXPECT(std::memcmp(src1.data(), zero.data(), inc) != 0);
    EXPECT(std::memcmp(src0.data(), src1.data(), inc) != 0);

    // Out-of-range heap types must be safely ignored without modifying destination slots.
    const D3D12_DESCRIPTOR_HEAP_TYPE bad_types[] = {
        D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES,
        static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(99),
        static_cast<D3D12_DESCRIPTOR_HEAP_TYPE>(-1),
    };
    for (const D3D12_DESCRIPTOR_HEAP_TYPE bad : bad_types) {
        dev->CopyDescriptorsSimple(2, dst_base, src_base, bad);
        EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 0 * inc), zero.data(), inc) == 0);
        EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 1 * inc), zero.data(), inc) == 0);

        const UINT count = 2;
        dev->CopyDescriptors(1, &dst_base, &count, 1, &src_base, &count, bad);
        EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 0 * inc), zero.data(), inc) == 0);
        EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 1 * inc), zero.data(), inc) == 0);
    }

    // Valid CopyDescriptorsSimple copies exact bytes.
    dev->CopyDescriptorsSimple(2, dst_base, src_base, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 0 * inc), src0.data(), inc) == 0);
    EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 1 * inc), src1.data(), inc) == 0);

    // Valid CopyDescriptors copies to slots 2 and 3.
    const D3D12_CPU_DESCRIPTOR_HANDLE dst_ranges[2] = {{dst_base.ptr + 2 * inc}, {dst_base.ptr + 3 * inc}};
    const D3D12_CPU_DESCRIPTOR_HANDLE src_ranges[1] = {src_base};
    const UINT src_sizes[1] = {2};
    dev->CopyDescriptors(2, dst_ranges, nullptr, 1, src_ranges, src_sizes, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 2 * inc), src0.data(), inc) == 0);
    EXPECT(std::memcmp(reinterpret_cast<const void*>(dst_base.ptr + 3 * inc), src1.data(), inc) == 0);

    // Count == 0 is a safe no-op.
    dev->CopyDescriptorsSimple(0, dst_base, src_base, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return 0;
}

}  // namespace

int main() {
    Ref<ID3D12Device> ours;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&ours.p))) || !ours.p) {
        std::printf("d3d12_heap_desc_parity_test: no Vulkan device, skipped\n");
        return 77;
    }

    const HeapCase cases[] = {
        // Capability probe (out == nullptr)
        {"DEFAULT valid (out==null)", make_heap(D3D12_HEAP_TYPE_DEFAULT), true, false, S_FALSE},
        {"DEFAULT valid (out==null, bad riid)", make_heap(D3D12_HEAP_TYPE_DEFAULT), true, true, S_FALSE},
        {"UPLOAD valid (out==null)", make_heap(D3D12_HEAP_TYPE_UPLOAD), true, false, S_FALSE},
        {"READBACK valid (out==null)", make_heap(D3D12_HEAP_TYPE_READBACK), true, false, S_FALSE},
        {"DEFAULT Size=0 (out==null)", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 0), true, false,
         E_INVALIDARG},
        {"DEFAULT Align=4096 (out==null)", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 65536, 4096), true,
         false, E_INVALIDARG},
        {"DEFAULT unknown flag 0x2000 (out==null)",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, static_cast<D3D12_HEAP_FLAGS>(0x2000)), true, false, E_INVALIDARG},
        {"DEFAULT SHARED_CROSS_ADAPTER alone (out==null)",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER), true, false, E_INVALIDARG},

        // Bad riid with non-null output
        {"DEFAULT valid (bad riid)", make_heap(D3D12_HEAP_TYPE_DEFAULT), false, true, E_NOINTERFACE},

        // Size and alignment
        {"DEFAULT Size=0", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 0), false, false, E_INVALIDARG},
        {"DEFAULT Align=0", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 65536, 0), false, false, S_OK},
        {"DEFAULT Align=64K",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 65536, D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT),
         false, false, S_OK},
        {"DEFAULT Align=4M",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT,
                   D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT),
         false, false, S_OK},
        {"DEFAULT Align=4096", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 65536, 4096), false, false,
         E_INVALIDARG},
        {"DEFAULT Align=128K", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 131072, 131072), false, false,
         E_INVALIDARG},
        {"DEFAULT Align=8M", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 8 * 1024 * 1024, 8 * 1024 * 1024),
         false, false, E_INVALIDARG},

        // Unknown flag bits and ALLOW_DISPLAY
        {"DEFAULT unknown flag 0x2", make_heap(D3D12_HEAP_TYPE_DEFAULT, static_cast<D3D12_HEAP_FLAGS>(0x2)), false,
         false, E_INVALIDARG},
        {"DEFAULT unknown flag 0x10", make_heap(D3D12_HEAP_TYPE_DEFAULT, static_cast<D3D12_HEAP_FLAGS>(0x10)), false,
         false, E_INVALIDARG},
        {"DEFAULT unknown flag 0x2000", make_heap(D3D12_HEAP_TYPE_DEFAULT, static_cast<D3D12_HEAP_FLAGS>(0x2000)),
         false, false, E_INVALIDARG},
        {"DEFAULT unknown flag 0x80000000",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, static_cast<D3D12_HEAP_FLAGS>(0x80000000u)), false, false, E_INVALIDARG},
        {"DEFAULT ALLOW_DISPLAY", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_DISPLAY), false, false,
         E_INVALIDARG},

        // DENY flag combinations
        {"DEFAULT DENY_BUFFERS alone", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_DENY_BUFFERS), false, false,
         E_INVALIDARG},
        {"DEFAULT DENY_RT_DS_TEXTURES alone", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES),
         false, false, E_INVALIDARG},
        {"DEFAULT DENY_NON_RT_DS_TEXTURES alone",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES), false, false, E_INVALIDARG},
        {"DEFAULT all three DENY flags",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES |
                                                D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES),
         false, false, E_INVALIDARG},
        {"DEFAULT ALLOW_ONLY_BUFFERS", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), false,
         false, S_OK},
        {"DEFAULT ALLOW_ONLY_NON_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES), false, false, S_OK},
        {"DEFAULT ALLOW_ONLY_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES), false, false, S_OK},

        // UPLOAD / READBACK DENY and flag parity
        {"UPLOAD DENY_BUFFERS alone", make_heap(D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_DENY_BUFFERS), false, false,
         E_INVALIDARG},
        {"UPLOAD ALLOW_ONLY_BUFFERS", make_heap(D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), false,
         false, S_OK},
        {"UPLOAD ALLOW_ONLY_NON_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES), false, false, S_OK},
        {"UPLOAD ALLOW_ONLY_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES), false, false, S_OK},
        {"READBACK DENY_BUFFERS alone", make_heap(D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_DENY_BUFFERS), false, false,
         E_INVALIDARG},
        {"READBACK ALLOW_ONLY_BUFFERS", make_heap(D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS), false,
         false, S_OK},
        {"READBACK ALLOW_ONLY_NON_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES), false, false, S_OK},
        {"READBACK ALLOW_ONLY_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES), false, false, S_OK},

        // SHARED and SHARED_CROSS_ADAPTER
        {"DEFAULT SHARED", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED), false, false, S_OK},
        {"UPLOAD SHARED", make_heap(D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_SHARED), false, false, E_INVALIDARG},
        {"READBACK SHARED", make_heap(D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_SHARED), false, false, E_INVALIDARG},
        {"CUSTOM WC L0 SHARED",
         make_heap(D3D12_HEAP_TYPE_CUSTOM, D3D12_HEAP_FLAG_SHARED, 65536, 0, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                   D3D12_MEMORY_POOL_L0),
         false, false, E_INVALIDARG},
        {"CUSTOM NA L0 SHARED",
         make_heap(D3D12_HEAP_TYPE_CUSTOM, D3D12_HEAP_FLAG_SHARED, 65536, 0, D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
                   D3D12_MEMORY_POOL_L0),
         false, false, S_OK},
        {"DEFAULT SHARED_CROSS_ADAPTER alone",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER), false, false, E_INVALIDARG},
        {"DEFAULT SHARED | SHARED_CROSS_ADAPTER",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER), false,
         false, S_OK},
        {"DEFAULT SHARED | SHARED_CROSS_ADAPTER | ALLOW_ONLY_BUFFERS",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
                                                D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
         false, false, E_INVALIDARG},
        {"DEFAULT SHARED | SHARED_CROSS_ADAPTER | ALLOW_ONLY_NON_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
                                                D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES),
         false, false, E_INVALIDARG},
        {"DEFAULT SHARED | SHARED_CROSS_ADAPTER | ALLOW_ONLY_RT_DS_TEXTURES",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER |
                                                D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES),
         false, false, E_INVALIDARG},

        // ALLOW_WRITE_WATCH and ALLOW_SHADER_ATOMICS
        {"DEFAULT ALLOW_WRITE_WATCH", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH), false,
         false, E_INVALIDARG},
        {"UPLOAD ALLOW_WRITE_WATCH", make_heap(D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH), false, false,
         S_OK},
        {"READBACK ALLOW_WRITE_WATCH", make_heap(D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH), false,
         false, S_OK},
        {"DEFAULT ALLOW_SHADER_ATOMICS", make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS),
         false, false, S_OK},
        {"UPLOAD ALLOW_SHADER_ATOMICS", make_heap(D3D12_HEAP_TYPE_UPLOAD, D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS), false,
         false, E_INVALIDARG},
        {"READBACK ALLOW_SHADER_ATOMICS", make_heap(D3D12_HEAP_TYPE_READBACK, D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS),
         false, false, E_INVALIDARG},
        {"DEFAULT CREATE_NOT_RESIDENT | CREATE_NOT_ZEROED",
         make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT | D3D12_HEAP_FLAG_CREATE_NOT_ZEROED),
         false, false, S_OK},
    };

    Ref<ID3D12Device> hw, warp;
    create_host_devices(hw, warp);

    int mismatches = 0;
    int hw_compared = 0;
    int warp_compared = 0;
    for (const HeapCase& c : cases) {
        const HRESULT got = run_heap_case(ours.p, c);
        if (got != c.expected) {
            std::fprintf(stderr, "ours mismatch [%s]: got 0x%08lx, expected 0x%08lx\n", c.name, got, c.expected);
            ++mismatches;
        }
        if (hw.p) {
            const HRESULT hw_hr = run_heap_case(hw.p, c);
            ++hw_compared;
            if (got != hw_hr) {
                std::fprintf(stderr, "host-hw mismatch [%s]: ours 0x%08lx, host-hw 0x%08lx\n", c.name, got, hw_hr);
                ++mismatches;
            }
        }
        if (warp.p) {
            const HRESULT warp_hr = run_heap_case(warp.p, c);
            ++warp_compared;
            if (got != warp_hr) {
                std::fprintf(stderr, "warp mismatch [%s]: ours 0x%08lx, warp 0x%08lx\n", c.name, got, warp_hr);
                ++mismatches;
            }
        }
    }
    EXPECT(mismatches == 0);

    // Capability probe with out == nullptr must not reserve GPU address-arena space even for a huge heap size.
    const D3D12_HEAP_DESC huge = make_heap(D3D12_HEAP_TYPE_DEFAULT, D3D12_HEAP_FLAG_NONE, 16ull * 1024 * 1024 * 1024);
    EXPECT(ours->CreateHeap(&huge, __uuidof(ID3D12Heap), nullptr) == S_FALSE);

    // Null desc must return E_INVALIDARG cleanly.
    void* out = reinterpret_cast<void*>(0x1);
    EXPECT(ours->CreateHeap(nullptr, __uuidof(ID3D12Heap), &out) == E_INVALIDARG);
    EXPECT(out == nullptr);

    if (test_copy_descriptors(ours.p) != 0) return 1;

    std::printf("d3d12_heap_desc_parity_test: %zu cases passed (host-hw compared: %d, warp compared: %d, mismatches: %d)\n",
                sizeof(cases) / sizeof(cases[0]), hw_compared, warp_compared, mismatches);
    return 0;
}
