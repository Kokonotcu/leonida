// A heap's GPU address arena range on the Vulkan-backed device, with a small arena (CTest sets LEONIDA_ARENA_MB=64):
//   - texture-only heaps reserve nothing: four of 128 MiB each are created and take a placed texture, eight times
//     the arena (before, CreateHeap refused the first with E_OUTOFMEMORY);
//   - a heap that allows buffers reserves its size when the first buffer is placed: in a 128 MiB one a texture is
//     placed, and a buffer is refused with E_OUTOFMEMORY, the arena being too small;
//   - two buffers in one small heap share its one range: their VAs are their placed offsets apart, and the range goes
//     back to the arena with the heap.
// No game data or shaders. LEONIDA_VK_VALIDATION=1 adds VK_LAYER_KHRONOS_validation to the layer's device.
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

constexpr UINT64 kMiB = 1024 * 1024;
constexpr UINT64 k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
unsigned failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++failures;
}

D3D12_HEAP_DESC heap_desc(UINT64 bytes, D3D12_HEAP_FLAGS flags) {
    D3D12_HEAP_DESC desc{};
    desc.SizeInBytes = bytes;
    desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC texture_desc() {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 256;
    desc.Height = 256;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

HRESULT place(ID3D12Device* device, ID3D12Heap* heap, UINT64 offset, const D3D12_RESOURCE_DESC& desc,
              Ref<ID3D12Resource>& out) {
    return device->CreatePlacedResource(heap, offset, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                        IID_PPV_ARGS(&out.p));
}

// Two 64 KiB buffers placed 64 KiB apart in a 1 MiB heap: their VAs are 64 KiB apart (one range for the heap).
bool small_buffer_heap(ID3D12Device* device) {
    Ref<ID3D12Heap> heap;
    const D3D12_HEAP_DESC desc = heap_desc(kMiB, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
    if (FAILED(device->CreateHeap(&desc, IID_PPV_ARGS(&heap.p)))) return false;
    Ref<ID3D12Resource> a, b;
    if (FAILED(place(device, heap.p, 0, buffer_desc(k64), a)) || FAILED(place(device, heap.p, k64, buffer_desc(k64), b)))
        return false;
    const D3D12_GPU_VIRTUAL_ADDRESS va_a = a->GetGPUVirtualAddress(), va_b = b->GetGPUVirtualAddress();
    std::printf("      placed buffer VAs 0x%llx, 0x%llx\n", static_cast<unsigned long long>(va_a),
                static_cast<unsigned long long>(va_b));
    return va_a && va_b == va_a + k64;
}
}  // namespace

int main() {
    Ref<ID3D12Device> device;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p))) || !device.p) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;
    }

    {
        Ref<ID3D12Heap> heaps[4];
        Ref<ID3D12Resource> textures[4];
        bool created = true, placed = true;
        for (int i = 0; i < 4; ++i) {
            const D3D12_HEAP_DESC desc =
                heap_desc(128 * kMiB, D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES);
            created = created && SUCCEEDED(device->CreateHeap(&desc, IID_PPV_ARGS(&heaps[i].p)));
            placed = placed && created && SUCCEEDED(place(device.p, heaps[i].p, 0, texture_desc(), textures[i]));
        }
        check(created, "four 128 MiB texture-only heaps with a 64 MiB arena");
        check(placed, "a placed texture in each");
    }

    {
        Ref<ID3D12Heap> heap;
        const D3D12_HEAP_DESC desc = heap_desc(128 * kMiB, D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES);
        const bool created = SUCCEEDED(device->CreateHeap(&desc, IID_PPV_ARGS(&heap.p)));
        check(created, "a 128 MiB heap for buffers and textures");
        Ref<ID3D12Resource> texture, buffer;
        check(created && SUCCEEDED(place(device.p, heap.p, 0, texture_desc(), texture)), "a texture placed in it");
        const HRESULT hr = created ? place(device.p, heap.p, 64 * kMiB, buffer_desc(k64), buffer) : E_FAIL;
        check(hr == E_OUTOFMEMORY && !buffer.p, "a buffer placed in it: E_OUTOFMEMORY, the arena can't take the heap");
    }

    check(small_buffer_heap(device.p), "two buffers in a 1 MiB heap, 64 KiB apart");
    // The first heap's range went back with it, so the arena takes the same again.
    check(small_buffer_heap(device.p), "again, after the first heap was released");

    std::printf(failures ? "FAIL (%u)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
