// A heap's GPU address arena range on the Vulkan-backed device, with a small arena (CTest sets LEONIDA_ARENA_MB=96:
// the title page table and the quad indices take the first 32.8 MiB, leaving about 63 MiB):
//   - texture-only heaps reserve nothing: four of 128 MiB each are created and take a placed texture, over five times
//     the arena (before, CreateHeap refused the first with E_OUTOFMEMORY);
//   - a heap that allows buffers reserves its size when the first buffer is placed: in a 128 MiB one a texture is
//     placed, and a buffer is refused with E_OUTOFMEMORY, the arena being too small;
//   - two buffers in one small heap share its one range: their VAs are their placed offsets apart, and the range goes
//     back to the arena with the heap;
//   - a buffer refused by a heap whose texture tiles have their own sparse backing reserves nothing (skipped without
//     sparse texture tiles).
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
// A 48 MiB heap for buffers and textures whose texture tiles have their own sparse backing refuses a placed buffer;
// the refusal must leave the arena free, so a 32 MiB buffer heap still fits the arena (63 MiB free). 1: pass, 0: fail,
// -1: no sparse texture tiles here (skipped).
int refused_buffer_keeps_arena(ID3D12Device* device) {
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Heap> mixed;
    const D3D12_HEAP_DESC mixed_desc = heap_desc(48 * kMiB, D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES);
    if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateHeap(&mixed_desc, IID_PPV_ARGS(&mixed.p))))
        return 0;
    D3D12_RESOURCE_DESC tiled = texture_desc();
    tiled.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    Ref<ID3D12Resource> reserved;
    if (FAILED(device->CreateReservedResource(&tiled, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                              IID_PPV_ARGS(&reserved.p))))
        return 0;
    if (!rt::d3d12::is_sparse_image(reserved.p)) return -1;
    const D3D12_TILED_RESOURCE_COORDINATE origin{};
    const D3D12_TILE_REGION_SIZE one{1, FALSE, 0, 0, 0};
    const UINT heap_offset = 0;
    queue->UpdateTileMappings(reserved.p, 1, &origin, &one, mixed.p, 1, nullptr, &heap_offset, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    Ref<ID3D12Resource> refused;
    const HRESULT hr = place(device, mixed.p, 0, buffer_desc(k64), refused);
    std::printf("      buffer in the heap with texture tiles: %#lx\n", static_cast<unsigned long>(hr));
    if (SUCCEEDED(hr)) return 0;
    Ref<ID3D12Heap> buffers;
    const D3D12_HEAP_DESC buffer_heap = heap_desc(32 * kMiB, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS);
    Ref<ID3D12Resource> buffer;
    return SUCCEEDED(device->CreateHeap(&buffer_heap, IID_PPV_ARGS(&buffers.p))) &&
           SUCCEEDED(place(device, buffers.p, 0, buffer_desc(k64), buffer));
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
        check(created, "four 128 MiB texture-only heaps with a 96 MiB arena");
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

    const int refused = refused_buffer_keeps_arena(device.p);
    if (refused < 0)
        std::printf("skip  a refused buffer placement keeps no arena space: no sparse texture tiles on this device\n");
    else
        check(refused == 1, "a buffer refused by a heap with texture tiles reserves nothing: a 32 MiB buffer heap fits");

    std::printf(failures ? "FAIL (%u)\n" : "PASS\n", failures);
    return failures ? 1 : 0;
}
