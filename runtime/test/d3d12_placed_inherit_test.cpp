// CreatePlacedResource leaves the heap's bytes alone. D3D12 zeroes a heap when it is created (unless
// D3D12_HEAP_FLAG_CREATE_NOT_ZEROED), and placing a buffer only names a range of it: a buffer placed over bytes another
// buffer already holds reads them ("Memory Aliasing and Data Inheritance": for buffers "the resource transition state
// is not important and all states are compatible"), and the buffer that held them keeps them.
// https://learn.microsoft.com/en-us/windows/win32/direct3d12/memory-aliasing-and-data-inheritance
// Checked on the Vulkan-backed device and, on Windows, on the host devices too (hardware and WARP). CPU-only: an
// upload heap, its placed buffers mapped. No shaders or game data. Skips (77) without the standalone device.
#include "host_wine.h"
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", who, __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT64 k64K = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;

D3D12_RESOURCE_DESC buffer(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_HEAP_DESC upload_heap(UINT64 bytes) {
    D3D12_HEAP_DESC d{};
    d.SizeInBytes = bytes;
    d.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
    d.Alignment = k64K;
    return d;
}

std::uint8_t pattern(UINT64 i) { return std::uint8_t(i * 7 + 1); }

HRESULT place(ID3D12Device* device, ID3D12Heap* heap, UINT64 offset, UINT64 bytes, ID3D12Resource** out) {
    const D3D12_RESOURCE_DESC desc = buffer(bytes);
    return device->CreatePlacedResource(heap, offset, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                        IID_PPV_ARGS(out));
}

// The first byte in [from, to) that is not what `want` says, or `to` when every one is.
template<class Want>
UINT64 first_mismatch(const std::uint8_t* bytes, UINT64 from, UINT64 to, Want want) {
    for (UINT64 i = from; i < to; ++i)
        if (bytes[i] != want(i)) return i;
    return to;
}

int check(ID3D12Device* device, const char* who) {
    // Two 64 KiB halves. A fills the heap and is written through its mapping.
    Ref<ID3D12Heap> heap;
    const D3D12_HEAP_DESC heap_desc = upload_heap(2 * k64K);
    EXPECT(SUCCEEDED(device->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
    Ref<ID3D12Resource> a;
    EXPECT(SUCCEEDED(place(device, heap.p, 0, 2 * k64K, &a.p)));
    std::uint8_t* a_bytes = nullptr;
    EXPECT(SUCCEEDED(a->Map(0, nullptr, reinterpret_cast<void**>(&a_bytes))) && a_bytes);
    // A new heap reads as zeroed.
    EXPECT(first_mismatch(a_bytes, 0, 2 * k64K, [](UINT64) { return std::uint8_t(0); }) == 2 * k64K);
    for (UINT64 i = 0; i < 2 * k64K; ++i) a_bytes[i] = pattern(i);

    // B is placed over the first half, C over the second. Neither changes a byte: A still reads its pattern
    // everywhere, and B and C read the part of it they cover.
    Ref<ID3D12Resource> b, c;
    EXPECT(SUCCEEDED(place(device, heap.p, 0, 4096, &b.p)));
    EXPECT(SUCCEEDED(place(device, heap.p, k64K, k64K, &c.p)));
    const UINT64 a_kept = first_mismatch(a_bytes, 0, 2 * k64K, pattern);
    if (a_kept != 2 * k64K)
        std::fprintf(stderr, "%s: after placing B and C, A's byte %llu reads 0x%02x, it held 0x%02x\n", who,
                     static_cast<unsigned long long>(a_kept), a_bytes[a_kept], pattern(a_kept));
    EXPECT(a_kept == 2 * k64K);
    std::uint8_t* b_bytes = nullptr;
    std::uint8_t* c_bytes = nullptr;
    EXPECT(SUCCEEDED(b->Map(0, nullptr, reinterpret_cast<void**>(&b_bytes))) && b_bytes);
    EXPECT(SUCCEEDED(c->Map(0, nullptr, reinterpret_cast<void**>(&c_bytes))) && c_bytes);
    EXPECT(first_mismatch(b_bytes, 0, 4096, pattern) == 4096);
    EXPECT(first_mismatch(c_bytes, 0, k64K, [](UINT64 i) { return pattern(k64K + i); }) == k64K);

    // Writes through B reach A, the same bytes.
    for (UINT64 i = 0; i < 4096; ++i) b_bytes[i] = std::uint8_t(~pattern(i));
    EXPECT(first_mismatch(a_bytes, 0, 4096, [](UINT64 i) { return std::uint8_t(~pattern(i)); }) == 4096);
    a->Unmap(0, nullptr);
    b->Unmap(0, nullptr);
    c->Unmap(0, nullptr);

    // A heap made after this one is gone reads as zeroed again, wherever it lands, before anything is placed in
    // it twice.
    a.reset();
    b.reset();
    c.reset();
    heap.reset();
    Ref<ID3D12Heap> again;
    EXPECT(SUCCEEDED(device->CreateHeap(&heap_desc, IID_PPV_ARGS(&again.p))));
    Ref<ID3D12Resource> d;
    EXPECT(SUCCEEDED(place(device, again.p, 0, 2 * k64K, &d.p)));
    std::uint8_t* d_bytes = nullptr;
    EXPECT(SUCCEEDED(d->Map(0, nullptr, reinterpret_cast<void**>(&d_bytes))) && d_bytes);
    EXPECT(first_mismatch(d_bytes, 0, 2 * k64K, [](UINT64) { return std::uint8_t(0); }) == 2 * k64K);
    d->Unmap(0, nullptr);

    std::printf("%s: placing buffers keeps the heap's bytes; a new heap reads as zeroed\n", who);
    return 0;
}

}  // namespace

int main() {
    Ref<ID3D12Device> ours;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&ours.p))) || !ours.p) {
        std::printf("placed inherit: no Vulkan device, skipped\n");
        return 77;
    }
    int failures = check(ours.p, "vulkan");
    ours.reset();

    Ref<IDXGIFactory4> factory;
    if (!hostwine::host_is_wine() && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        Ref<IDXGIAdapter1> adapter;
        for (UINT i = 0; factory->EnumAdapters1(i, &adapter.p) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 desc{};
            adapter->GetDesc1(&desc);
            Ref<ID3D12Device> host;
            if (SUCCEEDED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
                char name[128] = {};
                WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
                failures += check(host.p, (std::string("host ") + name).c_str());
            }
            adapter.reset();
        }
    }
    if (failures) return 1;
    std::printf("d3d12_placed_inherit_test: OK\n");
    return 0;
}
