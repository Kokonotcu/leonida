// Console Xbox CreateDescriptorHeap (issue #313 comment 7107265058) passes a 16-byte PC
// D3D12_DESCRIPTOR_HEAP_DESC followed by address at +0x10 and size at +0x18. The title asserts
// GetCPUDescriptorHandleForHeapStart() == address. This tests the explicit Leonida console-heap
// creation path used by the Xbox device proxy slot 14. It does not cover title-memory aliasing /
// section 3.3 backing.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <cstdint>
#include <cstdio>

namespace {

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T** put() {
        if (p) {
            p->Release();
            p = nullptr;
        }
        return &p;
    }
    T* get() const { return p; }
    T* operator->() const { return p; }
};

bool check_console_addressed_heap(ID3D12Device* device) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 8;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    desc.NodeMask = 0;
    const D3D12_GPU_VIRTUAL_ADDRESS address = 0x112233440000ull;
    const std::uint64_t size = 8ull * 32ull;

    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = rt::d3d12::create_console_descriptor_heap(
        device, &desc, address, size, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "create_console_descriptor_heap failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return false;
    }
    const auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    if (cpu.ptr != static_cast<SIZE_T>(address)) {
        std::fprintf(stderr, "CPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(cpu.ptr), static_cast<unsigned long long>(address));
        return false;
    }
    const auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
    if (gpu.ptr != address) {
        std::fprintf(stderr, "GPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(gpu.ptr), static_cast<unsigned long long>(address));
        return false;
    }
    const D3D12_DESCRIPTOR_HEAP_DESC got = heap->GetDesc();
    if (got.Type != desc.Type || got.NumDescriptors != desc.NumDescriptors || got.Flags != desc.Flags ||
        got.NodeMask != 1) {
        std::fprintf(stderr, "GetDesc mismatch\n");
        return false;
    }
    return true;
}

bool check_xs_proxy_slot14(ID3D12Device* real_device) {
    // Slot 14's console-extended path is opt-in (LEONIDA_CONSOLE_HEAP_VA=1 registers this handler at device creation).
    rt::d3d12_xs::set_console_descriptor_heap_handler(&rt::d3d12::create_console_descriptor_heap);
    struct Unregister {
        ~Unregister() { rt::d3d12_xs::set_console_descriptor_heap_handler(nullptr); }
    } unregister;
    // The proxy takes ownership of one reference to the device (d3d12_xs_device.h); main keeps its own.
    real_device->AddRef();
    void* proxy_raw = nullptr;
    if (FAILED(rt::d3d12_xs::create_device_proxy(real_device, __uuidof(ID3D12Device), &proxy_raw)) || !proxy_raw) {
        real_device->Release();
        std::fprintf(stderr, "create_device_proxy failed\n");
        return false;
    }

    struct ConsoleDescriptorHeapDesc {
        D3D12_DESCRIPTOR_HEAP_DESC desc{};
        std::uint64_t address = 0;
        std::uint64_t size = 0;
    } console{};
    console.desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    console.desc.NumDescriptors = 4;
    console.desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    console.address = 0x556677880000ull;
    console.size = 4ull * 32ull;

    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = proxy->CreateDescriptorHeap(&console.desc, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "proxy CreateDescriptorHeap(console) failed: 0x%08lx\n",
                     static_cast<unsigned long>(hr));
        return false;
    }
    // The heap proxy answers in the console's ABI, as the title calls it: slot 9 returns the handle in RAX. The PC
    // C++ call (a hidden return pointer) is not how the proxy is reached.
    using CpuStartFn = std::uint64_t(WINAPI*)(void*);
    void* const heap_object = heap.get();
    const std::uint64_t cpu_ptr = reinterpret_cast<CpuStartFn>((*reinterpret_cast<void***>(heap_object))[9])(heap_object);
    if (cpu_ptr != console.address) {
        std::fprintf(stderr, "proxy CPU heap start mismatch: got 0x%llx want 0x%llx\n",
                     static_cast<unsigned long long>(cpu_ptr),
                     static_cast<unsigned long long>(console.address));
        return false;
    }
    return true;
}

bool check_normal_heap_unchanged(ID3D12Device* device) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 4;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    Ref<ID3D12DescriptorHeap> heap;
    const HRESULT hr = device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(heap.put()));
    if (FAILED(hr) || !heap.get()) {
        std::fprintf(stderr, "CreateDescriptorHeap(normal) failed: 0x%08lx\n", static_cast<unsigned long>(hr));
        return false;
    }
    const auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
    if (cpu.ptr == 0) {
        std::fprintf(stderr, "normal heap CPU start is null\n");
        return false;
    }
    return true;
}

}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    Ref<ID3D12Device> device;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(device.put()))) || !device.get()) {
        std::fprintf(stderr, "create_vulkan_device failed\n");
        return 1;
    }

    if (!check_normal_heap_unchanged(device.get())) return 1;
    if (!check_console_addressed_heap(device.get())) return 1;
    if (!check_xs_proxy_slot14(device.get())) return 1;
    std::fprintf(stderr, "ok\n");
    return 0;
}
