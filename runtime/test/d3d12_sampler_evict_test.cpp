// Samplers that no descriptor-heap slot holds any more are evicted when set 0's sampler array gets near full (D5).
// Before, every distinct S# CreateSampler saw kept its VkSampler, its set-0 index and its hash-table key for the
// device's life: dynamic resolution's mip-bias steps (a new S# each) filled the array (at most
// maxSamplerAllocationCount less the reserved ones), after which every new sampler was refused and its shaders
// sampled with the fallback (point, clamp).
//
// A D3D12 device on a real Vulkan device; vkCreateSampler and vkDestroySampler are intercepted to count live
// samplers. One sampler-heap slot is rewritten with more distinct S#s than the device can hold samplers, with an
// ExecuteCommandLists and a fence wait every 64 writes so the evictions' submissions complete:
//   - the last S# written still gets its sampler (before: the array was full, the new sampler destroyed at once);
//   - live samplers never exceed maxSamplerAllocationCount;
//   - the sampler of an S# another slot keeps is never evicted.
// Skips with exit code 77 without a Vulkan 1.3 device with the bindless features. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_set>

namespace {

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

namespace driver {
PFN_vkGetInstanceProcAddr real_instance = nullptr;
PFN_vkGetDeviceProcAddr real_device = nullptr;
PFN_vkCreateSampler real_create = nullptr;
PFN_vkDestroySampler real_destroy = nullptr;
std::mutex mutex;
std::unordered_set<std::uint64_t> live;
std::size_t peak = 0;
VkSampler last_created = VK_NULL_HANDLE;

std::uint64_t raw(VkSampler s) { return std::uint64_t(reinterpret_cast<std::uintptr_t>(s)); }

VKAPI_ATTR VkResult VKAPI_CALL create_sampler(VkDevice device, const VkSamplerCreateInfo* info,
                                              const VkAllocationCallbacks* allocator, VkSampler* sampler) {
    const VkResult result = real_create(device, info, allocator, sampler);
    if (result == VK_SUCCESS) {
        std::lock_guard lock(mutex);
        live.insert(raw(*sampler));
        peak = std::max(peak, live.size());
        last_created = *sampler;
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL destroy_sampler(VkDevice device, VkSampler sampler, const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(mutex);
        live.erase(raw(sampler));
    }
    real_destroy(device, sampler, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkCreateSampler") == 0) return reinterpret_cast<PFN_vkVoidFunction>(create_sampler);
    if (std::strcmp(name, "vkDestroySampler") == 0) return reinterpret_cast<PFN_vkVoidFunction>(destroy_sampler);
    return real_device(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (std::strcmp(name, "vkGetDeviceProcAddr") == 0) return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_instance(instance, name);
}
}  // namespace driver

int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

D3D12_SAMPLER_DESC sampler_desc(float bias) {
    D3D12_SAMPLER_DESC desc{};
    desc.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    desc.AddressU = desc.AddressV = desc.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    desc.MipLODBias = bias;
    desc.MaxAnisotropy = 1;
    desc.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    desc.MaxLOD = D3D12_FLOAT32_MAX;
    return desc;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!vk.bindless) {
        std::printf("SKIP: the device has no bindless features\n");
        return 77;
    }
    driver::real_instance = vk.instance_proc;
    driver::real_device =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk.instance_proc(vk.instance, "vkGetDeviceProcAddr"));
    if (!driver::real_device) return 1;
    driver::real_create = reinterpret_cast<PFN_vkCreateSampler>(driver::real_device(vk.device, "vkCreateSampler"));
    driver::real_destroy = reinterpret_cast<PFN_vkDestroySampler>(driver::real_device(vk.device, "vkDestroySampler"));
    if (!driver::real_create || !driver::real_destroy) return 1;
    const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        vk.instance_proc(vk.instance, "vkGetPhysicalDeviceProperties"));
    if (!properties) return 1;
    VkPhysicalDeviceProperties limits{};
    properties(vk.physical, &limits);
    const std::uint32_t max_samplers = limits.limits.maxSamplerAllocationCount;
    if (max_samplers > 15000) {
        std::printf("SKIP: maxSamplerAllocationCount %u: more distinct S#s than one test should make\n", max_samplers);
        return 77;
    }
    // More distinct S#s than the device can hold samplers: MipLODBias steps of 1/256 from -16 up.
    const std::uint32_t writes = max_samplers + 300;

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.buffer_device_address = true;
        info.bindless = vk.bindless;
        info.bindless_partially_bound = vk.partially_bound;
        info.image_view_min_lod = vk.image_view_min_lod;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        if (FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
            return 1;
        HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) return 1;
        UINT64 fence_value = 0;
        const auto settle = [&] {  // a submission (whose acquire drains completed releases), finished
            queue->ExecuteCommandLists(0, nullptr);
            ++fence_value;
            if (FAILED(queue->Signal(fence.p, fence_value)) ||
                FAILED(fence->SetEventOnCompletion(fence_value, event)) ||
                WaitForSingleObject(event, 5000) != WAIT_OBJECT_0)
                return false;
            return true;
        };

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
        heap_desc.NumDescriptors = 2;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Ref<ID3D12DescriptorHeap> heap;
        if (FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p)))) return 1;
        const D3D12_CPU_DESCRIPTOR_HANDLE first = heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE kept = first;
        kept.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);

        // Slot 1 keeps one S# throughout; slot 0 sees a new one every write.
        const D3D12_SAMPLER_DESC kept_desc = sampler_desc(-31.0f);
        device->CreateSampler(&kept_desc, kept);
        VkSampler kept_sampler = VK_NULL_HANDLE;
        {
            std::lock_guard lock(driver::mutex);
            kept_sampler = driver::last_created;
        }
        bool settled = true;
        for (std::uint32_t i = 0; i < writes; ++i) {
            const D3D12_SAMPLER_DESC desc = sampler_desc(-16.0f + float(i) / 256.0f);
            device->CreateSampler(&desc, first);
            if (i % 64 == 63) settled = settle() && settled;
        }
        expect(settled, "every settling submission completed");
        VkSampler last = VK_NULL_HANDLE;
        std::size_t live = 0, peak = 0;
        bool last_live = false, kept_live = false;
        {
            std::lock_guard lock(driver::mutex);
            last = driver::last_created;
            live = driver::live.size();
            peak = driver::peak;
            last_live = driver::live.count(driver::raw(last)) != 0;
            kept_live = driver::live.count(driver::raw(kept_sampler)) != 0;
        }
        std::printf("%u S# writes into one slot: %zu samplers live, at most %zu, maxSamplerAllocationCount %u\n",
                    writes, live, peak, max_samplers);
        expect(last_live, "the last S# written has its sampler (the array did not fill)");
        expect(kept_live, "the S# another slot keeps was not evicted");
        expect(peak <= max_samplers, "live samplers stay within maxSamplerAllocationCount");
        expect(device->GetDeviceRemovedReason() == S_OK, "the device is not removed");
        CloseHandle(event);
    }

    if (failures) return 1;
    std::printf("d3d12_sampler_evict_test: OK\n");
    return 0;
}
