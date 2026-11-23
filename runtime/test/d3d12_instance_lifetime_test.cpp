// A D3D12 device released and another created keep the process's first VkInstance and its VkPhysicalDevice valid
// (#5144, RT-398). RivaTuner Statistics Server's injected RTSSHooks64.dll hooks vkQueuePresentKHR and calls
// vkGetPhysicalDeviceMemoryProperties with the physical device of the first instance the process made; when that
// instance was destroyed, the loader's trampoline rejects the handle ("Invalid physicalDevice") and ends the
// process. This test makes the same call through vulkan-1.dll's export, without RTSS. No game files.
#include "../shims/d3d12_com.h"

#include <atomic>
#include <cstdio>
#include <thread>

namespace {

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

struct Handles {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
};

bool make(Ref<ID3D12Device>& device, Handles& handles) {
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device.p)))) return false;
    const rt::d3d12::VulkanDeviceInfo* info = rt::d3d12::vulkan_device_info(device.p);
    if (!info) return false;
    handles = {info->instance, info->physical};
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Ref<ID3D12Device> first;
    Handles a;
    if (!make(first, a)) {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    // The loader's exported trampoline, which is what an injected hook calls: it validates the handle.
    const auto memory_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleA("vulkan-1.dll"), "vkGetPhysicalDeviceMemoryProperties")));
    if (!memory_properties) {
        std::puts("SKIP: vulkan-1.dll has no vkGetPhysicalDeviceMemoryProperties export");
        return 77;
    }
    VkPhysicalDeviceMemoryProperties before{};
    memory_properties(a.physical, &before);

    for (int round = 1; round <= 3; ++round) {
        first.reset();  // the device that made (or first used) the instance goes away
        Handles b;
        if (!make(first, b)) {
            std::puts("FAIL: a second standalone device could not be created");
            return 1;
        }
        const bool same = b.instance == a.instance && b.physical == a.physical;
        check(same, "a device created after another was released uses the process's first VkInstance and VkPhysicalDevice");
        if (!same) {
            // Calling the trampoline with a.physical now would end the process, as it does with RTSS.
            std::puts("FAIL: the first instance's VkPhysicalDevice is no longer valid (the loader would abort)");
            return 1;
        }
        VkPhysicalDeviceMemoryProperties after{};
        memory_properties(a.physical, &after);
        check(after.memoryTypeCount == before.memoryTypeCount && after.memoryHeapCount == before.memoryHeapCount,
              "the first VkPhysicalDevice still answers vkGetPhysicalDeviceMemoryProperties through the loader");
    }

    // Two devices alive at once, created on two threads, share the instance too.
    Ref<ID3D12Device> left, right;
    Handles l, r;
    std::atomic<bool> left_ok{false}, right_ok{false};
    std::thread t1([&] { left_ok = make(left, l); });
    std::thread t2([&] { right_ok = make(right, r); });
    t1.join();
    t2.join();
    check(left_ok && right_ok, "two devices created on two threads at once");
    check(l.instance == a.instance && r.instance == a.instance, "and both use the process's first VkInstance");
    left.reset();
    right.reset();
    first.reset();
    VkPhysicalDeviceMemoryProperties last{};
    memory_properties(a.physical, &last);
    check(last.memoryTypeCount == before.memoryTypeCount,
          "with every device released, the first VkPhysicalDevice is still valid");

    std::puts(failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
}
