// ID3D12CommandQueue::Signal claims the fence value only once its submission succeeded (B15). Before, the value was
// claimed first: when vkQueueSubmit then failed (a transient error, not device loss), the fence remembered a value
// the GPU would never signal. A later CPU ID3D12Fence::Signal above it was held until the GPU passed that value, that
// is forever, and a retry of the same queue Signal was refused as "not above".
//
// The D3D12 device runs on a real Vulkan device whose vkQueueSubmit fails once, with VK_ERROR_OUT_OF_HOST_MEMORY, when
// the test arms it (the failure is never forwarded to the driver):
//   1. a queue Signal(F, 5) whose submission fails returns an error, and the same Signal retried reaches 5;
//   2. after a failed queue Signal(F, 8), a CPU Signal(F, 9) is applied at once.
// Skips with exit code 77 without a Vulkan 1.3 device. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

template<class T> struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
    bool ready(DWORD ms) const { return handle && WaitForSingleObject(handle, ms) == WAIT_OBJECT_0; }
};

namespace driver {
PFN_vkGetInstanceProcAddr real_instance = nullptr;
PFN_vkGetDeviceProcAddr real_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
std::atomic<bool> fail_next_submit{false};
std::atomic<unsigned> failed_submits{0};
DWORD test_thread = 0;

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    // Only the armed call on the test thread fails, so the fence dispatcher and other threads are unaffected.
    if (GetCurrentThreadId() == test_thread && fail_next_submit.exchange(false)) {
        failed_submits.fetch_add(1);
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::strcmp(name, "vkQueueSubmit") == 0) return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
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

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    driver::real_instance = vk.instance_proc;
    driver::real_device =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(vk.instance_proc(vk.instance, "vkGetDeviceProcAddr"));
    if (!driver::real_device) return 1;
    driver::real_submit = reinterpret_cast<PFN_vkQueueSubmit>(driver::real_device(vk.device, "vkQueueSubmit"));
    if (!driver::real_submit) return 1;
    driver::test_thread = GetCurrentThreadId();

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = driver::get_instance;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p)))) return 1;
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))) ||
            FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
            return 1;
        Event event;
        if (!event.handle) return 1;

        // 1. A failed submission claims nothing: the same Signal retried goes through.
        driver::fail_next_submit = true;
        expect(FAILED(queue->Signal(fence.p, 5)), "1 the Signal whose submission failed returns an error");
        expect(driver::failed_submits == 1, "1 the armed vkQueueSubmit failed");
        expect(SUCCEEDED(queue->Signal(fence.p, 5)), "1 the retried Signal(5) succeeds");
        expect(SUCCEEDED(fence->SetEventOnCompletion(5, event.handle)) && event.ready(5000),
               "1 the fence reaches 5 after the retry");
        expect(fence->GetCompletedValue() == 5, "1 GetCompletedValue reads 5");
        std::printf("1: a queue Signal whose submit failed claimed nothing; the retry reached 5\n");

        // 2. A CPU Signal after a failed queue Signal is not held behind a value the GPU never signals.
        driver::fail_next_submit = true;
        expect(FAILED(queue->Signal(fence.p, 8)), "2 the Signal whose submission failed returns an error");
        expect(driver::failed_submits == 2, "2 the armed vkQueueSubmit failed");
        expect(SUCCEEDED(fence->Signal(9)), "2 the CPU Signal(9) succeeds");
        expect(SUCCEEDED(fence->SetEventOnCompletion(9, event.handle)) && event.ready(5000),
               "2 the fence reaches 9 at once (not held behind 8)");
        expect(fence->GetCompletedValue() == 9, "2 GetCompletedValue reads 9");
        std::printf("2: a CPU Signal after a failed queue Signal was applied at once\n");

        // The queue still works.
        expect(SUCCEEDED(queue->Signal(fence.p, 10)), "a later queue Signal succeeds");
        expect(SUCCEEDED(fence->SetEventOnCompletion(10, event.handle)) && event.ready(5000), "the fence reaches 10");
        expect(device->GetDeviceRemovedReason() == S_OK, "the device is not removed");
    }

    if (failures) return 1;
    std::printf("d3d12_queue_signal_claim_test: OK\n");
    return 0;
}
