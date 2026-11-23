// VulkanContext::wait_retire_ticket() progress-semaphore lifetime, deduplication, and device-loss tests
// (runtime/src/vulkan_backend.h):
//   1. Destroying a VulkanQueue (~VulkanQueue) while another thread is inside wait_retire_ticket() must
//      defer vkDestroySemaphore(progress_) until the active waiter exits, then destroy it cleanly outside
//      submit_mutex_ without leaking or calling vkDestroySemaphore twice.
//   2. Multiple pending tickets for the same queue must be deduplicated into one vkWaitSemaphores entry per
//      semaphore with the maximum required value.
//   3. wait_retire_ticket() must leave device_lost false on transient VK_ERROR_OUT_OF_HOST_MEMORY and mark
//      the context lost (returning 0) on VK_ERROR_DEVICE_LOST.
//   4. wait_retire_ticket() must use a bounded vkWaitSemaphores timeout (never UINT64_MAX) and abort
//      immediately if another thread marks the device lost.
//   5. VulkanQueue::signal() that is not held keeps its pin (the fence) until the submission has completed.
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

struct ScopeExit {
    std::function<void()> fn;
    ~ScopeExit() {
        if (fn) fn();
    }
};

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkWaitSemaphores real_wait_semaphores = nullptr;
PFN_vkDestroySemaphore real_destroy_semaphore = nullptr;

std::mutex mu;
std::condition_variable cv;
std::unordered_set<VkSemaphore> destroyed;

// Case 1 teardown barrier state:
bool arm_pause = false;
bool hook_active = false;
bool entered_wait = false;
bool teardown_done = false;
bool destroyed_while_waiting = false;
VkSemaphore awaited_sem = VK_NULL_HANDLE;

// Inspection / injection state for cases 2-4:
std::uint32_t wait_calls = 0;
std::uint32_t last_sem_count = 0;
std::uint64_t last_first_val = 0;
std::uint64_t last_timeout = 0;
std::function<VkResult(const VkSemaphoreWaitInfo*, std::uint64_t)> wait_override;

VKAPI_ATTR VkResult VKAPI_CALL wait_semaphores(VkDevice device, const VkSemaphoreWaitInfo* info,
                                                std::uint64_t timeout) {
    bool paused = false;
    bool destroyed_during_pause = false;
    std::function<VkResult(const VkSemaphoreWaitInfo*, std::uint64_t)> override_fn;
    {
        std::unique_lock lock(mu);
        ++wait_calls;
        last_timeout = timeout;
        if (info) {
            last_sem_count = info->semaphoreCount;
            last_first_val = (info->semaphoreCount > 0 && info->pValues) ? info->pValues[0] : 0;
        } else {
            last_sem_count = 0;
            last_first_val = 0;
        }
        if (arm_pause && info && info->semaphoreCount > 0 && info->pSemaphores) {
            arm_pause = false;
            paused = true;
            hook_active = true;
            awaited_sem = info->pSemaphores[0];
            entered_wait = true;
            cv.notify_all();
            cv.wait_for(lock, std::chrono::seconds(5), [] { return teardown_done; });
            destroyed_during_pause = destroyed_while_waiting;
            hook_active = false;
        } else if (wait_override) {
            override_fn = std::move(wait_override);
            wait_override = {};
        }
    }
    if (paused && destroyed_during_pause) {
        // Unmodified origin/main destroyed the awaited semaphore concurrently; avoid UB in real driver.
        return VK_ERROR_UNKNOWN;
    }
    if (override_fn) return override_fn(info, timeout);
    return real_wait_semaphores(device, info, timeout);
}

VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice device, VkSemaphore semaphore,
                                             const VkAllocationCallbacks* allocator) {
    {
        std::lock_guard lock(mu);
        destroyed.insert(semaphore);
        if (hook_active && semaphore != VK_NULL_HANDLE && semaphore == awaited_sem) {
            destroyed_while_waiting = true;
        }
    }
    real_destroy_semaphore(device, semaphore, allocator);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n(name);
    if (n == "vkWaitSemaphores") return reinterpret_cast<PFN_vkVoidFunction>(wait_semaphores);
    if (n == "vkDestroySemaphore") return reinterpret_cast<PFN_vkVoidFunction>(destroy_semaphore);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr proc, VkDevice device) {
    real_get_device = proc;
    real_wait_semaphores = reinterpret_cast<PFN_vkWaitSemaphores>(proc(device, "vkWaitSemaphores"));
    real_destroy_semaphore = reinterpret_cast<PFN_vkDestroySemaphore>(proc(device, "vkDestroySemaphore"));
}

void reset_case() {
    std::lock_guard lock(mu);
    destroyed.clear();
    arm_pause = false;
    hook_active = false;
    entered_wait = false;
    teardown_done = false;
    destroyed_while_waiting = false;
    awaited_sem = VK_NULL_HANDLE;
    wait_calls = 0;
    last_sem_count = 0;
    last_first_val = 0;
    last_timeout = 0;
    wait_override = {};
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    vktest::Device d;
    if (!d.create(false)) {
        std::printf("SKIP: no suitable Vulkan 1.3 device\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    EXPECT(driver::real_wait_semaphores != nullptr && driver::real_destroy_semaphore != nullptr);

    CommandList empty;
    EXPECT(empty.close());
    const CommandList* p_empty = &empty;

    // 1. Concurrent ~VulkanQueue() while wait_retire_ticket() is inside vkWaitSemaphores must defer
    //    semaphore destruction until wait_retire_ticket() releases it outside submit_mutex_.
    {
        driver::reset_case();
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        auto q = std::make_unique<VulkanQueue>(context, d.queue, d.family);
        EXPECT(q->ok());
        EXPECT(q->execute({&p_empty, 1}) == ReplayResult::success);
        const std::uint64_t ticket = context.submitted_retire_ticket();
        EXPECT(ticket == 1);

        {
            std::lock_guard lock(driver::mu);
            driver::arm_pause = true;
        }

        std::uint64_t returned_ticket = 0;
        std::thread waiter([&] { returned_ticket = context.wait_retire_ticket(ticket); });
        ScopeExit cleanup{[&] {
            {
                std::lock_guard lock(driver::mu);
                driver::teardown_done = true;
            }
            driver::cv.notify_all();
            if (waiter.joinable()) waiter.join();
        }};

        {
            std::unique_lock lock(driver::mu);
            EXPECT(driver::cv.wait_for(lock, std::chrono::seconds(5), [] { return driver::entered_wait; }));
        }
        const VkSemaphore sem = [] {
            std::lock_guard lock(driver::mu);
            return driver::awaited_sem;
        }();
        EXPECT(sem != VK_NULL_HANDLE);

        q.reset();  // ~VulkanQueue() while waiter is inside vkWaitSemaphores

        bool destroyed_before_wait_exit = false;
        {
            std::lock_guard lock(driver::mu);
            destroyed_before_wait_exit = driver::destroyed.count(sem) != 0;
            driver::teardown_done = true;
        }
        driver::cv.notify_all();
        waiter.join();
        cleanup.fn = {};

        EXPECT(!destroyed_before_wait_exit);
        {
            std::lock_guard lock(driver::mu);
            EXPECT(driver::destroyed.count(sem) == 1);
        }
        EXPECT(returned_ticket >= ticket);
        std::printf("case 1 (concurrent ~VulkanQueue during wait_retire_ticket defers destroy cleanly): ok\n");
    }

    // 2. Deduplication of repeated per-queue semaphores across multiple pending tickets.
    {
        driver::reset_case();
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue q(context, d.queue, d.family);
        EXPECT(q.ok());
        VulkanFence gate(context, 0);
        EXPECT(gate.semaphore() != VK_NULL_HANDLE);

        ScopeExit gate_cleanup{[&] {
            gate.signal(1);
            q.idle();
        }};
        context.holds().signaled(gate.semaphore(), 1);
        q.wait(gate.semaphore(), 1);
        for (int i = 0; i < 5; ++i) EXPECT(q.execute({&p_empty, 1}) == ReplayResult::success);
        const std::uint64_t ticket = context.submitted_retire_ticket();
        EXPECT(ticket == 5);

        {
            std::lock_guard lock(driver::mu);
            // Signal gate right inside override then delegate to real vkWaitSemaphores.
            driver::wait_override = [&](const VkSemaphoreWaitInfo* info, std::uint64_t timeout) -> VkResult {
                gate.signal(1);
                return driver::real_wait_semaphores(d.device, info, timeout);
            };
        }

        const std::uint64_t done = context.wait_retire_ticket(ticket);
        gate_cleanup.fn = {};
        EXPECT(q.idle());
        EXPECT(done >= ticket);
        {
            std::lock_guard lock(driver::mu);
            EXPECT(driver::last_sem_count == 1);
            EXPECT(driver::last_first_val == 5);
        }
        std::printf("case 2 (deduplicated 5 tickets into 1 semaphore wait val=5): ok\n");
    }

    // 3. VK_ERROR_OUT_OF_HOST_MEMORY stays nonfatal; VK_ERROR_DEVICE_LOST propagates mark_device_lost and returns 0.
    {
        driver::reset_case();
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue q(context, d.queue, d.family);
        EXPECT(q.ok());

        EXPECT(q.execute({&p_empty, 1}) == ReplayResult::success);
        EXPECT(q.idle());
        const std::uint64_t t1 = context.submitted_retire_ticket();
        {
            std::lock_guard lock(driver::mu);
            driver::wait_override = [](const VkSemaphoreWaitInfo*, std::uint64_t) {
                return VK_ERROR_OUT_OF_HOST_MEMORY;
            };
        }
        const std::uint64_t done1 = context.wait_retire_ticket(t1);
        EXPECT(!context.is_device_lost());
        EXPECT(done1 >= t1);

        EXPECT(q.execute({&p_empty, 1}) == ReplayResult::success);
        EXPECT(q.idle());
        const std::uint64_t t2 = context.submitted_retire_ticket();
        {
            std::lock_guard lock(driver::mu);
            driver::wait_override = [](const VkSemaphoreWaitInfo*, std::uint64_t) { return VK_ERROR_DEVICE_LOST; };
        }
        const std::uint64_t done2 = context.wait_retire_ticket(t2);
        EXPECT(done2 == 0);
        EXPECT(context.is_device_lost());
        std::printf("case 3 (OOM nonfatal vs VK_ERROR_DEVICE_LOST marks context lost): ok\n");
    }

    // 4. Bounded wait slice wakes on concurrent device loss without passing UINT64_MAX.
    {
        driver::reset_case();
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue q(context, d.queue, d.family);
        EXPECT(q.ok());
        EXPECT(q.execute({&p_empty, 1}) == ReplayResult::success);
        const std::uint64_t ticket = context.submitted_retire_ticket();

        {
            std::lock_guard lock(driver::mu);
            driver::wait_calls = 0;
            driver::wait_override = [&context](const VkSemaphoreWaitInfo*, std::uint64_t) {
                context.mark_device_lost();
                return VK_TIMEOUT;
            };
        }
        const std::uint64_t done = context.wait_retire_ticket(ticket);
        EXPECT(done == 0);
        {
            std::lock_guard lock(driver::mu);
            EXPECT(driver::last_timeout > 0 && driver::last_timeout < UINT64_MAX);
            EXPECT(driver::wait_calls == 1);
        }
        EXPECT(q.idle());
        std::printf("case 4 (bounded wait slice wakes on concurrent device loss): ok\n");
    }

    // 5. An unheld queue Signal keeps its fence pinned until the submission that signals it has completed (B11): an
    //    application may release the fence right after ID3D12CommandQueue::Signal, and destroying a semaphore with a
    //    pending signal is undefined (VUID-vkDestroySemaphore-semaphore-05149). The submission waits on a gate the
    //    CPU opens later, so the signal is still pending when the call returns.
    {
        driver::reset_case();
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        VulkanQueue q(context, d.queue, d.family);
        EXPECT(q.ok());
        VulkanFence gate(context, 0), fence(context, 0);
        EXPECT(gate.semaphore() != VK_NULL_HANDLE && fence.semaphore() != VK_NULL_HANDLE);
        ScopeExit gate_cleanup{[&] {
            gate.signal(1);
            q.idle();
        }};
        context.holds().signaled(gate.semaphore(), 1);  // counts as submitted: the Signal below is not held
        q.wait(gate.semaphore(), 1);
        auto object = std::make_shared<int>(0);  // stands for the ID3D12Fence the pin holds a reference to
        const std::weak_ptr<int> watch = object;
        bool pinned = false;
        EXPECT(q.signal(fence.semaphore(), 1, [&]() -> std::shared_ptr<void> {
            pinned = true;
            return object;
        }));
        object.reset();  // the application's own reference
        EXPECT(pinned && !watch.expired());
        EXPECT(fence.completed() == 0);
        gate.signal(1);
        EXPECT(fence.wait(1, 5'000'000'000ull));
        gate_cleanup.fn = {};
        EXPECT(q.execute({&p_empty, 1}) == ReplayResult::success);  // drains the pins whose submission completed
        EXPECT(q.idle());
        EXPECT(q.execute({&p_empty, 1}) == ReplayResult::success);
        EXPECT(watch.expired());
        std::printf("case 5 (unheld Signal keeps its fence until the GPU signal completes): ok\n");
    }

    const unsigned val_errors = d.validation_errors();
    std::printf("Vulkan validation errors: %u\n", val_errors);
    if (val_errors != 0) return 1;
    std::printf("vk_retire_wait_lifetime_test: ok\n");
    return 0;
}
