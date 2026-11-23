// VulkanQueue::~VulkanQueue() must clear VulkanContext::bindless_upload_semaphore_ before destroying progress_.
// Otherwise, after a queue that published a bindless table upload is destroyed, another queue's next submission
// puts the destroyed VkSemaphore handle into vkQueueSubmit::pWaitSemaphores (use-after-free / VUID violation).
//
// Intercepts vkCreateSemaphore, vkDestroySemaphore, and vkQueueSubmit on top of a real Vulkan 1.3 device to
// deterministically verify:
//   1. Queue A publishes a bindless upload and stays alive: Queue B's next submission waits on Queue A's live
//      progress semaphore, and a subsequent submission on Queue B does not re-wait on the same version.
//   2. Queue A publishes a new bindless upload and is destroyed: Queue B's (and a brand-new Queue C's) next
//      submission submits cleanly with zero destroyed semaphores in pWaitSemaphores.
//   3. Control: Queue A publishes storage_init_semaphore_ (via zero_on_next_submit) and is destroyed; Queue B
//      submits with zero destroyed semaphores.
//
// Skips with exit code 77 when no Vulkan 1.3 device is available. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string_view>
#include <unordered_set>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {

PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCreateSemaphore real_create_sem = nullptr;
PFN_vkDestroySemaphore real_destroy_sem = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;

std::mutex sem_mutex;
std::unordered_set<std::uint64_t> live_semaphores;
std::unordered_set<std::uint64_t> destroyed_semaphores;
std::uint32_t last_wait_count = 0;
std::uint64_t last_wait_sem = 0;
std::uint32_t destroyed_wait_hits = 0;

std::uint64_t raw_handle(VkSemaphore s) {
    return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(s));
}

VKAPI_ATTR VkResult VKAPI_CALL create_semaphore(VkDevice device, const VkSemaphoreCreateInfo* info,
                                                const VkAllocationCallbacks* alloc, VkSemaphore* out) {
    const VkResult res = real_create_sem(device, info, alloc, out);
    if (res == VK_SUCCESS && out && *out) {
        std::lock_guard lock(sem_mutex);
        const std::uint64_t h = raw_handle(*out);
        live_semaphores.insert(h);
        destroyed_semaphores.erase(h);
    }
    return res;
}

VKAPI_ATTR void VKAPI_CALL destroy_semaphore(VkDevice device, VkSemaphore sem, const VkAllocationCallbacks* alloc) {
    if (sem) {
        std::lock_guard lock(sem_mutex);
        const std::uint64_t h = raw_handle(sem);
        live_semaphores.erase(h);
        destroyed_semaphores.insert(h);
    }
    real_destroy_sem(device, sem, alloc);
}

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    bool bad = false;
    {
        std::lock_guard lock(sem_mutex);
        last_wait_count = 0;
        last_wait_sem = 0;
        for (std::uint32_t i = 0; i < count; ++i) {
            last_wait_count += submits[i].waitSemaphoreCount;
            for (std::uint32_t w = 0; w < submits[i].waitSemaphoreCount; ++w) {
                const std::uint64_t h = raw_handle(submits[i].pWaitSemaphores[w]);
                if (!last_wait_sem) last_wait_sem = h;
                if (destroyed_semaphores.count(h) || !live_semaphores.count(h)) {
                    ++destroyed_wait_hits;
                    bad = true;
                    std::fprintf(stderr, "FAIL: vkQueueSubmit waited on destroyed VkSemaphore 0x%llx\n",
                                 static_cast<unsigned long long>(h));
                }
            }
        }
    }
    // Never forward a known-destroyed semaphore handle into the real Vulkan driver.
    if (bad) return VK_ERROR_DEVICE_LOST;
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkCreateSemaphore") return reinterpret_cast<PFN_vkVoidFunction>(create_semaphore);
    if (n == "vkDestroySemaphore") return reinterpret_cast<PFN_vkVoidFunction>(destroy_semaphore);
    if (n == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    return real_get_device(device, name);
}

void install(PFN_vkGetDeviceProcAddr device_proc, VkDevice device) {
    real_get_device = device_proc;
    real_create_sem = reinterpret_cast<PFN_vkCreateSemaphore>(device_proc(device, "vkCreateSemaphore"));
    real_destroy_sem = reinterpret_cast<PFN_vkDestroySemaphore>(device_proc(device, "vkDestroySemaphore"));
    real_submit = reinterpret_cast<PFN_vkQueueSubmit>(device_proc(device, "vkQueueSubmit"));
    std::lock_guard lock(sem_mutex);
    live_semaphores.clear();
    destroyed_semaphores.clear();
    last_wait_count = 0;
    last_wait_sem = 0;
    destroyed_wait_hits = 0;
}

}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_bindless_upload_semaphore_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::install(d.device_proc, d.device);
    EXPECT(driver::real_create_sem && driver::real_destroy_sem && driver::real_submit);

    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    const VkBuffer src_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    const VkBuffer dst_buf = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_ptr);
    EXPECT(src_buf && dst_buf);

    std::atomic<bool> emit_bindless_upload{false};

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());

        ResourceBinding src_rb{};
        src_rb.buffer = src_buf;
        src_rb.size = 64;
        ResourceBinding dst_rb{};
        dst_rb.buffer = dst_buf;
        dst_rb.size = 64;
        context.add_resource(1, src_rb);
        context.add_resource(2, dst_rb);

        context.set_bindless_prepare([&](VkCommandBuffer) {
            if (emit_bindless_upload.exchange(false)) context.note_bindless_upload();
            return true;
        });

        CommandList copy_list;
        EXPECT(copy_list.copy_buffer(CopyBuffer{2, 0, 1, 0, 64}));
        EXPECT(copy_list.close());
        const CommandList* lists[] = {&copy_list};

        auto queue_a = std::make_unique<VulkanQueue>(context, d.queue, d.family);
        auto queue_b = std::make_unique<VulkanQueue>(context, d.queue, d.family);
        EXPECT(queue_a->ok() && queue_b->ok());

        // 1. Live cross-queue sync: Queue A publishes a bindless upload and stays alive; Queue B must wait on it once.
        emit_bindless_upload.store(true);
        EXPECT(queue_a->execute(lists) == ReplayResult::success);
        EXPECT(queue_b->execute(lists) == ReplayResult::success);
        {
            std::lock_guard lock(driver::sem_mutex);
            EXPECT(driver::destroyed_wait_hits == 0);
            EXPECT(driver::last_wait_count == 1);
            EXPECT(driver::live_semaphores.count(driver::last_wait_sem) == 1);
        }
        // Second submit on Queue B sees the same version and does not re-wait.
        EXPECT(queue_b->execute(lists) == ReplayResult::success);
        {
            std::lock_guard lock(driver::sem_mutex);
            EXPECT(driver::destroyed_wait_hits == 0);
            EXPECT(driver::last_wait_count == 0);
        }

        // 2. Regression: Queue A publishes a new bindless upload, then Queue A is destroyed before Queue B executes.
        emit_bindless_upload.store(true);
        EXPECT(queue_a->execute(lists) == ReplayResult::success);
        queue_a.reset();  // destroys Queue A's progress_ semaphore after vkQueueWaitIdle

        const ReplayResult res_b = queue_b->execute(lists);
        {
            std::lock_guard lock(driver::sem_mutex);
            EXPECT(driver::destroyed_wait_hits == 0);
            EXPECT(driver::last_wait_count == 0);
        }
        EXPECT(res_b == ReplayResult::success);

        // Brand-new Queue C must also submit cleanly with no wait on the destroyed semaphore.
        auto queue_c = std::make_unique<VulkanQueue>(context, d.queue, d.family);
        EXPECT(queue_c->ok());
        EXPECT(queue_c->execute(lists) == ReplayResult::success);
        {
            std::lock_guard lock(driver::sem_mutex);
            EXPECT(driver::destroyed_wait_hits == 0);
            EXPECT(driver::last_wait_count == 0);
        }

        // 3. Control: storage_init_semaphore_ published by Queue C, then Queue C destroyed, then Queue B executes.
        context.zero_on_next_submit(2, dst_buf);
        EXPECT(queue_c->execute(lists) == ReplayResult::success);
        queue_c.reset();
        EXPECT(queue_b->execute(lists) == ReplayResult::success);
        {
            std::lock_guard lock(driver::sem_mutex);
            EXPECT(driver::destroyed_wait_hits == 0);
            EXPECT(driver::last_wait_count == 0);
        }

        EXPECT(queue_b->idle());
    }

    std::printf("vk_bindless_upload_semaphore_test: PASS\n");
    return 0;
}
