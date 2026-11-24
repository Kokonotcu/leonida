// An in-place (add-only) upload of the bindless hash table on one queue is waited for by the next submission of every
// other queue (B9). Before, only a flush that started a new table generation (BindlessSet::last_flush_replaced) made
// other queues wait: a queue B submitting right after queue A's add-only flush could probe the table before A's
// vkCmdUpdateBuffer landed, and a descriptor created before B's ExecuteCommandLists read the fallback texture or
// sampler. BindlessSet::last_flush_uploaded reports any table upload, and the fallback images' first initialization.
//
// A real BindlessSet and two VulkanQueues share one VulkanContext whose bindless prepare hook flushes the set as the
// D3D12 device does (d3d12_vulkan_device.cpp, set_bindless_prepare). vkQueueSubmit is intercepted to count the
// semaphores each submission waits on:
//   1. the first flush (fallback initialization and the first table upload) on A: B's next submission waits on A;
//   2. an added key, uploaded in place on A (no new generation): B's next submission waits on A;
//   3. a flush with nothing to upload on A: B does not wait;
//   4. only a descriptor written (an image's view changed, the table unchanged): B does not wait.
// Skips with exit code 77 without a Vulkan 1.3 device with the bindless features. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"
#include "../src/vulkan_bindless.h"

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string_view>
#include <utility>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {

PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
std::mutex mutex;
std::uint32_t last_wait_count = 0;
bool fail_next = false;  // the next submission fails with VK_ERROR_OUT_OF_HOST_MEMORY, not reaching the driver

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    {
        std::lock_guard lock(mutex);
        if (std::exchange(fail_next, false)) return VK_ERROR_OUT_OF_HOST_MEMORY;
        last_wait_count = 0;
        for (std::uint32_t i = 0; i < count; ++i) last_wait_count += submits[i].waitSemaphoreCount;
    }
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    return real_get_device(device, name);
}

std::uint32_t waits() {
    std::lock_guard lock(mutex);
    return last_wait_count;
}

}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false) || !d.bindless) {
        std::fprintf(stderr, "vk_bindless_add_wait_test: no Vulkan 1.3 device with the bindless features, skipped\n");
        return 77;
    }
    driver::real_get_device = d.device_proc;
    driver::real_submit = reinterpret_cast<PFN_vkQueueSubmit>(d.device_proc(d.device, "vkQueueSubmit"));
    EXPECT(driver::real_submit);

    // Set 0 sized for this device, as the D3D12 device does.
    const auto properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties2"));
    EXPECT(properties2);
    VkPhysicalDeviceVulkan12Properties v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, &v12};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &v11};
    properties2(d.physical, &properties);
    BindlessFeatures features{};
    bindless::size_bindings(features, properties.properties.limits, v11, v12);

    {
        BindlessSet set(d.device_proc, d.device, d.memory_properties(), features);
        EXPECT(set.ok());
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        bool uploaded = false;  // what the last prepare reported
        context.set_bindless_prepare([&](VkCommandBuffer command_buffer) {
            if (!set.fallbacks_ready()) set.record_fallback_init(command_buffer);
            const bool flushed = set.flush(command_buffer, context.submitted_retire_ticket());
            uploaded = set.last_flush_uploaded();
            if (uploaded) context.note_bindless_upload();
            return flushed;
        });
        context.set_bindless_commit([&](bool submitted) { set.table_upload_result(submitted); });

        CommandList empty;
        EXPECT(empty.close());
        const CommandList* lists[] = {&empty};
        VulkanQueue a(context, d.queue, d.family);
        VulkanQueue b(context, d.queue, d.family);
        EXPECT(a.ok() && b.ok());

        // 0. B17: the submission that carries the fallback images' first initialization fails: they are not ready, and
        //    the next submission records the initialization again.
        {
            std::lock_guard lock(driver::mutex);
            driver::fail_next = true;
        }
        EXPECT(a.execute(lists) != ReplayResult::success);
        EXPECT(!set.fallbacks_ready());
        std::printf("0 a failed first submission leaves the fallback images to initialize again\n");

        // 1. The first flush: the fallback images' initialization and the first upload of the table.
        EXPECT(a.execute(lists) == ReplayResult::success);
        EXPECT(set.fallbacks_ready());
        EXPECT(uploaded);
        EXPECT(b.execute(lists) == ReplayResult::success);
        EXPECT(driver::waits() == 1);
        std::printf("1 first flush on A: B waits for it\n");

        // 2. A key added in place: no new generation, but B's shaders probe the same table.
        DescriptorKey key{};
        key[0] = 0x1234;
        key[7] = 0x5678;
        const std::uint32_t index = set.add_image(ImageDimension::dim_2d, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(index != 0);
        EXPECT(set.insert_key(key, index));
        EXPECT(!set.flush_replaces_table());
        EXPECT(a.execute(lists) == ReplayResult::success);
        EXPECT(uploaded && !set.last_flush_replaced());
        EXPECT(b.execute(lists) == ReplayResult::success);
        EXPECT(driver::waits() == 1);
        // B has seen it: its next submission does not wait again.
        EXPECT(b.execute(lists) == ReplayResult::success);
        EXPECT(driver::waits() == 0);
        std::printf("2 add-only upload on A: B waits for it, once\n");

        // 3. Nothing to upload.
        EXPECT(a.execute(lists) == ReplayResult::success);
        EXPECT(!uploaded);
        EXPECT(b.execute(lists) == ReplayResult::success);
        EXPECT(driver::waits() == 0);
        std::printf("3 nothing uploaded on A: B does not wait\n");

        // 4. A descriptor write only (vkUpdateDescriptorSets on the CPU): the table is unchanged.
        set.update_image(ImageDimension::dim_2d, index, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(a.execute(lists) == ReplayResult::success);
        EXPECT(!uploaded);
        EXPECT(b.execute(lists) == ReplayResult::success);
        EXPECT(driver::waits() == 0);
        std::printf("4 a descriptor write only on A: B does not wait\n");

        EXPECT(a.idle() && b.idle());
    }

    const unsigned errors = d.validation_errors();
    if (errors) {
        std::fprintf(stderr, "vk_bindless_add_wait_test: %u validation errors\n", errors);
        return 1;
    }
    std::printf("vk_bindless_add_wait_test: PASS\n");
    return 0;
}
