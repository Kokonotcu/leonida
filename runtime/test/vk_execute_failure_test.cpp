// A failed ExecuteCommandLists names its cause (I4): VulkanQueue::last_failure gives the list and command prepare
// refused, that command's kind (command_name), the ReplayResult, and the VkResult when vkQueueSubmit was what
// failed; VulkanQueue::describe words it for the log line the D3D12 queue prints once per cause. Before, the log
// line held only the ReplayResult's name, which no dropped frame could be traced back to.
//
// A real Vulkan device; vkQueueSubmit can be told to fail once with VK_ERROR_OUT_OF_HOST_MEMORY (not forwarded):
//   1. the second of two lists copies past the end of a buffer at its second command: list 1, command 1,
//      CopyBufferRegion, invalid_copy, no VkResult;
//   2. a list that is not closed: list 0 as a whole, list_open;
//   3. a submission the driver refuses: no single command, unsupported, vkQueueSubmit returned -1;
//   4. a later success clears it.
// Skips with exit code 77 without a Vulkan 1.3 device. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
std::atomic<bool> fail_next{false};

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    if (fail_next.exchange(false)) return VK_ERROR_OUT_OF_HOST_MEMORY;
    return real_submit(queue, count, submits, fence);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkQueueSubmit") return reinterpret_cast<PFN_vkVoidFunction>(queue_submit);
    return real_get_device(device, name);
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_execute_failure_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::real_get_device = d.device_proc;
    driver::real_submit = d.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    EXPECT(driver::real_submit);
    void* src_ptr = nullptr;
    void* dst_ptr = nullptr;
    const VkBuffer src = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &src_ptr);
    const VkBuffer dst = d.buffer(64, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &dst_ptr);
    EXPECT(src && dst);

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        ResourceBinding a{}, b{};
        a.buffer = src;
        a.size = 64;
        b.buffer = dst;
        b.size = 64;
        context.add_resource(1, a);
        context.add_resource(2, b);
        VulkanQueue queue(context, d.queue, d.family);
        EXPECT(queue.ok());

        CommandList good, bad, open;
        EXPECT(good.copy_buffer({2, 0, 1, 0, 16}) && good.close());
        EXPECT(bad.uav_barrier({0}) && bad.copy_buffer({2, 60, 1, 0, 16}) && bad.close());  // 60 + 16 > 64
        EXPECT(open.copy_buffer({2, 0, 1, 0, 16}));                                         // never closed

        // 1. A command prepare refuses.
        const CommandList* two[] = {&good, &bad};
        const ReplayResult refused = queue.execute(two);
        EXPECT(refused != ReplayResult::success);
        VulkanQueue::Failure f = queue.last_failure();
        EXPECT(f.result == refused && f.list == 1 && f.command == 1);
        EXPECT(f.kind && std::string_view(f.kind) == "CopyBufferRegion" && f.vk_result == VK_SUCCESS);
        std::string text = VulkanQueue::describe(f);
        std::printf("1 %s (result %d)\n", text.c_str(), int(f.result));
        EXPECT(text == "list 1, command 1 (CopyBufferRegion)");

        // 2. A list that is not closed: the list as a whole.
        const CommandList* one[] = {&open};
        EXPECT(queue.execute(one) == ReplayResult::list_open);
        f = queue.last_failure();
        EXPECT(f.result == ReplayResult::list_open && f.list == 0 && f.command == VulkanQueue::Failure::none);
        EXPECT(VulkanQueue::describe(f) == "list 0 as a whole");

        // 3. The driver refuses the submission.
        const CommandList* fine[] = {&good};
        driver::fail_next = true;
        EXPECT(queue.execute(fine) != ReplayResult::success);
        f = queue.last_failure();
        text = VulkanQueue::describe(f);
        std::printf("3 %s (result %d)\n", text.c_str(), int(f.result));
        EXPECT(f.vk_result == VK_ERROR_OUT_OF_HOST_MEMORY && f.command == VulkanQueue::Failure::none);
        EXPECT(text == "no single command, vkQueueSubmit returned -1");

        // 4. Success clears it.
        EXPECT(queue.execute(fine) == ReplayResult::success);
        EXPECT(queue.last_failure().result == ReplayResult::success);
        EXPECT(queue.idle());
    }

    const unsigned errors = d.validation_errors();
    if (errors) {
        std::fprintf(stderr, "vk_execute_failure_test: %u validation errors\n", errors);
        return 1;
    }
    std::printf("vk_execute_failure_test: PASS\n");
    return 0;
}
