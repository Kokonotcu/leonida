// Creating an unordered-access view does not make other queues wait when the texture is initialized already (D7).
// VulkanContext::note_shader_image (every CreateUnorderedAccessView on a texture) queued the image for the next
// shader-running submission, and that submission advanced the storage-initialization version whenever it had any
// queued image, even when initialize() found nothing UNDEFINED: every other queue's next submission then waited for
// it. A title that writes UAV descriptors into ring heaps every frame serialized async compute and copy behind
// graphics. Now an image the tracker already has in defined layouts is not queued, and the version advances only
// when a submission initialized something.
//
// A real Vulkan device, two VulkanQueues on one VulkanContext, a do-nothing compute pipeline; vkQueueSubmit is
// intercepted to count the semaphores each submission waits on:
//   1. the first UAV of a never-used texture: queue A's dispatch initializes it and B's next submission waits on A;
//   2. a second UAV of the same, now initialized, texture: A's dispatch initializes nothing and B does not wait;
//   3. a UAV noted while the texture was UNDEFINED, which a clear on B (no shaders) then initializes: A's dispatch
//      finds nothing to initialize and B does not wait.
// Skips with exit code 77 without a Vulkan 1.3 device. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string_view>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
std::mutex mutex;
std::uint32_t last_wait_count = 0;

VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits,
                                            VkFence fence) {
    {
        std::lock_guard lock(mutex);
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

// glslang-equivalent of `layout(local_size_x = 1) in; void main() {}`, assembled by hand.
constexpr std::uint32_t cs_empty_spirv[] = {
    0x07230203u, 0x00010000u, 0x00000000u, 0x00000005u, 0x00000000u,
    0x00020011u, 0x00000001u,                                         // OpCapability Shader
    0x0003000Eu, 0x00000000u, 0x00000001u,                            // OpMemoryModel Logical GLSL450
    0x0005000Fu, 0x00000005u, 0x00000001u, 0x6E69616Du, 0x00000000u,  // OpEntryPoint GLCompute %1 "main"
    0x00060010u, 0x00000001u, 0x00000011u, 0x00000001u, 0x00000001u, 0x00000001u,  // LocalSize 1 1 1
    0x00020013u, 0x00000002u,                                         // %2 = OpTypeVoid
    0x00030021u, 0x00000003u, 0x00000002u,                            // %3 = OpTypeFunction %2
    0x00050036u, 0x00000002u, 0x00000001u, 0x00000000u, 0x00000003u,  // %1 = OpFunction %2 None %3
    0x000200F8u, 0x00000004u,                                         // OpLabel
    0x000100FDu,                                                      // OpReturn
    0x00010038u,                                                      // OpFunctionEnd
};

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_storage_init_wait_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::real_get_device = d.device_proc;
    driver::real_submit = d.fn<PFN_vkQueueSubmit>("vkQueueSubmit");
    EXPECT(driver::real_submit);

    // The pipeline every dispatch below names: no descriptors, the 144-byte push-constant block every translated
    // shader declares (ShaderUserData).
    const VkDevice device = d.device;
    const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ShaderUserData)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push;
    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    EXPECT(d.fn<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(device, &layout_info, nullptr,
                                                                      &pipeline_layout) == VK_SUCCESS);
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = sizeof(cs_empty_spirv);
    module_info.pCode = cs_empty_spirv;
    VkShaderModule module = VK_NULL_HANDLE;
    EXPECT(d.fn<PFN_vkCreateShaderModule>("vkCreateShaderModule")(device, &module_info, nullptr, &module) ==
           VK_SUCCESS);
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    EXPECT(d.fn<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                                                          nullptr, &pipeline) == VK_SUCCESS);

    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    const VkImage first = d.image(VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, usage);
    const VkImage second = d.image(VK_FORMAT_R8G8B8A8_UNORM, 16, 16, 1, usage);
    EXPECT(first && second);

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        ComputePipeline compute{};
        compute.pipeline = pipeline;
        compute.layout = pipeline_layout;
        constexpr std::uintptr_t pso = 1;
        context.add_pipeline(pso, compute);

        ResourceBinding binding{};
        binding.format = VK_FORMAT_R8G8B8A8_UNORM;
        binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        binding.width = binding.height = 16;
        binding.color_attachment = true;
        constexpr ResourceId uav_a = 1, uav_b = 2;
        binding.image = first;
        context.add_resource(uav_a, binding, VK_IMAGE_LAYOUT_UNDEFINED);
        binding.image = second;
        context.add_resource(uav_b, binding, VK_IMAGE_LAYOUT_UNDEFINED);

        CommandList dispatch, empty, clear;
        EXPECT(dispatch.dispatch(Dispatch{pso, {}, 1, 1, 1}) && dispatch.close());
        EXPECT(empty.close());
        EXPECT(clear.clear_render_target({uav_b, 0, 0, 1}, {0.0f, 0.0f, 0.0f, 0.0f}) && clear.close());
        const CommandList* dispatch_list[] = {&dispatch};
        const CommandList* empty_list[] = {&empty};
        const CommandList* clear_list[] = {&clear};

        VulkanQueue a(context, d.queue, d.family);
        VulkanQueue b(context, d.queue, d.family);
        EXPECT(a.ok() && b.ok());
        EXPECT(b.execute(empty_list) == ReplayResult::success);  // B's first submission, before anything is noted

        // 1. The first UAV of a never-used texture: A initializes it, B waits for that.
        context.note_shader_image(uav_a);
        EXPECT(a.execute(dispatch_list) == ReplayResult::success);
        EXPECT(context.layout(uav_a, 0) == VK_IMAGE_LAYOUT_GENERAL);
        EXPECT(b.execute(empty_list) == ReplayResult::success);
        EXPECT(driver::waits() == 1);
        std::printf("1 first UAV of a new texture: A initializes it, B waits once\n");

        // 2. A second UAV of the initialized texture: nothing to initialize, nothing to wait for.
        context.note_shader_image(uav_a);
        EXPECT(a.execute(dispatch_list) == ReplayResult::success);
        EXPECT(b.execute(empty_list) == ReplayResult::success);
        EXPECT(driver::waits() == 0);
        std::printf("2 second UAV of the initialized texture: B does not wait\n");

        // 3. Noted while UNDEFINED, initialized by B's clear before A's dispatch runs: A initializes nothing.
        context.note_shader_image(uav_b);
        EXPECT(b.execute(clear_list) == ReplayResult::success);
        EXPECT(context.layout(uav_b, 0) != VK_IMAGE_LAYOUT_UNDEFINED);
        EXPECT(a.execute(dispatch_list) == ReplayResult::success);
        EXPECT(b.execute(empty_list) == ReplayResult::success);
        EXPECT(driver::waits() == 0);
        std::printf("3 queued UAV texture initialized by a clear first: B does not wait for A\n");

        EXPECT(a.idle() && b.idle());
    }

    d.fn<PFN_vkDestroyPipeline>("vkDestroyPipeline")(device, pipeline, nullptr);
    d.fn<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(device, module, nullptr);
    d.fn<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(device, pipeline_layout, nullptr);
    const unsigned errors = d.validation_errors();
    if (errors) {
        std::fprintf(stderr, "vk_storage_init_wait_test: %u validation errors\n", errors);
        return 1;
    }
    std::printf("vk_storage_init_wait_test: PASS\n");
    return 0;
}
