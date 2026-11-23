// VulkanContext::view() caches image views by every field of the subresource range (B19). The key used to leave out
// levelCount, so two requests that differed only in how many mips they cover shared the first one's view: a
// two-mip request after a one-mip one got the one-mip view. Every caller passes one mip today, so this was latent.
//
// A real Vulkan device; vkCreateImageView is intercepted to record the ranges the views are made with:
//   1. a one-mip and a two-mip view of the same image, base mip and layers, are two views, each made with its range;
//   2. the same requests again return the cached views and make none.
// Skips with exit code 77 without a Vulkan 1.3 device. No game data.
#include "vk_test_device.h"
#include "../src/vulkan_backend.h"

#include <cstdint>
#include <cstdio>
#include <string_view>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

namespace driver {
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCreateImageView real_create_view = nullptr;
std::vector<VkImageSubresourceRange> made;

VKAPI_ATTR VkResult VKAPI_CALL create_view(VkDevice device, const VkImageViewCreateInfo* info,
                                           const VkAllocationCallbacks* allocator, VkImageView* view) {
    made.push_back(info->subresourceRange);
    return real_create_view(device, info, allocator, view);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    if (std::string_view(name) == "vkCreateImageView") return reinterpret_cast<PFN_vkVoidFunction>(create_view);
    return real_get_device(device, name);
}
}  // namespace driver

}  // namespace

int main() {
    using namespace leonida;

    vktest::Device d;
    if (!d.create(false)) {
        std::fprintf(stderr, "vk_view_cache_test: no Vulkan 1.3 device, skipped\n");
        return 77;
    }
    driver::real_get_device = d.device_proc;
    driver::real_create_view = d.fn<PFN_vkCreateImageView>("vkCreateImageView");
    EXPECT(driver::real_create_view);

    // A 16x16 RGBA8 colour attachment with two mips.
    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
    image_info.extent = {16, 16, 1};
    image_info.mipLevels = 2;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    VkImage image = VK_NULL_HANDLE;
    EXPECT(d.fn<PFN_vkCreateImage>("vkCreateImage")(d.device, &image_info, nullptr, &image) == VK_SUCCESS);
    VkMemoryRequirements requirements{};
    d.fn<PFN_vkGetImageMemoryRequirements>("vkGetImageMemoryRequirements")(d.device, image, &requirements);
    const VkPhysicalDeviceMemoryProperties& memory = d.memory_properties();
    std::uint32_t type = UINT32_MAX;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX; ++i)
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            type = i;
    EXPECT(type != UINT32_MAX);
    VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    VkDeviceMemory image_memory = VK_NULL_HANDLE;
    EXPECT(d.fn<PFN_vkAllocateMemory>("vkAllocateMemory")(d.device, &allocate, nullptr, &image_memory) == VK_SUCCESS);
    EXPECT(d.fn<PFN_vkBindImageMemory>("vkBindImageMemory")(d.device, image, image_memory, 0) == VK_SUCCESS);

    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        ResourceBinding binding{};
        binding.image = image;
        binding.format = VK_FORMAT_R8G8B8A8_UNORM;
        binding.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        binding.width = binding.height = 16;
        binding.mip_levels = 2;
        binding.color_attachment = true;
        binding.sampled = true;

        const VkImageSubresourceRange one_mip{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const VkImageSubresourceRange two_mips{VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 0, 1};

        // 1. Two views, each made with its own mip count.
        driver::made.clear();
        const VkImageView first = context.view(binding, one_mip);
        const VkImageView second = context.view(binding, two_mips);
        EXPECT(first != VK_NULL_HANDLE && second != VK_NULL_HANDLE);
        EXPECT(first != second);
        EXPECT(driver::made.size() == 2);
        EXPECT(driver::made[0].levelCount == 1 && driver::made[1].levelCount == 2);
        std::printf("1 a one-mip and a two-mip view of the same image are two views\n");

        // 2. Cached: the same requests make nothing new.
        EXPECT(context.view(binding, one_mip) == first);
        EXPECT(context.view(binding, two_mips) == second);
        EXPECT(driver::made.size() == 2);
        std::printf("2 the same requests return the cached views\n");
    }  // the context destroys its views

    d.fn<PFN_vkDestroyImage>("vkDestroyImage")(d.device, image, nullptr);
    d.fn<PFN_vkFreeMemory>("vkFreeMemory")(d.device, image_memory, nullptr);
    const unsigned errors = d.validation_errors();
    if (errors) {
        std::fprintf(stderr, "vk_view_cache_test: %u validation errors\n", errors);
        return 1;
    }
    std::printf("vk_view_cache_test: PASS\n");
    return 0;
}
