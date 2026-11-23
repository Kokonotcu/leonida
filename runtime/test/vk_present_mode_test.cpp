// Unit test for detail::chained_present_mode (runtime/src/vulkan_present_mode.h): the present mode VulkanPresenter
// chains in VkSwapchainPresentModeInfo. CPU-only, no Vulkan instance and no GPU. A chained mode applies to that present
// and every later one on the swapchain, so on a switchable swapchain every present names its mode: the tear mode for
// one that tears, FIFO for one that does not. A swapchain that is not switchable chains nothing.
#include "../src/vulkan_present_mode.h"

#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

int main() {
    using leonida::detail::chained_present_mode;
    constexpr VkPresentModeKHR none = VK_PRESENT_MODE_MAX_ENUM_KHR;

    // Switchable vsync swapchain: the tear mode when tearing, FIFO otherwise (so one tear does not stick).
    const VkPresentModeKHR tear_modes[] = {VK_PRESENT_MODE_IMMEDIATE_KHR, VK_PRESENT_MODE_FIFO_RELAXED_KHR};
    for (const VkPresentModeKHR tear_mode : tear_modes) {
        EXPECT(chained_present_mode(true, true, tear_mode) == tear_mode);
        EXPECT(chained_present_mode(true, false, tear_mode) == VK_PRESENT_MODE_FIFO_KHR);
    }
    // Not switchable (no swapchain_maintenance1, no compatible tear mode): nothing chained, tearing or not.
    EXPECT(chained_present_mode(true, true, none) == none);
    EXPECT(chained_present_mode(true, false, none) == none);
    // No vsync (MAILBOX/IMMEDIATE swapchain, created without the switchable list): nothing chained.
    EXPECT(chained_present_mode(false, true, VK_PRESENT_MODE_IMMEDIATE_KHR) == none);
    EXPECT(chained_present_mode(false, false, VK_PRESENT_MODE_IMMEDIATE_KHR) == none);

    std::printf("vk_present_mode_test: ok\n");
    return 0;
}
