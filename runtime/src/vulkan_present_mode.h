// Which present mode VulkanPresenter (vulkan_present.h) chains in VkSwapchainPresentModeInfoEXT/KHR for one present.
// Kept apart from the presenter so the choice can be tested without a device (test/vk_present_mode_test.cpp).
#pragma once

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

namespace leonida::detail {

// `vsync`: the current swapchain is the FIFO one. `tear`: this present asks to tear. `tear_mode`: the second mode the
// swapchain was created switchable with (VkSwapchainPresentModesCreateInfo {FIFO, tear_mode}), or
// VK_PRESENT_MODE_MAX_ENUM_KHR when it was not. Returns the mode to chain, or VK_PRESENT_MODE_MAX_ENUM_KHR to chain
// nothing.
//
// A chained mode applies to that present and to every later one on the swapchain (VK_KHR_swapchain_maintenance1,
// VkSwapchainPresentModeInfoKHR), so a switchable swapchain chains a mode on every present: the tear mode when this
// one tears, FIFO when it does not. Chaining only the tear mode would leave every present after the first tear in it.
// A swapchain that is not switchable chains nothing: a mode outside its create-time list would be invalid.
inline VkPresentModeKHR chained_present_mode(bool vsync, bool tear, VkPresentModeKHR tear_mode) {
    if (!vsync || tear_mode == VK_PRESENT_MODE_MAX_ENUM_KHR) return VK_PRESENT_MODE_MAX_ENUM_KHR;
    return tear ? tear_mode : VK_PRESENT_MODE_FIFO_KHR;
}

}  // namespace leonida::detail
