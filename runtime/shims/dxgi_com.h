#pragma once

// COM wrappers for IDXGIFactory7, IDXGIAdapter3, IDXGIOutput6, and IDXGISwapChain3 backed by VulkanSwapchain (dxgi_swapchain.h).
// Keeps the host dxgi.dll out of the Vulkan presentation path (docs/d3d12_vulkan.md section 3.1 & 4).

#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>

#include "dxgi_swapchain.h"

namespace rt::dxgi {

// Creates an IDXGIFactory4 COM object whose CreateSwapChainForHwnd creates a VulkanSwapchain
// and registers its back-buffer images with `context` (keyed by the returned ID3D12Resource* pointer).
// `swapchain_maintenance1` is VulkanDeviceInfo::swapchain_maintenance1: with it Present switches the present mode per
// frame from its SyncInterval and DXGI_PRESENT_ALLOW_TEARING (VulkanSwapchain::present_mode_for).
HRESULT create_factory(leonida::VulkanContext& context, PFN_vkGetInstanceProcAddr gipa, VkInstance instance,
                       VkPhysicalDevice physical, PFN_vkGetDeviceProcAddr gdpa, VkQueue queue,
                       std::uint32_t queue_family, bool swapchain_maintenance1, REFIID riid, void** out);

// The Vulkan present mode the last Present of `swapchain` was queued in, or VK_PRESENT_MODE_MAX_ENUM_KHR when it has
// not presented yet or is not a swap chain from create_factory. For tests.
VkPresentModeKHR last_present_mode(IUnknown* swapchain);

// The present modes Present can switch `swapchain` between, FIFO first (more than one only with
// swapchain_maintenance1 and an IMMEDIATE or MAILBOX mode compatible with FIFO), or its creation mode alone when it
// cannot switch; empty when it is not a swap chain from create_factory. For tests.
std::vector<VkPresentModeKHR> switchable_present_modes(IUnknown* swapchain);

}  // namespace rt::dxgi
