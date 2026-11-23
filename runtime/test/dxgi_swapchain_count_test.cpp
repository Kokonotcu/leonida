// Synthetic regression test for the DXGI swapchain buffer-count contract (D3D-389), with no GPU and no
// window: VulkanSwapchain (runtime/shims/dxgi_swapchain.h) runs on a fake Vulkan function table, so
// vkGetSwapchainImagesKHR can report more images than the app's requested BufferCount.
//
// DXGI rule: GetDesc/GetDesc1 report the BufferCount the app created with, and GetCurrentBackBufferIndex is
// always below it, one stable resource per index. The shim must not expose images the driver added above
// minImageCount, and must hide a caps.minImageCount above the request. When the native count differs, the
// exposed buffers form an offscreen ring; each present blits the current slot into the acquired native image.
//
// The same fake surface also checks the per-present mode (R5-5): with swapchain_maintenance1 a FIFO swapchain switches
// to the mode Present's SyncInterval and DXGI_PRESENT_ALLOW_TEARING ask for, among the modes the fake surface lists and
// reports compatible with FIFO, and Present(0, 0) never tears.
#include "../shims/dxgi_swapchain.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <vector>

namespace {

constexpr std::uintptr_t kSurface = 0x5100;
constexpr std::uintptr_t kSwapchain = 0x5101;
constexpr std::uintptr_t kFence = 0x5102;
constexpr std::uintptr_t kPool = 0x5103;
constexpr std::uintptr_t kCmd = 0x5104;
constexpr std::uintptr_t kNativeBase = 0x6000;
constexpr std::uintptr_t kOffscreenBase = 0x7000;

struct Fake {
    std::uint32_t min_image_count = 2;
    std::uint32_t max_image_count = 0;
    std::uint32_t returned_images = 2;
    std::uint32_t create_min_image_count = 0;
    std::uint32_t acquire_cursor = 0;
    std::uint32_t last_acquired = 0;
    std::uint32_t presented_image = UINT32_MAX;
    VkImage blit_src = VK_NULL_HANDLE;
    VkImage blit_dst = VK_NULL_HANDLE;
    unsigned presents = 0;
    unsigned created_offscreen = 0;
    // Present modes: what the surface lists, which of them it reports compatible with FIFO, what the swapchain was
    // created with and what each present asked for (MAX_ENUM: no VkSwapchainPresentModeInfoEXT).
    std::vector<VkPresentModeKHR> listed_modes{VK_PRESENT_MODE_FIFO_KHR};
    std::vector<VkPresentModeKHR> fifo_compatible;
    VkPresentModeKHR create_mode = VK_PRESENT_MODE_MAX_ENUM_KHR;
    std::uint32_t create_switchable = 0;
    VkPresentModeKHR presented_mode = VK_PRESENT_MODE_MAX_ENUM_KHR;
    VkResult present_result = VK_SUCCESS;
};
Fake g;

VkImage native_image(std::uint32_t i) {
    return reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kNativeBase + i));
}

template <typename T>
PFN_vkVoidFunction as_function(T fn) {
    return reinterpret_cast<PFN_vkVoidFunction>(fn);
}

VKAPI_ATTR VkResult VKAPI_CALL create_win32_surface(VkInstance, const VkWin32SurfaceCreateInfoKHR*,
                                                    const VkAllocationCallbacks*, VkSurfaceKHR* out) {
    if (out) *out = reinterpret_cast<VkSurfaceKHR>(kSurface);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_surface(VkInstance, VkSurfaceKHR, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL surface_support(VkPhysicalDevice, std::uint32_t, VkSurfaceKHR, VkBool32* out) {
    if (out) *out = VK_TRUE;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_caps(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* caps) {
    if (!caps) return VK_ERROR_INITIALIZATION_FAILED;
    std::memset(caps, 0, sizeof(*caps));
    caps->minImageCount = g.min_image_count;
    caps->maxImageCount = g.max_image_count;
    caps->currentExtent = {64, 64};
    caps->minImageExtent = {1, 1};
    caps->maxImageExtent = {4096, 4096};
    caps->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    caps->currentTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_formats(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                               VkSurfaceFormatKHR* formats) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (formats && *count) {
        formats[0].format = VK_FORMAT_B8G8R8A8_UNORM;
        formats[0].colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    }
    *count = 1;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL surface_present_modes(VkPhysicalDevice, VkSurfaceKHR, std::uint32_t* count,
                                                     VkPresentModeKHR* modes) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    const auto n = static_cast<std::uint32_t>(g.listed_modes.size());
    if (!modes) {
        *count = n;
        return VK_SUCCESS;
    }
    const std::uint32_t written = *count < n ? *count : n;
    for (std::uint32_t i = 0; i < written; ++i) modes[i] = g.listed_modes[i];
    *count = written;
    return written < n ? VK_INCOMPLETE : VK_SUCCESS;
}
// The base capabilities in every mode; FIFO reports g.fifo_compatible, any other mode only itself.
VKAPI_ATTR VkResult VKAPI_CALL surface_caps2(VkPhysicalDevice physical, const VkPhysicalDeviceSurfaceInfo2KHR* info,
                                             VkSurfaceCapabilities2KHR* caps) {
    if (!info || !caps) return VK_ERROR_INITIALIZATION_FAILED;
    surface_caps(physical, info->surface, &caps->surfaceCapabilities);
#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
    VkPresentModeKHR asked = VK_PRESENT_MODE_MAX_ENUM_KHR;
    for (auto* in = static_cast<const VkBaseInStructure*>(info->pNext); in; in = in->pNext)
        if (in->sType == VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_EXT)
            asked = reinterpret_cast<const VkSurfacePresentModeEXT*>(in)->presentMode;
    for (auto* out = static_cast<VkBaseOutStructure*>(caps->pNext); out; out = out->pNext) {
        if (out->sType != VK_STRUCTURE_TYPE_SURFACE_PRESENT_MODE_COMPATIBILITY_EXT) continue;
        auto* compat = reinterpret_cast<VkSurfacePresentModeCompatibilityEXT*>(out);
        const std::vector<VkPresentModeKHR> modes =
            asked == VK_PRESENT_MODE_FIFO_KHR ? g.fifo_compatible : std::vector<VkPresentModeKHR>{asked};
        const auto n = static_cast<std::uint32_t>(modes.size());
        if (compat->pPresentModes) {
            const std::uint32_t written = compat->presentModeCount < n ? compat->presentModeCount : n;
            for (std::uint32_t i = 0; i < written; ++i) compat->pPresentModes[i] = modes[i];
            compat->presentModeCount = written;
        } else {
            compat->presentModeCount = n;
        }
    }
#endif
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL memory_properties(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties* props) {
    std::memset(props, 0, sizeof(*props));
    props->memoryTypeCount = 1;
    props->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    props->memoryHeapCount = 1;
    props->memoryHeaps[0].size = 1ull << 30;
    props->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}
VKAPI_ATTR VkResult VKAPI_CALL create_swapchain(VkDevice, const VkSwapchainCreateInfoKHR* info,
                                                const VkAllocationCallbacks*, VkSwapchainKHR* out) {
    if (!info || !out) return VK_ERROR_INITIALIZATION_FAILED;
    g.create_min_image_count = info->minImageCount;
    g.create_mode = info->presentMode;
    g.create_switchable = 0;
#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
    for (auto* in = static_cast<const VkBaseInStructure*>(info->pNext); in; in = in->pNext)
        if (in->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODES_CREATE_INFO_EXT)
            g.create_switchable = reinterpret_cast<const VkSwapchainPresentModesCreateInfoEXT*>(in)->presentModeCount;
#endif
    *out = reinterpret_cast<VkSwapchainKHR>(kSwapchain);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_swapchain(VkDevice, VkSwapchainKHR, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL get_images(VkDevice, VkSwapchainKHR, std::uint32_t* count, VkImage* images) {
    if (!count) return VK_ERROR_INITIALIZATION_FAILED;
    if (images) {
        for (std::uint32_t i = 0; i < g.returned_images && i < *count; ++i) images[i] = native_image(i);
    }
    *count = g.returned_images;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL acquire_next_image(VkDevice, VkSwapchainKHR, std::uint64_t, VkSemaphore, VkFence,
                                                  std::uint32_t* out) {
    if (!out || g.returned_images == 0) return VK_ERROR_INITIALIZATION_FAILED;
    g.last_acquired = g.acquire_cursor % g.returned_images;
    ++g.acquire_cursor;
    *out = g.last_acquired;
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL queue_present(VkQueue, const VkPresentInfoKHR* info) {
    if (!info || info->swapchainCount != 1 || !info->pImageIndices) return VK_ERROR_INITIALIZATION_FAILED;
    g.presented_image = info->pImageIndices[0];
    g.presented_mode = VK_PRESENT_MODE_MAX_ENUM_KHR;
#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
    for (auto* in = static_cast<const VkBaseInStructure*>(info->pNext); in; in = in->pNext)
        if (in->sType == VK_STRUCTURE_TYPE_SWAPCHAIN_PRESENT_MODE_INFO_EXT)
            g.presented_mode = reinterpret_cast<const VkSwapchainPresentModeInfoEXT*>(in)->pPresentModes[0];
#endif
    ++g.presents;
    return g.present_result;
}
VKAPI_ATTR VkResult VKAPI_CALL create_fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*,
                                            VkFence* out) {
    if (out) *out = reinterpret_cast<VkFence>(kFence);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_fence(VkDevice, VkFence, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL wait_for_fences(VkDevice, std::uint32_t, const VkFence*, VkBool32, std::uint64_t) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_fences(VkDevice, std::uint32_t, const VkFence*) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL device_wait_idle(VkDevice) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL create_image(VkDevice, const VkImageCreateInfo*, const VkAllocationCallbacks*,
                                            VkImage* out) {
    if (!out) return VK_ERROR_INITIALIZATION_FAILED;
    *out = reinterpret_cast<VkImage>(static_cast<std::uintptr_t>(kOffscreenBase + g.created_offscreen++));
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL image_mem_req(VkDevice, VkImage, VkMemoryRequirements* req) {
    std::memset(req, 0, sizeof(*req));
    req->size = 4096;
    req->alignment = 1;
    req->memoryTypeBits = 1;
}
VKAPI_ATTR VkResult VKAPI_CALL alloc_mem(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*,
                                         VkDeviceMemory* out) {
    if (out) *out = reinterpret_cast<VkDeviceMemory>(kOffscreenBase);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL free_mem(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL bind_image_mem(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL create_cmd_pool(VkDevice, const VkCommandPoolCreateInfo*,
                                               const VkAllocationCallbacks*, VkCommandPool* out) {
    if (out) *out = reinterpret_cast<VkCommandPool>(kPool);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_cmd_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL alloc_cmd_bufs(VkDevice, const VkCommandBufferAllocateInfo*,
                                              VkCommandBuffer* out) {
    if (out) *out = reinterpret_cast<VkCommandBuffer>(kCmd);
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL reset_cmd_buf(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VKAPI_ATTR VkResult VKAPI_CALL begin_cmd_buf(VkCommandBuffer, const VkCommandBufferBeginInfo*) {
    return VK_SUCCESS;
}
VKAPI_ATTR VkResult VKAPI_CALL end_cmd_buf(VkCommandBuffer) { return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL cmd_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, std::uint32_t,
                                       const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                       std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL cmd_blit(VkCommandBuffer, VkImage src, VkImageLayout, VkImage dst, VkImageLayout,
                                    std::uint32_t, const VkImageBlit*, VkFilter) {
    g.blit_src = src;
    g.blit_dst = dst;
}
VKAPI_ATTR VkResult VKAPI_CALL queue_submit(VkQueue, std::uint32_t, const VkSubmitInfo*, VkFence) {
    return VK_SUCCESS;
}

struct Entry {
    const char* name;
    PFN_vkVoidFunction fn;
};
const Entry kEntries[] = {
    {"vkCreateWin32SurfaceKHR", as_function(create_win32_surface)},
    {"vkDestroySurfaceKHR", as_function(destroy_surface)},
    {"vkGetPhysicalDeviceSurfaceSupportKHR", as_function(surface_support)},
    {"vkGetPhysicalDeviceSurfaceCapabilitiesKHR", as_function(surface_caps)},
    {"vkGetPhysicalDeviceSurfaceFormatsKHR", as_function(surface_formats)},
    {"vkGetPhysicalDeviceSurfacePresentModesKHR", as_function(surface_present_modes)},
    {"vkGetPhysicalDeviceSurfaceCapabilities2KHR", as_function(surface_caps2)},
    {"vkGetPhysicalDeviceMemoryProperties", as_function(memory_properties)},
    {"vkCreateSwapchainKHR", as_function(create_swapchain)},
    {"vkDestroySwapchainKHR", as_function(destroy_swapchain)},
    {"vkGetSwapchainImagesKHR", as_function(get_images)},
    {"vkAcquireNextImageKHR", as_function(acquire_next_image)},
    {"vkQueuePresentKHR", as_function(queue_present)},
    {"vkCreateFence", as_function(create_fence)},
    {"vkDestroyFence", as_function(destroy_fence)},
    {"vkWaitForFences", as_function(wait_for_fences)},
    {"vkResetFences", as_function(reset_fences)},
    {"vkDeviceWaitIdle", as_function(device_wait_idle)},
    {"vkCreateImage", as_function(create_image)},
    {"vkDestroyImage", as_function(destroy_image)},
    {"vkGetImageMemoryRequirements", as_function(image_mem_req)},
    {"vkAllocateMemory", as_function(alloc_mem)},
    {"vkFreeMemory", as_function(free_mem)},
    {"vkBindImageMemory", as_function(bind_image_mem)},
    {"vkCreateCommandPool", as_function(create_cmd_pool)},
    {"vkDestroyCommandPool", as_function(destroy_cmd_pool)},
    {"vkAllocateCommandBuffers", as_function(alloc_cmd_bufs)},
    {"vkResetCommandBuffer", as_function(reset_cmd_buf)},
    {"vkBeginCommandBuffer", as_function(begin_cmd_buf)},
    {"vkEndCommandBuffer", as_function(end_cmd_buf)},
    {"vkCmdPipelineBarrier", as_function(cmd_barrier)},
    {"vkCmdBlitImage", as_function(cmd_blit)},
    {"vkQueueSubmit", as_function(queue_submit)},
};

PFN_vkVoidFunction lookup(const char* name) {
    for (const Entry& entry : kEntries)
        if (name && std::strcmp(name, entry.name) == 0) return entry.fn;
    return nullptr;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance_proc(VkInstance, const char* name) { return lookup(name); }
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice, const char* name) { return lookup(name); }

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

// Requests `requested` buffers while faking `native` images and a surface minimum of `minimum`.
int run_case(const char* label, std::uint32_t requested, std::uint32_t native, std::uint32_t minimum,
             bool expect_offscreen) {
    g = Fake{};
    g.min_image_count = minimum;
    g.returned_images = native;
    std::mutex mutex;
    leonida::VulkanSwapchain swapchain(get_instance_proc, reinterpret_cast<VkInstance>(1),
                                      reinterpret_cast<VkPhysicalDevice>(2), get_device_proc,
                                      reinterpret_cast<VkDevice>(3), reinterpret_cast<VkQueue>(4), 0, mutex);
    if (!swapchain.create(reinterpret_cast<HWND>(static_cast<std::uintptr_t>(1)), 64, 64,
                          VK_FORMAT_B8G8R8A8_UNORM, requested))
        return fail("VulkanSwapchain::create");

    // GetDesc/GetDesc1 read image_count(): the app's request, never the native count.
    if (swapchain.image_count() != requested) {
        std::fprintf(stderr, "FAIL: %s: exposed BufferCount %u, requested %u (native %u)\n", label,
                     swapchain.image_count(), requested, native);
        return 1;
    }
    const std::uint32_t expected_min = minimum > requested ? minimum : requested;
    if (g.create_min_image_count != expected_min) {
        std::fprintf(stderr, "FAIL: %s: vkCreateSwapchainKHR minImageCount %u, expected %u\n", label,
                     g.create_min_image_count, expected_min);
        return 1;
    }
    if (swapchain.is_offscreen() != expect_offscreen)
        return fail(expect_offscreen ? "expected the offscreen ring" : "expected direct native images");

    if (expect_offscreen) {
        if (g.created_offscreen != requested) return fail("offscreen ring size != requested BufferCount");
        std::vector<VkImage> exposed;
        for (std::uint32_t i = 0; i < requested; ++i) {
            const leonida::ResourceBinding binding = swapchain.resource_binding(i);
            if (!binding.image || binding.presentable) return fail("offscreen backbuffer binding is not offscreen");
            if (binding.image != swapchain.image(i)) return fail("binding/image disagree");
            for (VkImage other : exposed)
                if (other == binding.image) return fail("two exposed backbuffers share one image");
            exposed.push_back(binding.image);
        }
        if (swapchain.current_back_buffer_index() >= requested)
            return fail("initial current index out of range");

        std::uint32_t index = 0;
        if (!swapchain.acquire_next_image(index)) return fail("first acquire");
        constexpr unsigned kFrames = 20;
        for (unsigned frame = 0; frame < kFrames; ++frame) {
            const std::uint32_t current = swapchain.current_back_buffer_index();
            if (current >= requested) return fail("GetCurrentBackBufferIndex >= BufferCount");
            if (current != frame % requested) return fail("current index does not cycle over the exposed ring");
            const std::uint32_t acquired = g.last_acquired;
            g.blit_src = g.blit_dst = VK_NULL_HANDLE;
            g.presented_image = UINT32_MAX;
            if (!swapchain.present(current)) return fail("present");
            if (g.blit_src != exposed[current]) return fail("present did not blit the current backbuffer");
            if (g.blit_dst != native_image(acquired)) return fail("present blitted to the wrong acquired image");
            if (g.presented_image != acquired) return fail("presented an image other than the acquired one");
            if (frame + 1 < kFrames && !swapchain.acquire_next_image(index)) return fail("acquire");
        }
        // ResizeBuffers(count=0) keeps the exposed BufferCount, not the native image count.
        if (!swapchain.resize(64, 64)) return fail("resize");
        if (swapchain.image_count() != requested) return fail("resize(count=0) changed the exposed BufferCount");
        if (!swapchain.is_offscreen()) return fail("resize dropped the offscreen ring");
        return 0;
    }

    // Direct path: exposed index and native image index are the same, and nothing is blitted.
    if (swapchain.current_back_buffer_index() != 0) return fail("direct initial index");
    std::uint32_t index = 0;
    if (!swapchain.acquire_next_image(index)) return fail("direct first acquire");
    for (unsigned frame = 0; frame < 9; ++frame) {
        const std::uint32_t current = swapchain.current_back_buffer_index();
        if (current >= requested) return fail("direct GetCurrentBackBufferIndex >= BufferCount");
        if (current != g.last_acquired) return fail("direct current index is not the acquired image");
        if (swapchain.resource_binding(current).image != native_image(current))
            return fail("direct binding is not the native image");
        g.blit_src = g.blit_dst = VK_NULL_HANDLE;
        g.presented_image = UINT32_MAX;
        if (!swapchain.present(current)) return fail("direct present");
        if (g.blit_src || g.blit_dst) return fail("direct present took the blit path");
        if (g.presented_image != current) return fail("direct present index");
        if (frame + 1 < 9 && !swapchain.acquire_next_image(index)) return fail("direct acquire");
    }
    return 0;
}

constexpr VkPresentModeKHR kFifo = VK_PRESENT_MODE_FIFO_KHR;
constexpr VkPresentModeKHR kImmediate = VK_PRESENT_MODE_IMMEDIATE_KHR;
constexpr VkPresentModeKHR kMailbox = VK_PRESENT_MODE_MAILBOX_KHR;

struct ModeCase {
    const char* label;
    bool maintenance1;                            // the device has swapchain_maintenance1
    std::vector<VkPresentModeKHR> listed;         // vkGetPhysicalDeviceSurfacePresentModesKHR
    std::vector<VkPresentModeKHR> compatible;     // VkSurfacePresentModeCompatibilityEXT for FIFO
    bool tearing_chain;                           // created with DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING
    bool switches;                                // created with VkSwapchainPresentModesCreateInfoEXT
    VkPresentModeKHR sync1, torn, sync0, sync4;   // Present(1,0), (0,ALLOW_TEARING), (0,0), (4,0)
};

// Creates the swapchain on the fake surface and checks the mode each Present is queued in.
int run_mode_case(const ModeCase& c) {
    g = Fake{};
    g.listed_modes = c.listed;
    g.fifo_compatible = c.compatible;
    std::mutex mutex;
    leonida::VulkanSwapchain swapchain(get_instance_proc, reinterpret_cast<VkInstance>(1),
                                      reinterpret_cast<VkPhysicalDevice>(2), get_device_proc,
                                      reinterpret_cast<VkDevice>(3), reinterpret_cast<VkQueue>(4), 0, mutex,
                                      c.maintenance1);
    if (!swapchain.create(reinterpret_cast<HWND>(static_cast<std::uintptr_t>(1)), 64, 64,
                          VK_FORMAT_B8G8R8A8_UNORM, 2, c.tearing_chain))
        return fail("VulkanSwapchain::create (present modes)");
    if ((g.create_switchable > 1) != c.switches) {
        std::fprintf(stderr, "FAIL: %s: created with %u switchable modes\n", c.label, g.create_switchable);
        return 1;
    }
    if (c.switches && g.create_mode != kFifo) return fail("a switchable swapchain must be created in FIFO");
    if (swapchain.last_present_mode() != VK_PRESENT_MODE_MAX_ENUM_KHR)
        return fail("last_present_mode before a present");
    struct Step {
        std::uint32_t sync;
        bool tear;
        VkPresentModeKHR want;
    };
    const Step steps[] = {{1, false, c.sync1}, {0, true, c.torn}, {0, false, c.sync0}, {4, false, c.sync4}};
    for (const Step& step : steps) {
        if (step.tear && !c.tearing_chain) continue;  // Present rejects the flag on a plain swap chain
        std::uint32_t index = 0;
        if (!swapchain.acquire_next_image(index)) return fail("acquire (present modes)");
        if (!swapchain.present(swapchain.current_back_buffer_index(), step.sync, step.tear))
            return fail("present (present modes)");
        // Only a switchable swapchain names the mode per present; otherwise the creation mode applies.
        const VkPresentModeKHR queued = c.switches ? g.presented_mode : g.create_mode;
        if (c.switches ? g.presented_mode == VK_PRESENT_MODE_MAX_ENUM_KHR
                       : g.presented_mode != VK_PRESENT_MODE_MAX_ENUM_KHR)
            return fail("VkSwapchainPresentModeInfoEXT chained iff the swapchain switches");
        if (queued != step.want || swapchain.last_present_mode() != step.want) {
            std::fprintf(stderr, "FAIL: %s: Present(%u, %s) queued mode %d (hook %d), expected %d\n", c.label,
                         step.sync, step.tear ? "ALLOW_TEARING" : "0", queued, swapchain.last_present_mode(),
                         step.want);
            return 1;
        }
    }
    // A present the driver refuses does not change the mode the hook reports (the last one, Present(4, 0), was FIFO
    // on a switchable swapchain; this one asks for c.torn or c.sync0).
    const VkPresentModeKHR before = swapchain.last_present_mode();
    std::uint32_t index = 0;
    if (!swapchain.acquire_next_image(index)) return fail("acquire (failed present)");
    g.present_result = VK_ERROR_OUT_OF_DATE_KHR;
    if (swapchain.present(swapchain.current_back_buffer_index(), 0, c.tearing_chain))
        return fail("present should fail when vkQueuePresentKHR does");
    if (swapchain.last_present_mode() != before) return fail("a failed present changed last_present_mode");
    return 0;
}

struct Case {
    const char* label;
    std::uint32_t requested;
    std::uint32_t native;
    std::uint32_t minimum;
    bool offscreen;
};

}  // namespace

int main() {
    const Case cases[] = {
        // Vulkan returns more images than minImageCount: the app asked for 2 and must see 2.
        {"driver returns 4 for BufferCount 2", 2, 4, 2, true},
        // The surface minimum exceeds the request (2): still expose the app's 2.
        {"minImageCount 3 for BufferCount 2", 2, 3, 3, true},
        // Control: the app asks for more than minImageCount; native and exposed counts agree, no blit.
        {"BufferCount 3 above minImageCount 2", 3, 3, 2, false},
    };
    for (const Case& c : cases) {
        if (const int result = run_case(c.label, c.requested, c.native, c.minimum, c.offscreen)) return result;
    }
#ifdef LEONIDA_SWAPCHAIN_MAINTENANCE1
    const ModeCase mode_cases[] = {
        // The fix: a plain swap chain never tears, so Present(0, 0) without MAILBOX stays FIFO, not IMMEDIATE.
        {"plain, IMMEDIATE only", true, {kFifo, kImmediate}, {kFifo, kImmediate}, false, true,
         kFifo, kFifo, kFifo, kFifo},
        {"ALLOW_TEARING, IMMEDIATE only", true, {kFifo, kImmediate}, {kFifo, kImmediate}, true, true,
         kFifo, kImmediate, kFifo, kFifo},
        {"plain, MAILBOX and IMMEDIATE", true, {kImmediate, kMailbox, kFifo}, {kFifo, kMailbox, kImmediate}, false,
         true, kFifo, kFifo, kMailbox, kFifo},
        {"ALLOW_TEARING, MAILBOX and IMMEDIATE", true, {kImmediate, kMailbox, kFifo}, {kFifo, kMailbox, kImmediate},
         true, true, kFifo, kImmediate, kMailbox, kFifo},
        {"ALLOW_TEARING, MAILBOX only", true, {kFifo, kMailbox}, {kFifo, kMailbox}, true, true,
         kFifo, kMailbox, kMailbox, kFifo},
        // Cannot switch: listed but not compatible with FIFO, or no swapchain_maintenance1. Every present keeps the
        // creation mode, the original behaviour (IMMEDIATE for an ALLOW_TEARING swap chain, FIFO for a plain one).
        {"ALLOW_TEARING, nothing compatible with FIFO", true, {kFifo, kImmediate, kMailbox}, {kFifo}, true, false,
         kImmediate, kImmediate, kImmediate, kImmediate},
        {"plain, nothing compatible with FIFO", true, {kFifo, kImmediate, kMailbox}, {kFifo}, false, false,
         kFifo, kFifo, kFifo, kFifo},
        {"ALLOW_TEARING, no swapchain_maintenance1", false, {kFifo, kImmediate, kMailbox},
         {kFifo, kImmediate, kMailbox}, true, false, kImmediate, kImmediate, kImmediate, kImmediate},
    };
    for (const ModeCase& c : mode_cases) {
        if (const int result = run_mode_case(c)) return result;
    }
    std::printf("dxgi_swapchain_count_test: PASS (3 buffer-count cases: extra native images, min above request, "
                "request above min; %u present-mode cases)\n",
                static_cast<unsigned>(std::size(mode_cases)));
#else
    std::printf("dxgi_swapchain_count_test: PASS (3 cases: extra native images, min above request, "
                "request above min; present-mode cases need the swapchain_maintenance1 headers)\n");
#endif
    return 0;
}
