// Review I12: what the presenter does per frame that it need not.
//   1. A composed present (present_planes) made a VkImageView for each plane on every present; the views are now
//      cached with their images. Counts vkCreateImageView over 300 composed presents (was 300, one plane) and the
//      CPU time per present.
//   2. A present while the window is minimised asked recreate() for a swapchain, which waited for the queue to idle
//      before finding the window still has no size: every frame. Counts vkQueueWaitIdle over 300 such presents (was
//      300) and the CPU time per present. Restored, the window is presented to again.
// Prints the measurements. Skips with exit code 77 without a Vulkan 1.3 device with Win32 presentation. No game data.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>

#include "vk_test_device.h"
#include "../src/vulkan_present.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string_view>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w, LPARAM l) {
    return DefWindowProcW(window, message, w, l);
}

void pump() {
    MSG message;
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
}

namespace driver {
PFN_vkGetInstanceProcAddr real_get_instance = nullptr;
PFN_vkGetDeviceProcAddr real_get_device = nullptr;
PFN_vkCreateImageView real_create_view = nullptr;
PFN_vkQueueWaitIdle real_wait_idle = nullptr;
std::atomic<std::uint32_t> views_made{0};
std::atomic<std::uint32_t> idle_waits{0};

VKAPI_ATTR VkResult VKAPI_CALL create_view(VkDevice device, const VkImageViewCreateInfo* info,
                                           const VkAllocationCallbacks* allocator, VkImageView* view) {
    views_made.fetch_add(1);
    return real_create_view(device, info, allocator, view);
}
VKAPI_ATTR VkResult VKAPI_CALL wait_idle(VkQueue queue) {
    idle_waits.fetch_add(1);
    return real_wait_idle(queue);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device(VkDevice device, const char* name) {
    const std::string_view n = name;
    if (n == "vkCreateImageView") return reinterpret_cast<PFN_vkVoidFunction>(create_view);
    if (n == "vkQueueWaitIdle") return reinterpret_cast<PFN_vkVoidFunction>(wait_idle);
    return real_get_device(device, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_instance(VkInstance instance, const char* name) {
    if (std::string_view(name) == "vkGetDeviceProcAddr") return reinterpret_cast<PFN_vkVoidFunction>(get_device);
    return real_get_instance(instance, name);
}
}  // namespace driver

double us_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main() {
    using namespace leonida;
    vktest::Device d;
    if (!d.create(true)) {
        std::fprintf(stderr, "vk_presenter_cost_test: no Vulkan device with presentation, skipped\n");
        return 77;
    }
    driver::real_get_instance = d.instance_proc;
    driver::real_get_device =
        reinterpret_cast<PFN_vkGetDeviceProcAddr>(d.instance_proc(d.instance, "vkGetDeviceProcAddr"));
    driver::real_create_view =
        reinterpret_cast<PFN_vkCreateImageView>(driver::real_get_device(d.device, "vkCreateImageView"));
    driver::real_wait_idle = reinterpret_cast<PFN_vkQueueWaitIdle>(driver::real_get_device(d.device, "vkQueueWaitIdle"));
    EXPECT(driver::real_create_view && driver::real_wait_idle);

    WNDCLASSW wc{};
    wc.lpfnWndProc = window_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"LeonidaPresenterCostTest";
    EXPECT(RegisterClassW(&wc));
    HWND window = CreateWindowExW(0, wc.lpszClassName, L"Leonida presenter cost test", WS_OVERLAPPEDWINDOW, 80, 80,
                                  320, 240, nullptr, nullptr, wc.hInstance, nullptr);
    EXPECT(window);
    ShowWindow(window, SW_SHOWNOACTIVATE);
    UpdateWindow(window);
    pump();
    {
        VulkanContext context(driver::get_device, d.device, true);
        EXPECT(context.loaded());
        const VkImage plane_img = d.image(VK_FORMAT_R8G8B8A8_UNORM, 64, 64, 1,
                                          VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                              VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        EXPECT(plane_img != VK_NULL_HANDLE);
        ResourceBinding plane{};
        plane.image = plane_img;
        plane.format = VK_FORMAT_R8G8B8A8_UNORM;
        plane.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        plane.width = plane.height = 64;
        plane.mip_levels = plane.array_layers = plane.depth = plane.samples = 1;
        plane.color_attachment = true;
        plane.sampled = true;
        context.add_resource(2, plane, VK_IMAGE_LAYOUT_UNDEFINED);

        const auto create_surface = reinterpret_cast<PFN_vkCreateWin32SurfaceKHR>(
            d.instance_proc(d.instance, "vkCreateWin32SurfaceKHR"));
        EXPECT(create_surface);
        VkWin32SurfaceCreateInfoKHR surface_info{VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
        surface_info.hinstance = GetModuleHandleW(nullptr);
        surface_info.hwnd = window;
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        EXPECT(create_surface(d.instance, &surface_info, nullptr, &surface) == VK_SUCCESS);
        VulkanPresenter presenter(context, driver::get_instance, d.instance, d.physical, d.queue, d.family, surface);
        EXPECT(presenter.ok());

        PresentPlanes planes;
        planes.count = 1;
        planes.plane[0].source = 2;
        for (int i = 0; i < 5; ++i) {  // warm up: swapchain, compose pass and target
            pump();
            EXPECT(presenter.present_planes(planes, false) == VK_SUCCESS);
        }

        constexpr int kPresents = 300;
        driver::views_made = 0;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kPresents; ++i) {
            pump();
            EXPECT(presenter.present_planes(planes, false) == VK_SUCCESS);
        }
        const double composed_us = us_since(start) / kPresents;
        const std::uint32_t composed_views = driver::views_made.load();
        std::printf("composed presents: %u image views made over %d (%.1f us per present)\n", composed_views,
                    kPresents, composed_us);
        int failures = 0;  // both measurements are printed before the checks fail the test
        if (composed_views != 0) {
            std::fprintf(stderr, "FAIL: composed presents made image views\n");
            ++failures;
        }

        ShowWindow(window, SW_MINIMIZE);
        for (int i = 0; i < 50; ++i) pump();
        for (int i = 0; i < 5; ++i) {  // the first ones find the swapchain out of date and release it: they wait
            pump();
            EXPECT(presenter.present_planes(planes, false) == VK_SUCCESS);
        }
        const std::uint64_t shown = presenter.presented();
        driver::idle_waits = 0;
        start = std::chrono::steady_clock::now();
        for (int i = 0; i < kPresents; ++i) {
            pump();
            EXPECT(presenter.present_planes(planes, false) == VK_SUCCESS);
        }
        const double minimised_us = us_since(start) / kPresents;
        const std::uint32_t minimised_waits = driver::idle_waits.load();
        std::printf("presents while minimised: %u queue idle waits over %d (%.1f us per present), %llu shown\n",
                    minimised_waits, kPresents, minimised_us,
                    static_cast<unsigned long long>(presenter.presented() - shown));
        const bool minimised = presenter.presented() == shown;  // a window manager may not minimise a test window
        if (!minimised) {
            std::printf("the window did not minimise: the idle waits are not checked\n");
        } else if (minimised_waits != 0) {
            std::fprintf(stderr, "FAIL: presents while minimised waited for the queue to idle\n");
            ++failures;
        }

        ShowWindow(window, SW_RESTORE);
        for (int i = 0; i < 50; ++i) pump();
        EXPECT(presenter.present_planes(planes, false) == VK_SUCCESS);
        EXPECT(presenter.presented() > shown);
        context.remove_resource(2);
        if (failures) return 1;
    }
    DestroyWindow(window);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    EXPECT(d.validation_errors() == 0);
    std::printf("vk_presenter_cost_test: OK\n");
    return 0;
}
