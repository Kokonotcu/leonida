// The present sequences a first frame takes through the DXGI shim (runtime/shims/dxgi_swapchain.h,
// runtime/shims/dxgi_com.{h,cpp}) that the other tests leave out, run on the real driver. Host DXGI is compared
// live where it is safe to do so. The window resize, minimise/restore and zero-sized cases are the ones that used
// to leave Present returning a sticky E_FAIL (and presenting an image the swapchain no longer owned):
//   A  create with 2 and 3 buffers -> Present -> ResizeBuffers -> Present, and again with a changed buffer count
//   B  SetFullscreenState(TRUE/FALSE) around presents
//   C  a swapchain created on a zero-sized window (Width = Height = 0), then presented
//   D  a present with the window minimised, then restored, and ResizeBuffers while minimised
//   E  the frame-latency waitable object: default latency, wait/present loop, ResizeBuffers, bounds
//   F  3 buffers + ALLOW_TEARING: the back-buffer index cycles over the app's count for 9 presents
//   G  the window resized after creation, Present with no ResizeBuffers
// A call that does not return a value the app can survive is printed as FAIL with the HRESULT. A present that
// hangs (no return in 5 s) ends the process with code 3: that is a result, not a test bug. Without a window
// station (headless CI) the test skips with 77.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include "../shims/d3d12_com.h"
#include "../shims/dxgi_com.h"
#include "vulkan_loader.h"
#include "../src/vulkan_device_filter.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

unsigned& validation_errors() {
    static unsigned count = 0;
    return count;
}

VKAPI_ATTR VkBool32 VKAPI_CALL on_validation(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT types,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) &&
        (types & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT))
        ++validation_errors();
    std::fprintf(stderr, "validation: %s\n", data && data->pMessage ? data->pMessage : "(no message)");
    return VK_FALSE;
}

std::string hr_text(HRESULT hr) {
    char text[64];
    const char* name = nullptr;
    switch (hr) {
    case S_OK: name = "S_OK"; break;
    case S_FALSE: name = "S_FALSE"; break;
    case E_FAIL: name = "E_FAIL"; break;
    case E_INVALIDARG: name = "E_INVALIDARG"; break;
    case E_NOTIMPL: name = "E_NOTIMPL"; break;
    case DXGI_ERROR_INVALID_CALL: name = "DXGI_ERROR_INVALID_CALL"; break;
    case DXGI_ERROR_UNSUPPORTED: name = "DXGI_ERROR_UNSUPPORTED"; break;
    case DXGI_ERROR_NOT_CURRENTLY_AVAILABLE: name = "DXGI_ERROR_NOT_CURRENTLY_AVAILABLE"; break;
    case DXGI_STATUS_OCCLUDED: name = "DXGI_STATUS_OCCLUDED"; break;
    case DXGI_STATUS_MODE_CHANGED: name = "DXGI_STATUS_MODE_CHANGED"; break;
    default: break;
    }
    if (name)
        std::snprintf(text, sizeof(text), "%s (0x%08lx)", name, static_cast<unsigned long>(hr));
    else
        std::snprintf(text, sizeof(text), "0x%08lx", static_cast<unsigned long>(hr));
    return text;
}

void check(bool ok, const char* what, const char* detail = nullptr) {
    ++g_checks;
    if (ok) {
        std::printf("  ok    %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    } else {
        ++g_failures;
        std::printf("  FAIL  %s%s%s\n", what, detail ? ": " : "", detail ? detail : "");
    }
}

void note(const char* what, const std::string& value) { std::printf("  note  %s: %s\n", what, value.c_str()); }

LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) { return DefWindowProcW(hwnd, msg, wp, lp); }

void ensure_class() {
    static bool done = false;
    if (done) return;
    WNDCLASSW wc{};
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"LeonidaPresentRiskWnd";
    RegisterClassW(&wc);
    done = true;
}

HWND make_window(int width, int height, DWORD style = WS_OVERLAPPEDWINDOW) {
    ensure_class();
    RECT rc{0, 0, width, height};
    AdjustWindowRect(&rc, style, FALSE);
    HWND hwnd = CreateWindowExW(0, L"LeonidaPresentRiskWnd", L"Leonida present risk", style, CW_USEDEFAULT,
                                CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr,
                                GetModuleHandleW(nullptr), nullptr);
    if (hwnd) ShowWindow(hwnd, SW_SHOWNORMAL);
    return hwnd;
}

void pump() {
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
}

bool wait_fence(ID3D12Fence* fence, UINT64 value) {
    HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!ev) return false;
    const HRESULT hr = fence->SetEventOnCompletion(value, ev);
    if (FAILED(hr)) {
        CloseHandle(ev);
        return false;
    }
    const DWORD w = WaitForSingleObject(ev, 5000);
    CloseHandle(ev);
    return w == WAIT_OBJECT_0;
}

HRESULT create_guarded(IDXGIFactory4* factory, IUnknown* queue, HWND hwnd, const DXGI_SWAP_CHAIN_DESC1* desc,
                       IDXGISwapChain1** out) {
    __try {
        return factory->CreateSwapChainForHwnd(queue, hwnd, desc, nullptr, nullptr, out);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return HRESULT(0xC0000005);
    }
}

DXGI_SWAP_CHAIN_DESC1 base_desc(UINT width, UINT height, UINT count) {
    DXGI_SWAP_CHAIN_DESC1 desc{};
    desc.Width = width;
    desc.Height = height;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    desc.BufferCount = count;
    desc.Scaling = DXGI_SCALING_STRETCH;
    desc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    desc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    return desc;
}

// A present that may never return: run it on a thread and give up after `ms`.
struct PresentProbe {
    std::thread thread;
    std::atomic<bool> finished{false};
    HRESULT hr = E_PENDING;
};

bool run_present(IDXGISwapChain* chain, UINT sync, UINT flags, DWORD timeout_ms, HRESULT* out) {
    PresentProbe probe;
    probe.thread = std::thread([&probe, chain, sync, flags] {
        probe.hr = chain->Present(sync, flags);
        probe.finished.store(true, std::memory_order_release);
    });
    const DWORD slice = 10;
    DWORD waited = 0;
    while (!probe.finished.load(std::memory_order_acquire) && waited < timeout_ms) {
        Sleep(slice);
        waited += slice;
    }
    if (!probe.finished.load(std::memory_order_acquire)) {
        std::printf("  HANG  Present(sync=%u, flags=0x%x) did not return in %lu ms\n", sync, flags, timeout_ms);
        std::fflush(stdout);
        probe.thread.detach();
        std::_Exit(3);
    }
    probe.thread.join();
    *out = probe.hr;
    return true;
}

struct Host {
    ID3D12Device* device = nullptr;
    IDXGIFactory4* factory = nullptr;
    ID3D12CommandQueue* queue = nullptr;
};

Host make_host() {
    using CreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    using CreateFactory2 = HRESULT(WINAPI*)(UINT, REFIID, void**);
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const HMODULE dxgi = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create_device =
        d3d12 ? reinterpret_cast<CreateDevice>(reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
              : nullptr;
    const auto create_factory =
        dxgi ? reinterpret_cast<CreateFactory2>(reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory2")))
             : nullptr;
    Host host;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    if (!create_device || !create_factory ||
        FAILED(create_device(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.device))) ||
        FAILED(create_factory(0, IID_PPV_ARGS(&host.factory))) ||
        FAILED(host.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&host.queue)))) {
        if (host.factory) host.factory->Release();
        if (host.device) host.device->Release();
        return {};
    }
    return host;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

// Clears the current back buffer to `rgba`, presents it, and checks the clear reached the newest back buffer's
// pixels through a readback (the same check dxgi_swapchain_test does at creation).
int clear_present_and_verify(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12CommandAllocator* alloc,
                             ID3D12GraphicsCommandList* cl, ID3D12Fence* fence, ID3D12DescriptorHeap* rtv_heap,
                             IDXGISwapChain3* sc, UINT64& fence_value, const float rgba[4], std::uint8_t want_r,
                             std::uint8_t want_g, std::uint8_t want_b) {
    DXGI_SWAP_CHAIN_DESC1 desc{};
    if (FAILED(sc->GetDesc1(&desc))) return 1;
    const UINT index = sc->GetCurrentBackBufferIndex();
    ID3D12Resource* bb = nullptr;
    if (FAILED(sc->GetBuffer(index, IID_PPV_ARGS(&bb))) || !bb) return 2;
    const D3D12_RESOURCE_DESC bb_desc = bb->GetDesc();
    device->CreateRenderTargetView(bb, nullptr, rtv_heap->GetCPUDescriptorHandleForHeapStart());

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&bb_desc, 0, 1, 0, &fp, nullptr, nullptr, &total);
    D3D12_HEAP_PROPERTIES rb_hp{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC rb_desc{};
    rb_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rb_desc.Width = total;
    rb_desc.Height = 1;
    rb_desc.DepthOrArraySize = 1;
    rb_desc.MipLevels = 1;
    rb_desc.SampleDesc.Count = 1;
    rb_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* rb = nullptr;
    if (FAILED(device->CreateCommittedResource(&rb_hp, D3D12_HEAP_FLAG_NONE, &rb_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&rb)))) {
        bb->Release();
        return 3;
    }

    alloc->Reset();
    cl->Reset(alloc, nullptr);
    auto b_rt = transition(bb, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cl->ResourceBarrier(1, &b_rt);
    cl->ClearRenderTargetView(rtv_heap->GetCPUDescriptorHandleForHeapStart(), rgba, 0, nullptr);
    auto b_src = transition(bb, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    cl->ResourceBarrier(1, &b_src);
    D3D12_TEXTURE_COPY_LOCATION src_loc{bb, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_TEXTURE_COPY_LOCATION dst_loc{};
    dst_loc.pResource = rb;
    dst_loc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst_loc.PlacedFootprint = fp;
    cl->CopyTextureRegion(&dst_loc, 0, 0, 0, &src_loc, nullptr);
    auto b_pres = transition(bb, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_PRESENT);
    cl->ResourceBarrier(1, &b_pres);
    cl->Close();

    ID3D12CommandList* lists[1] = {cl};
    queue->ExecuteCommandLists(1, lists);
    queue->Signal(fence, ++fence_value);
    int result = 0;
    void* mapped = nullptr;
    if (!wait_fence(fence, fence_value) || FAILED(rb->Map(0, nullptr, &mapped)) || !mapped) {
        result = 4;
    } else {
        const auto* base = static_cast<const std::uint8_t*>(mapped);
        const bool bgra = bb_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                          bb_desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
        for (UINT y = 0; y < static_cast<UINT>(bb_desc.Height) && result == 0; ++y) {
            const std::uint8_t* row = base + static_cast<std::size_t>(y) * fp.Footprint.RowPitch;
            for (UINT x = 0; x < static_cast<UINT>(bb_desc.Width); ++x) {
                const std::uint8_t* px = row + static_cast<std::size_t>(x) * 4;
                const int r = bgra ? px[2] : px[0];
                const int g = px[1];
                const int b = bgra ? px[0] : px[2];
                if (std::abs(r - int(want_r)) > 1 || std::abs(g - int(want_g)) > 1 ||
                    std::abs(b - int(want_b)) > 1 || px[3] != 255) {
                    result = 5;
                    break;
                }
            }
        }
        rb->Unmap(0, nullptr);
    }
    rb->Release();
    bb->Release();
    return result;
}

}  // namespace

int main() {
    const PFN_vkGetInstanceProcAddr gipa = load_vulkan();
    if (!gipa) {
        std::printf("SKIP: vulkan-1.dll not available\n");
        return 77;
    }
    const auto create_inst = reinterpret_cast<PFN_vkCreateInstance>(gipa(nullptr, "vkCreateInstance"));
    const auto enum_layers =
        reinterpret_cast<PFN_vkEnumerateInstanceLayerProperties>(gipa(nullptr, "vkEnumerateInstanceLayerProperties"));
    if (!create_inst) return 1;

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
    const char* env_val = std::getenv("LEONIDA_VK_VALIDATION");
#ifdef _MSC_VER
#pragma warning(pop)
#endif

    const bool want_val = env_val && std::strcmp(env_val, "0") != 0;
    const char* val_layer = "VK_LAYER_KHRONOS_validation";
    bool val_active = false;
    if (want_val && enum_layers) {
        std::uint32_t cnt = 0;
        enum_layers(&cnt, nullptr);
        std::vector<VkLayerProperties> layers(cnt);
        enum_layers(&cnt, layers.data());
        for (const auto& l : layers)
            if (!std::strcmp(l.layerName, val_layer)) val_active = true;
        if (!val_active) {
            std::printf("FAIL: LEONIDA_VK_VALIDATION set but the layer is not installed\n");
            return 1;
        }
    }

    std::vector<const char*> inst_exts = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME};
    if (val_active) inst_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    VkDebugUtilsMessengerCreateInfoEXT dbg{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    dbg.pfnUserCallback = on_validation;
    const VkValidationFeatureEnableEXT sync_val = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT val_feats{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, &dbg};
    val_feats.enabledValidationFeatureCount = 1;
    val_feats.pEnabledValidationFeatures = &sync_val;

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = static_cast<std::uint32_t>(inst_exts.size());
    ici.ppEnabledExtensionNames = inst_exts.data();
    if (val_active) {
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &val_layer;
        ici.pNext = &val_feats;
    }
    VkInstance instance = VK_NULL_HANDLE;
    if (create_inst(&ici, nullptr, &instance) != VK_SUCCESS) return 1;
    const auto destroy_inst = reinterpret_cast<PFN_vkDestroyInstance>(gipa(instance, "vkDestroyInstance"));
    const auto gdpa = reinterpret_cast<PFN_vkGetDeviceProcAddr>(gipa(instance, "vkGetDeviceProcAddr"));
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_dbg = nullptr;
    if (val_active) {
        const auto create_dbg =
            reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(gipa(instance, "vkCreateDebugUtilsMessengerEXT"));
        destroy_dbg =
            reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(gipa(instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (create_dbg) create_dbg(instance, &dbg, nullptr, &messenger);
    }

    const auto enum_devs = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(gipa(instance, "vkEnumeratePhysicalDevices"));
    const auto get_props = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(gipa(instance, "vkGetPhysicalDeviceProperties"));
    const auto get_qf =
        reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(gipa(instance, "vkGetPhysicalDeviceQueueFamilyProperties"));
    const auto create_dev = reinterpret_cast<PFN_vkCreateDevice>(gipa(instance, "vkCreateDevice"));
    std::uint32_t pcnt = 0;
    enum_devs(instance, &pcnt, nullptr);
    std::vector<VkPhysicalDevice> physicals(pcnt);
    enum_devs(instance, &pcnt, physicals.data());
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    std::uint32_t family = 0;
    VkPhysicalDeviceProperties chosen{};
    int best_rank = -1;
    for (auto cand : physicals) {
        VkPhysicalDeviceProperties props{};
        get_props(cand, &props);
        if (props.apiVersion < VK_API_VERSION_1_3 || !leonida::vulkan_device_wanted(props.deviceName)) continue;
        std::uint32_t fcnt = 0;
        get_qf(cand, &fcnt, nullptr);
        std::vector<VkQueueFamilyProperties> fams(fcnt);
        get_qf(cand, &fcnt, fams.data());
        for (std::uint32_t i = 0; i < fcnt; ++i) {
            if (!(fams[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
            if (const int rank = leonida::vulkan_device_rank(static_cast<int>(props.deviceType), props.vendorID);
                rank > best_rank) {
                best_rank = rank;
                physical = cand;
                family = i;
                chosen = props;
            }
            break;
        }
    }
    if (!physical) {
        const std::string filter = leonida::vulkan_device_filter();
        if (filter.empty()) std::printf("SKIP: no Vulkan 1.3 GPU\n");
        else std::printf("FAIL: LEONIDA_VK_DEVICE=%s matched no Vulkan 1.3 GPU\n", filter.c_str());
        return filter.empty() ? 77 : 1;
    }

    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13};
    f12.timelineSemaphore = VK_TRUE;
    f12.separateDepthStencilLayouts = VK_TRUE;
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char* dev_ext = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &dev_ext;
    VkDevice vk_dev = VK_NULL_HANDLE;
    if (create_dev(physical, &dci, nullptr, &vk_dev) != VK_SUCCESS) return 1;
    const auto destroy_dev = reinterpret_cast<PFN_vkDestroyDevice>(gdpa(vk_dev, "vkDestroyDevice"));
    const auto wait_idle = reinterpret_cast<PFN_vkDeviceWaitIdle>(gdpa(vk_dev, "vkDeviceWaitIdle"));
    VkQueue vk_queue = VK_NULL_HANDLE;
    reinterpret_cast<PFN_vkGetDeviceQueue>(gdpa(vk_dev, "vkGetDeviceQueue"))(vk_dev, family, 0, &vk_queue);

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = gipa;
    info.instance = instance;
    info.physical = physical;
    info.device = vk_dev;
    info.queue = vk_queue;
    info.queue_family = family;
    info.separate_depth_stencil_layouts = true;
    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))) || !device) return 1;

    IDXGIFactory4* factory = nullptr;
    if (FAILED(rt::d3d12::create_dxgi_factory(device, IID_PPV_ARGS(&factory))) || !factory) return 1;

    // Headless CI (a service session, no window station) cannot create a window: every case below needs one, and
    // the API calls that take an HWND would fail for a reason that is not the layer's. Skip, as ctest expects (77).
    if (HWND probe = make_window(32, 32)) {
        DestroyWindow(probe);
    } else {
        std::printf("SKIP: CreateWindowExW failed (no window station)\n");
        factory->Release();
        device->Release();
        wait_idle(vk_dev);
        destroy_dev(vk_dev, nullptr);
        if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
        destroy_inst(instance, nullptr);
        return 77;
    }

    D3D12_COMMAND_QUEUE_DESC qd{D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* cl = nullptr;
    ID3D12Fence* fence = nullptr;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cl))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)))) {
        std::printf("FAIL: queue/alloc/cl/fence setup\n");
        return 1;
    }
    cl->Close();
    D3D12_DESCRIPTOR_HEAP_DESC rtv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ID3D12DescriptorHeap* rtv_heap = nullptr;
    if (FAILED(device->CreateDescriptorHeap(&rtv_hd, IID_PPV_ARGS(&rtv_heap)))) {
        std::printf("FAIL: CreateDescriptorHeap\n");
        return 1;
    }

    const Host host = make_host();
    std::printf("device: %s (validation %s)\n", chosen.deviceName, val_active ? "on" : "off");
    std::printf("host comparison: %s\n", host.factory ? "on" : "unavailable");

    UINT64 fence_value = 0;

    // ---------------------------------------------------------------- A: buffer counts, present, resize, present
    std::printf("\nA. create with 2 and 3 buffers, Present, ResizeBuffers, Present again\n");
    for (const UINT count : {2u, 3u}) {
        HWND hwnd = make_window(128, 128);
        if (!hwnd) {
            check(false, "CreateWindowExW");
            continue;
        }
        pump();
        DXGI_SWAP_CHAIN_DESC1 desc = base_desc(128, 128, count);
        IDXGISwapChain1* sc1 = nullptr;
        HRESULT hr = create_guarded(factory, queue, hwnd, &desc, &sc1);
        check(hr == S_OK && sc1, "create", hr_text(hr).c_str());
        if (FAILED(hr) || !sc1) {
            DestroyWindow(hwnd);
            continue;
        }
        IDXGISwapChain3* sc = nullptr;
        sc1->QueryInterface(IID_PPV_ARGS(&sc));
        sc1->Release();

        DXGI_SWAP_CHAIN_DESC1 got{};
        sc->GetDesc1(&got);
        check(got.BufferCount == count, "GetDesc1 reports the requested BufferCount",
              (std::to_string(got.BufferCount) + " vs " + std::to_string(count)).c_str());
        bool index_ok = true;
        for (UINT i = 0; i < count; ++i) index_ok = index_ok && sc->GetCurrentBackBufferIndex() < count;
        check(index_ok, "GetCurrentBackBufferIndex < BufferCount");
        hr = sc->Present(0, 0);
        check(hr == S_OK, "Present #1", hr_text(hr).c_str());
        hr = sc->Present(0, 0);
        check(hr == S_OK, "Present #2 (no draw)", hr_text(hr).c_str());

        hr = sc->ResizeBuffers(0, 96, 96, DXGI_FORMAT_UNKNOWN, 0);
        check(hr == S_OK, "ResizeBuffers(0, 96x96)", hr_text(hr).c_str());
        sc->GetDesc1(&got);
        check(got.Width == 96 && got.Height == 96 && got.BufferCount == count,
              "GetDesc1 after resize (96x96, same count)",
              (std::to_string(got.Width) + "x" + std::to_string(got.Height) + " c" + std::to_string(got.BufferCount)).c_str());
        hr = sc->Present(0, 0);
        check(hr == S_OK, "Present after resize", hr_text(hr).c_str());

        const UINT changed = count == 2 ? 3 : 2;
        hr = sc->ResizeBuffers(changed, 64, 64, DXGI_FORMAT_UNKNOWN, 0);
        check(hr == S_OK, "ResizeBuffers(count change)", hr_text(hr).c_str());
        sc->GetDesc1(&got);
        check(got.BufferCount == changed && got.Width == 64 && got.Height == 64,
              "GetDesc1 after count change",
              (std::to_string(got.Width) + "x" + std::to_string(got.Height) + " c" + std::to_string(got.BufferCount)).c_str());
        hr = sc->Present(0, 0);
        check(hr == S_OK, "Present after count change", hr_text(hr).c_str());

        const float rgba[4] = {0.1f, 0.8f, 0.2f, 1.0f};
        const int pixels = clear_present_and_verify(device, queue, alloc, cl, fence, rtv_heap, sc, fence_value, rgba,
                                                    26, 204, 51);
        check(pixels == 0, "clear + readback + Present after the resizes", pixels ? "pixel check failed" : nullptr);

        sc->Release();
        DestroyWindow(hwnd);
    }

    // ---------------------------------------------------------------- E: the frame-latency waitable object
    {
        std::printf("\nE. frame-latency waitable object (3 buffers)\n");
        HWND hwnd = make_window(160, 120);
        pump();
        DXGI_SWAP_CHAIN_DESC1 desc = base_desc(160, 120, 3);
        desc.Flags = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
        IDXGISwapChain1* sc1 = nullptr;
        HRESULT hr = create_guarded(factory, queue, hwnd, &desc, &sc1);
        check(hr == S_OK && sc1, "create with FRAME_LATENCY_WAITABLE_OBJECT", hr_text(hr).c_str());
        if (SUCCEEDED(hr) && sc1) {
            IDXGISwapChain2* sc = nullptr;
            sc1->QueryInterface(IID_PPV_ARGS(&sc));
            sc1->Release();
            UINT latency = 0;
            hr = sc->GetMaximumFrameLatency(&latency);
            check(hr == S_OK && latency == 1, "default latency is 1",
                  (hr_text(hr) + " latency=" + std::to_string(latency)).c_str());
            HANDLE waitable = sc->GetFrameLatencyWaitableObject();
            check(waitable != nullptr, "GetFrameLatencyWaitableObject is non-null");
            if (waitable) {
                // DXGI: the semaphore starts at the frame latency (1), so one Present is allowed before the first wait;
                // every wait takes a count and every Present gives one back.
                check(WaitForSingleObject(waitable, 5000) == WAIT_OBJECT_0, "signalled before the first Present");
                hr = sc->Present(0, 0);
                check(hr == S_OK, "Present #1 on the waitable chain", hr_text(hr).c_str());
                bool presented = true;
                DWORD worst = 0;
                for (int i = 0; i < 6; ++i) {
                    const DWORD before = GetTickCount();
                    const DWORD waited = WaitForSingleObject(waitable, 5000);
                    const DWORD wait_ms = GetTickCount() - before;
                    hr = sc->Present(0, 0);
                    if (waited != WAIT_OBJECT_0 || hr != S_OK) presented = false;
                    if (waited != WAIT_OBJECT_0) {
                        note(("frame " + std::to_string(i) + ": wait " +
                              (waited == WAIT_TIMEOUT ? "TIMEOUT" : "failed") + " after " + std::to_string(wait_ms) +
                              " ms, Present=" + hr_text(hr))
                                 .c_str(),
                             "");
                    }
                    if (wait_ms > worst) worst = wait_ms;
                }
                check(presented, "6 wait-then-Present frames");
                note("worst wait", std::to_string(worst) + " ms");

                hr = sc->SetMaximumFrameLatency(2);
                check(hr == S_OK, "SetMaximumFrameLatency(2)", hr_text(hr).c_str());
                sc->GetMaximumFrameLatency(&latency);
                check(latency == 2, "GetMaximumFrameLatency reads back 2", std::to_string(latency).c_str());
                check(WaitForSingleObject(waitable, 0) == WAIT_OBJECT_0, "raise to 2 releases a count immediately");
                hr = sc->SetMaximumFrameLatency(0);
                check(hr == DXGI_ERROR_INVALID_CALL, "SetMaximumFrameLatency(0) is refused", hr_text(hr).c_str());
                hr = sc->SetMaximumFrameLatency(41);
                check(hr == DXGI_ERROR_INVALID_CALL, "SetMaximumFrameLatency(41) is refused", hr_text(hr).c_str());

                hr = sc->ResizeBuffers(3, 100, 100, DXGI_FORMAT_UNKNOWN,
                                       DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT);
                check(hr == S_OK, "ResizeBuffers keeping the waitable flag", hr_text(hr).c_str());
                hr = sc->Present(0, 0);
                check(hr == S_OK, "Present after the waitable resize", hr_text(hr).c_str());
                check(WaitForSingleObject(waitable, 5000) == WAIT_OBJECT_0, "waitable still signals after resize");
                hr = sc->ResizeBuffers(0, 0, 100, DXGI_FORMAT_UNKNOWN, 0);
                check(hr == E_INVALIDARG, "ResizeBuffers dropping the waitable flag is refused", hr_text(hr).c_str());
                CloseHandle(waitable);
            }
            sc->Release();
        }
        DestroyWindow(hwnd);
    }

    // ---------------------------------------------------------------- F: 3 buffers + tearing, index cycle
    {
        std::printf("\nF. 3 buffers + ALLOW_TEARING, 9 presents, back-buffer index cycle\n");
        HWND hwnd = make_window(160, 120);
        pump();
        DXGI_SWAP_CHAIN_DESC1 desc = base_desc(160, 120, 3);
        desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
        IDXGISwapChain1* sc1 = nullptr;
        HRESULT hr = create_guarded(factory, queue, hwnd, &desc, &sc1);
        check(hr == S_OK && sc1, "create with ALLOW_TEARING", hr_text(hr).c_str());
        if (SUCCEEDED(hr) && sc1) {
            IDXGISwapChain3* sc = nullptr;
            sc1->QueryInterface(IID_PPV_ARGS(&sc));
            sc1->Release();
            bool cycle_ok = true, presents_ok = true;
            std::string seen;
            for (int i = 0; i < 9; ++i) {
                const UINT index = sc->GetCurrentBackBufferIndex();
                seen += std::to_string(index);
                cycle_ok = cycle_ok && index < 3;
                if (sc->Present(0, DXGI_PRESENT_ALLOW_TEARING) != S_OK) presents_ok = false;
            }
            check(cycle_ok, "every GetCurrentBackBufferIndex < BufferCount");
            note("index sequence", seen);
            check(presents_ok, "9 tearing presents return S_OK");
            UINT count = 0;
            sc->GetLastPresentCount(&count);
            check(count == 9, "GetLastPresentCount is 9", std::to_string(count).c_str());
            hr = sc->Present(1, DXGI_PRESENT_ALLOW_TEARING);
            check(hr == DXGI_ERROR_INVALID_CALL, "tearing present at sync 1 is refused", hr_text(hr).c_str());
            sc->Release();
        }
        DestroyWindow(hwnd);
    }

    // ---------------------------------------------------------------- B: fullscreen toggle
    {
        std::printf("\nB. SetFullscreenState(TRUE) / FALSE around presents\n");
        HWND hwnd = make_window(200, 150);
        pump();
        DXGI_SWAP_CHAIN_DESC1 desc = base_desc(200, 150, 2);
        IDXGISwapChain1* sc1 = nullptr;
        HRESULT hr = create_guarded(factory, queue, hwnd, &desc, &sc1);
        check(hr == S_OK && sc1, "create", hr_text(hr).c_str());
        if (SUCCEEDED(hr) && sc1) {
            BOOL is_fullscreen = TRUE;
            hr = sc1->SetFullscreenState(TRUE, nullptr);
            note("SetFullscreenState(TRUE, null output) returned", hr_text(hr));
            const HRESULT state_hr = sc1->GetFullscreenState(&is_fullscreen, nullptr);
            note("GetFullscreenState after TRUE",
                 hr_text(state_hr) + " fullscreen=" + (is_fullscreen ? "TRUE" : "FALSE"));
            check(hr == S_OK, "SetFullscreenState(TRUE) does not fail", hr_text(hr).c_str());
            hr = sc1->Present(0, 0);
            check(hr == S_OK, "Present while 'fullscreen'", hr_text(hr).c_str());
            hr = sc1->SetFullscreenState(FALSE, nullptr);
            check(hr == S_OK, "SetFullscreenState(FALSE)", hr_text(hr).c_str());
            hr = sc1->Present(0, 0);
            check(hr == S_OK, "Present after leaving fullscreen", hr_text(hr).c_str());
            sc1->Release();
        }
        DestroyWindow(hwnd);
    }

    // ---------------------------------------------------------------- C: zero-sized window
    {
        std::printf("\nC. a swapchain created on a zero-sized window, then presented\n");
        HWND hwnd = make_window(0, 0, WS_POPUP);
        pump();
        RECT client{};
        GetClientRect(hwnd, &client);
        note("client area at create", std::to_string(client.right) + "x" + std::to_string(client.bottom));
        DXGI_SWAP_CHAIN_DESC1 desc = base_desc(0, 0, 2);
        IDXGISwapChain1* sc1 = nullptr;
        HRESULT hr = create_guarded(factory, queue, hwnd, &desc, &sc1);
        note("create with Width=Height=0 on a 0x0 window", hr_text(hr));
        if (host.factory) {
            DXGI_SWAP_CHAIN_DESC1 hdesc = desc;
            IDXGISwapChain1* hsc = nullptr;
            const HRESULT hhr = create_guarded(host.factory, host.queue, hwnd, &hdesc, &hsc);
            note("host CreateSwapChainForHwnd on the same 0x0 window", hr_text(hhr));
            if (SUCCEEDED(hhr) && hsc) hsc->Release();
        }
        if (FAILED(hr)) {
            // Host DXGI refuses a swapchain on a window with no client area; the shim must not create a Vulkan
            // swapchain with a zero extent (VUID-VkSwapchainCreateInfoKHR-imageExtent-01274) instead.
            check(sc1 == nullptr, "a refused create returns no swapchain");
        } else if (sc1) {
            DXGI_SWAP_CHAIN_DESC1 got{};
            sc1->GetDesc1(&got);
            note("GetDesc1", std::to_string(got.Width) + "x" + std::to_string(got.Height));
            HRESULT present = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &present)) check(present == S_OK, "Present on the 0x0 window", hr_text(present).c_str());
            HRESULT second = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &second)) check(second == S_OK, "second Present on the 0x0 window", hr_text(second).c_str());
            // The window gets its real size later: does the same chain recover without ResizeBuffers?
            SetWindowPos(hwnd, nullptr, 0, 0, 320, 240, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
            pump();
            UINT presents_before = 0, presents_after = 0;
            sc1->GetLastPresentCount(&presents_before);
            HRESULT after = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &after)) {
                check(after == S_OK, "Present after the window is resized (no ResizeBuffers)", hr_text(after).c_str());
                sc1->GetLastPresentCount(&presents_after);
                note("GetLastPresentCount across the failed Present",
                     std::to_string(presents_before) + " -> " + std::to_string(presents_after) +
                         (presents_after == presents_before + 1 ? " (the present itself was taken; the re-acquire failed)"
                                                                : " (the present itself failed; no new image was acquired)"));
            }
            HRESULT after2 = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &after2)) note("second Present without ResizeBuffers", hr_text(after2));
            hr = sc1->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
            check(hr == S_OK, "ResizeBuffers(0,0,0) after the window got a size", hr_text(hr).c_str());
            DXGI_SWAP_CHAIN_DESC1 after_desc{};
            sc1->GetDesc1(&after_desc);
            note("GetDesc1 after the resize", std::to_string(after_desc.Width) + "x" + std::to_string(after_desc.Height));
            HRESULT restored = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &restored))
                check(restored == S_OK, "Present after ResizeBuffers", hr_text(restored).c_str());
            sc1->Release();
        }
        DestroyWindow(hwnd);
    }

    // ---------------------------------------------------------------- D: minimised window
    {
        std::printf("\nD. a present with the window minimised, then restored\n");
        HWND hwnd = make_window(200, 150);
        pump();
        DXGI_SWAP_CHAIN_DESC1 desc = base_desc(200, 150, 2);
        IDXGISwapChain1* sc1 = nullptr;
        HRESULT hr = create_guarded(factory, queue, hwnd, &desc, &sc1);
        check(hr == S_OK && sc1, "create", hr_text(hr).c_str());
        if (SUCCEEDED(hr) && sc1) {
            HRESULT hr_now = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &hr_now)) check(hr_now == S_OK, "Present while shown", hr_text(hr_now).c_str());
            ShowWindow(hwnd, SW_MINIMIZE);
            pump();
            HRESULT hr_min = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &hr_min)) {
                note("Present while minimised", hr_text(hr_min));
                check(SUCCEEDED(hr_min), "Present while minimised does not fail", hr_text(hr_min).c_str());
            }
            HRESULT hr_min2 = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &hr_min2)) note("second Present while minimised", hr_text(hr_min2));
            if (host.factory) {
                HWND hwnd_h = make_window(200, 150);
                pump();
                DXGI_SWAP_CHAIN_DESC1 hdesc = desc;
                IDXGISwapChain1* hsc = nullptr;
                if (SUCCEEDED(create_guarded(host.factory, host.queue, hwnd_h, &hdesc, &hsc)) && hsc) {
                    HRESULT h = E_PENDING;
                    if (run_present(hsc, 0, 0, 5000, &h)) note("host Present while shown", hr_text(h));
                    ShowWindow(hwnd_h, SW_MINIMIZE);
                    pump();
                    if (run_present(hsc, 0, 0, 5000, &h)) note("host Present while minimised", hr_text(h));
                    ShowWindow(hwnd_h, SW_RESTORE);
                    pump();
                    if (run_present(hsc, 0, 0, 5000, &h)) note("host Present after restore", hr_text(h));
                    hsc->Release();
                }
                DestroyWindow(hwnd_h);
            }
            ShowWindow(hwnd, SW_RESTORE);
            pump();
            HRESULT hr_restored = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &hr_restored))
                check(hr_restored == S_OK, "Present after restoring the window", hr_text(hr_restored).c_str());
            // The only thing that gets the chain back: ResizeBuffers.
            hr = sc1->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
            check(hr == S_OK, "ResizeBuffers(0,0,0) after the restore", hr_text(hr).c_str());
            HRESULT hr_recovered = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &hr_recovered))
                check(hr_recovered == S_OK, "Present after ResizeBuffers recovers", hr_text(hr_recovered).c_str());

            // ResizeBuffers while minimised: DXGI resizes the buffers even with no window to show them in.
            ShowWindow(hwnd, SW_MINIMIZE);
            pump();
            hr = sc1->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
            note("ours ResizeBuffers(0,0,0) while minimised", hr_text(hr));
            check(hr == S_OK, "ResizeBuffers(0,0,0) while minimised", hr_text(hr).c_str());
            HRESULT resize_explicit = sc1->ResizeBuffers(0, 320, 240, DXGI_FORMAT_UNKNOWN, 0);
            note("ours ResizeBuffers(0, 320x240) while minimised", hr_text(resize_explicit));
            DXGI_SWAP_CHAIN_DESC1 min_desc{};
            sc1->GetDesc1(&min_desc);
            note("ours GetDesc1 after the minimised resize",
                 std::to_string(min_desc.Width) + "x" + std::to_string(min_desc.Height));
            if (host.factory) {
                HWND hwnd_m = make_window(200, 150);
                pump();
                DXGI_SWAP_CHAIN_DESC1 hdesc = desc;
                IDXGISwapChain1* hsc = nullptr;
                if (SUCCEEDED(create_guarded(host.factory, host.queue, hwnd_m, &hdesc, &hsc)) && hsc) {
                    ShowWindow(hwnd_m, SW_MINIMIZE);
                    pump();
                    const HRESULT h0 = hsc->ResizeBuffers(0, 0, 0, DXGI_FORMAT_UNKNOWN, 0);
                    note("host ResizeBuffers(0,0,0) while minimised", hr_text(h0));
                    DXGI_SWAP_CHAIN_DESC1 hmin{};
                    hsc->GetDesc1(&hmin);
                    note("host GetDesc1 after the minimised resize",
                         std::to_string(hmin.Width) + "x" + std::to_string(hmin.Height));
                    const HRESULT h1 = hsc->ResizeBuffers(0, 320, 240, DXGI_FORMAT_UNKNOWN, 0);
                    note("host ResizeBuffers(0, 320x240) while minimised", hr_text(h1));
                    hsc->Release();
                }
                DestroyWindow(hwnd_m);
            }
            ShowWindow(hwnd, SW_RESTORE);
            pump();
            HRESULT hr_after_min_resize = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &hr_after_min_resize))
                check(hr_after_min_resize == S_OK, "Present after the minimised resize and restore",
                      hr_text(hr_after_min_resize).c_str());
            sc1->Release();
        }
        DestroyWindow(hwnd);
    }

    // ---------------------------------------------------------------- G: window resized, no ResizeBuffers
    {
        std::printf("\nG. the window is resized after creation, Present without ResizeBuffers\n");
        HWND hwnd = make_window(200, 150);
        pump();
        DXGI_SWAP_CHAIN_DESC1 desc = base_desc(200, 150, 2);
        IDXGISwapChain1* sc1 = nullptr;
        HRESULT hr = create_guarded(factory, queue, hwnd, &desc, &sc1);
        check(hr == S_OK && sc1, "create", hr_text(hr).c_str());
        if (SUCCEEDED(hr) && sc1) {
            HRESULT first = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &first)) check(first == S_OK, "Present before the resize", hr_text(first).c_str());
            SetWindowPos(hwnd, nullptr, 0, 0, 420, 320, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
            pump();
            Sleep(50);  // let the presentation engine see the new client area
            HRESULT after = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &after))
                check(after == S_OK, "Present after the window resize, no ResizeBuffers", hr_text(after).c_str());
            DXGI_SWAP_CHAIN_DESC1 got{};
            sc1->GetDesc1(&got);
            note("GetDesc1 after the resize (DXGI keeps the back-buffer size)", std::to_string(got.Width) + "x" + std::to_string(got.Height));
            HRESULT again = E_PENDING;
            if (run_present(sc1, 0, 0, 5000, &again)) note("second Present after the resize", hr_text(again));

            if (host.factory) {
                HWND hwnd_h = make_window(200, 150);
                pump();
                DXGI_SWAP_CHAIN_DESC1 hdesc = desc;
                IDXGISwapChain1* hsc = nullptr;
                if (SUCCEEDED(create_guarded(host.factory, host.queue, hwnd_h, &hdesc, &hsc)) && hsc) {
                    HRESULT h = E_PENDING;
                    if (run_present(hsc, 0, 0, 5000, &h)) note("host Present before the resize", hr_text(h));
                    SetWindowPos(hwnd_h, nullptr, 0, 0, 420, 320, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
                    pump();
                    Sleep(50);
                    if (run_present(hsc, 0, 0, 5000, &h)) note("host Present after the resize, no ResizeBuffers", hr_text(h));
                    hsc->Release();
                }
                DestroyWindow(hwnd_h);
            }
            sc1->Release();
        }
        DestroyWindow(hwnd);
    }

    rtv_heap->Release();
    fence->Release();
    cl->Release();
    alloc->Release();
    queue->Release();
    factory->Release();
    device->Release();
    if (host.queue) host.queue->Release();
    if (host.factory) host.factory->Release();
    if (host.device) host.device->Release();
    wait_idle(vk_dev);
    destroy_dev(vk_dev, nullptr);
    if (messenger && destroy_dbg) destroy_dbg(instance, messenger, nullptr);
    destroy_inst(instance, nullptr);

    std::printf("\npresent_risk_test: %d checks, %d failed, %u validation errors\n", g_checks, g_failures,
                validation_errors());
    return (g_failures == 0 && validation_errors() == 0) ? 0 : 1;
}
