// The retail executable's present path through the Xbox-ABI device and queue proxies: device slots 78/79/80
// (SetFrameIntervalX/ScheduleFrameEventX/WaitFrameEventX), queue slot 32 (PresentX) and device slot 41
// (CreateCommandSignature with the console signature IID). The test creates a real window and a Vulkan-backed
// device, registers the same kind of presentation factory d3d12_xs.cpp registers, presents a known texture through
// the proxy's queue slot 32, reads the swapchain image back and writes a PPM.
//
// Measured from game_scarlett_master.exe (1.0.2.1): the title creates a window class "sgaWindowClass" first, then
// the device, then calls [device+0x278]/[device+0x270] for frame pacing and [queue+0x100] with
// (plane_count, planes, params) for PresentX. The plane it builds is our XboxPresentPlaneParameters layout (0x48
// bytes: token +0, ResourceCount 1 at +8, ppResources at +0x10, source rect at +0x18, full output at +0x20,
// colour space at +0x28).
//
// Needs a Vulkan GPU with presentation and a window; exits 77 (skipped) otherwise. Set LEONIDA_VK_VALIDATION=1
// for the validation layer.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/xbox_present.h"
#include "../shims/xbox_present_types.h"

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

using rt::d3d12::XboxPresentParameters;
using rt::d3d12::XboxPresentPlaneParameters;

constexpr UINT kWidth = 32, kHeight = 24;
constexpr IID kXboxCommandSignatureIid = {
    0xc36a797c, 0xec80, 0x4f0a, {0x89, 0x85, 0xa7, 0xb2, 0x47, 0x50, 0x82, 0xd1}};

template <class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

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

HWND g_window = nullptr;
rt::d3d12::XboxPresentation* g_presentation = nullptr;

// The proxy hands the layer device and (optionally) an explicit window; the test owns the window and uses it.
class TestPresentation final : public rt::d3d12_xs::XboxPresentationSink {
public:
    explicit TestPresentation(std::unique_ptr<rt::d3d12::XboxPresentation> presentation)
        : presentation_(std::move(presentation)) {}

    HRESULT SetFrameIntervalX(std::uint64_t output, std::uint32_t interval_us, std::uint32_t latency,
                              std::uint32_t flags) override {
        return presentation_->SetFrameIntervalX(reinterpret_cast<IUnknown*>(static_cast<std::uintptr_t>(output)),
                                                interval_us, latency, flags);
    }
    HRESULT ScheduleFrameEventX(std::uint32_t event, std::uint32_t offset_us, std::uint32_t flags) override {
        return presentation_->ScheduleFrameEventX(event, offset_us, flags);
    }
    HRESULT WaitFrameEventX(std::uint32_t event, std::uint32_t timeout_ms, std::uint32_t flags,
                            std::uint64_t* token) override {
        return presentation_->WaitFrameEventX(event, timeout_ms, flags, token);
    }
    HRESULT PresentX(std::uint32_t plane_count, const XboxPresentPlaneParameters* planes,
                     const XboxPresentParameters* params) override {
        return presentation_->PresentX(plane_count, planes, params);
    }

private:
    std::unique_ptr<rt::d3d12::XboxPresentation> presentation_;
};

rt::d3d12_xs::XboxPresentationSink* test_factory(ID3D12Device* real, void* device_proxy, std::uint64_t) {
    auto presenter = rt::d3d12::create_window_presenter(real, g_window);
    if (!presenter) return nullptr;
    presenter->enable_capture();
    // Share the device's frame tokens, as the retail factory does, so a device-side WaitFrameEventX token presents.
    auto presentation = std::make_unique<rt::d3d12::XboxPresentation>(
        std::move(presenter), nullptr, rt::d3d12_xs::device_frame_tokens(device_proxy));
    g_presentation = presentation.get();
    return new TestPresentation(std::move(presentation));
}

using SlotFn = std::uint64_t(WINAPI*)(void*, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t,
                                      std::uint64_t);

std::uint64_t call_slot(void* object, unsigned slot, std::uint64_t a1 = 0, std::uint64_t a2 = 0,
                        std::uint64_t a3 = 0, std::uint64_t a4 = 0, std::uint64_t a5 = 0) {
    void** vtable = *reinterpret_cast<void***>(object);
    return reinterpret_cast<SlotFn>(vtable[slot])(object, a1, a2, a3, a4, a5);
}

HRESULT as_hr(std::uint64_t value) { return static_cast<HRESULT>(static_cast<std::uint32_t>(value)); }

bool write_ppm(const char* path, const std::vector<std::uint8_t>& bytes, VkExtent2D extent, VkFormat format) {
    const bool bgr = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << "P6\n" << extent.width << ' ' << extent.height << "\n255\n";
    for (UINT y = 0; y < extent.height; ++y) {
        const std::uint8_t* row = bytes.data() + std::size_t(y) * extent.width * 4;
        for (UINT x = 0; x < extent.width; ++x) {
            const std::uint8_t pixel[3] = {row[x * 4 + (bgr ? 2u : 0u)], row[x * 4 + 1], row[x * 4 + (bgr ? 0u : 2u)]};
            out.write(reinterpret_cast<const char*>(pixel), 3);
        }
    }
    return out.good();
}

bool make_texture(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12CommandAllocator* allocator,
                  ID3D12GraphicsCommandList* list, ID3D12Fence* fence, std::uint64_t& signalled,
                  Ref<ID3D12Resource>& texture, Ref<ID3D12Resource>& upload) {
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = kWidth;
    desc.Height = kHeight;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_ALLOW_DISPLAY, &desc,
                                                D3D12_RESOURCE_STATE_PRESENT, nullptr, IID_PPV_ARGS(&texture.p))))
        return false;

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total_bytes = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total_bytes);
    if (total_bytes == 0 || footprint.Footprint.RowPitch < kWidth * 4) return false;

    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC upload_desc{};
    upload_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    upload_desc.Width = total_bytes;
    upload_desc.Height = 1;
    upload_desc.DepthOrArraySize = 1;
    upload_desc.MipLevels = 1;
    upload_desc.Format = DXGI_FORMAT_UNKNOWN;
    upload_desc.SampleDesc.Count = 1;
    upload_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (FAILED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &upload_desc,
                                                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                IID_PPV_ARGS(&upload.p))))
        return false;

    void* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, &mapped)) || !mapped) return false;
    auto* bytes = static_cast<std::uint8_t*>(mapped);
    for (UINT y = 0; y < kHeight; ++y) {
        std::uint8_t* row = bytes + std::size_t(y) * footprint.Footprint.RowPitch;
        for (UINT x = 0; x < kWidth; ++x) {
            // Known colour: B=30, G=160, R=220, A=255 (orange).
            row[x * 4 + 0] = 30;
            row[x * 4 + 1] = 160;
            row[x * 4 + 2] = 220;
            row[x * 4 + 3] = 255;
        }
    }
    upload->Unmap(0, nullptr);

    if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator, nullptr))) return false;
    D3D12_RESOURCE_BARRIER to_copy{};
    to_copy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    to_copy.Transition = {texture.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_PRESENT,
                          D3D12_RESOURCE_STATE_COPY_DEST};
    list->ResourceBarrier(1, &to_copy);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = texture.p;
    destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = upload.p;
    source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    source.PlacedFootprint = footprint;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    D3D12_RESOURCE_BARRIER to_present = to_copy;
    to_present.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    to_present.Transition.StateAfter = D3D12_RESOURCE_STATE_PRESENT;
    list->ResourceBarrier(1, &to_present);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list};
    queue->ExecuteCommandLists(1, lists);
    ++signalled;
    // The queue comes from device slot 48, so it is called in the console's slot layout (docs/d3d12_command_objects.md):
    // Signal is its slot 16, as the retail title signals; PC slot 14 is refused there.
    using SignalFn = HRESULT(WINAPI*)(ID3D12CommandQueue*, ID3D12Fence*, UINT64);
    const auto console_signal = reinterpret_cast<SignalFn>((*reinterpret_cast<void***>(queue))[16]);
    return SUCCEEDED(console_signal(queue, fence, signalled)) &&
           SUCCEEDED(fence->SetEventOnCompletion(signalled, nullptr));
}

}  // namespace

int main() {
    // The host d3d12.dll stays loaded for the whole process, as it does in the runtime (d3d12_xs.cpp never frees it).
    // This test does not import it, and a driver that presents a windowed Vulkan swapchain through DXGI (NVIDIA) loads
    // it for the swapchain and frees it at teardown while D3D12Core.dll's worker threads still read it: an access
    // violation in D3D12Core.dll on an address of the unloaded d3d12.dll, in roughly one run in four on an RTX 3080
    // Laptop with driver 617.42 (Windows 10). Holding a reference of our own keeps it mapped.
    LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);

    vktest::Device vk;
    if (!vk.create(true)) {
        std::printf("presentx_window: no Vulkan device with presentation: skipped\n");
        return 77;
    }

    WNDCLASSW window_class{};
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = GetModuleHandleW(nullptr);
    // The measured retail class name: a title window with it must still be attachable when the factory discovers it.
    window_class.lpszClassName = L"sgaWindowClass";
    window_class.hCursor = LoadCursorW(nullptr, reinterpret_cast<LPCWSTR>(IDC_ARROW));
    EXPECT(RegisterClassW(&window_class));
    RECT frame_rect{0, 0, int(kWidth), int(kHeight)};
    AdjustWindowRect(&frame_rect, WS_OVERLAPPEDWINDOW, FALSE);
    g_window = CreateWindowExW(0, window_class.lpszClassName, L"Leonida: retail PresentX path", WS_OVERLAPPEDWINDOW,
                               80, 80, frame_rect.right - frame_rect.left, frame_rect.bottom - frame_rect.top,
                               nullptr, nullptr, window_class.hInstance, nullptr);
    EXPECT(g_window);
    ShowWindow(g_window, SW_SHOWNOACTIVATE);
    UpdateWindow(g_window);
    pump();

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    info.presentation = true;
    info.swapchain_maintenance1 = vk.swapchain_maintenance1;
    // create_device_proxy takes the layer device's only reference: the proxy releases it with its own last
    // reference, so `inner` is a borrowed pointer and is never released here.
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);

    rt::d3d12_xs::set_xbox_presentation_factory(&test_factory);
    rt::d3d12_xs::set_device_presentation_window(device.p, reinterpret_cast<std::uint64_t>(g_window));

    // Queue through the retail's slot 48 (CreateCommandQueueX), so the queue is the proxy the title presents on.
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    void* created_queue = nullptr;
    EXPECT(SUCCEEDED(as_hr(call_slot(device.p, 48, reinterpret_cast<std::uint64_t>(&queue_desc),
                                     reinterpret_cast<std::uint64_t>(&__uuidof(ID3D12CommandQueue)),
                                     reinterpret_cast<std::uint64_t>(&created_queue)))));
    EXPECT(created_queue != nullptr);
    queue.p = static_cast<ID3D12CommandQueue*>(created_queue);

    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(list->Close()));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    Ref<ID3D12Resource> texture;
    Ref<ID3D12Resource> upload;
    std::uint64_t signalled = 0;
    EXPECT(make_texture(device.p, queue.p, allocator.p, list.p, fence.p, signalled, texture, upload));

    // Device slot 41: the measured CreateCommandSignature with the console signature IID. The five retail calls are
    // DRAW stride 16, DRAW_INDEXED stride 20, and three DISPATCH-shaped signatures; one DRAW signature is representative.
    D3D12_INDIRECT_ARGUMENT_DESC argument{};
    argument.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DRAW;
    D3D12_COMMAND_SIGNATURE_DESC signature_desc{};
    signature_desc.ByteStride = 16;
    signature_desc.NumArgumentDescs = 1;
    signature_desc.pArgumentDescs = &argument;
    void* signature = nullptr;
    EXPECT(SUCCEEDED(as_hr(call_slot(device.p, 41, reinterpret_cast<std::uint64_t>(&signature_desc), 0,
                                     reinterpret_cast<std::uint64_t>(&kXboxCommandSignatureIid),
                                     reinterpret_cast<std::uint64_t>(&signature)))));
    EXPECT(signature != nullptr);
    static_cast<ID3D12CommandSignature*>(signature)->Release();

    // Frame pacing through the retail device slots. The first call creates the presenter (and its swapchain).
    EXPECT(SUCCEEDED(as_hr(call_slot(device.p, 78, 0, 16667, 1, 0))));  // SetFrameIntervalX(0, 16667, 1, 0)
    EXPECT(SUCCEEDED(as_hr(call_slot(device.p, 79, 0, 0, 0, 0))));      // ScheduleFrameEventX(origin, 0, 0, 0)
    std::uint64_t token = 0;
    // WaitFrameEventX(event=0, timeout_ms=100, reserved=0, flags=0, &token): the token is the fifth argument.
    EXPECT(SUCCEEDED(as_hr(call_slot(device.p, 80, 0, 100, 0, 0, reinterpret_cast<std::uint64_t>(&token)))));
    EXPECT(token != 0);
    EXPECT(g_presentation != nullptr);

    XboxPresentPlaneParameters plane{};
    plane.Token = token;
    plane.ResourceCount = 1;
    plane.ppResources = &texture.p;
    plane.ColorSpace = rt::d3d12::xbox_color_space_g22_p709;
    const HRESULT present = as_hr(call_slot(queue.p, 32, 1, reinterpret_cast<std::uint64_t>(&plane), 0, 0));
    EXPECT(present == S_OK);
    EXPECT(g_presentation->presents() == 1);

    std::vector<std::uint8_t> pixels;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    EXPECT(g_presentation->presenter().read_back(pixels, &extent, &format));
    EXPECT(!pixels.empty());
    EXPECT(extent.width > 0 && extent.height > 0);
    // The 32x24 source is a solid orange, stretched to the swapchain, so every captured pixel is that colour.
    const bool bgr = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    const std::uint8_t* first = pixels.data();
    const int red = first[bgr ? 2 : 0], green = first[1], blue = first[bgr ? 0 : 2];
    EXPECT(red >= 217 && red <= 223 && green >= 157 && green <= 163 && blue >= 27 && blue <= 33);
    char path[MAX_PATH] = {};
    EXPECT(GetTempPathA(MAX_PATH, path) != 0);
    std::string ppm = std::string(path) + "leonida_presentx_window.ppm";
    EXPECT(write_ppm(ppm.c_str(), pixels, extent, format));
    std::ifstream check(ppm, std::ios::binary);
    EXPECT(check.good());
    std::printf("presentx_window: device 78/79/80 + queue 32 present S_OK, %ux%u PPM written; slot 41 signature S_OK\n",
                extent.width, extent.height);
    DeleteFileA(ppm.c_str());

    // Release the queue and its children before the device proxy (and the presenter it owns).
    fence.reset();
    list.reset();
    allocator.reset();
    upload.reset();
    texture.reset();
    queue.reset();
    g_presentation = nullptr;
    device.reset();  // releases the presenter, then the layer device (`inner`)
    DestroyWindow(g_window);
    return 0;
}
