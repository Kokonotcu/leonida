// GetGPUVirtualAddress of buffers through the Xbox-ABI device proxy on the Vulkan device. A title builds root
// descriptors and V#s from this address, so a buffer that answers 0 sends the GPU to address 0. Every 1 KiB buffer
// must answer a non-zero, aligned address that no other live buffer shares, whatever its heap and initial state.
// The test makes the device three times: with bufferDeviceAddress and its arena (the device the game gets); without
// bufferDeviceAddress, where there is no arena; and with the arena refused while the driver answers addresses above
// 48 bits and not 64 KiB aligned. Without an arena the address must be 64 KiB aligned and below 2^48, and a V#
// built from it (a UAV clear at byte 256, a WriteBufferImmediate at address + 4) must still find the buffer.
// No game data or shaders.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_xs_device.h"

#include <cstdint>
#include <utility>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

constexpr UINT64 kBytes = 1024;
constexpr UINT64 kMinimumAlignment = 256;  // what a V# or a root descriptor may assume of a buffer's address

struct Case {
    const char* name;
    D3D12_HEAP_TYPE heap;
    D3D12_RESOURCE_STATES state;
    D3D12_RESOURCE_FLAGS flags;
};

constexpr Case kCases[] = {
    {"default COMMON", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_FLAG_NONE},
    {"default COPY_DEST", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE},
    {"default COPY_SOURCE", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_FLAG_NONE},
    {"default GENERIC_READ", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE},
    {"default UAV", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS},
    {"default COMMON + UAV flag", D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS},
    {"upload GENERIC_READ", D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE},
    {"readback COPY_DEST", D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE},
};

enum class Mode { arena, no_device_address, arena_refused };

// Mode::arena_refused: vkCreateBuffer fails for the GPU address arena (16 MiB and up), and vkGetBufferDeviceAddress
// answers what a driver may: above 48 bits, 256 apart. Nothing here submits, so no GPU work reads those addresses.
PFN_vkGetInstanceProcAddr g_real_instance = nullptr;
PFN_vkGetDeviceProcAddr g_real_device = nullptr;
PFN_vkCreateBuffer g_real_create_buffer = nullptr;
int g_arenas_refused = 0;
std::uint64_t g_next_raw = (std::uint64_t(1) << 52) + 0x100100;
int g_raw_addresses = 0;

VKAPI_ATTR VkResult VKAPI_CALL refusing_create_buffer(VkDevice device, const VkBufferCreateInfo* info,
                                                      const VkAllocationCallbacks* allocator, VkBuffer* buffer) {
    if (info->size >= (VkDeviceSize(16) << 20)) {
        ++g_arenas_refused;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    return g_real_create_buffer(device, info, allocator, buffer);
}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL raw_buffer_address(VkDevice, const VkBufferDeviceAddressInfo*) {
    ++g_raw_addresses;
    return std::exchange(g_next_raw, g_next_raw + 0x2000);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL hooked_device_proc(VkDevice device, const char* name) {
    if (!std::strcmp(name, "vkCreateBuffer")) {
        g_real_create_buffer = reinterpret_cast<PFN_vkCreateBuffer>(g_real_device(device, name));
        return reinterpret_cast<PFN_vkVoidFunction>(refusing_create_buffer);
    }
    if (!std::strcmp(name, "vkGetBufferDeviceAddress") || !std::strcmp(name, "vkGetBufferDeviceAddressKHR"))
        return reinterpret_cast<PFN_vkVoidFunction>(raw_buffer_address);
    return g_real_device(device, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL hooked_instance_proc(VkInstance instance, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) {
        g_real_device = reinterpret_cast<PFN_vkGetDeviceProcAddr>(g_real_instance(instance, name));
        return reinterpret_cast<PFN_vkVoidFunction>(hooked_device_proc);
    }
    return g_real_instance(instance, name);
}

// Records a clear of a view at byte 256 of `buffer` and a write at its address + 4: both find the buffer by an
// address that a V# or a root parameter carries, so the list must close cleanly.
bool addresses_resolve(ID3D12Device* device, ID3D12Resource* buffer) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R32_UINT;
    view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    view.Buffer.FirstElement = kMinimumAlignment / 4;
    view.Buffer.NumElements = 16;
    // An Xbox-ABI descriptor handle is the address of the 32-byte slot the view is written to.
    alignas(32) std::uint32_t slot[8] = {};
    const D3D12_CPU_DESCRIPTOR_HANDLE cpu{reinterpret_cast<SIZE_T>(slot)};
    device->CreateUnorderedAccessView(buffer, nullptr, &view, cpu);
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))))
        return false;
    const UINT values[4] = {1, 2, 3, 4};
    list->ClearUnorderedAccessViewUint({}, cpu, buffer, values, 0, nullptr);
    Ref<ID3D12GraphicsCommandList2> list2;
    if (FAILED(list->QueryInterface(IID_PPV_ARGS(&list2.p)))) return false;
    const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER write{buffer->GetGPUVirtualAddress() + 4, 7};
    list2->WriteBufferImmediate(1, &write, nullptr);
    const HRESULT closed = list->Close();
    if (FAILED(closed)) std::fprintf(stderr, "Close of a list using the buffer's address: 0x%lx\n", (unsigned long)closed);
    return SUCCEEDED(closed);
}

int run(vktest::Device& d, Mode mode) {
    const bool addressable = mode == Mode::arena;
    const char* label = mode == Mode::arena ? "arena" : mode == Mode::arena_refused ? "raw" : "no bda";
    rt::d3d12::VulkanDeviceInfo info{};
    if (mode == Mode::arena_refused) {
        g_real_instance = d.instance_proc;
        info.get_instance_proc = hooked_instance_proc;
    } else {
        info.get_instance_proc = d.instance_proc;
    }
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = mode != Mode::no_device_address;
    ID3D12Device* inner = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                       reinterpret_cast<void**>(&device.p))));
    rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`

    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kBytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    std::vector<Ref<ID3D12Resource>> buffers(std::size(kCases));
    std::vector<UINT64> addresses;
    for (size_t i = 0; i < std::size(kCases); ++i) {
        const Case& c = kCases[i];
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = c.heap;
        desc.Flags = c.flags;
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, c.state, nullptr,
                                                         IID_PPV_ARGS(&buffers[i].p))));
        const UINT64 address = buffers[i]->GetGPUVirtualAddress();
        std::printf("%s: %s -> 0x%llx\n", label, c.name, (unsigned long long)address);
        if (!address) std::fprintf(stderr, "%s: GetGPUVirtualAddress is 0\n", c.name);
        EXPECT(address != 0);
        EXPECT(address % kMinimumAlignment == 0);
        EXPECT(addressable || address % D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT == 0);
        EXPECT(address + kBytes <= (UINT64(1) << 48));  // a V# keeps 48 bits of address
        for (const UINT64 other : addresses) EXPECT(address >= other + kBytes || other >= address + kBytes);
        addresses.push_back(address);
    }
    if (mode == Mode::arena_refused) {  // the hooks were really in the path, or the raw branch was never reached
        EXPECT(g_arenas_refused > 0);
        EXPECT(g_raw_addresses >= int(std::size(kCases)));
    }
    EXPECT(addresses_resolve(device.p, buffers[4].p));  // "default UAV"
    return 0;
}

}  // namespace

int main() {
    vktest::Device d;
    d.request_buffer_device_address = true;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (status != vktest::DeviceStatus::ready) return 1;
    if (!d.buffer_device_address) {
        std::printf("SKIP: the device has no bufferDeviceAddress\n");
        return 77;
    }
    if (const int result = run(d, Mode::arena)) return result;
    if (const int result = run(d, Mode::no_device_address)) return result;
    return run(d, Mode::arena_refused);
}
