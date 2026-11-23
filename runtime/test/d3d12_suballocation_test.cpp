// Committed resources' memory on the Vulkan-backed device (src/vulkan_suballocator.h, DeviceState::allocate_resource),
// measured: 6000 small textures, 2000 small upload buffers and 256 CPU-visible textures, with suballocation (the
// default) and with LEONIDA_SUBALLOCATE=0, counting the layer's vkAllocateMemory calls through a wrapped
// vkGetDeviceProcAddr. The number of live allocations is held to the device's maxMemoryAllocationCount, and to no
// more than 4096 (what AMD's and older NVIDIA Windows drivers report), as a driver may hold it
// (VK_ERROR_TOO_MANY_OBJECTS): one allocation per resource runs into it, suballocation must not. Each upload buffer and CPU-visible texture keeps a pattern of its own, so ranges of a block
// that overlapped would show, and every allocation must be freed by the time the device is released. Skips (77)
// without a Vulkan GPU. No shaders or game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

constexpr int kTextures = 6000;
constexpr int kUploads = 2000;
constexpr int kCpuTextures = 256;
constexpr UINT kCpuSide = 32;
constexpr std::uint32_t kSpecMinimumAllocations = 4096;

struct Counter {
    PFN_vkGetInstanceProcAddr instance_proc = nullptr;
    PFN_vkGetDeviceProcAddr device_proc = nullptr;
    PFN_vkAllocateMemory allocate = nullptr;
    PFN_vkFreeMemory free = nullptr;
    std::atomic<std::uint32_t> calls{0};
    std::atomic<std::uint32_t> live{0};
    std::atomic<std::uint32_t> peak{0};
    std::atomic<std::uint32_t> refused{0};
    std::uint32_t limit = UINT32_MAX;
} counter;

VKAPI_ATTR VkResult VKAPI_CALL counted_allocate(VkDevice device, const VkMemoryAllocateInfo* info,
                                                const VkAllocationCallbacks* callbacks, VkDeviceMemory* memory) {
    ++counter.calls;
    if (counter.live.load() >= counter.limit) {
        ++counter.refused;
        return VK_ERROR_TOO_MANY_OBJECTS;
    }
    const VkResult result = counter.allocate(device, info, callbacks, memory);
    if (result == VK_SUCCESS) {
        const std::uint32_t now = ++counter.live;
        std::uint32_t peak = counter.peak.load();
        while (now > peak && !counter.peak.compare_exchange_weak(peak, now)) {}
    }
    return result;
}

VKAPI_ATTR void VKAPI_CALL counted_free(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* callbacks) {
    if (memory) --counter.live;
    counter.free(device, memory, callbacks);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (!std::strcmp(name, "vkAllocateMemory")) {
        counter.allocate = reinterpret_cast<PFN_vkAllocateMemory>(counter.device_proc(device, name));
        return reinterpret_cast<PFN_vkVoidFunction>(&counted_allocate);
    }
    if (!std::strcmp(name, "vkFreeMemory")) {
        counter.free = reinterpret_cast<PFN_vkFreeMemory>(counter.device_proc(device, name));
        return reinterpret_cast<PFN_vkVoidFunction>(&counted_free);
    }
    return counter.device_proc(device, name);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) {
        counter.device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(counter.instance_proc(instance, name));
        return reinterpret_cast<PFN_vkVoidFunction>(&device_proc);
    }
    return counter.instance_proc(instance, name);
}

D3D12_RESOURCE_DESC texture(UINT side) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = side;
    d.Height = side;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = 1;
    return d;
}

D3D12_RESOURCE_DESC buffer(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

struct Result {
    int textures = 0, uploads = 0, cpu_textures = 0;
    HRESULT first_failure = S_OK;
    const char* failed_at = nullptr;
    std::uint32_t device_calls = 0;    // vkAllocateMemory calls while the device was made
    std::uint32_t resource_calls = 0;  // and while the resources were
    std::uint32_t peak = 0, refused = 0, live_after = 0;
    bool patterns_ok = true;
    double create_ms = 0;
};

void fail(Result& r, const char* what, HRESULT hr) {
    if (r.failed_at) return;
    r.failed_at = what;
    r.first_failure = hr;
}

int run(vktest::Device& d, bool suballocate, Result& r) {
    _putenv_s("LEONIDA_SUBALLOCATE", suballocate ? "1" : "0");
    counter.calls = 0;
    counter.live = 0;
    counter.peak = 0;
    counter.refused = 0;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = &instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = true;  // upload buffers are addressable: a device-address, mapped pool
    ID3D12Device* device = nullptr;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))));
    r.device_calls = counter.calls.load();
    std::vector<ID3D12Resource*> made;
    made.reserve(kTextures + kUploads + kCpuTextures);
    const auto start = std::chrono::steady_clock::now();

    const D3D12_HEAP_PROPERTIES gpu{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC small_desc = texture(64);
    for (int i = 0; i < kTextures; ++i) {
        ID3D12Resource* t = nullptr;
        const HRESULT hr = device->CreateCommittedResource(&gpu, D3D12_HEAP_FLAG_NONE, &small_desc,
                                                           D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t));
        if (FAILED(hr)) {
            fail(r, "texture", hr);
            break;
        }
        made.push_back(t);
        ++r.textures;
    }

    const D3D12_HEAP_PROPERTIES upload{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_RESOURCE_DESC bytes = buffer(4096);
    std::vector<std::uint32_t*> mapped;
    for (int i = 0; i < kUploads; ++i) {
        ID3D12Resource* b = nullptr;
        HRESULT hr = device->CreateCommittedResource(&upload, D3D12_HEAP_FLAG_NONE, &bytes,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&b));
        if (FAILED(hr)) {
            fail(r, "upload buffer", hr);
            break;
        }
        made.push_back(b);
        void* data = nullptr;
        hr = b->Map(0, nullptr, &data);
        if (FAILED(hr) || !data) {
            fail(r, "upload Map", FAILED(hr) ? hr : E_POINTER);
            break;
        }
        auto* words = static_cast<std::uint32_t*>(data);
        for (int w = 0; w < 1024; ++w) words[w] = std::uint32_t(i) * 0x9E3779B1u + std::uint32_t(w);
        mapped.push_back(words);
        ++r.uploads;
    }
    for (std::size_t i = 0; i < mapped.size(); ++i)
        for (int w = 0; w < 1024; ++w)
            if (mapped[i][w] != std::uint32_t(i) * 0x9E3779B1u + std::uint32_t(w)) r.patterns_ok = false;

    const D3D12_HEAP_PROPERTIES write_back{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                                           D3D12_MEMORY_POOL_L0};
    const D3D12_RESOURCE_DESC cpu_desc = texture(kCpuSide);
    std::vector<ID3D12Resource*> cpu;
    std::vector<std::uint32_t> texels(kCpuSide * kCpuSide);
    for (int i = 0; i < kCpuTextures; ++i) {
        ID3D12Resource* t = nullptr;
        const HRESULT hr = device->CreateCommittedResource(&write_back, D3D12_HEAP_FLAG_NONE, &cpu_desc,
                                                           D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&t));
        if (FAILED(hr)) {
            fail(r, "CPU-visible texture", hr);
            break;
        }
        made.push_back(t);
        cpu.push_back(t);
        for (UINT k = 0; k < kCpuSide * kCpuSide; ++k) texels[k] = std::uint32_t(i) << 16 | k;
        if (FAILED(t->WriteToSubresource(0, nullptr, texels.data(), kCpuSide * 4, kCpuSide * kCpuSide * 4))) {
            fail(r, "WriteToSubresource", E_FAIL);
            break;
        }
        ++r.cpu_textures;
    }
    for (std::size_t i = 0; i < cpu.size() && int(i) < r.cpu_textures; ++i) {
        std::fill(texels.begin(), texels.end(), 0u);
        if (FAILED(cpu[i]->ReadFromSubresource(texels.data(), kCpuSide * 4, kCpuSide * kCpuSide * 4, 0, nullptr))) {
            r.patterns_ok = false;
            continue;
        }
        for (UINT k = 0; k < kCpuSide * kCpuSide; ++k)
            if (texels[k] != (std::uint32_t(i) << 16 | k)) r.patterns_ok = false;
    }
    r.create_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    r.resource_calls = counter.calls.load() - r.device_calls;
    r.peak = counter.peak.load();
    r.refused = counter.refused.load();
    for (ID3D12Resource* resource : made) resource->Release();
    device->Release();
    r.live_after = counter.live.load();
    return 0;
}

void print(const char* mode, const Result& r) {
    std::printf("%s: %d/%d textures, %d/%d upload buffers, %d/%d CPU-visible textures in %.0f ms; vkAllocateMemory: "
                "%u for the device, %u for the resources (%u refused at the limit), peak %u live, %u live after the "
                "device was released; patterns %s",
                mode, r.textures, kTextures, r.uploads, kUploads, r.cpu_textures, kCpuTextures, r.create_ms,
                r.device_calls, r.resource_calls, r.refused, r.peak, r.live_after, r.patterns_ok ? "kept" : "BROKEN");
    if (r.failed_at)
        std::printf("; first failure: %s, 0x%08lx", r.failed_at, static_cast<unsigned long>(r.first_failure));
    std::printf("\n");
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (status != vktest::DeviceStatus::ready) return 1;
    counter.instance_proc = d.instance_proc;
    if (const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            d.instance_proc(d.instance, "vkGetPhysicalDeviceProperties"))) {
        VkPhysicalDeviceProperties p{};
        properties(d.physical, &p);
        // At most 4096, the least the Vulkan spec guarantees and what AMD's and older NVIDIA Windows drivers report:
        // newer NVIDIA drivers report 4294967295, and a title has to run on the others too.
        counter.limit = std::min<std::uint32_t>(p.limits.maxMemoryAllocationCount, kSpecMinimumAllocations);
        std::printf("d3d12_suballocation: %s, maxMemoryAllocationCount %u, held to %u\n", p.deviceName,
                    p.limits.maxMemoryAllocationCount, counter.limit);
    }

    Result sub, own;
    EXPECT(run(d, true, sub) == 0);
    print("suballocated", sub);
    EXPECT(run(d, false, own) == 0);
    print("LEONIDA_SUBALLOCATE=0", own);

    // Suballocated: everything made, within a few dozen blocks, every byte kept, every allocation freed.
    EXPECT(!sub.failed_at);
    EXPECT(sub.patterns_ok);
    EXPECT(sub.refused == 0);
    EXPECT(sub.resource_calls < 100);
    EXPECT(sub.live_after == 0);
    // One allocation per resource: one call each for what it made, until the limit stops it.
    const int own_made = own.textures + own.uploads + own.cpu_textures;
    EXPECT(own.resource_calls >= std::uint32_t(own_made));
    EXPECT(own.patterns_ok);
    EXPECT(own.live_after == 0);
    if (std::uint64_t(kTextures + kUploads + kCpuTextures) + own.device_calls > counter.limit) EXPECT(own.failed_at);
    std::printf("d3d12_suballocation_test: OK\n");
    return 0;
}
