// ID3D12GraphicsCommandList4::BeginRenderPass / EndRenderPass (M3). They were unimplemented stubs: the pass's clears
// and resolves were lost and its draws went to whatever targets were bound. Now BeginRenderPass binds the targets and
// clears those whose beginning access is CLEAR; EndRenderPass resolves those whose ending access is RESOLVE.
//
// On the layer's D3D12 device over a real Vulkan device, read back through CopyTextureRegion:
//   1. a 4x4 RGBA8 target, beginning access CLEAR to (1, 0.5, 0, 1), ending PRESERVE: every texel is the clear colour;
//   2. a 4x-multisampled RGBA8 target cleared by the pass and resolved by its ending access (RESOLVE, AVERAGE) into a
//      single-sample texture: every resolved texel is the clear colour;
//   3. a D32 depth buffer, depth beginning access CLEAR to 0.25: every texel reads 0.25;
//   4. EndRenderPass without BeginRenderPass, and Close inside an open pass, make Close fail (E_INVALIDARG).
// No game data or shaders. Skips (77) without a Vulkan 1.3 device.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Rig {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList4> list;
    Ref<ID3D12Fence> fence;
    UINT64 value = 0;
};

ID3D12Resource* texture(ID3D12Device* device, DXGI_FORMAT format, UINT samples, D3D12_RESOURCE_FLAGS flags,
                        D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = 4;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = samples;
    desc.Flags = flags;
    ID3D12Resource* resource = nullptr;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                               IID_PPV_ARGS(&resource))))
        return nullptr;
    return resource;
}

// Copies subresource 0 of `image` (in COPY_SOURCE) to a readback buffer, runs the list and returns the texels.
bool read_back(ID3D12Device* device, Rig& rig, ID3D12Resource* image, unsigned bytes, std::vector<unsigned char>& out) {
    const D3D12_RESOURCE_DESC desc = image->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &placed, nullptr, nullptr, &total);
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = buffer.SampleDesc.Count = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> readback;
    if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_COPY_DEST,
                                               nullptr, IID_PPV_ARGS(&readback.p))))
        return false;
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = image;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.p;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = placed;
    rig.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (FAILED(rig.list->Close())) return false;
    ID3D12CommandList* lists[] = {rig.list.p};
    rig.queue->ExecuteCommandLists(1, lists);
    if (FAILED(rig.queue->Signal(rig.fence.p, ++rig.value))) return false;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    const bool done = event && SUCCEEDED(rig.fence->SetEventOnCompletion(rig.value, event)) &&
                      WaitForSingleObject(event, 5000) == WAIT_OBJECT_0;
    if (event) CloseHandle(event);
    if (!done) return false;
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    if (FAILED(readback->Map(0, &range, &mapped)) || !mapped) return false;
    out.clear();
    for (UINT row = 0; row < placed.Footprint.Height; ++row) {
        const auto* start = static_cast<unsigned char*>(mapped) + placed.Offset + row * placed.Footprint.RowPitch;
        out.insert(out.end(), start, start + placed.Footprint.Width * bytes);
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return SUCCEEDED(rig.allocator->Reset()) && SUCCEEDED(rig.list->Reset(rig.allocator.p, nullptr));
}

void transition(Rig& rig, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    rig.list->ResourceBarrier(1, &barrier);
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;
    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        Ref<ID3D12Device> device;
        CHECK(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        Rig rig;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&rig.queue.p))));
        CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&rig.allocator.p))));
        CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, rig.allocator.p, nullptr,
                                                 IID_PPV_ARGS(&rig.list.p))));
        CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&rig.fence.p))));

        Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
        const D3D12_DESCRIPTOR_HEAP_DESC rtv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC dsv_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&rtv_desc, IID_PPV_ARGS(&rtv_heap.p))));
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&dsv_desc, IID_PPV_ARGS(&dsv_heap.p))));
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE ms_rtv = rtv;
        ms_rtv.ptr += device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        const D3D12_CPU_DESCRIPTOR_HANDLE dsv = dsv_heap->GetCPUDescriptorHandleForHeapStart();
        const unsigned char want[4] = {255, 128, 0, 255};  // (1, 0.5, 0, 1) in RGBA8, rounded

        // 1. A cleared, preserved target.
        {
            Ref<ID3D12Resource> target;
            target.p = texture(device.p, DXGI_FORMAT_R8G8B8A8_UNORM, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                               D3D12_RESOURCE_STATE_RENDER_TARGET);
            CHECK(target.p);
            device->CreateRenderTargetView(target.p, nullptr, rtv);
            D3D12_RENDER_PASS_RENDER_TARGET_DESC pass{};
            pass.cpuDescriptor = rtv;
            pass.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
            pass.BeginningAccess.Clear.ClearValue = {DXGI_FORMAT_R8G8B8A8_UNORM, {1.0f, 0.5f, 0.0f, 1.0f}};
            pass.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
            rig.list->BeginRenderPass(1, &pass, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
            rig.list->EndRenderPass();
            transition(rig, target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            std::vector<unsigned char> texels;
            CHECK(read_back(device.p, rig, target.p, 4, texels));
            for (std::size_t i = 0; i < texels.size(); i += 4) CHECK(std::memcmp(&texels[i], want, 4) == 0);
            std::printf("1 a render pass clears its target (%zu texels)\n", texels.size() / 4);
        }

        // 2. Cleared multisampled, resolved at the end of the pass.
        {
            Ref<ID3D12Resource> ms, resolved;
            ms.p = texture(device.p, DXGI_FORMAT_R8G8B8A8_UNORM, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                           D3D12_RESOURCE_STATE_RENDER_TARGET);
            resolved.p = texture(device.p, DXGI_FORMAT_R8G8B8A8_UNORM, 1, D3D12_RESOURCE_FLAG_NONE,
                                 D3D12_RESOURCE_STATE_RESOLVE_DEST);
            CHECK(ms.p && resolved.p);
            device->CreateRenderTargetView(ms.p, nullptr, ms_rtv);
            D3D12_RENDER_PASS_ENDING_ACCESS_RESOLVE_SUBRESOURCE_PARAMETERS sub{0, 0, 0, 0, {0, 0, 4, 4}};
            D3D12_RENDER_PASS_RENDER_TARGET_DESC pass{};
            pass.cpuDescriptor = ms_rtv;
            pass.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
            pass.BeginningAccess.Clear.ClearValue = {DXGI_FORMAT_R8G8B8A8_UNORM, {1.0f, 0.5f, 0.0f, 1.0f}};
            pass.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_RESOLVE;
            pass.EndingAccess.Resolve.pSrcResource = ms.p;
            pass.EndingAccess.Resolve.pDstResource = resolved.p;
            pass.EndingAccess.Resolve.SubresourceCount = 1;
            pass.EndingAccess.Resolve.pSubresourceParameters = &sub;
            pass.EndingAccess.Resolve.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            pass.EndingAccess.Resolve.ResolveMode = D3D12_RESOLVE_MODE_AVERAGE;
            rig.list->BeginRenderPass(1, &pass, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
            // D3D12 resolves with the source in RENDER_TARGET at EndRenderPass; the layer's resolve tracks layouts.
            rig.list->EndRenderPass();
            transition(rig, resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            std::vector<unsigned char> texels;
            CHECK(read_back(device.p, rig, resolved.p, 4, texels));
            for (std::size_t i = 0; i < texels.size(); i += 4) CHECK(std::memcmp(&texels[i], want, 4) == 0);
            std::printf("2 a render pass resolves its multisampled target at its end\n");
        }

        // 3. A cleared depth buffer.
        {
            Ref<ID3D12Resource> depth;
            depth.p = texture(device.p, DXGI_FORMAT_D32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                              D3D12_RESOURCE_STATE_DEPTH_WRITE);
            CHECK(depth.p);
            device->CreateDepthStencilView(depth.p, nullptr, dsv);
            D3D12_RENDER_PASS_DEPTH_STENCIL_DESC pass{};
            pass.cpuDescriptor = dsv;
            pass.DepthBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_CLEAR;
            pass.DepthBeginningAccess.Clear.ClearValue.Format = DXGI_FORMAT_D32_FLOAT;
            pass.DepthBeginningAccess.Clear.ClearValue.DepthStencil = {0.25f, 0};
            pass.StencilBeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_NO_ACCESS;
            pass.DepthEndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
            pass.StencilEndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_NO_ACCESS;
            rig.list->BeginRenderPass(0, nullptr, &pass, D3D12_RENDER_PASS_FLAG_NONE);
            rig.list->EndRenderPass();
            transition(rig, depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
            std::vector<unsigned char> texels;
            CHECK(read_back(device.p, rig, depth.p, 4, texels));
            for (std::size_t i = 0; i < texels.size(); i += 4) {
                float d = 0;
                std::memcpy(&d, &texels[i], 4);
                CHECK(d == 0.25f);
            }
            std::printf("3 a render pass clears its depth buffer\n");
        }

        // 4. Misuse fails Close.
        rig.list->EndRenderPass();
        CHECK(rig.list->Close() == E_INVALIDARG);
        CHECK(SUCCEEDED(rig.allocator->Reset()) && SUCCEEDED(rig.list->Reset(rig.allocator.p, nullptr)));
        D3D12_RENDER_PASS_RENDER_TARGET_DESC open{};
        open.cpuDescriptor = rtv;
        open.BeginningAccess.Type = D3D12_RENDER_PASS_BEGINNING_ACCESS_TYPE_PRESERVE;
        open.EndingAccess.Type = D3D12_RENDER_PASS_ENDING_ACCESS_TYPE_PRESERVE;
        rig.list->BeginRenderPass(1, &open, nullptr, D3D12_RENDER_PASS_FLAG_NONE);
        CHECK(rig.list->Close() == E_INVALIDARG);
        std::printf("4 EndRenderPass without a pass, and Close inside one, fail Close\n");
    }
    if (vk.validation_errors()) {
        std::fprintf(stderr, "validation errors: %u\n", vk.validation_errors());
        return 1;
    }
    std::printf("d3d12_render_pass_test: OK\n");
    return 0;
}
