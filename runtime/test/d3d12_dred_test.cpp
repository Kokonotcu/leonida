// ID3D12DeviceRemovedExtendedData/1/2 on the Vulkan device (shims/d3d12_dred.h), against the host runtime on WARP
// and the first hardware adapter:
//   - QueryInterface answers the three interfaces with the device's identity, DRED enabled or not;
//   - DRED not enabled ("off", LEONIDA_DRED unset): every output is DXGI_ERROR_UNSUPPORTED, before and after
//     RemoveDevice;
//   - enabled ("on": LEONIDA_DRED=1 for the layer, ID3D12DeviceRemovedExtendedDataSettings FORCED_ON for the host):
//     DXGI_ERROR_NOT_CURRENTLY_AVAILABLE on a live device, a null argument included; after RemoveDevice S_OK with no
//     node and page-fault VA 0 (no GPU loss: nothing to report); GetDeviceState UNKNOWN throughout;
//   - behind the Xbox-ABI device proxy the interfaces are refused: they are not in the Xbox device chain the proxy
//     models, and the Xbox layout of DRED is not measured. Leonida's own code asks the layer's device;
//   - in "on", lists executed on the Vulkan device with VK_AMD_buffer_marker, under validation;
//   - the conversion of the layer's breadcrumbs and VK_EXT_device_fault addresses into D3D12 nodes and a VA, on
//     synthetic data (a real device loss is not triggered).
// argv[1] is "off" or "on"; the settings are per process, so ctest runs the two modes separately. 77 when no
// device at all ran. No game data.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_dred.h"
#include "../shims/d3d12_xs_device.h"

#include <d3d12.h>
#include <dxgi1_4.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "%s line %d: %s\n", label, __LINE__, #condition); return 1; } \
} while (0)

namespace {

template<class T>
struct Ref {
    T* p = nullptr;
    Ref() = default;
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

bool same_object(IUnknown* a, IUnknown* b) {
    Ref<IUnknown> ia, ib;
    a->QueryInterface(IID_PPV_ARGS(&ia.p));
    b->QueryInterface(IID_PPV_ARGS(&ib.p));
    return ia.p && ia.p == ib.p;
}

// Every output method of the three interfaces returns `expected`; with `null_too` also for a null argument.
int outputs_return(const char* label, ID3D12DeviceRemovedExtendedData2* dred, HRESULT expected, bool null_too) {
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT o0{};
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 o1{};
    D3D12_DRED_PAGE_FAULT_OUTPUT p0{};
    D3D12_DRED_PAGE_FAULT_OUTPUT1 p1{};
    D3D12_DRED_PAGE_FAULT_OUTPUT2 p2{};
    EXPECT(dred->GetAutoBreadcrumbsOutput(&o0) == expected);
    EXPECT(dred->GetAutoBreadcrumbsOutput1(&o1) == expected);
    EXPECT(dred->GetPageFaultAllocationOutput(&p0) == expected);
    EXPECT(dred->GetPageFaultAllocationOutput1(&p1) == expected);
    EXPECT(dred->GetPageFaultAllocationOutput2(&p2) == expected);
    if (expected == S_OK) {
        EXPECT(!o0.pHeadAutoBreadcrumbNode && !o1.pHeadAutoBreadcrumbNode);
        EXPECT(p0.PageFaultVA == 0 && !p0.pHeadExistingAllocationNode && !p0.pHeadRecentFreedAllocationNode);
        EXPECT(p1.PageFaultVA == 0 && !p1.pHeadExistingAllocationNode && !p1.pHeadRecentFreedAllocationNode);
        EXPECT(p2.PageFaultVA == 0 && !p2.pHeadExistingAllocationNode && !p2.pHeadRecentFreedAllocationNode);
    }
    if (null_too) {
        EXPECT(dred->GetAutoBreadcrumbsOutput(nullptr) == expected);
        EXPECT(dred->GetAutoBreadcrumbsOutput1(nullptr) == expected);
        EXPECT(dred->GetPageFaultAllocationOutput(nullptr) == expected);
        EXPECT(dred->GetPageFaultAllocationOutput1(nullptr) == expected);
        EXPECT(dred->GetPageFaultAllocationOutput2(nullptr) == expected);
    }
    EXPECT(dred->GetDeviceState() == D3D12_DRED_DEVICE_STATE_UNKNOWN);
    return 0;
}

// Two lists, one with an event, executed and waited for; a third held behind a Wait that is never satisfied.
int execute_lists(const char* label, ID3D12Device* device) {
    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 256;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> dst, src;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&dst.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&src.p))));
    Ref<ID3D12GraphicsCommandList> a, b;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&a.p))));
    a->CopyBufferRegion(dst.p, 0, src.p, 0, 16);
    EXPECT(SUCCEEDED(a->Close()));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&b.p))));
    b->BeginEvent(0, L"copy", sizeof(L"copy"));
    b->CopyBufferRegion(dst.p, 16, src.p, 16, 16);
    b->EndEvent();
    EXPECT(SUCCEEDED(b->Close()));
    ID3D12CommandList* lists[] = {a.p, b.p};
    queue->ExecuteCommandLists(2, lists);
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));  // blocks until the GPU is there
    return 0;
}

int check_device(const char* label, ID3D12Device* device, bool on, bool host) {
    Ref<ID3D12DeviceRemovedExtendedData> d0;
    Ref<ID3D12DeviceRemovedExtendedData1> d1;
    Ref<ID3D12DeviceRemovedExtendedData2> d2;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&d0.p))));
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&d1.p))));
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&d2.p))));
    EXPECT(same_object(device, d0.p) && same_object(device, d2.p));
    Ref<ID3D12Device> back;
    EXPECT(SUCCEEDED(d2->QueryInterface(IID_PPV_ARGS(&back.p))));
    Ref<ID3D12DeviceRemovedExtendedDataSettings> settings;  // a debug interface, not one of the device
    EXPECT(device->QueryInterface(IID_PPV_ARGS(&settings.p)) == E_NOINTERFACE);

    const HRESULT live = on ? DXGI_ERROR_NOT_CURRENTLY_AVAILABLE : DXGI_ERROR_UNSUPPORTED;
    if (outputs_return(label, d2.p, live, true)) return 1;
    if (execute_lists(label, device)) return 1;
    if (outputs_return(label, d2.p, live, true)) return 1;

    Ref<ID3D12Device5> device5;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device5.p))));
    device5->RemoveDevice();
    EXPECT(device->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED);
    // After the removal the host runtime dereferences a null argument, so it is only asked of the layer.
    if (outputs_return(label, d2.p, on ? S_OK : DXGI_ERROR_UNSUPPORTED, !on)) return 1;
    if (on && !host) {
        D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 o{};
        EXPECT(d1->GetAutoBreadcrumbsOutput1(nullptr) == E_INVALIDARG);
        EXPECT(d1->GetAutoBreadcrumbsOutput1(&o) == S_OK);
    }
    std::printf("%s: DRED %s: QueryInterface, outputs live, after ExecuteCommandLists and after RemoveDevice as on "
                "the host runtime\n", label, on ? "on" : "off");
    return 0;
}

int run_host(IDXGIAdapter1* adapter, const char* label, bool on) {
    Ref<ID3D12Device> device;
    if (FAILED(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device.p)))) {
        std::printf("%s: no D3D12 device, skipped\n", label);
        return 77;
    }
    return check_device(label, device.p, on, true);
}

// The layer's data after a loss, as D3D12 nodes and a VA.
int snapshot() {
    const char* label = "snapshot";
    using leonida::BreadcrumbState;
    namespace op = leonida::breadcrumb_op;
    std::vector<leonida::BreadcrumbRing::Entry> entries = {
        {1, 0x10, {op::copy_buffer_region}, BreadcrumbState::completed},
        {2, 0x20, {op::begin_event, op::draw_instanced, op::end_event}, BreadcrumbState::started},
        {3, 0x30, {}, BreadcrumbState::submitted},
    };
    entries[2].list_name = L"post";
    entries[2].queue_name = L"direct queue";
    constexpr VkDeviceAddress arena = 0x1'0000'0000;
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(3);
    addresses[0].addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_INSTRUCTION_POINTER_FAULT_EXT;  // not a data address
    addresses[0].reportedAddress = arena + 0x40;
    addresses[1].addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT;
    addresses[1].reportedAddress = arena + 0x12340;
    addresses[2].addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT;
    addresses[2].reportedAddress = arena + 0x99990;

    rt::d3d12::DredSnapshot s;
    s.build(entries, addresses, arena);
    const D3D12_AUTO_BREADCRUMB_NODE1* n = s.head1();
    EXPECT(n && n->BreadcrumbCount == 3);  // the completed list is left out
    EXPECT(n->pCommandHistory[0] == D3D12_AUTO_BREADCRUMB_OP_BEGINEVENT);
    EXPECT(n->pCommandHistory[1] == D3D12_AUTO_BREADCRUMB_OP_DRAWINSTANCED);
    EXPECT(n->pCommandHistory[2] == D3D12_AUTO_BREADCRUMB_OP_ENDEVENT);
    EXPECT(n->pLastBreadcrumbValue && *n->pLastBreadcrumbValue == 0);
    EXPECT(!n->pCommandList && !n->pCommandQueue && !n->pCommandListDebugNameW && !n->BreadcrumbContextsCount);
    n = n->pNext;
    EXPECT(n && n->BreadcrumbCount == 0 && !n->pCommandHistory && !n->pNext);
    EXPECT(n->pCommandListDebugNameW && std::wcscmp(n->pCommandListDebugNameW, L"post") == 0);
    EXPECT(n->pCommandListDebugNameA && std::strcmp(n->pCommandListDebugNameA, "post") == 0);
    EXPECT(n->pCommandQueueDebugNameW && std::wcscmp(n->pCommandQueueDebugNameW, L"direct queue") == 0);
    EXPECT(n->pCommandQueueDebugNameA && std::strcmp(n->pCommandQueueDebugNameA, "direct queue") == 0);
    EXPECT(!n->pCommandList && !n->pCommandQueue);  // never a COM pointer the layer does not hold
    EXPECT(s.head()->pNext->pCommandListDebugNameW == n->pCommandListDebugNameW);
    EXPECT(s.head() && s.head()->BreadcrumbCount == 3 && s.head()->pNext && !s.head()->pNext->pNext);
    EXPECT(s.fault_va() == 0x12340);  // the first invalid access, as a D3D12 VA

    addresses.resize(1);
    s.build(entries, addresses, arena);
    EXPECT(s.fault_va() == 0);  // only an instruction pointer
    addresses[0].addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT;
    addresses[0].reportedAddress = arena - 0x1000;
    s.build(entries, addresses, arena);
    EXPECT(s.fault_va() == 0);  // below the arena: no D3D12 VA
    s.build({entries[0]}, {}, 0);
    EXPECT(!s.head() && !s.head1() && s.fault_va() == 0);
    // GPU progress unknown (no VK_AMD_buffer_marker): no node claims a list in flight, the fault VA still counts.
    std::vector<leonida::BreadcrumbRing::Entry> unknown = entries;
    for (auto& e : unknown) e.state = BreadcrumbState::unknown;
    addresses.assign(1, {});
    addresses[0].addressType = VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT;
    addresses[0].reportedAddress = arena + 0x700;
    s.build(unknown, addresses, arena);
    EXPECT(!s.head() && !s.head1() && s.fault_va() == 0x700);
    EXPECT(D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION == op::copy_buffer_region &&
           D3D12_AUTO_BREADCRUMB_OP_BARRIER == op::barrier &&
           D3D12_AUTO_BREADCRUMB_OP_WRITEBUFFERIMMEDIATE == op::write_buffer_immediate &&
           D3D12_AUTO_BREADCRUMB_OP_RESOLVESUBRESOURCEREGION == op::resolve_subresource_region);
    // Per-operation progress (BUILD-1655, synthetic entries, not GPU-observed): BreadcrumbCount is the list's
    // operations, pCommandHistory their order, *pLastBreadcrumbValue the ones finished; a list whose progress is
    // not known (list markers only) claims none.
    std::vector<leonida::BreadcrumbRing::Entry> progress = {
        {4, 0x40, {op::copy_buffer_region, op::resource_barrier, op::dispatch, op::draw_indexed_instanced},
         BreadcrumbState::started},
        {5, 0x50, {op::clear_render_target_view, op::draw_instanced}, BreadcrumbState::submitted},
        {6, 0x60, {op::copy_resource, op::resolve_subresource}, BreadcrumbState::started},
    };
    progress[0].progress_known = true;
    progress[0].completed_ops = 2;  // two of four finished
    progress[1].progress_known = true;
    progress[1].completed_ops = 0;  // not started
    progress.push_back({7, 0x70, {}, BreadcrumbState::started});  // no operations, progress unknown
    leonida::BreadcrumbRing::Entry untracked{8, 0x80, {op::dispatch}, BreadcrumbState::unknown};
    untracked.no_slot = true;  // every slot was busy: not tracked on the GPU, so no node
    untracked.list_name = L"untracked";
    progress.push_back(untracked);
    s.build(progress, {}, arena);
    n = s.head1();
    EXPECT(n && n->BreadcrumbCount == 4 && n->pLastBreadcrumbValue && *n->pLastBreadcrumbValue == 2);
    EXPECT(n->pCommandHistory[0] == D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION &&
           n->pCommandHistory[1] == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER &&
           n->pCommandHistory[2] == D3D12_AUTO_BREADCRUMB_OP_DISPATCH &&
           n->pCommandHistory[3] == D3D12_AUTO_BREADCRUMB_OP_DRAWINDEXEDINSTANCED);
    n = n->pNext;
    EXPECT(n && n->BreadcrumbCount == 2 && *n->pLastBreadcrumbValue == 0 &&
           n->pCommandHistory[0] == D3D12_AUTO_BREADCRUMB_OP_CLEARRENDERTARGETVIEW);
    n = n->pNext;
    EXPECT(n && n->BreadcrumbCount == 2 && *n->pLastBreadcrumbValue == 0);  // unknown: none claimed
    n = n->pNext;
    EXPECT(n && n->BreadcrumbCount == 0 && !n->pCommandHistory && *n->pLastBreadcrumbValue == 0 && !n->pNext);
    EXPECT(s.head()->pLastBreadcrumbValue && *s.head()->pLastBreadcrumbValue == 2);  // the same in NODE and NODE1
    int nodes = 0;
    for (auto [a, b] = std::pair{s.head(), s.head1()}; a || b; a = a->pNext, b = b->pNext, ++nodes) {
        EXPECT(a && b && a->BreadcrumbCount == b->BreadcrumbCount && a->pCommandHistory == b->pCommandHistory &&
               a->pLastBreadcrumbValue == b->pLastBreadcrumbValue && *a->pLastBreadcrumbValue <= a->BreadcrumbCount);
    }
    EXPECT(nodes == 4);  // the untracked list has none
    for (auto* a = s.head(); a; a = a->pNext)
        EXPECT(!a->pCommandListDebugNameW || std::wcscmp(a->pCommandListDebugNameW, L"untracked") != 0);
    std::printf("snapshot: nodes for the lists that did not complete, operations and fault VA ok\n");
    return 0;
}

// A controlled loss for the D3D12 path (as in vk_device_lost_test): vkQueueSubmit returns VK_ERROR_DEVICE_LOST when
// armed, without reaching the driver; vkCmdWriteBufferMarkerAMD records nothing (every list then reads "not
// started"); vkGetDeviceFaultInfoEXT reports one invalid write at `fault_address`; the first vkGetBufferDeviceAddress
// result is the GPU address arena's (create_arena asks for it first).
namespace loss {
PFN_vkGetInstanceProcAddr real_instance = nullptr;
PFN_vkGetDeviceProcAddr real_device = nullptr;
PFN_vkQueueSubmit real_submit = nullptr;
PFN_vkGetBufferDeviceAddress real_bda = nullptr;
std::atomic<bool> arm{false}, lost{false};
std::atomic<VkDeviceAddress> arena{0}, fault_address{0};

VKAPI_ATTR VkResult VKAPI_CALL submit(VkQueue queue, std::uint32_t count, const VkSubmitInfo* submits, VkFence fence) {
    if (lost) return VK_ERROR_DEVICE_LOST;
    if (arm.exchange(false)) {
        lost = true;
        return VK_ERROR_DEVICE_LOST;
    }
    return real_submit(queue, count, submits, fence);
}
VKAPI_ATTR void VKAPI_CALL marker(VkCommandBuffer, VkPipelineStageFlagBits, VkBuffer, VkDeviceSize, std::uint32_t) {}
VKAPI_ATTR VkResult VKAPI_CALL fault(VkDevice, VkDeviceFaultCountsEXT* counts, VkDeviceFaultInfoEXT* info) {
    if (!lost) return VK_ERROR_UNKNOWN;
    if (!info) {
        counts->addressInfoCount = 1;
        counts->vendorInfoCount = 0;
        counts->vendorBinarySize = 0;
        return VK_SUCCESS;
    }
    std::snprintf(info->description, sizeof info->description, "mock fault");
    if (info->pAddressInfos && counts->addressInfoCount)
        info->pAddressInfos[0] = {VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT, fault_address.load(), 0};
    return VK_SUCCESS;
}
VKAPI_ATTR VkDeviceAddress VKAPI_CALL bda(VkDevice device, const VkBufferDeviceAddressInfo* address) {
    const VkDeviceAddress result = real_bda(device, address);
    VkDeviceAddress none = 0;
    arena.compare_exchange_strong(none, result);
    return result;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    if (!std::strcmp(name, "vkQueueSubmit")) return reinterpret_cast<PFN_vkVoidFunction>(submit);
    if (!std::strcmp(name, "vkCmdWriteBufferMarkerAMD")) return reinterpret_cast<PFN_vkVoidFunction>(marker);
    if (!std::strcmp(name, "vkGetDeviceFaultInfoEXT")) return reinterpret_cast<PFN_vkVoidFunction>(fault);
    if (!std::strcmp(name, "vkGetBufferDeviceAddress") || !std::strcmp(name, "vkGetBufferDeviceAddressKHR")) {
        if (!real_bda) real_bda = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(real_device(device, name));
        return real_bda ? reinterpret_cast<PFN_vkVoidFunction>(bda) : nullptr;
    }
    return real_device(device, name);
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    if (!std::strcmp(name, "vkGetDeviceProcAddr")) return reinterpret_cast<PFN_vkVoidFunction>(device_proc);
    return real_instance(instance, name);
}
}  // namespace loss

// After a (controlled) loss the outputs hold the layer's breadcrumbs: the named list as a node with its queue's
// name, and the fault as a D3D12 VA.
int run_loss(vktest::Device& d) {
    const char* label = "Vulkan loss";
    loss::real_instance = d.instance_proc;
    loss::real_device = d.device_proc;
    loss::real_submit = reinterpret_cast<PFN_vkQueueSubmit>(d.device_proc(d.device, "vkQueueSubmit"));
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = loss::instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.buffer_device_address = d.buffer_device_address;
    info.buffer_marker = true;  // the mock stands in for the extension
    info.device_fault = true;
    Ref<ID3D12Device> device;
    EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
    Ref<ID3D12DeviceRemovedExtendedData1> dred;
    EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dred.p))));

    Ref<ID3D12CommandQueue> queue;
    D3D12_COMMAND_QUEUE_DESC qd{};
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(queue->SetName(L"dred queue")));
    Ref<ID3D12CommandAllocator> allocator;
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = 256;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    Ref<ID3D12Resource> dst, src;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&dst.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON,
                                                     nullptr, IID_PPV_ARGS(&src.p))));
    Ref<ID3D12GraphicsCommandList> a, b;
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&a.p))));
    EXPECT(SUCCEEDED(a->SetName(L"list A")));
    a->CopyBufferRegion(dst.p, 0, src.p, 0, 16);
    EXPECT(SUCCEEDED(a->Close()));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&b.p))));
    b->CopyBufferRegion(dst.p, 16, src.p, 16, 16);
    EXPECT(SUCCEEDED(b->Close()));
    ID3D12CommandList* first[] = {a.p};
    queue->ExecuteCommandLists(1, first);
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)));
    EXPECT(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));  // the GPU is idle before the loss
    const D3D12_GPU_VIRTUAL_ADDRESS va = info.buffer_device_address ? dst->GetGPUVirtualAddress() : 0;
    loss::fault_address = va && loss::arena ? loss::arena + va + 0x40 : 0x40;
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 out{};
    EXPECT(dred->GetAutoBreadcrumbsOutput1(&out) == DXGI_ERROR_NOT_CURRENTLY_AVAILABLE);

    loss::arm = true;
    ID3D12CommandList* second[] = {b.p};
    queue->ExecuteCommandLists(1, second);  // VK_ERROR_DEVICE_LOST
    EXPECT(device->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED);

    EXPECT(dred->GetAutoBreadcrumbsOutput1(&out) == S_OK);
    const D3D12_AUTO_BREADCRUMB_NODE1* named = nullptr;
    unsigned nodes = 0;
    for (auto* n = out.pHeadAutoBreadcrumbNode; n; n = n->pNext, ++nodes) {
        EXPECT(!n->pCommandList && !n->pCommandQueue && n->pLastBreadcrumbValue && *n->pLastBreadcrumbValue == 0);
        if (n->pCommandListDebugNameW && std::wcscmp(n->pCommandListDebugNameW, L"list A") == 0) named = n;
        EXPECT(!n->pCommandListDebugNameW || std::wcscmp(n->pCommandListDebugNameW, L"list A") == 0);  // B: lost
    }
    EXPECT(named && named->BreadcrumbCount == 1 && named->pCommandHistory[0] == D3D12_AUTO_BREADCRUMB_OP_COPYBUFFERREGION);
    EXPECT(named->pCommandListDebugNameA && std::strcmp(named->pCommandListDebugNameA, "list A") == 0);
    EXPECT(named->pCommandQueueDebugNameW && std::wcscmp(named->pCommandQueueDebugNameW, L"dred queue") == 0);
    D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1 again{};
    EXPECT(dred->GetAutoBreadcrumbsOutput1(&again) == S_OK && again.pHeadAutoBreadcrumbNode == out.pHeadAutoBreadcrumbNode);
    D3D12_DRED_PAGE_FAULT_OUTPUT1 page{};
    EXPECT(dred->GetPageFaultAllocationOutput1(&page) == S_OK);
    EXPECT(page.PageFaultVA == (va && loss::arena ? va + 0x40 : 0));  // no arena: no D3D12 VA to give
    EXPECT(!page.pHeadExistingAllocationNode && !page.pHeadRecentFreedAllocationNode);
    std::printf("%s: after VK_ERROR_DEVICE_LOST, %u node(s), \"list A\" on \"dred queue\" not completed, page fault "
                "VA %#llx%s\n", label, nodes, static_cast<unsigned long long>(page.PageFaultVA),
                va ? " (the buffer's VA + 0x40)" : " (no arena)");
    return 0;
}

void no_dialogs() {
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    SetErrorMode(GetErrorMode() | SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
}

}  // namespace

int main(int argc, char** argv) {
    no_dialogs();
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char* label = "d3d12_dred_test";
    const bool on = argc > 1 && std::strcmp(argv[1], "on") == 0;
    EXPECT(on == rt::d3d12::dred_requested());  // ctest sets LEONIDA_DRED=1 for "on" only
    if (snapshot()) return 1;
    int ran = 0;

    const bool wine = GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version") != nullptr;
    if (wine) std::printf("host D3D12: under Wine, skipped\n");
    if (!wine && on) {
        Ref<ID3D12DeviceRemovedExtendedDataSettings> settings;
        EXPECT(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&settings.p))));
        settings->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        settings->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    }
    Ref<IDXGIFactory4> factory;
    if (!wine && SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        Ref<IDXGIAdapter1> warp;
        if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p)))) {
            const int r = run_host(warp.p, "host WARP", on);
            if (r == 1) return 1;
            ran += r == 0;
        }
        for (UINT i = 0;; ++i) {
            Ref<IDXGIAdapter1> adapter;
            if (factory->EnumAdapters1(i, &adapter.p) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
            char name[128] = {};
            WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name) - 1, nullptr, nullptr);
            char host_label[160];
            std::snprintf(host_label, sizeof(host_label), "host %s", name);
            const int r = run_host(adapter.p, host_label, on);
            if (r == 1) return 1;
            ran += r == 0;
            break;
        }
    }

    vktest::Device d;
    d.request_buffer_marker = on;
    d.request_buffer_device_address = on;  // the loss case maps a fault to a D3D12 VA
    if (d.create()) {
        label = "Vulkan";
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.buffer_marker = d.buffer_marker;  // the breadcrumbs run under validation; no loss shows them
        if (on && !d.buffer_marker) std::printf("Vulkan: no VK_AMD_buffer_marker, breadcrumbs off\n");
        {
            ID3D12Device* inner = nullptr;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
            inner->AddRef();  // the backend's own reference; the proxy takes the other
            ID3D12Device* proxy = nullptr;
            EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                               reinterpret_cast<void**>(&proxy))));
            rt::d3d12::set_device_outer(inner, proxy);
            Ref<ID3D12DeviceRemovedExtendedData1> through_proxy, direct;
            EXPECT(proxy->QueryInterface(IID_PPV_ARGS(&through_proxy.p)) == E_NOINTERFACE && !through_proxy.p);
            EXPECT(SUCCEEDED(inner->QueryInterface(IID_PPV_ARGS(&direct.p))));
            direct.reset();
            proxy->Release();
            EXPECT(inner->Release() == 0);  // the tear-off kept no reference
            std::printf("Vulkan: DRED refused behind the Xbox-ABI device proxy, answered by the layer's device\n");
        }
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        if (check_device(label, device.p, on, false)) return 1;
        if (on && run_loss(d)) return 1;
        device.reset();
        ++ran;
        if (d.validation_errors()) {
            std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
            return 1;
        }
    } else {
        std::printf("Vulkan: no device, skipped\n");
    }
    if (!ran) {
        std::printf("SKIP d3d12_dred_test: neither a host D3D12 device nor a Vulkan device\n");
        return 77;
    }
    std::printf("d3d12_dred_test %s: %d backend(s) passed\n", on ? "on" : "off", ran);
    return 0;
}
