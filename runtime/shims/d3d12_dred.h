// ID3D12DeviceRemovedExtendedData, ...1 and ...2 for the Vulkan-backed device (LEONIDA_DRED=1), answered from the
// layer's device-lost diagnostics (src/vulkan_breadcrumbs.h, VK_EXT_device_fault). Private to the runtime; include
// after d3d12_com_object.h.
//
// What the host runtime does (measured with tools in no repository file: WARP and an RTX 4060 Ti, driver 621.01,
// Windows 11 26200, the same on both):
//   - QueryInterface for the three interfaces succeeds on every device, DRED enabled or not, and the result has the
//     device's IUnknown identity;
//   - DRED not enabled: every output method returns DXGI_ERROR_UNSUPPORTED, before and after RemoveDevice;
//   - enabled (ID3D12DeviceRemovedExtendedDataSettings, FORCED_ON) on a device that is not removed:
//     DXGI_ERROR_NOT_CURRENTLY_AVAILABLE, a null argument included;
//   - enabled, after RemoveDevice: S_OK with no breadcrumb node and page-fault VA 0, even with a list held behind
//     a Wait that was never satisfied; GetDeviceState is D3D12_DRED_DEVICE_STATE_UNKNOWN throughout. With a null
//     argument the host runtime dereferences it; here it is E_INVALIDARG.
// The layer does the same, with LEONIDA_DRED=1 in place of the settings interface. After a real loss
// (VK_ERROR_DEVICE_LOST) the outputs hold what the layer knows, and nothing it does not:
//   - a node for each list kept by the breadcrumb ring that the GPU did not complete, oldest first. The
//     granularity is the list: *pLastBreadcrumbValue is 0 (one that started may have run some of its operations).
//     The list and queue debug names are copies taken at submission (dred_debug_name); pCommandList and
//     pCommandQueue stay null, since nothing keeps those objects alive after ExecuteCommandLists, and there are no
//     breadcrumb contexts. Without VK_AMD_buffer_marker the GPU progress is unknown and there is no node at all;
//     the stderr report lists the submitted lists instead;
//   - PageFaultVA: the first fault address VK_EXT_device_fault reports as an invalid read, write or execute,
//     as a D3D12 VA (Vulkan device address minus the arena's, as GetGPUVirtualAddress computes it), 0 without one
//     or when it lies below the arena; no allocation nodes (the layer keeps no allocation history);
//   - GetDeviceState stays UNKNOWN: a Vulkan loss does not say whether the GPU hung or faulted.
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi.h>

#include "../src/vulkan_backend.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace rt::d3d12 {

// The D3D12 debug name of `object` as it is now: SetName stores it as WKPDID_D3DDebugObjectNameW private data with
// its terminator (d3d12_com_object.h), and GetPrivateData reads it under the object's own lock. Empty without one.
// The caller holds a reference to `object`; the name is a copy.
template<class T>
std::wstring dred_debug_name(T* object) {
    if (!object) return {};
    for (int attempt = 0; attempt < 4; ++attempt) {
        UINT size = 0;
        if (object->GetPrivateData(T::kDebugObjectNameW, &size, nullptr) != S_OK || size < sizeof(wchar_t)) return {};
        std::vector<wchar_t> text(size / sizeof(wchar_t) + 1, L'\0');
        UINT read = size;
        const HRESULT hr = object->GetPrivateData(T::kDebugObjectNameW, &read, text.data());
        if (hr == DXGI_ERROR_MORE_DATA) continue;  // renamed to something longer between the two calls
        if (hr != S_OK) return {};
        return std::wstring(text.data(), wcsnlen(text.data(), read / sizeof(wchar_t)));
    }
    return {};
}

// LEONIDA_DRED, read once: 1 asks for device-lost diagnostics with a breadcrumb per command list (two GPU markers
// per list, close to free); 2 adds one after every D3D12 operation, so *pLastBreadcrumbValue says how far a list
// got (BUILD-1655; on an RTX 4060 Ti about 0.4 us of GPU time per operation, docs/d3d12_vulkan.md section 4.11).
// Anything else is off.
inline int dred_level() {
    static const int level = [] {
        char text[4] = {};
        std::size_t length = 0;
        if (getenv_s(&length, text, "LEONIDA_DRED") != 0 || length != 2) return 0;
        return text[0] == '1' ? 1 : text[0] == '2' ? 2 : 0;
    }();
    return level;
}
inline bool dred_requested() { return dred_level() > 0; }

// The D3D12 view of the layer's diagnostics at one moment: the nodes and the fault VA, with storage that lives as
// long as this object (the host runtime's pointers stay valid as long as the device).
class DredSnapshot {
public:
    void build(const std::vector<leonida::BreadcrumbRing::Entry>& entries,
               const std::vector<VkDeviceFaultAddressInfoEXT>& addresses, VkDeviceAddress arena) {
        histories_.clear();
        last_.clear();
        names_.clear();
        for (const auto& e : entries) {
            // Completed lists are left out, as on D3D12. So are lists whose GPU progress is unknown (no
            // VK_AMD_buffer_marker): a node has no way to say "unknown", and claiming one in flight would be a guess.
            if (e.state == leonida::BreadcrumbState::completed || e.state == leonida::BreadcrumbState::unknown)
                continue;
            histories_.emplace_back();
            for (const std::uint32_t op : e.ops) histories_.back().push_back(static_cast<D3D12_AUTO_BREADCRUMB_OP>(op));
            last_.push_back(e.progress_known ? e.completed_ops : 0);  // operations the GPU finished, in order
            names_.push_back(Names{e.list_name, ansi(e.list_name), e.queue_name, ansi(e.queue_name)});
        }
        const std::size_t count = histories_.size();
        nodes_.assign(count, {});
        nodes1_.assign(count, {});
        for (std::size_t i = 0; i < count; ++i) {
            auto& n = nodes_[i];
            n.BreadcrumbCount = static_cast<UINT>(histories_[i].size());
            n.pLastBreadcrumbValue = &last_[i];
            n.pCommandHistory = histories_[i].empty() ? nullptr : histories_[i].data();
            n.pNext = i + 1 < count ? &nodes_[i + 1] : nullptr;
            const Names& names = names_[i];  // the names are copies held here; pCommandList/pCommandQueue stay null
            n.pCommandListDebugNameW = names.list_w.empty() ? nullptr : names.list_w.c_str();
            n.pCommandListDebugNameA = names.list_a.empty() ? nullptr : names.list_a.c_str();
            n.pCommandQueueDebugNameW = names.queue_w.empty() ? nullptr : names.queue_w.c_str();
            n.pCommandQueueDebugNameA = names.queue_a.empty() ? nullptr : names.queue_a.c_str();
            auto& n1 = nodes1_[i];
            n1.BreadcrumbCount = n.BreadcrumbCount;
            n1.pLastBreadcrumbValue = n.pLastBreadcrumbValue;
            n1.pCommandHistory = n.pCommandHistory;
            n1.pCommandListDebugNameW = n.pCommandListDebugNameW;
            n1.pCommandListDebugNameA = n.pCommandListDebugNameA;
            n1.pCommandQueueDebugNameW = n.pCommandQueueDebugNameW;
            n1.pCommandQueueDebugNameA = n.pCommandQueueDebugNameA;
            n1.pNext = i + 1 < count ? &nodes1_[i + 1] : nullptr;
        }
        fault_va_ = 0;
        for (const auto& a : addresses) {
            if (a.addressType != VK_DEVICE_FAULT_ADDRESS_TYPE_READ_INVALID_EXT &&
                a.addressType != VK_DEVICE_FAULT_ADDRESS_TYPE_WRITE_INVALID_EXT &&
                a.addressType != VK_DEVICE_FAULT_ADDRESS_TYPE_EXECUTE_INVALID_EXT)
                continue;
            if (arena && a.reportedAddress >= arena) fault_va_ = a.reportedAddress - arena;
            break;
        }
    }

    const D3D12_AUTO_BREADCRUMB_NODE* head() const { return nodes_.empty() ? nullptr : nodes_.data(); }
    const D3D12_AUTO_BREADCRUMB_NODE1* head1() const { return nodes1_.empty() ? nullptr : nodes1_.data(); }
    D3D12_GPU_VIRTUAL_ADDRESS fault_va() const { return fault_va_; }

private:
    struct Names {
        std::wstring list_w;
        std::string list_a;
        std::wstring queue_w;
        std::string queue_a;
    };
    // The ANSI form D3D12's *DebugNameA fields hold, in the process code page.
    static std::string ansi(const std::wstring& name) {
        if (name.empty()) return {};
        const int length = WideCharToMultiByte(CP_ACP, 0, name.c_str(), static_cast<int>(name.size()), nullptr, 0,
                                               nullptr, nullptr);
        if (length <= 0) return {};
        std::string out(static_cast<std::size_t>(length), '\0');
        WideCharToMultiByte(CP_ACP, 0, name.c_str(), static_cast<int>(name.size()), out.data(), length, nullptr,
                            nullptr);
        return out;
    }

    std::vector<std::vector<D3D12_AUTO_BREADCRUMB_OP>> histories_;
    std::vector<Names> names_;
    std::vector<UINT> last_;
    std::vector<D3D12_AUTO_BREADCRUMB_NODE> nodes_;
    std::vector<D3D12_AUTO_BREADCRUMB_NODE1> nodes1_;
    D3D12_GPU_VIRTUAL_ADDRESS fault_va_ = 0;
};

// A tear-off living inside the device: AddRef, Release and QueryInterface go to the device, so the interface has
// the device's identity and lifetime, as on the host runtime.
class DeviceRemovedData final : public ID3D12DeviceRemovedExtendedData2 {
public:
    static bool answers(REFIID riid) {
        return riid == __uuidof(ID3D12DeviceRemovedExtendedData) || riid == __uuidof(ID3D12DeviceRemovedExtendedData1) ||
               riid == __uuidof(ID3D12DeviceRemovedExtendedData2);
    }

    // False until attach(): a device whose construction stopped early does not hand the tear-off out.
    bool attached() const { return device_ != nullptr; }

    // `context` may be null (no Vulkan context: nothing is ever lost); `removed` is the device's RemoveDevice flag;
    // `arena` the GPU address arena's device address, read when the data is taken (0: no arena, no VA).
    void attach(IUnknown* device, leonida::VulkanContext* context, const std::atomic<bool>* removed,
                const VkDeviceAddress* arena, bool enabled) {
        device_ = device;
        context_ = context;
        removed_ = removed;
        arena_ = arena;
        enabled_ = enabled;
    }

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
        return device_->QueryInterface(riid, out);
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return device_->AddRef(); }
    ULONG STDMETHODCALLTYPE Release() override { return device_->Release(); }

    HRESULT STDMETHODCALLTYPE GetAutoBreadcrumbsOutput(D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT* out) override {
        return ready(out, [&] { out->pHeadAutoBreadcrumbNode = snapshot_.head(); });
    }
    HRESULT STDMETHODCALLTYPE GetPageFaultAllocationOutput(D3D12_DRED_PAGE_FAULT_OUTPUT* out) override {
        return ready(out, [&] { *out = {snapshot_.fault_va(), nullptr, nullptr}; });
    }
    HRESULT STDMETHODCALLTYPE GetAutoBreadcrumbsOutput1(D3D12_DRED_AUTO_BREADCRUMBS_OUTPUT1* out) override {
        return ready(out, [&] { out->pHeadAutoBreadcrumbNode = snapshot_.head1(); });
    }
    HRESULT STDMETHODCALLTYPE GetPageFaultAllocationOutput1(D3D12_DRED_PAGE_FAULT_OUTPUT1* out) override {
        return ready(out, [&] { *out = {snapshot_.fault_va(), nullptr, nullptr}; });
    }
    HRESULT STDMETHODCALLTYPE GetPageFaultAllocationOutput2(D3D12_DRED_PAGE_FAULT_OUTPUT2* out) override {
        return ready(out, [&] { *out = {snapshot_.fault_va(), nullptr, nullptr, D3D12_DRED_PAGE_FAULT_FLAGS_NONE}; });
    }
    D3D12_DRED_DEVICE_STATE STDMETHODCALLTYPE GetDeviceState() override { return D3D12_DRED_DEVICE_STATE_UNKNOWN; }

private:
    // The host runtime's order: not enabled, then not removed, then the output. The data is taken once, at the
    // first call after the loss, so the pointers handed out stay valid; `fill` reads it under the same lock, since a
    // caller that saw only RemoveDevice may read while another one takes the data after a real loss.
    template<class T, class Fill>
    HRESULT ready(T* out, Fill&& fill) {
        if (!enabled_) return DXGI_ERROR_UNSUPPORTED;
        const bool lost = context_ && context_->is_device_lost();
        if (!lost && !(removed_ && removed_->load(std::memory_order_acquire))) return DXGI_ERROR_NOT_CURRENTLY_AVAILABLE;
        if (!out) return E_INVALIDARG;
        *out = {};
        std::lock_guard lock(mutex_);
        if (!taken_ && lost) {  // RemoveDevice alone leaves the outputs empty, as on the host runtime
            try {
                std::vector<leonida::BreadcrumbRing::Entry> entries;
                if (const auto* ring = context_->breadcrumbs()) entries = ring->snapshot();
                std::vector<VkDeviceFaultAddressInfoEXT> addresses;
                std::string description;
                context_->device_fault(addresses, description);
                snapshot_.build(entries, addresses, arena_ ? *arena_ : 0);
            } catch (...) {
                return E_OUTOFMEMORY;
            }
            taken_ = true;
        }
        fill();
        return S_OK;
    }

    IUnknown* device_ = nullptr;
    leonida::VulkanContext* context_ = nullptr;
    const std::atomic<bool>* removed_ = nullptr;
    const VkDeviceAddress* arena_ = nullptr;
    bool enabled_ = false;
    std::mutex mutex_;
    bool taken_ = false;
    DredSnapshot snapshot_;
};

}  // namespace rt::d3d12
