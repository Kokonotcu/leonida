// Xbox-ABI proxy around a PC ID3D12Device.
//
// Measured Xbox device deviations used here (vtable indices, x64):
//   slot 26   GetCustomHeapProperties, the PC slot, but the retail executable calls it with the buffer for the
//             returned struct in RCX and the device in RDX (node mask in R8, heap type in R9; three calls with node
//             mask 1 and types DEFAULT/UPLOAD/READBACK). A PC C++ method has the device in RCX and the buffer in RDX.
//             The thunk takes the console form when RDX holds a device proxy and forwards the PC form otherwise.
//             Slots 25 and 43 return structs too and are still forwarded in the PC form: the executable calls 25 the
//             same way (#6982 and its own call site), with console resource descriptions the PC method cannot take as
//             they are, and no call of 43 has been seen.
//   slot 41   CreateCommandSignature at its PC position; a console argument of type 105 (a 12-byte dispatch, read
//             from the retail title's call sites) is created as D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH.
//   slot 61   CreateSamplerX(desc, dest) -> PC CreateSampler (slot 22); the description's PC
//             D3D12_SAMPLER_DESC prefix is passed. The PC slot 61 is CreateMetaCommand. Measured on the retail title's boot.
//   slot 62   CreateComputePipelineState(desc, w2, w3, iid, out): two words (0 from XboxBvhBuilder, meaning unknown)
//             before the IID. The PC slot 62 is CreateStateObject.
//   slot 67   CreateComputePipelineState from native code (desc, iid, out): converts the Xbox native-code desc
//             (+0x00 root signature, +0x28 code, +0x30 size) to D3D12_COMPUTE_PIPELINE_STATE_DESC and forwards
//             to PC CreateComputePipelineState (slot 11). The PC slot 67 is CreateProtectedResourceSession1.
//   slot 70   (texture words, view desc, destination handle), no return value: a shader-resource view from raw words
//             and a GPU address, with no resource (retail boot chain stop 14). The PC slot 70 is CreatePlacedResource1.
//             A measured shape (PR 7066, eugen1763): a raw buffer SRV. arg1 starts with a desktop D3D12_RESOURCE_DESC
//             (buffer), arg2 with a desktop D3D12_SHADER_RESOURCE_VIEW_DESC (R32_TYPELESS, BUFFER, RAW) followed at
//             once by a D3D12_GPU_VIRTUAL_ADDRESS, the console VA a slot-133 call registered. When the VA is inside a
//             buffer the device's resource provider knows, the destination gets a raw view of it (PC
//             CreateShaderResourceView with that resource). Any other shape, or an unknown VA, gets the PC null view
//             (CreateShaderResourceView with no resource) and the call returns.
//   slot 76   resource creation -> PC CreateCommittedResource (PC slot 27), with the Xbox layout word ignored.
//   slot 48   CreateCommandQueueX(desc, riid, out) -> the same queue proxy as slot 8. The Xbox desc's first fields are
//             D3D12_COMMAND_QUEUE_DESC's (public HLE).
//   slot 73   CreateCommandListX(desc, allocator, initialState, riid, out) -> PC CreateCommandList(0, type, allocator,
//             initialState, riid, out). The type is the first word of the Xbox desc; there is no node-mask argument
//             in the console call (public HLE; the PC six-argument form differs).
//   slots 78/79/80  SetFrameIntervalX, ScheduleFrameEventX, WaitFrameEventX. They answer through the registered
//             XboxPresentationSink when there is one (the Vulkan retail path below); otherwise a per-device token
//             source answers them before the PC vtable is indexed and WaitFrameEventX returns at once with a fresh
//             non-zero token. The token source is shared with the sink's frame pacer, so a token from slot 80 is one
//             PresentX accepts. Slot 80 is outside a PC ID3D12Device11 table, so it is checked before the
//             beyond-table refusal.
//   slot 86   CreatePipelineState(desc, riid, out) -> PC ID3D12Device2::CreatePipelineState (PC slot 47).
//   slot 101  CreateStateObject(desc, riid, out) -> PC ID3D12Device5::CreateStateObject (PC slot 62, displaced on
//             Xbox by compute-PSO slot 62).
//   slot 108  GetResourceAllocationInfo2(ret, this, visible_mask, count, descs, info1) -> PC
//             ID3D12Device8::GetResourceAllocationInfo2 (PC slot 68), accepting the console C-ABI struct return
//             (return buffer in RCX, device in RDX) as well as the PC C++ order, with Xbox layout words and flag bits
//             normalized on a copy of the descriptors.
//   slot 127  CreateCommittedResource3 with Xbox layout/flags -> the PC device's ID3D12Device10 method; the
//             caller's heap, layout, clear value, session and castable formats are kept.
// The slots 48/73/78/79/80 are the public HLE Freminet/ScarlettX's observation, not Microsoft documentation, and can
// be moved with LEONIDA_XS_QUEUE_X_SLOT, LEONIDA_XS_LIST_X_SLOT, LEONIDA_XS_FRAME_INTERVAL_SLOT,
// LEONIDA_XS_SCHEDULE_EVENT_SLOT and LEONIDA_XS_WAIT_EVENT_SLOT (3..127; anything else is logged once and ignored).
//   slot 13   CheckFeatureSupport keeps the PC signature, but a D3D12_OPTIONS request with a structure larger than
//             the PC's (0x40 against 0x3C, measured on a retail console title) is answered from the PC runtime's
//             structure with the extra bytes zeroed; the PC runtime refuses that size. See d3d12_console_feature.h.
//   slot 133  CreatePlacedResourceX(va, desc, state, clear, riid, out) -> PC CreateCommittedResource on a DEFAULT heap,
//             with the Xbox flag bits dropped and an all-ones state as COMMON. Not backed by the memory at va.
// The measured slots 9/10 (GetCPU/GPUDescriptorHandleForHeapStart) are ID3D12DescriptorHeap methods, not device
// methods: the PC device's 9/10 are CreateCommandAllocator/CreateGraphicsPipelineState and are forwarded unchanged.
// On the PC heap those getters use the hidden return pointer in RDX; the Xbox ABI returns the 8-byte handle in
// RAX. Device slot 14 therefore wraps the created heap in a heap proxy that performs that conversion. The heap
// proxy holds a reference to the device proxy, and its ID3D12DeviceChild::GetDevice returns that same proxy.
// The proxy's heap slots 9/10 expose only the Xbox shape: a PC C++ caller using the PC header's return convention
// would not receive the value, while GetDesc and the other heap slots forward the PC convention unchanged. A
// direct one-argument call against a real PC heap is invalid, so the device test exercises the conversion on the
// wrapper and compares it with a hidden-pointer call against the real heap.
//
// Every other PC device slot is forwarded by a generated thunk that swaps `this` to the PC device. The proxy's
// vtable therefore follows the PC device order, with slots 62 and 76 and the public HLE's 48/73/78/79/80 deliberately
// shadowed. The full Xbox vtable order is not known from the spec; slots beyond the PC device's own vtable return
// E_NOTIMPL and log once.
//
// Command queues and fences (device slots 8, 75, 36, and 32 for a shared fence) come back wrapped in queue and fence
// proxies of the same kind: the public chain (19 queue slots, 11 or 12 fence slots in SDK 10.0.26100) is forwarded
// in PC order, and a slot past it, up to the end of the proxy's table (128 queue, 64 fence slots), is logged with its
// first four argument words and refused with E_NOTIMPL. Fence slot 12 is the served exception: the retail executable
// calls it as (value, descriptor, flags) and reads the return as a pointer to the fence's completed value, so the
// proxy returns a live per-fence mirror (see the section in d3d12_xs_device.cpp). Queue slot 32 is PresentX in the
// retail executable: it goes
// to the real presenter when one is registered, else to the opt-in capture when LEONIDA_PRESENTX_SLOT/_CAPTURE names
// it, else it is refused like any other tail slot (slot 30 can be named as the alternate). A slot past the table
// faults on the vtable read inside a reserved, never-committed region, and the
// fault is logged with the slot. That the console keeps the public slots is an assumption
// (docs/d3d12_command_objects.md §5.2). Heap, queue and fence proxies are
// unwrapped wherever a device or queue method hands them to the real runtime: queue Signal/Wait (14/15), device
// CreateSharedHandle (31), MakeResident/Evict (34/35), SetEventOnMultipleFenceCompletion (45), SetResidencyPriority
// (46) and EnqueueMakeResident (50). A command list created for the console command-list IID is wrapped: on it,
// console slot 139 is the retail title's Reset(allocator, pso) and forwards to PC slot 10 with both proxies unwrapped.
// A heap proxy given to a list's SetDescriptorHeaps reaches the real list as it is. QueryInterface answers only the
// modelled interfaces; one the real object has but the proxy does not model is logged once and refused.
#pragma once

#include <d3d12.h>

#include <cstddef>
#include "xbox_present_types.h"

#include <cstdint>
#include <memory>

namespace leonida {
class FrameTokenSource;
}

namespace rt::d3d12_xs {

// Wraps `real` and returns the requested device interface through `out`. Takes ownership of one reference to
// `real` on success; on failure the caller keeps its reference. Returns E_NOINTERFACE if `riid` is not a D3D12
// device interface the PC device implements.
HRESULT create_device_proxy(ID3D12Device* real, REFIID riid, void** out);

// True when `device` points at a proxy created by create_device_proxy. Used by the tests.
bool is_device_proxy(const void* device);

// The host path of D3D12XboxCreateDevice (BL-2, docs/d3d12_xs_surface.md): the PC D3D12CreateDevice (`create`)
// asked for kHostDeviceFloor, the lowest feature level D3D12 creates a device at. It is a minimum, so asking for it
// refuses no adapter the title can run on and still gives the device its highest level, which FEATURE_LEVELS then
// reports. A null `create`, or S_OK with no device, is E_NOTIMPL / E_FAIL; `*out` is null on failure.
constexpr D3D_FEATURE_LEVEL kHostDeviceFloor = D3D_FEATURE_LEVEL_11_0;
using HostCreateDeviceFn = HRESULT(WINAPI*)(IUnknown* adapter, D3D_FEATURE_LEVEL minimum, REFIID riid, void** out);
HRESULT create_host_device(HostCreateDeviceFn create, IUnknown* adapter, ID3D12Device** out);

// The highest of 11_0, 11_1, 12_0, 12_1 and 12_2 `device` reports through D3D12_FEATURE_FEATURE_LEVELS, or 0 when
// the query fails.
D3D_FEATURE_LEVEL max_feature_level(ID3D12Device* device);

// IDXGIDevice1 on the device. A GDKX title asks its device for IDXGIDevice1, then GetAdapter and EnumOutputs(0),
// to find the output it gives SetFrameIntervalX (docs/d3d12_vulkan.md section 4.5). The proxy answers IDXGIObject,
// IDXGIDevice and IDXGIDevice1 itself, as one COM object with the device (same reference count, QueryInterface
// both ways). GetAdapter and GetParent return the adapter of the device behind the proxy: by default the host DXGI
// adapter with that device's LUID. A provider replaces that: `real` is the device behind the proxy, and it returns
// the adapter as `riid`. rt::d3d12::set_device_outer sets one for the Vulkan-backed device (the layer's own adapter).
//
// A title may ask D3D12XboxCreateDevice (so create_device_proxy) for IDXGIDevice1 directly. What comes back then is
// the proxy's DXGI pointer, not the proxy: is_device_proxy is false for it. Code that takes "the proxy" from a
// creation call goes through device_proxy_primary first.
using DxgiAdapterProvider = HRESULT (*)(ID3D12Device* real, REFIID riid, void** out);
// `device` is the proxy or its DXGI pointer. False, with a line on stderr, when it is neither: nothing is set then.
bool set_device_adapter_provider(void* device, DxgiAdapterProvider provider);
// Finds the buffer resource that holds a console GPU virtual address, for device slot 70. `real` is the device behind
// the proxy; `bytes` is how many bytes from `va` the caller wants, so a resource that holds all of them is preferred
// over a shorter alias. Returns the resource with a reference added (the proxy releases it), or null when no resource holds
// `va`; `start` and `size` are then unused, otherwise the resource's first VA and size in bytes.
using ResourceByVaProvider = ID3D12Resource* (*)(ID3D12Device* real, std::uint64_t va, std::uint64_t bytes,
                                                std::uint64_t* start, std::uint64_t* size);
// `device` is the proxy or its DXGI pointer. False, with a line on stderr, when it is neither. Without a provider slot
// 70 writes only the null view.
bool set_device_resource_provider(void* device, ResourceByVaProvider provider);
// The texture at a console GPU address, for a slot-70 view of a texture: the title-placed texture registered there with
// the description's extent, or a new one placed there as device slot 133 places one. Returns it with a reference
// added (the proxy releases it), or null.
using TextureByVaProvider = ID3D12Resource* (*)(ID3D12Device* real, std::uint64_t va, const D3D12_RESOURCE_DESC* desc);
bool set_device_texture_provider(void* device, TextureByVaProvider provider);
// The device proxy an IDXGIDevice pointer handed out by one belongs to, or null.
void* device_proxy_from_dxgi_device(void* dxgi_device);
// The device proxy itself, given the proxy or its DXGI pointer; null for anything else.
void* device_proxy_primary(void* device);

// True when `heap` is a descriptor-heap proxy created by device slot 14.
bool is_heap_proxy(const void* heap);
ID3D12DescriptorHeap* heap_proxy_real_heap(const void* heap);

// Console D3D12 descriptor heaps allocate their backing through XMem (allocator ID 0x80 = eXALLOCAllocatorId_D3D in
// bits 16..23, object type 4 in the low 14 bits). A title that intercepts this allocation via __XMemAllocationHooks__
// returns its own virtual address and expects GetCPUDescriptorHandleForHeapStart() to return that address.
inline constexpr std::uint64_t kXmemDescriptorHeapAttributes = (0x80ULL << 16) | 4ULL;
using DescriptorHeapBackingAlloc = void* (*)(std::size_t byte_size, std::uint64_t attributes);
using DescriptorHeapBackingFree = void (*)(void* ptr, std::uint64_t attributes);
// Installs (or clears when both are null) the backing allocator consulted by CreateDescriptorHeap (slot 14).
void set_descriptor_heap_allocator(DescriptorHeapBackingAlloc alloc, DescriptorHeapBackingFree free);
// Installs the default bridge (used by D3D12XboxCreateDevice) only when no explicit allocator has been set.
void set_default_descriptor_heap_allocator(DescriptorHeapBackingAlloc alloc, DescriptorHeapBackingFree free);
// Translates a CPU descriptor handle from a title-backed descriptor heap to the underlying PC/Vulkan heap handle.
// Returns `handle` unchanged when it does not fall inside any active custom-backed heap.
D3D12_CPU_DESCRIPTOR_HANDLE translate_cpu_descriptor_handle(const void* device, D3D12_CPU_DESCRIPTOR_HANDLE handle);

// LEONIDA_TITLE_HEAP_SYNC=1 (off by default): title memory is the canonical copy of a title-backed heap. Its CPU
// handles are not translated, so every descriptor write (slots 17-24, 61) and every byte the title copies in itself
// land there, and sync_title_heaps() copies the 4 KiB chunks of shader-visible ones that changed since the last sync
// (a host shadow, compared with memcmp) into the real heap. The Vulkan device runs it at every ExecuteCommandLists.
// set_title_heap_sync: tests; 1 on, 0 off, -1 back to the environment. Read at heap creation.
void set_title_heap_sync(int mode);
bool title_heap_sync_enabled();
void sync_title_heaps();
struct TitleHeapSyncStats {
    std::uint64_t syncs = 0;            // sync_title_heaps() calls that had heaps to look at
    std::uint64_t chunks_compared = 0;
    std::uint64_t chunks_copied = 0;
    std::uint64_t heaps = 0;            // shader-visible title heaps synced now
};
TitleHeapSyncStats title_heap_sync_stats();
// The sampler-heap slots in [first, end) (title addresses, the device's slot keys in this mode) hold new bytes: the
// device gives each S# its sampler and its slot reference, and drops the slot's old one (an all-zero slot holds none).
// Installed by the Vulkan device; without one sampler heaps sync bytes only.
using TitleSamplerSlotsFn = void (*)(ID3D12Device* real, std::uintptr_t first, std::uintptr_t end);
void set_title_sampler_slots_handler(TitleSamplerSlotsFn handler);

// Number of PC vtable slots the proxy forwards (detected from the interfaces the PC device supports).
unsigned device_proxy_slot_count(const void* device);

// The PC device behind a proxy. Null when `device` is not a proxy.
ID3D12Device* device_proxy_real_device(const void* device);

// The device's frame-token counter, shared with the presentation pacer when one is attached: device slot 80
// (WaitFrameEventX) hands out its tokens and PresentX checks them, so a title that waits through the device and
// presents through its XboxPresentation never sees a token the pacer did not issue. Null for a non-proxy.
std::shared_ptr<leonida::FrameTokenSource> device_frame_tokens(const void* device);

// The real presentation behind the retail title's Xbox slots. The device proxy does not link the Vulkan presenter
// itself: whichever module owns the presentation (d3d12_xs.cpp on the Vulkan backend) registers a factory once, and
// a device asks it for a sink the first time an Xbox presentation slot is called. The factory gets the layer's
// ID3D12Device and the window supplied to set_device_presentation_window (or null: find the process's game window).
// Without a factory, or when it returns null, the device keeps the token stand-in and the queue keeps the capture
// route, so unit tests and a host-D3D12 run do not require a window.
//
// Measured calls from the retail executable (1.0.2.1):
//   device slots 78/79/80  SetFrameIntervalX/ScheduleFrameEventX/WaitFrameEventX (0x270/0x278/0x280).
//   queue slot 32          PresentX(plane_count, planes, params) (0x100). The plane layout is the provisional
//                          XboxPresentPlaneParameters above: the executable writes 0x48 bytes per plane, token at
//                          +0, ResourceCount 1 at +8, ppResources at +0x10, source rect at +0x18, full-output
//                          placement at +0x20 and the colour space at +0x28.
class XboxPresentationSink {
public:
    virtual ~XboxPresentationSink() = default;
    virtual HRESULT SetFrameIntervalX(std::uint64_t output, std::uint32_t interval_us, std::uint32_t latency,
                                      std::uint32_t flags) = 0;
    virtual HRESULT ScheduleFrameEventX(std::uint32_t event, std::uint32_t offset_us, std::uint32_t flags) = 0;
    virtual HRESULT WaitFrameEventX(std::uint32_t event, std::uint32_t timeout_ms, std::uint32_t flags,
                                    std::uint64_t* token) = 0;
    virtual HRESULT PresentX(std::uint32_t plane_count, const rt::d3d12::XboxPresentPlaneParameters* planes,
                             const rt::d3d12::XboxPresentParameters* params) = 0;
};
// The factory returns a sink owned by the device proxy (delete on device release), or null. `device_proxy` is the
// Xbox-ABI device the sink belongs to, so a factory that also drives leonida::FramePacer can share the device's
// frame-token source through device_frame_tokens instead of numbering tokens on its own.
using XboxPresentationFactory = XboxPresentationSink* (*)(ID3D12Device* real, void* device_proxy,
                                                          std::uint64_t window);
// Process-wide; call before the first Xbox presentation slot. A null factory restores the stand-ins.
void set_xbox_presentation_factory(XboxPresentationFactory factory);
// Optional window for this device's factory (an HWND as std::uint64_t); null lets the factory discover it.
void set_device_presentation_window(void* device, std::uint64_t window);
// Device slot 133 (the console's CreatePlacedResourceX) makes a buffer on the title's pages at `va`. The proxy does
// not create it itself: the device behind it does, through this handler, which the Vulkan layer registers
// (rt::d3d12::create_buffer_at_host_address). With none registered, or a PC device behind the proxy, the slot
// answers E_NOTIMPL.
// `layout` is the title's third argument, a D3D12_BARRIER_LAYOUT (0xFFFFFFFF = UNDEFINED), measured on a retail
// boot: 0x6 SHADER_RESOURCE, 0x3 UNORDERED_ACCESS, as PC CreatePlacedResource2 takes it.
using PlacedAtVaHandler = HRESULT (*)(ID3D12Device* real, void* va, const D3D12_RESOURCE_DESC& desc,
                                      std::uint32_t layout, REFIID riid, void** out);
void set_placed_at_va_handler(PlacedAtVaHandler handler);

// Device slot 57: the title's native graphics pipeline creation (desc, riid, out), from an Xbox native PSO
// description that names its stages by DXBC HASH digest (not implemented yet: see the RT-SLOT57 work upstream). With
// LEONIDA_MISSING_SHADER_STUB=1 and a handler registered, the slot answers a pipeline state that draws nothing (its
// draws are skipped) instead of forwarding to PC slot 57, CreateLifetimeTracker, whose refusal the title read through
// as a null pipeline (retail boot 6).
using StubPipelineHandler = HRESULT (*)(ID3D12Device* real, REFIID riid, void** out);
void set_stub_pipeline_handler(StubPipelineHandler handler);

// Called by slot 133 for a buffer, before the handler, with the buffer's range [va, va + width). gta6rt registers
// rt::xmem::commit_physical_range: xmem commits physical pages on first touch, so a range the title has mapped but
// not written yet is still reserved, and the handler's import refuses reserved pages (it would answer with its
// fallback buffer, which the title's writes never reach). A false return is logged once and the handler is still
// called. With none registered nothing is done.
using PlacedAtVaPagesHook = bool (*)(const void* va, std::size_t bytes);
void set_placed_at_va_pages_hook(PlacedAtVaPagesHook hook);

// Device slot 14 with the console's extended description (a PC D3D12_DESCRIPTOR_HEAP_DESC followed by the heap's
// console address and size, issue #313) goes to this handler when one is registered and the address is not 0. The
// Vulkan layer registers rt::d3d12::create_console_descriptor_heap only under LEONIDA_CONSOLE_HEAP_VA=1, because that
// heap reports the title's address without backing it; the default is the XMem-backed heap slot 14 makes otherwise.
// With none, the extra words are ignored and the PC description makes the heap.
using ConsoleDescriptorHeapHandler = HRESULT (*)(ID3D12Device* real, const D3D12_DESCRIPTOR_HEAP_DESC* desc,
                                                 D3D12_GPU_VIRTUAL_ADDRESS address, std::uint64_t size, REFIID riid,
                                                 void** out);
void set_console_descriptor_heap_handler(ConsoleDescriptorHeapHandler handler);

// Presentation wiring: connects device frame-event slots (78/79/80/81: SetFrameIntervalX, ScheduleFrameEventX,
// WaitFrameEventX, GetFrameStatisticsX; 81 is LEONIDA_XS_FRAME_STATISTICS_SLOT) and queue presentation slots (PresentX
// at 30, or LEONIDA_XS_PRESENT_X_SLOT; SuspendX and ResumeX are routed only when LEONIDA_XS_SUSPEND_X_SLOT /
// LEONIDA_XS_RESUME_X_SLOT are set) to an external presenter such as rt::d3d12::XboxPresentation without coupling
// d3d12_xs_device to Vulkan headers. Passing nullptr detaches the handler and restores the default mock/fallback
// behavior.
struct PresentationHandler {
    void* context = nullptr;
    HRESULT (*set_frame_interval)(void* context, IUnknown* output, UINT interval_us, UINT latency, UINT flags) = nullptr;
    HRESULT (*schedule_frame_event)(void* context, UINT event, UINT offset_us, UINT flags) = nullptr;
    HRESULT (*wait_frame_event)(void* context, UINT event, DWORD timeout_ms, UINT flags, std::uint64_t* token) = nullptr;
    HRESULT (*get_frame_statistics)(void* context, std::uint64_t token, UINT type_mask, UINT* count, void* out) = nullptr;
    HRESULT (*present_x)(void* context, UINT plane_count, const void* planes, const void* params) = nullptr;
    HRESULT (*suspend_x)(void* context, UINT flags) = nullptr;
    HRESULT (*resume_x)(void* context) = nullptr;
};
bool set_presentation_handler(void* device, const PresentationHandler* handler);

// True for slots the proxy handles itself rather than forwarding verbatim: IUnknown (0-2), 8 and 75 (queue
// proxies), 9 (allocator proxies), 13 (console-sized D3D12_OPTIONS), 14 (descriptor heap proxy), 17-24 (CPU
// descriptor handle translation), 25/26 (the console's struct-return form), 31/34/35/45/46/50 (proxies unwrapped),
// 32 and 36 (fence proxies), 41 (CreateCommandSignature with the console signature IID), 48/73 and 78/79/80 (the
// Xbox slots above, at their configured indices), 56 (GetResourceAllocationInfo1), 62, 67, 70, 76, 86, 101, 108 and
// 133. Slot 10 is intentionally false: the measured 9/10 handle conversion is on the heap proxy, and device slot 9 is
// CreateCommandAllocator, which the proxy handles for the allocator proxy.
bool device_slot_overridden(unsigned slot);

// Queue and fence proxies created by device slots 8/75 and 36. The real_* accessors return the wrapped object
// (no reference added), or null when the pointer is not such a proxy.
bool is_queue_proxy(const void* queue);
bool is_fence_proxy(const void* fence);
ID3D12CommandQueue* queue_proxy_real_queue(const void* queue);
ID3D12Fence* fence_proxy_real_fence(const void* fence);
// Public slots forwarded (19 for a queue; 11 or 12 for a fence), and the size of the proxy's vtable.
unsigned queue_proxy_slot_count(const void* queue);
unsigned fence_proxy_slot_count(const void* fence);
unsigned queue_proxy_table_size();
unsigned fence_proxy_table_size();
// Live references on a queue, fence or allocator proxy (0 for anything else). Test helper.
unsigned long command_object_proxy_ref_count(const void* object);
// Per-slot call counters and first-call flags, shared by all queue proxies and all fence proxies.
std::uint64_t queue_slot_hits(unsigned slot);
bool queue_slot_first_call_seen(unsigned slot);
std::uint64_t fence_slot_hits(unsigned slot);
bool fence_slot_first_call_seen(unsigned slot);
// Vtable reads past a queue or fence proxy table that faulted in its guard region (counted by the fault logger).
std::uint64_t queue_table_overruns();
std::uint64_t fence_table_overruns();
// True when the queue, fence and allocator tables sit in front of their guard regions (false only if a reservation
// or the fault logger failed).
bool command_object_tables_guarded();

// Command allocator proxies (device slot 9): the public chain (9 slots: the base chain and Reset) is forwarded, a
// slot past it up to the end of the table (64 slots) is logged and refused with E_NOTIMPL, and a read past the
// table faults in a guard region, as for queues and fences. Slot 9 is the served exception (the retail title's
// allocator-pool path): it logs its first calls and answers S_OK without forwarding, and if its first word is
// writable memory holding all ones (the poison the title reads through) it sets that word to null. The console
// allocator's other extra methods are not measured and not guessed. Slot 9 of the device wraps only an allocator
// that answers kLeonidaCommandAllocatorTag, Leonida's
// own Vulkan-layer allocator: its command lists resolve the proxy (CreateCommandList, Reset). A PC runtime
// allocator goes back unchanged, because the PC runtime's lists would take a proxy for their own allocator; a
// command-list proxy that unwraps at Reset is what lets the host backend wrap allocators too.
// {7d0e6b51-34a2-4f0c-9a61-0c5b2e8f4a10}: the Vulkan layer's CommandAllocator answers it with its implementation.
inline constexpr GUID kLeonidaCommandAllocatorTag{
    0x7d0e6b51, 0x34a2, 0x4f0c, {0x9a, 0x61, 0x0c, 0x5b, 0x2e, 0x8f, 0x4a, 0x10}};
bool is_allocator_proxy(const void* allocator);
// The wrapped allocator (no reference added), or null when `allocator` is not an allocator proxy.
ID3D12CommandAllocator* allocator_proxy_real_allocator(const void* allocator);
unsigned allocator_proxy_slot_count(const void* allocator);
unsigned allocator_proxy_table_size();
std::uint64_t allocator_slot_hits(unsigned slot);
bool allocator_slot_first_call_seen(unsigned slot);
std::uint64_t allocator_table_overruns();
// Command list proxies created by device slot 73 (CreateCommandListX):
bool is_command_list_proxy(const void* list);
ID3D12GraphicsCommandList* command_list_proxy_real_list(const void* list);
unsigned command_list_proxy_table_size();
std::uint64_t list_slot_hits(unsigned slot);
bool list_slot_first_call_seen(unsigned slot);

// Heap slots 0-2, 7 (GetDevice -> device proxy) and 9/10 (handle conversion).
bool heap_slot_overridden(unsigned slot);

// Console list slot mapping: returns PC vtable slot or ~0u if console-only / unforwarded.
inline constexpr unsigned kListNotForwarded = ~0u;
inline constexpr unsigned console_to_pc_list_slot(unsigned slot) {
    if (slot <= 26) return slot;               // PC order, measured
    if (slot <= 30) return kListNotForwarded;  // console-only extras
    if (slot <= 59) return slot - 4;           // PC chain, shifted +4
    if (slot == 60) return 58;                 // EndEvent (no console SetMarker/BeginEvent)
    if (slot == 61) return 59;                 // ExecuteIndirect (Package)
    if (slot == 85) return 16;                 // CopyTextureRegionX -> CopyTextureRegion
    if (slot == 146) return 80;                // D3D12_BARRIER_GROUP (Package; PC GCL7)
    if (slot == 139) return 10;                // Reset(allocator, pso): measured on the retail title after Close
    return kListNotForwarded;
}


// Number of live references on a device proxy. Test helper for the heap's GetDevice/lifetime checks.
unsigned long device_proxy_ref_count(const void* device);

// Per-slot call counters and first-call flags for the device and heap proxies. Slot 0 also counts the
// QueryInterface call made while a proxy is created, so tests reset the counters after creation.
// reset_device_proxy_stats also resets the queue and fence counters.
// The hit counters (device, heap, queue, fence and allocator) are for tests, and they are off at the start.
// A test must read a counter (any *_slot_hits function) or call reset_device_proxy_stats BEFORE the calls that it
// counts. Calls made before that are not counted (the counter stays 0); they set only the first-call flags.
std::uint64_t device_slot_hits(unsigned slot);
bool device_slot_first_call_seen(unsigned slot);
std::uint64_t heap_slot_hits(unsigned slot);
bool heap_slot_first_call_seen(unsigned slot);
void reset_device_proxy_stats();

}  // namespace rt::d3d12_xs
