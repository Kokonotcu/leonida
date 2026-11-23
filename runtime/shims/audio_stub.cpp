// AcpHal.dll, xapu.dll, chad.dll: the console audio path, silent and safe for boot.
//
// The retail executable imports 19 functions from these three modules. None has a host namesake, so every import
// used to bind to the loader's unimplemented thunk, which returns 0 and writes nothing. For most of the imports that
// is enough. Three of them, however, are expected to produce a value that the caller then dereferences, and a bare 0
// there is a null read before the first frame:
//
//   AcpHalCreate(&out)                 0x1443aed98: 0x1443aedb7 reads [out] and calls its vtable
//   AcpHalAllocateShapeContexts(&cfg)  0x1443aec73: 0x1443aec80 reads cfg+0x38 and memsets it
//   ApuAlloc(&out, 0, size, align, fl) 0x1443aed13: 0x1443aed5b memsets the returned buffer
//
// So those three hand out real, zeroed host memory. The object behind AcpHalCreate is a static page whose first
// qword is a vtable; every slot answers S_OK, so the caller's vtable calls ([obj], [obj+0x10], [obj+8], [obj+0x28])
// do nothing and cannot fault. Every other function is a silent no-op returning the value its measured caller tests
// for. No APU address is fabricated: ApuMapApuAddress returns null (both call sites check for null before using the
// result) and ApuIsVirtualAddressValid returns FALSE, so the engine never asks for a host-to-APU mapping.
//
// The RVAs above are from game_scarlett_master.exe (Xbox 1.0.2.1). The signatures beyond those call sites are not
// public; audio-stub-notes.md records what is measured, inferred or still unknown.
#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

constexpr int32_t kOk = 0;
constexpr int32_t kInvalidArg = static_cast<int32_t>(0x80070057u);   // E_INVALIDARG
constexpr int32_t kOutOfMemory = static_cast<int32_t>(0x8007000Eu);  // E_OUTOFMEMORY
constexpr int32_t kNoWork = static_cast<int32_t>(0x80070490u);       // HRESULT_FROM_WIN32(ERROR_NOT_FOUND)

// One line per function, once per process: the audio path in the startup trace, without a log per command. Logging
// must never be the reason a boot fails, so an allocation failure while formatting the line is swallowed.
void audio_log(const char* name, const char* detail, const void* caller) noexcept {
    try {
        const std::string at = rt::startup_trace::where(reinterpret_cast<uint64_t>(caller));
        std::fprintf(stderr, "[rt] audio: %s: %s (caller %s)\n", name, detail, at.c_str());
        std::fflush(stderr);
        if (rt::startup_trace::enabled()) {
            rt::startup_trace::event("audio_stub", std::string("\"api\":") + rt::startup_trace::quote(name) +
                                                       ",\"stub\":" + rt::startup_trace::quote(detail) +
                                                       ",\"caller\":" + rt::startup_trace::quote(at));
        }
    } catch (...) {
    }
}

#define AUDIO_LOG_ONCE(name, detail)                                                                       \
    do {                                                                                                   \
        static std::atomic<bool> logged{false};                                                            \
        if (!logged.exchange(true, std::memory_order_relaxed)) audio_log((name), (detail), _ReturnAddress()); \
    } while (0)

// VirtualQuery-backed proof that a range is committed and has the requested access. The writes here are 8 bytes, so
// one region is enough; a range that crosses a region boundary is refused rather than guessed at.
bool range_is(uintptr_t address, size_t size, bool write) {
    if (size == 0) return true;
    if (address == 0 || address + size < address) return false;
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(reinterpret_cast<const void*>(address), &info, sizeof(info)) != sizeof(info)) return false;
    if (info.State != MEM_COMMIT) return false;
    const DWORD protection = info.Protect & 0xFFu;
    if (protection == PAGE_NOACCESS || (info.Protect & PAGE_GUARD) != 0) return false;
    const bool readable = protection == PAGE_READONLY || protection == PAGE_READWRITE ||
                          protection == PAGE_WRITECOPY || protection == PAGE_EXECUTE_READ ||
                          protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY;
    const bool writable = protection == PAGE_READWRITE || protection == PAGE_WRITECOPY ||
                          protection == PAGE_EXECUTE_READWRITE || protection == PAGE_EXECUTE_WRITECOPY;
    if (!readable || (write && !writable)) return false;
    const uintptr_t base = reinterpret_cast<uintptr_t>(info.BaseAddress);
    const uintptr_t end = base + info.RegionSize;
    if (end < base) return false;
    return address + size <= end;
}

// ---- AcpHalCreate's object ---------------------------------------------------------------------------------------
//
// The measured caller stores the first argument at 0x149ae98b8, calls AcpHalCreate with its address, then reads
// [0x149ae98b8] as an object and calls [obj], [obj+0x10]; the shutdown path calls [obj+8] and [obj+0x28]. A static
// object with a full page of S_OK methods covers all of them without any console state.
using HalMethod = int32_t(__stdcall*)(void*);

int32_t __stdcall HalNoop(void* /*self*/) { return kOk; }

struct HalVtable {
    HalMethod slot[64];
    constexpr HalVtable() : slot{} {
        for (auto& entry : slot) entry = &HalNoop;
    }
};
constexpr HalVtable kHalVtable{};

struct HalObject {
    const HalMethod* vtable;
    uint8_t body[0x1000 - sizeof(const HalMethod*)];
};
HalObject g_hal_object = {kHalVtable.slot, {}};

int32_t __stdcall AcpHalCreate(void* out) {
    AUDIO_LOG_ONCE("AcpHalCreate",
                   "writes a static fake object with a no-op vtable; S_OK when the out pointer is writable");
    if (!range_is(reinterpret_cast<uintptr_t>(out), sizeof(void*), true)) return kInvalidArg;
    void* object = &g_hal_object;
    std::memcpy(out, &object, sizeof object);
    return kOk;
}

// The measured config has the context count at +0x10 and the context-array pointer at +0x38; the caller memsets
// count * 64 bytes. Allocate exactly that, zeroed, and write the pointer only after proving it is writable.
constexpr uint32_t kShapeContextStride = 64;
constexpr uint32_t kMaxShapeContexts = 1u << 20;  // 64 MiB of contexts

int32_t __stdcall AcpHalAllocateShapeContexts(void* config) {
    AUDIO_LOG_ONCE("AcpHalAllocateShapeContexts",
                   "allocates the measured count * 64 context bytes and writes them at config+0x38; S_OK");
    if (!range_is(reinterpret_cast<uintptr_t>(config), 0x18, false)) return kInvalidArg;
    uint32_t count = 0;
    std::memcpy(&count, static_cast<const uint8_t*>(config) + 0x10, sizeof count);
    if (count == 0) count = 1;  // keep the pointer non-null even for an empty request
    if (count > kMaxShapeContexts) return kInvalidArg;
    const size_t bytes = static_cast<size_t>(count) * kShapeContextStride;
    void* buffer = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buffer) return kOutOfMemory;
    void* slot = static_cast<uint8_t*>(config) + 0x38;
    if (!range_is(reinterpret_cast<uintptr_t>(slot), sizeof buffer, true)) {
        VirtualFree(buffer, 0, MEM_RELEASE);
        return kInvalidArg;
    }
    std::memcpy(slot, &buffer, sizeof buffer);
    return kOk;
}

// The measured shutdown path calls this once, after the object's [vtable+0x28], and never reads the contexts again.
// Leaving the allocation to process exit keeps a late use impossible; the array is only count * 64 bytes.
int32_t __stdcall AcpHalReleaseShapeContexts() {
    AUDIO_LOG_ONCE("AcpHalReleaseShapeContexts", "accepts any state and returns S_OK; the context array is process-lifetime");
    return kOk;
}

// ---- ApuAlloc / ApuFree -------------------------------------------------------------------------------------------
//
// The measured wrapper at 0x1443a6110 calls ApuAlloc(out, 0, size, 0x800, flags): the pointer is the first argument,
// not the return value, and both call sites (0x1443aed13, 0x141e74620) treat the written pointer as ordinary memory.
// A null or fake pointer would be memset by the caller, so this returns zeroed host pages. The requested alignment is
// 0x800; VirtualAlloc is 64 KiB aligned, which satisfies every power-of-two alignment up to 64 KiB.
constexpr uint32_t kMaxApuAllocBytes = 256u * 1024u * 1024u;
constexpr size_t kMaxTrackedAllocations = 256;

struct Allocation {
    void* pointer;
    size_t size;
};
Allocation g_allocations[kMaxTrackedAllocations] = {};
std::mutex g_allocations_mutex;

void track_allocation(void* pointer, size_t size) {
    std::lock_guard lock(g_allocations_mutex);
    for (auto& entry : g_allocations) {
        if (!entry.pointer) {
            entry.pointer = pointer;
            entry.size = size;
            return;
        }
    }
}

bool untrack_allocation(void* pointer) {
    std::lock_guard lock(g_allocations_mutex);
    for (auto& entry : g_allocations) {
        if (entry.pointer == pointer) {
            entry.pointer = nullptr;
            entry.size = 0;
            return true;
        }
    }
    return false;
}

int32_t __stdcall ApuAlloc(void** out, uint32_t reserved, uint32_t size, uint32_t alignment, uint32_t flags) {
    AUDIO_LOG_ONCE("ApuAlloc",
                   "returns zeroed host memory through the measured out pointer; reserved and flags are ignored");
    (void)reserved;
    (void)flags;
    if (!range_is(reinterpret_cast<uintptr_t>(out), sizeof(void*), true)) return kInvalidArg;
    if (alignment != 0 && ((alignment & (alignment - 1)) != 0 || alignment > 0x10000)) return kInvalidArg;
    if (size > kMaxApuAllocBytes) return kOutOfMemory;
    const SIZE_T bytes = size ? static_cast<SIZE_T>(size) : 1;
    void* buffer = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buffer) return kOutOfMemory;
    track_allocation(buffer, static_cast<size_t>(bytes));
    std::memcpy(out, &buffer, sizeof buffer);
    return kOk;
}

int32_t __stdcall ApuFree(void* pointer) {
    AUDIO_LOG_ONCE("ApuFree", "accepts null, fake and unknown handles; only an ApuAlloc block is released");
    if (pointer && untrack_allocation(pointer)) VirtualFree(pointer, 0, MEM_RELEASE);
    return kOk;
}

int32_t __stdcall ApuCreateHeap(uint32_t size, uint32_t reserved) {
    AUDIO_LOG_ONCE("ApuCreateHeap", "accepts the size pair and returns S_OK; ApuAlloc supplies the memory");
    (void)size;
    (void)reserved;
    return kOk;
}

// A host pointer has no APU virtual address. TRUE here would make the engine store a fabricated APU address in its
// shape contexts and later hand it to the (absent) audio processor; FALSE leaves the "no mapping" branch in charge.
BOOL __stdcall ApuIsVirtualAddressValid(const void* address, uint32_t size) {
    AUDIO_LOG_ONCE("ApuIsVirtualAddressValid", "returns FALSE: host memory has no APU virtual mapping");
    (void)address;
    (void)size;
    return FALSE;
}

// Both measured callers test the result for null before adding an offset and copying. Null is the only safe answer
// without a real APU mapping; a fake non-null pointer would be written through.
void* __stdcall ApuMapApuAddress(uint32_t apu_address) {
    AUDIO_LOG_ONCE("ApuMapApuAddress", "returns null: no APU-to-host mapping exists; both callers check for null");
    (void)apu_address;
    return nullptr;
}

uint32_t __stdcall ApuMapVirtualAddress(const void* address) {
    AUDIO_LOG_ONCE("ApuMapVirtualAddress",
                   "returns 0: no host-to-APU mapping exists; not reached while ApuIsVirtualAddressValid is FALSE");
    (void)address;
    return 0;
}

// ---- xapu.dll ----------------------------------------------------------------------------------------------------
//
// The measured XApuConnect wrapper (0x1443aee50) passes three pointers, the last one the game's state block at
// 0x149ae8960, and treats a non-negative return as success. The game then passes the (zero) first qword of that block
// to XApuEnqueueCommand/GetResultCount/Disconnect; those shims accept the zero handle. Writing a fabricated handle
// into the block would invite code that dereferences it, so the pointers stay untouched.
int32_t __stdcall XApuConnect(void* arg1, void* arg2, void* arg3) {
    AUDIO_LOG_ONCE("XApuConnect", "returns S_OK without writing the three opaque pointers; the zero handle is accepted");
    (void)arg1;
    (void)arg2;
    (void)arg3;
    return kOk;
}

int32_t __stdcall XApuDisconnect(void* handle) {
    AUDIO_LOG_ONCE("XApuDisconnect", "accepts null and fake handles; returns S_OK");
    (void)handle;
    return kOk;
}

// 0x144eaa8a3 reads a result out of the buffer after a success. With GetResultCount always 0 the loop never reaches
// this, and a direct caller gets a negative "no result" status instead of an unwritten buffer it would parse.
int32_t __stdcall XApuDequeueResult(void* handle, void* out) {
    AUDIO_LOG_ONCE("XApuDequeueResult",
                   "returns HRESULT_FROM_WIN32(ERROR_NOT_FOUND): no result and the out buffer is untouched");
    (void)handle;
    (void)out;
    return kNoWork;
}

int32_t __stdcall XApuEnqueueCommand(void* handle, void* command, uint32_t flags) {
    AUDIO_LOG_ONCE("XApuEnqueueCommand", "accepts and ignores the command; returns S_OK");
    (void)handle;
    (void)command;
    (void)flags;
    return kOk;
}

int32_t __stdcall XApuGetResultCount(void* handle) {
    AUDIO_LOG_ONCE("XApuGetResultCount", "returns 0: no results are ever queued");
    (void)handle;
    return 0;
}

// ---- chad.dll ----------------------------------------------------------------------------------------------------
int32_t __stdcall XDspActivate(void* arg1, void* arg2, void* arg3, void* arg4) {
    AUDIO_LOG_ONCE("XDspActivate", "returns 0 (success); no DSP is started");
    (void)arg1;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    return kOk;
}

int32_t __stdcall XDspConnectWithMaximumStreamLimit(void* handle, uint32_t arg2, uint32_t arg3, uint32_t arg4,
                                                    uint32_t arg5, void* arg6) {
    AUDIO_LOG_ONCE("XDspConnectWithMaximumStreamLimit", "returns 0 (success); no stream is connected");
    (void)handle;
    (void)arg2;
    (void)arg3;
    (void)arg4;
    (void)arg5;
    (void)arg6;
    return kOk;
}

int32_t __stdcall XDspDeactivate(void* handle) {
    AUDIO_LOG_ONCE("XDspDeactivate", "accepts null and fake handles; returns 0");
    (void)handle;
    return kOk;
}

int32_t __stdcall XDspDisconnect(void* handle) {
    AUDIO_LOG_ONCE("XDspDisconnect", "accepts null and fake handles; returns 0");
    (void)handle;
    return kOk;
}

int32_t __stdcall XDspSubmitCommand(void* arg1, void* arg2) {
    AUDIO_LOG_ONCE("XDspSubmitCommand", "accepts and ignores the command; returns 0 (success)");
    (void)arg1;
    (void)arg2;
    return kOk;
}

}  // namespace

#define AUDIO_ENTRY(dll, name) {dll, #name, reinterpret_cast<void*>(&name)}

SHIM_TABLE(acphal,
    AUDIO_ENTRY("acphal.dll", AcpHalAllocateShapeContexts),
    AUDIO_ENTRY("acphal.dll", AcpHalCreate),
    AUDIO_ENTRY("acphal.dll", AcpHalReleaseShapeContexts),
    AUDIO_ENTRY("acphal.dll", ApuAlloc),
    AUDIO_ENTRY("acphal.dll", ApuCreateHeap),
    AUDIO_ENTRY("acphal.dll", ApuFree),
    AUDIO_ENTRY("acphal.dll", ApuIsVirtualAddressValid),
    AUDIO_ENTRY("acphal.dll", ApuMapApuAddress),
    AUDIO_ENTRY("acphal.dll", ApuMapVirtualAddress),
);

SHIM_TABLE(xapu,
    AUDIO_ENTRY("xapu.dll", XApuConnect),
    AUDIO_ENTRY("xapu.dll", XApuDequeueResult),
    AUDIO_ENTRY("xapu.dll", XApuDisconnect),
    AUDIO_ENTRY("xapu.dll", XApuEnqueueCommand),
    AUDIO_ENTRY("xapu.dll", XApuGetResultCount),
);

SHIM_TABLE(chad,
    AUDIO_ENTRY("chad.dll", XDspActivate),
    AUDIO_ENTRY("chad.dll", XDspConnectWithMaximumStreamLimit),
    AUDIO_ENTRY("chad.dll", XDspDeactivate),
    AUDIO_ENTRY("chad.dll", XDspDisconnect),
    AUDIO_ENTRY("chad.dll", XDspSubmitCommand),
);
