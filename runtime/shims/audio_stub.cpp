// xapu.dll, chad.dll: the console audio processor and DSP imports, silent and safe for boot.
//
// The retail executable imports 10 functions from these two modules. None has a host namesake, so every import used
// to bind to the loader's unimplemented thunk, which returns 0 and writes nothing. Each is a silent no-op returning
// the value its measured caller tests for. acphal.dll, the third module of the audio path, is shims/acphal.cpp's
// (upstream's APU heap and SHAPE contexts); this file once stubbed it too and gave that up when acphal.cpp landed.
//
// The RVAs below are from game_scarlett_master.exe (Xbox 1.0.2.1). The signatures beyond those call sites are not
// public; audio-stub-notes.md records what is measured, inferred or still unknown.
#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "shims.h"
#include "startup_trace.h"

namespace {

constexpr int32_t kOk = 0;
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
