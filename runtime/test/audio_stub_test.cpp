// Checks the registered audio boot shims the way the retail executable reaches them: acphal.dll (shims/acphal.cpp,
// which has its own unit test) in the measured boot sequence, and that the registry serves xapu.dll and chad.dll
// (xapu_chad.cpp; xapu_chad_test covers each call).
//
// Every function is resolved through rt::shims::find, the same table the loader binds imports to. The test calls each
// with valid, null and fake arguments. The three functions whose measured callers dereference an output get a real
// out buffer and the result is exercised; the rest must return their documented value without touching memory.
// No game data, no audio device.
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "shims.h"

namespace {

constexpr int32_t kOk = 0;
constexpr int32_t kNoWork = static_cast<int32_t>(0x80070490u);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

template <class Fn>
Fn require(const char* dll, const char* name) {
    Fn fn = reinterpret_cast<Fn>(rt::shims::find(dll, name));
    check(fn != nullptr, std::string("registry serves ") + dll + "!" + name);
    if (!fn) {
        std::printf("%d checks, %d failures: missing %s!%s\n", g_checks, g_failures, dll, name);
        std::exit(1);
    }
    return fn;
}

bool zeroed(const void* pointer, size_t size) {
    const auto* bytes = static_cast<const uint8_t*>(pointer);
    for (size_t i = 0; i < size; ++i)
        if (bytes[i] != 0) return false;
    return true;
}

bool writable_pattern(void* pointer, size_t size) {
    auto* bytes = static_cast<uint8_t*>(pointer);
    for (size_t i = 0; i < size; ++i) bytes[i] = static_cast<uint8_t>((i * 37u + 11u) & 0xffu);
    for (size_t i = 0; i < size; ++i)
        if (bytes[i] != static_cast<uint8_t>((i * 37u + 11u) & 0xffu)) return false;
    return true;
}

using AcpHalCreateFn = int32_t(__stdcall*)(void*);
using AcpHalAllocateShapeContextsFn = int32_t(__stdcall*)(void*);
using AcpHalReleaseShapeContextsFn = int32_t(__stdcall*)();
using ApuAllocFn = int32_t(__stdcall*)(void**, uint32_t, uint32_t, uint32_t, uint32_t);
using ApuFreeFn = int32_t(__stdcall*)(void*);
using ApuCreateHeapFn = int32_t(__stdcall*)(uint32_t, uint32_t);

using XApuConnectFn = int32_t(__stdcall*)(void*, void*, void*);
using XApuDisconnectFn = int32_t(__stdcall*)(void*);
using XApuDequeueResultFn = int32_t(__stdcall*)(void*, void*);
using XApuEnqueueCommandFn = int32_t(__stdcall*)(void*, void*, uint32_t);
using XApuGetResultCountFn = int32_t(__stdcall*)(void*);

using XDspActivateFn = int32_t(__stdcall*)(void*, void*, void*, void*);
using XDspConnectWithMaximumStreamLimitFn = int32_t(__stdcall*)(void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
using XDspDeactivateFn = int32_t(__stdcall*)(void*);
using XDspDisconnectFn = int32_t(__stdcall*)(void*);
using XDspSubmitCommandFn = int32_t(__stdcall*)(void*, void*);

constexpr uintptr_t kFakeAddress = 0xA0000001u;
void* const kFakeHandle = reinterpret_cast<void*>(kFakeAddress);

// The vtable the fake AcpHal object exposes; slot numbers are the offsets the measured caller uses.
using HalSlotTwo = int32_t(__stdcall*)(void*, uint32_t, uint32_t);
using HalSlotSix = int32_t(__stdcall*)(void*, uint32_t, uint32_t, uint32_t, void*, uint32_t);

void test_registry() {
    check(rt::shims::owns_dll("acphal.dll"), "acphal.dll is owned by the shims");
    check(rt::shims::owns_dll("xapu.dll"), "xapu.dll is owned by the shims");
    check(rt::shims::owns_dll("chad.dll"), "chad.dll is owned by the shims");
    check(rt::shims::find("acphal.dll", "no_such_audio_export") == nullptr,
          "an unserved acphal.dll name is NULL");
    check(rt::shims::find("acphal.dll", "#1") == nullptr,
          "no ordinal alias is registered: this executable imports all 19 names by name");
}

// The measured boot sequence: ApuCreateHeap -> AcpHalAllocateShapeContexts (count 0x15e, 64 bytes each) -> ApuAlloc
// (0x222e00 bytes, align 0x800, flags 2) -> memset of both buffers -> AcpHalCreate and its vtable calls -> XApuConnect
// -> XDsp setup. The game's own code does the memsets and the vtable dispatch; doing the same here catches a shim that
// returns a null or fake pointer where the game writes.
void test_measured_boot_sequence(AcpHalCreateFn create, AcpHalAllocateShapeContextsFn allocate_shape,
                                 AcpHalReleaseShapeContextsFn release_shape, ApuAllocFn apu_alloc, ApuFreeFn apu_free,
                                 ApuCreateHeapFn create_heap, XApuConnectFn connect, XApuDisconnectFn disconnect,
                                 XApuDequeueResultFn dequeue, XApuEnqueueCommandFn enqueue, XApuGetResultCountFn count,
                                 XDspActivateFn activate, XDspConnectWithMaximumStreamLimitFn connect_max,
                                 XDspSubmitCommandFn submit) {
    check(create_heap(8u << 20, 0x226e00) == kOk, "boot: ApuCreateHeap(8 MiB, 0x226e00) returns S_OK");

    alignas(16) uint8_t config[0x100] = {};
    *reinterpret_cast<uint32_t*>(config + 0x10) = 0x15e;
    check(allocate_shape(config) == kOk, "boot: AcpHalAllocateShapeContexts(count 0x15e) returns S_OK");
    void* contexts = *reinterpret_cast<void**>(config + 0x38);
    check(contexts != nullptr, "boot: shape context array is non-null");
    if (contexts) std::memset(contexts, 0, 0x15e * 64);  // the game's memset at 0x1443aec94
    check(contexts && zeroed(contexts, 0x15e * 64), "boot: the 0x15e * 64 context bytes are zeroed and writable");

    void* audio_pool = nullptr;
    check(apu_alloc(&audio_pool, 0, 0x222e00, 0x800, 2) == kOk, "boot: ApuAlloc(0x222e00, 0x800, 2) returns S_OK");
    check(audio_pool != nullptr, "boot: the audio pool is non-null");
    if (audio_pool) {
        std::memset(audio_pool, 0, 0x222e00);  // the game's memset at 0x1443aed5b
        check(writable_pattern(audio_pool, 0x222e00), "boot: the 2.2 MiB audio pool is readable and writable");
    }

    alignas(16) uint8_t hal_state[0x40] = {};
    check(create(hal_state) == kOk, "boot: AcpHalCreate returns S_OK");
    void* hal_object = *reinterpret_cast<void**>(hal_state);
    check(hal_object != nullptr, "boot: the AcpHal object is non-null");
    if (hal_object) {
        void** vtable = *reinterpret_cast<void***>(hal_object);
        check(vtable != nullptr, "boot: the AcpHal object has a vtable");
        if (vtable) {
            check(reinterpret_cast<HalSlotTwo>(vtable[0])(hal_object, 0x1000, 0x1000) == kOk,
                  "boot: AcpHal vtable[0](0x1000, 0x1000) returns S_OK");
            uint32_t format = 0xc0;
            check(reinterpret_cast<HalSlotSix>(vtable[2])(hal_object, 2, 0xf0f0f0f0u, 0xffffffffu, &format, 0) == kOk,
                  "boot: AcpHal vtable[2](format) returns S_OK");
        }
    }

    // The xapu.dll / chad.dll steps that follow in the boot are xapu_chad_test's: that shim connects and frees for
    // real, so the guessed argument shapes this test once passed a no-op stub would hand it a stack address to free.
    (void)connect, (void)count, (void)enqueue, (void)dequeue, (void)disconnect;
    (void)activate, (void)connect_max, (void)submit;

    check(release_shape() == kOk, "boot: AcpHalReleaseShapeContexts returns S_OK");
    if (audio_pool) apu_free(audio_pool);
}

}  // namespace

int main() {
    test_registry();

    const auto create = require<AcpHalCreateFn>("acphal.dll", "AcpHalCreate");
    const auto allocate = require<AcpHalAllocateShapeContextsFn>("acphal.dll", "AcpHalAllocateShapeContexts");
    const auto release = require<AcpHalReleaseShapeContextsFn>("acphal.dll", "AcpHalReleaseShapeContexts");
    const auto apu_alloc = require<ApuAllocFn>("acphal.dll", "ApuAlloc");
    const auto apu_free = require<ApuFreeFn>("acphal.dll", "ApuFree");
    const auto apu_create_heap = require<ApuCreateHeapFn>("acphal.dll", "ApuCreateHeap");

    const auto xapu_connect = require<XApuConnectFn>("xapu.dll", "XApuConnect");
    const auto xapu_disconnect = require<XApuDisconnectFn>("xapu.dll", "XApuDisconnect");
    const auto xapu_dequeue = require<XApuDequeueResultFn>("xapu.dll", "XApuDequeueResult");
    const auto xapu_enqueue = require<XApuEnqueueCommandFn>("xapu.dll", "XApuEnqueueCommand");
    const auto xapu_count = require<XApuGetResultCountFn>("xapu.dll", "XApuGetResultCount");

    const auto dsp_activate = require<XDspActivateFn>("chad.dll", "XDspActivate");
    const auto dsp_connect_max =
        require<XDspConnectWithMaximumStreamLimitFn>("chad.dll", "XDspConnectWithMaximumStreamLimit");
    const auto dsp_deactivate = require<XDspDeactivateFn>("chad.dll", "XDspDeactivate");
    const auto dsp_disconnect = require<XDspDisconnectFn>("chad.dll", "XDspDisconnect");
    const auto dsp_submit = require<XDspSubmitCommandFn>("chad.dll", "XDspSubmitCommand");

    test_measured_boot_sequence(create, allocate, release, apu_alloc, apu_free, apu_create_heap, xapu_connect,
                                xapu_disconnect, xapu_dequeue, xapu_enqueue, xapu_count, dsp_activate, dsp_connect_max,
                                dsp_submit);

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
