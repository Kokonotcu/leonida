// Checks the registered audio boot shims (AcpHal.dll, xapu.dll, chad.dll) the way the retail executable reaches them.
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
constexpr int32_t kInvalidArg = static_cast<int32_t>(0x80070057u);
constexpr int32_t kOutOfMemory = static_cast<int32_t>(0x8007000Eu);
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
using ApuIsVirtualAddressValidFn = BOOL(__stdcall*)(const void*, uint32_t);
using ApuMapApuAddressFn = void*(__stdcall*)(uint32_t);
using ApuMapVirtualAddressFn = uint32_t(__stdcall*)(const void*);

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

void* fake_pointer(uintptr_t value) { return reinterpret_cast<void*>(value); }

constexpr uintptr_t kFakeAddress = 0xA0000001u;
void* const kFakeHandle = reinterpret_cast<void*>(kFakeAddress);

// The vtable the fake AcpHal object exposes; slot numbers are the offsets the measured caller uses.
using HalSlotSimple = int32_t(__stdcall*)(void*);
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

void test_acphal(AcpHalCreateFn create, AcpHalAllocateShapeContextsFn allocate,
                 AcpHalReleaseShapeContextsFn release) {
    alignas(16) uint8_t slot[0x40] = {};
    std::memset(slot, 0xcc, sizeof slot);
    check(create(nullptr) == kInvalidArg, "AcpHalCreate(NULL) returns E_INVALIDARG");
    check(create(kFakeHandle) == kInvalidArg, "AcpHalCreate(fake pointer) returns E_INVALIDARG");

    check(create(slot) == kOk, "AcpHalCreate(valid out) returns S_OK");
    void* object = *reinterpret_cast<void**>(slot);
    check(object != nullptr, "AcpHalCreate writes a non-null object");
    void** vtable = object ? *reinterpret_cast<void***>(object) : nullptr;
    check(vtable != nullptr, "the fake AcpHal object has a vtable");
    if (vtable) {
        check(reinterpret_cast<HalSlotTwo>(vtable[0])(object, 0x1000, 0x1000) == kOk,
              "AcpHal object vtable[0] returns S_OK");
        check(reinterpret_cast<HalSlotSimple>(vtable[1])(object) == kOk,
              "AcpHal object vtable[1] returns S_OK");
        uint32_t format = 0xc0;
        check(reinterpret_cast<HalSlotSix>(vtable[2])(object, 2, 0xf0f0f0f0u, 0xffffffffu, &format, 0) == kOk,
              "AcpHal object vtable[2] returns S_OK");
        check(reinterpret_cast<HalSlotSimple>(vtable[5])(object) == kOk,
              "AcpHal object vtable[5] returns S_OK");
    }

    check(allocate(nullptr) == kInvalidArg, "AcpHalAllocateShapeContexts(NULL) returns E_INVALIDARG");
    check(allocate(kFakeHandle) == kInvalidArg,
          "AcpHalAllocateShapeContexts(fake pointer) returns E_INVALIDARG");

    alignas(16) uint8_t config[0x100] = {};
    *reinterpret_cast<uint32_t*>(config + 0x10) = 4;
    *reinterpret_cast<void**>(config + 0x38) = fake_pointer(0x1234);
    check(allocate(config) == kOk, "AcpHalAllocateShapeContexts(count 4) returns S_OK");
    void* contexts = *reinterpret_cast<void**>(config + 0x38);
    check(contexts != nullptr && contexts != fake_pointer(0x1234),
          "AcpHalAllocateShapeContexts writes a non-null context array");
    if (contexts) {
        check(zeroed(contexts, 4 * 64), "the context array is zeroed");
        check(writable_pattern(contexts, 4 * 64), "the context array is readable and writable");
    }

    *reinterpret_cast<uint32_t*>(config + 0x10) = 0xffffffffu;
    void* before = *reinterpret_cast<void**>(config + 0x38);
    check(allocate(config) == kInvalidArg, "an absurd context count returns E_INVALIDARG");
    check(*reinterpret_cast<void**>(config + 0x38) == before,
          "the context out pointer is unchanged after a rejected count");

    *reinterpret_cast<uint32_t*>(config + 0x10) = 0;
    check(allocate(config) == kOk, "a zero context count returns S_OK");
    check(*reinterpret_cast<void**>(config + 0x38) != nullptr, "a zero context count still writes a non-null array");

    check(release() == kOk, "AcpHalReleaseShapeContexts returns S_OK");
    check(release() == kOk, "AcpHalReleaseShapeContexts is repeatable");
}

void test_apu(ApuAllocFn allocate, ApuFreeFn free_block, ApuCreateHeapFn create_heap,
              ApuIsVirtualAddressValidFn valid, ApuMapApuAddressFn map_apu, ApuMapVirtualAddressFn map_virtual) {
    void* block = fake_pointer(0x1234);
    check(allocate(nullptr, 0, 0x1000, 0x800, 0) == kInvalidArg, "ApuAlloc(NULL out) returns E_INVALIDARG");
    check(allocate(reinterpret_cast<void**>(const_cast<void*>(kFakeHandle)), 0, 0x1000, 0x800, 0) == kInvalidArg,
          "ApuAlloc(fake out) returns E_INVALIDARG");

    check(allocate(&block, 0, 0x1000, 0x800, 0) == kOk, "ApuAlloc(4096, align 0x800) returns S_OK");
    check(block != nullptr && block != fake_pointer(0x1234), "ApuAlloc writes a non-null pointer");
    check((reinterpret_cast<uintptr_t>(block) & 0xfffu) == 0, "ApuAlloc returns a page-aligned pointer");
    check(zeroed(block, 0x1000), "ApuAlloc memory is zeroed");
    check(writable_pattern(block, 0x1000), "ApuAlloc memory is readable and writable");
    check(free_block(block) == kOk, "ApuFree releases an ApuAlloc block");
    check(free_block(block) == kOk, "ApuFree is safe to repeat");
    check(free_block(nullptr) == kOk, "ApuFree(NULL) returns S_OK");
    check(free_block(kFakeHandle) == kOk, "ApuFree(fake handle) returns S_OK");

    void* empty = nullptr;
    check(allocate(&empty, 0, 0, 0x800, 0) == kOk, "ApuAlloc(size 0) returns S_OK");
    check(empty != nullptr, "ApuAlloc(size 0) still writes a non-null pointer");
    check(free_block(empty) == kOk, "ApuFree releases the size-0 block");

    void* untouched = fake_pointer(0x1234);
    check(allocate(&untouched, 0, 0xffffffffu, 0x800, 0) == kOutOfMemory,
          "ApuAlloc(4 GiB) returns E_OUTOFMEMORY");
    check(untouched == fake_pointer(0x1234), "the ApuAlloc out pointer is unchanged on failure");

    untouched = fake_pointer(0x1234);
    check(allocate(&untouched, 0, 0x1000, 3, 0) == kInvalidArg, "ApuAlloc(non-power-of-two alignment) is rejected");
    check(untouched == fake_pointer(0x1234), "the ApuAlloc out pointer is unchanged after bad alignment");

    void* aligned = nullptr;
    check(allocate(&aligned, 0, 0x1000, 0x10000, 0) == kOk, "ApuAlloc(align 64 KiB) returns S_OK");
    check(aligned && (reinterpret_cast<uintptr_t>(aligned) & 0xffffu) == 0,
          "ApuAlloc(align 64 KiB) returns a 64 KiB-aligned pointer");
    if (aligned) free_block(aligned);

    check(create_heap(0, 0x226e00) == kOk, "ApuCreateHeap returns S_OK");
    check(create_heap(0xffffffffu, 0xffffffffu) == kOk, "ApuCreateHeap accepts arbitrary size words");

    int local = 0;
    check(valid(&local, sizeof local) == FALSE, "ApuIsVirtualAddressValid returns FALSE for host memory");
    check(valid(nullptr, 0) == FALSE, "ApuIsVirtualAddressValid(NULL) returns FALSE");
    check(map_apu(0x1234) == nullptr, "ApuMapApuAddress returns null");
    check(map_virtual(&local) == 0, "ApuMapVirtualAddress returns 0");
}

void test_xapu(XApuConnectFn connect, XApuDisconnectFn disconnect, XApuDequeueResultFn dequeue,
               XApuEnqueueCommandFn enqueue, XApuGetResultCountFn count) {
    check(connect(nullptr, nullptr, nullptr) == kOk, "XApuConnect(NULLs) returns S_OK");
    check(connect(kFakeHandle, kFakeHandle, kFakeHandle) == kOk, "XApuConnect(fake pointers) returns S_OK");

    check(disconnect(nullptr) == kOk, "XApuDisconnect(NULL) returns S_OK");
    check(disconnect(kFakeHandle) == kOk, "XApuDisconnect(fake handle) returns S_OK");

    check(count(nullptr) == 0, "XApuGetResultCount(NULL) returns 0");
    check(count(kFakeHandle) == 0, "XApuGetResultCount(fake handle) returns 0");

    uint8_t result[16];
    std::memset(result, 0xcc, sizeof result);
    check(dequeue(nullptr, result) == kNoWork, "XApuDequeueResult(NULL handle) returns no-work");
    check(result[0] == 0xcc && result[15] == 0xcc, "XApuDequeueResult leaves the result buffer untouched");
    check(dequeue(kFakeHandle, result) == kNoWork, "XApuDequeueResult(fake handle) returns no-work");
    check(dequeue(nullptr, nullptr) == kNoWork, "XApuDequeueResult(NULL out) returns no-work without writing");

    uint8_t command[32];
    std::memset(command, 0xaa, sizeof command);
    check(enqueue(kFakeHandle, command, 0) == kOk, "XApuEnqueueCommand returns S_OK");
    check(command[0] == 0xaa && command[31] == 0xaa, "XApuEnqueueCommand leaves the command untouched");
}

void test_chad(XDspActivateFn activate, XDspConnectWithMaximumStreamLimitFn connect_max, XDspDeactivateFn deactivate,
               XDspDisconnectFn disconnect, XDspSubmitCommandFn submit) {
    check(activate(nullptr, nullptr, nullptr, nullptr) == kOk, "XDspActivate(NULLs) returns 0");
    check(activate(kFakeHandle, kFakeHandle, kFakeHandle, kFakeHandle) == kOk,
          "XDspActivate(fake handles) returns 0");

    int state = 0;
    check(connect_max(kFakeHandle, 0, 0x80, 0x80, 1, &state) == kOk,
          "XDspConnectWithMaximumStreamLimit returns 0");
    check(connect_max(nullptr, 0, 0, 0, 0, nullptr) == kOk,
          "XDspConnectWithMaximumStreamLimit(NULLs) returns 0");

    check(deactivate(nullptr) == kOk, "XDspDeactivate(NULL) returns 0");
    check(deactivate(kFakeHandle) == kOk, "XDspDeactivate(fake handle) returns 0");
    check(disconnect(nullptr) == kOk, "XDspDisconnect(NULL) returns 0");
    check(disconnect(kFakeHandle) == kOk, "XDspDisconnect(fake handle) returns 0");

    uint8_t command[32];
    std::memset(command, 0xbb, sizeof command);
    check(submit(command, kFakeHandle) == kOk, "XDspSubmitCommand returns 0");
    check(command[0] == 0xbb && command[31] == 0xbb, "XDspSubmitCommand leaves the command untouched");
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

    alignas(16) uint8_t state_block[0x400] = {};
    check(connect(state_block, state_block + 0x100, state_block + 0x200) == kOk, "boot: XApuConnect returns S_OK");
    check(count(state_block + 0x200) == 0, "boot: XApuGetResultCount returns 0");
    uint8_t command[0x40] = {};
    check(enqueue(state_block + 0x200, command, 0) == kOk, "boot: XApuEnqueueCommand returns S_OK");
    uint8_t result[0x20] = {};
    check(dequeue(state_block + 0x200, result) == kNoWork, "boot: XApuDequeueResult returns no-work");
    check(disconnect(state_block + 0x200) == kOk, "boot: XApuDisconnect returns S_OK");

    check(activate(state_block, state_block + 0x100, state_block + 0x200, state_block + 0x300) == kOk,
          "boot: XDspActivate returns 0");
    check(connect_max(state_block, 0, 0x80, 0x80, 1, state_block + 0x300) == kOk,
          "boot: XDspConnectWithMaximumStreamLimit returns 0");
    check(submit(command, state_block) == kOk, "boot: XDspSubmitCommand returns 0");

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
    const auto apu_valid = require<ApuIsVirtualAddressValidFn>("acphal.dll", "ApuIsVirtualAddressValid");
    const auto apu_map = require<ApuMapApuAddressFn>("acphal.dll", "ApuMapApuAddress");
    const auto apu_map_virtual = require<ApuMapVirtualAddressFn>("acphal.dll", "ApuMapVirtualAddress");

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

    test_acphal(create, allocate, release);
    test_apu(apu_alloc, apu_free, apu_create_heap, apu_valid, apu_map, apu_map_virtual);
    test_xapu(xapu_connect, xapu_disconnect, xapu_dequeue, xapu_enqueue, xapu_count);
    test_chad(dsp_activate, dsp_connect_max, dsp_deactivate, dsp_disconnect, dsp_submit);
    test_measured_boot_sequence(create, allocate, release, apu_alloc, apu_free, apu_create_heap, xapu_connect,
                                xapu_disconnect, xapu_dequeue, xapu_enqueue, xapu_count, dsp_activate, dsp_connect_max,
                                dsp_submit);

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
