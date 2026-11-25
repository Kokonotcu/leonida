// LEONIDA_LOG_BIG_ALLOCS=1: every reservation or commit of 256 MiB or more anywhere in the process (game, runtime,
// driver), with the first caller outside ntdll, kernelbase and kernel32, logged (the first 64). Diagnostics for the
// process's commit: in the first loading screen it holds seven 1 GiB and two 2 GiB private blocks that neither the
// title's own allocation calls (xmem's shims) nor the runtime's large Vulkan allocations account for.
//
// Every allocation in the process ends in ntdll!NtAllocateVirtualMemory, a system-call stub (mov r10, rcx; mov eax,
// <number>; ... syscall; ret). Its first 12 bytes are replaced by a jump to log_then_allocate, which makes the same
// system call through a stub of its own (mov r10, rcx; mov eax, <number>; syscall; ret). x64 Windows only; nothing is
// installed unless the stub has that shape.
#pragma once

#include <windows.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace leonida::big_alloc_log {

using NtAllocate = LONG(NTAPI*)(HANDLE, PVOID*, ULONG_PTR, PSIZE_T, ULONG, ULONG);

inline NtAllocate g_syscall = nullptr;  // our own stub with the same system-call number
inline std::atomic<int> g_told{0};
inline thread_local bool g_inside = false;  // a plain bool: no destructor (a TLS callback hangs our loader)

// The first return address outside the system DLLs, as module+offset.
inline void describe_caller(char* out, std::size_t size) {
    void* frames[24] = {};
    const USHORT n = RtlCaptureStackBackTrace(2, 24, frames, nullptr);
    const HMODULE system[3] = {GetModuleHandleW(L"ntdll.dll"), GetModuleHandleW(L"kernelbase.dll"),
                               GetModuleHandleW(L"kernel32.dll")};
    for (USHORT i = 0; i < n; ++i) {
        HMODULE module = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                static_cast<LPCWSTR>(frames[i]), &module)) {
            std::snprintf(out, size, "%p (no module: the title's code or a thunk)", frames[i]);
            return;
        }
        if (module == system[0] || module == system[1] || module == system[2]) continue;
        char name[MAX_PATH] = {};
        GetModuleFileNameA(module, name, MAX_PATH);
        const char* base = std::strrchr(name, '\\');
        std::snprintf(out, size, "%s+0x%llx", base ? base + 1 : name,
                      static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(frames[i]) -
                                                      reinterpret_cast<std::uintptr_t>(module)));
        return;
    }
    std::snprintf(out, size, "(only system frames)");
}

inline LONG NTAPI log_then_allocate(HANDLE process, PVOID* base, ULONG_PTR zero_bits, PSIZE_T size, ULONG type,
                                    ULONG protect) {
    const SIZE_T asked = size ? *size : 0;
    const LONG status = g_syscall(process, base, zero_bits, size, type, protect);
    if (asked >= (SIZE_T(256) << 20) && !g_inside && g_told.load(std::memory_order_relaxed) < 64) {
        g_inside = true;
        g_told.fetch_add(1, std::memory_order_relaxed);
        char caller[MAX_PATH + 64] = {};
        describe_caller(caller, sizeof caller);
        std::fprintf(stderr, "[rt] big allocation: %llu MiB, type 0x%lx, protect 0x%lx, status 0x%lx, at %p, from %s\n",
                     static_cast<unsigned long long>(asked >> 20), type, protect, static_cast<unsigned long>(status),
                     base ? *base : nullptr, caller);
        g_inside = false;
    }
    return status;
}

// Installs the hook when LEONIDA_LOG_BIG_ALLOCS=1. Call once, early, before other threads allocate much.
inline void install() {
    char value[4] = {};
    if (GetEnvironmentVariableA("LEONIDA_LOG_BIG_ALLOCS", value, sizeof value) != 1 || value[0] != '1') return;
    auto* target = reinterpret_cast<std::uint8_t*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtAllocateVirtualMemory"));
    // 4C 8B D1 (mov r10, rcx), B8 xx xx xx xx (mov eax, number): the Windows 10/11 x64 stub.
    if (!target || target[0] != 0x4C || target[1] != 0x8B || target[2] != 0xD1 || target[3] != 0xB8) {
        std::fprintf(stderr, "[rt] big allocation log: NtAllocateVirtualMemory has an unknown shape; not installed\n");
        return;
    }
    std::uint32_t number = 0;
    std::memcpy(&number, target + 4, 4);
    auto* stub = static_cast<std::uint8_t*>(VirtualAlloc(nullptr, 64, MEM_RESERVE | MEM_COMMIT, PAGE_EXECUTE_READWRITE));
    if (!stub) return;
    const std::uint8_t code[] = {0x4C, 0x8B, 0xD1, 0xB8, 0, 0, 0, 0, 0x0F, 0x05, 0xC3};  // mov r10,rcx; mov eax,n; syscall; ret
    std::memcpy(stub, code, sizeof code);
    std::memcpy(stub + 4, &number, 4);
    g_syscall = reinterpret_cast<NtAllocate>(stub);
    // mov rax, imm64; jmp rax (12 bytes) over the start of the stub.
    std::uint8_t jump[12] = {0x48, 0xB8, 0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xE0};
    const auto handler = reinterpret_cast<std::uint64_t>(&log_then_allocate);
    std::memcpy(jump + 2, &handler, 8);
    DWORD old = 0;
    if (!VirtualProtect(target, sizeof jump, PAGE_EXECUTE_READWRITE, &old)) return;
    std::memcpy(target, jump, sizeof jump);
    VirtualProtect(target, sizeof jump, old, &old);
    FlushInstructionCache(GetCurrentProcess(), target, sizeof jump);
    std::fprintf(stderr, "[rt] big allocation log: on (NtAllocateVirtualMemory, system call 0x%x)\n", number);
}

// LEONIDA_LOG_BIG_ALLOCS=2: the same question without patching code. The import tables of the NVIDIA driver, the
// runtime, the CRT and the Vulkan loader get wrappers for VirtualAlloc, VirtualAllocEx and HeapAlloc; an allocation of
// 8 MiB or more is logged (the first 600) with six return addresses as module+offset, from a module list taken at
// install (no loader lock while logging). For the first loading screen's 1 GiB private arenas and the growing
// 0xFD0000-byte blocks, which no caller of ours accounts for. Install after the Vulkan device exists (the driver's
// module is loaded by then).
namespace iat {

struct Module {
    std::uintptr_t base = 0, end = 0;
    char name[64] = {};
};
inline Module g_modules[512];
inline int g_module_count = 0;
inline std::atomic<int> g_logged{0};
using VirtualAllocFn = LPVOID(WINAPI*)(LPVOID, SIZE_T, DWORD, DWORD);
using VirtualAllocExFn = LPVOID(WINAPI*)(HANDLE, LPVOID, SIZE_T, DWORD, DWORD);
using HeapAllocFn = LPVOID(WINAPI*)(HANDLE, DWORD, SIZE_T);
inline VirtualAllocFn g_virtual_alloc = nullptr;
inline VirtualAllocExFn g_virtual_alloc_ex = nullptr;
inline HeapAllocFn g_heap_alloc = nullptr;
constexpr SIZE_T kThreshold = SIZE_T(8) << 20;

inline void name_of(std::uintptr_t address, char* out, std::size_t size) {
    for (int i = 0; i < g_module_count; ++i)
        if (address >= g_modules[i].base && address < g_modules[i].end) {
            std::snprintf(out, size, "%s+0x%llx", g_modules[i].name,
                          static_cast<unsigned long long>(address - g_modules[i].base));
            return;
        }
    std::snprintf(out, size, "%p", reinterpret_cast<void*>(address));
}

inline void log(const char* what, SIZE_T size, DWORD type, void* result) {
    if (g_logged.fetch_add(1, std::memory_order_relaxed) >= 600) return;
    void* frames[8] = {};
    const USHORT n = RtlCaptureStackBackTrace(2, 8, frames, nullptr);
    char text[1024];
    int at = std::snprintf(text, sizeof text, "[rt] big allocation (iat): %s %llu KiB type 0x%lx at %p from", what,
                           static_cast<unsigned long long>(size >> 10), type, result);
    for (USHORT i = 0; i < n && i < 6 && at > 0 && std::size_t(at) < sizeof text - 96; ++i) {
        char where[96];
        name_of(reinterpret_cast<std::uintptr_t>(frames[i]), where, sizeof where);
        at += std::snprintf(text + at, sizeof text - at, " %s", where);
    }
    std::fprintf(stderr, "%s\n", text);
}

inline LPVOID WINAPI virtual_alloc(LPVOID address, SIZE_T size, DWORD type, DWORD protect) {
    LPVOID r = g_virtual_alloc(address, size, type, protect);
    if (size >= kThreshold) log("VirtualAlloc", size, type, r);
    return r;
}
inline LPVOID WINAPI virtual_alloc_ex(HANDLE process, LPVOID address, SIZE_T size, DWORD type, DWORD protect) {
    LPVOID r = g_virtual_alloc_ex(process, address, size, type, protect);
    if (size >= kThreshold) log("VirtualAllocEx", size, type, r);
    return r;
}
inline LPVOID WINAPI heap_alloc(HANDLE heap, DWORD flags, SIZE_T size) {
    LPVOID r = g_heap_alloc(heap, flags, size);
    if (size >= kThreshold) log("HeapAlloc", size, 0, r);
    return r;
}

// Every import of `module` named `name` (any DLL) now calls `hook`.
inline int patch(HMODULE module, const char* name, void* hook) {
    auto* base = reinterpret_cast<std::uint8_t*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return 0;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) return 0;
    int patched = 0;
    for (auto* d = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress); d->Name; ++d) {
        if (!d->OriginalFirstThunk) continue;
        auto* names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(base + d->OriginalFirstThunk);
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + d->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
            const auto* by_name = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (std::strcmp(reinterpret_cast<const char*>(by_name->Name), name) != 0) continue;
            DWORD old = 0;
            if (!VirtualProtect(&slots->u1.Function, sizeof(slots->u1.Function), PAGE_READWRITE, &old)) continue;
            slots->u1.Function = reinterpret_cast<ULONGLONG>(hook);
            VirtualProtect(&slots->u1.Function, sizeof(slots->u1.Function), old, &old);
            ++patched;
        }
    }
    return patched;
}

inline void install() {
    char value[4] = {};
    if (GetEnvironmentVariableA("LEONIDA_LOG_BIG_ALLOCS", value, sizeof value) != 1 || value[0] != '2') return;
    static std::atomic<bool> once{false};
    if (once.exchange(true)) return;
    const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
    g_virtual_alloc = reinterpret_cast<VirtualAllocFn>(GetProcAddress(kernel, "VirtualAlloc"));
    g_virtual_alloc_ex = reinterpret_cast<VirtualAllocExFn>(GetProcAddress(kernel, "VirtualAllocEx"));
    g_heap_alloc = reinterpret_cast<HeapAllocFn>(GetProcAddress(kernel, "HeapAlloc"));
    if (!g_virtual_alloc || !g_virtual_alloc_ex || !g_heap_alloc) return;
    // The module list, once (name_of reads it without the loader lock).
    HMODULE modules[512] = {};
    DWORD needed = 0;
    using EnumFn = BOOL(WINAPI*)(HANDLE, HMODULE*, DWORD, LPDWORD);
    const auto enumerate = reinterpret_cast<EnumFn>(GetProcAddress(kernel, "K32EnumProcessModules"));
    if (enumerate && enumerate(GetCurrentProcess(), modules, sizeof modules, &needed)) {
        for (DWORD i = 0; i < needed / sizeof(HMODULE) && g_module_count < 512; ++i) {
            const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(modules[i]);
            const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(reinterpret_cast<const std::uint8_t*>(dos) +
                                                                          dos->e_lfanew);
            Module& m = g_modules[g_module_count++];
            m.base = reinterpret_cast<std::uintptr_t>(modules[i]);
            m.end = m.base + nt->OptionalHeader.SizeOfImage;
            char path[MAX_PATH] = {};
            GetModuleFileNameA(modules[i], path, MAX_PATH);
            const char* leaf = std::strrchr(path, '\\');
            std::snprintf(m.name, sizeof m.name, "%s", leaf ? leaf + 1 : path);
        }
    }
    // Every module loaded now but the ones the hooks call through (their own imports would recurse into the log).
    int patched = 0;
    for (int i = 0; i < g_module_count; ++i) {
        const char* n = g_modules[i].name;
        if (!_stricmp(n, "ntdll.dll") || !_stricmp(n, "kernelbase.dll") || !_stricmp(n, "kernel32.dll")) continue;
        const auto m = reinterpret_cast<HMODULE>(g_modules[i].base);
        patched += patch(m, "VirtualAlloc", reinterpret_cast<void*>(&virtual_alloc));
        patched += patch(m, "VirtualAllocEx", reinterpret_cast<void*>(&virtual_alloc_ex));
        patched += patch(m, "HeapAlloc", reinterpret_cast<void*>(&heap_alloc));
    }
    std::fprintf(stderr, "[rt] big allocation log (import tables): %d imports patched, %d modules listed\n", patched,
                 g_module_count);
}

}  // namespace iat

}  // namespace leonida::big_alloc_log
