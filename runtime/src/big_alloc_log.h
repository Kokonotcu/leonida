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

}  // namespace leonida::big_alloc_log
