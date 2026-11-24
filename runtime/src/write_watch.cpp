// LEONIDA_WATCH_WRITE=<allocation size hex>:<offset hex>: a diagnostic that reports every write to one 8-byte word of
// the title's memory, with the writing instruction. The word is named by the size of the allocation it lies in and its
// offset from that allocation's base, because the base moves between runs while both stay the same (retail boot 15:
// an object at +0x41414E0 of a 0x25400000-byte allocation was overwritten with texels).
//
// A thread looks for the allocation (a committed MEM_PRIVATE allocation of exactly that size) until it exists, then puts
// a write breakpoint (DR0, 8 bytes) on the word in every thread of the process, new ones included, every 20 ms. The
// vectored handler logs each hit: the instruction after the write, the value now there, and the first return
// addresses on the stack. Hits past the first 64 are counted, not logged.
#include <windows.h>
#include <tlhelp32.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace rt::write_watch {
namespace {

std::atomic<std::uint64_t> g_address{0};
std::atomic<unsigned> g_hits{0};
std::uint64_t g_size = 0;
std::uint64_t g_offset = 0;
bool g_exec = false;  // LEONIDA_WATCH_EXEC: an execute breakpoint instead of a write watch

// The base of a committed private allocation of exactly g_size bytes, or 0.
std::uint64_t find_allocation() {
    std::uint8_t* at = nullptr;
    MEMORY_BASIC_INFORMATION info{};
    void* base = nullptr;
    std::uint64_t total = 0;
    while (VirtualQuery(at, &info, sizeof(info)) == sizeof(info)) {
        if (info.AllocationBase != base) {
            if (base && total == g_size) return reinterpret_cast<std::uint64_t>(base);
            base = info.AllocationBase;
            total = 0;
        }
        if (info.State != MEM_FREE && info.Type == MEM_PRIVATE) total += info.RegionSize;
        at = static_cast<std::uint8_t*>(info.BaseAddress) + info.RegionSize;
        if (reinterpret_cast<std::uint64_t>(at) >= 0x7FFFFFFF0000ull) break;
    }
    return base && total == g_size ? reinterpret_cast<std::uint64_t>(base) : 0;
}

void arm_all_threads(std::uint64_t address) {
    const DWORD self = GetCurrentThreadId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return;
    THREADENTRY32 entry{sizeof(entry)};
    for (BOOL ok = Thread32First(snapshot, &entry); ok; ok = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) continue;
        HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE,
                                   entry.th32ThreadID);
        if (!thread) continue;
        if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
            CONTEXT c{};
            c.ContextFlags = CONTEXT_DEBUG_REGISTERS;
            if (GetThreadContext(thread, &c) && c.Dr0 != address) {
                c.Dr0 = address;
                // L0, RW0 = 01 (write), LEN0 = 10 (8 bytes); an execute breakpoint has RW0 = LEN0 = 00.
                c.Dr7 = (c.Dr7 & ~0xF0003ull) | 1ull | (g_exec ? 0ull : (1ull << 16) | (2ull << 18));
                SetThreadContext(thread, &c);
            }
            ResumeThread(thread);
        }
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
}

DWORD WINAPI watcher(void*) {
    std::uint64_t base = 0;
    while (!(base = find_allocation())) Sleep(50);
    const std::uint64_t address = base + g_offset;
    g_address.store(address, std::memory_order_release);
    std::fprintf(stderr, "[rt] write-watch: allocation %p (0x%llx bytes), watching %p\n", reinterpret_cast<void*>(base),
                 static_cast<unsigned long long>(g_size), reinterpret_cast<void*>(address));
    for (;;) {
        arm_all_threads(address);
        Sleep(20);
    }
}

void dump_words(const char* name, DWORD64 object, int count) {
    MEMORY_BASIC_INFORMATION info{};
    if (object < 0x10000 || VirtualQuery(reinterpret_cast<void*>(object), &info, sizeof(info)) != sizeof(info) ||
        info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
        return;
    const auto* words = reinterpret_cast<const std::uint64_t*>(object);
    std::fprintf(stderr, "[rt] write-watch:   %s [%p]:", name, reinterpret_cast<void*>(object));
    for (int i = 0; i < count && object + 8 * (i + 1) <= reinterpret_cast<DWORD64>(info.BaseAddress) + info.RegionSize;
         ++i)
        std::fprintf(stderr, " %016llx", static_cast<unsigned long long>(words[i]));
    std::fprintf(stderr, "\n");
}

LONG CALLBACK on_single_step(EXCEPTION_POINTERS* info) {
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) return EXCEPTION_CONTINUE_SEARCH;
    CONTEXT& c = *info->ContextRecord;
    if (!(c.Dr6 & 1)) return EXCEPTION_CONTINUE_SEARCH;
    c.Dr6 = 0;
    if (g_exec) {
        c.EFlags |= 0x10000;  // RF: run the instruction without breaking again
        const unsigned hit = g_hits.fetch_add(1, std::memory_order_relaxed);
        if (hit < 16) {
            std::fprintf(stderr, "[rt] write-watch: exec hit %u rip=%p rcx=%p rdx=%p r8=%p r9=%p ret=%p\n", hit,
                         reinterpret_cast<void*>(c.Rip), reinterpret_cast<void*>(c.Rcx), reinterpret_cast<void*>(c.Rdx),
                         reinterpret_cast<void*>(c.R8), reinterpret_cast<void*>(c.R9),
                         *reinterpret_cast<void* const*>(c.Rsp));
            dump_words("rcx", c.Rcx, 16);
            dump_words("rdx", c.Rdx, 8);
            dump_words("rax", c.Rax, 16);
            dump_words("rsi", c.Rsi, 16);
            MEMORY_BASIC_INFORMATION mi{};
            if (c.Rdx > 0x10000 && VirtualQuery(reinterpret_cast<void*>(c.Rdx), &mi, sizeof(mi)) == sizeof(mi) &&
                mi.State == MEM_COMMIT && !(mi.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
                dump_words("[rdx+0x10]", *reinterpret_cast<const std::uint64_t*>(c.Rdx + 0x10), 12);
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    const std::uint64_t address = g_address.load(std::memory_order_acquire);
    const unsigned hit = g_hits.fetch_add(1, std::memory_order_relaxed);
    if (hit < 64 && address) {
        const auto* stack = reinterpret_cast<const std::uint64_t*>(c.Rsp);
        std::fprintf(stderr,
                     "[rt] write-watch: hit %u rip=%p value=%016llx tid=%lu rcx=%p rdx=%p r8=%p | stack %016llx %016llx "
                     "%016llx %016llx %016llx %016llx\n",
                     hit, reinterpret_cast<void*>(c.Rip),
                     static_cast<unsigned long long>(*reinterpret_cast<const std::uint64_t*>(address)),
                     GetCurrentThreadId(), reinterpret_cast<void*>(c.Rcx), reinterpret_cast<void*>(c.Rdx),
                     reinterpret_cast<void*>(c.R8), static_cast<unsigned long long>(stack[0]),
                     static_cast<unsigned long long>(stack[1]), static_cast<unsigned long long>(stack[2]),
                     static_cast<unsigned long long>(stack[3]), static_cast<unsigned long long>(stack[4]),
                     static_cast<unsigned long long>(stack[5]));
        std::fprintf(stderr, "[rt] write-watch:   stack[6..27]:");
        for (int i = 6; i < 28; ++i) std::fprintf(stderr, " %016llx", static_cast<unsigned long long>(stack[i]));
        std::fprintf(stderr, "\n");
        // The callee-saved registers, which in a leaf writer (memcpy) still hold the caller's state.
        std::fprintf(stderr, "[rt] write-watch:   rbx=%p rbp=%p rsi=%p rdi=%p r12=%p r13=%p r14=%p r15=%p\n",
                     reinterpret_cast<void*>(c.Rbx), reinterpret_cast<void*>(c.Rbp), reinterpret_cast<void*>(c.Rsi),
                     reinterpret_cast<void*>(c.Rdi), reinterpret_cast<void*>(c.R12), reinterpret_cast<void*>(c.R13),
                     reinterpret_cast<void*>(c.R14), reinterpret_cast<void*>(c.R15));
        const DWORD64 objects[] = {c.R14, c.Rbx};
        for (const DWORD64 object : objects) {
            MEMORY_BASIC_INFORMATION info{};
            if (object < 0x10000 || VirtualQuery(reinterpret_cast<void*>(object), &info, sizeof(info)) != sizeof(info) ||
                info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
                continue;
            const auto* words = reinterpret_cast<const std::uint64_t*>(object);
            std::fprintf(stderr, "[rt] write-watch:   [%p]:", reinterpret_cast<void*>(object));
            for (int i = 0; i < 12 && object + 8 * (i + 1) <= reinterpret_cast<DWORD64>(info.BaseAddress) + info.RegionSize;
                 ++i)
                std::fprintf(stderr, " %016llx", static_cast<unsigned long long>(words[i]));
            std::fprintf(stderr, "\n");
        }
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

}  // namespace

// Starts the watch when LEONIDA_WATCH_WRITE is set; a malformed value is reported and ignored.
void start_from_environment() {
    // LEONIDA_WATCH_EXEC=<address hex>: an execute breakpoint there; each hit logs the argument registers and the
    // words they point at (the first 16 hits).
    if (const char* exec = std::getenv("LEONIDA_WATCH_EXEC"); exec && *exec) {
        const std::uint64_t address = std::strtoull(exec, nullptr, 16);
        if (!address) return;
        g_exec = true;
        g_address.store(address, std::memory_order_release);
        AddVectoredExceptionHandler(1, &on_single_step);
        if (HANDLE t = CreateThread(
                nullptr, 0,
                [](void*) -> DWORD {
                    const std::uint64_t at = g_address.load(std::memory_order_acquire);
                    for (;;) {
                        arm_all_threads(at);
                        Sleep(20);
                    }
                },
                nullptr, 0, nullptr))
            CloseHandle(t);
        std::fprintf(stderr, "[rt] write-watch: execute breakpoint at %p\n", reinterpret_cast<void*>(address));
        return;
    }
    const char* spec = std::getenv("LEONIDA_WATCH_WRITE");
    if (!spec || !*spec) return;
    char* end = nullptr;
    g_size = std::strtoull(spec, &end, 16);
    if (!end || *end != ':' || !g_size) {
        std::fprintf(stderr, "[rt] write-watch: LEONIDA_WATCH_WRITE=%s is not <size hex>:<offset hex>; ignored\n", spec);
        return;
    }
    g_offset = std::strtoull(end + 1, nullptr, 16) & ~7ull;
    AddVectoredExceptionHandler(1, &on_single_step);
    if (HANDLE t = CreateThread(nullptr, 0, &watcher, nullptr, 0, nullptr)) CloseHandle(t);
    std::fprintf(stderr, "[rt] write-watch: looking for a 0x%llx-byte allocation, offset 0x%llx\n",
                 static_cast<unsigned long long>(g_size), static_cast<unsigned long long>(g_offset));
}

}  // namespace rt::write_watch
