// gta6rt: native runtime for the Xbox Series build.
//
//   gta6rt --dry-run <image> [--game-dir DIR] [--overlay DIR]
//       Maps the image and its bundled modules, applies relocations, binds every import, installs static TLS for
//       this thread and prints a report. No code from the mapped images
//       runs. TLS needs the runtime's own gta6rt_tls_host.dll, which reserves the module TLS index; system DLLs are
//       loaded to resolve host imports, as before.
//
//   gta6rt --init <image> [--game-dir DIR] [--overlay DIR]
//       The same mapping, then each loaded DLL's TLS callbacks and DllMain(DLL_PROCESS_ATTACH) are run.
//
//   gta6rt --precompile-pipelines [--threads N] [--big-threads N] [--big-bytes N] [--all] [--smallest-first]
//          [--limit N] [--min-bytes N] [--max-bytes N] [--filter PART]... [--list]
//       No game: creates the Vulkan device as --run does and builds every compute pipeline of the shader library
//       (LEONIDA_SHADERS) into the pipeline journal and the driver's cache, so a title's first dispatch of a large one
//       does not stall it. Resumable: entries already journaled are skipped (--all: built anyway). See
//       shims/d3d12_precompile.h.
//
//   gta6rt --run <exe> [--game-dir DIR] [--overlay DIR] [--trace-d3d12] [--trace FILE] [-- game arguments...]
//       The same, then the EXE's own TLS callbacks and its entry point, on a game thread. The game's argv and
//       command line are <exe name> plus the arguments after "--"; exit/ExitProcess from the game end the run
//       (shims/process.cpp). gta6rt prints the binding report first and returns the game's exit code.
//
// --game-dir is also where a title's GAME: paths read from; anything it writes or deletes goes to the overlay
// (--overlay, default <game-dir>/overlay) so the package is never modified (shims/file_system.cpp).
#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <exception>
#include <filesystem>
#include <malloc.h>
#include <map>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>

namespace rt::write_watch {
void start_from_environment();
}
#include <cstring>

namespace {

// The retail executable (1.0.2.1) runs a hang watchdog thread (function at RVA 0x2215a60): when the game's main loop
// has not set its "kick" byte (RVA 0x71b6331) for 30 one-second ticks, it calls a crash callback and then writes to
// address 0 (`movl $0, 0` at RVA 0x2215c35). Under the runtime the game thread can legitimately stall for longer than
// that, e.g. while the driver compiles pipelines against a cold cache (first boot after a driver update), and the
// title then kills itself. By default the watchdog stays armed, as on the console. With LEONIDA_TITLE_WATCHDOG=0, and
// only when that exact trap instruction sits where it does in this build, a runtime thread sets the kick byte twice a
// second so the title's watchdog never fires.
constexpr std::uint32_t kWatchdogTrapRva = 0x2215c35;
constexpr std::uint32_t kWatchdogKickRva = 0x71b6331;
constexpr unsigned char kWatchdogTrap[] = {0xC7, 0x04, 0x25, 0, 0, 0, 0, 0, 0, 0, 0};  // movl $0, 0x0

void keep_title_watchdog_kicked(std::uint8_t* base) {
    char value[4]{};
    if (GetEnvironmentVariableA("LEONIDA_TITLE_WATCHDOG", value, sizeof(value)) != 1 || value[0] != '0') return;
    MEMORY_BASIC_INFORMATION trap{}, kick{};
    if (!VirtualQuery(base + kWatchdogTrapRva, &trap, sizeof(trap)) || trap.State != MEM_COMMIT ||
        std::memcmp(base + kWatchdogTrapRva, kWatchdogTrap, sizeof(kWatchdogTrap)) != 0)
        return;  // not the build this was measured on: leave the title's watchdog alone
    const DWORD writable = PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    if (!VirtualQuery(base + kWatchdogKickRva, &kick, sizeof(kick)) || kick.State != MEM_COMMIT ||
        !(kick.Protect & writable))
        return;
    volatile std::uint8_t* const byte = base + kWatchdogKickRva;
    const HANDLE thread = CreateThread(
        nullptr, 0,
        [](LPVOID p) -> DWORD {
            auto* b = static_cast<volatile std::uint8_t*>(p);
            for (;;) {
                *b = 1;
                Sleep(500);
            }
        },
        const_cast<std::uint8_t*>(byte), 0, nullptr);
    if (thread) {
        CloseHandle(thread);
        std::fprintf(stderr, "[rt] loader: keeping the title's hang watchdog kicked (LEONIDA_TITLE_WATCHDOG=0; unset or 1 keeps it armed)\n");
    }
}

// LEONIDA_PEEK="0x9e1dac0:3,0x7446510,0x98f828c" (diagnostic): a thread prints the 8-byte values at these RVAs of the
// title's image (":n" = n consecutive values, at most 32; "RVA>OFF" reads through the pointer stored at the RVA, at
// pointer + OFF, both hex) every LEONIDA_PEEK_SECONDS (default 10). For answering "what does this
// global hold at run time" without a debugger: the reverse-engineering notes (docs/re/requests.md) ask for exactly
// that. Reads are checked; nothing is written.
void start_title_peek(std::uint8_t* base) {
    static char spec[1024];
    const DWORD n = GetEnvironmentVariableA("LEONIDA_PEEK", spec, sizeof(spec));
    if (!n || n >= sizeof(spec)) return;
    static std::uint8_t* image = base;
    const HANDLE thread = CreateThread(
        nullptr, 0,
        [](LPVOID) -> DWORD {
            char seconds[16]{};
            DWORD period = 10;
            if (GetEnvironmentVariableA("LEONIDA_PEEK_SECONDS", seconds, sizeof(seconds)) > 0)
                period = std::max<DWORD>(1, static_cast<DWORD>(std::strtoul(seconds, nullptr, 10)));
            for (unsigned tick = 0;; ++tick) {
                std::string line = "[rt] peek t=" + std::to_string(tick * period) + "s:";
                for (const char* at = spec; *at;) {
                    char* end = nullptr;
                    const std::uint64_t rva = std::strtoull(at, &end, 16);
                    if (end == at) break;
                    // "RVA>OFF": the 8-byte pointer at the RVA, then the values at pointer + OFF (both hex).
                    bool deref = false;
                    std::uint64_t offset = 0;
                    if (*end == '>') {
                        deref = true;
                        offset = std::strtoull(end + 1, &end, 16);
                    }
                    unsigned count = 1;
                    if (*end == ':') count = static_cast<unsigned>(std::strtoul(end + 1, &end, 10));
                    char text[48];
                    if (deref)
                        std::snprintf(text, sizeof(text), " 0x%llx>%llx=", static_cast<unsigned long long>(rva),
                                      static_cast<unsigned long long>(offset));
                    else
                        std::snprintf(text, sizeof(text), " 0x%llx=", static_cast<unsigned long long>(rva));
                    line += text;
                    const std::uint8_t* from = image + rva;
                    if (deref) {
                        std::uint64_t pointer = 0;
                        SIZE_T got = 0;
                        if (!ReadProcessMemory(GetCurrentProcess(), from, &pointer, sizeof(pointer), &got) ||
                            got != sizeof(pointer) || !pointer)
                            count = 0, line += "?";
                        from = reinterpret_cast<const std::uint8_t*>(pointer + offset);
                    }
                    for (unsigned i = 0; i < count && i < 32; ++i) {
                        std::uint64_t value = 0;
                        SIZE_T got = 0;
                        const bool ok = ReadProcessMemory(GetCurrentProcess(), from + 8ull * i, &value,
                                                          sizeof(value), &got) && got == sizeof(value);
                        std::snprintf(text, sizeof(text), ok ? "%s%llx" : "%s?", i ? "," : "",
                                      static_cast<unsigned long long>(value));
                        line += text;
                    }
                    at = *end == ',' ? end + 1 : end;
                    if (at == end && *end) break;
                }
                std::fprintf(stderr, "%s\n", line.c_str());
                Sleep(period * 1000);
            }
        },
        nullptr, 0, nullptr);
    if (thread) CloseHandle(thread);
}

}  // namespace

#include "base_reservation.h"
#include "big_alloc_log.h"
#include "code_hooks.h"
#include "cpu_emulation.h"
#include "host_cpu.h"
#include "mods.h"
#include "pe_loader.h"
#include "thread_stacks.h"
#include "dev_overlay.h"
#include "thread_sampler.h"
#include "title_page_watch.h"
#include "title_geometry_watch.h"
#include "../shims/d3d12_trace.h"
#include "../shims/d3d12_xs_device.h"
#include "../shims/file_system.h"
#ifdef LEONIDA_VULKAN_DEVICE
#include "vulkan_device_fault.h"
#include "../shims/d3d12_compile_exit.h"
#include "../shims/d3d12_precompile.h"
#include "../shims/package_splash.h"
#endif
#include "../shims/process_shim.h"
#include "../shims/shims.h"
#include "../shims/startup_trace.h"
#include "../shims/xmem.h"

#ifdef LEONIDA_VULKAN_DEVICE
namespace rt::d3d12 {
int prebuild_recorded_pipelines(const std::filesystem::path& recipes, unsigned threads);  // shims/d3d12_com.h
int prebuild_shader_binaries(unsigned threads);                                            // shims/d3d12_com.h
}
#endif
#include "quiet_av.h"  // leonida_quiet_av: a guarded title read is running (shims/d3d12_vulkan_device.cpp)

namespace {

const char* binding_name(rt::Binding b) {
    switch (b) {
        case rt::Binding::Shim: return "shim";
        case rt::Binding::Host: return "host";
        case rt::Binding::Module: return "module";
        default: return "UNIMPLEMENTED";
    }
}

// Module TLS index 0 for the game's executable (rt::reserve_tls_index0), from the CRT's initializer table: .CRT$XCT
// runs after the C runtime is set up and before every C++ dynamic initializer of gta6rt (.CRT$XCU), so before any of
// gta6rt's code can have started a thread. The frames above it (wmainCRTStartup, __scrt_common_main_seh, _initterm)
// hold no TLS slot address. The result is read again where an image is run.
void __cdecl reserve_tls_index0_at_start() { rt::reserve_tls_index0(); }
#pragma section(".CRT$XCT", read)
extern "C" __declspec(allocate(".CRT$XCT")) void(__cdecl* const gta6rt_reserve_tls_index0)() =
    reserve_tls_index0_at_start;
#pragma comment(linker, "/INCLUDE:gta6rt_reserve_tls_index0")

// Select the settings used to display the offline first-run UI. Keep explicit caller settings.
bool configure_first_run_ui(const std::filesystem::path& package, std::filesystem::path& overlay) {
    // A variable that is set but empty is not a setting: Windows cannot hold one, but a caller on Wine can pass one
    // (NAME= in the environment), and asking for the size alone reports it as present there.
    const auto set_default = [](const wchar_t* name, const wchar_t* value) {
        wchar_t first[2]{};
        return GetEnvironmentVariableW(name, first, 2) != 0 || _wputenv_s(name, value) == 0;
    };
    const struct { const wchar_t* name; const wchar_t* value; } settings[] = {
        {L"LEONIDA_D3D12", L"vulkan"}, {L"LEONIDA_ARENA_MB", L"2048"},
        {L"LEONIDA_GRAPHICS_STAGE_LAYOUT", L"1"}, {L"LEONIDA_PIPELINE_COMPILE", L"on-demand"},
        {L"LEONIDA_FRAME_PACING", L"1"},
        // Focused, not always: keys typed into another window must not press anything on a consent page.
        // Scripted presses (LEONIDA_INPUT_SCRIPT) still run while the window is unfocused.
        {L"LEONIDA_KEYBOARD_PAD", L"1"},
    };
    for (const auto& setting : settings) {
        if (!set_default(setting.name, setting.value)) return false;
    }
    std::wstring executable(32768, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
    if (!length || length >= executable.size()) return false;
    executable.resize(length);
    const auto state = std::filesystem::path(executable).parent_path() / L"first-run-ui";
    if (overlay.empty()) overlay = state / L"overlay";
    if (!GetEnvironmentVariableW(L"LEONIDA_XMEM_BACKING_DIR", nullptr, 0)) {
        std::error_code error;
        const auto backing = state / L"xmem";
        std::filesystem::create_directories(backing, error);
        if (error || !set_default(L"LEONIDA_XMEM_BACKING_DIR", backing.c_str())) return false;
    }
    return SetCurrentDirectoryW(package.c_str()) != FALSE;
}

int usage() {
    std::fprintf(stderr,
                 "usage: gta6rt --dry-run <image> [--game-dir DIR] [--overlay DIR] [--mods DIR] [--archetype-lookup RVA[,ARG]] [--view-mode-getter RVA] [--trace FILE]\n"
                 "       gta6rt --init <image> [--game-dir DIR] [--overlay DIR] [--mods DIR] [--archetype-lookup RVA[,ARG]] [--view-mode-getter RVA] [--trace-d3d12] [--trace FILE]\n"
                 "       gta6rt --first-run-ui <exe> [--game-dir DIR] [--overlay DIR] [--trace FILE]\n"
                 "       gta6rt --run <exe> [--game-dir DIR] [--overlay DIR] [--mods DIR] [--archetype-lookup RVA[,ARG]] [--view-mode-getter RVA] [--trace-d3d12] [--trace FILE] [-- game arguments...]\n"
                 "       gta6rt --precompile-pipelines [--threads N] [--big-threads N] [--big-bytes N] [--all] [--smallest-first] [--limit N] [--min-bytes N] [--max-bytes N] [--filter PART]... [--list] [--via-helper]\n"
                 "       gta6rt --compile-server [--parent PID] [--threads N] [--idle-threads N] [--idle-exit SECONDS]\n");
    return 2;
}

// gta6rt --prebuild-pipelines [--recipes FILE] [--jobs N] (docs/aot_pipelines.md): every graphics pipeline a game run
// recorded (pso_recipes.jsonl beside gta6rt.exe by default) is built on the game's Vulkan device into the pipeline
// cache, which is saved, so the next run compiles none of them. gta6rt --prebuild-shaders [--jobs N]: every translated
// graphics shader compiled once as a shader object and its driver binary stored (LEONIDA_SHADER_OBJECTS=1 runs load
// them). Neither runs game code.
int prebuild_pipelines_command(int argc, wchar_t** argv) {
#ifdef LEONIDA_VULKAN_DEVICE
    const bool shaders = std::wstring(argv[1]) == L"--prebuild-shaders";
    std::filesystem::path recipes;
    unsigned jobs = 0;
    for (int i = 2; i < argc; ++i) {
        const std::wstring a = argv[i];
        if (a == L"--recipes" && i + 1 < argc && !shaders) recipes = argv[++i];
        else if (a == L"--jobs" && i + 1 < argc) {
            wchar_t* end = nullptr;
            const unsigned long n = std::wcstoul(argv[++i], &end, 10);
            if (!end || *end || n == 0 || n > 256) return usage();
            jobs = static_cast<unsigned>(n);
        } else {
            return usage();
        }
    }
    if (shaders) return rt::d3d12::prebuild_shader_binaries(jobs);
    if (recipes.empty()) {
        std::wstring self(32768, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, self.data(), static_cast<DWORD>(self.size()));
        if (!n || n >= self.size()) return 2;
        self.resize(n);
        recipes = std::filesystem::path(self).parent_path() / L"pso_recipes.jsonl";
    }
    return rt::d3d12::prebuild_recorded_pipelines(recipes, jobs);
#else
    (void)argc;
    (void)argv;
    std::fprintf(stderr, "gta6rt: --prebuild-pipelines and --prebuild-shaders need the Vulkan device (build with the "
                         "Vulkan SDK)\n");
    return 2;
#endif
}

}  // namespace

namespace {

// An int 3 in the title's own code (no host module holds the address) is one of its soft asserts: the code after it
// resumes normally (e.g. exe+0x4ffd879: int 3; jmp back into the function), which a dev kit's debugger or
// GTA6-BootTest's harness (-skip) steps over. Stepped over here too, each address named once; LEONIDA_ASSERTS=stop
// keeps them fatal. True when the exception was handled.
bool step_over_title_assert(EXCEPTION_POINTERS* info) {
    const EXCEPTION_RECORD& r = *info->ExceptionRecord;
    if (r.ExceptionCode != 0x80000003 || !info->ContextRecord) return false;
    static const bool stop = [] {
        char v[8] = {};
        return GetEnvironmentVariableA("LEONIDA_ASSERTS", v, sizeof v) && std::strcmp(v, "stop") == 0;
    }();
    HMODULE owner = nullptr;
    const bool host_code = GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                                  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                              static_cast<LPCSTR>(r.ExceptionAddress), &owner) && owner;
    const auto* at = static_cast<const unsigned char*>(r.ExceptionAddress);
    if (stop || host_code || !at || *at != 0xCC) return false;
    static std::mutex m;
    static std::vector<const void*> seen;
    {
        std::lock_guard lock(m);
        if (std::find(seen.begin(), seen.end(), r.ExceptionAddress) == seen.end() && seen.size() < 64) {
            seen.push_back(r.ExceptionAddress);
            std::fprintf(stderr, "[rt] title assert (int 3) at %p stepped over (tid %lu)\n", r.ExceptionAddress,
                         GetCurrentThreadId());
        }
    }
    info->ContextRecord->Rip += 1;
    return true;
}

LONG CALLBACK crash_logger(EXCEPTION_POINTERS* info) {
    if (!info || !info->ExceptionRecord) return EXCEPTION_CONTINUE_SEARCH;
    const EXCEPTION_RECORD& r = *info->ExceptionRecord;
#ifdef LEONIDA_VULKAN_DEVICE
    // A guarded read of the title's pages (d3d12_vulkan_device.cpp) expects this fault and handles it itself.
    if (leonida_quiet_av && r.ExceptionCode == 0xC0000005) return EXCEPTION_CONTINUE_SEARCH;
#endif
    if (step_over_title_assert(info)) return EXCEPTION_CONTINUE_EXECUTION;
    // A C++ exception (0xE06D7363) is logged too, the first 8: an unhandled one ended runs with nothing in the log (the
    // throw site's stack below names it; most are caught and harmless, hence the cap).
    if (r.ExceptionCode == 0xE06D7363) {
        static std::atomic<int> cpp_told{0};
        if (cpp_told.fetch_add(1) >= 8) return EXCEPTION_CONTINUE_SEARCH;
    } else if (r.ExceptionCode != 0xC0000005 && r.ExceptionCode != 0xC0000374 && r.ExceptionCode != 0x80000003 &&
               r.ExceptionCode != 0xC0000409 && r.ExceptionCode != 0xC00000FD) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const void* access = r.NumberParameters >= 2 ? reinterpret_cast<void*>(r.ExceptionInformation[1]) : nullptr;
    char module[MAX_PATH] = {};
    HMODULE host = nullptr;
    const auto address = reinterpret_cast<LPCSTR>(r.ExceptionAddress);
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           address, &host) &&
        host)
        GetModuleFileNameA(host, module, MAX_PATH);
    std::fprintf(stderr, "[rt] crash-veh: code=0x%08lx addr=%p module=%s+0x%llx access=%p tid=%lu\n", r.ExceptionCode,
                 r.ExceptionAddress, module[0] ? module : "?", host ? static_cast<unsigned long long>(
                     reinterpret_cast<uintptr_t>(r.ExceptionAddress) - reinterpret_cast<uintptr_t>(host)) : 0ull,
                 access, GetCurrentThreadId());
    if (info->ContextRecord) {
        const CONTEXT& c = *info->ContextRecord;
        std::fprintf(stderr,
                     "[rt] crash-context: rip=%p rsp=%p rax=%p rbx=%p rcx=%p rdx=%p rsi=%p rdi=%p rbp=%p "
                     "r8=%p r9=%p r10=%p r11=%p r12=%p r13=%p r14=%p r15=%p\n",
                     reinterpret_cast<void*>(c.Rip), reinterpret_cast<void*>(c.Rsp),
                     reinterpret_cast<void*>(c.Rax), reinterpret_cast<void*>(c.Rbx),
                     reinterpret_cast<void*>(c.Rcx), reinterpret_cast<void*>(c.Rdx),
                     reinterpret_cast<void*>(c.Rsi), reinterpret_cast<void*>(c.Rdi),
                     reinterpret_cast<void*>(c.Rbp), reinterpret_cast<void*>(c.R8),
                     reinterpret_cast<void*>(c.R9), reinterpret_cast<void*>(c.R10),
                     reinterpret_cast<void*>(c.R11), reinterpret_cast<void*>(c.R12),
                     reinterpret_cast<void*>(c.R13), reinterpret_cast<void*>(c.R14),
                     reinterpret_cast<void*>(c.R15));
        if (c.Rip == 0) {
            __try {
                const auto object = static_cast<std::uint64_t>(c.Rax);
                const auto vtable = *reinterpret_cast<const std::uint64_t*>(object);
                const auto slot146 = *reinterpret_cast<const std::uint64_t*>(vtable + 0x490);
                char where[MAX_PATH] = {};
                HMODULE holder = nullptr;
                GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCSTR>(vtable), &holder);
                if (holder) GetModuleFileNameA(holder, where, MAX_PATH);
                std::fprintf(stderr, "[rt] nullcall: rax=%p vtable=%p slot146=%p module=%s\n",
                             reinterpret_cast<void*>(object), reinterpret_cast<void*>(vtable),
                             reinterpret_cast<void*>(slot146), where[0] ? where : "?");
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                std::fprintf(stderr, "[rt] nullcall: unreadable\n");
            }
        }
        __try {
            const auto* stack = reinterpret_cast<const std::uint64_t*>(c.Rsp);
            std::fprintf(stderr, "[rt] crash-stack-raw:");
            for (int i = 0; i < 8; ++i) std::fprintf(stderr, " %016llx", static_cast<unsigned long long>(stack[i]));
            std::fprintf(stderr, "\n");
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            std::fprintf(stderr, "[rt] crash-stack-raw: unreadable\n");
        }
        // Diagnostic: the memory around each general register that points at readable memory, with its region, so a
        // corrupted object can be told apart from an overrun leading into it (retail boot 15: an object whose vtable
        // word read 0x3F8000003F800000, two floats of 1.0).
        const struct {
            const char* name;
            DWORD64 value;
        } regs[] = {{"rax", c.Rax}, {"rbx", c.Rbx}, {"rcx", c.Rcx}, {"rdx", c.Rdx}, {"rsi", c.Rsi}, {"rdi", c.Rdi},
                    {"r8", c.R8},   {"r9", c.R9},   {"r12", c.R12}, {"r13", c.R13}, {"r14", c.R14}, {"r15", c.R15}};
        for (const auto& reg : regs) {
            MEMORY_BASIC_INFORMATION info{};
            if (reg.value < 0x10000 ||
                VirtualQuery(reinterpret_cast<void*>(reg.value), &info, sizeof(info)) != sizeof(info) ||
                info.State != MEM_COMMIT || (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)))
                continue;
            const auto region_begin = reinterpret_cast<DWORD64>(info.BaseAddress);
            const DWORD64 region_end = region_begin + info.RegionSize;
            const DWORD64 at_reg = reg.value & ~DWORD64(7);
            const DWORD64 from = at_reg - region_begin >= 0x80 ? at_reg - 0x80 : region_begin;
            const DWORD64 to = region_end - at_reg >= 0x80 ? at_reg + 0x80 : region_end;
            std::fprintf(stderr, "[rt] crash-mem %s=%p region base=%p size=0x%llx alloc=%p type=0x%lx", reg.name,
                         reinterpret_cast<void*>(reg.value), info.BaseAddress,
                         static_cast<unsigned long long>(info.RegionSize), info.AllocationBase, info.Type);
            __try {
                for (DWORD64 at = from; at < to; at += 8) {
                    if (((at - from) & 0x3f) == 0)
                        std::fprintf(stderr, "\n[rt] crash-mem   %+05lld:",
                                     static_cast<long long>(at) - static_cast<long long>(at_reg));
                    std::fprintf(stderr, " %016llx", *reinterpret_cast<const unsigned long long*>(at));
                }
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                std::fprintf(stderr, " (unreadable)");
            }
            std::fprintf(stderr, "\n");
        }
    }
    // Diagnostic: the retail frame path reads through 0x100000000 (the synthetic-VA base). Commit one guard page
    // there so a read of that address is a value, not a fault, while the caller is being identified.
    __try {
        void* probe = VirtualAlloc(reinterpret_cast<void*>(0x100000000ull), 0x10000,
                                   MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
        std::fprintf(stderr, "[rt] crash-pad: 0x100000000 -> %p\n", probe);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::fprintf(stderr, "[rt] crash-pad: failed\n");
    }
    // Diagnostic: the faulting thread's TEB TLS slots (the console main image reads slot 0 directly).
    __try {
        auto** tls = reinterpret_cast<void**>(__readgsqword(0x58));
        std::fprintf(stderr, "[rt] crash-tls: slots=%p slot0=%p slot1=%p slot2=%p slot3=%p\n",
                     static_cast<void*>(tls), tls ? tls[0] : nullptr, tls ? tls[1] : nullptr,
                     tls ? tls[2] : nullptr, tls ? tls[3] : nullptr);
        if (tls && tls[0]) {
            MEMORY_BASIC_INFORMATION info{};
            if (VirtualQuery(tls[0], &info, sizeof(info)) == sizeof(info))
                std::fprintf(stderr, "[rt] crash-tls: slot0 region base=%p size=0x%llx state=0x%lx protect=0x%lx\n",
                             info.BaseAddress, static_cast<unsigned long long>(info.RegionSize), info.State,
                             info.Protect);
            else
                std::fprintf(stderr, "[rt] crash-tls: slot0 not mapped\n");
            auto* base = static_cast<unsigned char*>(tls[0]);
            std::fprintf(stderr, "[rt] crash-tls: tls+0x12a8=%p tls+0x12a0=%p tls+0x12b0=%p\n",
                         *reinterpret_cast<void**>(base + 0x12a8), *reinterpret_cast<void**>(base + 0x12a0),
                         *reinterpret_cast<void**>(base + 0x12b0));
        }
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        std::fprintf(stderr, "[rt] crash-tls: unreadable\n");
    }
    void* frames[32] = {};
    const USHORT count = CaptureStackBackTrace(0, 32, frames, nullptr);
    for (USHORT i = 0; i < count; ++i) {
        HMODULE frame_host = nullptr;
        char frame_module[MAX_PATH] = {};
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               reinterpret_cast<LPCSTR>(frames[i]), &frame_host) &&
            frame_host)
            GetModuleFileNameA(frame_host, frame_module, MAX_PATH);
        const char* slash = std::strrchr(frame_module, '\\');
        std::fprintf(stderr, "[rt] crash-stack %u: %s+0x%llx\n", i, slash ? slash + 1 : frame_module,
                     frame_host ? static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(frames[i]) -
                                                                  reinterpret_cast<uintptr_t>(frame_host))
                                : 0ull);
    }
    std::fflush(stderr);
    // The retail frame path reads the per-thread context pointer at 0x100000000 (4 GiB): its TLS block is laid out
    // so the pointer field lands exactly at 4 GiB. This thread's own TLS block already holds that pointer at
    // +0x12a8, so put it in the slot the title reads, then let the faulting instruction run again against it.
    // Diagnostic: the A8 render-target format assert (title RVA 0xCC2936). The format is supported by the layer but
    // the title's own info switch still traps; step over the int3 so the next real stop is visible.
    if (r.ExceptionCode == 0x80000003 && info->ContextRecord &&
        info->ContextRecord->Rip == 0x140CC2936ull) {
        static std::atomic<int> skipped_format{0};
        if (skipped_format.fetch_add(1) < 64) {
            std::fprintf(stderr, "[rt] format-int3: rbp=%p r10=%p r13=%p r14=%p rsi=%p rdi=%p\n",
                         reinterpret_cast<void*>(info->ContextRecord->Rbp),
                         reinterpret_cast<void*>(info->ContextRecord->R10),
                         reinterpret_cast<void*>(info->ContextRecord->R13),
                         reinterpret_cast<void*>(info->ContextRecord->R14),
                         reinterpret_cast<void*>(info->ContextRecord->Rsi),
                         reinterpret_cast<void*>(info->ContextRecord->Rdi));
            __try {
                const auto rsp = static_cast<std::uint64_t>(info->ContextRecord->Rsp);
                const auto caller_rsi = *reinterpret_cast<const std::uint64_t*>(rsp + 0x48);
                const auto holder = caller_rsi ? *reinterpret_cast<const std::uint64_t*>(caller_rsi + 0x60) : 0;
                const auto holder_vt = holder ? *reinterpret_cast<const std::uint64_t*>(holder) : 0;
                std::fprintf(stderr, "[rt] format-object: caller_rsi=%p holder=%p holder_vtable=%p\n",
                             reinterpret_cast<void*>(caller_rsi), reinterpret_cast<void*>(holder),
                             reinterpret_cast<void*>(holder_vt));
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                std::fprintf(stderr, "[rt] format-object: unreadable\n");
            }
            __try {
                const auto* st = reinterpret_cast<const std::uint64_t*>(info->ContextRecord->Rsp);
                std::fprintf(stderr, "[rt] format-stack-raw:");
                for (int i = 0; i < 10; ++i) std::fprintf(stderr, " %016llx", static_cast<unsigned long long>(st[i]));
                std::fprintf(stderr, "\n");
                for (int i = 0; i < 14; ++i) {
                    const std::uint64_t a = st[i];
                    if (a < 0x10000) continue;
                    HMODULE holder = nullptr;
                    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                           reinterpret_cast<LPCSTR>(a), &holder) &&
                        holder) {
                        char name[MAX_PATH] = {};
                        GetModuleFileNameA(holder, name, MAX_PATH);
                        const char* slash = std::strrchr(name, '\\');
                        std::fprintf(stderr, "[rt] format-frame %d: %s+0x%llx\n", i, slash ? slash + 1 : name,
                                     static_cast<unsigned long long>(a - reinterpret_cast<std::uint64_t>(holder)));
                    } else if (a >= 0x140000000ull && a < 0x150000000ull) {
                        std::fprintf(stderr, "[rt] format-frame %d: game+0x%llx\n", i,
                                     static_cast<unsigned long long>(a - 0x140000000ull));
                    }
                }
            } __except (EXCEPTION_EXECUTE_HANDLER) {
                std::fprintf(stderr, "[rt] format-stack-raw: unreadable\n");
            }
            info->ContextRecord->Rip += 1;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    // An unknown console method reached through a vtable with no entry at the called offset: skip the call and
    // answer S_OK (RAX = 0), popping its return address. Bounded and logged, so the next real stop is visible.
    if (r.ExceptionCode == 0xC0000005 && info->ContextRecord && info->ContextRecord->Rip == 0) {
        static std::atomic<int> skipped{0};
        const auto ret = *reinterpret_cast<const std::uint64_t*>(info->ContextRecord->Rsp);
        if (ret >= 0x140000000ull && ret < 0x150000000ull && skipped.fetch_add(1) < 16) {
            std::fprintf(stderr, "[rt] nullcall-skip: return=%p rcx=%p rdx=%p\n",
                         reinterpret_cast<void*>(ret),
                         reinterpret_cast<void*>(info->ContextRecord->Rcx),
                         reinterpret_cast<void*>(info->ContextRecord->Rdx));
            info->ContextRecord->Rip = ret;
            info->ContextRecord->Rsp += 8;
            info->ContextRecord->Rax = 0;
            return EXCEPTION_CONTINUE_EXECUTION;
        }
    }
    if (r.ExceptionCode == 0xC0000005 && access == reinterpret_cast<void*>(0x100000000ull)) {
        __try {
            VirtualAlloc(reinterpret_cast<void*>(0x100000000ull), 0x1000, MEM_COMMIT, PAGE_READWRITE);
            auto** tls = reinterpret_cast<void**>(__readgsqword(0x58));
            void* context = nullptr;
            if (tls && tls[0]) context = *reinterpret_cast<void**>(static_cast<unsigned char*>(tls[0]) + 0x12a8);
            *reinterpret_cast<void**>(0x100000000ull) = context;
            std::fprintf(stderr, "[rt] crash-pad: wrote context %p at 0x100000000\n", context);
        } __except (EXCEPTION_EXECUTE_HANDLER) {
            std::fprintf(stderr, "[rt] crash-pad: could not write the context slot\n");
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}
#ifdef LEONIDA_VULKAN_DEVICE
// gta6rt --precompile-pipelines ...: argv[1] is the mode.
int precompile_main(int argc, wchar_t** argv) {
    rt::d3d12::PrecompileOptions options;
    const auto number = [&](int& i, auto& out) {
        if (i + 1 >= argc) return false;
        wchar_t* end = nullptr;
        const unsigned long long v = std::wcstoull(argv[++i], &end, 10);
        if (!end || *end) return false;
        out = static_cast<std::remove_reference_t<decltype(out)>>(v);
        return true;
    };
    for (int i = 2; i < argc; ++i) {
        const std::wstring a = argv[i];
        bool ok = true;
        if (a == L"--threads") ok = number(i, options.threads);
        else if (a == L"--big-threads") ok = number(i, options.big_threads);
        else if (a == L"--big-bytes") ok = number(i, options.big_bytes);
        else if (a == L"--limit") ok = number(i, options.limit);
        else if (a == L"--min-bytes") ok = number(i, options.min_bytes);
        else if (a == L"--max-bytes") ok = number(i, options.max_bytes);
        else if (a == L"--all") options.rebuild = true;
        else if (a == L"--smallest-first") options.smallest_first = true;
        else if (a == L"--list") options.dry_run = true;
        else if (a == L"--via-helper") options.via_helper = true;
        else if (a == L"--filter" && i + 1 < argc) {
            const std::wstring part = argv[++i];
            options.filters.emplace_back(part.begin(), part.end());  // library names are ASCII
        } else ok = false;
        if (!ok) return usage();
    }
    return rt::d3d12::precompile_compute_pipelines(options);
}

// gta6rt --compile-server [--parent PID] [--threads N] [--idle-threads N] [--idle-exit SECONDS]
// (shims/d3d12_compile_helper.h).
int compile_server_main(int argc, wchar_t** argv) {
    rt::d3d12::CompileServerOptions options;
    for (int i = 2; i < argc; ++i) {
        const std::wstring a = argv[i];
        if (i + 1 >= argc) return usage();
        wchar_t* end = nullptr;
        const unsigned long long v = std::wcstoull(argv[++i], &end, 10);
        if (!end || *end) return usage();
        if (a == L"--parent") options.parent_pid = static_cast<unsigned long>(v);
        else if (a == L"--threads") options.threads = static_cast<unsigned>(v);
        else if (a == L"--idle-threads") options.idle_threads = static_cast<unsigned>(v);
        else if (a == L"--idle-exit") options.idle_exit_s = static_cast<unsigned>(v);
        else return usage();
    }
    return rt::d3d12::compile_server(options);
}
#endif

}  // namespace

#ifdef LEONIDA_VULKAN_DEVICE
// The run ends with the title's Vulkan device alive, so no ~Device stops the background pipeline compiler: stop it
// here, before ExitProcess ends its thread, maybe inside the driver's compiler (rt::d3d12::stop_pipeline_compiles).
constexpr unsigned kExitCompileWaitMs = 2000;  // a console close gives the whole process 5 s
// For a build still running on a title thread, then the session cache write (rt::d3d12::save_pipeline_caches_at_exit).
constexpr unsigned kExitCacheWaitMs = 1000;

static BOOL WINAPI stop_compiles_on_console_exit(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
        rt::d3d12_xs::stop_list_values(500);  // as at the run's end: no list-value thread cut off inside the layer
        rt::d3d12::stop_pipeline_compiles(kExitCompileWaitMs);
        rt::d3d12::save_pipeline_caches_at_exit(kExitCacheWaitMs);
    }
    return FALSE;  // the next handler (the default one: ExitProcess) ends the process, as before
}

// A run that ends in an exception nothing handles (a fault in the title or one of its DLLs, the int3 after a device
// loss) would otherwise keep none of the pipelines it compiled, and the next boot builds them all again. The device is
// left alone apart from that write; the exception then goes on as before (the previous filter, or Windows' own
// handling), so the exit code is the exception's. Not for a stack overflow: no stack is left to write with.
static LPTOP_LEVEL_EXCEPTION_FILTER previous_fatal_filter = nullptr;
// The retail title installs a filter of its own (SetUnhandledExceptionFilter from api-ms-win-core-errorhandling),
// which replaced this one and ended the process without it (measured: a title fault at 354 s wrote nothing). Its
// call is served here instead: this filter stays installed and hands the exception to the title's, as Windows would.
static void* volatile title_fatal_filter = nullptr;

static LPTOP_LEVEL_EXCEPTION_FILTER WINAPI title_set_unhandled_exception_filter(LPTOP_LEVEL_EXCEPTION_FILTER filter) {
    return reinterpret_cast<LPTOP_LEVEL_EXCEPTION_FILTER>(
        InterlockedExchangePointer(&title_fatal_filter, reinterpret_cast<void*>(filter)));
}

static LONG WINAPI save_caches_on_fatal_exception(EXCEPTION_POINTERS* info) {
    const DWORD code = info && info->ExceptionRecord ? info->ExceptionRecord->ExceptionCode : 0;
    if (code != EXCEPTION_STACK_OVERFLOW) {
        // Several title threads can fault at once (measured: eight within a second). The first writes; the others
        // wait here for it, so none of them reaches the title's filter, which may end the process, mid-write.
        static std::once_flag once;
        std::call_once(once, [code] {
            std::fprintf(stderr, "[rt] unhandled exception 0x%08lx: writing the session pipeline cache before the "
                                 "process ends\n", static_cast<unsigned long>(code));
            rt::d3d12::stop_pipeline_compiles(kExitCompileWaitMs);
            rt::d3d12::save_pipeline_caches_at_exit(kExitCacheWaitMs);
            std::fflush(stderr);
        });
    }
    if (auto* title = reinterpret_cast<LPTOP_LEVEL_EXCEPTION_FILTER>(
            InterlockedCompareExchangePointer(&title_fatal_filter, nullptr, nullptr)))
        return title(info);
    return previous_fatal_filter ? previous_fatal_filter(info) : EXCEPTION_CONTINUE_SEARCH;
}

static BOOL WINAPI show_package_window(HWND window, int command) {
    const BOOL previous = ShowWindow(window, command);
    const DWORD error = GetLastError();
    if (command != SW_HIDE && rt::package_splash::paint_window(window))
        rt::startup_trace::event("package_splash_window");
    SetLastError(error);
    return previous;
}
#endif

int wmain(int argc, wchar_t** argv) {
    leonida::big_alloc_log::install();  // LEONIDA_LOG_BIG_ALLOCS=1 (src/big_alloc_log.h)
#ifdef LEONIDA_VULKAN_DEVICE
    if (argc > 1 && std::wstring(argv[1]) == L"--precompile-pipelines") return precompile_main(argc, argv);
    if (argc > 1 && std::wstring(argv[1]) == L"--compile-server") return compile_server_main(argc, argv);
#endif
    std::filesystem::path image, game_dir, overlay_dir, mods_dir;
    uint32_t archetype_lookup_rva = 0;  // --archetype-lookup RVA[,ARG]: the game's archetype lookup by hash
    int archetype_lookup_arg = 0;
    uint32_t view_mode_getter_rva = 0;  // --view-mode-getter RVA: the game's view mode of a camera context
    bool dry_run = false, init = false, run = false, trace_d3d12 = false, first_run_ui = false;
    std::vector<std::wstring> game_args;  // wide: path::string() throws on a character the code page lacks (RT-84)
    std::wstring trace_file;
    if (argc >= 2 && (std::wstring(argv[1]) == L"--prebuild-pipelines" || std::wstring(argv[1]) == L"--prebuild-shaders"))
        return prebuild_pipelines_command(argc, argv);
    for (int i = 1; i < argc; ++i) {
        std::wstring a = argv[i];
        if (a == L"--") {
            for (++i; i < argc; ++i) game_args.push_back(argv[i]);
            break;
        }
        if (a == L"--dry-run") dry_run = true;
        else if (a == L"--init") init = true;
        else if (a == L"--first-run-ui") first_run_ui = true;
        else if (a == L"--run") run = true;
        else if (a == L"--trace-d3d12") trace_d3d12 = true;
        else if (a == L"--trace" && i + 1 < argc) trace_file = argv[++i];
        else if (a == L"--game-dir" && i + 1 < argc) game_dir = argv[++i];
        else if (a == L"--overlay" && i + 1 < argc) overlay_dir = argv[++i];
        else if (a == L"--mods" && i + 1 < argc) mods_dir = argv[++i];
        else if (a == L"--view-mode-getter" && i + 1 < argc) {
            wchar_t* end = nullptr;
            const unsigned long rva = std::wcstoul(argv[++i], &end, 16);
            if (!end || *end || rva == 0) return usage();
            view_mode_getter_rva = static_cast<uint32_t>(rva);
        }
        else if (a == L"--archetype-lookup" && i + 1 < argc) {
            const std::wstring v = argv[++i];
            const auto comma = v.find(L',');
            const std::wstring address = v.substr(0, comma);
            wchar_t* end = nullptr;
            const unsigned long rva = std::wcstoul(address.c_str(), &end, 16);
            if (!end || *end || rva == 0) return usage();
            archetype_lookup_rva = static_cast<uint32_t>(rva);
            if (comma != std::wstring::npos) archetype_lookup_arg = v.substr(comma + 1) == L"1" ? 1 : 0;
        }
        else if (image.empty()) image = a;
        else return usage();
    }
    if (image.empty()) return usage();
    // Exactly one mode: --dry-run --run would otherwise initialise the modules and call the entry point.
    if (dry_run + init + run + first_run_ui != 1) return usage();
    run = run || first_run_ui;
    // The game runs AVX code from its TLS callbacks on; without AVX that is a bare 0xC000001D (src/host_cpu.h).
    if ((init || run) && !rt::host_cpu::check_skipped())
        if (const auto missing = rt::host_cpu::missing_host_features(); !missing.empty()) {
            std::fprintf(stderr, "%s\n", rt::host_cpu::message(missing).c_str());
            return rt::host_cpu::kExitCode;
        }
    // The console folders are fixed once, as absolute paths. A bare image name (`gta6rt --run game.exe` from the
    // game folder) has an empty parent path, which left GAME: with no root; and a relative --game-dir or --overlay
    // would follow the title's later SetCurrentDirectory. If absolute() fails the path stays as given.
    const auto absolute_or_same = [](const std::filesystem::path& p) {
        std::error_code ec;
        std::filesystem::path a = std::filesystem::absolute(p, ec);
        return ec ? p : a;
    };
    if (first_run_ui) {
        image = absolute_or_same(image);
        if (!mods_dir.empty()) mods_dir = absolute_or_same(mods_dir);
        if (!trace_file.empty()) trace_file = absolute_or_same(trace_file).wstring();
    }
    game_dir = game_dir.empty() ? absolute_or_same(image).parent_path() : absolute_or_same(game_dir);
    // An unattended run can select a fresh write root even when its launcher supplies --overlay.
    if (const DWORD size = GetEnvironmentVariableW(L"LEONIDA_WRITE_DIR", nullptr, 0)) {
        std::wstring dir(size, 0);
        const DWORD n = GetEnvironmentVariableW(L"LEONIDA_WRITE_DIR", dir.data(), size);
        if (n && n < size) {
            dir.resize(n);
            overlay_dir = dir;
        }
    }
    if (!overlay_dir.empty()) overlay_dir = absolute_or_same(overlay_dir);
    // LEONIDA_SHADERS too: the D3D12 device reads it at the title's first pipeline state, after it may have moved.
    if (const DWORD size = GetEnvironmentVariableW(L"LEONIDA_SHADERS", nullptr, 0)) {
        std::wstring shaders(size, L'\0');
        const DWORD n = GetEnvironmentVariableW(L"LEONIDA_SHADERS", shaders.data(), size);
        if (n && n < size) {
            shaders.resize(n);
            SetEnvironmentVariableW(L"LEONIDA_SHADERS", absolute_or_same(shaders).c_str());
        }
    }
    // An image without relocations whose range this process already uses (arm64 Wine on macOS) runs in a new
    // process that has the range reserved from the start (base_reservation.h); that process does not do this again.
    if (!rt::base_reservation::inherited())
        if (const auto range = rt::base_reservation::needed(image)) return rt::base_reservation::relaunch(*range);
    // GAME: paths read under --game-dir; whatever a title writes lands in the overlay, never in the package.
    if (first_run_ui && !configure_first_run_ui(game_dir, overlay_dir)) {
        std::fprintf(stderr, "cannot initialize the first-run UI paths or graphics settings\n");
        return 1;
    }
    if (overlay_dir.empty()) overlay_dir = game_dir / "overlay";
    rt::fs::configure(game_dir, overlay_dir);
#ifdef LEONIDA_VULKAN_DEVICE
    if (run) {
        // Before the image is loaded, so its import binds here (save_caches_on_fatal_exception).
        const rt::shims::Entry filter_imports[] = {
            {"api-ms-win-core-errorhandling-l1-1-0.dll", "SetUnhandledExceptionFilter",
             reinterpret_cast<void*>(&title_set_unhandled_exception_filter)},
            {"kernel32.dll", "SetUnhandledExceptionFilter", reinterpret_cast<void*>(&title_set_unhandled_exception_filter)},
        };
        rt::shims::Table{filter_imports, std::size(filter_imports)};
        rt::package_splash::configure(game_dir);
        if (rt::package_splash::configured()) {
            const rt::shims::Entry window_imports[] = {
                {"ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", "ShowWindow", reinterpret_cast<void*>(&show_package_window)},
                {"user32.dll", "ShowWindow", reinterpret_cast<void*>(&show_package_window)},
            };
            rt::shims::Table{window_imports, std::size(window_imports)};
        }
    }
#endif
    // Mods (docs/modding.md): <folder of gta6rt.exe>\mods unless --mods names another. Loaded and reported here;
    // the archetype swaps take effect once the game's archetype lookup is hooked for this executable version.
    if (mods_dir.empty()) {
        std::wstring self(32768, L'\0');
        const DWORD n = GetModuleFileNameW(nullptr, self.data(), static_cast<DWORD>(self.size()));
        if (n && n < self.size()) mods_dir = std::filesystem::path(self.substr(0, n)).parent_path() / L"mods";
    }
    // Mods are optional: whatever goes wrong reading them is reported and the run goes on without them.
    try {
        const leonida::mods::LoadResult mods = leonida::mods::load_mods(mods_dir);
        for (const std::string& e : mods.errors) std::fprintf(stderr, "[rt] mods: skipped %s\n", e.c_str());
        for (const auto& m : mods.mods) {
            std::fprintf(stderr, "[rt] mods: %s%s\n", m.name.c_str(), m.enabled ? "" : " (disabled)");
            for (const auto& sw : m.archetype_swaps)
                std::fprintf(stderr, "[rt] mods:   archetype %s (0x%08X) -> %s (0x%08X)\n", sw.from_name.c_str(),
                             sw.from, sw.to_name.c_str(), sw.to);
        }
        leonida::mods::set_active(leonida::mods::ArchetypeSwaps(mods.mods));
        const leonida::mods::CameraView camera = leonida::mods::camera_of(mods.mods);
        leonida::mods::set_active_camera(camera);
        if (camera.first_person)
            std::fprintf(stderr, "[rt] mods: first-person view (value %u) for %s%s\n", camera.first_person_value,
                         camera.contexts.empty() ? "every context" : "the listed contexts",
                         view_mode_getter_rva ? "" : "; not applied: no --view-mode-getter RVA for this executable");
        if (leonida::mods::active().size())
            std::fprintf(stderr, archetype_lookup_rva ? "[rt] mods: %zu archetype swap(s) loaded\n"
                                 : "[rt] mods: %zu archetype swap(s) loaded; not applied: no --archetype-lookup RVA for "
                                   "this executable (docs/modding.md)\n", leonida::mods::active().size());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[rt] mods: not loaded (%s); running without mods\n", e.what());
    }
    rt::d3d12_trace::set_enabled(trace_d3d12);
    AddVectoredExceptionHandler(1, &crash_logger);
    // No modal "Application Error" box when the title faults (its GPU watchdog writes to address 0 on purpose): the
    // dialog sat on the user's screen after run 29 until clicked away. crash_logger has already written the fault and
    // its stack to the log; the process just ends. LEONIDA_CRASH_DIALOG=1 keeps the box (to attach a debugger).
    if (const char* v = std::getenv("LEONIDA_CRASH_DIALOG"); !(v && v[0] == '1'))
        SetErrorMode(GetErrorMode() | SEM_NOGPFAULTERRORBOX | SEM_FAILCRITICALERRORS);
    {
        // A minimum working set the size of what the title keeps resident anyway (it runs at 10-17 GB): the GPU's host
        // imports of title memory lock its pages, which Windows counts against the process's lockable pages (bounded by
        // the minimum working set, VirtualLock's rule; MmProbeAndLockPages raises STATUS_WORKING_SET_QUOTA past it). The
        // driver refused imports at 0.4-3.4 GB in all, at no fixed size (runs 17-23). Soft limits (the *_DISABLE flags):
        // nothing is trimmed or capped that would not be otherwise. LEONIDA_MIN_WORKING_SET_GB=0 leaves the default.
        double gb = 6.0;
        if (const char* v = std::getenv("LEONIDA_MIN_WORKING_SET_GB")) gb = std::strtod(v, nullptr);
        if (gb > 0) {
            const SIZE_T min_bytes = static_cast<SIZE_T>(gb * (1ull << 30));
            const SIZE_T max_bytes = min_bytes + (SIZE_T{16} << 30);
            const BOOL ok = SetProcessWorkingSetSizeEx(GetCurrentProcess(), min_bytes, max_bytes,
                                                       QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            std::fprintf(stderr, "[rt] working set: minimum %.1f GB %s (error %lu)\n", gb, ok ? "set" : "refused",
                         ok ? 0ul : GetLastError());
        }
    }
    rt::write_watch::start_from_environment();  // LEONIDA_WATCH_WRITE (src/write_watch.cpp), added after: runs first
    // Slot 133 imports the title's pages; xmem commits them on first touch, so they are committed first.
    rt::d3d12_xs::set_placed_at_va_pages_hook(&rt::xmem::commit_physical_range);
    if (rt::xmem::slot31_runs() && rt::xmem::queue_page_mappings_enabled())
        rt::d3d12_xs::set_copy_page_mappings_hook(&rt::xmem::copy_page_mappings);
    // So do the file shims' reads, whose kernel writes do not fault them in: on ERROR_NOACCESS (title_page_watch.h).
    leonida::geometry_writes::aliases.store(&rt::xmem::geometry_aliases);
    leonida::kernel_writes::set_committer(&rt::xmem::commit_physical_range);
    rt::d3d12_xs::set_texture_page_provider(&rt::xmem::pin_mapped_data);
    leonida::kernel_writes::set_unguarder(&rt::xmem::reopen_unowned);
    if (int trace_error = 0; !trace_file.empty() && !rt::startup_trace::open(trace_file.c_str(), &trace_error)) {
        std::fprintf(stderr, trace_error == EEXIST ? "error: the trace file already exists; --trace only creates a new file\n"
                                                   : "error: cannot create the trace file\n");
        return 2;
    }
    // The console's Zen 2 instructions this CPU may lack (docs/cpu_features.md): emulated on #UD. Installed after the
    // trace's handler so it runs first, and an emulated instruction is not recorded as an exception.
    if (const auto host = rt::cpu_emulation::detect_host(); !host.sse4a || !host.clzero || !host.monitorx || !host.clwb)
        std::fprintf(stderr, "[rt] cpu: this CPU lacks%s%s%s%s; the game's uses are emulated (docs/cpu_features.md)\n",
                     host.sse4a ? "" : " SSE4a", host.clzero ? "" : " CLZERO", host.monitorx ? "" : " MONITORX",
                     host.clwb ? "" : " CLWB");
    if (!rt::cpu_emulation::install())
        std::fprintf(stderr, "[rt] cpu: AddVectoredExceptionHandler failed (error %lu); instructions this CPU lacks "
                             "will stop the game\n", GetLastError());

    try {
        rt::Loader loader(game_dir);
        rt::Module& main_module = loader.load(image);  // maps, binds and installs TLS; runs no image code
        // An executable with TLS that did not get index 0: its /GA code would read and write the TLS in slot 0, which
        // is gta6rt's own (or another executable's), so no code of it may run.
        const auto* main_nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(
            main_module.base + reinterpret_cast<const IMAGE_DOS_HEADER*>(main_module.base)->e_lfanew);
        const bool main_is_exe = (main_nt->FileHeader.Characteristics & IMAGE_FILE_DLL) == 0;
        if ((init || run) && main_is_exe && main_module.tls_slot != rt::kNoTlsIndex && main_module.tls_index != 0) {
            std::string why;
            if (rt::reserve_tls_index0(&why)) why = "another mapped executable holds it";
            std::fprintf(stderr,
                         "error: %s cannot have module TLS index 0 (%s); code built with /GA would read and write "
                         "another module's TLS through slot 0, so it is not run\n",
                         main_module.name.c_str(), why.c_str());
            return 1;
        }
        // The mods' archetype swaps (docs/modding.md): hooked here, before any image code runs, when the lookup's RVA
        // for this executable is given. The count of redirected calls shows whether the RVA is the real lookup.
        if (view_mode_getter_rva) {
            const auto r = leonida::hooks::hook_image_results(main_module.base, view_mode_getter_rva,
                                                              &leonida::mods::override_view_mode);
            if (r.ok)
                std::fprintf(stderr, "[rt] mods: view-mode getter at RVA 0x%X hooked: %zu direct call(s) redirected\n",
                             view_mode_getter_rva, r.redirected);
            else
                std::fprintf(stderr, "[rt] mods: view-mode getter at RVA 0x%X not hooked: %s\n", view_mode_getter_rva,
                             r.error);
        }
        if (archetype_lookup_rva) {
            const auto r = leonida::hooks::hook_image_calls(
                main_module.base, archetype_lookup_rva,
                [](uint32_t hash) { return leonida::mods::active().remap(hash); }, archetype_lookup_arg);
            if (r.ok)
                std::fprintf(stderr, "[rt] mods: archetype lookup at RVA 0x%X hooked (argument %d): %zu direct "
                                     "call(s) redirected\n", archetype_lookup_rva, archetype_lookup_arg, r.redirected);
            else
                std::fprintf(stderr, "[rt] mods: archetype lookup at RVA 0x%X not hooked: %s\n", archetype_lookup_rva,
                             r.error);
        }
        if (run) rt::process::set_command_line(image.filename().wstring(), game_args);
        std::exception_ptr init_failure;
        if (init || run) {
            try {
                loader.initialise();  // TLS callbacks, then DllMain(DLL_PROCESS_ATTACH) for DLLs
                std::printf("%zu module(s) initialised\n", loader.modules().size());
            } catch (const std::exception&) {
                // Keep the mapped modules available for the binding report on an attach failure.
                init_failure = std::current_exception();
            }
        }
        int unimplemented = 0, unimplemented_hresult = 0;
        for (auto* m : loader.modules()) {
            std::map<rt::Binding, int> counts;
            for (auto& r : m->imports) counts[r.binding]++;
            std::printf("%-36s base=%p%s size=%u entry=+%#x  shim=%d host=%d module=%d unimplemented=%d\n",
                        m->name.c_str(), m->base, m->relocated ? " (relocated)" : "", m->size, m->entry_rva,
                        counts[rt::Binding::Shim], counts[rt::Binding::Host], counts[rt::Binding::Module],
                        counts[rt::Binding::Unimplemented]);
            for (auto& r : m->imports)
                if (r.binding == rt::Binding::Unimplemented || r.binding == rt::Binding::Shim) {
                    const bool missing = r.binding == rt::Binding::Unimplemented;
                    const bool hresult =
                        missing && rt::shims::classify_import(r.dll, r.name) == rt::shims::ReturnClass::HResult;
                    std::printf("    %-13s %s!%s%s\n", binding_name(r.binding), r.dll.c_str(), r.name.c_str(),
                                hresult ? "  [returns HRESULT]" : "");
                    unimplemented += missing;
                    unimplemented_hresult += hresult;
                }
        }
        std::printf("%zu module(s) mapped, %d unimplemented import(s)\n", loader.modules().size(), unimplemented);
        std::printf("unimplemented by return policy: hresult=%d other=%d\n", unimplemented_hresult,
                    unimplemented - unimplemented_hresult);
        if (init_failure) {
            std::fflush(stdout);
            std::rethrow_exception(init_failure);
        }
        if (std::getenv("LEONIDA_HEAP_PROBE")) {
            std::fprintf(stderr, "[rt] heap-probe: after mapping\n");
            std::fflush(stderr);
            const BOOL ok = HeapValidate(reinterpret_cast<HANDLE>(_get_heap_handle()), 0, nullptr);
            std::fprintf(stderr, "[rt] heap-probe: after mapping ok=%d\n", ok);
            std::fflush(stderr);
        }
        if (!run) {
            rt::startup_trace::event("exit", "\"code\":0");
            return 0;
        }
        if (!main_module.entry_rva) {
            std::fprintf(stderr, "error: %s has no entry point\n", main_module.name.c_str());
            return 1;
        }
        const std::filesystem::path scratch_dir = rt::fs::scratch_dir();
        const std::wstring paths = L"console paths: GAME: -> " + game_dir.native() + L", writes -> " +
                                   overlay_dir.native() + L", T: -> " + scratch_dir.native();
        const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
        DWORD mode;
        bool printed = false;
        if (GetConsoleMode(out, &mode)) {
            std::fflush(stdout);
            const std::wstring line = paths + L"\r\n";
            DWORD written;
            printed = WriteConsoleW(out, line.data(), static_cast<DWORD>(line.size()), &written, nullptr) != FALSE;
        }
        if (!printed) {
            const auto game = game_dir.u8string();
            const auto overlay = overlay_dir.u8string();
            const auto scratch = scratch_dir.u8string();
            std::printf("console paths: GAME: -> %s, writes -> %s, T: -> %s\n",
                        reinterpret_cast<const char*>(game.c_str()), reinterpret_cast<const char*>(overlay.c_str()),
                        reinterpret_cast<const char*>(scratch.c_str()));
        }
        std::printf("running %s\n", main_module.name.c_str());
        std::fflush(stdout);
        keep_title_watchdog_kicked(main_module.base);
        start_title_peek(main_module.base);
        leonida::dev_overlay::start();  // LEONIDA_DEV_OVERLAY=1
        leonida::thread_sampler::start(main_module.base, main_module.size);  // LEONIDA_SAMPLE_MS (diagnostic)
#ifdef LEONIDA_VULKAN_DEVICE
        SetConsoleCtrlHandler(&stop_compiles_on_console_exit, TRUE);
        previous_fatal_filter = SetUnhandledExceptionFilter(&save_caches_on_fatal_exception);
#endif
        int code;
#ifdef LEONIDA_VULKAN_DEVICE
        leonida::thread_stack_hook.store(&rt::print_thread_stacks_now, std::memory_order_release);
#endif
        {
            auto stacks = rt::start_thread_stacks();
            code = rt::process::run_entry(main_module.base + main_module.entry_rva);
        }
        std::fflush(stdout);
        std::printf("game exited with code %d\n", code);
        rt::shims::report_called_unimplemented();
        rt::startup_trace::event("exit", "\"code\":" + std::to_string(code));
        std::fflush(stdout);
        std::fflush(stderr);
        // Not `return`: a normal host exit runs the CRT's atexit table, which the game shares with gta6rt (its
        // handlers would run against a game that has stopped) and whose last word decides the exit status.
        // ExitProcess also skips atexit, so the TLS clone cleanup registered there is run here instead.
        // The list-value threads end first: ExitProcess must not cut one off inside the layer (in a held submission,
        // holding its queues' lock).
        rt::d3d12_xs::stop_list_values(500);
#ifdef LEONIDA_VULKAN_DEVICE
        rt::d3d12::stop_pipeline_compiles(kExitCompileWaitMs);
        rt::d3d12::save_pipeline_caches_at_exit(kExitCacheWaitMs);
#endif
        rt::cleanup_tls_clones();
        ExitProcess(static_cast<UINT>(code));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        rt::startup_trace::event("error", "\"what\":" + rt::startup_trace::quote(e.what()));
        return 1;
    }
}
