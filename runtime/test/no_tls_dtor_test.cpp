// gta6rt.exe must not have a thread_local with a destructor. One makes the linker pull in the CRT's TLS destructor
// support (__dyn_tls_dtor), which the exe registers at startup through _register_thread_local_exe_atexit_callback;
// the retail title's CRT then makes the same registration in the same process, which the CRT ends with abort()
// (0xC0000409, 0.9 s into every boot: upstream 861bb0546's thread_local std::vector in d3d12_vulkan_device.cpp).
// Usage: no_tls_dtor_test <gta6rt.exe> (its PDB beside it). Fails when the symbol is in the PDB, or the PDB is missing.
#include <windows.h>
#include <dbghelp.h>
#include <cstdio>
#pragma comment(lib, "dbghelp.lib")

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: no_tls_dtor_test <exe>\n"); return 2; }
    const HANDLE process = reinterpret_cast<HANDLE>(static_cast<ULONG_PTR>(0x5157));  // any unique value, no process
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_FAIL_CRITICAL_ERRORS);
    if (!SymInitialize(process, nullptr, FALSE)) { std::fprintf(stderr, "SymInitialize failed %lu\n", GetLastError()); return 1; }
    const DWORD64 base = SymLoadModuleEx(process, nullptr, argv[1], nullptr, 0x10000000, 0, nullptr, 0);
    IMAGEHLP_MODULE64 module{};
    module.SizeOfStruct = sizeof(module);
    if (!base || !SymGetModuleInfo64(process, base, &module) || module.SymType != SymPdb) {
        std::fprintf(stderr, "no PDB symbols for %s: cannot check\n", argv[1]);
        SymCleanup(process);
        return 1;
    }
    // A symbol every build has, so a lookup that cannot work is not taken for an absent destructor callback.
    char buffer[sizeof(SYMBOL_INFO) + 256]{};
    auto* symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = 255;
    if (!SymFromName(process, "main", symbol) && !SymFromName(process, "wmain", symbol)) {
        std::fprintf(stderr, "symbol lookup does not work on %s: cannot check\n", argv[1]);
        SymCleanup(process);
        return 1;
    }
    const bool dtor = SymFromName(process, "__dyn_tls_dtor", symbol) != FALSE;
    SymCleanup(process);
    if (dtor) {
        std::fprintf(stderr, "%s has a thread_local with a destructor (__dyn_tls_dtor is linked in): the title's CRT "
                             "aborts at startup. Hold such a value through a plain pointer instead.\n", argv[1]);
        return 1;
    }
    std::printf("no thread_local destructor in %s\n", argv[1]);
    return 0;
}
