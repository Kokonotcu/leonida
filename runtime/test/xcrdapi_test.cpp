// XCRDAPI.dll!DStorageGetFactory: the console's DirectStorage entry point answers a non-null factory token.
// The retail executable's first XCRDAPI call stops at this name; this test keeps the registry binding and the
// IUnknown head honest without loading DirectStorage or running the title.
#include <windows.h>

#include <cstdio>

#include "shims.h"

int main() {
    void* fn = rt::shims::find("xcrdapi.dll", "DStorageGetFactory");
    if (!fn) {
        std::fprintf(stderr, "xcrdapi.dll!DStorageGetFactory is not registered\n");
        return 1;
    }
    using GetFactory = HRESULT(WINAPI*)(const IID*, void**);
    IID iid{};
    void* factory = nullptr;
    const HRESULT hr = reinterpret_cast<GetFactory>(fn)(&iid, &factory);
    if (FAILED(hr) || !factory) {
        std::fprintf(stderr, "DStorageGetFactory returned hr=0x%08lx out=%p\n", static_cast<unsigned long>(hr),
                     factory);
        return 1;
    }
    auto** vtable = *reinterpret_cast<void***>(factory);
    if (!vtable || !vtable[1] || !vtable[2]) {
        std::fprintf(stderr, "the factory token has no IUnknown vtable\n");
        return 1;
    }
    using AddRef = ULONG(WINAPI*)(void*);
    using Release = ULONG(WINAPI*)(void*);
    if (reinterpret_cast<AddRef>(vtable[1])(factory) != 2) {
        std::fprintf(stderr, "AddRef did not reach 2\n");
        return 1;
    }
    if (reinterpret_cast<Release>(vtable[2])(factory) != 1) {
        std::fprintf(stderr, "Release did not return to 1\n");
        return 1;
    }
    std::printf("DStorageGetFactory returned a non-null IUnknown factory token\n");
    return 0;
}
