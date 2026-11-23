// XCRDAPI.dll: the console's DirectStorage entry point.
//
// The retail executable imports one name, DStorageGetFactory. Its call site (game+0x1d8e4d0) passes an interface id
// (game+0x62fc7d8, 62959d11-d45f-4884-a0a3-f7862ea243ae) and an out pointer, then requires a non-null factory and
// S_OK before calling the factory's vtable slot 8 with 0x02000000 and requiring a non-negative HRESULT. There is no
// dstorage.dll on this PC to forward to.
//
// This is the minimal safe factory: a process-wide object whose observed slot 8 succeeds. Unknown methods return
// E_NOTIMPL and never write through an out pointer; QueryInterface/AddRef/Release follow COM. It is a bring-up
// stopgap, not a DirectStorage implementation: no queue, file or memory is faked.
#include <windows.h>
#include <intrin.h>

#include <atomic>
#include <cstdint>
#include <cstdio>

#include "shims.h"
#include "startup_trace.h"

namespace {

struct Factory {
    const void** vtable;
    // The observed caller AddRefs/Releases the factory like any COM object (the registry classifies the import as
    // HRESULT, not void); the object itself is process-wide, so the count exists only to balance those calls.
    std::atomic<LONG> refs{1};
};

struct StorageFile {
    const void** vtable;
};
extern StorageFile g_storage_file;

HRESULT __stdcall factory_query_interface(Factory* self, const IID* /*iid*/, void** out) {
    if (!out) return E_POINTER;
    self->refs.fetch_add(1, std::memory_order_relaxed);
    *out = self;
    return S_OK;
}

ULONG __stdcall factory_add_ref(Factory* self) {
    return static_cast<ULONG>(self->refs.fetch_add(1, std::memory_order_relaxed) + 1);
}
ULONG __stdcall factory_release(Factory* self) {
    return static_cast<ULONG>(self->refs.fetch_sub(1, std::memory_order_relaxed) - 1);
}

// Observed slot 8: SetStagingBufferSize-like, one integer argument, HRESULT. A PC has no staging buffer to size, so
// accepting it is safe.
HRESULT __stdcall factory_set_staging_buffer_size(Factory* /*self*/, uint32_t /*size*/) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: factory slot 8 (staging buffer size) accepted; no PC staging buffer\n");
    }
    return S_OK;
}

// Every other unmeasured method: refuse, and do not touch an out pointer. One instantiation per slot so the log
// names the method the game actually reached.
template <int Slot>
HRESULT __stdcall factory_not_implemented_slot(Factory* /*self*/) {
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: factory slot %d -> E_NOTIMPL (caller %s)\n", Slot,
                     rt::startup_trace::where(reinterpret_cast<uint64_t>(_ReturnAddress())).c_str());
    }
    return E_NOTIMPL;
}

// Slot 4 is OpenFile: (const wchar_t* path, REFIID riid, void** file). The measured caller
// (game+0x28c8be0) stores the returned object and then opens the path itself with Win32 CreateFileW (via the file
// shims), so the object only has to be a valid non-null COM identity. No file data is faked here.
HRESULT __stdcall factory_slot4(Factory* /*self*/, const wchar_t* path, const IID* /*iid*/, void** out) {
    static std::atomic<bool> logged{false};
    if (!out) return E_POINTER;
    *out = &g_storage_file;
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        char utf8[512] = {};
        if (path) WideCharToMultiByte(CP_UTF8, 0, path, -1, utf8, sizeof utf8 - 1, nullptr, nullptr);
        std::fprintf(stderr, "[rt] xcrdapi: OpenFile(\"%s\") -> dummy file object; the caller opens it with Win32\n",
                     path ? utf8 : "(null)");
    }
    return S_OK;
}

HRESULT __stdcall file_query_interface(StorageFile* self, const IID* /*iid*/, void** out) {
    if (!out) return E_POINTER;
    *out = self;
    return S_OK;
}
ULONG __stdcall file_add_ref(StorageFile* /*self*/) { return 1; }
ULONG __stdcall file_release(StorageFile* /*self*/) { return 1; }
HRESULT __stdcall file_not_implemented(StorageFile* /*self*/) { return E_NOTIMPL; }

const void* g_file_vtable[] = {
    reinterpret_cast<const void*>(&file_query_interface),
    reinterpret_cast<const void*>(&file_add_ref),
    reinterpret_cast<const void*>(&file_release),
    reinterpret_cast<const void*>(&file_not_implemented),
    reinterpret_cast<const void*>(&file_not_implemented),
    reinterpret_cast<const void*>(&file_not_implemented),
    reinterpret_cast<const void*>(&file_not_implemented),
    reinterpret_cast<const void*>(&file_not_implemented),
};
StorageFile g_storage_file{g_file_vtable};

const void* g_factory_vtable[] = {
    reinterpret_cast<const void*>(&factory_query_interface),  // 0 IUnknown::QueryInterface
    reinterpret_cast<const void*>(&factory_add_ref),          // 1 IUnknown::AddRef
    reinterpret_cast<const void*>(&factory_release),          // 2 IUnknown::Release
    reinterpret_cast<const void*>(&factory_not_implemented_slot<3>),
    reinterpret_cast<const void*>(&factory_slot4),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<5>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<6>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<7>),
    reinterpret_cast<const void*>(&factory_set_staging_buffer_size),  // 8 observed by the retail exe
    reinterpret_cast<const void*>(&factory_not_implemented_slot<9>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<10>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<11>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<12>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<13>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<14>),
    reinterpret_cast<const void*>(&factory_not_implemented_slot<15>),
};

Factory g_factory{g_factory_vtable};

}  // namespace

HRESULT WINAPI DStorageGetFactory(REFIID /*riid*/, void** out) {
    static std::atomic<bool> logged{false};
    if (!out) return E_POINTER;
    if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::fprintf(stderr, "[rt] xcrdapi: DStorageGetFactory -> software factory (observed slot 8, other methods "
                             "E_NOTIMPL)\n");
    }
    *out = &g_factory;
    return S_OK;
}

SHIM_TABLE(xcrdapi,
    {"xcrdapi.dll", "DStorageGetFactory", reinterpret_cast<void*>(&DStorageGetFactory)},
);
