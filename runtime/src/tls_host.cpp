// Minimal TLS anchor DLL for the native loader.
//
// Loading this image makes the Windows loader assign it a per-module TLS index and allocate a block for that index in
// every thread's TEB. Loader::initialize_tls reuses that reserved index for the manually mapped image, so
// __declspec(thread) access in mapped code reads a valid TEB slot. The anchor itself is never used; the loader saves
// and restores the helper's own block, so the helper's CRT TLS state is not disturbed.
//
// The Windows loader also calls this DLL's DllMain with DLL_THREAD_ATTACH on every new thread (after it has built that
// thread's TLS array) and with DLL_THREAD_DETACH when a thread exits (before it frees the array). The loader registers
// a callback here so it can give each new thread its own copy of the mapped image's TLS, and put this DLL's block back
// before the thread exits. DllMain itself must not touch TLS: this DLL's slot is the one being swapped.
#include <windows.h>

__declspec(thread) unsigned char g_tls_host_anchor = 0;

namespace {

using ThreadEventFn = void(__cdecl*)(DWORD reason);

// A plain global, not TLS (see above). Written once by the loader, read on every thread event.
void* volatile g_thread_event = nullptr;

}  // namespace

// Never called. The export keeps the TLS variable (and therefore the image's TLS directory) from being dropped by
// the linker without making DllMain touch TLS during thread attach/detach.
extern "C" __declspec(dllexport) void gta6rt_tls_host_keep() { g_tls_host_anchor = 1; }

// Called by the loader. `fn` runs inside DllMain (under the Windows loader lock) for DLL_THREAD_ATTACH and
// DLL_THREAD_DETACH; nullptr stops the calls.
extern "C" __declspec(dllexport) void gta6rt_tls_host_set_thread_event(ThreadEventFn fn) {
    InterlockedExchangePointer(&g_thread_event, reinterpret_cast<void*>(fn));
}

// This is the image's entry point, not DllMain behind the C runtime's _DllMainCRTStartup. On DLL_THREAD_ATTACH the
// CRT wrapper does its own per-thread bookkeeping *after* this function returns, in the module TLS block. The loader
// has replaced that block with the mapped image's block, so the wrapper would treat the image's __declspec(thread)
// data as its _ptd and corrupt the process heap during thread attach/detach. This DLL uses no CRT state, so handle
// the thread events directly (the entry point receives exactly DllMain's arguments).
extern "C" BOOL WINAPI gta6rt_tls_host_entry(HINSTANCE, DWORD reason, LPVOID) {
    // The OS loader reserves a module TLS index only for a DLL with a non-empty import directory; an image with none
    // is initialised without one (so its TLS index storage stays 0, colliding with the host EXE's index). Keep a
    // kernel32 import in the image even though this CRT-free entry needs no other import.
    (void)GetCurrentProcessId();
    if (reason == DLL_THREAD_ATTACH || reason == DLL_THREAD_DETACH) {
        void* fn = InterlockedCompareExchangePointer(&g_thread_event, nullptr, nullptr);
        if (fn) reinterpret_cast<ThreadEventFn>(fn)(reason);
    }
    return TRUE;
}
