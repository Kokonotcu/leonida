// Import stubs for the ANSI window entry points the retail executable imports from
// ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll. window_guest.dll links against this DLL, which gives its import
// table that exact module name. The stub DLL is kept out of the guest's folder and is never loaded: rt::Loader binds
// the names to runtime/shims/window.cpp instead. No game data.
#include <windows.h>

extern "C" {

ATOM WINAPI RegisterClassExA(const WNDCLASSEXA*) { return 0; }

HWND WINAPI CreateWindowExA(DWORD, LPCSTR, LPCSTR, DWORD, int, int, int, int, HWND, HMENU, HINSTANCE, LPVOID) {
    return nullptr;
}

LRESULT WINAPI DefWindowProcA(HWND, UINT, WPARAM, LPARAM) { return 0; }

}  // extern "C"
