// Import stubs for the wide window entry points and ShowWindow, which the retail executable imports from
// ext-ms-win-rtcore-ntuser-window-l1-1-0.dll. Like window_ansi_stub, this DLL supplies only import names: the loader
// binds them to runtime/shims/window.cpp. No game data.
#include <windows.h>

extern "C" {

ATOM WINAPI RegisterClassExW(const WNDCLASSEXW*) { return 0; }

HWND WINAPI CreateWindowExW(DWORD, LPCWSTR, LPCWSTR, DWORD, int, int, int, int, HWND, HMENU, HINSTANCE, LPVOID) {
    return nullptr;
}

LRESULT WINAPI DefWindowProcW(HWND, UINT, WPARAM, LPARAM) { return 0; }

BOOL WINAPI ShowWindow(HWND, int) { return FALSE; }

BOOL WINAPI UnregisterClassA(LPCSTR, HINSTANCE) { return FALSE; }

BOOL WINAPI UnregisterClassW(LPCWSTR, HINSTANCE) { return FALSE; }

BOOL WINAPI GetClassInfoExA(HINSTANCE, LPCSTR, WNDCLASSEXA*) { return FALSE; }

BOOL WINAPI GetClassInfoExW(HINSTANCE, LPCWSTR, WNDCLASSEXW*) { return FALSE; }

}  // extern "C"
