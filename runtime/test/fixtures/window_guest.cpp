// A stand-in for the retail executable's window setup, mapped by the loader in window_test.cpp.
//
// It imports the ANSI window calls from ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll, ShowWindow from
// ext-ms-win-rtcore-ntuser-window-l1-1-0.dll, and the W variants plus UnregisterClass/GetClassInfoEx from the
// second name. The import stubs live in fixtures/stubs/window, not beside this DLL, and /NODEFAULTLIB:user32.lib
// keeps the linker from resolving the calls to the host's user32.lib instead. At run time the shim registry answers
// every one of them, so the mapped WNDPROC below is what user32 calls. No game data.
#include <windows.h>

#pragma comment(linker, "/NODEFAULTLIB:user32.lib")

namespace {

constexpr char kClassA[] = "sgaWindowClass";
constexpr wchar_t kClassW[] = L"sgaWindowClassW";

volatile LONG g_wndproc_a_calls = 0;
volatile LONG g_wndproc_w_calls = 0;

LRESULT CALLBACK wndproc_a(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_APP + 1) InterlockedIncrement(&g_wndproc_a_calls);
    return DefWindowProcA(hwnd, message, wparam, lparam);
}

LRESULT CALLBACK wndproc_w(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    if (message == WM_APP + 1) InterlockedIncrement(&g_wndproc_w_calls);
    return DefWindowProcW(hwnd, message, wparam, lparam);
}

}  // namespace

extern "C" __declspec(dllexport) HMODULE window_guest_instance() { return GetModuleHandleW(nullptr); }

extern "C" __declspec(dllexport) const char* window_guest_class_a() { return kClassA; }
extern "C" __declspec(dllexport) const wchar_t* window_guest_class_w() { return kClassW; }
extern "C" __declspec(dllexport) void* window_guest_wndproc_a() { return reinterpret_cast<void*>(&wndproc_a); }
extern "C" __declspec(dllexport) void* window_guest_wndproc_w() { return reinterpret_cast<void*>(&wndproc_w); }
extern "C" __declspec(dllexport) long window_guest_wndproc_a_calls() { return g_wndproc_a_calls; }
extern "C" __declspec(dllexport) long window_guest_wndproc_w_calls() { return g_wndproc_w_calls; }

extern "C" __declspec(dllexport) ATOM window_guest_register_a() {
    WNDCLASSEXA description{};
    description.cbSize = sizeof(description);
    description.style = CS_HREDRAW | CS_VREDRAW;
    description.lpfnWndProc = &wndproc_a;
    description.hInstance = GetModuleHandleW(nullptr);
    description.hbrBackground = reinterpret_cast<HBRUSH>(static_cast<INT_PTR>(COLOR_WINDOW + 1));
    description.lpszClassName = kClassA;
    return RegisterClassExA(&description);
}

extern "C" __declspec(dllexport) ATOM window_guest_register_w() {
    WNDCLASSEXW description{};
    description.cbSize = sizeof(description);
    description.style = CS_HREDRAW | CS_VREDRAW;
    description.lpfnWndProc = &wndproc_w;
    description.hInstance = GetModuleHandleW(nullptr);
    description.hbrBackground = reinterpret_cast<HBRUSH>(static_cast<INT_PTR>(COLOR_WINDOW + 1));
    description.lpszClassName = kClassW;
    return RegisterClassExW(&description);
}

extern "C" __declspec(dllexport) HWND window_guest_create_a() {
    return CreateWindowExA(0, kClassA, "Leonida window shim guest (ANSI)", WS_POPUP, 0, 0, 64, 64, nullptr,
                           nullptr, GetModuleHandleW(nullptr), nullptr);
}

extern "C" __declspec(dllexport) HWND window_guest_create_w() {
    return CreateWindowExW(0, kClassW, L"Leonida window shim guest (wide)", WS_POPUP, 0, 0, 64, 64, nullptr,
                           nullptr, GetModuleHandleW(nullptr), nullptr);
}

extern "C" __declspec(dllexport) BOOL window_guest_show_a(HWND hwnd) { return ShowWindow(hwnd, SW_HIDE); }
extern "C" __declspec(dllexport) BOOL window_guest_show_w(HWND hwnd) { return ShowWindow(hwnd, SW_HIDE); }

extern "C" __declspec(dllexport) BOOL window_guest_class_info_a(void** wndproc, void** instance) {
    WNDCLASSEXA description{};
    description.cbSize = sizeof(description);
    if (!GetClassInfoExA(GetModuleHandleW(nullptr), kClassA, &description)) return FALSE;
    if (wndproc) *wndproc = reinterpret_cast<void*>(description.lpfnWndProc);
    if (instance) *instance = description.hInstance;
    return TRUE;
}

extern "C" __declspec(dllexport) BOOL window_guest_class_info_w(void** wndproc, void** instance) {
    WNDCLASSEXW description{};
    description.cbSize = sizeof(description);
    if (!GetClassInfoExW(GetModuleHandleW(nullptr), kClassW, &description)) return FALSE;
    if (wndproc) *wndproc = reinterpret_cast<void*>(description.lpfnWndProc);
    if (instance) *instance = description.hInstance;
    return TRUE;
}

extern "C" __declspec(dllexport) BOOL window_guest_unregister_a() {
    return UnregisterClassA(kClassA, GetModuleHandleW(nullptr));
}

extern "C" __declspec(dllexport) BOOL window_guest_unregister_w() {
    return UnregisterClassW(kClassW, GetModuleHandleW(nullptr));
}
