// Window-class and window creation for a manually mapped title.
//
// The retail executable imports the ANSI window calls from
// ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll and ShowWindow from
// ext-ms-win-rtcore-ntuser-window-l1-1-0.dll (pe_inspect on the shipped exe). It obtains its hInstance from
// GetModuleHandle(nullptr), which this runtime answers with the mapped image base (library_loader.cpp). user32 does
// not know that address as a module, so class registration and window creation can fail.
//
// When the hInstance is the mapped main image, register and create against the host's own module handle instead.
// The WNDPROC in the class is left untouched: it is game code in the mapped image, and user32 calls it directly.
// The class is therefore owned by gta6rt's image, which is what UnregisterClass and GetClassInfoEx must use too.
//
// Everything else is forwarded unchanged. All functions are registered under the two ext-ms names the exe imports
// and under user32.dll, for code linked the classic way.
#include <windows.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

#include "shims.h"

namespace rt::shims::window {

namespace {

// The handle user32 should see. The mapped main image is not an OS module; gta6rt's own instance is. A null
// hInstance keeps its documented meaning (the module that created the class, or a child window).
HINSTANCE host_instance(HINSTANCE instance) {
    if (!instance) return instance;
    if (void* mapped = bundled_main_module(); mapped && reinterpret_cast<void*>(instance) == mapped)
        return GetModuleHandleW(nullptr);
    return instance;
}

std::string atom_name(const void* name) {
    return "#" + std::to_string(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(name)));
}

std::string narrow_class(const wchar_t* name) {
    if (!name) return "(null)";
    if (reinterpret_cast<uintptr_t>(name) < 0x10000) return atom_name(name);  // MAKEINTATOM
    char buffer[256];
    const int length = WideCharToMultiByte(CP_UTF8, 0, name, -1, buffer, sizeof(buffer), nullptr, nullptr);
    if (length <= 0) return "(wide class name)";
    return buffer;
}

std::string narrow_class(const char* name) {
    if (!name) return "(null)";
    if (reinterpret_cast<uintptr_t>(name) < 0x10000) return atom_name(name);  // MAKEINTATOM
    return name;
}

void log_register_once(const char* api, const std::string& class_name, HINSTANCE requested, HINSTANCE used, ATOM atom,
                       DWORD error) {
    static std::once_flag once;
    std::call_once(once, [&] {
        std::fprintf(stderr, "[rt] window: %s class=\"%s\" hInstance=%p used=%p -> %s", api, class_name.c_str(),
                     requested, used, atom ? "registered" : "failed");
        if (!atom) std::fprintf(stderr, " (error %lu)", static_cast<unsigned long>(error));
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    });
}

void log_create_once(const char* api, const std::string& class_name, DWORD style, HINSTANCE requested,
                     HINSTANCE used, HWND hwnd, DWORD error) {
    static std::once_flag once;
    std::call_once(once, [&] {
        std::fprintf(stderr, "[rt] window: %s class=\"%s\" style=0x%08lx hInstance=%p used=%p -> %s", api,
                     class_name.c_str(), static_cast<unsigned long>(style), requested, used,
                     hwnd ? "created" : "failed");
        if (!hwnd) std::fprintf(stderr, " (error %lu)", static_cast<unsigned long>(error));
        std::fprintf(stderr, "\n");
        std::fflush(stderr);
    });
}

}  // namespace

ATOM WINAPI register_class_ex_a(const WNDCLASSEXA* description) {
    if (!description) return RegisterClassExA(description);
    WNDCLASSEXA copy{};
    const UINT bytes = description->cbSize < sizeof(copy) ? description->cbSize : sizeof(copy);
    std::memcpy(&copy, description, bytes);
    const HINSTANCE requested = description->hInstance;
    copy.hInstance = host_instance(requested);
    const ATOM atom = RegisterClassExA(&copy);
    const DWORD error = GetLastError();
    log_register_once("RegisterClassExA", narrow_class(description->lpszClassName), requested, copy.hInstance, atom,
                      error);
    return atom;
}

ATOM WINAPI register_class_ex_w(const WNDCLASSEXW* description) {
    if (!description) return RegisterClassExW(description);
    WNDCLASSEXW copy{};
    const UINT bytes = description->cbSize < sizeof(copy) ? description->cbSize : sizeof(copy);
    std::memcpy(&copy, description, bytes);
    const HINSTANCE requested = description->hInstance;
    copy.hInstance = host_instance(requested);
    const ATOM atom = RegisterClassExW(&copy);
    const DWORD error = GetLastError();
    log_register_once("RegisterClassExW", narrow_class(description->lpszClassName), requested, copy.hInstance, atom,
                      error);
    return atom;
}

HWND WINAPI create_window_ex_a(DWORD ex_style, LPCSTR class_name, LPCSTR window_name, DWORD style, int x, int y,
                               int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param) {
    const HINSTANCE used = host_instance(instance);
    HWND hwnd = CreateWindowExA(ex_style, class_name, window_name, style, x, y, width, height, parent, menu, used,
                                param);
    const DWORD error = GetLastError();
    log_create_once("CreateWindowExA", narrow_class(class_name), style, instance, used, hwnd, error);
    return hwnd;
}

HWND WINAPI create_window_ex_w(DWORD ex_style, LPCWSTR class_name, LPCWSTR window_name, DWORD style, int x, int y,
                               int width, int height, HWND parent, HMENU menu, HINSTANCE instance, LPVOID param) {
    const HINSTANCE used = host_instance(instance);
    HWND hwnd = CreateWindowExW(ex_style, class_name, window_name, style, x, y, width, height, parent, menu, used,
                                param);
    const DWORD error = GetLastError();
    log_create_once("CreateWindowExW", narrow_class(class_name), style, instance, used, hwnd, error);
    return hwnd;
}

BOOL WINAPI unregister_class_a(LPCSTR class_name, HINSTANCE instance) {
    return UnregisterClassA(class_name, host_instance(instance));
}

BOOL WINAPI unregister_class_w(LPCWSTR class_name, HINSTANCE instance) {
    return UnregisterClassW(class_name, host_instance(instance));
}

BOOL WINAPI get_class_info_ex_a(HINSTANCE instance, LPCSTR class_name, WNDCLASSEXA* description) {
    return GetClassInfoExA(host_instance(instance), class_name, description);
}

BOOL WINAPI get_class_info_ex_w(HINSTANCE instance, LPCWSTR class_name, WNDCLASSEXW* description) {
    return GetClassInfoExW(host_instance(instance), class_name, description);
}

LRESULT WINAPI def_window_proc_a(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    return DefWindowProcA(hwnd, message, wparam, lparam);
}

LRESULT WINAPI def_window_proc_w(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    return DefWindowProcW(hwnd, message, wparam, lparam);
}

BOOL WINAPI show_window(HWND hwnd, int command) { return ShowWindow(hwnd, command); }

}  // namespace rt::shims::window

// merge: CreateWindowExA/W are registered once, by lifecycle.cpp (the title-window fitting and SC_CLOSE hook), which
// creates through create_window_ex_a/w here; two registrations made the winner depend on link order.
#define WINDOW(dll)                                                                                                  \
    {dll, "RegisterClassExA", reinterpret_cast<void*>(&rt::shims::window::register_class_ex_a)},                     \
    {dll, "RegisterClassExW", reinterpret_cast<void*>(&rt::shims::window::register_class_ex_w)},                     \
    {dll, "DefWindowProcA", reinterpret_cast<void*>(&rt::shims::window::def_window_proc_a)},                         \
    {dll, "DefWindowProcW", reinterpret_cast<void*>(&rt::shims::window::def_window_proc_w)},                         \
    {dll, "ShowWindow", reinterpret_cast<void*>(&rt::shims::window::show_window)},                                   \
    {dll, "UnregisterClassA", reinterpret_cast<void*>(&rt::shims::window::unregister_class_a)},                      \
    {dll, "UnregisterClassW", reinterpret_cast<void*>(&rt::shims::window::unregister_class_w)},                      \
    {dll, "GetClassInfoExA", reinterpret_cast<void*>(&rt::shims::window::get_class_info_ex_a)},                      \
    {dll, "GetClassInfoExW", reinterpret_cast<void*>(&rt::shims::window::get_class_info_ex_w)}

SHIM_TABLE(window,
    WINDOW("ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll"),
    WINDOW("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll"),
    WINDOW("user32.dll"),
);
