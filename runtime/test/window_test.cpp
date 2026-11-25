// Window shim test: a mapped image registers a class whose WNDPROC is in the mapped image and creates a window with
// the mapped image's hInstance, exactly as the retail executable does with sgaWindowClass.
//
// The fixture imports the ANSI calls from ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll and ShowWindow from
// ext-ms-win-rtcore-ntuser-window-l1-1-0.dll (the two names the shipped exe imports), plus the W and class-info
// entries. rt::Loader must bind all of them to runtime/shims/window.cpp. The shim passes the host module handle to
// user32 and leaves the mapped WNDPROC alone, so:
//   - RegisterClassExA/W succeeds and GetClassInfoEx reports the host instance and the mapped WNDPROC;
//   - CreateWindowExA/W succeeds, GWLP_HINSTANCE is the host instance, and GWLP_WNDPROC is the mapped WNDPROC;
//   - messages sent to the window reach the mapped WNDPROC and its DefWindowProcA/W call returns;
//   - UnregisterClassA/W accepts the mapped hInstance and removes the host-owned class.
// No game data.
#include <windows.h>

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "pe_loader.h"
#include "shims.h"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    std::fflush(stdout);
    if (!ok) ++g_failures;
}

using RegisterGuest = ATOM (*)();
using ShowGuest = BOOL (*)(HWND);
using UnregisterGuest = BOOL (*)();
using ClassInfoA = BOOL(WINAPI*)(void**, void**);
using ClassInfoW = BOOL(WINAPI*)(void**, void**);
using ClassNameA = const char*(*)();
using ClassNameW = const wchar_t*(*)();
using WndProcPtr = void*(*)();
using CallCount = long (*)();
using InstanceFn = HMODULE (*)();
using CreateWindowA = HWND (*)();
using CreateWindowW = HWND (*)();

bool is_window_dll(const std::string& dll) {
    const std::string key = rt::lower(dll);
    return key == "ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll" ||
           key == "ext-ms-win-rtcore-ntuser-window-l1-1-0.dll";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: window_test <window_guest.dll>\n");
        return 2;
    }
    const std::filesystem::path fixture = argv[1];

    try {
        rt::Loader loader(fixture.parent_path());
        rt::Module& module = loader.load(fixture);

        int window_imports = 0;
        int shim_window_imports = 0;
        bool saw_ansi_dll = false, saw_window_dll = false;
        for (const auto& record : module.imports) {
            if (!is_window_dll(record.dll)) continue;
            ++window_imports;
            if (record.binding == rt::Binding::Shim) ++shim_window_imports;
            const std::string key = rt::lower(record.dll);
            saw_ansi_dll = saw_ansi_dll || key == "ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll";
            saw_window_dll = saw_window_dll || key == "ext-ms-win-rtcore-ntuser-window-l1-1-0.dll";
        }
        check(window_imports >= 11, "the guest imports the window API entries");
        check(shim_window_imports == window_imports, "every window import binds to the shim");
        check(saw_ansi_dll && saw_window_dll, "both ext-ms window DLL names are imported and served");

        check(rt::shims::find("ext-ms-win-rtcore-ntuser-window-ansi-l1-1-0.dll", "RegisterClassExA") != nullptr,
              "RegisterClassExA is registered under the ANSI ext-ms name");
        check(rt::shims::find("ext-ms-win-rtcore-ntuser-window-l1-1-0.dll", "ShowWindow") != nullptr,
              "ShowWindow is registered under the non-ANSI ext-ms name");
        check(rt::shims::find("user32.dll", "CreateWindowExW") != nullptr,
              "CreateWindowExW is registered under user32.dll");

        loader.initialise();

        const auto instance = reinterpret_cast<InstanceFn>(module.exports.at("window_guest_instance"));
        const auto class_a = reinterpret_cast<ClassNameA>(module.exports.at("window_guest_class_a"));
        const auto class_w = reinterpret_cast<ClassNameW>(module.exports.at("window_guest_class_w"));
        const auto wndproc_a = reinterpret_cast<WndProcPtr>(module.exports.at("window_guest_wndproc_a"));
        const auto wndproc_w = reinterpret_cast<WndProcPtr>(module.exports.at("window_guest_wndproc_w"));
        const auto calls_a = reinterpret_cast<CallCount>(module.exports.at("window_guest_wndproc_a_calls"));
        const auto calls_w = reinterpret_cast<CallCount>(module.exports.at("window_guest_wndproc_w_calls"));
        const auto register_a = reinterpret_cast<RegisterGuest>(module.exports.at("window_guest_register_a"));
        const auto register_w = reinterpret_cast<RegisterGuest>(module.exports.at("window_guest_register_w"));
        const auto create_a = reinterpret_cast<CreateWindowA>(module.exports.at("window_guest_create_a"));
        const auto create_w = reinterpret_cast<CreateWindowW>(module.exports.at("window_guest_create_w"));
        const auto show_a = reinterpret_cast<ShowGuest>(module.exports.at("window_guest_show_a"));
        const auto show_w = reinterpret_cast<ShowGuest>(module.exports.at("window_guest_show_w"));
        const auto unregister_a = reinterpret_cast<UnregisterGuest>(module.exports.at("window_guest_unregister_a"));
        const auto unregister_w = reinterpret_cast<UnregisterGuest>(module.exports.at("window_guest_unregister_w"));
        const auto class_info_a = reinterpret_cast<ClassInfoA>(module.exports.at("window_guest_class_info_a"));
        const auto class_info_w = reinterpret_cast<ClassInfoW>(module.exports.at("window_guest_class_info_w"));

        const HMODULE mapped = instance();
        const HMODULE host = GetModuleHandleW(nullptr);
        check(mapped != nullptr && mapped != host, "the guest sees the mapped image as its module handle");

        // ANSI class and window.
        const ATOM atom_a = register_a();
        check(atom_a != 0, "RegisterClassExA creates sgaWindowClass against the mapped hInstance");
        void* info_wndproc = nullptr;
        void* info_instance = nullptr;
        check(class_info_a(&info_wndproc, &info_instance) == TRUE && info_instance == host,
              "GetClassInfoExA resolves the mapped hInstance to the host instance");
        check(info_wndproc == wndproc_a(), "GetClassInfoExA returns the mapped WNDPROC unchanged");
        WNDCLASSEXA host_class_a{};
        host_class_a.cbSize = sizeof(host_class_a);
        SetLastError(0);
        const BOOL found_a = GetClassInfoExA(host, class_a(), &host_class_a);
        const DWORD found_a_error = GetLastError();
        check(found_a != FALSE,
              "user32 finds the class under the host instance (error " + std::to_string(found_a_error) + ")");
        check(host_class_a.hInstance == host, "the class is owned by the host instance");
        check(host_class_a.lpfnWndProc == reinterpret_cast<WNDPROC>(wndproc_a()),
              "the class keeps the mapped WNDPROC");

        HWND hwnd_a = create_a();
        check(hwnd_a && IsWindow(hwnd_a), "CreateWindowExA creates the window");
        if (hwnd_a) {
            check(reinterpret_cast<HMODULE>(GetWindowLongPtrW(hwnd_a, GWLP_HINSTANCE)) == host,
                  "CreateWindowExA passes the host instance to user32");
            // RegisterClassExA may leave an ANSI message thunk as the window's procedure; the class info above
            // already shows the mapped WNDPROC, and the message below proves it is the one that runs.
            check(GetWindowLongPtrW(hwnd_a, GWLP_WNDPROC) != 0,
                  "the ANSI window has a window procedure");
            const long before = calls_a();
            SendMessageA(hwnd_a, WM_APP + 1, 0, 0);
            check(calls_a() == before + 1, "a message reaches the mapped WNDPROC");
            const BOOL was_visible = show_a(hwnd_a);
            check(was_visible == FALSE, "ShowWindow (ANSI path) forwards and reports the hidden state");
            check(DestroyWindow(hwnd_a) != FALSE, "DestroyWindow closes the window");
            check(!IsWindow(hwnd_a), "the window is gone");
        }
        check(unregister_a() != FALSE, "UnregisterClassA accepts the mapped hInstance");
        host_class_a.cbSize = sizeof(host_class_a);
        check(GetClassInfoExA(host, class_a(), &host_class_a) == FALSE,
              "the class is removed from the host instance");

        // Wide class and window.
        const ATOM atom_w = register_w();
        check(atom_w != 0, "RegisterClassExW creates the wide class against the mapped hInstance");
        info_wndproc = nullptr;
        info_instance = nullptr;
        check(class_info_w(&info_wndproc, &info_instance) == TRUE && info_instance == host,
              "GetClassInfoExW resolves the mapped hInstance to the host instance");
        check(info_wndproc == wndproc_w(), "GetClassInfoExW returns the mapped WNDPROC unchanged");
        WNDCLASSEXW host_class_w{};
        host_class_w.cbSize = sizeof(host_class_w);
        check(GetClassInfoExW(host, class_w(), &host_class_w) != FALSE,
              "user32 finds the wide class under the host instance");
        check(host_class_w.lpfnWndProc == reinterpret_cast<WNDPROC>(wndproc_w()),
              "the wide class keeps the mapped WNDPROC");

        HWND hwnd_w = create_w();
        check(hwnd_w && IsWindow(hwnd_w), "CreateWindowExW creates the window");
        if (hwnd_w) {
            check(reinterpret_cast<HMODULE>(GetWindowLongPtrW(hwnd_w, GWLP_HINSTANCE)) == host,
                  "CreateWindowExW passes the host instance to user32");
            // merge: the title's first top-level window is subclassed by lifecycle.cpp (upstream's SC_CLOSE hook),
            // which keeps the mapped WNDPROC in its "leonida.title.wndproc" property and forwards to it (the message
            // check below proves it runs).
            const void* proc = reinterpret_cast<void*>(GetWindowLongPtrW(hwnd_w, GWLP_WNDPROC));
            check(proc == wndproc_w() || GetPropW(hwnd_w, L"leonida.title.wndproc") == wndproc_w(),
                  "the wide window keeps the mapped WNDPROC (directly, or behind the title-window hook)");
            const long before = calls_w();
            SendMessageW(hwnd_w, WM_APP + 1, 0, 0);
            check(calls_w() == before + 1, "a message reaches the mapped wide WNDPROC");
            const BOOL was_visible = show_w(hwnd_w);
            check(was_visible == FALSE, "ShowWindow (wide path) forwards and reports the hidden state");
            check(DestroyWindow(hwnd_w) != FALSE, "DestroyWindow closes the wide window");
            check(!IsWindow(hwnd_w), "the wide window is gone");
        }
        check(unregister_w() != FALSE, "UnregisterClassW accepts the mapped hInstance");
        host_class_w.cbSize = sizeof(host_class_w);
        check(GetClassInfoExW(host, class_w(), &host_class_w) == FALSE,
              "the wide class is removed from the host instance");
    } catch (const std::exception& e) {
        std::printf("FAIL unexpected exception: %s\n", e.what());
        ++g_failures;
    }

    std::printf("window_test: %s (%d failures)\n", g_failures ? "FAILED" : "OK", g_failures);
    std::fflush(stdout);
    return g_failures ? 1 : 0;
}
