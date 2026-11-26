#pragma once

// RenderDoc frame captures (the debugging environment for rendering: every draw, dispatch, resource, shader and
// pipeline state of a frame, replayable and bisectable offline in qrenderdoc or renderdoccmd, with nothing of it
// built into the runtime).
//
// LEONIDA_RENDERDOC=<list>: load RenderDoc's in-process library before the Vulkan instance is made, which turns on its
// Vulkan layer, and capture the presents in <list> (comma-separated present numbers, as LEONIDA_CAPTURE numbers its
// frames). "key" captures only on the capture key (F12, or Print Screen). Captures go to LEONIDA_RENDERDOC_DIR
// (default D:\GTA6\port\work\rdc) as leonida_frame<N>.rdc. LEONIDA_RENDERDOC_DLL names another renderdoc.dll. The layer
// hides the extensions it cannot capture, and the runtime takes the paths it uses without them.

#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../third_party/renderdoc/renderdoc_app.h"

namespace leonida::renderdoc {

struct State {
    RENDERDOC_API_1_6_0* api = nullptr;
    std::vector<std::uint64_t> presents;  // sorted; a capture is triggered at the present before each
    bool loaded = false;
};

inline State& state() {
    static State s;
    return s;
}

// Before vkCreateInstance. Nothing without LEONIDA_RENDERDOC.
inline void load() {
    State& s = state();
    if (s.loaded) return;
    s.loaded = true;
    const char* list = std::getenv("LEONIDA_RENDERDOC");
    if (!list || !*list) return;
    HMODULE module = GetModuleHandleW(L"renderdoc.dll");
    if (!module) {
        const char* path = std::getenv("LEONIDA_RENDERDOC_DLL");
        module = path && *path ? LoadLibraryA(path) : LoadLibraryW(L"C:\\Program Files\\RenderDoc\\renderdoc.dll");
    }
    const auto get_api = module ? reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(module, "RENDERDOC_GetAPI")) : nullptr;
    if (!get_api || !get_api(eRENDERDOC_API_Version_1_6_0, reinterpret_cast<void**>(&s.api)) || !s.api) {
        std::fprintf(stderr, "[rt] renderdoc: LEONIDA_RENDERDOC is set but renderdoc.dll could not be loaded\n");
        s.api = nullptr;
        return;
    }
    const char* dir = std::getenv("LEONIDA_RENDERDOC_DIR");
    const std::string folder = dir && *dir ? dir : "D:\\GTA6\\port\\work\\rdc";
    CreateDirectoryA(folder.c_str(), nullptr);
    const std::string base = folder + "\\leonida";
    s.api->SetCaptureFilePathTemplate(base.c_str());
    s.api->SetCaptureOptionU32(eRENDERDOC_Option_CaptureCallstacks, 0);
    s.api->SetCaptureOptionU32(eRENDERDOC_Option_RefAllResources, 0);
    RENDERDOC_InputButton keys[] = {eRENDERDOC_Key_F12, eRENDERDOC_Key_PrtScrn};
    s.api->SetCaptureKeys(keys, 2);
    if (std::string(list) != "key") {
        for (const char* at = list; *at;) {
            char* end = nullptr;
            const unsigned long long n = std::strtoull(at, &end, 10);
            if (end == at) break;
            if (n > 1) s.presents.push_back(n);
            at = *end == ',' ? end + 1 : end;
        }
        std::sort(s.presents.begin(), s.presents.end());
    }
    std::fprintf(stderr, "[rt] renderdoc: loaded; captures %s into %s\n",
                 s.presents.empty() ? "on the capture key (F12)" : list, folder.c_str());
}

// After present number `shown` was queued: the next present is captured when it is in the list.
inline void after_present(std::uint64_t shown) {
    State& s = state();
    if (!s.api || s.presents.empty()) return;
    if (std::binary_search(s.presents.begin(), s.presents.end(), shown + 1)) s.api->TriggerCapture();
}

}  // namespace leonida::renderdoc
