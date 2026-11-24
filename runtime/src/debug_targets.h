#pragma once

// LEONIDA_DUMP_TARGETS=<present number>: the colour render targets the title draws into, collected as draws are
// recorded (d3d12_com.cpp record_draw), written out as PPM images by the presenter at that present
// (VulkanPresenter::dump_targets), into LEONIDA_CAPTURE's folder. A diagnostic: it shows what each target holds
// when the presented frame is black, without guessing from side effects.

#include "command_list.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace leonida::debug_targets {

inline std::uint64_t dump_at() {
    static const std::uint64_t at = [] {
        const char* v = std::getenv("LEONIDA_DUMP_TARGETS");
        return v ? std::strtoull(v, nullptr, 10) : 0ull;
    }();
    return at;
}

inline std::mutex& mutex() {
    static std::mutex m;
    return m;
}
inline std::vector<ResourceId>& targets() {
    static std::vector<ResourceId> t;
    return t;
}

// A render target a draw writes: the 32 most recently drawn distinct ones (the title recreates some targets every
// frame, so the first ones seen are gone by the dump).
inline void note(ResourceId id) {
    if (!dump_at() || !id) return;
    std::lock_guard lock(mutex());
    auto& t = targets();
    for (auto it = t.begin(); it != t.end(); ++it)
        if (*it == id) {
            t.erase(it);
            break;
        }
    t.push_back(id);
    if (t.size() > 32) t.erase(t.begin());
}

inline std::vector<ResourceId> snapshot() {
    std::lock_guard lock(mutex());
    return targets();
}

// The GPU address arena's CPU mapping, for diagnostics that read descriptors the shaders will load (set by the
// device when it creates the arena).
inline std::uint8_t* arena_mapped = nullptr;
inline std::uint64_t arena_size = 0;

// Logs a draw's user data and, for each dword that is an arena VA, the 8 descriptor words there (a T# / V#).
inline void log_user_data(const char* what, const std::uint32_t* s, std::size_t n) {
    std::fprintf(stderr, "[rt] descriptors of %s: user data", what);
    for (std::size_t i = 0; i < n && i < 16; ++i) std::fprintf(stderr, " %08x", s[i]);
    std::fprintf(stderr, "\n");
    if (!arena_mapped) return;
    for (std::size_t i = 0; i < n && i < 16; ++i) {
        const std::uint64_t va = s[i];
        if (va < 0x10000 || va + 64 > arena_size || (va & 3)) continue;
        const auto* w = reinterpret_cast<const std::uint32_t*>(arena_mapped + va);
        std::fprintf(stderr, "[rt]   s%zu -> arena %#llx: %08x %08x %08x %08x %08x %08x %08x %08x\n", i,
                     static_cast<unsigned long long>(va), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
    }
}

}  // namespace leonida::debug_targets
