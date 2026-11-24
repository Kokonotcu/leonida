#pragma once

// LEONIDA_DUMP_TARGETS=<present number>: the colour render targets the title draws into, collected as draws are
// recorded (d3d12_com.cpp record_draw), written out as PPM images by the presenter at that present
// (VulkanPresenter::dump_targets), into LEONIDA_CAPTURE's folder. A diagnostic: it shows what each target holds
// when the presented frame is black, without guessing from side effects.

#include "command_list.h"

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

}  // namespace leonida::debug_targets
