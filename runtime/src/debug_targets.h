#pragma once

// LEONIDA_DUMP_TARGETS=<present number>: the colour render targets the title draws into, collected as draws are
// recorded (d3d12_com.cpp record_draw), written out as PPM images by the presenter at that present
// (VulkanPresenter::dump_targets), into LEONIDA_CAPTURE's folder. A diagnostic: it shows what each target holds
// when the presented frame is black, without guessing from side effects.

#include "command_list.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
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
// Reads `bytes` of the title's own memory at `va` (its CPU address) when it is committed and readable; false otherwise
// (set by the device: the check needs VirtualQuery).
inline bool (*read_title)(std::uint64_t va, void* out, std::size_t bytes) = nullptr;

// Reads `bytes` at GPU VA `va` for a diagnostic: the arena below 4 GiB, the title's memory above. False when neither
// holds it.
inline bool read_gpu_va(std::uint64_t va, void* out, std::size_t bytes) {
    if (va < (1ull << 32)) {
        if (!arena_mapped || va + bytes > arena_size) return false;
        std::memcpy(out, arena_mapped + va, bytes);
        return true;
    }
    return read_title && read_title(va, out, bytes);
}

// A hang report's look through a descriptor table: the 0x80 bytes at `table` (a user-data pointer) and, for each
// 16 bytes there that read as a V# (base and num_records non-zero), the first 32 bytes of its buffer. A shader stuck in
// a loop whose count comes from a buffer (CS_DefragHeap_ProcessFreeRequests: s_ff1 of a count of 0 gave 2^32 rounds)
// shows the count it read.
inline void log_table(std::FILE* out, int user_data, std::uint32_t table) {
    std::uint32_t w[32];
    if (!read_gpu_va(table, w, sizeof(w))) return;
    for (int row = 0; row < 4; ++row)
        std::fprintf(out, "\n[rt] d3d12 DRED:          table %d +%02x: %08x %08x %08x %08x %08x %08x %08x %08x", user_data,
                     row * 32, w[8 * row], w[8 * row + 1], w[8 * row + 2], w[8 * row + 3], w[8 * row + 4],
                     w[8 * row + 5], w[8 * row + 6], w[8 * row + 7]);
    for (int q = 0; q < 8; ++q) {
        const std::uint32_t* v = w + 4 * q;
        const std::uint64_t base = v[0] | (std::uint64_t(v[1] & 0xFFFF) << 32);
        if (!base || !v[2]) continue;
        std::uint32_t b[8];
        if (!read_gpu_va(base, b, sizeof(b))) {
            std::fprintf(out, "\n[rt] d3d12 DRED:            V# at +%02x: base %#llx stride %u records %u: not readable", q * 16,
                         static_cast<unsigned long long>(base), (v[1] >> 16) & 0x3FFF, v[2]);
            continue;
        }
        std::fprintf(out, "\n[rt] d3d12 DRED:            V# at +%02x: base %#llx stride %u records %u: %08x %08x %08x %08x %08x "
                          "%08x %08x %08x",
                     q * 16, static_cast<unsigned long long>(base), (v[1] >> 16) & 0x3FFF, v[2], b[0], b[1], b[2], b[3],
                     b[4], b[5], b[6], b[7]);
    }
}

// Logs a draw's user data and, for each dword that is an arena VA, the 8 descriptor words there (a T# / V#).
inline void log_user_data(const char* what, const std::uint32_t* s, std::size_t n) {
    std::fprintf(stderr, "[rt] descriptors of %s: user data", what);
    for (std::size_t i = 0; i < n && i < 16; ++i) std::fprintf(stderr, " %08x", s[i]);
    std::fprintf(stderr, "\n");
    if (!arena_mapped) return;
    for (std::size_t i = 0; i < n && i < 16; ++i) {
        const std::uint64_t va = s[i];
        if (va < 0x10000 || va + 128 > arena_size || (va & 3)) continue;
        const auto* w = reinterpret_cast<const std::uint32_t*>(arena_mapped + va);
        for (int row = 0; row < 4; ++row)
            std::fprintf(stderr, "[rt]   s%zu -> arena %#llx+%02x: %08x %08x %08x %08x %08x %08x %08x %08x\n", i,
                         static_cast<unsigned long long>(va), row * 32, w[8 * row], w[8 * row + 1], w[8 * row + 2],
                         w[8 * row + 3], w[8 * row + 4], w[8 * row + 5], w[8 * row + 6], w[8 * row + 7]);
    }
}

}  // namespace leonida::debug_targets
