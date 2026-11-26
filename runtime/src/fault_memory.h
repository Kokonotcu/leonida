#pragma once

// What a device-fault or hang report (src/vulkan_breadcrumbs.h) reads behind a stuck dispatch's descriptor tables:
// the GPU address arena's CPU mapping and the title's own memory, set up by the device and the d3d12_xs layer.

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace leonida::fault_memory {

// The GPU address arena's CPU mapping (set by the device when it creates the arena).
inline std::uint8_t* arena_mapped = nullptr;
inline std::uint64_t arena_size = 0;
// Reads `bytes` of the title's own memory at `va` (its CPU address) when it is committed and readable; false otherwise
// (set by the device: the check needs VirtualQuery).
inline bool (*read_title)(std::uint64_t va, void* out, std::size_t bytes) = nullptr;
// For an arena address in a title-backed shader-visible heap's real copy: the title's own bytes there now (`now`) and
// what the last heap sync copied (`copied`), 32 bytes each; false when the address is in no such heap (set by the
// d3d12_xs layer). Tells a table the title wrote after its submission's sync from one the sync missed.
inline bool (*title_heap_of)(const void* real, void* now, void* copied) = nullptr;

// Reads `bytes` at GPU VA `va`: the arena below 4 GiB, the title's memory above. False when neither holds it.
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
    if (arena_mapped && title_heap_of && table < arena_size) {
        std::uint32_t now[8] = {}, copied[8] = {};
        if (title_heap_of(arena_mapped + table, now, copied)) {
            std::fprintf(out, "\n[rt] d3d12 DRED:          table %d in a title heap: title now %08x %08x %08x %08x %08x "
                              "%08x %08x %08x; last synced %08x %08x %08x %08x %08x %08x %08x %08x",
                         user_data, now[0], now[1], now[2], now[3], now[4], now[5], now[6], now[7], copied[0], copied[1],
                         copied[2], copied[3], copied[4], copied[5], copied[6], copied[7]);
        } else {
            std::fprintf(out, "\n[rt] d3d12 DRED:          table %d in no title heap", user_data);
        }
    }
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

}  // namespace leonida::fault_memory
