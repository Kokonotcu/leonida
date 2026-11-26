#pragma once
// LEONIDA_DEBUG_RING=1: the run's memory history, for the crash and stall reports. Every write path of the runtime into
// the title's memory and every change of what the title's addresses map is appended to a fixed ring (2^18 entries of
// 40 bytes, about 10 MB, one atomic add and five stores an event, no lock, no output). Nothing is printed until a report
// asks: dump() prints the entries that touch an address or a physical page, newest last.
// Off (the default) nothing is recorded and record() is one relaxed load.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace leonida::debug_ring {

enum class Kind : std::uint32_t {
    none = 0,
    xmem_map,        // a: address (64 KiB), b: page number, c: section scratch + offset (the page's identity)
    xmem_unmap,      // a: address, b: page number it showed, c: identity
    xmem_alloc,      // a: page number, c: identity
    xmem_free,       // a: page number, c: identity
    cpu_write,       // a: address, b: bytes, c: caller (return address)
    value_write,     // a: title address, b: value, c: the page table entry of its page when the list was submitted
    value_wait,      // a: title address, b: reference value, c: comparison
    list_closed,     // a: list, b: type, c: the queue it ran on at its Close (0: held)
    list_executed,   // a: list, b: queue, c: number of lists in the call
    title_copy,      // a: destination, b: source, c: bytes (console list copies, slots 62 and 86)
};

inline const char* name(Kind k) {
    switch (k) {
        case Kind::xmem_map: return "xmem map";
        case Kind::xmem_unmap: return "xmem unmap";
        case Kind::xmem_alloc: return "xmem alloc";
        case Kind::xmem_free: return "xmem free";
        case Kind::cpu_write: return "cpu write";
        case Kind::value_write: return "value write";
        case Kind::value_wait: return "value wait";
        case Kind::list_closed: return "list closed";
        case Kind::list_executed: return "list executed";
        case Kind::title_copy: return "title copy";
        default: return "?";
    }
}

struct Entry {
    std::atomic<std::uint64_t> seq{0};  // 0: empty; else the event's number + 1, written last
    std::uint64_t ms = 0;               // milliseconds since the ring started
    std::uint64_t a = 0, b = 0, c = 0;
    std::uint32_t kind = 0, tid = 0;
};

constexpr std::size_t kEntries = std::size_t(1) << 18;

inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_DEBUG_RING");
        return v && v[0] == '1';
    }();
    return on;
}

struct Ring {
    std::atomic<std::uint64_t> next{0};
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    Entry* entries = nullptr;
};
inline Ring& ring() {
    static Ring* r = [] {
        auto* made = new Ring;
#ifdef _WIN32
        // Committed up front from the OS, not the C heap: the crash report reads it while a heap lock may be held.
        made->entries = static_cast<Entry*>(
            VirtualAlloc(nullptr, sizeof(Entry) * kEntries, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
#endif
        return made;
    }();
    return *r;
}

// The physical identity of the page behind a title address now (xmem installs it: section scratch + offset of the
// 64 KiB page, 0 when unmapped or when xmem's lock is busy; it never waits).
inline std::atomic<std::uint64_t (*)(std::uint64_t address)> identity_of{nullptr};
inline std::uint64_t identity_now(std::uint64_t address) {
    const auto fn = identity_of.load(std::memory_order_acquire);
    return fn ? fn(address) : 0;
}

inline void record(Kind kind, std::uint64_t a, std::uint64_t b = 0, std::uint64_t c = 0) {
    if (!enabled()) return;
    Ring& r = ring();
    if (!r.entries) return;
    const std::uint64_t n = r.next.fetch_add(1, std::memory_order_relaxed);
    Entry& e = r.entries[n & (kEntries - 1)];
    e.seq.store(0, std::memory_order_relaxed);
    e.ms = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - r.start).count());
    e.a = a;
    e.b = b;
    e.c = c;
    e.kind = static_cast<std::uint32_t>(kind);
#ifdef _WIN32
    e.tid = GetCurrentThreadId();
#endif
    e.seq.store(n + 1, std::memory_order_release);
}

// What a dump looks for: entries whose address range covers [address, address + span), or that name `page` (a page
// number) or `identity` (a page's section scratch address + offset, or a page table entry), each when non-zero.
struct Match {
    std::uint64_t address = 0, span = 0, page = 0, identity = 0;
};

inline bool matches(const Entry& e, const Match& m) {
    const auto k = static_cast<Kind>(e.kind);
    const auto overlaps = [&](std::uint64_t start, std::uint64_t bytes) {
        return m.address && bytes && start < m.address + std::max<std::uint64_t>(m.span, 1) && m.address < start + bytes;
    };
    switch (k) {
        case Kind::xmem_map:
        case Kind::xmem_unmap:
            return overlaps(e.a, 0x10000) || (m.page && e.b == m.page) ||
                   (m.identity && (e.c & ~0xFFFFull) == (m.identity & ~0xFFFFull));
        case Kind::xmem_alloc:
        case Kind::xmem_free: return (m.page && e.a == m.page) || (m.identity && e.c == (m.identity & ~0xFFFFull));
        case Kind::cpu_write: return overlaps(e.a, e.b);
        case Kind::title_copy: return overlaps(e.a, e.c) || overlaps(e.b, e.c);
        case Kind::value_write:
            return overlaps(e.a, 8) || (m.identity && e.c && (e.c & ~0xFFFFull) == (m.identity & ~0xFFFFull));
        case Kind::value_wait: return overlaps(e.a, 8);
        default: return false;
    }
}

// The newest `limit` entries matching `m`, printed oldest first with their age, under the tag `why`.
inline void dump(const char* why, const Match& m, unsigned limit = 48) {
    if (!enabled()) return;
    Ring& r = ring();
    if (!r.entries) return;
    const std::uint64_t end = r.next.load(std::memory_order_acquire);
    const std::uint64_t begin = end > kEntries ? end - kEntries : 0;
    static std::uint64_t picked[256];
    unsigned found = 0;
    for (std::uint64_t n = end; n-- > begin && found < limit && found < 256;) {
        const Entry& e = r.entries[n & (kEntries - 1)];
        if (e.seq.load(std::memory_order_acquire) != n + 1) continue;
        if (matches(e, m)) picked[found++] = n;
    }
    const std::uint64_t now = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - r.start).count());
    std::fprintf(stderr, "[rt] ring %s: %u entries (of %llu recorded) for address %#llx page %#llx identity %#llx\n",
                 why, found, static_cast<unsigned long long>(end), static_cast<unsigned long long>(m.address),
                 static_cast<unsigned long long>(m.page), static_cast<unsigned long long>(m.identity));
    for (unsigned i = found; i-- > 0;) {
        const Entry& e = r.entries[picked[i] & (kEntries - 1)];
        std::fprintf(stderr, "[rt] ring   -%llu ms %s a=%#llx b=%#llx c=%#llx tid %u\n",
                     static_cast<unsigned long long>(now - e.ms), name(static_cast<Kind>(e.kind)),
                     static_cast<unsigned long long>(e.a), static_cast<unsigned long long>(e.b),
                     static_cast<unsigned long long>(e.c), e.tid);
    }
}

}  // namespace leonida::debug_ring
