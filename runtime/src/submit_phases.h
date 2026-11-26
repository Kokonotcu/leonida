#pragma once
// Where a submission's CPU time goes, always on and cheap: each phase of VulkanQueue::execute_now adds its time here
// (two steady_clock reads and a relaxed atomic add), for the dev overlay (src/dev_overlay.h). The overlay's time is the
// process's age (GetProcessTimes), so it lines up with the boot.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace leonida::submit_phases {

enum Phase : unsigned {
    wait_room,  // VulkanQueue::wait_for_room: the queue at its cap waits for the GPU
    heaps,      // DeviceState::refresh_console_heaps
    shadows,    // slot-133 shadow buffers and title windows (ShadowPages::refresh)
    geometry,   // GeometryViews::refresh
    textures,   // Device::refresh_title_textures
    pipelines,  // no longer timed (the pre-pass that waited for pipelines is gone); kept for the column order
    replay,     // the rest of execute_now: preparing, recording and submitting the lists
    kCount
};

inline const char* name(unsigned p) {
    static const char* const names[kCount] = {"wait", "heaps", "shadows", "geometry", "textures", "pipes",
                                                    "replay"};
    return p < kCount ? names[p] : "?";
}

struct Totals {
    std::atomic<std::uint64_t> ns[kCount]{};
    std::atomic<std::uint64_t> all_submissions{0};
    std::atomic<std::uint64_t> present_serial{0};  // frames the presenter showed (VulkanPresenter), never reset
    // Threads whose innermost Scope is this phase now, and when the first of them entered it (steady_clock). Counted
    // per phase, since queues submit from several threads: one value would go stale when their scopes overlap.
    std::atomic<unsigned> active[kCount]{};
    std::atomic<std::int64_t> active_since[kCount]{};
};
inline Totals& totals() {
    static Totals t;
    return t;
}

inline std::int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Seconds since the process started.
inline double process_age() {
#ifdef _WIN32
    FILETIME created{}, exited{}, kernel{}, user{}, now{};
    if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) return 0;
    GetSystemTimeAsFileTime(&now);
    const auto ticks = [](const FILETIME& f) { return (std::uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime; };
    return double(ticks(now) - ticks(created)) * 1e-7;
#else
    return 0;
#endif
}

inline void add(Phase p, std::uint64_t ns) {
    Totals& t = totals();
    t.ns[p].fetch_add(ns, std::memory_order_relaxed);
}

// Times its own lifetime into a phase.
struct Scope {
    Phase phase;
    std::int64_t start = now_ns();
    int outer;  // the phase of this thread's Scope around this one, -1 for none
    explicit Scope(Phase p) : phase(p), outer(innermost()) {
        if (outer >= 0) leave(outer);
        enter(int(p), start);
        innermost() = int(p);
    }
    ~Scope() {
        const std::int64_t end = now_ns();
        add(phase, std::uint64_t(end - start));
        leave(int(phase));
        if (outer >= 0) enter(outer, end);
        innermost() = outer;
    }
    static int& innermost() {
        thread_local int phase_of_this_thread = -1;
        return phase_of_this_thread;
    }
    static void enter(int p, std::int64_t at) {
        Totals& t = totals();
        if (t.active[p].fetch_add(1, std::memory_order_relaxed) == 0) t.active_since[p].store(at, std::memory_order_relaxed);
    }
    static void leave(int p) { totals().active[p].fetch_sub(1, std::memory_order_relaxed); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
};

// The phase that some thread has been in longest, -1 when none is in one.
struct Running {
    int phase = -1;
    std::int64_t since = 0;
};
inline Running running() {
    Totals& t = totals();
    Running r;
    for (unsigned p = 0; p < kCount; ++p) {
        if (!t.active[p].load(std::memory_order_relaxed)) continue;
        const std::int64_t since = t.active_since[p].load(std::memory_order_relaxed);
        if (r.phase < 0 || since < r.since) r = {int(p), since};
    }
    return r;
}

// After each submission: counts it.
inline void submitted() { totals().all_submissions.fetch_add(1, std::memory_order_relaxed); }

}  // namespace leonida::submit_phases
