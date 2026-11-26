#pragma once
// What the runtime is building now, for the dev overlay (dev_overlay.h). Always on: two relaxed atomics a build, and a
// short copy under a lock for the name.
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

namespace leonida::dev_status {

struct Builds {
    std::atomic<unsigned> active{0};
    std::atomic<std::uint64_t> done{0};

    std::string last() {
        std::lock_guard lock(mutex_);
        return name_;
    }
    void name(const char* text) {
        std::lock_guard lock(mutex_);
        const std::size_t n = std::min(std::strlen(text), sizeof name_ - 1);
        std::memcpy(name_, text, n);
        name_[n] = 0;
    }

private:
    std::mutex mutex_;
    char name_[96]{};
};
inline Builds& builds() {
    static Builds instance;
    return instance;
}

// The GPU's progress over all queues, in retirement tickets (VulkanContext): what was submitted, what has completed,
// and when the completed figure last moved (steady-clock nanoseconds). Relaxed atomics, written where the context
// already counts them. The title ends itself after 30 s with no GPU progress (its "GPU Loc Sig" watchdog), so the
// overlay shows the time since the last progress while work is outstanding.
struct Gpu {
    std::atomic<std::uint64_t> submitted{0}, completed{0};
    std::atomic<std::int64_t> progress_ns{0};
};
inline Gpu& gpu() {
    static Gpu instance;
    return instance;
}
inline void gpu_completed(std::uint64_t ticket, std::int64_t now_ns) {
    Gpu& g = gpu();
    std::uint64_t seen = g.completed.load(std::memory_order_relaxed);
    while (ticket > seen && !g.completed.compare_exchange_weak(seen, ticket, std::memory_order_relaxed)) {
    }
    if (ticket > seen) g.progress_ns.store(now_ns, std::memory_order_relaxed);
}

// Compute dispatches the replay left out because their pipeline was not ready (LEONIDA_ASYNC_PSO, LEONIDA_LAZY_COMPUTE,
// past the compute-wait budget): work the title asked for that never ran, whose outputs keep what they had.
inline std::atomic<std::uint64_t>& skipped_dispatches() {
    static std::atomic<std::uint64_t> count{0};
    return count;
}

// Reports the dev overlay prints when it sees a stall (the GPU or the title's main loop still for 5 s), installed by
// the layers that own the state: the console memory waits pending (d3d12_com.cpp). Null: nothing.
inline std::atomic<void (*)()> stall_report_waits{nullptr};

// One pipeline build, from construction to destruction.
struct Build {
    explicit Build(const char* name) {
        builds().name(name);
        builds().active.fetch_add(1, std::memory_order_relaxed);
    }
    ~Build() {
        builds().active.fetch_sub(1, std::memory_order_relaxed);
        builds().done.fetch_add(1, std::memory_order_relaxed);
    }
    Build(const Build&) = delete;
    Build& operator=(const Build&) = delete;
};

}  // namespace leonida::dev_status
