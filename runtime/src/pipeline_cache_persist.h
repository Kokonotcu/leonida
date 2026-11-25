// D3D-704: the runtime keeps the VkPipelineCache entries it learns in a session and offers them to the next one.
//
// Two files feed a device's cache on its first PSO: the `pipelines` build stage's pipeline_cache.bin (the seed, D3D-34)
// and the runtime's own pipeline_cache_runtime_<vendor>_<device>.bin, which device teardown writes. Each is taken only
// when its VkPipelineCacheHeaderVersionOne names this device's vendor, device and pipelineCacheUUID; when both are,
// the second is merged into the first (vkMergePipelineCaches). Teardown writes the cache's data when it grew since it
// was loaded, through a same-directory temporary file and a rename, so a failed save leaves the old file as it was.
// The vendor and device in the name keep two GPUs of one machine from replacing each other's file; a driver update
// changes the UUID, so the old file is ignored once and then replaced.
// A run that never reaches device teardown (killed by a debugger, crashed, timed out) would keep nothing, so the device
// also writes checkpoints while it lives (CheckpointWorker below): one worker thread saves the cache the same way when
// pipelines were created since the last checkpoint, at most once per LEONIDA_PIPELINE_CACHE_CHECKPOINT_S seconds.
// Portable: no Vulkan device and no Windows API here, so runtime/test/pipeline_cache_persist_test.cpp runs anywhere.
#pragma once

#include "cached_pso.h"

#include <vulkan/vulkan_core.h>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace leonida::pipeline_cache {

// Why a file's data was or was not given to vkCreatePipelineCache.
enum class Header {
    ok,
    truncated,     // shorter than VkPipelineCacheHeaderVersionOne
    bad_header,    // headerSize or headerVersion is not VK_PIPELINE_CACHE_HEADER_VERSION_ONE's
    other_device,  // another vendorID or deviceID
    other_driver,  // the same GPU, another pipelineCacheUUID (a driver update)
};

inline const char* describe(Header header) {
    switch (header) {
        case Header::ok: return "matches this device";
        case Header::truncated: return "is shorter than a Vulkan pipeline cache header";
        case Header::bad_header: return "has no Vulkan pipeline cache header (version one)";
        case Header::other_device: return "was made on another GPU";
        case Header::other_driver: return "was made by another driver version (pipelineCacheUUID)";
    }
    return "?";
}

inline Header check_header(const void* data, std::size_t size, const cached_pso::Identity& identity) {
    static_assert(sizeof(VkPipelineCacheHeaderVersionOne) == 32);
    VkPipelineCacheHeaderVersionOne header{};
    if (!data || size < sizeof(header)) return Header::truncated;
    // The header is little-endian, like the host. The driver interprets the opaque payload.
    std::memcpy(&header, data, sizeof(header));
    if (header.headerSize != sizeof(header) || header.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE)
        return Header::bad_header;
    if (header.vendorID != identity.vendor || header.deviceID != identity.device) return Header::other_device;
    if (std::memcmp(header.pipelineCacheUUID, identity.cache_uuid, VK_UUID_SIZE) != 0) return Header::other_driver;
    return Header::ok;
}

// The saved file's name for this device: pipeline_cache_runtime_1002_164e.bin.
inline std::string file_name(const cached_pso::Identity& identity) {
    char name[64];
    std::snprintf(name, sizeof(name), "pipeline_cache_runtime_%04x_%04x.bin", unsigned(identity.vendor),
                  unsigned(identity.device));
    return name;
}

enum class Saved {
    written,
    unchanged,  // the cache holds what it was loaded with: no new pipeline, nothing to write
    empty,      // the driver keeps nothing beyond the header
    failed,     // reading the cache or writing the file failed; an existing file is left as it was
};

struct SaveResult {
    Saved saved = Saved::failed;
    std::size_t bytes = 0;  // the cache's data size
    std::string error;      // what failed, for the log
};

// Teardown's decision and write. `get` is vkGetPipelineCacheData bound to the device and cache:
// VkResult(size_t* size, void* data). `snapshot` reads the whole cache through `get` (pipeline_cache_snapshot, which
// copes with VK_INCOMPLETE) and may throw. `write` publishes the bytes atomically and returns an error code
// (write_pipeline_cache_file). `loaded_bytes` is the cache's data size right after it was created: an in-memory cache
// only ever gains entries, so the same size means nothing was added.
template<class Get, class Snapshot, class Write>
SaveResult save(Get get, Snapshot snapshot, Write write, std::size_t loaded_bytes) {
    SaveResult out;
    try {
        std::size_t size = 0;
        if (const VkResult result = get(&size, nullptr); result != VK_SUCCESS) {
            out.error = "vkGetPipelineCacheData returned " + std::to_string(int(result));
            return out;
        }
        out.bytes = size;
        if (size <= sizeof(VkPipelineCacheHeaderVersionOne)) {
            out.saved = Saved::empty;
            return out;
        }
        if (size == loaded_bytes) {
            out.saved = Saved::unchanged;
            return out;
        }
        const std::vector<char> data = snapshot(get);
        out.bytes = data.size();
        if (const std::error_code error = write(data.data(), data.size())) {
            out.error = error.message();
            return out;
        }
        out.saved = Saved::written;
    } catch (const std::exception& error) {
        out.saved = Saved::failed;
        out.error = error.what();
    }
    return out;
}

// ---- Checkpoints while the device lives ----

// 180 s: each checkpoint has the driver serialize the whole cache and copies it (about 260 MB in the first loading
// screen, 2026-11-25: four 260 MB buffers in four minutes, while the PC was short of memory).
constexpr std::chrono::seconds default_checkpoint_interval{180};
// After the first checkpoint line, at most one line per this long.
constexpr std::chrono::seconds checkpoint_log_interval{300};

// LEONIDA_PIPELINE_CACHE_CHECKPOINT_S: whole seconds between checkpoints, 0 for none. Empty, signed, fractional or
// not a number (or more than nine digits): nullopt, and the caller uses default_checkpoint_interval.
inline std::optional<std::chrono::seconds> parse_checkpoint_seconds(std::string_view text) {
    if (text.empty() || text.size() > 9) return std::nullopt;
    long long seconds = 0;
    for (const char c : text) {
        if (c < '0' || c > '9') return std::nullopt;
        seconds = seconds * 10 + (c - '0');
    }
    return std::chrono::seconds(seconds);
}

// When a checkpoint is due: pipelines were created since the last one (a counter, so an idle device never asks the
// driver for its cache) and the interval has passed since the last one ran, or since the cache was loaded. Pure: the
// caller passes the clock and the counter, so pipeline_cache_persist_test drives it without waiting.
class CheckpointPolicy {
public:
    using Clock = std::chrono::steady_clock;

    CheckpointPolicy() = default;
    CheckpointPolicy(Clock::duration interval, Clock::time_point start) : interval_(interval), last_(start) {}

    bool enabled() const { return interval_ > Clock::duration::zero(); }
    // Pipelines were created after the last checkpoint that saved (or found nothing new).
    bool pending(std::uint64_t created) const { return created != saved_created_; }
    // The earliest time the next checkpoint may run.
    Clock::time_point next() const { return last_ + interval_; }
    bool due(std::uint64_t created, Clock::time_point now) const {
        return enabled() && pending(created) && now >= next();
    }
    // A checkpoint that read the counter as `created` finished at `now`. When it wrote the file or found the cache
    // unchanged, those pipelines count as saved; after a failed one they are tried again an interval later.
    void finished(std::uint64_t created, Clock::time_point now, bool saved) {
        last_ = now;
        if (saved) saved_created_ = created;
    }

private:
    Clock::duration interval_{};
    Clock::time_point last_{};
    std::uint64_t saved_created_ = 0;
};

// Which checkpoints get a log line: the first, then at most one per checkpoint_log_interval. The rest are counted, so
// the next line can say how many went unlogged.
class CheckpointLog {
public:
    using Clock = std::chrono::steady_clock;

    // True when the checkpoint at `now` is logged; `skipped` is then the number left out since the last line.
    bool take(Clock::time_point now, std::uint64_t& skipped) {
        if (logged_ && now - last_ < checkpoint_log_interval) {
            ++skipped_;
            return false;
        }
        skipped = skipped_;
        skipped_ = 0;
        logged_ = true;
        last_ = now;
        return true;
    }

private:
    bool logged_ = false;
    Clock::time_point last_{};
    std::uint64_t skipped_ = 0;
};

// The one thread that writes checkpoints. Pipeline creation calls created() (a short lock and a counter, never a
// write); the worker sleeps until a creation, then until the policy's next(), and calls `save` with no lock held.
// Only the worker calls `save`, so writes never overlap; stop() (and the destructor) joins it, after which the caller
// does the final save at teardown. `save` returns true when the cache is on disk or held nothing new.
class CheckpointWorker {
public:
    using Clock = CheckpointPolicy::Clock;

    CheckpointWorker(Clock::duration interval, std::function<bool()> save)
        : policy_(interval, Clock::now()), save_(std::move(save)) {
        if (policy_.enabled()) thread_ = std::thread([this] { run(); });
    }
    ~CheckpointWorker() { stop(); }
    CheckpointWorker(const CheckpointWorker&) = delete;
    CheckpointWorker& operator=(const CheckpointWorker&) = delete;

    bool enabled() const { return policy_.enabled(); }

    void created() {
        const std::lock_guard lock(mutex_);
        ++created_;
        if (idle_) wake_.notify_one();
    }

    // Waits for a checkpoint in progress; none starts afterwards.
    void stop() {
        {
            const std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

    // For tests: checkpoints that called `save`, and the ones that returned true.
    std::uint64_t runs() const {
        const std::lock_guard lock(mutex_);
        return runs_;
    }
    std::uint64_t saves() const {
        const std::lock_guard lock(mutex_);
        return saves_;
    }

private:
    void run() {
        std::unique_lock lock(mutex_);
        while (!stopping_) {
            if (!policy_.pending(created_)) {
                idle_ = true;
                wake_.wait(lock, [&] { return stopping_ || policy_.pending(created_); });
                idle_ = false;
                continue;
            }
            if (Clock::now() < policy_.next()) {
                wake_.wait_until(lock, policy_.next(), [&] { return stopping_; });
                continue;
            }
            const std::uint64_t created = created_;
            lock.unlock();
            const bool saved = save_();
            lock.lock();
            ++runs_;
            if (saved) ++saves_;
            policy_.finished(created, Clock::now(), saved);
        }
    }

    CheckpointPolicy policy_;
    std::function<bool()> save_;
    mutable std::mutex mutex_;
    std::condition_variable wake_;
    std::uint64_t created_ = 0;
    std::uint64_t runs_ = 0;
    std::uint64_t saves_ = 0;
    bool idle_ = false;
    bool stopping_ = false;
    std::thread thread_;  // last: started after every other member is ready
};

}  // namespace leonida::pipeline_cache
