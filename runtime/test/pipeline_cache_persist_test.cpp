// D3D-704 without a GPU: which saved pipeline caches a device takes (src/pipeline_cache_persist.h), and the teardown
// write: nothing for an empty or unchanged cache, the whole cache when it grew, the previous file kept when the write
// fails. The atomic writer and the VK_INCOMPLETE-safe reader are the ones the runtime uses.
// Then the checkpoints while a device lives: LEONIDA_PIPELINE_CACHE_CHECKPOINT_S's parsing, the rate limit and the
// creation counter on an injected clock, the log limiter, and the worker thread on a short interval (nothing without
// a creation, one save at a time, creation not held up by a slow save, a failed save retried, stop() waits).
#include "../src/pipeline_cache_persist.h"
#include "pipeline_cache_file.h"
#include "pipeline_cache_snapshot.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using leonida::cached_pso::Identity;
using leonida::pipeline_cache::Header;
using leonida::pipeline_cache::Saved;

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace {

Identity device_identity() {
    Identity identity;
    identity.vendor = 0x1002;
    identity.device = 0x164e;
    identity.driver = 0x800000;
    for (int i = 0; i < 16; ++i) identity.cache_uuid[i] = std::uint8_t(0xa0 + i);
    return identity;
}

// A cache's data as a driver hands it out: VkPipelineCacheHeaderVersionOne, then `payload` opaque bytes.
std::vector<char> cache_data(const Identity& identity, std::size_t payload, char fill = 'p') {
    VkPipelineCacheHeaderVersionOne header{};
    header.headerSize = sizeof(header);
    header.headerVersion = VK_PIPELINE_CACHE_HEADER_VERSION_ONE;
    header.vendorID = identity.vendor;
    header.deviceID = identity.device;
    std::memcpy(header.pipelineCacheUUID, identity.cache_uuid, VK_UUID_SIZE);
    std::vector<char> out(sizeof(header) + payload, fill);
    std::memcpy(out.data(), &header, sizeof(header));
    return out;
}

Header check(const std::vector<char>& data, const Identity& identity) {
    return leonida::pipeline_cache::check_header(data.data(), data.size(), identity);
}

int test_header() {
    const Identity identity = device_identity();
    const auto good = cache_data(identity, 100);
    CHECK(check(good, identity) == Header::ok);
    CHECK(check(cache_data(identity, 0), identity) == Header::ok);  // a header alone is a valid, empty cache

    Identity other = identity;
    other.cache_uuid[15] ^= 1;
    CHECK(check(good, other) == Header::other_driver);
    other = identity;
    other.vendor = 0x10de;
    CHECK(check(good, other) == Header::other_device);
    other = identity;
    other.device = 0x744c;
    CHECK(check(good, other) == Header::other_device);
    other = identity;
    other.driver ^= 1;  // driverVersion is not in the header; pipelineCacheUUID speaks for the driver
    CHECK(check(good, other) == Header::ok);

    CHECK(check(std::vector<char>(good.begin(), good.begin() + 31), identity) == Header::truncated);
    CHECK(check({}, identity) == Header::truncated);
    CHECK(leonida::pipeline_cache::check_header(nullptr, 64, identity) == Header::truncated);

    auto changed = good;
    changed[0] = 33;  // headerSize
    CHECK(check(changed, identity) == Header::bad_header);
    changed = good;
    changed[4] = 2;  // headerVersion
    CHECK(check(changed, identity) == Header::bad_header);
    CHECK(check(std::vector<char>(200, '\x5a'), identity) == Header::bad_header);
    CHECK(check(std::vector<char>(200, '\0'), identity) == Header::bad_header);

    for (const Header h : {Header::ok, Header::truncated, Header::bad_header, Header::other_device,
                           Header::other_driver})
        CHECK(std::strlen(leonida::pipeline_cache::describe(h)) > 0);
    CHECK(leonida::pipeline_cache::file_name(identity) == "pipeline_cache_runtime_1002_164e.bin");
    other.vendor = 0x10005;  // Mesa's VK_VENDOR_ID_MESA (llvmpipe)
    other.device = 0;
    CHECK(leonida::pipeline_cache::file_name(other) == "pipeline_cache_runtime_10005_0000.bin");
    return 0;
}

// vkGetPipelineCacheData over a vector, as a driver answers it: the size with a null pointer, VK_INCOMPLETE and a
// prefix when the buffer is short.
struct FakeCache {
    std::vector<char> data;
    VkResult fail = VK_SUCCESS;
    int calls = 0;
    VkResult operator()(std::size_t* size, void* out) {
        ++calls;
        if (fail != VK_SUCCESS) return fail;
        if (!out) {
            *size = data.size();
            return VK_SUCCESS;
        }
        const std::size_t n = (std::min)(*size, data.size());
        std::memcpy(out, data.data(), n);
        *size = n;
        return n < data.size() ? VK_INCOMPLETE : VK_SUCCESS;
    }
};

std::string contents(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), {}};
}

int test_save(const fs::path& dir) {
    const Identity identity = device_identity();
    const auto path = dir / leonida::pipeline_cache::file_name(identity);
    FakeCache cache;
    const auto get = [&](std::size_t* size, void* out) { return cache(size, out); };
    const auto snapshot = [](const auto& read) { return pipeline_cache_snapshot(read); };
    int writes = 0;
    const auto write = [&](const char* data, std::size_t size) {
        ++writes;
        return write_pipeline_cache_file(path, data, size);
    };

    // A driver that keeps nothing: no file.
    cache.data = cache_data(identity, 0);
    auto result = leonida::pipeline_cache::save(get, snapshot, write, 0);
    CHECK(result.saved == Saved::empty && writes == 0 && !fs::exists(path));

    // A cache that grew: the whole of it, and the file is one the next device takes.
    cache.data = cache_data(identity, 4000, 'a');
    result = leonida::pipeline_cache::save(get, snapshot, write, 32);
    CHECK(result.saved == Saved::written && result.bytes == cache.data.size() && writes == 1);
    const std::string first(cache.data.begin(), cache.data.end());
    CHECK(contents(path) == first);
    const std::string saved = contents(path);
    CHECK(leonida::pipeline_cache::check_header(saved.data(), saved.size(), identity) == Header::ok);

    // Loaded with that and nothing added: the file is left alone.
    result = leonida::pipeline_cache::save(get, snapshot, write, cache.data.size());
    CHECK(result.saved == Saved::unchanged && writes == 1);

    // The driver cannot be asked, or the read fails: nothing is written.
    cache.fail = VK_ERROR_OUT_OF_HOST_MEMORY;
    result = leonida::pipeline_cache::save(get, snapshot, write, 32);
    CHECK(result.saved == Saved::failed && !result.error.empty() && writes == 1);
    cache.fail = VK_SUCCESS;
    const auto throwing = [](const auto&) -> std::vector<char> { throw std::runtime_error("read failed"); };
    result = leonida::pipeline_cache::save(get, throwing, write, 32);
    CHECK(result.saved == Saved::failed && result.error == "read failed" && writes == 1);
    CHECK(contents(path) == first);

    // A write that fails keeps the previous file whole, and leaves no staging file behind.
    cache.data = cache_data(identity, 9000, 'b');
#ifdef _WIN32
    // A reader that did not share DELETE access prevents the replacement.
    HANDLE held = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(held != INVALID_HANDLE_VALUE);
    result = leonida::pipeline_cache::save(get, snapshot, write, 32);
    CloseHandle(held);
    CHECK(result.saved == Saved::failed && !result.error.empty());
#else
    if (geteuid() != 0) {  // root writes into a read-only directory anyway
        CHECK(chmod(dir.c_str(), 0555) == 0);
        result = leonida::pipeline_cache::save(get, snapshot, write, 32);
        CHECK(chmod(dir.c_str(), 0755) == 0);
        CHECK(result.saved == Saved::failed && !result.error.empty());
    } else {
        std::printf("SKIP: running as root, the read-only directory case is not checked\n");
    }
#endif
    CHECK(contents(path) == first);
    CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 1);

    // Then a write that works replaces it.
    result = leonida::pipeline_cache::save(get, snapshot, write, 32);
    CHECK(result.saved == Saved::written);
    CHECK(contents(path) == std::string(cache.data.begin(), cache.data.end()));
    CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 1);
    return 0;
}

using namespace std::chrono_literals;
using leonida::pipeline_cache::CheckpointLog;
using leonida::pipeline_cache::CheckpointPolicy;
using leonida::pipeline_cache::CheckpointWorker;

int test_checkpoint_interval() {
    using leonida::pipeline_cache::parse_checkpoint_seconds;
    CHECK(parse_checkpoint_seconds("60") == 60s);
    CHECK(parse_checkpoint_seconds("0") == 0s);
    CHECK(parse_checkpoint_seconds("007") == 7s);
    CHECK(parse_checkpoint_seconds("999999999") == 999999999s);
    for (const char* bad : {"", "-1", "+5", "1.5", "5s", " 5", "abc", "1234567890"})
        CHECK(!parse_checkpoint_seconds(bad));
    CHECK(leonida::pipeline_cache::default_checkpoint_interval == 60s);
    return 0;
}

// The rate limit and the creation counter, on a clock the test sets.
int test_checkpoint_policy() {
    const CheckpointPolicy::Clock::time_point t0{};
    CheckpointPolicy policy(60s, t0);
    CHECK(policy.enabled());
    // Nothing created: never due, however long the device lives.
    CHECK(!policy.pending(0));
    CHECK(!policy.due(0, t0 + 3600s));
    // A creation: due once the interval since the load has passed, not before.
    CHECK(policy.pending(1));
    CHECK(!policy.due(1, t0 + 59s));
    CHECK(policy.due(1, t0 + 60s));
    CHECK(policy.next() == t0 + 60s);
    // Saved with 3 created; nothing created since: not due again.
    policy.finished(3, t0 + 61s, true);
    CHECK(!policy.pending(3));
    CHECK(!policy.due(3, t0 + 3600s));
    // One more creation right after: it waits for the interval since the last checkpoint.
    CHECK(!policy.due(4, t0 + 62s));
    CHECK(!policy.due(4, t0 + 120s));
    CHECK(policy.due(4, t0 + 121s));
    // A failed checkpoint: the creations stay pending and are tried again an interval later.
    policy.finished(4, t0 + 121s, false);
    CHECK(policy.pending(4));
    CHECK(!policy.due(4, t0 + 180s));
    CHECK(policy.due(4, t0 + 181s));
    policy.finished(4, t0 + 181s, true);
    CHECK(!policy.pending(4));
    // A creation counted after a checkpoint read the counter stays pending.
    policy.finished(5, t0 + 300s, true);
    CHECK(!policy.pending(5));
    CHECK(policy.pending(6));
    // LEONIDA_PIPELINE_CACHE_CHECKPOINT_S=0: never due.
    const CheckpointPolicy off(0s, t0);
    CHECK(!off.enabled());
    CHECK(!off.due(100, t0 + 3600s));
    return 0;
}

int test_checkpoint_log() {
    const CheckpointLog::Clock::time_point t0{};
    CheckpointLog log;
    std::uint64_t skipped = 99;
    CHECK(log.take(t0, skipped) && skipped == 0);  // the first checkpoint is logged
    CHECK(!log.take(t0 + 60s, skipped));
    CHECK(!log.take(t0 + 299s, skipped));
    CHECK(log.take(t0 + 300s, skipped) && skipped == 2);  // then one line per 300 s, with the count left out
    CHECK(!log.take(t0 + 360s, skipped));
    CHECK(log.take(t0 + 1000s, skipped) && skipped == 1);
    return 0;
}

template<class Predicate>
bool wait_for(Predicate predicate, std::chrono::milliseconds limit = 5000ms) {
    const auto end = std::chrono::steady_clock::now() + limit;
    while (!predicate()) {
        if (std::chrono::steady_clock::now() > end) return false;
        std::this_thread::sleep_for(5ms);
    }
    return true;
}

// The worker thread, on the real clock and short intervals.
int test_checkpoint_worker(const fs::path& dir) {
    // Interval 0: no thread; creations are counted for nothing and nothing is saved.
    {
        int calls = 0;
        CheckpointWorker worker(0s, [&] { ++calls; return true; });
        CHECK(!worker.enabled());
        worker.created();
        worker.stop();
        CHECK(calls == 0 && worker.runs() == 0);
    }
    // No creation: no checkpoint. Creations: one checkpoint, then none while nothing new is created.
    {
        std::atomic<int> calls{0};
        CheckpointWorker worker(50ms, [&] { ++calls; return true; });
        CHECK(worker.enabled());
        std::this_thread::sleep_for(200ms);
        CHECK(calls == 0 && worker.runs() == 0);
        worker.created();
        worker.created();
        worker.created();
        CHECK(wait_for([&] { return worker.runs() == 1; }));
        std::this_thread::sleep_for(200ms);
        CHECK(calls == 1 && worker.runs() == 1 && worker.saves() == 1);
        worker.created();
        CHECK(wait_for([&] { return worker.runs() == 2; }));
    }
    // A steady stream of creations: at most one checkpoint per interval.
    {
        std::atomic<int> calls{0};
        CheckpointWorker worker(300ms, [&] { ++calls; return true; });
        const auto start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < 1000ms) {
            worker.created();
            std::this_thread::sleep_for(5ms);
        }
        worker.stop();
        const auto elapsed = std::chrono::steady_clock::now() - start;
        const int allowed = int(elapsed / 300ms);  // the first no earlier than 300 ms in, then one per 300 ms
        std::printf("checkpoints over %lld ms of creations at a 300 ms interval: %d (at most %d)\n",
                    static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()),
                    calls.load(), allowed);
        CHECK(calls >= 1 && calls <= allowed);
    }
    // A slow save: creations are not held up by it, and saves never overlap.
    {
        std::atomic<int> in_flight{0}, most{0}, calls{0};
        CheckpointWorker worker(20ms, [&] {
            const int now = ++in_flight;
            most = (std::max)(most.load(), now);
            ++calls;
            std::this_thread::sleep_for(300ms);
            --in_flight;
            return true;
        });
        worker.created();
        CHECK(wait_for([&] { return in_flight == 1; }));
        const auto before = std::chrono::steady_clock::now();
        for (int i = 0; i < 100; ++i) worker.created();
        const auto took = std::chrono::steady_clock::now() - before;
        CHECK(in_flight == 1);  // still in the first save
        CHECK(took < 100ms);    // 100 creations did not wait for it
        CHECK(wait_for([&] { return calls == 2 && in_flight == 0; }));  // the creations during it: one more checkpoint
        worker.stop();
        CHECK(most == 1 && calls == 2);
    }
    // A failed save is tried again an interval later; stop() waits for a save in progress, and none starts after it.
    {
        std::atomic<int> calls{0};
        std::atomic<bool> finished{false};
        CheckpointWorker worker(50ms, [&] {
            const int n = ++calls;
            if (n == 2) {
                std::this_thread::sleep_for(200ms);
                finished = true;
            }
            return n != 1;
        });
        worker.created();
        CHECK(wait_for([&] { return calls == 2; }));
        CHECK(worker.runs() == 1 && worker.saves() == 0);  // the failed one; the second is in progress
        worker.stop();
        CHECK(finished && worker.runs() == 2 && worker.saves() == 1);
        worker.created();
        std::this_thread::sleep_for(150ms);
        CHECK(calls == 2);
    }
    // The device's use: the worker writes the file through save() while the device lives, and a later checkpoint over
    // a cache that did not grow (a creation that hit the cache) writes nothing; neither does teardown's save then.
    {
        const Identity identity = device_identity();
        const auto path = dir / leonida::pipeline_cache::file_name(identity);
        FakeCache cache;
        cache.data = cache_data(identity, 0);
        std::size_t saved_bytes = cache.data.size();  // the size at the load
        std::atomic<int> writes{0};
        std::mutex cache_mutex;  // the fake cache is a vector; a driver's cache synchronizes itself
        const auto checkpoint = [&] {
            const std::lock_guard lock(cache_mutex);
            const auto get = [&](std::size_t* size, void* out) { return cache(size, out); };
            const auto snapshot = [](const auto& read) { return pipeline_cache_snapshot(read); };
            const auto write = [&](const char* data, std::size_t size) {
                ++writes;
                return write_pipeline_cache_file(path, data, size);
            };
            const auto result = leonida::pipeline_cache::save(get, snapshot, write, saved_bytes);
            if (result.saved == Saved::written) saved_bytes = result.bytes;
            return result.saved != Saved::failed;
        };
        CheckpointWorker worker(50ms, checkpoint);
        {
            const std::lock_guard lock(cache_mutex);
            cache.data = cache_data(identity, 2000, 'c');  // a pipeline was created
        }
        worker.created();
        CHECK(wait_for([&] { return worker.runs() == 1; }));
        CHECK(writes == 1 && fs::exists(path));
        const std::string bytes = contents(path);
        CHECK(bytes == std::string(cache.data.begin(), cache.data.end()));
        CHECK(leonida::pipeline_cache::check_header(bytes.data(), bytes.size(), identity) == Header::ok);
        worker.created();  // the same pipeline again: the cache does not grow
        CHECK(wait_for([&] { return worker.runs() == 2; }));
        CHECK(writes == 1 && worker.saves() == 2);
        worker.stop();
        CHECK(checkpoint());  // teardown's final save
        CHECK(writes == 1);
        std::error_code ignored;
        fs::remove(path, ignored);
    }
    return 0;
}

}  // namespace

int main() {
    if (const int failed = test_header()) return failed;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = fs::temp_directory_path() / ("leonida_pipeline_cache_persist_" + std::to_string(stamp));
    CHECK(fs::create_directory(dir));
    int result = test_save(dir);
    if (!result) result = test_checkpoint_interval();
    if (!result) result = test_checkpoint_policy();
    if (!result) result = test_checkpoint_log();
    if (!result) {
        std::error_code ignored;
        for (const auto& entry : fs::directory_iterator(dir)) fs::remove(entry.path(), ignored);
        result = test_checkpoint_worker(dir);
    }
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    if (result == 0)
        std::printf("pipeline cache persistence: header checks, teardown writes and checkpoints behave\n");
    return result;
}
