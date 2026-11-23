// D3D-704 without a GPU: which saved pipeline caches a device takes (src/pipeline_cache_persist.h), and the teardown
// write: nothing for an empty or unchanged cache, the whole cache when it grew, the previous file kept when the write
// fails. The atomic writer and the VK_INCOMPLETE-safe reader are the ones the runtime uses.
#include "../src/pipeline_cache_persist.h"
#include "pipeline_cache_file.h"
#include "pipeline_cache_snapshot.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
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

}  // namespace

int main() {
    if (const int failed = test_header()) return failed;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = fs::temp_directory_path() / ("leonida_pipeline_cache_persist_" + std::to_string(stamp));
    CHECK(fs::create_directory(dir));
    const int result = test_save(dir);
    std::error_code ignored;
    fs::remove_all(dir, ignored);
    if (result == 0) std::printf("pipeline cache persistence: header checks and teardown writes behave\n");
    return result;
}
