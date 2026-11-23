// The VAs of buffers on a device with no GPU address arena (src/synthetic_va.h) without a GPU.
#include "../src/synthetic_va.h"

#include <cstdint>
#include <cstdio>
#include <set>
#include <thread>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {
using leonida::SyntheticVa;
constexpr std::uint64_t kKiB = 1024;

// A refused request leaves the space it asked for: the next one that fits still gets it.
int refusal_keeps_space() {
    SyntheticVa va(SyntheticVa::kAlignment, SyntheticVa::kAlignment + 192 * kKiB);
    std::uint64_t a = 0, b = 0, c = 0;
    EXPECT(va.allocate(100 * kKiB, a));              // 128 KiB of the 192
    EXPECT(!va.allocate(64 * kKiB + 1, b));          // would need 128 KiB: refused
    EXPECT(va.allocate(64 * kKiB, c));               // the last 64 KiB is still there
    EXPECT(c == a + 128 * kKiB);
    EXPECT(!va.allocate(1, b));                      // now it is full
    return 0;
}

int huge_request_is_refused() {
    SyntheticVa va;
    std::uint64_t a = 0, b = 0;
    EXPECT(!va.allocate(~std::uint64_t(0), a));
    EXPECT(!va.allocate(SyntheticVa::kLimit, a));
    EXPECT(va.allocate(64 * kKiB, b));
    EXPECT(b == SyntheticVa::kFirst);
    return 0;
}

// Every address is 64 KiB aligned, ranges do not overlap, and every byte of a range is below 2^47 (a V# keeps 48 bits).
int addresses_are_aligned_disjoint_and_in_range() {
    SyntheticVa va;
    std::uint64_t previous_end = 0;
    for (const std::uint64_t bytes : {std::uint64_t(1), 1 * kKiB, 64 * kKiB, 64 * kKiB + 1, 96 * kKiB, 3 * 64 * kKiB}) {
        std::uint64_t address = 0;
        EXPECT(va.allocate(bytes, address));
        EXPECT(address != 0 && address % SyntheticVa::kAlignment == 0);
        EXPECT(address >= previous_end);
        EXPECT(address + bytes <= SyntheticVa::kLimit);
        previous_end = address + bytes;
    }
    return 0;
}

int threads_get_distinct_addresses() {
    SyntheticVa va;
    constexpr int kThreads = 8, kEach = 2000;
    std::vector<std::vector<std::uint64_t>> got(kThreads);
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            for (int i = 0; i < kEach; ++i) {
                std::uint64_t address = 0;
                if (va.allocate(1 * kKiB, address)) got[t].push_back(address);
            }
        });
    for (auto& thread : threads) thread.join();
    std::set<std::uint64_t> all;
    for (const auto& list : got) all.insert(list.begin(), list.end());
    EXPECT(all.size() == std::size_t(kThreads) * kEach);
    return 0;
}
}  // namespace

int main() {
    if (const int r = refusal_keeps_space()) return r;
    if (const int r = huge_request_is_refused()) return r;
    if (const int r = addresses_are_aligned_disjoint_and_in_range()) return r;
    if (const int r = threads_get_distinct_addresses()) return r;
    std::printf("synthetic_va: ok\n");
    return 0;
}
