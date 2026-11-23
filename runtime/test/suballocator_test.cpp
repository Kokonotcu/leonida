// leonida::Suballocator (src/vulkan_suballocator.h), the bookkeeping behind the D3D12 layer's committed resources'
// memory: block sizes, the dedicated threshold, alignment, best fit, coalescing, block growth and release, pool
// separation, mapped pointers, and a multi-threaded smoke test. No Vulkan, no GPU.
#include "../src/vulkan_suballocator.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                                 \
    do {                                                                                 \
        if (!(condition)) {                                                              \
            std::fprintf(stderr, "suballocator_test:%d: %s\n", __LINE__, #condition);    \
            ++failures;                                                                  \
        }                                                                                \
    } while (0)

using leonida::Suballocator;
using leonida::SuballocPool;
constexpr std::uint64_t MiB = leonida::kSuballocMiB;
constexpr std::uint64_t GiB = 1024 * MiB;

// What DeviceState::allocate_resource does without the Vulkan calls: existing blocks first, then a new block.
struct Harness {
    Suballocator s;
    std::atomic<std::uint64_t> next_memory{1};
    std::atomic<int> blocks_made{0};
    leonida::SuballocBlockSizes sizes = leonida::suballoc_block_sizes(16 * GiB);

    Suballocator::Allocation get(SuballocPool pool, std::uint64_t size, std::uint64_t alignment, void* base = nullptr) {
        Suballocator::Allocation a;
        if (s.allocate(pool, size, alignment, a)) return a;
        const std::uint64_t block = s.next_block_size(pool, sizes, size);
        ++blocks_made;
        return s.add_block(pool, next_memory++, base, block, size);
    }
};

void block_sizes() {
    const auto big = leonida::suballoc_block_sizes(16 * GiB);
    CHECK(big.first == 64 * MiB && big.max == 256 * MiB);
    const auto two = leonida::suballoc_block_sizes(2 * GiB);
    CHECK(two.first == 64 * MiB && two.max == 128 * MiB);
    const auto bar = leonida::suballoc_block_sizes(256 * MiB);  // the BAR window without Resizable BAR
    CHECK(bar.first == 16 * MiB && bar.max == 16 * MiB);
    const auto nvidia_bar = leonida::suballoc_block_sizes(214 * MiB);
    CHECK(nvidia_bar.first == 8 * MiB && nvidia_bar.max == 8 * MiB);
    const auto tiny = leonida::suballoc_block_sizes(4 * MiB);
    CHECK(tiny.first == MiB && tiny.max == MiB);
    const auto capped = leonida::suballoc_block_sizes(16 * GiB, 100 * MiB);  // maxMemoryAllocationSize
    CHECK(capped.first == 64 * MiB && capped.max == 64 * MiB);
}

void dedicated_threshold() {
    const auto sizes = leonida::suballoc_block_sizes(16 * GiB);
    CHECK(!leonida::suballoc_dedicated(128 * MiB, sizes, false));
    CHECK(leonida::suballoc_dedicated(128 * MiB + 1, sizes, false));
    CHECK(leonida::suballoc_dedicated(4096, sizes, true));  // VkMemoryDedicatedRequirements asked for it
    const auto bar = leonida::suballoc_block_sizes(256 * MiB);
    CHECK(leonida::suballoc_dedicated(8 * MiB + 1, bar, false));
}

void alignment_and_padding() {
    Harness h;
    const SuballocPool pool{3, false, false, false};
    const auto a = h.get(pool, 100, 1);
    CHECK(a.offset == 0 && h.blocks_made == 1);
    const auto b = h.get(pool, 4096, 65536);
    CHECK(b.offset == 65536 && b.memory == a.memory);
    // The padding [100, 65536) stays free; a 1000-byte range at 256 fits it best.
    const auto c = h.get(pool, 1000, 256);
    CHECK(c.offset == 256);
    const auto ranges = h.s.free_ranges(pool);
    CHECK(ranges.size() == 3);
    if (ranges.size() == 3) {
        CHECK(ranges[0] == std::make_pair(std::uint64_t(100), std::uint64_t(156)));
        CHECK(ranges[1] == std::make_pair(std::uint64_t(1256), std::uint64_t(65536 - 1256)));
        CHECK(ranges[2].first == 65536 + 4096 && ranges[2].second == 64 * MiB - 65536 - 4096);
    }
    // Every offset honours its alignment, whatever the order (non-power-of-two too).
    std::mt19937 random(7);
    for (int i = 0; i < 500; ++i) {
        const std::uint64_t alignment = std::uint64_t(1) << (random() % 17);
        const auto r = h.get(pool, 1 + random() % 70000, alignment);
        CHECK(r.offset % alignment == 0);
    }
    const auto odd = h.get(pool, 10, 3000);
    CHECK(odd.offset % 3000 == 0);
}

void best_fit() {
    Harness h;
    const SuballocPool pool{0, true, false, false};
    // [hole 3000][used][hole 1000][used][hole 2000][used][rest]
    const auto h1 = h.get(pool, 3000, 1);
    const auto u1 = h.get(pool, 10, 1);
    const auto h2 = h.get(pool, 1000, 1);
    const auto u2 = h.get(pool, 10, 1);
    const auto h3 = h.get(pool, 2000, 1);
    const auto u3 = h.get(pool, 10, 1);
    CHECK(h.s.free(h1) == 0);
    CHECK(h.s.free(h2) == 0);
    CHECK(h.s.free(h3) == 0);
    const auto x = h.get(pool, 1500, 1);
    CHECK(x.offset == h3.offset);  // the 2000 hole, not the 3000 one or the rest of the block
    const auto y = h.get(pool, 900, 1);
    CHECK(y.offset == h2.offset);
    const auto z = h.get(pool, 2500, 1);
    CHECK(z.offset == h1.offset);
    const auto w = h.get(pool, 600, 1);  // no hole is big enough: the rest of the block
    CHECK(w.offset == u3.offset + 10);
    (void)u1;
    (void)u2;
}

void coalescing_and_spare() {
    Harness h;
    const SuballocPool pool{1, false, true, false};
    std::vector<Suballocator::Allocation> all;
    for (int i = 0; i < 64; ++i) all.push_back(h.get(pool, 4096 + i * 512, 256));
    CHECK(h.blocks_made == 1);
    std::shuffle(all.begin(), all.end(), std::mt19937(11));
    for (const auto& a : all) CHECK(h.s.free(a) == 0);  // the pool's only block is kept, empty
    const auto ranges = h.s.free_ranges(pool);
    CHECK(ranges.size() == 1 && ranges[0].first == 0 && ranges[0].second == 64 * MiB);
    const auto stats = h.s.stats();
    CHECK(stats.blocks == 1 && stats.empty_blocks == 1 && stats.allocations == 0 && stats.used_bytes == 0);
    // Reused, not a new block.
    const auto again = h.get(pool, 4096, 256);
    CHECK(again.offset == 0 && h.blocks_made == 1);
}

void growth_and_release() {
    Harness h;
    const SuballocPool pool{2, false, false, false};
    CHECK(h.s.next_block_size(pool, h.sizes, 4096) == 64 * MiB);
    const auto first = h.get(pool, 40 * MiB, 4096);
    CHECK(h.s.next_block_size(pool, h.sizes, 4096) == 128 * MiB);
    const auto second = h.get(pool, 40 * MiB, 4096);  // 24 MiB left in the first: a 128 MiB block
    CHECK(second.memory != first.memory && h.blocks_made == 2);
    CHECK(h.s.next_block_size(pool, h.sizes, 4096) == 256 * MiB);
    const auto third = h.get(pool, 100 * MiB, 4096);  // 88 MiB left in the second
    CHECK(third.memory != second.memory && h.blocks_made == 3);
    CHECK(h.s.next_block_size(pool, h.sizes, 4096) == 256 * MiB);  // capped
    CHECK(h.s.stats().block_bytes == (64 + 128 + 256) * MiB);
    // A pool with only small blocks still makes one that holds a large request.
    Harness small;
    small.sizes = {MiB, 256 * MiB};
    CHECK(small.s.next_block_size(pool, small.sizes, 100 * MiB) == 128 * MiB);
    // Emptying blocks: the first stays as the spare, the next ones are returned for vkFreeMemory.
    CHECK(h.s.free(first) == 0);
    CHECK(h.s.free(second) == second.memory);
    CHECK(h.s.free(third) == third.memory);
    const auto stats = h.s.stats();
    CHECK(stats.blocks == 1 && stats.empty_blocks == 1 && stats.block_bytes == 64 * MiB);
    const auto left = h.s.release_all();
    CHECK(left.size() == 1 && left[0] == first.memory);
    CHECK(h.s.stats().blocks == 0);
}

void pools_are_separate() {
    Harness h;
    const SuballocPool buffers{4, false, false, true};
    const SuballocPool images{4, true, false, false};
    const SuballocPool addressed{4, false, true, true};
    static std::uint8_t backing[8192];
    const auto b = h.get(buffers, 4096, 256, backing);
    Suballocator::Allocation a;
    CHECK(!h.s.allocate(images, 4096, 256, a));     // never next to a buffer: no bufferImageGranularity
    CHECK(!h.s.allocate(addressed, 4096, 256, a));  // device-address blocks are allocated with the flag
    const auto i = h.get(images, 4096, 256);
    CHECK(i.memory != b.memory && !i.mapped);
    // A mapped pool hands out base + offset.
    const auto b2 = h.get(buffers, 100, 1024);
    CHECK(b2.memory == b.memory && b2.mapped == backing + b2.offset && b.mapped == backing);
    const SuballocPool last{31, true, true, true};
    CHECK(last.index() == 255);
}

void threads() {
    Harness h;
    constexpr int kThreads = 8;
    constexpr int kSteps = 20000;
    std::vector<std::vector<Suballocator::Allocation>> kept(kThreads);
    std::vector<std::thread> workers;
    for (int t = 0; t < kThreads; ++t) {
        workers.emplace_back([&, t] {
            std::mt19937 random(1000 + t);
            auto& mine = kept[t];
            for (int step = 0; step < kSteps; ++step) {
                const SuballocPool pool{std::uint32_t(random() % 3), (random() & 1) != 0, false, false};
                if (!mine.empty() && random() % 2) {
                    const std::size_t which = random() % mine.size();
                    h.s.free(mine[which]);
                    mine[which] = mine.back();
                    mine.pop_back();
                } else {
                    mine.push_back(h.get(pool, 256 + random() % (256 * 1024), std::uint64_t(256) << (random() % 9)));
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    // No two live ranges of a block overlap.
    std::map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> by_memory;
    std::size_t live = 0;
    for (const auto& mine : kept)
        for (const auto& a : mine) {
            by_memory[a.memory].emplace_back(a.offset, a.offset + a.size);
            ++live;
        }
    for (auto& [memory, ranges] : by_memory) {
        std::sort(ranges.begin(), ranges.end());
        for (std::size_t i = 1; i < ranges.size(); ++i) CHECK(ranges[i - 1].second <= ranges[i].first);
    }
    CHECK(h.s.stats().allocations == live);
    for (const auto& mine : kept)
        for (const auto& a : mine) h.s.free(a);
    const auto stats = h.s.stats();
    CHECK(stats.allocations == 0 && stats.used_bytes == 0);
    CHECK(stats.blocks == stats.empty_blocks && stats.blocks <= 6);  // at most one spare per pool (3 types x 2)
    std::printf("suballocator_test: threads: %d blocks made for %d steps, %zu left as spares\n", h.blocks_made.load(),
                kThreads * kSteps, stats.blocks);
}

}  // namespace

int main() {
    block_sizes();
    dedicated_threshold();
    alignment_and_padding();
    best_fit();
    coalescing_and_spare();
    growth_and_release();
    pools_are_separate();
    threads();
    if (failures) {
        std::fprintf(stderr, "suballocator_test: %d failures\n", failures);
        return 1;
    }
    std::printf("suballocator_test: OK\n");
    return 0;
}
