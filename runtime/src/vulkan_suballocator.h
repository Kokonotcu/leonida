#pragma once

// Suballocation of VkDeviceMemory for the D3D12 layer's committed resources (textures, upload and readback buffers,
// default-heap buffers outside the arena, WriteToSubresource staging). Drivers on Windows report
// maxMemoryAllocationCount = 4096, and a game creates far more resources than that, so one vkAllocateMemory per
// resource runs out of allocations long before it runs out of memory (and each one is slow). This file is the
// bookkeeping only: which block and offset a resource gets. It makes no Vulkan call, so suballocator_test checks it
// without a GPU; the caller (DeviceState in shims/d3d12_vulkan_device.cpp) allocates, maps and frees the blocks.
//
// - A pool is one memory type, one resource class and one set of allocation flags (SuballocPool). Optimal-tiling
//   images and buffers never share a block, so bufferImageGranularity never applies and is not modelled.
//   Blocks of a pool that names device addresses are allocated with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, and
//   blocks of a mapped pool are mapped once, whole, when they are made: a range's host pointer is base + offset
//   (vkMapMemory on a range of memory that is already mapped is invalid).
// - Block sizes (suballoc_block_sizes): a pool's first block is 64 MiB and each new one doubles the largest it has,
//   up to 256 MiB, as VMA does for large heaps. 4096 blocks of 256 MiB are 1 TiB, so the allocation count cannot run
//   out, while a title that needs little memory holds 64 MiB per pool it uses. A small heap (the 256 MiB BAR window
//   without Resizable BAR) gets blocks of at most 1/16 of it, so one pool cannot take the whole heap.
// - A resource larger than half the largest block keeps its own allocation (as does one the driver asks a dedicated
//   allocation for, which the caller decides): a block that holds it would be mostly that resource anyway.
// - Within a pool the smallest free range that holds the aligned size is taken (best fit over every block of the
//   pool); the padding in front of an aligned range stays free, and a freed range merges with free neighbours.
// - A block whose last range is freed is released, except one empty block per pool, kept so a title that frees and
//   recreates a resource every frame does not allocate and free a block every frame.
// - Every public method takes the allocator's lock; resources are created and released on several threads.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace leonida {

inline constexpr std::uint64_t kSuballocMiB = std::uint64_t(1) << 20;

// The largest power of two at or below `value` (0 for 0).
inline std::uint64_t suballoc_floor_pow2(std::uint64_t value) {
    if (!value) return 0;
    std::uint64_t p = 1;
    while (p <= value / 2) p <<= 1;
    return p;
}

struct SuballocBlockSizes {
    std::uint64_t first = 0;  // a pool's first block
    std::uint64_t max = 0;    // blocks double up to this
    // A resource above this gets its own allocation.
    std::uint64_t dedicated_above() const { return max / 2; }
};

// Block sizes for a memory heap of `heap_size` bytes. `max_allocation_size` is maxMemoryAllocationSize (0: unknown).
inline SuballocBlockSizes suballoc_block_sizes(std::uint64_t heap_size, std::uint64_t max_allocation_size = 0) {
    constexpr std::uint64_t first = 64 * kSuballocMiB;
    constexpr std::uint64_t largest = 256 * kSuballocMiB;
    constexpr std::uint64_t smallest = kSuballocMiB;
    std::uint64_t max = std::min(largest, suballoc_floor_pow2(heap_size / 16));
    if (max_allocation_size) max = std::min(max, suballoc_floor_pow2(max_allocation_size));
    max = std::max(max, smallest);
    return {std::min(first, max), max};
}

// A resource that gets its own allocation rather than a range of a block.
inline bool suballoc_dedicated(std::uint64_t size, const SuballocBlockSizes& sizes, bool driver_asks) {
    return driver_asks || size > sizes.dedicated_above();
}

// Which blocks an allocation may come from. Memory types are below VK_MAX_MEMORY_TYPES (32).
struct SuballocPool {
    std::uint32_t memory_type = 0;
    bool image = false;           // optimal-tiling images; buffers otherwise
    bool device_address = false;  // VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
    bool mapped = false;          // persistently mapped blocks
    std::uint32_t index() const {
        return (memory_type & 31u) << 3 | std::uint32_t(image) << 2 | std::uint32_t(device_address) << 1 |
               std::uint32_t(mapped);
    }
};

class Suballocator {
public:
    struct Block;

    struct Allocation {
        Block* block = nullptr;
        std::uint64_t memory = 0;  // the block's VkDeviceMemory, as an integer
        std::uint64_t offset = 0;
        std::uint64_t size = 0;
        void* mapped = nullptr;    // the range's host pointer in a mapped pool
        explicit operator bool() const { return block != nullptr; }
    };

    struct Stats {
        std::size_t blocks = 0;
        std::size_t empty_blocks = 0;
        std::size_t allocations = 0;
        std::uint64_t block_bytes = 0;
        std::uint64_t used_bytes = 0;
    };

    Suballocator() = default;
    Suballocator(const Suballocator&) = delete;
    Suballocator& operator=(const Suballocator&) = delete;

    // From the blocks `pool` already has: the smallest free range that holds `size` bytes at `alignment`.
    bool allocate(SuballocPool pool, std::uint64_t size, std::uint64_t alignment, Allocation& out) {
        if (!size) return false;
        std::lock_guard lock(mutex_);
        Pool& p = pools_[pool.index()];
        const std::uint64_t align = alignment ? alignment : 1;
        for (auto it = p.by_size.lower_bound(size); it != p.by_size.end(); ++it) {
            Block* block = it->second.block;
            const std::uint64_t offset = it->second.offset;
            const std::uint64_t free_size = it->first;
            const std::uint64_t aligned = align_up(offset, align);
            if (aligned < offset || aligned - offset > free_size || free_size - (aligned - offset) < size) continue;
            out = carve(p, *block, offset, free_size, aligned, size);
            return true;
        }
        return false;
    }

    // The size of the next block of `pool` for an allocation of `size` bytes its blocks could not hold: double the
    // largest it has (the first size when it has none), at least `size`, at most `sizes.max` unless `size` is larger.
    std::uint64_t next_block_size(SuballocPool pool, const SuballocBlockSizes& sizes, std::uint64_t size) const {
        std::lock_guard lock(mutex_);
        const Pool& p = pools_[pool.index()];
        std::uint64_t largest = 0;
        for (const auto& block : p.blocks) largest = std::max(largest, block->size);
        std::uint64_t next = largest ? std::min(sizes.max, largest * 2) : sizes.first;
        next = std::max(next, sizes.first);
        while (next < size && next < sizes.max) next = std::min(sizes.max, next * 2);
        if (next < size) next = align_up(size, kSuballocMiB);
        return next;
    }

    // Adds a block the caller allocated (`memory`, mapped at `base` when the pool is mapped) and takes the allocation
    // from its start. `block_size` must hold `size`.
    Allocation add_block(SuballocPool pool, std::uint64_t memory, void* base, std::uint64_t block_size,
                         std::uint64_t size) {
        std::lock_guard lock(mutex_);
        Pool& p = pools_[pool.index()];
        auto owned = std::make_unique<Block>();
        Block& block = *owned;
        block.pool = &p;
        block.memory = memory;
        block.base = base;
        block.size = block_size;
        p.blocks.push_back(std::move(owned));
        insert_free(p, block, 0, block_size);
        return carve(p, block, 0, block_size, 0, size);
    }

    // Gives the range back. When that empties its block and the pool already keeps an empty block, the block is
    // removed and its memory returned, for the caller to free; 0 otherwise.
    std::uint64_t free(const Allocation& allocation) {
        if (!allocation.block) return 0;
        std::lock_guard lock(mutex_);
        Block& block = *allocation.block;
        Pool& p = *block.pool;
        block.used -= allocation.size;
        --block.allocations;
        std::uint64_t offset = allocation.offset;
        std::uint64_t size = allocation.size;
        auto next = block.free.lower_bound(offset);
        if (next != block.free.end() && next->first == offset + size) {
            size += next->second->first;
            p.by_size.erase(next->second);
            next = block.free.erase(next);
        }
        if (next != block.free.begin()) {
            const auto prev = std::prev(next);
            const std::uint64_t prev_size = prev->second->first;
            if (prev->first + prev_size == offset) {
                offset = prev->first;
                size += prev_size;
                p.by_size.erase(prev->second);
                block.free.erase(prev);
            }
        }
        insert_free(p, block, offset, size);
        if (block.allocations) return 0;
        const bool another_empty = std::any_of(p.blocks.begin(), p.blocks.end(), [&](const auto& other) {
            return other.get() != &block && !other->allocations;
        });
        if (!another_empty) return 0;  // kept as the pool's spare
        const std::uint64_t memory = block.memory;
        for (const auto& [range_offset, entry] : block.free) p.by_size.erase(entry);
        p.blocks.erase(std::find_if(p.blocks.begin(), p.blocks.end(),
                                    [&](const auto& owned) { return owned.get() == &block; }));
        return memory;
    }

    // Every block's memory, for the device's teardown; the allocator is empty afterwards. Ranges still allocated
    // are dropped with their blocks.
    std::vector<std::uint64_t> release_all() {
        std::lock_guard lock(mutex_);
        std::vector<std::uint64_t> memory;
        for (Pool& p : pools_) {
            for (const auto& block : p.blocks) memory.push_back(block->memory);
            p.blocks.clear();
            p.by_size.clear();
        }
        return memory;
    }

    Stats stats() const {
        std::lock_guard lock(mutex_);
        Stats s;
        for (const Pool& p : pools_) {
            for (const auto& block : p.blocks) {
                ++s.blocks;
                s.empty_blocks += !block->allocations;
                s.allocations += block->allocations;
                s.block_bytes += block->size;
                s.used_bytes += block->used;
            }
        }
        return s;
    }

    // The free ranges of the pool's blocks in block order, then offset order, for tests.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> free_ranges(SuballocPool pool) const {
        std::lock_guard lock(mutex_);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> out;
        for (const auto& block : pools_[pool.index()].blocks)
            for (const auto& [offset, entry] : block->free) out.emplace_back(offset, entry->first);
        return out;
    }

    struct FreeRef {
        Block* block = nullptr;
        std::uint64_t offset = 0;
    };
    using SizeIndex = std::multimap<std::uint64_t, FreeRef>;  // free size -> where

    struct Pool;
    struct Block {
        Pool* pool = nullptr;
        std::uint64_t memory = 0;
        void* base = nullptr;
        std::uint64_t size = 0;
        std::uint64_t used = 0;
        std::uint32_t allocations = 0;
        std::map<std::uint64_t, SizeIndex::iterator> free;  // offset -> its entry in the pool's size index
    };
    struct Pool {
        std::vector<std::unique_ptr<Block>> blocks;
        SizeIndex by_size;
    };

private:
    static std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
        const std::uint64_t rest = value % alignment;
        return rest ? value + (alignment - rest) : value;
    }

    static void insert_free(Pool& p, Block& block, std::uint64_t offset, std::uint64_t size) {
        if (!size) return;
        block.free[offset] = p.by_size.emplace(size, FreeRef{&block, offset});
    }

    // Takes [aligned, aligned + size) out of the free range [offset, offset + free_size).
    static Allocation carve(Pool& p, Block& block, std::uint64_t offset, std::uint64_t free_size, std::uint64_t aligned,
                            std::uint64_t size) {
        const auto entry = block.free.find(offset);
        p.by_size.erase(entry->second);
        block.free.erase(entry);
        insert_free(p, block, offset, aligned - offset);
        insert_free(p, block, aligned + size, offset + free_size - (aligned + size));
        block.used += size;
        ++block.allocations;
        Allocation out;
        out.block = &block;
        out.memory = block.memory;
        out.offset = aligned;
        out.size = size;
        out.mapped = block.base ? static_cast<std::uint8_t*>(block.base) + aligned : nullptr;
        return out;
    }

    mutable std::mutex mutex_;
    Pool pools_[256];  // by SuballocPool::index()
};

}  // namespace leonida
