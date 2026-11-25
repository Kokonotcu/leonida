#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rt::xmem {

bool commit_physical_range(const void* address, std::size_t size);
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);
// Writes (or reads) `size` bytes of `staging` to the physical pages `pages` (one per 64 KiB, `count` of them), from
// `offset` in the first page. False when a page is not one of the title's (0: a list the title has not filled yet).
bool transfer_physical(const std::uint64_t* pages, std::size_t count, std::uint16_t offset, std::uint8_t* staging,
                       std::size_t size, bool write);

// commit_physical_range for code that may be linked without xmem.cpp (the D3D12 layer's tests): set by xmem when it
// first hands out physical pages, null before.
using CommitRangeFn = bool (*)(const void* address, std::size_t size);
inline std::atomic<CommitRangeFn> g_commit_range{nullptr};
// transfer_physical, the same way (DirectStorage requests into physical pages; dstorage_test links no xmem).
using TransferPhysicalFn = bool (*)(const std::uint64_t*, std::size_t, std::uint16_t, std::uint8_t*, std::size_t, bool);
inline std::atomic<TransferPhysicalFn> g_transfer_physical{nullptr};

// Bytes written into title memory behind the title's back (DirectStorage requests): the D3D12 layer sets this to the
// watch of its title-placed textures, which are uploaded again. Called with xmem's lock held for page writes, so it
// only records the range.
using TitleWrittenFn = void (*)(const void* address, std::size_t size);
inline std::atomic<TitleWrittenFn> g_title_written{nullptr};
// The same memory's address in the mirrored window, 0 for none (set by xmem once it hands out physical pages).
using MirrorAddressFn = std::uintptr_t (*)(std::uintptr_t address);
inline std::atomic<MirrorAddressFn> g_mirror_address{nullptr};
// Tells the watch of a write at `address`, and at its mirror.
inline void note_title_written(const void* address, std::size_t size) {
    const auto told = g_title_written.load(std::memory_order_acquire);
    if (!told || !size) return;
    told(address, size);
    if (const auto mirror = g_mirror_address.load(std::memory_order_acquire))
        if (const std::uintptr_t other = mirror(reinterpret_cast<std::uintptr_t>(address)))
            told(reinterpret_cast<const void*>(other), size);
}

// Changes whenever a physical page is mapped or unmapped anywhere: a reader that found a range readable need not ask
// VirtualQuery again (one call per 64 KiB view) until it changes.
inline std::atomic<std::uint64_t> g_map_generation{0};

}
