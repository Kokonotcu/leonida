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

// Host memory whose protection we flip ourselves: the title-heap dirty watch keeps a heap's pages read-only between
// syncs (d3d12_xs). True when [begin, end) overlaps such a range. Host imports stay away from it: on NVIDIA one import
// of memory that is not plain read-write is refused (-2) and every host import after it fails too, for the rest of the
// device's life (heapimport `alias` test, 2026-11-25; boykopovar/AnyPS5#1022 saw the same on Linux).
using ProtectWatchedFn = bool (*)(std::uintptr_t begin, std::uintptr_t end);
inline std::atomic<ProtectWatchedFn> g_protect_watched{nullptr};

// Changes whenever a physical page is mapped or unmapped anywhere: a reader that found a range readable need not ask
// VirtualQuery again (one call per 64 KiB view) until it changes.
inline std::atomic<std::uint64_t> g_map_generation{0};

// Calls that take the process's address-space lock exclusively, counted for the rate report (d3d12_xs sync): every
// VirtualQuery in the process (ours, the driver's in each host import: 15-30 ms per call in the game, 0.001-0.05 ms
// alone) waits behind them.
inline std::atomic<std::uint64_t> g_commits_on_touch{0};  // physical_page_fault: VirtualAlloc(MEM_COMMIT) of a page
inline std::atomic<std::uint64_t> g_view_maps{0};         // MapViewOfFile3 of physical pages
inline std::atomic<std::uint64_t> g_view_unmaps{0};       // UnmapViewOfFile2
inline std::atomic<std::uint64_t> g_placeholder_splits{0};  // VirtualFree(MEM_PRESERVE/COALESCE_PLACEHOLDER)

// The GPU's view of physical pages (the D3D12 layer's title page table). A page's own memory is its 64 KiB in the
// whole-section view of the section behind it (`section`, `section_bytes`), which stays at one address while the
// section lives, whatever window the page is mapped into. The page table points each window address at that memory,
// so a remap is one entry: an import of the window's views instead pins the pages mapped there at import time, and
// the GPU went on reading those after the title had mapped others (stale BVH inputs, then a hung GPU).
//
// `g_page_mapped(address, backing, section, section_bytes)`: the window page at `address` now shows `backing` (null:
// nothing). Called by xmem with its page lock held, after the host mapping changed.
using PageMappedFn = void (*)(std::uintptr_t address, void* backing, void* section, std::uint64_t section_bytes);
inline std::atomic<PageMappedFn> g_page_mapped{nullptr};
// A section's whole-section view is about to be unmapped (its last page freed and unmapped), with xmem's lock held.
using SectionReleasedFn = void (*)(void* section, std::uint64_t section_bytes);
inline std::atomic<SectionReleasedFn> g_section_released{nullptr};
// Calls `fn` for every 64 KiB page of [start, end) inside a physical-page window, mapped (with its backing) or not
// (null), with xmem's page lock held, and returns how many there were. Set by xmem when it first hands out pages.
using WindowPagesFn = std::uint64_t (*)(std::uintptr_t start, std::uintptr_t end, PageMappedFn fn);
inline std::atomic<WindowPagesFn> g_window_pages{nullptr};
// Calls `fn` for every window page mapped now (the page table's start), with xmem's page lock held; how many.
using MappedPagesFn = std::uint64_t (*)(PageMappedFn fn);
inline std::atomic<MappedPagesFn> g_mapped_pages{nullptr};

}
