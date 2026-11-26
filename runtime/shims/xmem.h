#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "../src/gpu_label_memory.h"  // leonida::label::Target: the console's value writes (list slots 71 and 72)

namespace rt::xmem {

// CopyPageMappingsBatchX, Xbox queue slot 31. Measured x64 layout (also documented by VIctory).
struct PageRange {
    std::uint32_t kind, start, count;
};
struct PageBatch {
    std::uintptr_t destination;
    std::uint32_t count;
    const PageRange* ranges;
};
static_assert(sizeof(PageRange) == 12 && sizeof(PageBatch) == 24 && offsetof(PageBatch, ranges) == 16);

// Record the two measured stream-pool reservation return sites; other callers are ignored.
void note_stream_reservation(std::uintptr_t base, std::size_t size, std::uintptr_t caller_rva);
bool queue_page_mappings_enabled();
bool slot31_runs();
bool copy_page_mappings(std::uint32_t count, const PageBatch* batches, std::uintptr_t pool,
                        std::uint32_t pool_pages, std::uint32_t flags);

void geometry_aliases(std::uintptr_t va, std::vector<std::uintptr_t>& addresses);
// Test seam: LEONIDA_GEOMETRY_REMAP_RELEASE is read again at the next remap.
void reload_remap_release_setting();
bool commit_physical_range(const void* address, std::size_t size);
// A lifetime-pinned alias of a contiguous range of the title's writable mapped physical pages.
std::shared_ptr<std::uint8_t> pin_mapped_data(const void* address, std::size_t size);
// Test seam: section pages the zero-page pool has handed out. A replaced plant returns its page, so a
// plant-and-replace loop does not grow this.
std::uint64_t zero_pages_taken();
// Test seam: the address of physical page `page` in its section's scratch view (xmem's own mirror of the section), or
// 0 for a page that is not live. A write there must fault: it is not the title's memory.
std::uintptr_t physical_page_scratch(std::uint64_t page);
// Test seam: called at the points of a fault where another thread's map, unmap or plant of the faulting page can
// interleave with it (FaultRaceStage). Null: none. The hook runs on the thread named by the stage, with the locks that
// stage says are held: it may wait on an event, but must not allocate, log or fault.
enum FaultRaceStage : int {
    kFaultAfterLook = 0,       // the handler, after the shared look, with its lock released
    kFaultBeforeExclusive = 1, // the handler, just before the exclusive lock (a write the look left to it)
    kFaultDecided = 2,         // the handler, inside the shared look after the query it decides on: the lock still held
    kRemapBetween = 3,         // XMemMapPhysicalPages, between the unmap and the map, the exclusive lock held
};
using FaultRaceHook = void (*)(int stage);
void set_fault_race_hook(FaultRaceHook hook);
// Test seam: how the fault handler left the access violations it saw, one count per exit (all since the start).
enum class FaultExit : int {
    nested,          // taken while this thread held the fault lock, or inside the handler: passed on unanswered
    answered,        // a mapped page committed, or found committed and allowed (another thread's map or commit)
    planted,         // a window page the title never mapped, backed by a zero page
    committed,       // a section page in the title's reservation with no view of xmem's, committed
    passed_guarded,  // a committed page whose protection refuses the access, or a host guarded read of an unmapped page
    passed_foreign,  // memory that is not the title's reservation and not only reserved: someone else's fault
    left_to_title,   // the title's reservation, or a write into memory that is only reserved: noted (FaultNote)
    count
};
std::uint64_t fault_exits(FaultExit exit);
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
// leonida::kernel_writes' unguarder (src/title_page_watch.h): pages of physical-page views in [begin, end) left
// read-only with no guard to reopen them are made writable again. True when such a page there is writable now.
bool reopen_unowned(std::uintptr_t begin, std::uintptr_t end);

// Copy a validated DirectStorage staging buffer through the page table. The page list has
// one frame number per 64 KiB page. All pages are resolved before a write starts.
bool transfer_physical(std::span<const std::uint64_t> pages, std::uint16_t offset,
                       std::span<std::uint8_t> staging, bool write);

// Console queue slot 31, CopyPageMappingsBatchX (d3d12_xs_device.cpp): maps at `destination` the physical pages that
// are mapped at `source` now, `pages` 64 KiB pages one after another, or the one page at `source` every time with
// `repeat_first`. A source page with nothing mapped leaves its destination page unmapped; a null `source` unmaps them
// all. False, with the last error set, for an unaligned address or a destination outside a physical-page window.
// Set by xmem.cpp when it is linked: the command-object proxy is also compiled into tests without it.
using CopyPageMappingsFn = bool (*)(void* destination, const void* source, std::uint64_t pages, bool repeat_first);
inline CopyPageMappingsFn copy_page_mappings_fn = nullptr;

// Writes `size` bytes at `address` in the title's memory, through the host pages xmem manages: its own mapping of the
// physical page, which the CPU may write where the title's view of the page is PAGE_NOACCESS. False when the range is
// not xmem's pages. Registered with leonida::value_sink (src/value_write_sink.h) as the fallback of the console's
// value writes (list slots 71 and 72), so that the D3D12 layer reaches it without linking this shim.
bool write_title_bytes(const void* address, const void* bytes, std::size_t size);

// The label target of an address in the pages the title maps: those pages are the title's memory too, and the
// console's value writes name them (runtime/src/vulkan_gpu_labels.h). The shim owns the page tables, so it answers
// the question. False when the address is not in one of the title's mapped pages.
using LabelTargetFn = bool (*)(std::uintptr_t address, unsigned bytes, leonida::label::Target& out);
inline LabelTargetFn label_target_fn = nullptr;

}
