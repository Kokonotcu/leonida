#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rt::xmem {

bool commit_physical_range(const void* address, std::size_t size);
bool commit_physical_pages(const std::uint64_t* page_array, std::size_t count);

// commit_physical_range for code that may be linked without xmem.cpp (the D3D12 layer's tests): set by xmem when it
// first hands out physical pages, null before.
using CommitRangeFn = bool (*)(const void* address, std::size_t size);
inline std::atomic<CommitRangeFn> g_commit_range{nullptr};

// Changes whenever a physical page is mapped or unmapped anywhere: a reader that found a range readable need not ask
// VirtualQuery again (one call per 64 KiB view) until it changes.
inline std::atomic<std::uint64_t> g_map_generation{0};

}
