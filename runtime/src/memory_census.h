#pragma once
// The process's address space by kind, for the stall, crash and exit reports: one VirtualQuery walk (a few tens of
// milliseconds over ~200,000 regions, so only at those events), summed by type (image, mapped, private) and state, with
// the private figures the memory line shows and the largest committed regions. Under `tag`.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

namespace leonida::memory_census {

inline void print(const char* tag) {
    struct Sum {
        unsigned long long regions = 0, bytes = 0;
    };
    Sum committed[3], reserved;  // image, mapped, private
    struct Big {
        std::uintptr_t base = 0;
        unsigned long long bytes = 0;
        DWORD type = 0, protect = 0;
    } big[8];
    MEMORY_BASIC_INFORMATION m{};
    std::uintptr_t at = 0x10000;
    unsigned long long regions = 0;
    while (at < 0x7FFFFFFF0000ull && VirtualQuery(reinterpret_cast<const void*>(at), &m, sizeof m) == sizeof m) {
        ++regions;
        if (m.State == MEM_COMMIT) {
            const int k = m.Type == MEM_IMAGE ? 0 : m.Type == MEM_MAPPED ? 1 : 2;
            ++committed[k].regions;
            committed[k].bytes += m.RegionSize;
            Big* smallest = std::min_element(std::begin(big), std::end(big),
                                             [](const Big& a, const Big& b) { return a.bytes < b.bytes; });
            if (m.RegionSize > smallest->bytes)
                *smallest = {reinterpret_cast<std::uintptr_t>(m.BaseAddress), m.RegionSize, m.Type, m.Protect};
        } else if (m.State == MEM_RESERVE) {
            ++reserved.regions;
            reserved.bytes += m.RegionSize;
        }
        const std::uintptr_t next = reinterpret_cast<std::uintptr_t>(m.BaseAddress) + m.RegionSize;
        if (next <= at) break;
        at = next;
    }
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    pmc.cb = sizeof pmc;
    GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof pmc);
    MEMORYSTATUSEX status{sizeof status};
    GlobalMemoryStatusEx(&status);
    std::fprintf(stderr,
                 "[rt] memory census %s: private %llu MiB, working set %llu MiB, %llu regions; committed: image %llu "
                 "MiB (%llu), mapped %llu MiB (%llu), private %llu MiB (%llu); reserved %llu MiB (%llu); PC free %llu "
                 "MiB of %llu, commit %llu of %llu MiB\n",
                 tag, static_cast<unsigned long long>(pmc.PrivateUsage >> 20),
                 static_cast<unsigned long long>(pmc.WorkingSetSize >> 20), regions, committed[0].bytes >> 20,
                 committed[0].regions, committed[1].bytes >> 20, committed[1].regions, committed[2].bytes >> 20,
                 committed[2].regions, reserved.bytes >> 20, reserved.regions,
                 static_cast<unsigned long long>(status.ullAvailPhys >> 20),
                 static_cast<unsigned long long>(status.ullTotalPhys >> 20),
                 static_cast<unsigned long long>((status.ullTotalPageFile - status.ullAvailPageFile) >> 20),
                 static_cast<unsigned long long>(status.ullTotalPageFile >> 20));
    std::sort(std::begin(big), std::end(big), [](const Big& a, const Big& b) { return a.bytes > b.bytes; });
    for (const Big& b : big)
        if (b.bytes)
            std::fprintf(stderr, "[rt] memory census %s:   %#llx %llu MiB %s protect %#lx\n", tag,
                         static_cast<unsigned long long>(b.base), b.bytes >> 20,
                         b.type == MEM_IMAGE ? "image" : b.type == MEM_MAPPED ? "mapped" : "private", b.protect);
}

}  // namespace leonida::memory_census
