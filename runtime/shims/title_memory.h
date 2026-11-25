// The memory a title is told it may use: XSystemGetAppMemoryUsageLimit and XMemGetWorkingSetStatistics (+0) answer
// with this one value, because a title sizes its memory pools from it (a #14 finding on the new repository: a 6 GiB
// override of both made a capped run live 78 s instead of 50 s before the OOM kill).
//
// Both used to report the host's total physical RAM: 31.7 GiB on a 32 GB PC, all of it on a 16 GB one, as if the OS,
// the Vulkan driver, the runtime and the title's GPU-visible memory were free. The answer is now the console's title
// budget, capped by what the host can back:
//   - 13.5 GiB, what a Series X title gets (ASSUMPTION: the documented 13.5 GB title budget, read as GiB; the game
//     was built against it);
//   - at most the host's physical RAM minus headroom: max(4 GiB, a quarter of it), for the OS, the driver and gta6rt;
//     never below 2 GiB;
//   - LEONIDA_TITLE_MEMORY_MB=<n> overrides it (n MiB, 1..1048576), to try a different budget without a rebuild.
#pragma once

#include <windows.h>

#include <cstdint>
#include <cstdlib>

namespace rt::title_memory {

inline constexpr std::uint64_t kGiB = 1024ull * 1024 * 1024;
inline constexpr std::uint64_t kConsoleTitleBudget = 27 * kGiB / 2;  // 13.5 GiB
inline constexpr std::uint64_t kFloor = 2 * kGiB;

// Pure rule, for tests: the budget for a host with `physical` bytes of RAM and an override of `override_mb` MiB (0:
// none).
inline std::uint64_t budget_for(std::uint64_t physical, std::uint64_t override_mb) {
    if (override_mb) return override_mb * 1024 * 1024;
    if (!physical) return 12 * kGiB;  // the host did not answer: the earlier fallback
    const std::uint64_t quarter = physical / 4;
    const std::uint64_t headroom = quarter > 4 * kGiB ? quarter : 4 * kGiB;
    const std::uint64_t host = physical > headroom + kFloor ? physical - headroom : kFloor;
    return host < kConsoleTitleBudget ? host : kConsoleTitleBudget;
}

inline std::uint64_t override_mb() {
    char value[32] = {};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_TITLE_MEMORY_MB", value, sizeof(value));
    if (n == 0 || n >= sizeof(value)) return 0;
    char* end = nullptr;
    const unsigned long long mb = std::strtoull(value, &end, 10);
    if (!end || *end || mb == 0 || mb > 1048576) return 0;
    return mb;
}

// An Xbox Series S title gets less: 8 GiB (ASSUMPTION: the documented ~8 GB a Series S game may use of its 10 GB).
// With LEONIDA_DEVICE_TYPE=6 (Series S) the title is told that, so it sizes its pools as on that console; told
// 13.5 GiB it committed ~13.7 GiB of pools on top of the GPU copies of its textures and ran a 32 GB PC out of memory
// in the first loading screen.
inline constexpr std::uint64_t kSeriesSTitleBudget = 8 * kGiB;

inline bool series_s() {
    char value[8] = {};
    const DWORD n = GetEnvironmentVariableA("LEONIDA_DEVICE_TYPE", value, sizeof(value));
    return n == 1 && value[0] == '6';
}

// What the title is told, computed once per process (the answer must not change under the title).
inline std::uint64_t limit() {
    static const std::uint64_t value = [] {
        MEMORYSTATUSEX ms{sizeof(ms)};
        const std::uint64_t physical = GlobalMemoryStatusEx(&ms) ? ms.ullTotalPhys : 0;
        const std::uint64_t mb = override_mb();
        const std::uint64_t budget = budget_for(physical, mb);
        return !mb && series_s() && budget > kSeriesSTitleBudget ? kSeriesSTitleBudget : budget;
    }();
    return value;
}

}  // namespace rt::title_memory
