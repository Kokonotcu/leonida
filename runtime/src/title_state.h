#pragma once
// The retail title's main state machine, loading-screen flags and hang watchdog, read for the stall and crash reports
// (GTA VI 1.0.2.1; addresses from five-star-recomp docs/debugging.md, "Hang watchdog"). Read-only, guarded: an address
// the title has not committed reads as "?". One line under `tag`.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <cstdint>
#include <cstdio>

namespace leonida::title_state {

template <class T>
inline bool read(std::uint64_t address, T& out) {
    SIZE_T got = 0;
    return ReadProcessMemory(GetCurrentProcess(), reinterpret_cast<const void*>(address), &out, sizeof out, &got) &&
           got == sizeof out;
}

inline void print(const char* tag) {
    std::uint16_t state = 0, next = 0;
    std::uint8_t loading = 0, pause = 0, session_flag = 0;
    std::uint64_t heartbeat = 0, handler = 0, session = 0;
    const bool s = read(0x1471273A0 + 0x10, state) && read(0x1471273A0 + 0x12, next);
    const bool l = read(0x147453ED4, loading);
    const bool h = read(0x1471B6338, heartbeat) && read(0x1471B6330, pause) && read(0x1471B6348, handler);
    const bool se = read(0x14AAE3B70, session) && session && read(session + 0x1B3, session_flag);
    LARGE_INTEGER now{}, frequency{};
    QueryPerformanceCounter(&now);
    QueryPerformanceFrequency(&frequency);
    const double beat_age =
        h && heartbeat && frequency.QuadPart ? double(now.QuadPart - std::int64_t(heartbeat)) / double(frequency.QuadPart)
                                             : -1;
    std::fprintf(stderr,
                 "[rt] title state %s: main state %d next %d, loading-screen flag %d, session %#llx flag %d, heartbeat "
                 "%.1f s ago, pause %d, watchdog handler %#llx\n",
                 tag, s ? int(state) : -1, s ? int(next) : -1, l ? int(loading) : -1,
                 static_cast<unsigned long long>(session), se ? int(session_flag) : -1, beat_age, h ? int(pause) : -1,
                 static_cast<unsigned long long>(handler));
}

}  // namespace leonida::title_state
