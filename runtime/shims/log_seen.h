// What a "print once" function (stub, note_once, log_once) already printed, for its repeated calls (review I11).
// Those run on hot paths: stub() on every RSSetViewports with more than one viewport, log_once() on every
// CheckFeatureSupport. Each took a global mutex and built a std::string to look the text up. The text is a literal
// at nearly every call site, so its address and the value name the site: a repeated call costs a hash and one
// relaxed atomic load here. A first call, or one whose slot another site has taken since, takes the slow path,
// where the caller's set decides by the text itself (literals at two addresses are still printed once).
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rt::log_seen {

class Sites {
public:
    // True when this text and value were printed (or found printed) already. False may still be a repeat.
    bool seen(const void* text, std::int64_t value = 0) const noexcept {
        const std::uint64_t key = mix(text, value);
        return slots_[key & mask].load(std::memory_order_relaxed) == key;
    }
    // After the slow path printed it or found it printed.
    void remember(const void* text, std::int64_t value = 0) noexcept {
        const std::uint64_t key = mix(text, value);
        slots_[key & mask].store(key, std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t size = 1024, mask = size - 1;
    // splitmix64 of the address and the value: never 0, the empty slot's key.
    static std::uint64_t mix(const void* text, std::int64_t value) noexcept {
        std::uint64_t x = std::uint64_t(reinterpret_cast<std::uintptr_t>(text)) ^
                          (std::uint64_t(value) * 0x9E3779B97F4A7C15ull);
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        x ^= x >> 31;
        return x | 1;
    }
    std::array<std::atomic<std::uint64_t>, size> slots_{};
};

}  // namespace rt::log_seen
