#pragma once

// Console VAs for buffers on a device with no GPU address arena (docs/d3d12_vulkan.md section 3.3): nothing on such a
// device can resolve them, but a title must still read a non-zero, 64 KiB-aligned, unique address, and the context
// finds the buffer by it. The range sits above 4 GiB (where the arena stays below) and under 2^47, so every address
// and every address inside a buffer fits the 48 bits a V# keeps. An address is never reused. Depends on nothing but
// the standard library, so it is tested without a GPU (synthetic_va_test).

#include <atomic>
#include <cstdint>

namespace leonida {

class SyntheticVa {
public:
    static constexpr std::uint64_t kAlignment = 64 * 1024;
    static constexpr std::uint64_t kFirst = std::uint64_t(1) << 36;
    static constexpr std::uint64_t kLimit = std::uint64_t(1) << 47;

    explicit SyntheticVa(std::uint64_t first = kFirst, std::uint64_t limit = kLimit) : next_(first), limit_(limit) {}

    // The first address of `bytes` bytes, rounded up to whole 64 KiB. Fails, and leaves the counter where it was, when
    // the range cannot hold them: a refused request must not use up space a smaller one could still have.
    bool allocate(std::uint64_t bytes, std::uint64_t& out) {
        if (bytes > limit_) return false;
        const std::uint64_t step = (bytes + kAlignment - 1) / kAlignment * kAlignment;
        std::uint64_t current = next_.load();
        do {
            if (step == 0 || step > limit_ - current) return false;
        } while (!next_.compare_exchange_weak(current, current + step));
        out = current;
        return true;
    }

private:
    std::atomic<std::uint64_t> next_;
    std::uint64_t limit_;
};

}  // namespace leonida
