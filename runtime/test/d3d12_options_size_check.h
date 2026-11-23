// CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS) with the console's larger structure, run by the tests of both
// backends: the Vulkan device (d3d12_vulkan_device_test, raw and behind the proxy) and the proxy over a PC runtime
// device (device_proxy_test, WARP). See shims/d3d12_console_feature.h for the rule and why only this feature has it.
#pragma once

#include <d3d12.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace optsize {

// 0x3C on the PC, 0x40 on the console (measured on a retail console title).
constexpr UINT kPcSize = sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS);
static_assert(kPcSize == 0x3C, "the PC structure is 0x3C bytes");

constexpr std::uint8_t kFill = 0xCD;  // what the caller's buffer holds before the call
constexpr size_t kGuard = 16;         // bytes after the caller's structure that no call may touch

// Runs the cases against `device` and returns the number of failures (each printed to stderr with `name`).
//   - the PC size works and every larger size, from 0x3D to far above 0x40, gets the same PC fields, S_OK, every
//     byte after them zero although the caller filled the buffer with 0xCD, and the guard bytes after the caller's
//     structure untouched;
//   - a size smaller than the PC structure, or no buffer with a larger size, is E_INVALIDARG and leaves the buffer
//     as it was;
//   - another feature asked with a larger structure than its PC one is still E_INVALIDARG (the rule is only for the
//     one measured feature), also when its structure is the PC size plus the same 4 bytes.
inline int check(ID3D12Device* device, const char* name) {
    int failures = 0;
    const auto fail = [&](const char* what, UINT size, HRESULT hr) {
        std::fprintf(stderr, "CheckFeatureSupport(D3D12_OPTIONS) %s: %s (size 0x%x, HRESULT 0x%08x)\n", name, what, size,
                     static_cast<unsigned>(hr));
        ++failures;
    };

    D3D12_FEATURE_DATA_D3D12_OPTIONS pc{};
    std::memset(&pc, kFill, sizeof(pc));
    HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &pc, kPcSize);
    if (hr != S_OK) {
        fail("the PC size is not S_OK", kPcSize, hr);
        return failures;  // nothing to compare the larger sizes with
    }
    D3D12_FEATURE_DATA_D3D12_OPTIONS again{};
    hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &again, kPcSize);
    if (hr != S_OK || std::memcmp(&pc, &again, sizeof(pc)) != 0) fail("the PC size answers differently twice", kPcSize, hr);

    // 0x3D (one byte over), 0x40 (the console), 0x44 and 0x80 (more fields), 0x1000 and 0x10000 (far larger).
    for (const UINT size : {kPcSize + 1, 0x40u, 0x44u, 0x80u, 0x1000u, 0x10000u}) {
        std::vector<std::uint8_t> buffer(size + kGuard, kFill);
        hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, buffer.data(), size);
        if (hr != S_OK) {
            fail("a larger size is not S_OK", size, hr);
            continue;
        }
        if (std::memcmp(buffer.data(), &pc, kPcSize) != 0) fail("the PC fields differ from the PC-size answer", size, hr);
        size_t not_zero = 0;
        for (size_t i = kPcSize; i < size; ++i) not_zero += buffer[i] != 0;
        if (not_zero) fail("bytes after the PC structure are not zero", size, hr);
        size_t touched = 0;
        for (size_t i = size; i < buffer.size(); ++i) touched += buffer[i] != kFill;
        if (touched) fail("bytes after the caller's structure were written", size, hr);
    }

    // Smaller than the PC structure: refused as before, buffer untouched (0x3B, 0x38, 4, 1, 0).
    for (const UINT size : {kPcSize - 1, kPcSize - 4, 4u, 1u, 0u}) {
        std::vector<std::uint8_t> buffer(kPcSize + kGuard, kFill);
        hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, buffer.data(), size);
        if (hr != E_INVALIDARG) fail("a smaller size is not E_INVALIDARG", size, hr);
        size_t touched = 0;
        for (const std::uint8_t byte : buffer) touched += byte != kFill;
        if (touched) fail("a refused call wrote to the buffer", size, hr);
    }

    // No buffer: refused for the console's sizes. (The PC size with no buffer is the PC runtime's own call: WARP
    // crashes on it where the host adapter answers E_INVALIDARG, so it is not made here.)
    for (const UINT size : {0x40u, 0x1000u}) {
        hr = device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, nullptr, size);
        if (hr != E_INVALIDARG) fail("no buffer is not E_INVALIDARG", size, hr);
    }

    // Other features keep their exact-size rule: one measured size, one feature. The PC structure plus 4 bytes is
    // the shape the console's D3D12_OPTIONS has; for these it stays E_INVALIDARG. The buffers start as zeros, so a
    // feature with an input field (FORMAT_SUPPORT's format, ARCHITECTURE's node) has a valid one and only the size
    // can be the reason for the refusal.
    struct Other {
        D3D12_FEATURE feature;
        UINT pc_size;
        const char* what;
    };
    const Other others[] = {
        {D3D12_FEATURE_D3D12_OPTIONS1, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS1), "OPTIONS1"},
        {D3D12_FEATURE_D3D12_OPTIONS2, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS2), "OPTIONS2"},
        {D3D12_FEATURE_D3D12_OPTIONS3, sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS3), "OPTIONS3"},
        {D3D12_FEATURE_ARCHITECTURE, sizeof(D3D12_FEATURE_DATA_ARCHITECTURE), "ARCHITECTURE"},
        {D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, sizeof(D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT),
         "GPU_VIRTUAL_ADDRESS_SUPPORT"},
        {D3D12_FEATURE_FORMAT_SUPPORT, sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT), "FORMAT_SUPPORT"},
    };
    for (const Other& other : others) {
        // The exact size must work for the input used below, so the refusals are about the size.
        std::vector<std::uint8_t> exact(other.pc_size, 0);
        if (other.feature == D3D12_FEATURE_FORMAT_SUPPORT)
            reinterpret_cast<D3D12_FEATURE_DATA_FORMAT_SUPPORT*>(exact.data())->Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        hr = device->CheckFeatureSupport(other.feature, exact.data(), other.pc_size);
        if (hr != S_OK) {
            std::fprintf(stderr, "CheckFeatureSupport(%s) %s: the PC size is not S_OK (HRESULT 0x%08x)\n", other.what, name,
                         static_cast<unsigned>(hr));
            ++failures;
            continue;
        }
        for (const UINT size : {other.pc_size + 4, other.pc_size + 1, 0x1000u}) {
            std::vector<std::uint8_t> buffer(size + kGuard, 0);
            std::memset(buffer.data() + size, kFill, kGuard);
            std::memcpy(buffer.data(), exact.data(), other.pc_size);
            hr = device->CheckFeatureSupport(other.feature, buffer.data(), size);
            if (hr != E_INVALIDARG) {
                std::fprintf(stderr, "CheckFeatureSupport(%s) %s: a larger size is not E_INVALIDARG (size 0x%x, HRESULT 0x%08x)\n",
                             other.what, name, size, static_cast<unsigned>(hr));
                ++failures;
            }
            size_t touched = 0;
            for (size_t i = size; i < buffer.size(); ++i) touched += buffer[i] != kFill;
            if (touched) {
                std::fprintf(stderr, "CheckFeatureSupport(%s) %s: a call wrote after the caller's structure (size 0x%x)\n",
                             other.what, name, size);
                ++failures;
            }
        }
    }
    return failures;
}

}  // namespace optsize
