#pragma once

#include <d3d12.h>

#include <cstdint>
#include <cstring>

// CheckFeatureSupport with the console's larger structure, shared by the Vulkan device and the Xbox-ABI device proxy
// (the PC runtime behind the proxy refuses any size but its own).
//
// Measured (a retail console title, as logged by the XCL project; GTA VI is not measured): it asks for D3D12_OPTIONS
// with 0x40 bytes where the PC structure, D3D12_FEATURE_DATA_D3D12_OPTIONS, is 0x3C, and stops on E_INVALIDARG. A
// request LARGER than the PC structure is therefore answered with the PC fields and the bytes after them zeroed (the
// console's extra fields read as "not supported"). A SMALLER one stays E_INVALIDARG.
//
// Only D3D12_OPTIONS is covered, because it is the only feature with a measured console size. A feature is not added
// from its PC size alone: a console structure may add input fields (FORMAT_SUPPORT, MULTISAMPLE_QUALITY_LEVELS and
// FEATURE_LEVELS carry the question in the structure) that zeros would answer wrongly, and for the rest an oversized
// call is more likely a caller bug, which the host refuses (d3d12_vulkan_device_test holds OPTIONS2-21 to E_INVALIDARG
// for one byte too many). Add a feature here when a boot trace shows a console title asking for it with more bytes
// than the PC structure has; both backends log the size of the first such D3D12_OPTIONS call.
namespace rt::d3d12 {

// True when `feature` asked with `size` bytes is a console-sized request this layer answers: a feature with a measured
// console structure, and more bytes than its PC structure has.
inline bool is_console_sized_feature_request(D3D12_FEATURE feature, UINT size) {
    return feature == D3D12_FEATURE_D3D12_OPTIONS && size > sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS);
}

// Stores the PC answer (`pc_size` bytes) at the start of the caller's structure and zeroes the rest of its `size`
// bytes. `size` is at least `pc_size`; the buffer is the caller's, with exactly `size` bytes.
inline void store_with_console_tail(void* data, UINT size, const void* pc, UINT pc_size) {
    std::memcpy(data, pc, pc_size);
    if (size > pc_size) std::memset(static_cast<std::uint8_t*>(data) + pc_size, 0, size - pc_size);
}

}  // namespace rt::d3d12
