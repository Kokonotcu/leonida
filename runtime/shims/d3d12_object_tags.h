// Private implementation tags of the Vulkan-backed D3D12 layer objects (d3d12_com_object.h, Object::kTag). They are
// not D3D12 interfaces: QueryInterface(kLeonidaResourceTag) succeeds only on this layer's Resource and returns its
// implementation. A pointer whose vtable looks executable is not enough to call ID3D12Resource methods on it; a host
// runtime resource, for example, does not answer this tag.
#pragma once

#include <guiddef.h>

namespace rt::d3d12 {

// {9b3e5f12-7c48-4d0e-a6b1-2f8d4c7e9a35}: the Resource class behind d3d12_vulkan_device.cpp.
inline constexpr GUID kLeonidaResourceTag{0x9b3e5f12, 0x7c48, 0x4d0e,
                                          {0xa6, 0xb1, 0x2f, 0x8d, 0x4c, 0x7e, 0x9a, 0x35}};

}  // namespace rt::d3d12
