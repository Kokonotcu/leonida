// Opt-in PresentX slot routing and a windowless first-frame capture for the trace-named queue slots
// (docs/d3d12_vulkan.md section 4.5). Nothing here runs unless LEONIDA_PRESENTX_SLOT (or the suspend/resume slot)
// names a queue slot past the public ID3D12CommandQueue chain; the routing itself lives in d3d12_xs_device.cpp.
//
// LEONIDA_CAPTURE=<dir> makes a routed PresentX write present-<n>-plane-<m>.ppm instead of presenting: no window,
// no swapchain, no presenter. Every pointer the call passes is validated with VirtualQuery before it is read, the
// resource must answer the layer's private Resource tag, and a malformed call returns E_INVALIDARG with one
// throttled log line, never a fault.
#pragma once

#include <d3d12.h>

namespace rt::d3d12 {

struct XboxPresentPlaneParameters;

// The queue slots named by LEONIDA_PRESENTX_SLOT, LEONIDA_PRESENTX_SLOT_ALT, LEONIDA_PRESENTX_SUSPEND_SLOT and
// LEONIDA_PRESENTX_RESUME_SLOT. 0 means "not configured". The public HLE reports PresentX at both 30 and 32; with
// LEONIDA_CAPTURE set those two are the defaults, so a first capture run needs no other variable. Read once, on the
// first queue call past the public chain, so the feature is inert (a branch that is not taken) when unset.
struct PresentXRouteConfig {
    unsigned present_slot = 0;
    unsigned present_slot_alt = 0;
    unsigned suspend_slot = 0;
    unsigned resume_slot = 0;
};
const PresentXRouteConfig& presentx_route_config();

// True when LEONIDA_CAPTURE names a non-empty directory. Read once with the route config.
bool presentx_capture_enabled();

// A routed PresentX. With capture off it returns S_OK without reading `planes` (a routed present is accepted and
// dropped: no window and no swapchain are created). With capture on it validates the plane array and every resource
// pointer, requires each resource to answer the layer's private Resource tag, copies each plane's first resource
// through `device` and `queue`, and writes one PPM per plane. E_INVALIDARG for a malformed call (logged once per
// distinct message per 250 ms), E_FAIL for a device, queue or file failure.
HRESULT presentx_capture_present(ID3D12Device* device, ID3D12CommandQueue* queue, UINT plane_count,
                                 const XboxPresentPlaneParameters* planes);

}  // namespace rt::d3d12
