// The provisional D3D12XBOX_PRESENT_* layouts of docs/d3d12_vulkan.md section 4.5, in a header with no Vulkan or
// presenter dependency so the queue proxy (d3d12_xs_device.cpp) can forward a routed PresentX and the windowless
// capture can read it without linking the presentation core.
//
// WHAT IS KNOWN: the public GDKX samples give the call order, the argument counts and the field names.
// WHAT IS OURS: every layout, constant value and size below is a provisional definition; the real slots, layouts and
// enum values are unmeasured.
#pragma once

#include <d3d12.h>

#include <cstdint>

namespace rt::d3d12 {

// Provisional (ours): D3D12XBOX_PRESENT_PLANE_PARAMETERS. The names are those the samples use; the layout, the
// token width (64 bits here) and the fields the console has beyond these are unmeasured.
//
// The fields after pSrcViewRects exist in the console's struct (names and order seen in a public binding, see the struct)
// but their types, enum values and flag bits are unmeasured: the values of ours below are placeholders that make the
// composition (section 4.5) testable. Zero everything is the default and means "as before": the whole output,
// bilinear when scaling, SDR, plane 1 premultiplied and not composed.
struct XboxViewRect {  // D3D12XBOX_VIEW_RECT: four floats; as fractions 0..1 of the output (hypothesis, section 4.5)
    float left, top, right, bottom;
};
// DXGI_COLOR_SPACE_TYPE values (documented DXGI numbers). Only these two are composed; any other is treated as the first.
inline constexpr int xbox_color_space_g22_p709 = 0;       // DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709: SDR
inline constexpr int xbox_color_space_g2084_p2020 = 12;   // DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020: HDR10
// Provisional (ours): ScaleFilter and Flags.
inline constexpr int xbox_scale_filter_default = 0;  // bilinear (point when not scaled makes no difference)
inline constexpr int xbox_scale_filter_point = 1;
inline constexpr int xbox_scale_filter_linear = 2;
// Plane 1 is composed over plane 0 as an overlay only with this flag (hypothesis: whether the console blends a second plane
// at all is unmeasured; the public HDR sample's plane 1 is the GameDVR copy of the frame, not shown, so without the flag
// plane 1 is not composed, as before).
inline constexpr UINT xbox_plane_flag_overlay = 1;
inline constexpr UINT xbox_plane_flag_straight_alpha = 2;  // plane 1's colour is not premultiplied by its alpha

struct XboxPresentPlaneParameters {
    // Field ORDER and names: the public Go binding of the GDK header in hajimehoshi/ebiten (Apache-2.0,
    // internal/graphicsdriver/directx/d3d12_windows.go, _D3D12XBOX_PRESENT_PLANE_PARAMETERS); the HDR10 sample confirms
    // the first four and ColorSpace. Types, padding and the console's real sizes are unmeasured (ours: 64-bit token, int
    // enums), so nothing here is the console's ABI.
    std::uint64_t Token;                  // from WaitFrameEventX
    UINT ResourceCount;                   // 1 in the samples
    ID3D12Resource* const* ppResources;   // ResourceCount committed resources with ALLOW_DISPLAY, in PRESENT
    const D3D12_RECT* pSrcViewRects;      // null, or one rectangle per resource (the HDR sample sets it)
    const XboxViewRect* pDestPlacementBase;  // null: the whole output
    int ColorSpace;                       // a DXGI_COLOR_SPACE_TYPE (see above)
    int ScaleFilter;                      // xbox_scale_filter_*; anything unknown is the default
    UINT ExtendedDescCount;               // not used (logged once when non-zero)
    const void* pExtendedDescs;
    UINT Flags;                           // xbox_plane_flag_*; unknown bits are ignored
};

// Provisional (ours): D3D12XBOX_PRESENT_PARAMETERS. The samples show at least this field.
struct XboxPresentParameters {
    // 0 (the default): a frame that missed its vblank waits for the next one. Above 0: a frame that missed it is shown
    // at once, mid-scan, when the scan is not further than this percentage down the screen (100: wherever it is).
    // Needs VK_KHR/EXT_swapchain_maintenance1 and an IMMEDIATE or FIFO_RELAXED mode the swapchain can switch to;
    // without them it is ignored (logged once). What late means: see FramePacer::present_timing.
    float ImmediateThresholdPercent;
};

}  // namespace rt::d3d12
