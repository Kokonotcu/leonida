#pragma once

// The Xbox presentation core (docs/d3d12_vulkan.md section 4.5): the frame events and PresentX of a GDKX title, which
// presents without a DXGI swapchain. FramePacer (src/frame_pacer.h) does the timing and the token bookkeeping;
// WindowPresenter (d3d12_com.h, over leonida::VulkanPresenter) puts the pixels in a window. This class is the
// glue, and the surface the Xbox device and queue proxy will forward to.
//
// WHAT IS KNOWN: the public GDKX samples give the call order and the argument counts; public code that is not
// Microsoft documentation gives most of the slots and the x64 layouts (docs/d3d12_xs_surface.md 7.1):
//   - ebiten (Apache-2.0) mirrors the GDK structs in Go: D3D12XBOX_PRESENT_PLANE_PARAMETERS is 0x48 bytes on x64,
//     D3D12XBOX_PRESENT_PARAMETERS 0x20 bytes, and D3D12XBOX_FRAME_PIPELINE_TOKEN is uint64.
//   - XCL (GPL-2.0-only, facts only) measured the frame calls on device slots 78-81 and PresentX on queue slot 30
//     (older runtime layout: XDefiant, Minecraft, Fortnite) or 32 (newer: Unicorn Overlord).
// WHAT IS OURS: the constant values, the HRESULT choices and the ScaleFilter/plane-flag values; the real enum
// values are still unmeasured. The SuspendX/ResumeX slots are not in any public source.
//
// Slots (0-based, QueryInterface = 0) before the proxy can forward:
//   ID3D12Device (Xbox):        SetFrameIntervalX   (output, interval, latency, flags)                    78
//                               ScheduleFrameEventX (event, offset, reserved pointer, flags)              79
//                               WaitFrameEventX     (event, timeout ms, reserved pointer, flags, &token)   80
//                               GetFrameStatisticsX (token, type mask, &count, array)                     81
//   ID3D12CommandQueue (Xbox):  PresentX (plane count, planes, params)                                    30 (older) / 32 (newer)
//                               SuspendX (flags)                                                        slot unknown
//                               ResumeX ()                                                              slot unknown
// The public numbers are measurements of other titles' runtimes; the retail executable's calls of device slots
// 78/79/80 and queue slot 32 are listed in d3d12_xs_device.h (XboxPresentationSink). The runtime version moves
// PresentX, so the queue slots can be named at run time by LEONIDA_PRESENTX_SLOT, LEONIDA_PRESENTX_SUSPEND_SLOT and
// LEONIDA_PRESENTX_RESUME_SLOT (presentx_capture.h), and a routed PresentX can be captured without a window.
// The queue-side calls do not look at the queue: the Vulkan device has one VkQueue behind every D3D12 queue, and
// the present is submitted after the work already executed on it.

#include "d3d12_com.h"
#include "xbox_present_types.h"
#include "../src/frame_pacer.h"
#include "../src/vulkan_present.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace rt::d3d12 {

// Provisional (ours): D3D12XBOX_FRAME_INTERVAL_* as periods in microseconds.
inline constexpr std::uint32_t xbox_frame_interval_30hz_us = 33333;
inline constexpr std::uint32_t xbox_frame_interval_60hz_us = 16667;
inline constexpr std::uint32_t xbox_frame_interval_120hz_us = 8333;

class XboxPresentation {
public:
    // `clock` null: real time (a steady high-resolution timer). Tests pass a leonida::ManualFrameClock, which must
    // outlive this object. `presenter` is what create_window_presenter returned. `tokens` null: the pacer owns its
    // own source; the retail sink passes the device's source so WaitFrameEventX tokens are accepted by PresentX.
    explicit XboxPresentation(std::unique_ptr<WindowPresenter> presenter, leonida::FrameClock* clock = nullptr,
                              std::shared_ptr<leonida::FrameTokenSource> tokens = nullptr);
    ~XboxPresentation();
    XboxPresentation(const XboxPresentation&) = delete;
    XboxPresentation& operator=(const XboxPresentation&) = delete;

    // Device side. `output` (an IDXGIOutput*, which the samples fetch from the device) is not used: there is one
    // window. `interval_period_us` is the period of a frame interval, e.g. xbox_frame_interval_60hz_us; `latency`
    // is the frames allowed in flight (the samples pass the buffer count minus 1, or 1).
    HRESULT SetFrameIntervalX(IUnknown* output, UINT interval_period_us, UINT latency, UINT flags);
    // Only the origin event (xbox_frame_event_origin) exists. `offset_us` delays it inside the frame. The pointer
    // argument the samples pass as null is not modelled.
    HRESULT ScheduleFrameEventX(UINT event, UINT offset_us, UINT flags);
    // Blocks for the next origin and while the latency is used up. HRESULT_FROM_WIN32(WAIT_TIMEOUT) after `timeout_ms`
    // (INFINITE waits for ever); `*token` is untouched then. E_INVALIDARG for a null token pointer or another event;
    // HRESULT_FROM_WIN32(ERROR_INVALID_STATE) when nothing is registered (not yet, or since SuspendX).
    // XCL's public implementation returns S_FALSE on a finite timeout and writes the token on timeout only with
    // flag 2, and E_INVALIDARG when the event is not scheduled; the WAIT_TIMEOUT and ERROR_INVALID_STATE here are
    // this layer's choices and known divergences (docs/d3d12_xs_surface.md 7.1).
    HRESULT WaitFrameEventX(UINT event, DWORD timeout_ms, UINT flags, std::uint64_t* token);
    // See FramePacer::get_frame_statistics: S_FALSE and count 0 when nothing is available.
    HRESULT GetFrameStatisticsX(std::uint64_t token, UINT type_mask, UINT* count, leonida::FrameStatistic* out);

    // Queue side.
    //   0 planes: a black frame (what a title does before it releases its resources); no token is looked at.
    //   1 plane:  plane 0's first resource, which must be a single-sample 2D colour texture in the PRESENT state,
    //             is blitted to the window (format conversion by the blit; an sRGB format gets an sRGB swapchain),
    //             limited to pSrcViewRects[0] when that is set. This is the whole behaviour while the plane asks for
    //             nothing else (SDR colour space, whole output, default filter).
    //   Composed: a plane with pDestPlacementBase, a point or linear ScaleFilter, or the HDR10 colour space (and any
    //             second plane that is an overlay) is composed by a compute pass (VulkanPresenter::present_planes): plane 0
    //             from its source rectangle to its destination placement, plane 1 alpha-blended over it, colour spaces
    //             converted to the swapchain's (HDR10 when the surface has it, else HDR10 is tone mapped to SDR). Needs sampled,
    //             single-sample, non-integer sources; others give DXGI_ERROR_UNSUPPORTED.
    //   2 planes: the two share one token. Plane 1 is composed over plane 0 only when xbox_plane_flag_overlay is set on it
    //             (hypothesis); otherwise it is not shown (the GameDVR copy of the public HDR sample), logged once.
    //             Placement, ScaleFilter and Flags values that are not plausible are ignored and logged once, not refused.
    //   More than 2: E_INVALIDARG (the UMD limit, section 4.5).
    //   ResourceCount above 1 on any plane: DXGI_ERROR_UNSUPPORTED (logged once), nothing presented; only resource 0 is read.
    // Token: the one from WaitFrameEventX. Real behaviour for a null, repeated or out-of-order token is unmeasured and a
    // title must not freeze on it, so those are presented anyway and logged once per kind; only a token that was never
    // handed out gives E_INVALIDARG (nothing presented). Skipping a token is fine. Sources the GPU cannot blit
    // (multisampled, integer, block-compressed) give DXGI_ERROR_UNSUPPORTED and present nothing; resolving a
    // multisampled plane into a temporary is not done. While suspended: HRESULT_FROM_WIN32(ERROR_INVALID_STATE), also
    // for a present that was waiting for a swapchain image when SuspendX came.
    // `params` may be null (threshold 0). ImmediateThresholdPercent: see XboxPresentParameters; a frame that was shown
    // at once has its display statistic's percent_scanned set and is counted in immediate_presents().
    // With --trace-d3d12 every call writes one `[d3d12-trace] ... XboxPresentation::PresentX <ok|failed>`
    // line (shims/d3d12_trace.h), so a run's presents can be counted and a refused one is visible.
    HRESULT PresentX(UINT plane_count, const XboxPresentPlaneParameters* planes, const XboxPresentParameters* params);
    // SuspendX drops the interval, the origin event and the frames in flight, and refuses presents. ResumeX allows
    // presents again; SetFrameIntervalX and ScheduleFrameEventX must be called again before the next wait.
    HRESULT SuspendX(UINT flags);
    HRESULT ResumeX();

    // What SDR white is in nits when SDR and HDR10 planes meet (docs section 4.5). `nits` > 0 overrides everything; 0 goes
    // back to automatic: while the swapchain is HDR10, the SDR content brightness Windows has for the output the window
    // is on (a background thread refreshes it once a second; PresentX only reads the cached value), else 203 (ITU-R
    // BT.2408's reference white). Applies to the next PresentX. The setters may be called from any thread.
    void set_sdr_white_nits(float nits) { sdr_white_override_ = nits > 0.0f ? nits : 0.0f; }
    // How SDR planes are decoded and an SDR output is encoded: piecewise sRGB (default) or gamma 2.2. The environment
    // variable LEONIDA_SDR_EOTF=gamma22 (or 2.2; any case; "srgb" is the default; other values are logged once and
    // ignored) chooses when the object is made.
    void set_sdr_eotf(leonida::PresentPlanes::SdrEotf eotf) { sdr_eotf_ = eotf; }
    // Peak of HDR10 content in nits, the white point of the tone map onto an SDR output (default 1000; no metadata is read).
    void set_hdr_peak_nits(float nits) { hdr_peak_nits_ = nits > 0.0f ? nits : 1000.0f; }

    leonida::FramePacer& pacer() { return pacer_; }
    WindowPresenter& presenter() { return *presenter_; }
    std::uint64_t presents() const { return presents_; }
    // Frames shown mid-scan because of ImmediateThresholdPercent.
    std::uint64_t immediate_presents() const { return pacer_.immediate_presents(); }

private:
    // PresentX without its `[d3d12-trace]` line.
    HRESULT present_x(UINT plane_count, const XboxPresentPlaneParameters* planes, const XboxPresentParameters* params);
    float effective_sdr_white_nits();  // override, else the cached Windows setting while HDR10, else 203
    void display_white_worker();       // refreshes display_white_ off the present path
    std::unique_ptr<WindowPresenter> presenter_;
    std::unique_ptr<leonida::FrameClock> own_clock_;
    leonida::FramePacer pacer_;
    std::mutex present_mutex_;  // PresentX, SuspendX and ResumeX one at a time
    std::atomic<std::uint64_t> presents_{0};
    bool logged_planes_ = false, logged_threshold_ = false, logged_display_ = false, logged_color_space_ = false,
         logged_extras_ = false, logged_resource_count_ = false, logged_sdr_plane_ = false;
    std::atomic<float> sdr_white_override_{0.0f};  // > 0: set_sdr_white_nits
    std::atomic<leonida::PresentPlanes::SdrEotf> sdr_eotf_{leonida::PresentPlanes::SdrEotf::srgb};
    std::atomic<float> hdr_peak_nits_{1000.0f};
    // The last good Windows SDR white level (nits, 0: none yet), refreshed by display_white_thread_ about once a second while
    // an HDR10 swapchain is in use (white_wanted_); a failed query keeps the previous value.
    std::atomic<float> display_white_{0.0f};
    std::atomic<bool> white_wanted_{false};
    std::mutex worker_mutex_;
    std::condition_variable worker_wake_;
    bool worker_stop_ = false;
    std::thread display_white_thread_;
    bool logged_null_token_ = false, logged_stale_token_ = false, logged_repeated_token_ = false;
};

inline constexpr UINT xbox_frame_event_origin = leonida::frame_event_origin;

}  // namespace rt::d3d12
