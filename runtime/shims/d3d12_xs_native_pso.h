// Xbox device slot 57: native graphics pipelines (layout and sources in runtime/src/native_pso.h).
//
// The handler reads the 0x108-byte description, finds each stage's source container through the HASH digest in its
// records (<shaders>/native_hashes.txt, written by tools/native_pso_hashes.py), and creates the pipeline on the
// device behind the proxy from the game's own containers, as a PC title would, as one pipeline state stream
// (VS/GS/PS or MS/PS) through the proxy's slot-86 path, which keeps the stream so slot 55 can derive from the
// pipeline. Blob A (blend, raster, depth context registers) is decoded into the description (native_state::apply);
// a blob that does not decode keeps blending and culling off and depth ALWAYS.
//
// Header-only, so every target that compiles d3d12_xs_device.cpp gets it without a build change.
//
// A stage that cannot be resolved refuses the call with E_NOTIMPL (the title skips a failed pipeline, issue #7496),
// or, with LEONIDA_STUB_PIPELINES set, asks the device for a pipeline whose vertex stage has no translation, which
// #7417's device answers with a no-op stub. LEONIDA_NATIVE_PSO=0 refuses every call.
#pragma once

#include "../src/native_pso.h"
#include "d3d12_xs_pso_stream.h"

#include <windows.h>

#include <d3d12.h>

#include <atomic>
#include <climits>
#include <cstring>
#include <cwchar>
#include <span>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

namespace rt::d3d12_xs {

// Where the translator's output is (<work>/shaders/spirv); asked once, on the first slot-57 call. The Vulkan device
// sets it. Without one (a host D3D12 device, which cannot take console containers) every call is refused.
using ShaderDirectoryProvider = std::filesystem::path (*)();

// Creates a pipeline from a stream on the device behind the proxy (ID3D12Device2::CreatePipelineState).
using PipelineStreamCreator = HRESULT (*)(void* context, const D3D12_PIPELINE_STATE_STREAM_DESC* desc,
                                          const IID* riid, void** out);

struct NativePsoStats {
    std::uint64_t calls, created, stubbed, refused;
};

namespace native_pso_detail {

namespace np = leonida::native_pso;

// No function-local statics: their guards fault on the first shim call under Wine with the static CRT (#7175).
inline std::atomic<ShaderDirectoryProvider> g_provider{nullptr};
inline SRWLOCK g_lock = SRWLOCK_INIT;
inline bool g_loaded = false;  // under g_lock
inline std::filesystem::path* g_dir = nullptr;
inline np::HashMap* g_map = nullptr;  // null when there is no usable map
inline std::atomic<std::uint64_t> g_calls{0}, g_created{0}, g_stubbed{0}, g_refused{0}, g_stub_asks{0},
    g_malformed{0};
inline std::atomic<bool> g_first_logged{false}, g_odd_logged{false};

// "0", "off", "false" and "no" (any case) are off; anything else set is on.
inline bool env_flag(const wchar_t* name, bool fallback) {
    wchar_t value[8] = {};
    const DWORD n = GetEnvironmentVariableW(name, value, 8);
    if (n == 0 || n >= 8) return fallback;
    for (const wchar_t* off : {L"0", L"off", L"false", L"no"})
        if (_wcsicmp(value, off) == 0) return false;
    return true;
}

// The map and its folder, loaded on the first call. Read-only afterwards, so callers use them without the lock.
inline const np::HashMap* hash_map(const std::filesystem::path** dir) {
    AcquireSRWLockExclusive(&g_lock);
    if (!g_loaded) {
        g_loaded = true;
        try {
            const ShaderDirectoryProvider provider = g_provider.load();
            auto path = provider ? provider() : std::filesystem::path();
            const auto file = path / L"native_hashes.txt";
            std::ifstream in(file, std::ios::binary);
            const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
            auto map = std::make_unique<np::HashMap>();
            std::string error;
            if (path.empty()) {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: no translated shader folder (needs the Vulkan device "
                             "and LEONIDA_SHADERS or shader_path.txt)\n");
            } else if (!in && text.empty()) {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: no native_hashes.txt in %s; rerun make_game.py's "
                             "shaders stage\n", reinterpret_cast<const char*>(path.u8string().c_str()));
            } else if (!map->parse(text, &error)) {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %s\n", error.c_str());
            } else {
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %zu native shader digests\n", map->size());
                g_map = map.release();
                g_dir = new std::filesystem::path(std::move(path));
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[rt] d3d12_xs: slot 57: native_hashes.txt not loaded: %s\n", e.what());
        }
    }
    ReleaseSRWLockExclusive(&g_lock);
    *dir = g_dir;
    return g_map;
}

inline bool read_container(const std::filesystem::path& dir, const std::string& relative,
                           std::vector<std::uint8_t>& out) {
    try {
        const auto path = dir / std::filesystem::path(std::u8string(relative.begin(), relative.end()));
        std::ifstream in(path, std::ios::binary | std::ios::ate);
        if (!in) return false;
        const std::streamoff size = in.tellg();
        if (size <= 0 || size > (64 << 20)) return false;
        out.resize(static_cast<std::size_t>(size));
        in.seekg(0);
        return static_cast<bool>(in.read(reinterpret_cast<char*>(out.data()), size));
    } catch (const std::exception&) {
        return false;
    }
}

// A container read from disk, shared by every pipeline that names it. Many pipelines reuse one stage, and the title
// creates thousands of them, so each file is read once. Not cached: a failed read (the file may appear later), and
// anything past the byte budget (read again on each use, as before).
using Container = std::shared_ptr<const std::vector<std::uint8_t>>;
inline constexpr std::size_t kContainerCacheBudget = std::size_t(256) << 20;
inline std::unordered_map<std::string, Container>* g_containers = nullptr;  // under g_lock; null until first use
inline std::size_t g_container_bytes = 0;                                    // under g_lock

inline Container load_container(const std::filesystem::path& dir, const std::string& relative) {
    AcquireSRWLockExclusive(&g_lock);
    if (g_containers) {
        const auto hit = g_containers->find(relative);
        if (hit != g_containers->end()) {
            Container found = hit->second;
            ReleaseSRWLockExclusive(&g_lock);
            return found;
        }
    }
    ReleaseSRWLockExclusive(&g_lock);

    auto bytes = std::make_shared<std::vector<std::uint8_t>>();
    if (!read_container(dir, relative, *bytes)) return nullptr;
    Container loaded = std::move(bytes);

    AcquireSRWLockExclusive(&g_lock);
    if (!g_containers) g_containers = new std::unordered_map<std::string, Container>();
    const auto [slot, inserted] = g_containers->try_emplace(relative, loaded);
    if (!inserted) {
        loaded = slot->second;  // another thread read it first: share its copy
    } else if (g_container_bytes + loaded->size() > kContainerCacheBudget) {
        g_containers->erase(slot);
    } else {
        g_container_bytes += loaded->size();
    }
    ReleaseSRWLockExclusive(&g_lock);
    return loaded;
}

inline bool log_this(std::uint64_t n) { return (n & (n - 1)) == 0; }

// The console's own formats past the PC list, as the device proxy maps them for resources and views (pc_format in
// d3d12_xs_device.cpp): the D16_UNORM_S8_UINT family 118/119/120 is made in the PC D24S8 family 45/46/47. A native PSO
// naming DSV format 118 was refused by the device (retail boot 29).
inline DXGI_FORMAT native_pc_format(unsigned format) {
    switch (format) {
        case 118: return DXGI_FORMAT_D24_UNORM_S8_UINT;
        case 119: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
        case 120: return DXGI_FORMAT_X24_TYPELESS_G8_UINT;
        default: return static_cast<DXGI_FORMAT>(format);
    }
}

// Blend and culling off; a depth target is written with ALWAYS (blob A is not decoded).
inline void fill_fixed_state(const np::FixedState& s, D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc) {
    for (auto& rt : pc.BlendState.RenderTarget) rt.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    pc.SampleMask = s.sample_mask ? s.sample_mask : UINT_MAX;
    pc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    pc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pc.RasterizerState.DepthClipEnable = TRUE;
    pc.DSVFormat = native_pc_format(s.dsv_format);
    pc.DepthStencilState.DepthEnable = pc.DSVFormat != DXGI_FORMAT_UNKNOWN;
    pc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    pc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_ALWAYS;
    pc.PrimitiveTopologyType = s.topology_type ? static_cast<D3D12_PRIMITIVE_TOPOLOGY_TYPE>(s.topology_type)
                                               : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pc.NumRenderTargets = s.rt_count;
    for (unsigned i = 0; i < s.rt_count; ++i) pc.RTVFormats[i] = native_pc_format(s.rtv_formats[i]);
    pc.SampleDesc.Count = s.sample_count;
}

// Blob A's registers in D3D12 terms (field layouts from AMD's RDNA2 register reference / Mesa's gfx10 tables). Before
// this every native pipeline drew with blending and culling off and depth ALWAYS: the Scaleform text pipelines wrote
// each glyph quad's whole colour (letters as solid blocks, or gone where the quad's colour was cleared later), and
// every translucent UI layer was opaque. LEONIDA_NATIVE_STATE=0 keeps the old fixed state; =blend applies only
// blending and the write masks.
namespace native_state {

inline D3D12_BLEND blend_factor(std::uint32_t f, bool alpha) {
    switch (f) {
        case 0: return D3D12_BLEND_ZERO;
        case 1: return D3D12_BLEND_ONE;
        case 2: return alpha ? D3D12_BLEND_SRC_ALPHA : D3D12_BLEND_SRC_COLOR;
        case 3: return alpha ? D3D12_BLEND_INV_SRC_ALPHA : D3D12_BLEND_INV_SRC_COLOR;
        case 4: return D3D12_BLEND_SRC_ALPHA;
        case 5: return D3D12_BLEND_INV_SRC_ALPHA;
        case 6: return D3D12_BLEND_DEST_ALPHA;
        case 7: return D3D12_BLEND_INV_DEST_ALPHA;
        case 8: return alpha ? D3D12_BLEND_DEST_ALPHA : D3D12_BLEND_DEST_COLOR;
        case 9: return alpha ? D3D12_BLEND_INV_DEST_ALPHA : D3D12_BLEND_INV_DEST_COLOR;
        case 10: return D3D12_BLEND_SRC_ALPHA_SAT;
        case 13: return alpha ? D3D12_BLEND_ALPHA_FACTOR : D3D12_BLEND_BLEND_FACTOR;
        case 14: return alpha ? D3D12_BLEND_INV_ALPHA_FACTOR : D3D12_BLEND_INV_BLEND_FACTOR;
        case 15: return alpha ? D3D12_BLEND_SRC1_ALPHA : D3D12_BLEND_SRC1_COLOR;
        case 16: return alpha ? D3D12_BLEND_INV_SRC1_ALPHA : D3D12_BLEND_INV_SRC1_COLOR;
        case 17: return D3D12_BLEND_SRC1_ALPHA;
        case 18: return D3D12_BLEND_INV_SRC1_ALPHA;
        case 19: return D3D12_BLEND_ALPHA_FACTOR;
        case 20: return D3D12_BLEND_INV_ALPHA_FACTOR;
        default: return D3D12_BLEND_ONE;
    }
}
inline bool dual_source(std::uint32_t f) { return f >= 15 && f <= 18; }
inline D3D12_BLEND_OP blend_op(std::uint32_t op) {
    switch (op) {
        case 1: return D3D12_BLEND_OP_SUBTRACT;      // COMB_SRC_MINUS_DST
        case 2: return D3D12_BLEND_OP_MIN;
        case 3: return D3D12_BLEND_OP_MAX;
        case 4: return D3D12_BLEND_OP_REV_SUBTRACT;  // COMB_DST_MINUS_SRC
        default: return D3D12_BLEND_OP_ADD;          // COMB_DST_PLUS_SRC
    }
}
inline D3D12_STENCIL_OP stencil_op(std::uint32_t op) {
    switch (op) {
        case 1: return D3D12_STENCIL_OP_ZERO;
        case 2: case 3: case 4: return D3D12_STENCIL_OP_REPLACE;  // ONES, REPLACE_TEST, REPLACE_OP
        case 5: return D3D12_STENCIL_OP_INCR_SAT;                 // ADD_CLAMP
        case 6: return D3D12_STENCIL_OP_DECR_SAT;                 // SUB_CLAMP
        case 7: return D3D12_STENCIL_OP_INVERT;
        case 8: return D3D12_STENCIL_OP_INCR;                     // ADD_WRAP
        case 9: return D3D12_STENCIL_OP_DECR;                     // SUB_WRAP
        default: return D3D12_STENCIL_OP_KEEP;
    }
}
inline D3D12_COMPARISON_FUNC compare(std::uint32_t f) { return static_cast<D3D12_COMPARISON_FUNC>((f & 7) + 1); }

inline const char* mode() {
    static const std::string* const m = new std::string([] {
        const char* v = std::getenv("LEONIDA_NATIVE_STATE");
        return v ? v : "1";
    }());
    return m->c_str();
}

inline void apply(const np::StateA& a, D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc) {
    const char* m = mode();
    if (m[0] == '0') return;
    // Blend and write masks, per target (CB_BLENDn_CONTROL: SRCBLEND 0-4, COMB_FCN 5-7, DESTBLEND 8-12, alpha
    // 16-20 / 21-23 / 24-28, SEPARATE_ALPHA_BLEND 29, ENABLE 30; CB_TARGET_MASK: 4 bits a target, R G B A as D3D12).
    D3D12_BLEND_DESC& bd = pc.BlendState;
    bool differ = false;
    for (unsigned i = 0; i < 8; ++i) {
        D3D12_RENDER_TARGET_BLEND_DESC& rt = bd.RenderTarget[i];
        const std::uint32_t c = a.blend[i];
        rt.RenderTargetWriteMask = static_cast<UINT8>((a.target_mask >> (4 * i)) & 0xF);
        rt.BlendEnable = (c >> 30) & 1;
        rt.LogicOpEnable = FALSE;
        rt.LogicOp = D3D12_LOGIC_OP_NOOP;
        const bool separate = (c >> 29) & 1;
        rt.SrcBlend = blend_factor(c & 0x1F, false);
        rt.BlendOp = blend_op((c >> 5) & 7);
        rt.DestBlend = blend_factor((c >> 8) & 0x1F, false);
        rt.SrcBlendAlpha = blend_factor(separate ? (c >> 16) & 0x1F : c & 0x1F, true);
        rt.BlendOpAlpha = blend_op(separate ? (c >> 21) & 7 : (c >> 5) & 7);
        rt.DestBlendAlpha = blend_factor(separate ? (c >> 24) & 0x1F : (c >> 8) & 0x1F, true);
        if (!rt.BlendEnable) {
            rt.SrcBlend = rt.SrcBlendAlpha = D3D12_BLEND_ONE;
            rt.DestBlend = rt.DestBlendAlpha = D3D12_BLEND_ZERO;
            rt.BlendOp = rt.BlendOpAlpha = D3D12_BLEND_OP_ADD;
        }
        if (i < pc.NumRenderTargets && i > 0 &&
            std::memcmp(&rt, &bd.RenderTarget[0], sizeof(rt)) != 0)
            differ = true;
    }
    // A dual-source factor on target 0: D3D12 then allows no independent blending; targets 1-7 take the default
    // (five-star-recomp's DesktopStateRules).
    const std::uint32_t c0 = a.blend[0];
    const bool dual = ((c0 >> 30) & 1) && (dual_source(c0 & 0x1F) || dual_source((c0 >> 8) & 0x1F) ||
                                          dual_source((c0 >> 16) & 0x1F) || dual_source((c0 >> 24) & 0x1F));
    if (dual) {
        for (unsigned i = 1; i < 8; ++i) bd.RenderTarget[i] = D3D12_RENDER_TARGET_BLEND_DESC{
            FALSE, FALSE, D3D12_BLEND_ONE, D3D12_BLEND_ZERO, D3D12_BLEND_OP_ADD, D3D12_BLEND_ONE, D3D12_BLEND_ZERO,
            D3D12_BLEND_OP_ADD, D3D12_LOGIC_OP_NOOP, bd.RenderTarget[i].RenderTargetWriteMask};
        differ = false;
    }
    bd.IndependentBlendEnable = differ;
    bd.AlphaToCoverageEnable = FALSE;
    if (std::strcmp(m, "blend") == 0) return;

    // Depth and stencil (DB_DEPTH_CONTROL: STENCIL_ENABLE 0, Z_ENABLE 1, Z_WRITE_ENABLE 2, ZFUNC 4-6, BACKFACE_ENABLE
    // 7, STENCILFUNC 8-10, STENCILFUNC_BF 20-22; DB_STENCIL_CONTROL: FAIL 0-3, ZPASS 4-7, ZFAIL 8-11, then the back
    // face's at 12-23; DB_STENCILREFMASK: MASK 8-15, WRITEMASK 16-23).
    const bool has_depth = pc.DSVFormat != DXGI_FORMAT_UNKNOWN;
    D3D12_DEPTH_STENCIL_DESC& ds = pc.DepthStencilState;
    const std::uint32_t dc = a.depth_control;
    ds.DepthEnable = has_depth && ((dc >> 1) & 1);
    ds.DepthWriteMask = (dc >> 2) & 1 ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;
    ds.DepthFunc = compare(dc >> 4);
    ds.StencilEnable = has_depth && (dc & 1);
    ds.StencilReadMask = static_cast<UINT8>((a.stencil_ref >> 8) & 0xFF);
    ds.StencilWriteMask = static_cast<UINT8>((a.stencil_ref >> 16) & 0xFF);
    const std::uint32_t sc = a.stencil_control;
    ds.FrontFace = {stencil_op(sc & 0xF), stencil_op((sc >> 8) & 0xF), stencil_op((sc >> 4) & 0xF), compare(dc >> 8)};
    ds.BackFace = (dc >> 7) & 1 ? D3D12_DEPTH_STENCILOP_DESC{stencil_op((sc >> 12) & 0xF), stencil_op((sc >> 20) & 0xF),
                                                             stencil_op((sc >> 16) & 0xF), compare(dc >> 20)}
                                : ds.FrontFace;

    // Rasterizer (PA_SU_SC_MODE_CNTL: CULL_FRONT 0, CULL_BACK 1, FACE 2 (1: clockwise is front), POLY_OFFSET_FRONT_ENABLE
    // 11; PA_CL_CLIP_CNTL: ZCLIP_NEAR_DISABLE 26, ZCLIP_FAR_DISABLE 27). Depth bias in AMD's units: slope = scale / 16.
    D3D12_RASTERIZER_DESC& rs = pc.RasterizerState;
    const std::uint32_t su = a.su_sc_mode;
    const bool cull_front = su & 1, cull_back = (su >> 1) & 1;
    rs.CullMode = cull_front && !cull_back ? D3D12_CULL_MODE_FRONT
                  : cull_back && !cull_front ? D3D12_CULL_MODE_BACK
                                             : D3D12_CULL_MODE_NONE;
    rs.FrontCounterClockwise = ((su >> 2) & 1) == 0;
    rs.DepthClipEnable = ((a.clip_cntl >> 26) & 3) == 0;
    if ((su >> 11) & 1) {
        rs.DepthBias = static_cast<INT>(a.poly_offset);
        rs.SlopeScaledDepthBias = a.poly_scale / 16.0f;
        rs.DepthBiasClamp = a.poly_clamp;
    }
    // A NaN in a float field would make the device refuse the pipeline (five-star-recomp's rule): zero instead.
    if (rs.SlopeScaledDepthBias != rs.SlopeScaledDepthBias) rs.SlopeScaledDepthBias = 0;
    if (rs.DepthBiasClamp != rs.DepthBiasClamp) rs.DepthBiasClamp = 0;
}

}  // namespace native_state

// A pipeline state stream: each subobject is a type word, then its payload at its own alignment, padded to 8 bytes
// (d3dx12's alignas(void*) subobjects).
class Stream {
public:
    template <typename P>
    void add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE type, const P& payload) {
        const std::size_t at = size_;
        const std::size_t offset = (at + sizeof(type) + alignof(P) - 1) & ~(alignof(P) - 1);
        size_ = (offset + sizeof(P) + 7) & ~std::size_t(7);
        words_.resize(size_ / 8);
        auto* bytes = reinterpret_cast<unsigned char*>(words_.data());
        std::memcpy(bytes + at, &type, sizeof(type));
        std::memcpy(bytes + offset, &payload, sizeof(P));
    }
    D3D12_PIPELINE_STATE_STREAM_DESC desc() { return {size_, words_.data()}; }

private:
    std::vector<std::uint64_t> words_;
    std::size_t size_ = 0;
};

// The stream of a graphics description; an MS stage makes a mesh pipeline (no input layout or topology).
inline void build_stream(const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc, D3D12_SHADER_BYTECODE ms, Stream& s) {
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, pc.pRootSignature);
    const bool mesh = ms.pShaderBytecode != nullptr;
    if (mesh) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, ms);
    if (pc.VS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, pc.VS);
    if (pc.GS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS, pc.GS);
    if (pc.PS.pShaderBytecode) s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, pc.PS);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, pc.BlendState);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, pc.SampleMask);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, pc.RasterizerState);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, pc.DepthStencilState);
    if (!mesh) {
        s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, pc.InputLayout);
        s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, pc.PrimitiveTopologyType);
    }
    D3D12_RT_FORMAT_ARRAY rtvs{};
    rtvs.NumRenderTargets = pc.NumRenderTargets;
    for (unsigned i = 0; i < 8; ++i) rtvs.RTFormats[i] = pc.RTVFormats[i];
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, rtvs);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, pc.DSVFormat);
    s.add(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC, pc.SampleDesc);
}

inline HRESULT create_from(PipelineStreamCreator create, void* context, const D3D12_GRAPHICS_PIPELINE_STATE_DESC& pc,
                           D3D12_SHADER_BYTECODE ms, const IID& riid, void** out) {
    Stream s;
    build_stream(pc, ms, s);
    const D3D12_PIPELINE_STATE_STREAM_DESC desc = s.desc();
    return create(context, &desc, &riid, out);
}

inline const char* status_name(np::PlanStatus s) {
    switch (s) {
        case np::PlanStatus::ok: return "ok";
        case np::PlanStatus::malformed: return "malformed description";
        case np::PlanStatus::missing: return "no translated container for the stage's digest";
        case np::PlanStatus::ambiguous: return "the stage's records name two containers";
        case np::PlanStatus::unsupported: return "unsupported stage set";
    }
    return "?";
}

// A bare FSSM blob that no index names: the device's own "no translated shader" path answers it.
inline constexpr std::uint8_t kStubProgram[16] = {'F', 'S', 'S', 'M'};

// LEONIDA_NATIVE_PSO_DEBUG=N: the layout of the first N descriptions (sizes and offsets only, no contents).
inline std::atomic<std::uint64_t> g_debugged{0};

inline void debug_hits(const char* what, std::span<const std::uint8_t> b, const np::HashMap& map) {
    std::string hits, tags;
    for (std::size_t at = 0; at + 16 <= b.size(); at += 4) {
        np::Digest d;
        std::memcpy(d.data(), b.data() + at, 16);
        if (const np::ShaderRef* r = map.find(d)) hits += " " + std::to_string(at) + ":" + r->stage;
    }
    for (std::size_t at = 0; at + 4 <= b.size(); at += 4)
        if (np::read_u32(b, at) == 0x43) tags += " " + std::to_string(at);
    std::fprintf(stderr, "[rt] d3d12_xs: slot 57 debug:   %s %zu B, 0x43 words at%s, digest hits at%s\n", what,
                 b.size(), tags.empty() ? " -" : tags.c_str(), hits.empty() ? " -" : hits.c_str());
}

inline void debug_layout(const np::GraphicsDesc& desc, const np::HashMap* map, std::uint64_t call) {
    wchar_t value[8] = {};
    const DWORD n = GetEnvironmentVariableW(L"LEONIDA_NATIVE_PSO_DEBUG", value, 8);
    if (n == 0 || n >= 8) return;
    const std::uint64_t limit = std::wcstoull(value, nullptr, 10);
    if (g_debugged.fetch_add(1) >= limit) return;
    const auto* words = reinterpret_cast<const std::uint64_t*>(&desc);
    std::string pairs;
    for (std::size_t i = 1; i < sizeof(desc) / 8; i += 2)
        pairs += " " + std::to_string(i * 8) + ":" + (words[i] ? "p" : "0") + "/" + std::to_string(words[i + 1]);
    std::fprintf(stderr, "[rt] d3d12_xs: slot 57 debug call %llu: offset:pointer/size%s\n",
                 static_cast<unsigned long long>(call), pairs.c_str());
    if (const DWORD dn = GetEnvironmentVariableW(L"LEONIDA_NATIVE_PSO_DUMP", nullptr, 0)) {
        std::wstring dir(dn, L'\0');
        dir.resize(GetEnvironmentVariableW(L"LEONIDA_NATIVE_PSO_DUMP", dir.data(), dn));
        const auto dump = [&](const std::wstring& name, const np::Pair& pair) {
            if (np::pair_state(pair) != np::PairState::present) return;
            std::ofstream(std::filesystem::path(dir) / name, std::ios::binary)
                .write(static_cast<const char*>(pair.data), static_cast<std::streamsize>(pair.size));
        };
        const std::wstring prefix = std::to_wstring(call) + L"_";
        dump(prefix + L"a.bin", desc.state_a);
        dump(prefix + L"b.bin", desc.state_b);
        for (std::size_t k = 0; k < np::kStageBlocks; ++k) {
            dump(prefix + std::to_wstring(k) + L"_code.bin", desc.stages[k].code);
            dump(prefix + std::to_wstring(k) + L"_rec.bin", desc.stages[k].records);
        }
    }
    if (!map) return;
    const np::HashMap& m = *map;
    if (np::pair_state(desc.state_a) == np::PairState::present) debug_hits("blob A", np::bytes(desc.state_a), m);
    if (np::pair_state(desc.state_b) == np::PairState::present) debug_hits("blob B", np::bytes(desc.state_b), m);
    for (std::size_t k = 0; k < np::kStageBlocks; ++k) {
        const std::string tag = "block " + std::to_string(k);
        if (np::pair_state(desc.stages[k].code) == np::PairState::present)
            debug_hits((tag + " code").c_str(), np::bytes(desc.stages[k].code), m);
        if (np::pair_state(desc.stages[k].records) == np::PairState::present)
            debug_hits((tag + " records").c_str(), np::bytes(desc.stages[k].records), m);
    }
}

// The device's answer: failures are logged and counted, an odd pointer is refused, successes are counted in `done`.
inline HRESULT finish(HRESULT hr, void** out, std::uint64_t call, const char* what, std::atomic<std::uint64_t>& done) {
    if (FAILED(hr)) {
        const std::uint64_t n = g_refused.fetch_add(1) + 1;
        if (log_this(n))
            std::fprintf(stderr, "[rt] d3d12_xs: slot 57 call %llu: the device refused 0x%08lx (%s) (%llu so far)\n",
                         static_cast<unsigned long long>(call), static_cast<unsigned long>(hr), what,
                         static_cast<unsigned long long>(n));
        return hr;
    }
    // The title keeps the pipeline shifted right by one (issue #7496): an odd pointer would lose its low bit.
    if (*out && (reinterpret_cast<std::uintptr_t>(*out) & 1)) {
        if (!g_odd_logged.exchange(true)) std::fprintf(stderr, "[rt] d3d12_xs: slot 57: odd pipeline pointer\n");
        static_cast<IUnknown*>(*out)->Release();
        *out = nullptr;
        g_refused.fetch_add(1);
        return E_FAIL;
    }
    done.fetch_add(1);
    return hr;
}

inline HRESULT unresolved(PipelineStreamCreator create, void* context, const np::GraphicsDesc& desc,
                          const np::FixedState* state, const IID& riid, void** out, std::uint64_t call,
                          const char* why, std::size_t block) {
    const bool stub = env_flag(L"LEONIDA_STUB_PIPELINES", false);
    static_assert(sizeof(kStubProgram) == 16);
    const std::uint64_t n = stub ? g_stub_asks.fetch_add(1) + 1 : g_refused.fetch_add(1) + 1;
    if (log_this(n))
        std::fprintf(stderr, "[rt] d3d12_xs: slot 57: %s (block %zu); %s (%llu so far)\n", why, block,
                     stub ? "asking the device for a stub" : "refused with E_NOTIMPL",
                     static_cast<unsigned long long>(n));
    if (!stub) return E_NOTIMPL;
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pc{};
    pc.pRootSignature = static_cast<ID3D12RootSignature*>(const_cast<void*>(desc.root_signature));
    fill_fixed_state(state ? *state : np::FixedState{}, pc);
    pc.VS = {kStubProgram, sizeof(kStubProgram)};
    return finish(create_from(create, context, pc, {}, riid, out), out, call, "stub", g_stubbed);
}

}  // namespace native_pso_detail

inline void set_native_shader_directory_provider(ShaderDirectoryProvider provider) {
    using namespace native_pso_detail;
    g_provider.store(provider);
}

inline NativePsoStats native_pso_stats() {
    using namespace native_pso_detail;
    return {g_calls.load(), g_created.load(), g_stubbed.load(), g_refused.load()};
}

// Test helper: forgets the loaded map, the cached containers and the counters, so the next call reads the provider again.
inline void reset_native_pso_state() {
    using namespace native_pso_detail;
    AcquireSRWLockExclusive(&g_lock);
    delete g_map;
    delete g_dir;
    delete g_containers;
    g_containers = nullptr;
    g_container_bytes = 0;
    g_map = nullptr;
    g_dir = nullptr;
    g_loaded = false;
    ReleaseSRWLockExclusive(&g_lock);
    g_calls = g_created = g_stubbed = g_refused = g_stub_asks = g_malformed = 0;
}

// slot 57 (desc, riid, out); `create` makes the pipeline from its stream on the device behind the proxy.
inline HRESULT create_native_graphics_pipeline(PipelineStreamCreator create, void* context, const void* desc_words,
                                               const IID* riid, void** out) {
    using namespace native_pso_detail;
    const std::uint64_t call = g_calls.fetch_add(1) + 1;
    if (desc_words && !g_first_logged.exchange(true)) {
        const auto& d = *static_cast<const np::GraphicsDesc*>(desc_words);
        std::fprintf(stderr, "[rt] d3d12_xs: slot 57 first call: blob A %llu B, blob B %llu B, riid %p, out %p\n",
                     static_cast<unsigned long long>(d.state_a.size), static_cast<unsigned long long>(d.state_b.size),
                     static_cast<const void*>(riid), static_cast<void*>(out));
    }
    if (!out) return E_POINTER;
    *out = nullptr;
    if (!riid) return E_POINTER;
    if (!desc_words) return E_INVALIDARG;
    const auto& desc = *static_cast<const np::GraphicsDesc*>(desc_words);
    if (!env_flag(L"LEONIDA_NATIVE_PSO", true)) {
        g_refused.fetch_add(1);
        return E_NOTIMPL;
    }

    const std::filesystem::path* dir = nullptr;
    const np::HashMap* map = hash_map(&dir);
    const np::HashMap empty;
    debug_layout(desc, map, call);
    const np::Plan plan = np::plan(desc, map ? *map : empty);
    if (plan.status == np::PlanStatus::malformed) {
        g_refused.fetch_add(1);
        const std::uint64_t n = g_malformed.fetch_add(1) + 1;
        if (log_this(n))
            std::fprintf(stderr, "[rt] d3d12_xs: slot 57 call %llu: malformed description: %s (block %zu, blob B "
                         "%llu B); E_INVALIDARG (%llu so far)\n", static_cast<unsigned long long>(call), plan.reason,
                         plan.failed_block, static_cast<unsigned long long>(desc.state_b.size),
                         static_cast<unsigned long long>(n));
        return E_INVALIDARG;
    }
    if (plan.status != np::PlanStatus::ok || !dir)
        return unresolved(create, context, desc, &plan.state, *riid, out, call,
                          map ? status_name(plan.status) : "no native_hashes.txt", plan.failed_block);

    Container code[np::kStageBlocks];  // keeps each stage's bytes alive until the device has the pipeline
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pc{};
    D3D12_SHADER_BYTECODE ms{};
    for (const np::PlannedStage& s : plan.stages) {
        code[s.block] = load_container(*dir, s.shader->path);
        if (!code[s.block])
            return unresolved(create, context, desc, &plan.state, *riid, out, call, "container not readable",
                              s.block);
        const D3D12_SHADER_BYTECODE bc{code[s.block]->data(), code[s.block]->size()};
        const std::string& stage = s.shader->stage;
        (stage == "vs" ? pc.VS : stage == "ps" ? pc.PS : stage == "gs" ? pc.GS : ms) = bc;
    }
    pc.pRootSignature = static_cast<ID3D12RootSignature*>(const_cast<void*>(desc.root_signature));
    fill_fixed_state(plan.state, pc);
    {
        const char* why = nullptr;
        if (const auto a = np::decode_state_a(np::bytes(desc.state_a), &why)) {
            native_state::apply(*a, pc);
        } else {
            static std::atomic<int> told{0};
            if (told.fetch_add(1) < 4)
                std::fprintf(stderr, "[rt] d3d12_xs: slot 57 blob A not decoded (%s): blending and culling off\n",
                             why ? why : "?");
        }
    }
    std::vector<D3D12_INPUT_ELEMENT_DESC> elements;
    for (const np::InputElement& e : plan.state.elements)
        elements.push_back({e.name.c_str(), e.semantic_index, static_cast<DXGI_FORMAT>(e.format), e.input_slot,
                            e.aligned_byte_offset, static_cast<D3D12_INPUT_CLASSIFICATION>(e.classification),
                            e.step_rate});
    pc.InputLayout = {elements.data(), static_cast<UINT>(elements.size())};

    // The kept stream (slot 55) references these containers instead of copying them (SharedBytecodeScope).
    std::vector<Container> shared;
    for (const Container& c : code)
        if (c) shared.push_back(c);
    const SharedBytecodeScope scope(std::move(shared));
    const HRESULT hr = create_from(create, context, pc, ms, *riid, out);
    return finish(hr, out, call, plan.stages.front().shader->path.c_str(), g_created);
}

}  // namespace rt::d3d12_xs
