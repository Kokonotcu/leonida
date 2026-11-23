// Differential clear fuzzer: one seeded random program of ClearRenderTargetView, ClearDepthStencilView and
// ClearUnorderedAccessViewUint/Float over one texture (typed views of a typeless resource among them; mips, array
// slices and rects), filled first with known bytes, runs on the layer and on WARP, and every byte of every
// subresource (every plane of a depth-stencil format) is read back and compared. The reference is WARP, as for the
// copy fuzzer (#5272): its clears follow the D3D12 conversion rules, where a GPU's own driver may not.
// Formats and views either device can't use for an operation (CheckFeatureSupport) are not generated, so a program
// depends on the seed and on both devices' format support; --verbose prints what was left out.
// --seeds N --start S choose the programs (default 40 from 1); --hardware compares with the GPU's D3D12 instead;
// --reduce SEED drops clears and rects while the layer still differs and prints the smallest program left.
// --all also generates the known differences the default leaves out (see Avoid).
// Needs a Vulkan GPU (exit 77 otherwise, and under Wine). No game data.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

enum class Kind { color, depth };

struct Family {
    Kind kind;
    DXGI_FORMAT resource;            // what the texture is created with
    std::vector<DXGI_FORMAT> views;  // RTV/UAV formats (colour) or the DSV format (depth)
    const char* name;
};

const std::vector<Family>& families() {
    static const std::vector<Family> list = {
        {Kind::color, DXGI_FORMAT_R8G8B8A8_TYPELESS,
         {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_R8G8B8A8_SNORM,
          DXGI_FORMAT_R8G8B8A8_UINT, DXGI_FORMAT_R8G8B8A8_SINT},
         "R8G8B8A8_TYPELESS"},
        {Kind::color, DXGI_FORMAT_B8G8R8A8_TYPELESS, {DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB},
         "B8G8R8A8_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R16G16B16A16_TYPELESS,
         {DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_R16G16B16A16_SNORM,
          DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SINT},
         "R16G16B16A16_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R16G16_TYPELESS,
         {DXGI_FORMAT_R16G16_FLOAT, DXGI_FORMAT_R16G16_UNORM, DXGI_FORMAT_R16G16_SNORM, DXGI_FORMAT_R16G16_UINT},
         "R16G16_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R32_TYPELESS, {DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R32_UINT, DXGI_FORMAT_R32_SINT},
         "R32_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R32G32B32A32_TYPELESS,
         {DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_R32G32B32A32_UINT, DXGI_FORMAT_R32G32B32A32_SINT},
         "R32G32B32A32_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R10G10B10A2_TYPELESS, {DXGI_FORMAT_R10G10B10A2_UNORM, DXGI_FORMAT_R10G10B10A2_UINT},
         "R10G10B10A2_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R8_TYPELESS,
         {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8_SNORM, DXGI_FORMAT_R8_UINT, DXGI_FORMAT_R8_SINT}, "R8_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R8G8_TYPELESS, {DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8_SNORM, DXGI_FORMAT_R8G8_UINT},
         "R8G8_TYPELESS"},
        {Kind::color, DXGI_FORMAT_R11G11B10_FLOAT, {DXGI_FORMAT_R11G11B10_FLOAT}, "R11G11B10_FLOAT"},
        {Kind::color, DXGI_FORMAT_B5G6R5_UNORM, {DXGI_FORMAT_B5G6R5_UNORM}, "B5G6R5_UNORM"},
        {Kind::color, DXGI_FORMAT_B5G5R5A1_UNORM, {DXGI_FORMAT_B5G5R5A1_UNORM}, "B5G5R5A1_UNORM"},
        {Kind::color, DXGI_FORMAT_B4G4R4A4_UNORM, {DXGI_FORMAT_B4G4R4A4_UNORM}, "B4G4R4A4_UNORM"},
        {Kind::depth, DXGI_FORMAT_D16_UNORM, {DXGI_FORMAT_D16_UNORM}, "D16_UNORM"},
        {Kind::depth, DXGI_FORMAT_D24_UNORM_S8_UINT, {DXGI_FORMAT_D24_UNORM_S8_UINT}, "D24_UNORM_S8_UINT"},
        {Kind::depth, DXGI_FORMAT_D32_FLOAT, {DXGI_FORMAT_D32_FLOAT}, "D32_FLOAT"},
        {Kind::depth, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, {DXGI_FORMAT_D32_FLOAT_S8X24_UINT}, "D32_FLOAT_S8X24_UINT"},
        {Kind::depth, DXGI_FORMAT_R32_TYPELESS, {DXGI_FORMAT_D32_FLOAT}, "R32_TYPELESS (D32_FLOAT view)"},
        {Kind::depth, DXGI_FORMAT_R24G8_TYPELESS, {DXGI_FORMAT_D24_UNORM_S8_UINT}, "R24G8_TYPELESS (D24S8 view)"},
    };
    return list;
}

const char* format_name(DXGI_FORMAT f) {
    switch (f) {
#define NAME(x) case DXGI_FORMAT_##x: return #x;
        NAME(R8G8B8A8_UNORM) NAME(R8G8B8A8_UNORM_SRGB) NAME(R8G8B8A8_SNORM) NAME(R8G8B8A8_UINT) NAME(R8G8B8A8_SINT)
        NAME(B8G8R8A8_UNORM) NAME(B8G8R8A8_UNORM_SRGB) NAME(R16G16B16A16_FLOAT) NAME(R16G16B16A16_UNORM)
        NAME(R16G16B16A16_SNORM) NAME(R16G16B16A16_UINT) NAME(R16G16B16A16_SINT) NAME(R16G16_FLOAT) NAME(R16G16_UNORM)
        NAME(R16G16_SNORM) NAME(R16G16_UINT) NAME(R32_FLOAT) NAME(R32_UINT) NAME(R32_SINT) NAME(R32G32B32A32_FLOAT)
        NAME(R32G32B32A32_UINT) NAME(R32G32B32A32_SINT) NAME(R10G10B10A2_UNORM) NAME(R10G10B10A2_UINT) NAME(R8_UNORM)
        NAME(R8_SNORM) NAME(R8_UINT) NAME(R8_SINT) NAME(R8G8_UNORM) NAME(R8G8_SNORM) NAME(R8G8_UINT)
        NAME(R11G11B10_FLOAT) NAME(B5G6R5_UNORM) NAME(B5G5R5A1_UNORM) NAME(B4G4R4A4_UNORM) NAME(D16_UNORM)
        NAME(D24_UNORM_S8_UINT) NAME(D32_FLOAT) NAME(D32_FLOAT_S8X24_UINT)
#undef NAME
    default: return "?";
    }
}

// What both devices can do with a view format.
struct Support {
    bool rtv = false, dsv = false, uav = false;
};

Support query(ID3D12Device* device, DXGI_FORMAT format) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{format};
    Support s;
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof fs))) return s;
    s.rtv = (fs.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) != 0;
    s.dsv = (fs.Support1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) != 0;
    s.uav = (fs.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
            (fs.Support2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE);
    return s;
}

// The families and views a program may use: supported by both devices for at least one clear.
struct Usable {
    const Family* family;
    std::vector<DXGI_FORMAT> rtv, uav;  // colour
    bool dsv = false;                   // depth
};

std::vector<Usable> usable(ID3D12Device* a, ID3D12Device* b, bool verbose) {
    std::vector<Usable> out;
    for (const Family& f : families()) {
        Usable u{&f};
        for (DXGI_FORMAT v : f.views) {
            const Support sa = query(a, v), sb = query(b, v);
            if (f.kind == Kind::depth) {
                u.dsv = sa.dsv && sb.dsv;
                continue;
            }
            if (sa.rtv && sb.rtv) u.rtv.push_back(v);
            else if (verbose) std::printf("  left out: %s render-target clears (not on both devices)\n", format_name(v));
            if (sa.uav && sb.uav) u.uav.push_back(v);
        }
        if (u.dsv || !u.rtv.empty() || !u.uav.empty()) out.push_back(u);
        else if (verbose) std::printf("  left out: %s\n", f.name);
    }
    return out;
}

struct Shape {
    UINT width, height, layers, mips;
};

struct Clear {
    enum Op { rtv, dsv, uav_uint, uav_float } op;
    DXGI_FORMAT view;
    UINT mip = 0, first_layer = 0, layers = 1;
    float color[4]{};
    UINT values[4]{};
    D3D12_CLEAR_FLAGS flags = D3D12_CLEAR_FLAG_DEPTH;
    float depth = 0;
    UINT8 stencil = 0;
    std::vector<D3D12_RECT> rects;
};

struct Program {
    const Usable* use = nullptr;
    Shape shape{};
    bool uav_flag = false;
    std::vector<Clear> clears;
};

UINT mip_extent(UINT size, UINT mip) { return std::max(1u, size >> mip); }

bool has_stencil(DXGI_FORMAT f) {
    return f == DXGI_FORMAT_D24_UNORM_S8_UINT || f == DXGI_FORMAT_D32_FLOAT_S8X24_UINT ||
           f == DXGI_FORMAT_R24G8_TYPELESS;
}

// A clear colour component: in range, out of range, special values, integers for integer views.
float component(std::mt19937_64& rng, DXGI_FORMAT view) {
    const auto unit = [&] { return float(rng() % 1000001) / 1000000.0f; };
    const bool integer = std::strstr(format_name(view), "INT") && !std::strstr(format_name(view), "NORM");
    switch (rng() % (integer ? 6 : 10)) {
    case 0: return 0.0f;
    case 1: return 1.0f;
    case 2: return float(int(rng() % 600) - 300);  // whole numbers, also past an 8-bit range
    case 3: return float(std::int64_t(rng() % 5000000000ull) - 2500000000ll);
    case 4: return float(int(rng() % 512) - 256) + 0.5f;  // halves: the rounding of a conversion
    case 5: return unit() * 4.0f - 2.0f;
    case 6: return -0.0f;
    case 7: return unit();
    case 8: {
        constexpr float special[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                     -std::numeric_limits<float>::infinity(), 1e-40f, -1e-40f, 65504.0f, 65520.0f,
                                     1e30f, 0.5f / 255.0f, 1.5f / 255.0f, 0.0031308f, 0.04045f};
        return special[rng() % std::size(special)];
    }
    default: return unit() * 2.0f - 1.0f;
    }
}

// What the default programs leave out (--all puts it back), each a known difference with its own entry:
// - `documented`: ClearUnorderedAccessViewUint bit patterns that runtime/src/uav_clear.h documents as inexact on a
//   texture that isn't typeless (R11G11B10_FLOAT here): float NaN patterns, whose payload a driver canonicalises. Such
//   a channel gets its top value bit cleared instead.
// - `nvidia_stencil`: a stencil clear with rects on a mip past 0. NVIDIA's Vulkan driver (621.14) writes the stencil of
//   a depth-stencil attachment view of mip > 0 into mip 0 (vkCmdClearAttachments and loadOp CLEAR alike); such a
//   clear goes to mip 0 instead.
struct Avoid {
    bool documented = true;
    bool nvidia_stencil = false;
};

// A ClearUnorderedAccessViewUint value whose bits uav_clear.h documents as inexact on `view`'s channel `i`, made exact.
UINT avoid_documented(DXGI_FORMAT view, UINT i, UINT value) {
    const std::string name = format_name(view);
    if (name == "R11G11B10_FLOAT") {
        if (i > 2) return value;
        const UINT mantissa = i == 2 ? 5 : 6, exponent = 0x1fu << mantissa, m = (1u << mantissa) - 1;
        return (value & exponent) == exponent && (value & m) ? value & ~(1u << (mantissa + 4)) : value;
    }
    if (name.find("16_FLOAT") != std::string::npos)
        return (value & 0x7c00) == 0x7c00 && (value & 0x3ff) ? value & ~0x4000u : value;
    if (name.find("32_FLOAT") != std::string::npos)
        return (value & 0x7f800000) == 0x7f800000 && (value & 0x7fffff) ? value & ~0x40000000u : value;
    if (name.find("SNORM") != std::string::npos) {
        const UINT bits = name.find("16") != std::string::npos ? 16 : 8;
        const UINT field = value & ((1u << bits) - 1);
        return field == (1u << (bits - 1)) ? value ^ 1u : value;  // 0x80.. -> 0x80..01
    }
    return value;
}

Program generate(std::uint64_t seed, const std::vector<Usable>& use, const Avoid& avoid) {
    std::mt19937_64 rng(seed * 0x9e3779b97f4a7c15ull + 0x1234567);
    const auto pick = [&](UINT n) { return UINT(rng() % n); };
    Program p;
    p.use = &use[pick(UINT(use.size()))];
    const bool depth = p.use->family->kind == Kind::depth;
    Shape& s = p.shape;
    s.width = 1 + pick(48);
    s.height = 1 + pick(48);
    s.layers = 1 + pick(4);
    UINT full = 0;
    for (UINT m = std::max(s.width, s.height); m; m >>= 1) ++full;
    s.mips = 1 + pick(full);
    p.uav_flag = !depth && !p.use->uav.empty() && (p.use->rtv.empty() || pick(2));
    const UINT n = 3 + pick(12);
    for (UINT i = 0; i < n; ++i) {
        Clear c{};
        if (depth) {
            c.op = Clear::dsv;
            c.view = p.use->family->views[0];
        } else if (p.uav_flag && (p.use->rtv.empty() || pick(2))) {
            c.op = pick(2) ? Clear::uav_uint : Clear::uav_float;
            c.view = p.use->uav[pick(UINT(p.use->uav.size()))];
        } else {
            c.op = Clear::rtv;
            c.view = p.use->rtv[pick(UINT(p.use->rtv.size()))];
        }
        c.mip = pick(s.mips);
        c.first_layer = pick(s.layers);
        c.layers = 1 + pick(s.layers - c.first_layer);
        for (float& v : c.color) v = component(rng, c.view);
        for (UINT& v : c.values) {
            switch (pick(4)) {
            case 0: v = UINT(rng()); break;
            case 1: v = pick(256); break;
            case 2: v = pick(65536); break;
            default: v = 0xffffffffu - pick(4); break;
            }
        }
        if (c.op == Clear::uav_float)  // float clears of integer views are invalid; keep them to typed float/norm views
            if (std::strstr(format_name(c.view), "INT") && !std::strstr(format_name(c.view), "NORM")) c.op = Clear::uav_uint;
        if (depth) {
            c.flags = has_stencil(p.use->family->resource)
                          ? D3D12_CLEAR_FLAGS(1 + pick(3))  // depth, stencil or both
                          : D3D12_CLEAR_FLAG_DEPTH;
            constexpr float edges[] = {0.0f, 1.0f, 0.5f, 1e-40f, 1.0f / 65535.0f, 1.0f / 16777215.0f, 0.9999999f};
            switch (pick(4)) {
            case 0: c.depth = edges[pick(UINT(std::size(edges)))]; break;
            case 1: c.depth = float((double(pick(65535)) + 0.5) / 65535.0); break;  // a D16 tie, where rounding rules part
            default: c.depth = float(rng() % 1000001) / 1000000.0f; break;
            }
            c.stencil = UINT8(rng());
        }
        // RTV and DSV clears take rects; UAV clears too. Some clears have none (the whole view).
        const UINT w = mip_extent(s.width, c.mip), h = mip_extent(s.height, c.mip);
        const UINT rects = pick(3) ? pick(4) : 0;
        for (UINT r = 0; r < rects; ++r) {
            const LONG x0 = LONG(pick(w)), y0 = LONG(pick(h));
            const LONG x1 = pick(8) ? x0 + 1 + LONG(pick(w - UINT(x0))) : x0;  // a few empty rects
            const LONG y1 = y0 + 1 + LONG(pick(h - UINT(y0)));
            c.rects.push_back({x0, y0, x1, y1});
        }
        // A typeless texture's UINT clears are exact (the replay clears through the family's UINT format); only the
        // others go through a float.
        if (c.op == Clear::uav_uint && avoid.documented && !std::strstr(p.use->family->name, "TYPELESS"))
            for (UINT k = 0; k < 4; ++k) c.values[k] = avoid_documented(c.view, k, c.values[k]);
        if (avoid.nvidia_stencil && c.op == Clear::dsv && (c.flags & D3D12_CLEAR_FLAG_STENCIL) && !c.rects.empty()) {
            c.mip = 0;
            for (D3D12_RECT& r : c.rects) {  // keep the rects inside mip 0 (they are already: mip 0 is the largest)
                r.right = std::min<LONG>(r.right, LONG(s.width));
                r.bottom = std::min<LONG>(r.bottom, LONG(s.height));
            }
        }
        p.clears.push_back(c);
    }
    return p;
}

std::string describe(const Clear& c) {
    char text[512];
    const char* op = c.op == Clear::rtv ? "ClearRenderTargetView" : c.op == Clear::dsv ? "ClearDepthStencilView"
                     : c.op == Clear::uav_uint ? "ClearUnorderedAccessViewUint" : "ClearUnorderedAccessViewFloat";
    int n = std::snprintf(text, sizeof text, "%s(%s mip %u layers %u+%u", op, format_name(c.view), c.mip, c.first_layer,
                          c.layers);
    if (c.op == Clear::rtv || c.op == Clear::uav_float)
        n += std::snprintf(text + n, sizeof text - n, ", %g %g %g %g", c.color[0], c.color[1], c.color[2], c.color[3]);
    else if (c.op == Clear::uav_uint)
        n += std::snprintf(text + n, sizeof text - n, ", 0x%x 0x%x 0x%x 0x%x", c.values[0], c.values[1], c.values[2],
                           c.values[3]);
    else
        n += std::snprintf(text + n, sizeof text - n, ", %s%s depth %.9g stencil %u",
                           (c.flags & D3D12_CLEAR_FLAG_DEPTH) ? "D" : "", (c.flags & D3D12_CLEAR_FLAG_STENCIL) ? "S" : "",
                           c.depth, c.stencil);
    for (const D3D12_RECT& r : c.rects)
        n += std::snprintf(text + n, sizeof text - n, ", [%ld,%ld-%ld,%ld]", r.left, r.top, r.right, r.bottom);
    std::snprintf(text + n, sizeof text - n, ")");
    return text;
}

void print_program(const Program& p) {
    std::printf("  texture %s %ux%u, %u layers, %u mips%s\n", p.use->family->name, p.shape.width, p.shape.height,
                p.shape.layers, p.shape.mips, p.uav_flag ? ", UAV" : "");
    for (const Clear& c : p.clears) std::printf("  %s\n", describe(c).c_str());
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc = {1, 0};
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC texture_desc(const Program& p) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = p.shape.width;
    d.Height = p.shape.height;
    d.DepthOrArraySize = UINT16(p.shape.layers);
    d.MipLevels = UINT16(p.shape.mips);
    d.Format = p.use->family->resource;
    d.SampleDesc = {1, 0};
    if (p.use->family->kind == Kind::depth) d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    else {
        if (!p.use->rtv.empty()) d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        if (p.uav_flag) d.Flags |= D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    }
    return d;
}

std::uint8_t pattern(std::uint64_t seed, UINT64 i) {
    std::uint64_t x = seed * 0x9e3779b97f4a7c15ull + i * 0x94d049bb133111ebull + 0xbf58476d1ce4e5b9ull;
    x ^= x >> 31; x *= 0xd6e8feb86659fd93ull; x ^= x >> 32;
    return std::uint8_t(x);
}

constexpr UINT64 kStagingBytes = 1 << 24;

// One readback region: a plane of a subresource, rows without padding. `mask` blanks bytes whose content D3D12
// leaves undefined (the X8 byte of a D24 depth plane).
struct Region {
    UINT sub, plane;
    UINT64 offset;
    UINT rows;
    UINT64 row_bytes;
    UINT pitch;
    DXGI_FORMAT format;
};

struct Result {
    std::vector<std::uint8_t> bytes;
    std::vector<Region> regions;
    std::vector<size_t> starts;  // where each region's bytes begin
    std::string error;
    bool gap = false;  // the device refused the resource
};

UINT plane_count(DXGI_FORMAT f) { return has_stencil(f) ? 2 : 1; }

Result run(ID3D12Device* device, const Program& p, std::uint64_t seed) {
    Result out;
    const auto fail = [&](const char* why) {
        out.error = why;
        return out;
    };
    D3D12_HEAP_PROPERTIES def{D3D12_HEAP_TYPE_DEFAULT}, up{D3D12_HEAP_TYPE_UPLOAD}, rb{D3D12_HEAP_TYPE_READBACK};
    const bool depth = p.use->family->kind == Kind::depth;
    const auto td = texture_desc(p);
    ComPtr<ID3D12Resource> tex, upload, readback;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COPY_DEST;
    if (FAILED(device->CreateCommittedResource(&def, D3D12_HEAP_FLAG_NONE, &td, state, nullptr, IID_PPV_ARGS(&tex)))) {
        out.gap = true;
        return fail("texture refused");
    }
    const auto sd = buffer_desc(kStagingBytes);
    if (FAILED(device->CreateCommittedResource(&up, D3D12_HEAP_FLAG_NONE, &sd, D3D12_RESOURCE_STATE_GENERIC_READ,
                                               nullptr, IID_PPV_ARGS(&upload))) ||
        FAILED(device->CreateCommittedResource(&rb, D3D12_HEAP_FLAG_NONE, &sd, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                               IID_PPV_ARGS(&readback))))
        return fail("staging");
    D3D12_COMMAND_QUEUE_DESC qd{};
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence))))
        return fail("queue");
    UINT64 fence_value = 0;
    const auto submit = [&]() {
        list->Close();
        ID3D12CommandList* l = list.Get();
        queue->ExecuteCommandLists(1, &l);
        queue->Signal(fence.Get(), ++fence_value);
        HANDLE e = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        fence->SetEventOnCompletion(fence_value, e);
        const bool ok = WaitForSingleObject(e, 20000) == WAIT_OBJECT_0;
        CloseHandle(e);
        alloc->Reset();
        list->Reset(alloc.Get(), nullptr);
        return ok;
    };
    const auto transition = [&](D3D12_RESOURCE_STATES to) {
        if (state == to) return;
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {tex.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state, to};
        list->ResourceBarrier(1, &barrier);
        state = to;
    };

    // Fill every plane of every subresource. Depth planes get valid depths (a float in [0, 1], or any D16/D24 value),
    // since a clear isn't what would differ on a NaN depth.
    const UINT subs = p.shape.mips * p.shape.layers, planes = plane_count(p.use->family->resource);
    const UINT all = subs * planes;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> fp(all);
    std::vector<UINT> rows(all);
    std::vector<UINT64> row_bytes(all);
    UINT64 total = 0;
    device->GetCopyableFootprints(&td, 0, all, 0, fp.data(), rows.data(), row_bytes.data(), &total);
    if (total > kStagingBytes) return fail("staging too small");
    std::uint8_t* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)))) return fail("map upload");
    for (UINT64 i = 0; i < total; ++i) mapped[i] = pattern(seed, i);
    for (UINT s = 0; s < all; ++s) {
        const DXGI_FORMAT f = fp[s].Footprint.Format;
        if (f != DXGI_FORMAT_R32_FLOAT && f != DXGI_FORMAT_R32_TYPELESS && f != DXGI_FORMAT_D32_FLOAT) continue;
        for (UINT r = 0; r < rows[s]; ++r) {
            float* row = reinterpret_cast<float*>(mapped + fp[s].Offset + UINT64(r) * fp[s].Footprint.RowPitch);
            for (UINT x = 0; x < fp[s].Footprint.Width; ++x)
                row[x] = float(pattern(seed, fp[s].Offset + r * 4096 + x) | (pattern(seed, x * 7 + r) << 8)) / 65535.0f;
        }
    }
    upload->Unmap(0, nullptr);
    for (UINT s = 0; s < all; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dst{tex.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        dst.SubresourceIndex = s;
        D3D12_TEXTURE_COPY_LOCATION src{upload.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        src.PlacedFootprint = fp[s];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (!submit()) return fail("fill timed out");

    // Descriptor heaps: one view per clear.
    const UINT n = UINT(p.clears.size());
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.NumDescriptors = n;
    ComPtr<ID3D12DescriptorHeap> rtv_heap, dsv_heap, uav_cpu, uav_gpu;
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&rtv_heap)))) return fail("rtv heap");
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&dsv_heap)))) return fail("dsv heap");
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&uav_cpu)))) return fail("uav heap");
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&uav_gpu)))) return fail("uav heap");
    const UINT rtv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    const UINT dsv_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
    const UINT uav_step = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    ID3D12DescriptorHeap* heaps[] = {uav_gpu.Get()};
    bool uav_before = false;
    for (UINT i = 0; i < n; ++i) {
        const Clear& c = p.clears[i];
        if (c.op == Clear::rtv) {
            D3D12_RENDER_TARGET_VIEW_DESC v{};
            v.Format = c.view;
            v.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            v.Texture2DArray = {c.mip, c.first_layer, c.layers, 0};
            D3D12_CPU_DESCRIPTOR_HANDLE h = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * rtv_step;
            device->CreateRenderTargetView(tex.Get(), &v, h);
            transition(D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ClearRenderTargetView(h, c.color, UINT(c.rects.size()), c.rects.empty() ? nullptr : c.rects.data());
            uav_before = false;
        } else if (c.op == Clear::dsv) {
            D3D12_DEPTH_STENCIL_VIEW_DESC v{};
            v.Format = c.view;
            v.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            v.Texture2DArray = {c.mip, c.first_layer, c.layers};
            D3D12_CPU_DESCRIPTOR_HANDLE h = dsv_heap->GetCPUDescriptorHandleForHeapStart();
            h.ptr += SIZE_T(i) * dsv_step;
            device->CreateDepthStencilView(tex.Get(), &v, h);
            transition(D3D12_RESOURCE_STATE_DEPTH_WRITE);
            list->ClearDepthStencilView(h, c.flags, c.depth, c.stencil, UINT(c.rects.size()),
                                        c.rects.empty() ? nullptr : c.rects.data());
        } else {
            D3D12_UNORDERED_ACCESS_VIEW_DESC v{};
            v.Format = c.view;
            v.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2DARRAY;
            v.Texture2DArray = {c.mip, c.first_layer, c.layers, 0};
            D3D12_CPU_DESCRIPTOR_HANDLE cpu = uav_cpu->GetCPUDescriptorHandleForHeapStart();
            cpu.ptr += SIZE_T(i) * uav_step;
            D3D12_CPU_DESCRIPTOR_HANDLE gpu_cpu = uav_gpu->GetCPUDescriptorHandleForHeapStart();
            gpu_cpu.ptr += SIZE_T(i) * uav_step;
            D3D12_GPU_DESCRIPTOR_HANDLE gpu = uav_gpu->GetGPUDescriptorHandleForHeapStart();
            gpu.ptr += UINT64(i) * uav_step;
            device->CreateUnorderedAccessView(tex.Get(), nullptr, &v, cpu);
            device->CreateUnorderedAccessView(tex.Get(), nullptr, &v, gpu_cpu);
            transition(D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (uav_before) {  // UAV writes are unordered without a UAV barrier
                D3D12_RESOURCE_BARRIER b{};
                b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                b.UAV.pResource = tex.Get();
                list->ResourceBarrier(1, &b);
            }
            list->SetDescriptorHeaps(1, heaps);
            const D3D12_RECT* r = c.rects.empty() ? nullptr : c.rects.data();
            if (c.op == Clear::uav_uint)
                list->ClearUnorderedAccessViewUint(gpu, cpu, tex.Get(), c.values, UINT(c.rects.size()), r);
            else
                list->ClearUnorderedAccessViewFloat(gpu, cpu, tex.Get(), c.color, UINT(c.rects.size()), r);
            uav_before = true;
        }
    }
    if (!submit()) return fail("program timed out");

    // Read back every plane of every subresource at the device's own footprints.
    transition(D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT s = 0; s < all; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dst{readback.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT};
        dst.PlacedFootprint = fp[s];
        D3D12_TEXTURE_COPY_LOCATION src{tex.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX};
        src.SubresourceIndex = s;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        out.regions.push_back({s % subs, s / subs, fp[s].Offset, rows[s], row_bytes[s], fp[s].Footprint.RowPitch,
                               fp[s].Footprint.Format});
    }
    if (!submit()) return fail("readback timed out");
    std::uint8_t* back = nullptr;
    const D3D12_RANGE range{0, SIZE_T(kStagingBytes)};
    if (FAILED(readback->Map(0, &range, reinterpret_cast<void**>(&back)))) return fail("map readback");
    for (const Region& rg : out.regions) {
        out.starts.push_back(out.bytes.size());
        for (UINT row = 0; row < rg.rows; ++row) {
            const std::uint8_t* from = back + rg.offset + UINT64(row) * rg.pitch;
            out.bytes.insert(out.bytes.end(), from, from + rg.row_bytes);
        }
        // The X8 byte of a D24 depth plane is undefined in D3D12.
        if (depth && rg.plane == 0 && (rg.format == DXGI_FORMAT_R24_UNORM_X8_TYPELESS ||
                                       rg.format == DXGI_FORMAT_R24G8_TYPELESS ||
                                       rg.format == DXGI_FORMAT_D24_UNORM_S8_UINT))
            for (size_t i = out.starts.back() + 3; i < out.bytes.size(); i += 4) out.bytes[i] = 0;
    }
    readback->Unmap(0, nullptr);
    return out;
}

// D3D12 gives the float-to-sRGB conversion a tolerance (D3D11.3 3.2.3.7), and the encoded value can sit on a code
// tie (0.349108 encodes to 159.50): where the program clears through an sRGB view, a colour byte (not alpha) of an
// RGBA8/BGRA8 texture may be one code off.
bool srgb_tolerance(const Program& p, const Result& layer, const Result& ref, size_t i) {
    const DXGI_FORMAT family = p.use->family->resource;
    if (family != DXGI_FORMAT_R8G8B8A8_TYPELESS && family != DXGI_FORMAT_B8G8R8A8_TYPELESS) return false;
    bool srgb = false;
    for (const Clear& c : p.clears)
        srgb = srgb || c.view == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || c.view == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    size_t r = 0;
    while (r + 1 < ref.starts.size() && ref.starts[r + 1] <= i) ++r;
    if (!srgb || (i - ref.starts[r]) % 4 == 3) return false;
    const int d = int(layer.bytes[i]) - int(ref.bytes[i]);
    return d == 1 || d == -1;
}

// D3D12 leaves the payload of a NaN that a float clear value converts to unspecified (WARP writes 0xfffd for a half,
// NVIDIA 0xfe00). Unless --all brings back the UINT clears whose bits are a NaN (Avoid::documented), every NaN comes
// from a float clear, and a texel channel that is a NaN on both sides counts as the same. Returns the bytes from `i`
// that are equal that way, or 0.
bool nan_equal = true;

size_t same_nan(const Program& p, const Result& layer, const Result& ref, size_t i) {
    if (!nan_equal) return 0;
    const DXGI_FORMAT family = p.use->family->resource;
    size_t r = 0;
    while (r + 1 < ref.starts.size() && ref.starts[r + 1] <= i) ++r;
    const size_t base = ref.starts[r];
    const auto word = [&](const Result& res, size_t at, size_t bytes) {
        std::uint32_t v = 0;
        std::memcpy(&v, res.bytes.data() + at, bytes);
        return v;
    };
    if (family == DXGI_FORMAT_R16G16B16A16_TYPELESS || family == DXGI_FORMAT_R16G16_TYPELESS) {
        const size_t at = base + (i - base) / 2 * 2;
        const auto nan = [](std::uint32_t h) { return (h & 0x7c00) == 0x7c00 && (h & 0x3ff); };
        return nan(word(layer, at, 2)) && nan(word(ref, at, 2)) ? at + 2 - i : 0;
    }
    if (family == DXGI_FORMAT_R32_TYPELESS || family == DXGI_FORMAT_R32G32B32A32_TYPELESS) {
        const size_t at = base + (i - base) / 4 * 4;
        const auto nan = [](std::uint32_t f) { return (f & 0x7f800000) == 0x7f800000 && (f & 0x7fffff); };
        return nan(word(layer, at, 4)) && nan(word(ref, at, 4)) ? at + 4 - i : 0;
    }
    if (family == DXGI_FORMAT_R11G11B10_FLOAT) {
        const size_t at = base + (i - base) / 4 * 4;
        const std::uint32_t a = word(layer, at, 4), b = word(ref, at, 4);
        constexpr unsigned shift[3] = {0, 11, 22}, mantissa[3] = {6, 6, 5};
        for (int c = 0; c < 3; ++c) {
            const unsigned bits = mantissa[c] + 5;
            const std::uint32_t mask = (1u << bits) - 1, x = (a >> shift[c]) & mask, y = (b >> shift[c]) & mask;
            const std::uint32_t exponent = 0x1fu << mantissa[c], m = (1u << mantissa[c]) - 1;
            const bool both_nan = (x & exponent) == exponent && (x & m) && (y & exponent) == exponent && (y & m);
            if (x != y && !both_nan) return 0;
        }
        return at + 4 - i;
    }
    return 0;
}

// Where two results first differ, and how many bytes differ; empty when equal.
std::string compare(const Program& p, const Result& layer, const Result& ref, size_t& count) {
    count = 0;
    if (layer.bytes.size() != ref.bytes.size()) {
        count = std::max(layer.bytes.size(), ref.bytes.size());
        return "readback sizes differ (" + std::to_string(layer.bytes.size()) + " vs " +
               std::to_string(ref.bytes.size()) + " bytes)";
    }
    std::string first;
    for (size_t i = 0; i < layer.bytes.size(); ++i) {
        if (layer.bytes[i] == ref.bytes[i]) continue;
        if (srgb_tolerance(p, layer, ref, i)) continue;
        if (const size_t skip = same_nan(p, layer, ref, i)) {
            i += skip - 1;
            continue;
        }
        if (count++ == 0) {
            size_t r = 0;
            while (r + 1 < ref.starts.size() && ref.starts[r + 1] <= i) ++r;
            const Region& rg = ref.regions[r];
            const size_t within = i - ref.starts[r];
            const UINT texel = rg.row_bytes && rg.pitch ? 0 : 0;
            (void)texel;
            const UINT y = UINT(within / rg.row_bytes);
            const size_t x_byte = within % rg.row_bytes;
            const UINT mip = rg.sub % p.shape.mips, layer_index = rg.sub / p.shape.mips;
            const size_t texel_bytes = std::max<size_t>(1, rg.row_bytes / std::max(1u, mip_extent(p.shape.width, mip)));
            const size_t t0 = i - (x_byte % texel_bytes);
            char text[512];
            int n = std::snprintf(text, sizeof text, "mip %u layer %u plane %u texel (%zu,%u): layer", mip, layer_index,
                                  rg.plane, x_byte / texel_bytes, y);
            for (size_t k = 0; k < texel_bytes && k < 16; ++k) n += std::snprintf(text + n, sizeof text - n, " %02x", layer.bytes[t0 + k]);
            n += std::snprintf(text + n, sizeof text - n, ", reference");
            for (size_t k = 0; k < texel_bytes && k < 16; ++k) n += std::snprintf(text + n, sizeof text - n, " %02x", ref.bytes[t0 + k]);
            first = text;
        }
    }
    return first;
}

}  // namespace

int main(int argc, char** argv) {
    unsigned seeds = 40, start = 1, reduce = 0;
    bool verbose = false, hardware = false, reducing = false, all = false;
    for (int i = 1; i < argc; ++i) {
        bool bad = false;
        const auto number = [&](unsigned& out) {
            const char* s = i + 1 < argc ? argv[i + 1] : "";
            char* end = nullptr;
            const unsigned long long v = std::strtoull(s, &end, 10);
            if (*s < '0' || *s > '9' || *end || v > 0xffffffffull) {
                bad = true;
                return;
            }
            out = unsigned(v);
            ++i;
        };
        if (!std::strcmp(argv[i], "--seeds")) number(seeds);
        else if (!std::strcmp(argv[i], "--start")) number(start);
        else if (!std::strcmp(argv[i], "--reduce")) { number(reduce); reducing = true; }
        else if (!std::strcmp(argv[i], "--hardware")) hardware = true;
        else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
        else if (!std::strcmp(argv[i], "--all")) all = true;
        else {
            std::fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
        if (bad) {
            std::fprintf(stderr, "%s needs a decimal count\n", argv[i]);
            return 2;
        }
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "wine_get_version")) {
        std::puts("SKIP: Wine's D3D12 is not a reference");
        return 77;
    }
    ComPtr<ID3D12Device> layer;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&layer)))) {
        std::puts("SKIP: no Vulkan device");
        return 77;
    }
    std::wstring ref_name;
    ComPtr<ID3D12Device> ref;
    {
        // WARP by default; the first hardware adapter with --hardware.
        ComPtr<IDXGIFactory4> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            if (!hardware) {
                ComPtr<IDXGIAdapter> warp;
                if (SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))) &&
                    SUCCEEDED(D3D12CreateDevice(warp.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ref))))
                    ref_name = L"WARP";
            } else {
                for (UINT i = 0; !ref; ++i) {
                    ComPtr<IDXGIAdapter1> adapter;
                    if (factory->EnumAdapters1(i, &adapter) != S_OK) break;
                    DXGI_ADAPTER_DESC1 desc{};
                    if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
                    if (SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&ref))))
                        ref_name = desc.Description;
                }
            }
        }
    }
    if (!ref) {
        std::puts("SKIP: no reference D3D12 device");
        return 77;
    }
    const std::vector<Usable> use = usable(layer.Get(), ref.Get(), verbose);
    Avoid avoid;
    if (all) avoid.documented = nan_equal = false;
    else if (const auto* info = rt::d3d12::vulkan_device_info(layer.Get())) {
        VkPhysicalDeviceProperties props{};
        const auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
            info->get_instance_proc(info->instance, "vkGetPhysicalDeviceProperties"));
        if (properties) properties(info->physical, &props);
        avoid.nvidia_stencil = props.vendorID == 0x10de;
    }
    if (!all)
        std::printf("left out (--all puts them back): uav_clear.h's documented UINT patterns%s\n",
                    avoid.nvidia_stencil ? "; stencil clears with rects past mip 0 (NVIDIA driver)" : "");
    if (use.empty()) {
        std::puts("SKIP: no format both devices can clear");
        return 77;
    }

    if (reducing) {
        Program p = generate(reduce, use, avoid);
        const auto differs = [&](const Program& q) {
            const Result a = run(layer.Get(), q, reduce), b = run(ref.Get(), q, reduce);
            size_t count = 0;
            return a.error.empty() && b.error.empty() && !compare(q, a, b, count).empty();
        };
        if (!differs(p)) {
            std::printf("seed %u: the layer matches %ls, nothing to reduce\n", reduce, ref_name.c_str());
            return 0;
        }
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t i = p.clears.size(); i-- > 0;) {
                Program q = p;
                q.clears.erase(q.clears.begin() + std::ptrdiff_t(i));
                if (!q.clears.empty() && differs(q)) {
                    p = q;
                    changed = true;
                }
            }
            for (size_t i = 0; i < p.clears.size(); ++i)
                for (size_t r = p.clears[i].rects.size(); r-- > 0;) {
                    Program q = p;
                    q.clears[i].rects.erase(q.clears[i].rects.begin() + std::ptrdiff_t(r));
                    if (differs(q)) {
                        p = q;
                        changed = true;
                    }
                }
        }
        const Result a = run(layer.Get(), p, reduce), b = run(ref.Get(), p, reduce);
        size_t count = 0;
        const std::string where = compare(p, a, b, count);
        std::printf("seed %u reduced to %zu clear(s), %zu byte(s) differ; first: %s\n", reduce, p.clears.size(), count,
                    where.c_str());
        print_program(p);
        return 1;
    }

    unsigned differ = 0, gaps = 0, ran = 0;
    for (unsigned k = 0; k < seeds; ++k) {
        const std::uint64_t seed = std::uint64_t(start) + k;
        const Program p = generate(seed, use, avoid);
        const Result b = run(ref.Get(), p, seed);
        if (!b.error.empty()) {
            if (verbose) std::printf("seed %llu: reference: %s\n", static_cast<unsigned long long>(seed), b.error.c_str());
            continue;  // the reference refused it: not a test of the layer
        }
        const Result a = run(layer.Get(), p, seed);
        ++ran;
        if (!a.error.empty()) {
            if (a.gap) ++gaps;
            else ++differ;
            std::printf("seed %llu: layer: %s\n", static_cast<unsigned long long>(seed), a.error.c_str());
            if (verbose) print_program(p);
            continue;
        }
        size_t count = 0;
        const std::string where = compare(p, a, b, count);
        if (!where.empty()) {
            ++differ;
            std::printf("seed %llu: %zu byte(s) differ from %ls, first at %s\n", static_cast<unsigned long long>(seed),
                        count, ref_name.c_str(), where.c_str());
            if (verbose) print_program(p);
        } else if (verbose) {
            std::printf("seed %llu: same (%s, %zu clears)\n", static_cast<unsigned long long>(seed), p.use->family->name,
                        p.clears.size());
        }
    }
    std::printf("clear fuzz (layer against %ls): %u seeds run, %u differ, %u refused by the layer\n", ref_name.c_str(),
                ran, differ, gaps);
    return differ || gaps ? 1 : 0;
}
