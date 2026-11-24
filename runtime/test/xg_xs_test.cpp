// Checks the registered xg_xs.dll shim (runtime/shims/xg_xs.cpp) against the call shapes measured in the game
// executable and the linear layout it computes (runtime/shims/xg_linear_layout.h, PR #8134's texture computer).
//
// Both exports are resolved through rt::shims::find, so this is the table the runtime installs. The checks:
// - XGSuggestSwizzleMode returns E_NOTIMPL and leaves the caller's field alone;
// - a description the layout does not cover still gets a computer (two measured callers use it without checking the
//   HRESULT), whose slots 2, 8 and 10 return E_NOTIMPL without writing what their arguments point to;
// - a texture gets a linear computer: slot 2 writes the console's XG_RESOURCE_LAYOUT records and nothing past its four
//   planes, every subresource is a D3D12 placed footprint (offsets of 512, row pitches of 256, a 3D mip's depth slices
//   row pitch * rows apart), slot 8 puts a subresource's rows where slot 2 says and touches nothing else, slot 10
//   reports the size in 64 KiB tiles, and slots 0/1 count references.
#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include "shims.h"
#include "xg_linear_layout.h"

namespace {

using CreateFn = HRESULT(__stdcall*)(const void*, void**);
using SuggestFn = HRESULT(__stdcall*)(const void*, uint32_t, uint32_t*);
using RefFn = ULONG(__stdcall*)(void*);
using Slot2Fn = HRESULT(__stdcall*)(void*, void*);
using Slot8Fn = HRESULT(__stdcall*)(void*, void*, uint32_t, uint32_t, const void*, uint32_t, uint32_t);
using Slot10Fn = HRESULT(__stdcall*)(void*, uint32_t*, void*, void*, uint32_t*, uint32_t, void*);
constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* what) {
    ++g_checks;
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition) ++g_failures;
}

bool all_bytes(const uint8_t* p, size_t n, uint8_t value) {
    for (size_t i = 0; i < n; ++i)
        if (p[i] != value) return false;
    return true;
}

void* const* vtable_of(void* object) { return *static_cast<void* const* const*>(object); }

D3D12_RESOURCE_DESC texture(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth, UINT16 mips,
                            DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = dimension;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = depth;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc.Count = 1;
    return d;
}

// Every subresource of `layout` as D3D12 asks of a placed footprint, and no two subresources overlapping.
bool footprints_valid(const D3D12_RESOURCE_DESC& d, const rt::xg::ResourceLayout& layout) {
    const bool is3d = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const UINT count = layout.mip_levels * (is3d ? 1u : d.DepthOrArraySize);
    std::uint64_t end = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (UINT s = 0; s < count; ++s) {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT f{};
        if (!rt::xg::footprint(d, layout, s, f)) return false;
        if (f.Offset % 512 || f.Footprint.RowPitch % 256) return false;
        const rt::xg::MipRecord& r = layout.plane[0].mip[s % layout.mip_levels];
        const std::uint64_t rows = r.height_elements;
        const std::uint64_t bytes = std::uint64_t{f.Footprint.RowPitch} * rows * f.Footprint.Depth;
        if (is3d && r.slice2d_size_bytes != std::uint64_t{f.Footprint.RowPitch} * rows) return false;
        ranges.push_back({f.Offset, f.Offset + bytes});
        end = std::max(end, f.Offset + bytes);
    }
    for (size_t i = 0; i < ranges.size(); ++i)
        for (size_t j = i + 1; j < ranges.size(); ++j)
            if (ranges[i].first < ranges[j].second && ranges[j].first < ranges[i].second) return false;
    return end <= layout.size_bytes && layout.size_bytes % 0x10000 == 0;
}

}  // namespace

int main() {
    auto create = reinterpret_cast<CreateFn>(rt::shims::find("xg_xs.dll", "XGCreateTextureComputer"));
    auto suggest = reinterpret_cast<SuggestFn>(rt::shims::find("xg_xs.dll", "XGSuggestSwizzleMode"));
    check(create != nullptr, "XGCreateTextureComputer is registered");
    check(suggest != nullptr, "XGSuggestSwizzleMode is registered");
    check(rt::shims::owns_dll("xg_xs.dll"), "the registry owns xg_xs.dll");
    if (!create || !suggest) return 1;

    uint8_t desc[0x38] = {};
    uint32_t mode = 0x1234;
    check(suggest(desc, 0, &mode) == kNotImpl, "XGSuggestSwizzleMode returns E_NOTIMPL");
    check(mode == 0x1234, "XGSuggestSwizzleMode leaves the mode field as the caller set it");

    // A description the layout does not cover (all zeros: no dimension).
    check(create(desc, nullptr) == E_POINTER, "XGCreateTextureComputer(desc, nullptr) is E_POINTER");
    void* computer = nullptr;
    check(create(desc, &computer) == kNotImpl, "an uncovered description: E_NOTIMPL");
    check(computer != nullptr, "an uncovered description still gets a computer for the callers that do not check");
    if (!computer) return 1;
    {
        void* const* vtable = vtable_of(computer);
        uint8_t layout[0x800];
        std::memset(layout, 0xcd, sizeof(layout));
        check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, layout) == kNotImpl, "static computer: slot 2 E_NOTIMPL");
        check(all_bytes(layout, sizeof(layout), 0xcd), "static computer: slot 2 writes nothing");
        uint8_t dst[64];
        std::memset(dst, 0xcd, sizeof(dst));
        const uint8_t src[64] = {};
        check(reinterpret_cast<Slot8Fn>(vtable[8])(computer, dst, 0, 0, src, 16, 64) == kNotImpl &&
                  all_bytes(dst, sizeof(dst), 0xcd),
              "static computer: slot 8 E_NOTIMPL, writes nothing");
        uint32_t count = 0, inout = 0x10;
        uint8_t a[12] = {}, b[12] = {}, list[0xc0] = {};
        check(reinterpret_cast<Slot10Fn>(vtable[10])(computer, &count, a, b, &inout, 0, list) == kNotImpl &&
                  count == 0 && inout == 0x10 && all_bytes(a, 12, 0) && all_bytes(b, 12, 0) &&
                  all_bytes(list, sizeof(list), 0),
              "static computer: slot 10 E_NOTIMPL, leaves its outputs");
        auto release = reinterpret_cast<RefFn>(vtable[1]);
        release(computer);
        release(computer);
        check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, layout) == kNotImpl,
              "the static computer stays usable after more Releases than it was handed out");
    }

    // A 13x7 RGBA8 2D array of 2 slices and 3 mips.
    const D3D12_RESOURCE_DESC array = texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 13, 7, 2, 3,
                                              DXGI_FORMAT_R8G8B8A8_UNORM);
    computer = nullptr;
    check(create(&array, &computer) == S_OK && computer, "a 2D array texture gets a linear computer");
    if (!computer) return 1;
    void* const* vtable = vtable_of(computer);
    std::vector<uint8_t> out(sizeof(rt::xg::ResourceLayout) + 64, 0xcd);
    check(reinterpret_cast<Slot2Fn>(vtable[2])(computer, out.data()) == S_OK, "slot 2 GetResourceLayout");
    check(all_bytes(out.data() + sizeof(rt::xg::ResourceLayout), 64, 0xcd),
          "slot 2 writes the header and four planes (0x1b38 bytes), nothing past them");
    rt::xg::ResourceLayout got{};
    std::memcpy(&got, out.data(), sizeof(got));
    const rt::xg::MipRecord& m0 = got.plane[0].mip[0];
    const rt::xg::MipRecord& m1 = got.plane[0].mip[1];
    check(got.mip_levels == 3 && got.planes == 1, "3 mips, 1 plane");
    // Mip 0: 13 texels * 4 bytes in a 256-byte row, 7 rows in 2048 bytes (a 512 multiple), two slices.
    check(m0.pitch_bytes == 256 && m0.slice2d_size_bytes == 2048 && m0.size_bytes == 4096 && m0.offset_bytes == 0,
          "mip 0: row pitch 256, 2D slice 2048, both slices 4096 at offset 0 (the title reads +0x00, +0x10, +0x1c)");
    check(m1.offset_bytes == 4096 && m1.width_elements == 6 && m1.height_elements == 3 &&
              m1.slice2d_size_bytes == 1024,
          "mip 1 follows mip 0, a mip holds both its slices");
    check(footprints_valid(array, got), "every subresource is a D3D12 placed footprint, none overlapping, in the size");
    check(got.size_bytes == 0x10000, "the size, in 64 KiB");

    // Slot 8: subresource 4 (mip 1 of slice 1) from linear rows 40 bytes apart; nothing else in the layout is written.
    {
        std::vector<uint8_t> base(static_cast<size_t>(got.size_bytes), 0xcd);
        std::vector<uint8_t> linear(40 * 3);
        for (size_t i = 0; i < linear.size(); ++i) linear[i] = static_cast<uint8_t>(i + 1);
        check(reinterpret_cast<Slot8Fn>(vtable[8])(computer, base.data(), 0, 4, linear.data(), 40, 120) == S_OK,
              "slot 8 CopyIntoSubresource");
        const size_t at = static_cast<size_t>(m1.offset_bytes + m1.slice2d_size_bytes);
        bool rows = true, rest = true;
        for (size_t y = 0; y < 3; ++y)
            rows = rows && std::memcmp(base.data() + at + y * 256, linear.data() + y * 40, 24) == 0;
        for (size_t i = 0; i < base.size(); ++i) {
            const bool inside = i >= at && i < at + 3 * 256 && (i - at) % 256 < 24;
            if (!inside && base[i] != 0xcd) rest = false;
        }
        check(rows, "slot 8 puts each row at offset + slice * 2D slice size + y * row pitch");
        check(rest, "slot 8 writes nothing else");
        check(reinterpret_cast<Slot8Fn>(vtable[8])(computer, base.data(), 0, 6, linear.data(), 40, 120) ==
                  E_INVALIDARG,
              "slot 8 refuses a subresource past the last");
    }

    // Slot 10: the size in 64 KiB tiles, every mip packed; the count clamped to the subresources from `first`.
    {
        uint32_t tiles = 0, count = 16;
        D3D12_PACKED_MIP_INFO packed{};
        D3D12_TILE_SHAPE shape{1, 1, 1};
        D3D12_SUBRESOURCE_TILING tilings[16]{};
        check(reinterpret_cast<Slot10Fn>(vtable[10])(computer, &tiles, &packed, &shape, &count, 2, tilings) == S_OK,
              "slot 10");
        check(tiles == 1 && packed.NumPackedMips == 3 && packed.NumTilesForPackedMips == 1 &&
                  shape.WidthInTexels == 0,
              "slot 10: one tile, three packed mips, no standard tile shape");
        check(count == 4 && tilings[0].StartTileIndexInOverallResource == D3D12_PACKED_TILE,
              "slot 10: the count clamped to the 4 subresources from 2, each packed");
    }

    // Slots 0/1 count references: AddRef then two Releases destroy it.
    check(reinterpret_cast<RefFn>(vtable[0])(computer) == 2, "slot 0 AddRef");
    check(reinterpret_cast<RefFn>(vtable[1])(computer) == 1, "slot 1 Release");
    check(reinterpret_cast<RefFn>(vtable[1])(computer) == 0, "the last Release");

    // A 3D BC1 volume: blocks, depth slices row pitch * rows apart, mips at 512-byte boundaries.
    const D3D12_RESOURCE_DESC volume = texture(D3D12_RESOURCE_DIMENSION_TEXTURE3D, 20, 12, 3, 0, DXGI_FORMAT_BC1_UNORM);
    rt::xg::ResourceLayout v{};
    check(rt::xg::linear_layout(volume, v), "a 3D BC1 volume is covered");
    check(v.mip_levels == 5 && v.plane[0].mip[0].width_elements == 5 && v.plane[0].mip[0].height_elements == 3 &&
              v.plane[0].mip[0].slice2d_size_bytes == 256 * 3 && v.plane[0].mip[0].depth_or_array == 3,
          "BC1 20x12x3: 5x3 blocks a slice, 256 * 3 bytes a depth slice, 3 deep, 5 mips");
    check(footprints_valid(volume, v), "every 3D subresource is a D3D12 placed footprint");
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT f{};
    check(rt::xg::footprint(volume, v, 4, f) && f.Footprint.Width == 4 && f.Footprint.Height == 4 &&
              f.Footprint.Depth == 1,
          "the last BC1 mip's footprint is one whole block");
    check(!rt::xg::linear_layout(texture(D3D12_RESOURCE_DIMENSION_BUFFER, 64, 1, 1, 1, DXGI_FORMAT_UNKNOWN), v) &&
              !rt::xg::linear_layout(texture(D3D12_RESOURCE_DIMENSION_TEXTURE2D, 64, 64, 1, 1, DXGI_FORMAT_NV12), v),
          "a buffer and a planar format are not covered");

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
