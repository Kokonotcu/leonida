// xg_xs.dll: the console's texture layout library. The game executable (1.0.2.1) imports two functions from it.
//
// Measured from the call sites in the executable, read statically (shapes and checks only):
//   HRESULT XGSuggestSwizzleMode(const desc*, 0, uint32_t* mode)
//     2 call sites, both in the game's helper that fills a texture description before XGCreateTextureComputer. The
//     out pointer is a 32-bit field of that description, which the helper first sets to 0 and, when the mode is
//     known, writes itself as 0x100 | mode. The HRESULT is only passed back up; no caller branches on it.
//   HRESULT XGCreateTextureComputer(const desc*, computer** out)
//     4 call sites. Two test the HRESULT and take their failure path when it is negative; two ignore it and call
//     through *out at once. On the computer the callers use vtable slot 1 after they are done (3 sites, read as
//     Release; no site calls slot 0, read as AddRef, nor QueryInterface), slot 2 with one out pointer to a large
//     stack structure (2 sites), slot 8 with six arguments in a loop over subresources (1 site) and slot 10 with
//     seven arguments (1 site). No site reads the return value of slots 2, 8 or 10.
// PR #8134 measured the rest on the same executable: the description is a D3D12_RESOURCE_DESC (0x38 bytes); slot 2 is
// GetResourceLayout(XG_RESOURCE_LAYOUT*) (the title copies its four planes whole and reads a mip's size, 2D slice size
// and row pitch); slot 8 is CopyIntoSubresource(base, plane, subresource, linear data, row pitch, slice pitch); slot 10
// takes ID3D12Device::GetResourceTiling's arguments, for reserved resources only.
//
// What this is (PR #8134's texture computer, ported): a LINEAR layout (xg_linear_layout.h). Every texture the title
// lays out with it is read by this runtime's own D3D12 layer, so the layout only has to agree with itself and with
// what the layer answers for the same description, and it does:
// - slot 2's records, slot 8's copies and slot 10's tile count all come from one rt::xg::linear_layout;
// - every subresource is a D3D12 placed footprint (rt::xg::footprint), so memory the title fills through slot 8 can be
//   copied into a texture with CopyTextureRegion;
// - device slots 25 and 108 answer a console texture description with at least this layout's size, so memory the
//   title sizes with them holds what slot 8 writes.
// A description it does not cover (a buffer, a format without a linear element size) gets the static computer below:
// E_NOTIMPL from every slot, as before, and still stored in *out for the two callers that do not check.
// XGSuggestSwizzleMode writes nothing (E_NOTIMPL): the field it would fill is the description's Layout, which the
// runtime reads as the PC's, and the computer is linear whatever it holds.
// Hypothesis, not measured: slots 0 and 1 being AddRef and Release; slot 9 (CopyFromSubresource?) and the slots past
// 10, which stay E_NOTIMPL and are logged once.
#include <windows.h>
#include <intrin.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <utility>

#include "shims.h"
#include "xg_linear_layout.h"
#include "../src/xg_console_layout.h"

namespace {

namespace xc = leonida::xg_console;

// A description whose layout word names a tiled console swizzle mode (0x100 + mode, mode not 0): the console layout
// (five-star-recomp's XG, ported in src/xg_console_layout.h). Textures streamed from the game's archives are stored
// in it, so the computer answers the same layout for them, or the title's own fills (slot 8) and the device's
// detiling of the pages would disagree.
std::optional<xc::Layout> console_layout(const D3D12_RESOURCE_DESC& d) {
    const auto word = static_cast<std::uint32_t>(d.Layout);
    if (!xc::is_console_tiled(word) || xc::swizzle_mode(word) == xc::mode_linear) return std::nullopt;
    if (d.Dimension < D3D12_RESOURCE_DIMENSION_TEXTURE1D || d.Dimension > D3D12_RESOURCE_DIMENSION_TEXTURE3D)
        return std::nullopt;
    xc::Desc x;
    x.dimension = static_cast<std::uint32_t>(d.Dimension);
    x.width = d.Width;
    x.height = d.Height;
    x.depth_or_array_size = d.DepthOrArraySize;
    x.mip_levels = d.MipLevels;
    x.format = static_cast<std::uint32_t>(d.Format);
    x.sample_count = std::max<std::uint32_t>(d.SampleDesc.Count, 1);
    x.layout = word;
    return xc::compute_layout(x);
}

// Slot 2's records of a console layout, in the order five-star-recomp measured the game reading them (docs/xg.md):
// plane +0x08 size, +0x10 array slice stride (the game addresses a slice as slice x stride + mip offset), +0x18 base
// offset, +0x20 alignment, +0x28 bytes per element; each mip its size, offset from the resource start, 2D slice size,
// pitch in pixels and bytes, alignment, padded and real extents in elements.
void console_records(const xc::Layout& l, rt::xg::ResourceLayout& out) {
    std::memset(&out, 0, sizeof(out));
    out.size_bytes = l.size;
    out.base_alignment_bytes = l.alignment;
    out.mip_levels = l.planes[0].mip_count;
    out.planes = l.plane_count;
    for (std::uint32_t p = 0; p < l.plane_count && p < 4; ++p) {
        const xc::Plane& plane = l.planes[p];
        auto* raw = reinterpret_cast<std::uint8_t*>(&out.plane[p]);
        const auto put64 = [&](std::size_t at, std::uint64_t v) { std::memcpy(raw + at, &v, 8); };
        const auto put32 = [&](std::size_t at, std::uint32_t v) { std::memcpy(raw + at, &v, 4); };
        put64(0x08, plane.size);
        put64(0x10, plane.slice_stride);
        put64(0x18, plane.offset);
        put64(0x20, plane.alignment);
        put32(0x28, plane.bytes_per_element);
        for (std::uint32_t m = 0; m < plane.mip_count && m < rt::xg::kMaxMips; ++m) {
            const xc::Mip& mip = plane.mips[m];
            rt::xg::MipRecord& r = out.plane[p].mip[m];
            r.size_bytes = mip.size;
            r.offset_bytes = xc::mip_offset(l, p, m);
            r.slice2d_size_bytes = mip.slice_2d_size;
            r.pitch_pixels = mip.pitch_elements * plane.block_dimension;
            r.pitch_bytes = mip.pitch_bytes;
            r.alignment_bytes = plane.block_bytes ? plane.block_bytes : xc::linear_mip_alignment;
            r.padded_width = mip.pitch_elements;
            r.padded_height = mip.padded_height_elements;
            r.padded_depth = mip.depth_elements;
            r.width_elements = mip.width_elements;
            r.height_elements = mip.height_elements;
            r.depth_or_array = mip.depth_elements;
            r.sample_count = 1;
            r.tile_mode = l.swizzle_mode;
        }
    }
}

constexpr HRESULT kNotImpl = static_cast<HRESULT>(rt::shims::kENotImpl);
constexpr size_t kSlots = 32;

// Prints once per entry point; `slot` is the vtable index for a computer method, or -1 for an export.
void warn_once(std::atomic<bool>& warned, const char* what, int slot, const void* caller) {
    if (warned.exchange(true, std::memory_order_relaxed)) return;
    char name[96];
    if (slot < 0) std::snprintf(name, sizeof(name), "%s", what);
    else std::snprintf(name, sizeof(name), "%s%d", what, slot);
    std::fprintf(stderr, "[rt] xg_xs: %s: not implemented; E_NOTIMPL (caller %s, logged once)\n", name,
                 rt::shims::module_offset(reinterpret_cast<uint64_t>(caller)).c_str());
}

struct Computer {
    const void* const* vtable;
    std::atomic<long> refs{1};
    D3D12_RESOURCE_DESC desc{};
    rt::xg::ResourceLayout layout{};
    xc::Layout console{};  // the console computer's layout (console_vtable)
};

// ---- The static computer: a description the layout does not cover. ----

ULONG __stdcall static_add_ref(void*) { return 1; }
ULONG __stdcall static_release(void*) { return 1; }

template <size_t Slot>
HRESULT __stdcall computer_slot(void*) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "texture computer slot ", static_cast<int>(Slot), _ReturnAddress());
    return kNotImpl;
}

template <size_t... Slot>
constexpr auto make_static_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0   ? reinterpret_cast<void*>(&static_add_ref)
         : Slot == 1 ? reinterpret_cast<void*>(&static_release)
                     : reinterpret_cast<void*>(&computer_slot<Slot>))...};
}

const auto kStaticVtable = make_static_vtable(std::make_index_sequence<kSlots>{});
struct StaticComputer {
    const void* const* vtable;
};
StaticComputer g_static{kStaticVtable.data()};

// ---- The linear computer. ----

ULONG __stdcall linear_add_ref(Computer* c) { return static_cast<ULONG>(++c->refs); }
ULONG __stdcall linear_release(Computer* c) {
    const long left = --c->refs;
    if (!left) delete c;
    return static_cast<ULONG>(left);
}

// Slot 2: GetResourceLayout(XG_RESOURCE_LAYOUT*): the header and the four planes, nothing past them.
HRESULT __stdcall linear_get_resource_layout(Computer* c, void* out) {
    if (!out) return E_POINTER;
    std::memcpy(out, &c->layout, sizeof(c->layout));
    return S_OK;
}

// Slot 8: CopyIntoSubresource(base, plane, subresource, linear data, row pitch, slice pitch): the subresource's rows
// from `linear` (row_pitch apart, depth slices slice_pitch apart) to their place in the layout at `base`.
HRESULT __stdcall linear_copy_into(Computer* c, std::uint8_t* base, std::uint32_t plane, std::uint32_t subresource,
                                   const std::uint8_t* linear, std::uint32_t row_pitch, std::uint32_t slice_pitch) {
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT f{};
    if (!base || !linear || plane != 0 || !rt::xg::footprint(c->desc, c->layout, subresource, f)) return E_INVALIDARG;
    const rt::xg::MipRecord& r = c->layout.plane[0].mip[subresource % c->layout.mip_levels];
    const std::uint64_t row_bytes =
        std::uint64_t{r.width_elements} * c->layout.plane[0].bytes_per_element * std::max(r.sample_count, 1u);
    const std::uint64_t copy = row_pitch ? std::min<std::uint64_t>(row_bytes, row_pitch) : row_bytes;
    for (std::uint32_t z = 0; z < f.Footprint.Depth; ++z)
        for (std::uint32_t y = 0; y < r.height_elements; ++y)
            std::memcpy(base + f.Offset + std::uint64_t{z} * r.slice2d_size_bytes + std::uint64_t{y} * r.pitch_bytes,
                        linear + std::uint64_t{z} * slice_pitch + std::uint64_t{y} * row_pitch, copy);
    // A texture the title placed on these pages (device slot 133) takes the new texels at its next submission.
    rt::xg::note_write(base + f.Offset, static_cast<std::size_t>(r.slice2d_size_bytes * f.Footprint.Depth));
    return S_OK;
}

// Slot 6: UINT64 GetTexelElementOffsetBytes(plane, mip, x, y, z or slice, sample, UINT* bit offset): the byte offset of
// one element from the resource start (x and y in elements: the caller divides by 4 for BC formats). Measured by
// five-star-recomp (ThatRecompDev, docs/xg.md): the game adds the return to the resource base to address one element
// (0x1455acdd0, inlined at five sites); E_NOTIMPL (0x80004001) came back as an offset and sent those writes 2 GiB away.
// An element outside the layout gives 0, as the console's does for a plane, mip or slice that does not exist.
std::uint64_t __stdcall linear_texel_offset(Computer* c, std::uint32_t plane, std::uint32_t mip, std::uint64_t x,
                                            std::uint64_t y, std::uint64_t z, std::uint64_t /*sample*/,
                                            std::uint32_t* bit_offset) {
    if (bit_offset) *bit_offset = 0;
    y &= 0xFFFFFFFFull;
    z &= 0xFFFFFFFFull;
    if (plane != 0 || mip >= c->layout.mip_levels) return 0;
    const bool volume = c->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const std::uint32_t subresource = volume ? mip : mip + static_cast<std::uint32_t>(z) * c->layout.mip_levels;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT f{};
    if (!rt::xg::footprint(c->desc, c->layout, subresource, f)) return 0;
    const rt::xg::MipRecord& r = c->layout.plane[0].mip[mip];
    return f.Offset + (volume ? z * r.slice2d_size_bytes : 0) + y * r.pitch_bytes +
           x * c->layout.plane[0].bytes_per_element;
}

// Slot 10: GetResourceTiling's arguments (UINT* tiles, D3D12_PACKED_MIP_INFO*, D3D12_TILE_SHAPE*, UINT* count in/out,
// UINT first, D3D12_SUBRESOURCE_TILING*). A linear layout has no standard tile shape: every mip is reported packed into
// the resource's 64 KiB tiles.
HRESULT __stdcall linear_get_resource_tiling(Computer* c, UINT* tiles, D3D12_PACKED_MIP_INFO* packed,
                                             D3D12_TILE_SHAPE* shape, UINT* count, UINT first,
                                             D3D12_SUBRESOURCE_TILING* out) {
    const auto total = static_cast<UINT>(c->layout.size_bytes / rt::xg::kResourceAlign);
    if (tiles) *tiles = total;
    if (packed) *packed = {0, static_cast<UINT8>(c->layout.mip_levels), total, 0};
    if (shape) *shape = {0, 0, 0};
    if (count) {
        const bool is3d = c->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        const UINT subresources = c->layout.mip_levels * (is3d ? 1u : c->desc.DepthOrArraySize);
        *count = std::min(*count, first < subresources ? subresources - first : 0u);
        for (UINT i = 0; out && i < *count; ++i) out[i] = {0, 0, 0, D3D12_PACKED_TILE};
    }
    return S_OK;
}

template <size_t... Slot>
constexpr auto make_linear_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0    ? reinterpret_cast<void*>(&linear_add_ref)
         : Slot == 1  ? reinterpret_cast<void*>(&linear_release)
         : Slot == 2  ? reinterpret_cast<void*>(&linear_get_resource_layout)
         : Slot == 6  ? reinterpret_cast<void*>(&linear_texel_offset)
         : Slot == 8  ? reinterpret_cast<void*>(&linear_copy_into)
         : Slot == 10 ? reinterpret_cast<void*>(&linear_get_resource_tiling)
                      : reinterpret_cast<void*>(&computer_slot<Slot>))...};
}

const auto kLinearVtable = make_linear_vtable(std::make_index_sequence<kSlots>{});

// ---- The console computer: a tiled layout word (console_layout). ----

// Slot 6 in the console layout: x and y in elements; z the depth of a 3D texture, else the array slice.
std::uint64_t __stdcall console_texel_offset(Computer* c, std::uint32_t plane, std::uint32_t mip, std::uint64_t x,
                                             std::uint64_t y, std::uint64_t z, std::uint64_t /*sample*/,
                                             std::uint32_t* bit_offset) {
    if (bit_offset) *bit_offset = 0;
    const bool volume = c->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const auto layer = static_cast<std::uint32_t>(z & 0xFFFFFFFFull);
    return xc::element_offset(c->console, plane, mip, volume ? 0 : layer, x, static_cast<std::uint32_t>(y),
                              volume ? layer : 0);
}

// Slot 8 in the console layout: the subresource's rows from `linear` tiled into the resource at `base`.
HRESULT __stdcall console_copy_into(Computer* c, std::uint8_t* base, std::uint32_t plane, std::uint32_t subresource,
                                    const std::uint8_t* linear, std::uint32_t row_pitch, std::uint32_t slice_pitch) {
    const std::uint32_t mips = c->console.planes[0].mip_count;
    if (!base || !linear || !mips) return E_INVALIDARG;
    const bool volume = c->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const std::uint32_t mip = subresource % mips, slice = volume ? 0 : subresource / mips;
    if (!xc::tile_subresource(c->console, plane, mip, slice, reinterpret_cast<const std::byte*>(linear), row_pitch,
                              slice_pitch,
                              {reinterpret_cast<std::byte*>(base), static_cast<std::size_t>(c->console.size)}))
        return E_INVALIDARG;
    rt::xg::note_write(base, static_cast<std::size_t>(c->console.size));
    return S_OK;
}

// Slot 9, CopyFromSubresource(linear, plane, subresource, a UINT the game passes as 0, resource base, row pitch,
// slice pitch) (five-star-recomp docs/xg.md): the subresource detiled into rows.
HRESULT __stdcall console_copy_from(Computer* c, std::uint8_t* linear, std::uint32_t plane, std::uint32_t subresource,
                                    std::uint32_t /*unknown*/, const std::uint8_t* base, std::uint32_t row_pitch,
                                    std::uint32_t slice_pitch) {
    const std::uint32_t mips = c->console.planes[0].mip_count;
    if (!base || !linear || !mips) return E_INVALIDARG;
    const bool volume = c->desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const std::uint32_t mip = subresource % mips, slice = volume ? 0 : subresource / mips;
    return xc::detile_subresource(c->console, plane, mip, slice,
                                  {reinterpret_cast<const std::byte*>(base), static_cast<std::size_t>(c->console.size)},
                                  reinterpret_cast<std::byte*>(linear), row_pitch, slice_pitch)
               ? S_OK
               : E_INVALIDARG;
}

template <size_t... Slot>
constexpr auto make_console_vtable(std::index_sequence<Slot...>) {
    return std::array<void*, sizeof...(Slot)>{
        (Slot == 0    ? reinterpret_cast<void*>(&linear_add_ref)
         : Slot == 1  ? reinterpret_cast<void*>(&linear_release)
         : Slot == 2  ? reinterpret_cast<void*>(&linear_get_resource_layout)
         : Slot == 6  ? reinterpret_cast<void*>(&console_texel_offset)
         : Slot == 8  ? reinterpret_cast<void*>(&console_copy_into)
         : Slot == 9  ? reinterpret_cast<void*>(&console_copy_from)
         : Slot == 10 ? reinterpret_cast<void*>(&linear_get_resource_tiling)
                      : reinterpret_cast<void*>(&computer_slot<Slot>))...};
}

const auto kConsoleVtable = make_console_vtable(std::make_index_sequence<kSlots>{});

HRESULT __stdcall XGCreateTextureComputer(const D3D12_RESOURCE_DESC* desc, void** out) {
    if (!out) return E_POINTER;
    if (desc) {
        if (const auto console = console_layout(*desc)) {
            auto* c = new (std::nothrow) Computer{kConsoleVtable.data()};
            if (!c) {
                *out = &g_static;
                return E_OUTOFMEMORY;
            }
            c->desc = *desc;
            c->console = *console;
            console_records(*console, c->layout);
            static std::atomic<unsigned> logged{0};
            if (logged.fetch_add(1, std::memory_order_relaxed) < 4)
                std::fprintf(stderr, "[rt] xg_xs: texture computer: dimension %d %llux%ux%u, %u mip(s), format %u, "
                                     "layout %#x -> %llu bytes (console tiled)\n",
                             static_cast<int>(desc->Dimension), static_cast<unsigned long long>(desc->Width),
                             desc->Height, static_cast<unsigned>(desc->DepthOrArraySize), console->planes[0].mip_count,
                             static_cast<unsigned>(desc->Format), static_cast<unsigned>(desc->Layout),
                             static_cast<unsigned long long>(console->size));
            *out = c;
            return S_OK;
        }
    }
    rt::xg::ResourceLayout layout{};
    if (!desc || !rt::xg::linear_layout(*desc, layout)) {
        static std::atomic<bool> warned{false};
        warn_once(warned, "XGCreateTextureComputer (a description the linear layout does not cover)", -1,
                  _ReturnAddress());
        *out = &g_static;  // see the top of the file: two measured callers use *out without checking the HRESULT
        return kNotImpl;
    }
    auto* c = new (std::nothrow) Computer{kLinearVtable.data()};
    if (!c) {
        *out = &g_static;
        return E_OUTOFMEMORY;
    }
    c->desc = *desc;
    c->layout = layout;
    static std::atomic<unsigned> logged{0};
    // Also every mip chain of 16-byte texels (the first 64): retail boot 15 died on an R32G32B32A32 mip chain the title
    // wrote past its memory, and whether this layout sized it is the question.
    static std::atomic<unsigned> wide_chains{0};
    const bool wide_chain = layout.mip_levels > 1 && layout.plane[0].bytes_per_element >= 16 &&
                            wide_chains.fetch_add(1, std::memory_order_relaxed) < 64;
    if (logged.fetch_add(1, std::memory_order_relaxed) < 4 || wide_chain)
        std::fprintf(stderr, "[rt] xg_xs: texture computer: dimension %d %llux%ux%u, %u mip(s), format %u -> %llu "
                             "bytes (linear)\n",
                     static_cast<int>(desc->Dimension), static_cast<unsigned long long>(desc->Width), desc->Height,
                     static_cast<unsigned>(desc->DepthOrArraySize), layout.mip_levels,
                     static_cast<unsigned>(desc->Format), static_cast<unsigned long long>(layout.size_bytes));
    *out = c;
    return S_OK;
}

// XGSuggestSwizzleMode(desc, flags, UINT32* layout): the game asks it for an asset whose mode byte is 0xFF and writes
// the answer into the desc's Layout (five-star-recomp docs/xg.md): 0x100 + 27 (64 KB blocks) when mip 0 holds 1 MiB or
// more, else 0x100 + 23 (4 KB), the split the shipped textures show. E_NOTIMPL left the layout unwritten.
HRESULT __stdcall XGSuggestSwizzleMode(const D3D12_RESOURCE_DESC* desc, uint32_t /*flags*/, uint32_t* layout) {
    if (!desc || !layout) return E_POINTER;
    xc::Desc x;
    x.dimension = static_cast<std::uint32_t>(desc->Dimension);
    x.width = desc->Width;
    x.height = desc->Height;
    x.depth_or_array_size = desc->DepthOrArraySize;
    x.mip_levels = desc->MipLevels;
    x.format = static_cast<std::uint32_t>(desc->Format);
    x.sample_count = std::max<std::uint32_t>(desc->SampleDesc.Count, 1);
    *layout = xc::suggest_layout(x);
    return S_OK;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
