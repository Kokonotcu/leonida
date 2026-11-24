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

namespace {

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
         : Slot == 8  ? reinterpret_cast<void*>(&linear_copy_into)
         : Slot == 10 ? reinterpret_cast<void*>(&linear_get_resource_tiling)
                      : reinterpret_cast<void*>(&computer_slot<Slot>))...};
}

const auto kLinearVtable = make_linear_vtable(std::make_index_sequence<kSlots>{});

HRESULT __stdcall XGCreateTextureComputer(const D3D12_RESOURCE_DESC* desc, void** out) {
    if (!out) return E_POINTER;
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
    if (logged.fetch_add(1, std::memory_order_relaxed) < 4)
        std::fprintf(stderr, "[rt] xg_xs: texture computer: dimension %d %llux%ux%u, %u mip(s), format %u -> %llu "
                             "bytes (linear)\n",
                     static_cast<int>(desc->Dimension), static_cast<unsigned long long>(desc->Width), desc->Height,
                     static_cast<unsigned>(desc->DepthOrArraySize), layout.mip_levels,
                     static_cast<unsigned>(desc->Format), static_cast<unsigned long long>(layout.size_bytes));
    *out = c;
    return S_OK;
}

HRESULT __stdcall XGSuggestSwizzleMode(const void* /*desc*/, uint32_t /*flags*/, uint32_t* /*mode*/) {
    static std::atomic<bool> warned{false};
    warn_once(warned, "XGSuggestSwizzleMode", -1, _ReturnAddress());
    return kNotImpl;
}

}  // namespace

SHIM_TABLE(xg_xs,
    {"xg_xs.dll", "XGCreateTextureComputer", reinterpret_cast<void*>(&XGCreateTextureComputer)},
    {"xg_xs.dll", "XGSuggestSwizzleMode", reinterpret_cast<void*>(&XGSuggestSwizzleMode)},
);
