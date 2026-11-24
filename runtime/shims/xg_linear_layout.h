// The texture layout xg_xs.dll's computer hands the title (shims/xg_xs.cpp): linear, and the same one wherever the
// runtime answers for a texture's memory (device slots 25 and 108 size a console description at least this large).
//
// The record shapes are the console's (XG_RESOURCE_LAYOUT, measured on the retail executable 1.0.2.1 by PR #8134):
// a 0x18-byte header (size, base alignment, mip levels, planes), then four planes of 0x6c8 bytes, each a 0x38-byte
// header and fifteen 0x70-byte mip records. The title copies the four planes whole and reads a mip's +0x00 (size, all
// slices), +0x10 (one 2D slice's size) and +0x1c (row pitch). The other fields follow the Xbox One XDK's
// XG_PLANE_LAYOUT / XG_MIPLEVEL_LAYOUT order (an ASSUMPTION: the title is not seen reading them). Nothing is written
// past the four planes: the XDK's trailing Dimension word is not measured.
//
// The layout: mips one after another, each holding all its array slices (mip-major, as the console's), every 2D
// slice at a 512-byte boundary with rows 256 bytes apart, the depth slices of a 3D mip row-pitch * rows apart. So
// every subresource is a D3D12 placed footprint (offset a multiple of 512, row pitch of 256, depth pitch row pitch *
// rows): a buffer over memory the title filled this way can be copied into the texture with CopyTextureRegion.
#pragma once

#include <windows.h>
#include <d3d12.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace rt::xg {

// Who is told that the texture computer wrote [address, address + bytes) (CopyIntoSubresource): the D3D12 layer's
// title-placed textures (d3d12_vulkan_device.cpp, "Title-placed textures"), set when it registers the first one.
using WriteListener = void (*)(const void* address, std::size_t bytes);
inline std::atomic<WriteListener> g_write_listener{nullptr};

inline void note_write(const void* address, std::size_t bytes) {
    if (const WriteListener listener = g_write_listener.load(std::memory_order_acquire)) listener(address, bytes);
}

#pragma pack(push, 1)
struct MipRecord {  // 0x70 bytes
    std::uint64_t size_bytes;          // +0x00: all array slices (or depth slices) of the mip
    std::uint64_t offset_bytes;        // +0x08: from the resource's start
    std::uint64_t slice2d_size_bytes;  // +0x10: one 2D slice (a depth slice of a 3D mip), padded
    std::uint32_t pitch_pixels;        // +0x18
    std::uint32_t pitch_bytes;         // +0x1c
    std::uint32_t alignment_bytes;     // +0x20
    std::uint32_t padded_width;        // +0x24 elements
    std::uint32_t padded_height;       // +0x28 elements
    std::uint32_t padded_depth;        // +0x2c
    std::uint32_t width_elements;      // +0x30
    std::uint32_t height_elements;     // +0x34
    std::uint32_t depth_or_array;      // +0x38
    std::uint32_t sample_count;        // +0x3c
    std::uint32_t tile_mode;           // +0x40: 0, linear
    std::uint8_t rest[0x70 - 0x44];
};
struct PlaneLayout {  // 0x6c8 bytes
    std::uint32_t usage;                 // +0x00
    std::uint32_t pad0;
    std::uint64_t size_bytes;            // +0x08
    std::uint64_t base_offset_bytes;     // +0x10
    std::uint64_t base_alignment_bytes;  // +0x18
    std::uint32_t bytes_per_element;     // +0x20
    std::uint8_t rest[0x38 - 0x24];
    MipRecord mip[15];
};
struct ResourceLayout {  // what the computer's slot 2 writes
    std::uint64_t size_bytes;
    std::uint64_t base_alignment_bytes;
    std::uint32_t mip_levels;
    std::uint32_t planes;
    PlaneLayout plane[4];
};
#pragma pack(pop)
static_assert(sizeof(MipRecord) == 0x70);
static_assert(sizeof(PlaneLayout) == 0x6c8);
static_assert(offsetof(ResourceLayout, plane) == 0x18);
static_assert(sizeof(ResourceLayout) == 0x18 + 4 * 0x6c8);

constexpr std::uint64_t kRowAlign = 256;          // D3D12_TEXTURE_DATA_PITCH_ALIGNMENT
constexpr std::uint64_t kSubresourceAlign = 512;  // D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT
constexpr std::uint64_t kResourceAlign = 64 * 1024;
constexpr std::uint32_t kMaxMips = 15;

// Bytes of one element (a texel, or a 4x4 block) and the block side, or bytes 0 for a format this layout does not
// cover (planar video formats, R1, the 2x1 packed formats).
struct Element {
    std::uint32_t bytes;
    std::uint32_t block;
};

inline Element element(DXGI_FORMAT f) {
    switch (f) {
    case DXGI_FORMAT_R32G32B32A32_TYPELESS: case DXGI_FORMAT_R32G32B32A32_FLOAT: case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return {16, 1};
    case DXGI_FORMAT_R32G32B32_TYPELESS: case DXGI_FORMAT_R32G32B32_FLOAT: case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
        return {12, 1};
    case DXGI_FORMAT_BC1_TYPELESS: case DXGI_FORMAT_BC1_UNORM: case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC4_TYPELESS: case DXGI_FORMAT_BC4_UNORM: case DXGI_FORMAT_BC4_SNORM:
        return {8, 4};
    case DXGI_FORMAT_BC2_TYPELESS: case DXGI_FORMAT_BC2_UNORM: case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_TYPELESS: case DXGI_FORMAT_BC3_UNORM: case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC5_TYPELESS: case DXGI_FORMAT_BC5_UNORM: case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_TYPELESS: case DXGI_FORMAT_BC6H_UF16: case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_TYPELESS: case DXGI_FORMAT_BC7_UNORM: case DXGI_FORMAT_BC7_UNORM_SRGB:
        return {16, 4};
    case DXGI_FORMAT_R1_UNORM: case DXGI_FORMAT_R8G8_B8G8_UNORM: case DXGI_FORMAT_G8R8_G8B8_UNORM:
        return {0, 1};
    default:
        break;
    }
    const auto v = static_cast<std::uint32_t>(f);
    if (v >= DXGI_FORMAT_R16G16B16A16_TYPELESS && v <= DXGI_FORMAT_X32_TYPELESS_G8X24_UINT) return {8, 1};  // 9..22
    if (v >= DXGI_FORMAT_R10G10B10A2_TYPELESS && v <= DXGI_FORMAT_X24_TYPELESS_G8_UINT) return {4, 1};      // 23..47
    if (v >= DXGI_FORMAT_R8G8_TYPELESS && v <= DXGI_FORMAT_R16_SINT) return {2, 1};                          // 48..59
    if (v >= DXGI_FORMAT_R8_TYPELESS && v <= DXGI_FORMAT_A8_UNORM) return {1, 1};                            // 60..65
    if (v == DXGI_FORMAT_R9G9B9E5_SHAREDEXP) return {4, 1};
    if (v == DXGI_FORMAT_B5G6R5_UNORM || v == DXGI_FORMAT_B5G5R5A1_UNORM || v == DXGI_FORMAT_B4G4R4A4_UNORM)
        return {2, 1};
    if (v >= DXGI_FORMAT_B8G8R8A8_UNORM && v <= DXGI_FORMAT_B8G8R8X8_UNORM_SRGB) return {4, 1};              // 87..93
    return {0, 1};
}

inline std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1) / a * a; }

// The layout of a texture description (1D, 2D or 3D). False for a buffer, a zero size or a format element() does not
// cover; `out` is then zeroed.
inline bool linear_layout(const D3D12_RESOURCE_DESC& d, ResourceLayout& out) {
    std::memset(&out, 0, sizeof(out));
    const bool is3d = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    if (d.Dimension < D3D12_RESOURCE_DIMENSION_TEXTURE1D || d.Dimension > D3D12_RESOURCE_DIMENSION_TEXTURE3D ||
        !d.Width || !d.Height || !d.DepthOrArraySize)
        return false;
    const Element e = element(d.Format);
    if (!e.bytes) return false;
    const std::uint64_t largest = std::max<std::uint64_t>({d.Width, d.Height, is3d ? d.DepthOrArraySize : 1u});
    std::uint32_t full = 1;
    while ((largest >> full) > 0) ++full;
    std::uint32_t mips = d.MipLevels ? std::min<std::uint32_t>(d.MipLevels, full) : full;
    mips = std::min(mips, kMaxMips);
    const std::uint32_t samples = std::max<std::uint32_t>(d.SampleDesc.Count, 1);
    const std::uint32_t slices = is3d ? 1u : d.DepthOrArraySize;
    out.mip_levels = mips;
    out.planes = 1;
    out.base_alignment_bytes = kResourceAlign;
    PlaneLayout& p = out.plane[0];
    p.base_alignment_bytes = kResourceAlign;
    p.bytes_per_element = e.bytes;
    std::uint64_t offset = 0;
    for (std::uint32_t m = 0; m < mips; ++m) {
        const std::uint64_t w = std::max<std::uint64_t>(d.Width >> m, 1);
        const std::uint32_t h = std::max<std::uint32_t>(d.Height >> m, 1);
        const std::uint32_t z = is3d ? std::max<std::uint32_t>(d.DepthOrArraySize >> m, 1) : slices;
        const auto we = static_cast<std::uint32_t>((w + e.block - 1) / e.block);
        const std::uint32_t he = (h + e.block - 1) / e.block;
        const std::uint64_t pitch = align_up(std::uint64_t{we} * e.bytes * samples, kRowAlign);
        // A 2D slice is a subresource and starts at 512 bytes; a depth slice is part of one, pitch * rows apart.
        const std::uint64_t slice = is3d ? pitch * he : align_up(pitch * he, kSubresourceAlign);
        MipRecord& r = p.mip[m];
        r.offset_bytes = offset;
        r.slice2d_size_bytes = slice;
        r.size_bytes = align_up(slice * z, kSubresourceAlign);
        r.pitch_bytes = static_cast<std::uint32_t>(pitch);
        r.pitch_pixels = static_cast<std::uint32_t>(pitch / e.bytes / samples * e.block);
        r.alignment_bytes = static_cast<std::uint32_t>(kSubresourceAlign);
        r.padded_width = static_cast<std::uint32_t>(pitch / e.bytes / samples);
        r.padded_height = he;
        r.padded_depth = z;
        r.width_elements = we;
        r.height_elements = he;
        r.depth_or_array = z;
        r.sample_count = samples;
        offset += r.size_bytes;
    }
    p.size_bytes = align_up(offset, kResourceAlign);
    out.size_bytes = p.size_bytes;
    return true;
}

// Subresource `subresource` (mip + slice * mips, D3D12's numbering) of `layout` as a D3D12 placed footprint, the
// offset from the resource's start. False past the last subresource.
inline bool footprint(const D3D12_RESOURCE_DESC& d, const ResourceLayout& layout, std::uint32_t subresource,
                      D3D12_PLACED_SUBRESOURCE_FOOTPRINT& out) {
    const std::uint32_t mips = layout.mip_levels;
    if (!mips) return false;
    const bool is3d = d.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const std::uint32_t mip = subresource % mips, slice = subresource / mips;
    if (slice >= (is3d ? 1u : d.DepthOrArraySize)) return false;
    const MipRecord& r = layout.plane[0].mip[mip];
    out.Offset = r.offset_bytes + (is3d ? 0 : std::uint64_t{slice} * r.slice2d_size_bytes);
    out.Footprint.Format = d.Format;
    // In texels, whole blocks for a block-compressed format (as GetCopyableFootprints gives them).
    const std::uint32_t block = element(d.Format).block;
    out.Footprint.Width = r.width_elements * block;
    out.Footprint.Height = r.height_elements * block;
    out.Footprint.Depth = is3d ? r.depth_or_array : 1;
    out.Footprint.RowPitch = r.pitch_bytes;
    return true;
}

}  // namespace rt::xg
