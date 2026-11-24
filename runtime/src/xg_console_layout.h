#pragma once

// The console's (XG, xg_xs.dll) texture layout: sizes, mip and element offsets, and copies between that layout and
// row-major memory. Standard library only; the DXGI format is a plain number and its element shape is looked up here.
//
// Ported from five-star-recomp (github.com/ThatRecompDev/five-star-recomp, GPL-3.0-or-later, Copyright (c) 2026 Five
// Star Recomp contributors): src/FiveStar.Loader/Shims/Xg/XgLayoutBuilder.cs, XgAddress.cs, XgCopier.cs (the copy
// loop), XgSwizzles.cs, XgSwizzle.cs, XgSwizzleModes.cs, XgFormats.cs, XgElementFormat.cs, XgMip.cs, XgPlane.cs,
// XgSurface.cs, XgLimits.cs, XgResourceDesc.cs, the volume bricks of FiveStar.Formats/Ttpf/SwizzlePatterns.cs and
// XGSuggestSwizzleMode of XgComputers.cs; the rules are those of its docs/xg.md and docs/textures.md. The COM
// computer, GetResourceTiling and the XG_RESOURCE_LAYOUT writer are not ported.
//
// The layout word is the D3D12_RESOURCE_DESC Layout field as the game writes it: 0x100 + a swizzle mode in AMD's
// ADDR_SW numbering (0 linear, 1-3 256 B blocks, 4-7 4 KB, 8-15 64 KB, 16-19 64 KB, 20-23 4 KB, 24-31 64 KB; the low
// two bits are the kind Z, S, D, R). Words below 0x100 are the desktop D3D12_TEXTURE_LAYOUT values and are not a
// console layout here (five-star sizes an UNKNOWN non-target texture with the suggested mode; a caller that wants
// that sets the word from suggest_layout first).
//
// Per plane, array slices follow one another, each holding one mip chain. An element is a 4x4 block for BC formats
// and a pixel otherwise. In a slice the mip tail block comes first, then the other mips smallest first, each padded to
// whole blocks with blocks row-major, so mip 0 ends the slice. The first tail mip is the first one no wider than half
// a block and no taller than a block with at most log2(block bytes) - 4 mips from it to the end of the chain; a chain
// of one mip, and any chain in 256-byte blocks, has no tail. Tail mip k sits in slot m = log2(block bytes) - 5 - k, at
// 16 << m bytes when m > 6 and m x 256 bytes otherwise, and is addressed with the whole block's swizzle from the
// element whose index is the slot start. Linear planes (mode 0, or 12-byte elements) hold the mips in order from mip 0
// with rows padded to 256 bytes and mips on 512 bytes. Depth-stencil formats get a second plane of one-byte stencil
// after the depth plane, aligned to its block. A 3D texture in a Z or S mode with 4 KB or 64 KB blocks is thick: one
// chain of bricks, each mip padded to whole bricks, bricks in x then y then z, smallest mip first, no tail; in other
// modes its depth slices are laid out as array slices of the full chain.
//
// The block swizzles give each bit of an element's index inside a block as the next bit of x or y (or z in a brick),
// least significant first; they are the patterns found on the shipped textures. A 256-byte block takes the first bits
// of the 4 KB pattern; the 1-byte 64 KB pattern takes y bit 1 before y bit 0. Bricks of 2- and 8-byte elements in
// 4 KB blocks are the shipped volume patterns; other bricks are five-star's extension (x, z and y bits in turn), not
// checked on data. MSAA layouts and the pipe/bank XOR are unknown, so multisampled descs get no layout.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

namespace leonida::xg_console {

// D3D12_RESOURCE_DIMENSION values.
inline constexpr std::uint32_t dimension_1d = 2;
inline constexpr std::uint32_t dimension_2d = 3;
inline constexpr std::uint32_t dimension_3d = 4;

// D3D12_REQ_MIP_LEVELS, and the planes produced: colour or depth, and stencil.
inline constexpr std::uint32_t max_mips = 15;
inline constexpr std::uint32_t max_planes = 2;

// The first layout word that carries a swizzle mode, and the modes XGSuggestSwizzleMode answers (ADDR_SW_4KB_R_X for
// mip 0 under 1 MiB, ADDR_SW_64KB_R_X from 1 MiB).
inline constexpr std::uint32_t mode_base = 0x100;
inline constexpr std::uint32_t mode_linear = 0;
inline constexpr std::uint32_t mode_suggested_small = 23;
inline constexpr std::uint32_t mode_suggested_large = 27;

inline constexpr std::uint32_t large_block_bytes = 0x10000;
inline constexpr std::uint32_t small_block_bytes = 0x1000;
inline constexpr std::uint32_t micro_block_bytes = 0x100;

// Linear rows are padded to 256 bytes and linear mips start on 512 bytes.
inline constexpr std::uint32_t linear_row_bytes = 256;
inline constexpr std::uint32_t linear_mip_alignment = 512;

// The description XGCreateTextureComputer reads (the D3D12_RESOURCE_DESC fields that matter for the layout).
struct Desc {
    std::uint32_t dimension = dimension_2d;
    std::uint64_t width = 1;
    std::uint32_t height = 1;
    std::uint16_t depth_or_array_size = 1;
    // 0 means the full chain, as in D3D12.
    std::uint16_t mip_levels = 1;
    std::uint32_t format = 0;
    std::uint32_t sample_count = 1;
    std::uint32_t layout = 0;
};

// The element shape of a DXGI format as XG tiles it.
struct ElementFormat {
    std::uint32_t bytes_per_element = 4;
    // Pixels per element side: 4 for BC formats, 1 otherwise.
    std::uint32_t block_dimension = 1;
    // Whether a second one-byte plane holds stencil.
    bool stencil_plane = false;
};

// Element sizes by DXGI enumeration range (XgFormats.cs). Format 118 is the console's D16_UNORM_S8_UINT. Unknown
// formats are taken as 4 bytes.
constexpr ElementFormat element_format(std::uint32_t dxgi) {
    if (dxgi < 1) return {4, 1, false};
    if (dxgi <= 4) return {16, 1, false};   // R32G32B32A32
    if (dxgi <= 8) return {12, 1, false};   // R32G32B32
    if (dxgi <= 18) return {8, 1, false};   // R16G16B16A16, R32G32
    if (dxgi <= 22) return {4, 1, true};    // R32G8X24, D32_FLOAT_S8X24_UINT
    if (dxgi <= 43) return {4, 1, false};   // R10G10B10A2 .. R32
    if (dxgi <= 47) return {4, 1, true};    // R24G8, D24_UNORM_S8_UINT
    if (dxgi <= 59) return {2, 1, false};   // R8G8, R16
    if (dxgi <= 66) return {1, 1, false};   // R8, A8, R1
    if (dxgi <= 69) return {4, 1, false};   // R9G9B9E5, R8G8_B8G8, G8R8_G8B8
    if (dxgi <= 72) return {8, 4, false};   // BC1
    if (dxgi <= 78) return {16, 4, false};  // BC2, BC3
    if (dxgi <= 81) return {8, 4, false};   // BC4
    if (dxgi <= 84) return {16, 4, false};  // BC5
    if (dxgi <= 86) return {2, 1, false};   // B5G6R5, B5G5R5A1
    if (dxgi <= 93) return {4, 1, false};   // B8G8R8A8, B8G8R8X8, R10G10B10_XR_BIAS_A2
    if (dxgi <= 99) return {16, 4, false};  // BC6H, BC7
    if (dxgi == 115) return {2, 1, false};  // B4G4R4A4
    if (dxgi == 118) return {2, 1, true};   // console D16_UNORM_S8_UINT
    return {4, 1, false};
}

namespace detail {

constexpr std::uint32_t popcount(std::uint32_t value) {
    std::uint32_t count = 0;
    for (; value != 0; value &= value - 1) ++count;
    return count;
}

constexpr std::uint32_t log2(std::uint64_t value) {
    std::uint32_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

constexpr bool is_pow2(std::uint32_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

constexpr std::uint32_t trailing_zeros(std::uint32_t value) {
    if (value == 0) return 32;
    std::uint32_t count = 0;
    while ((value & 1u) == 0) {
        value >>= 1;
        ++count;
    }
    return count;
}

constexpr std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// The bits of value spread onto the set bits of mask, lowest first (BMI2 pdep); bits beyond the mask are dropped.
constexpr std::uint32_t deposit(std::uint32_t value, std::uint32_t mask) {
    std::uint32_t result = 0;
    std::uint32_t bit = 0;
    for (std::uint32_t rest = mask; rest != 0; rest &= rest - 1) {
        const std::uint32_t lowest = rest & (~rest + 1);
        if ((value >> bit) & 1u) result |= lowest;
        ++bit;
    }
    return result;
}

// The bits of value under the set bits of mask, packed from bit 0 (BMI2 pext).
constexpr std::uint32_t extract(std::uint32_t value, std::uint32_t mask) {
    std::uint32_t result = 0;
    std::uint32_t bit = 0;
    for (std::uint32_t rest = mask; rest != 0; rest &= rest - 1) {
        const std::uint32_t lowest = rest & (~rest + 1);
        if ((value & lowest) != 0) result |= 1u << bit;
        ++bit;
    }
    return result;
}

}  // namespace detail

// A block swizzle as bit masks over the element index inside a block: index bit k is the next x bit of the element's
// position when bit k of x_mask is set, the next z bit when it is in z_mask, the next y bit otherwise. With
// low_y_swapped the first two y bits are y bit 1, then y bit 0.
struct Swizzle {
    std::uint32_t x_mask = 0;
    std::uint32_t y_mask = 0;
    std::uint32_t z_mask = 0;
    std::uint32_t bits = 0;
    bool low_y_swapped = false;
    std::uint32_t tile_width = 1;
    std::uint32_t tile_height = 1;
    std::uint32_t tile_depth = 1;

    static constexpr Swizzle make(std::uint32_t x_mask, std::uint32_t bits, bool low_y_swapped = false,
                                  std::uint32_t z_mask = 0) {
        Swizzle swizzle;
        swizzle.x_mask = x_mask;
        swizzle.z_mask = z_mask;
        swizzle.y_mask = ((1u << bits) - 1) & ~x_mask & ~z_mask;
        swizzle.bits = bits;
        swizzle.low_y_swapped = low_y_swapped;
        swizzle.tile_width = 1u << detail::popcount(swizzle.x_mask);
        swizzle.tile_height = 1u << detail::popcount(swizzle.y_mask);
        swizzle.tile_depth = 1u << detail::popcount(swizzle.z_mask);
        return swizzle;
    }

    constexpr std::uint32_t swap_low_y(std::uint32_t y) const {
        if (!low_y_swapped) return y;
        return (y & ~3u) | ((y & 1u) << 1) | ((y >> 1) & 1u);
    }

    // The element index inside a block of the element at (x, y) (and z for a brick) of that block.
    constexpr std::uint32_t index(std::uint32_t x, std::uint32_t y) const {
        return detail::deposit(x, x_mask) | detail::deposit(swap_low_y(y), y_mask);
    }

    constexpr std::uint32_t index(std::uint32_t x, std::uint32_t y, std::uint32_t z) const {
        return index(x, y) | detail::deposit(z, z_mask);
    }

    // The position inside its block of the element at an index.
    constexpr std::uint32_t x_of(std::uint32_t element_index) const {
        return detail::extract(element_index, x_mask);
    }

    constexpr std::uint32_t y_of(std::uint32_t element_index) const {
        return swap_low_y(detail::extract(element_index, y_mask));
    }
};

namespace detail {

// Element sizes with a pattern: 1, 2, 4, 8 and 16 bytes (index = log2 of the size).
inline constexpr std::uint32_t element_sizes = 5;

inline constexpr std::array<std::string_view, element_sizes> small_patterns = {
    "xxxxyyyyyxyx", "xxxyyyxyxyx", "xxyyyxyxyx", "xyyxxyxyx", "yyxxyxyx"};
inline constexpr std::array<std::string_view, element_sizes> large_patterns = {
    "xxxyyyxyyxyxyxyx", "xxxyyyxyxyxyxyx", "xxyyxyyxyxyxyx", "xyxxyyxyxyxyx", "xyxyyxyxyxyx"};

constexpr std::uint32_t x_mask_of(std::string_view pattern) {
    std::uint32_t mask = 0;
    for (std::size_t bit = 0; bit < pattern.size(); ++bit) {
        if (pattern[bit] == 'x') mask |= 1u << bit;
    }
    return mask;
}

// The 2D swizzle of a block size and element size (log2 of bytes); the 256-byte block is a prefix of the 4 KB one.
constexpr Swizzle find_swizzle(std::uint32_t block_bytes, std::uint32_t element_log2) {
    if (block_bytes == large_block_bytes) {
        const std::string_view pattern = large_patterns[element_log2];
        return Swizzle::make(x_mask_of(pattern), static_cast<std::uint32_t>(pattern.size()), element_log2 == 0);
    }
    const std::string_view pattern = small_patterns[element_log2];
    std::uint32_t bits = static_cast<std::uint32_t>(pattern.size());
    std::uint32_t x_mask = x_mask_of(pattern);
    if (block_bytes != small_block_bytes) {
        bits -= log2(small_block_bytes) - log2(micro_block_bytes);
        x_mask &= (1u << bits) - 1;
    }
    return Swizzle::make(x_mask, bits);
}

// The brick of a thick block: the shipped 4 KB bricks of 2- and 8-byte elements, extended to the brick shape (4 KB:
// 16x16x16, 8x16x16, 8x16x8, 8x8x8, 4x8x8; 64 KB: x4, y2, z2 of those) by appending x, z and y bits in turn.
constexpr Swizzle find_volume_swizzle(std::uint32_t block_bytes, std::uint32_t element_log2) {
    constexpr std::array<std::uint32_t, element_sizes> brick_x = {4, 3, 3, 3, 2};
    constexpr std::array<std::uint32_t, element_sizes> brick_y = {4, 4, 4, 3, 3};
    constexpr std::array<std::uint32_t, element_sizes> brick_z = {4, 4, 3, 3, 3};
    const bool large = block_bytes == large_block_bytes;
    // Axes in the order extra bits are taken: x, z, y.
    const std::array<std::uint32_t, 3> wanted = {brick_x[element_log2] + (large ? 2u : 0u),
                                                 brick_z[element_log2] + (large ? 1u : 0u),
                                                 brick_y[element_log2] + (large ? 1u : 0u)};
    std::string_view known;
    if (element_log2 == 1) known = "xzyzyxzyxzy";
    if (element_log2 == 3) known = "xzyxzyxzy";

    std::array<std::uint32_t, 3> have = {0, 0, 0};
    std::uint32_t x_mask = 0;
    std::uint32_t z_mask = 0;
    std::uint32_t bit = 0;
    for (const char letter : known) {
        if (letter == 'x') {
            x_mask |= 1u << bit;
            ++have[0];
        } else if (letter == 'z') {
            z_mask |= 1u << bit;
            ++have[1];
        } else {
            ++have[2];
        }
        ++bit;
    }
    for (bool added = true; added;) {
        added = false;
        for (std::uint32_t axis = 0; axis < 3; ++axis) {
            if (have[axis] < wanted[axis]) {
                if (axis == 0) x_mask |= 1u << bit;
                if (axis == 1) z_mask |= 1u << bit;
                ++bit;
                ++have[axis];
                added = true;
            }
        }
    }
    return Swizzle::make(x_mask, bit, false, z_mask);
}

}  // namespace detail

// Whether the layout word names a console swizzle mode (0x100 + mode). 0 (UNKNOWN), 1 (ROW_MAJOR) and the other
// desktop D3D12_TEXTURE_LAYOUT values are not console layouts.
constexpr bool is_console_tiled(std::uint32_t layout_word) {
    return layout_word >= mode_base;
}

// The swizzle mode of a console layout word.
constexpr std::uint32_t swizzle_mode(std::uint32_t layout_word) {
    return layout_word & 0xFFu;
}

// Bytes of one swizzle block of a mode, or 0 for linear.
constexpr std::uint32_t mode_block_bytes(std::uint32_t mode) {
    if (mode == mode_linear) return 0;
    if (mode <= 3) return micro_block_bytes;
    if (mode <= 7 || (mode >= 20 && mode <= 23)) return small_block_bytes;
    return large_block_bytes;
}

// Whether a 3D texture in this mode is laid out in thick bricks: the Z and S kinds of 4 KB and 64 KB blocks.
constexpr bool is_thick_mode(std::uint32_t mode) {
    return mode_block_bytes(mode) >= small_block_bytes && (mode & 3u) < 2;
}

// XGSuggestSwizzleMode: the layout word (0x100 + mode) for a desc, 64 KB blocks when mip 0 holds 1 MiB or more.
constexpr std::uint32_t suggest_layout(const Desc& desc) {
    const ElementFormat format = element_format(desc.format);
    const std::uint64_t width = desc.width == 0 ? 1 : desc.width;
    const std::uint64_t height = desc.height == 0 ? 1 : desc.height;
    const std::uint64_t mip0_bytes = (width + format.block_dimension - 1) / format.block_dimension *
                                     ((height + format.block_dimension - 1) / format.block_dimension) *
                                     format.bytes_per_element;
    return mode_base | (mip0_bytes >= 0x100000 ? mode_suggested_large : mode_suggested_small);
}

// Where one mip of one array slice of a plane lives.
struct Mip {
    // Byte offset from the start of the slice (for a tail mip, its slot inside the tail block at the slice start).
    // The offset from the resource start is plane.offset + slice x plane.slice_stride + offset (see mip_offset).
    std::uint64_t offset = 0;
    // Bytes the mip occupies (pitch x padded height, all bricks of a thick mip).
    std::uint64_t size = 0;
    // Bytes of one depth slice of the mip.
    std::uint64_t slice_2d_size = 0;
    std::uint32_t width_elements = 0;
    std::uint32_t height_elements = 0;
    std::uint32_t depth_elements = 1;
    std::uint32_t pitch_elements = 0;
    std::uint32_t padded_height_elements = 0;
    std::uint32_t pitch_bytes = 0;
    // For a mip in the tail: its first element's position inside the tail block.
    std::uint32_t tail_x = 0;
    std::uint32_t tail_y = 0;
    bool in_tail = false;
};

// One plane: array slices of slice_stride bytes from offset. A thick plane holds one chain of bricks instead, and
// slice_stride and size are the whole volume.
struct Plane {
    std::uint32_t bytes_per_element = 0;
    std::uint32_t block_dimension = 1;
    // Bytes of one swizzle block, or 0 for a linear plane.
    std::uint32_t block_bytes = 0;
    Swizzle swizzle;
    std::uint32_t mip_count = 0;
    // The first mip in the tail block, or mip_count when there is no tail.
    std::uint32_t first_tail_mip = 0;
    // Array size, or depth for a 3D texture.
    std::uint32_t slices = 1;
    bool thick = false;
    std::uint64_t offset = 0;
    std::uint64_t slice_stride = 0;
    std::uint64_t size = 0;
    std::uint64_t alignment = 0;
    std::array<Mip, max_mips> mips{};
};

struct Layout {
    Desc desc;
    std::uint32_t swizzle_mode = 0;
    std::uint32_t plane_count = 0;
    std::uint64_t size = 0;
    std::uint64_t alignment = 0;
    std::array<Plane, max_planes> planes{};
};

namespace detail {

inline void set_mip_extents(Plane& plane, std::uint32_t width, std::uint32_t height) {
    for (std::uint32_t index = 0; index < plane.mip_count; ++index) {
        Mip& mip = plane.mips[index];
        const std::uint32_t w = std::max(1u, width >> index);
        const std::uint32_t h = std::max(1u, height >> index);
        mip.width_elements = (w + plane.block_dimension - 1) / plane.block_dimension;
        mip.height_elements = (h + plane.block_dimension - 1) / plane.block_dimension;
        mip.depth_elements = 1;
    }
}

inline void build_linear(Plane& plane) {
    const std::uint32_t pitch_alignment =
        linear_row_bytes >> std::min(trailing_zeros(plane.bytes_per_element), log2(linear_row_bytes));
    std::uint64_t offset = 0;
    for (std::uint32_t index = 0; index < plane.mip_count; ++index) {
        Mip& mip = plane.mips[index];
        mip.pitch_elements = static_cast<std::uint32_t>(align_up(mip.width_elements, pitch_alignment));
        mip.padded_height_elements = mip.height_elements;
        mip.slice_2d_size = std::uint64_t{mip.pitch_elements} * mip.padded_height_elements * plane.bytes_per_element;
        mip.size = mip.slice_2d_size;
        offset = align_up(offset, linear_mip_alignment);
        mip.offset = offset;
        offset += mip.size;
    }
    plane.slice_stride = align_up(offset, linear_mip_alignment);
}

inline void build_thick(Plane& plane) {
    const Swizzle& swizzle = plane.swizzle;
    std::uint64_t offset = 0;
    for (std::uint32_t index = plane.mip_count; index-- > 0;) {
        Mip& mip = plane.mips[index];
        mip.depth_elements = std::max(1u, plane.slices >> index);
        mip.pitch_elements = static_cast<std::uint32_t>(align_up(mip.width_elements, swizzle.tile_width));
        mip.padded_height_elements = static_cast<std::uint32_t>(align_up(mip.height_elements, swizzle.tile_height));
        mip.slice_2d_size = std::uint64_t{mip.pitch_elements} * mip.padded_height_elements * plane.bytes_per_element;
        mip.size = mip.slice_2d_size * align_up(mip.depth_elements, swizzle.tile_depth);
        mip.offset = offset;
        offset += mip.size;
    }
    plane.slice_stride = offset;
}

inline void set_tail_mip(Mip& mip, const Swizzle& swizzle, std::uint32_t bytes_per_element, int slot) {
    const std::uint32_t clamped = static_cast<std::uint32_t>(std::max(0, slot));
    const std::uint64_t start = clamped > 6 ? std::uint64_t{1} << (clamped + 4) : std::uint64_t{clamped} << 8;
    const std::uint32_t element = static_cast<std::uint32_t>(start / bytes_per_element);
    mip.in_tail = true;
    mip.offset = start;
    mip.tail_x = swizzle.x_of(element);
    mip.tail_y = swizzle.y_of(element);
    mip.pitch_elements = swizzle.tile_width / 2;
    mip.padded_height_elements = swizzle.tile_height;
    mip.slice_2d_size = std::uint64_t{mip.pitch_elements} * mip.padded_height_elements * bytes_per_element;
    mip.size = mip.slice_2d_size;
}

inline void build_tiled(Plane& plane) {
    const Swizzle& swizzle = plane.swizzle;
    const std::uint32_t tail_mips = log2(plane.block_bytes) - 4;
    // A chain of one mip has no tail: a lone mip that would fit the tail starts at offset 0.
    const bool has_tail = plane.block_bytes > micro_block_bytes && plane.mip_count > 1;
    for (std::uint32_t index = 0; has_tail && index < plane.mip_count; ++index) {
        const Mip& mip = plane.mips[index];
        if (mip.width_elements <= swizzle.tile_width / 2 && mip.height_elements <= swizzle.tile_height &&
            plane.mip_count - index <= tail_mips) {
            plane.first_tail_mip = index;
            break;
        }
    }

    std::uint64_t offset = plane.first_tail_mip < plane.mip_count ? plane.block_bytes : 0;
    for (std::uint32_t index = plane.first_tail_mip; index-- > 0;) {
        Mip& mip = plane.mips[index];
        mip.pitch_elements = static_cast<std::uint32_t>(align_up(mip.width_elements, swizzle.tile_width));
        mip.padded_height_elements = static_cast<std::uint32_t>(align_up(mip.height_elements, swizzle.tile_height));
        mip.slice_2d_size = std::uint64_t{mip.pitch_elements} * mip.padded_height_elements * plane.bytes_per_element;
        mip.size = mip.slice_2d_size;
        mip.offset = offset;
        offset += mip.size;
    }
    for (std::uint32_t index = plane.first_tail_mip; index < plane.mip_count; ++index) {
        set_tail_mip(plane.mips[index], swizzle, plane.bytes_per_element,
                     static_cast<int>(tail_mips) - 1 - static_cast<int>(index - plane.first_tail_mip));
    }
    plane.slice_stride = offset;
}

inline void build_plane(Plane& plane, std::uint32_t bytes_per_element, std::uint32_t block_dimension,
                        std::uint32_t block_bytes, std::uint32_t width, std::uint32_t height, std::uint32_t mips,
                        std::uint32_t slices, bool thick) {
    plane.bytes_per_element = bytes_per_element;
    plane.block_dimension = block_dimension;
    plane.mip_count = mips;
    plane.slices = slices;
    plane.first_tail_mip = mips;
    set_mip_extents(plane, width, height);

    const bool swizzled = is_pow2(bytes_per_element) && log2(bytes_per_element) < element_sizes;
    plane.block_bytes = swizzled ? block_bytes : 0;
    plane.thick = thick && plane.block_bytes != 0;

    if (plane.block_bytes == 0) {
        build_linear(plane);
    } else if (plane.thick) {
        plane.swizzle = find_volume_swizzle(plane.block_bytes, log2(bytes_per_element));
        build_thick(plane);
    } else {
        plane.swizzle = find_swizzle(plane.block_bytes, log2(bytes_per_element));
        build_tiled(plane);
    }
    for (std::uint32_t index = 0; index < plane.mip_count; ++index) {
        plane.mips[index].pitch_bytes = plane.mips[index].pitch_elements * bytes_per_element;
    }
    plane.size = plane.thick ? plane.slice_stride : plane.slice_stride * slices;
    plane.alignment = plane.block_bytes == 0 ? linear_mip_alignment : plane.block_bytes;
}

}  // namespace detail

// The console layout of a texture, or nothing when the layout word is not a console layout, the dimension is not a
// texture, or the desc is multisampled.
inline std::optional<Layout> compute_layout(const Desc& desc) {
    if (!is_console_tiled(desc.layout)) return std::nullopt;
    if (desc.dimension < dimension_1d || desc.dimension > dimension_3d) return std::nullopt;
    if (desc.sample_count > 1) return std::nullopt;
    if (desc.width > std::numeric_limits<std::uint32_t>::max()) return std::nullopt;

    Layout layout;
    layout.desc = desc;
    const ElementFormat format = element_format(desc.format);
    const std::uint32_t width = desc.width == 0 ? 1u : static_cast<std::uint32_t>(desc.width);
    const std::uint32_t height = desc.dimension == dimension_1d ? 1u : (desc.height == 0 ? 1u : desc.height);
    const std::uint32_t full_chain = detail::log2(std::max(width, height)) + 1;
    const std::uint32_t mips = std::min<std::uint32_t>(desc.mip_levels == 0 ? full_chain : desc.mip_levels, max_mips);
    const std::uint32_t slices = desc.depth_or_array_size == 0 ? 1u : desc.depth_or_array_size;
    const std::uint32_t mode = swizzle_mode(desc.layout);
    const std::uint32_t block_bytes = mode_block_bytes(mode);
    const bool thick = desc.dimension == dimension_3d && is_thick_mode(mode);

    layout.swizzle_mode = mode;
    layout.plane_count = format.stencil_plane ? max_planes : 1;
    std::uint64_t offset = 0;
    for (std::uint32_t index = 0; index < layout.plane_count; ++index) {
        Plane& plane = layout.planes[index];
        const std::uint32_t bytes_per_element = index == 0 ? format.bytes_per_element : 1;
        const std::uint32_t block_dimension = index == 0 ? format.block_dimension : 1;
        detail::build_plane(plane, bytes_per_element, block_dimension, block_bytes, width, height, mips, slices, thick);
        offset = detail::align_up(offset, plane.alignment);
        plane.offset = offset;
        offset += plane.size;
        layout.alignment = plane.alignment;
    }
    layout.size = detail::align_up(offset, layout.alignment);
    return layout;
}

namespace detail {

// The offset from the resource start of element (x, y) of a mip of an array slice (the depth for a thick plane).
// Nothing is range-checked.
inline std::uint64_t element(const Plane& plane, std::uint32_t mip_index, std::uint32_t slice, std::uint32_t x,
                             std::uint32_t y) {
    const Mip& mip = plane.mips[mip_index];
    const std::uint64_t slice_start = plane.offset + std::uint64_t{slice} * plane.slice_stride;
    const std::uint64_t bytes_per_element = plane.bytes_per_element;
    if (plane.block_bytes == 0) {
        return slice_start + mip.offset + (std::uint64_t{y} * mip.pitch_elements + x) * bytes_per_element;
    }
    const Swizzle& swizzle = plane.swizzle;
    if (plane.thick) {
        const std::uint64_t columns = mip.pitch_elements / swizzle.tile_width;
        const std::uint64_t rows = mip.padded_height_elements / swizzle.tile_height;
        const std::uint64_t brick = (std::uint64_t{slice / swizzle.tile_depth} * rows + y / swizzle.tile_height) *
                                        columns +
                                    x / swizzle.tile_width;
        const std::uint64_t inside =
            std::uint64_t{swizzle.index(x % swizzle.tile_width, y % swizzle.tile_height, slice % swizzle.tile_depth)} *
            bytes_per_element;
        return plane.offset + mip.offset + brick * plane.block_bytes + inside;
    }
    if (mip.in_tail) {
        return slice_start + std::uint64_t{swizzle.index(mip.tail_x + x, mip.tail_y + y)} * bytes_per_element;
    }
    const std::uint64_t tile = std::uint64_t{y / swizzle.tile_height} * (mip.pitch_elements / swizzle.tile_width) +
                               x / swizzle.tile_width;
    const std::uint64_t inside =
        std::uint64_t{swizzle.index(x % swizzle.tile_width, y % swizzle.tile_height)} * bytes_per_element;
    return slice_start + mip.offset + tile * plane.block_bytes + inside;
}

// One row of a thin swizzled mip: the x part of each element's index comes from a table over one block's width (the
// x and y bits of an index never overlap, so the two parts add).
template <std::uint32_t Bytes, bool Into>
void copy_tiled_row(const Plane& plane, const Mip& mip, std::uint64_t slice_start, std::uint32_t y,
                    const std::uint32_t* x_part, std::byte* resource, std::byte* row) {
    const Swizzle& swizzle = plane.swizzle;
    const std::uint32_t width_mask = swizzle.tile_width - 1;
    std::uint64_t row_base;
    std::uint32_t x_start;
    std::uint64_t tiles_per_row_bytes = 0;
    if (mip.in_tail) {
        row_base = slice_start + std::uint64_t{deposit(swizzle.swap_low_y(mip.tail_y + y), swizzle.y_mask)} * Bytes;
        x_start = mip.tail_x;
    } else {
        const std::uint64_t tiles_per_row = mip.pitch_elements / swizzle.tile_width;
        row_base = slice_start + mip.offset + std::uint64_t{y / swizzle.tile_height} * tiles_per_row * plane.block_bytes +
                   std::uint64_t{deposit(swizzle.swap_low_y(y % swizzle.tile_height), swizzle.y_mask)} * Bytes;
        x_start = 0;
        tiles_per_row_bytes = plane.block_bytes;
    }
    const std::uint32_t shift = log2(swizzle.tile_width);
    for (std::uint32_t x = 0; x < mip.width_elements; ++x) {
        const std::uint32_t position = x_start + x;
        std::uint64_t offset = row_base + std::uint64_t{x_part[position & width_mask]} * Bytes;
        if (!mip.in_tail) offset += std::uint64_t{position >> shift} * tiles_per_row_bytes;
        if constexpr (Into) {
            std::memcpy(resource + offset, row + std::size_t{x} * Bytes, Bytes);
        } else {
            std::memcpy(row + std::size_t{x} * Bytes, resource + offset, Bytes);
        }
    }
}

template <bool Into>
void copy_tiled_row_any(const Plane& plane, const Mip& mip, std::uint64_t slice_start, std::uint32_t y,
                        const std::uint32_t* x_part, std::byte* resource, std::byte* row) {
    switch (plane.bytes_per_element) {
    case 1: copy_tiled_row<1, Into>(plane, mip, slice_start, y, x_part, resource, row); break;
    case 2: copy_tiled_row<2, Into>(plane, mip, slice_start, y, x_part, resource, row); break;
    case 4: copy_tiled_row<4, Into>(plane, mip, slice_start, y, x_part, resource, row); break;
    case 8: copy_tiled_row<8, Into>(plane, mip, slice_start, y, x_part, resource, row); break;
    default: copy_tiled_row<16, Into>(plane, mip, slice_start, y, x_part, resource, row); break;
    }
}

// CopyIntoSubresource / CopyFromSubresource for one mip of one array slice, or every depth slice of a 3D mip (slice
// then has to be 0, and depth slice d is at linear + d x depth_pitch).
template <bool Into>
bool copy_subresource(const Layout& layout, std::uint32_t plane_index, std::uint32_t mip_index, std::uint32_t slice,
                      std::byte* resource, std::size_t resource_bytes, std::byte* linear, std::uint64_t row_pitch,
                      std::uint64_t depth_pitch) {
    if (resource == nullptr || linear == nullptr || plane_index >= layout.plane_count) return false;
    if (resource_bytes < layout.size) return false;
    const Plane& plane = layout.planes[plane_index];
    if (mip_index >= plane.mip_count) return false;
    const bool volume = layout.desc.dimension == dimension_3d;
    if (volume ? slice != 0 : slice >= plane.slices) return false;
    const Mip& mip = plane.mips[mip_index];
    const std::uint32_t layers = volume ? std::max(1u, plane.slices >> mip_index) : 1u;
    const std::uint64_t row_bytes = std::uint64_t{mip.width_elements} * plane.bytes_per_element;
    if (row_pitch < row_bytes) return false;
    if (layers > 1 && depth_pitch < row_pitch * mip.height_elements) return false;

    std::array<std::uint32_t, 256> x_part{};
    const bool tiled_rows = plane.block_bytes != 0 && !plane.thick;
    if (tiled_rows) {
        for (std::uint32_t x = 0; x < plane.swizzle.tile_width; ++x) x_part[x] = deposit(x, plane.swizzle.x_mask);
    }

    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        const std::uint32_t layout_slice = slice + layer;
        const std::uint64_t slice_start = plane.offset + std::uint64_t{layout_slice} * plane.slice_stride;
        for (std::uint32_t y = 0; y < mip.height_elements; ++y) {
            std::byte* row = linear + layer * depth_pitch + std::uint64_t{y} * row_pitch;
            if (plane.block_bytes == 0) {
                std::byte* start = resource + element(plane, mip_index, layout_slice, 0, y);
                if constexpr (Into) {
                    std::memcpy(start, row, static_cast<std::size_t>(row_bytes));
                } else {
                    std::memcpy(row, start, static_cast<std::size_t>(row_bytes));
                }
            } else if (tiled_rows) {
                copy_tiled_row_any<Into>(plane, mip, slice_start, y, x_part.data(), resource, row);
            } else {
                for (std::uint32_t x = 0; x < mip.width_elements; ++x) {
                    std::byte* at = resource + element(plane, mip_index, layout_slice, x, y);
                    std::byte* to = row + std::size_t{x} * plane.bytes_per_element;
                    if constexpr (Into) {
                        std::memcpy(at, to, plane.bytes_per_element);
                    } else {
                        std::memcpy(to, at, plane.bytes_per_element);
                    }
                }
            }
        }
    }
    return true;
}

}  // namespace detail

// GetMipLevelOffsetBytes: the offset from the resource start of a mip of array slice 0, or 0 when it does not exist.
inline std::uint64_t mip_offset(const Layout& layout, std::uint32_t plane_index, std::uint32_t mip_index) {
    if (plane_index >= layout.plane_count || mip_index >= layout.planes[plane_index].mip_count) return 0;
    const Plane& plane = layout.planes[plane_index];
    return plane.offset + plane.mips[mip_index].offset;
}

// GetTexelElementOffsetBytes: the byte offset from the resource start of element (x, y) (in elements: blocks for BC
// formats) of a mip. For a 3D texture the depth is z and slice must be 0; for other textures the array slice is slice
// and z must be 0. Like XG it answers 0 when the plane, mip or slice does not exist; x and y are not range-checked.
inline std::uint64_t element_offset(const Layout& layout, std::uint32_t plane_index, std::uint32_t mip_index,
                                    std::uint32_t slice, std::uint64_t x_elem, std::uint32_t y_elem,
                                    std::uint32_t z) {
    if (plane_index >= layout.plane_count) return 0;
    const Plane& plane = layout.planes[plane_index];
    const bool volume = layout.desc.dimension == dimension_3d;
    if (volume ? slice != 0 : z != 0) return 0;
    const std::uint32_t layer = volume ? z : slice;
    if (mip_index >= plane.mip_count || layer >= plane.slices || x_elem > std::numeric_limits<std::uint32_t>::max()) {
        return 0;
    }
    return detail::element(plane, mip_index, layer, static_cast<std::uint32_t>(x_elem), y_elem);
}

// CopyFromSubresource: one mip of one array slice (of a 3D texture: every depth slice of the mip, slice 0) from the
// whole resource's console-layout bytes into rows of row_pitch bytes (rows of elements: block rows for BC formats),
// depth slices depth_pitch bytes apart. False when the plane, mip or slice does not exist, the memory is smaller than
// the layout, or the pitches cannot hold the mip.
inline bool detile_subresource(const Layout& layout, std::uint32_t plane_index, std::uint32_t mip_index,
                               std::uint32_t slice, std::span<const std::byte> console_memory, std::byte* linear,
                               std::uint64_t row_pitch, std::uint64_t depth_pitch) {
    // The From direction only reads the resource.
    return detail::copy_subresource<false>(layout, plane_index, mip_index, slice,
                                           const_cast<std::byte*>(console_memory.data()), console_memory.size(), linear,
                                           row_pitch, depth_pitch);
}

// CopyIntoSubresource: the reverse of detile_subresource, row-major data into the console layout.
inline bool tile_subresource(const Layout& layout, std::uint32_t plane_index, std::uint32_t mip_index,
                             std::uint32_t slice, const std::byte* linear, std::uint64_t row_pitch,
                             std::uint64_t depth_pitch, std::span<std::byte> console_memory) {
    // The Into direction only reads the linear data.
    return detail::copy_subresource<true>(layout, plane_index, mip_index, slice, console_memory.data(),
                                          console_memory.size(), const_cast<std::byte*>(linear), row_pitch,
                                          depth_pitch);
}

}  // namespace leonida::xg_console
