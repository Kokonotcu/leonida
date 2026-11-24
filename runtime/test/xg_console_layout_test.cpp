// The console (XG) texture layout (src/xg_console_layout.h), without a GPU or game data.
//
// The first cases are five-star-recomp's tests/FiveStar.Loader.Tests/XgTests.cs (GPL-3.0-or-later): the sizes and mip
// places of a 2048x2048 BC1 texture in 64 KB blocks and a 256x128 one in 4 KB blocks (the spans those textures occupy
// in the shipped page files), element offsets through the 64 KB blocks and the tail slot, the suggested mode, and the
// thick 256x32x8 R16_FLOAT volume. Its two dump tests (re-tiling shipped textures byte for byte) need the game's page
// files and are replaced by synthetic round trips: every subresource of a resource is tiled with distinct bytes, then
// each detiles back unchanged and every element sits at its element_offset. The 64x64 RGBA8 round trip is the CPU part
// of five-star's D3D12XGpuResourceTests.cs (its GPU read-back is not ported). The GetResourceTiling checks of the
// 64 KB case are left out with the tiling code.

#include "../src/xg_console_layout.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace xg = leonida::xg_console;

namespace {

int failures = 0;

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition);     \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

#define CHECK_EQ(actual, expected)                                                                             \
    do {                                                                                                       \
        const auto actual_value = static_cast<unsigned long long>(actual);                                     \
        const auto expected_value = static_cast<unsigned long long>(expected);                                 \
        if (actual_value != expected_value) {                                                                  \
            std::fprintf(stderr, "line %d: %s = 0x%llx, expected 0x%llx\n", __LINE__, #actual, actual_value,   \
                         expected_value);                                                                      \
            ++failures;                                                                                        \
        }                                                                                                      \
    } while (0)

constexpr std::uint32_t bc1 = 71;
constexpr std::uint32_t rgba8 = 28;
constexpr std::uint32_t large_64kb_r_x = 0x100 + 27;
constexpr std::uint32_t small_4kb_r_x = 0x100 + 23;

xg::Desc texture(std::uint64_t width, std::uint32_t height, std::uint16_t mips, std::uint32_t layout,
                 std::uint32_t format = bc1, std::uint16_t slices = 1, std::uint32_t dimension = xg::dimension_2d) {
    xg::Desc desc;
    desc.dimension = dimension;
    desc.width = width;
    desc.height = height;
    desc.depth_or_array_size = slices;
    desc.mip_levels = mips;
    desc.format = format;
    desc.sample_count = 1;
    desc.layout = layout;
    return desc;
}

// The offset of element (x, y) of a mip of plane 0, slice 0, as the game calls slot 6.
std::uint64_t texel(const xg::Layout& layout, std::uint32_t mip, std::uint64_t x, std::uint32_t y) {
    return xg::element_offset(layout, 0, mip, 0, x, y, 0);
}

// A 2048x2048 BC1 texture with 7 mips in 64 KB blocks: mips 3-6 share the tail block.
void large_texture_sizes_and_mip_places() {
    const auto layout = xg::compute_layout(texture(2048, 2048, 7, large_64kb_r_x));
    CHECK(layout.has_value());
    if (!layout) return;
    const xg::Plane& plane = layout->planes[0];

    // Mip 0 (512x512 blocks) ends the texture; mip 1 follows the tail block and mip 2; mip 3 is the first tail mip,
    // half a block in.
    CHECK_EQ(layout->size, 0x2b0000);
    CHECK_EQ(plane.slice_stride, 0x2b0000);
    CHECK_EQ(xg::mip_offset(*layout, 0, 0), 0xb0000);
    CHECK_EQ(plane.mips[0].size, 0x200000);
    CHECK_EQ(plane.mips[0].pitch_bytes, 4096);
    CHECK_EQ(xg::mip_offset(*layout, 0, 1), 0x10000 + 0x20000);
    CHECK_EQ(xg::mip_offset(*layout, 0, 3), 0x8000);
    CHECK_EQ(plane.first_tail_mip, 3);
    CHECK_EQ(layout->alignment, 0x10000);
}

void large_texture_element_offsets() {
    const auto layout = xg::compute_layout(texture(2048, 2048, 7, large_64kb_r_x));
    CHECK(layout.has_value());
    if (!layout) return;

    // Mip 0 starts at 0xb0000 with 4 blocks of 128x64 elements per row; mip 3 is the first tail mip, half a block in.
    CHECK_EQ(layout->size, 0x2b0000);
    CHECK_EQ(xg::mip_offset(*layout, 0, 0), 0xb0000);
    CHECK_EQ(texel(*layout, 0, 0, 0), 0xb0000);
    CHECK_EQ(texel(*layout, 0, 128, 0), 0xc0000);
    CHECK_EQ(texel(*layout, 0, 0, 64), 0xf0000);
    CHECK_EQ(texel(*layout, 3, 0, 0), 0x8000);
    CHECK_EQ(texel(*layout, 7, 0, 0), 0);
}

// A 256x128 BC1 texture with 8 mips in 4 KB blocks: mips 2-7 share the tail block.
void small_texture_size_and_suggestion() {
    const xg::Desc desc = texture(256, 128, 8, small_4kb_r_x);
    const auto layout = xg::compute_layout(desc);
    CHECK(layout.has_value());
    if (!layout) return;

    CHECK_EQ(layout->size, 0x6000);
    CHECK_EQ(xg::mip_offset(*layout, 0, 0), 0x2000);
    CHECK_EQ(layout->planes[0].first_tail_mip, 2);

    xg::Desc unknown = desc;
    unknown.layout = 0;
    CHECK_EQ(xg::suggest_layout(unknown), small_4kb_r_x);
    // 1 MiB of mip 0 or more gets 64 KB blocks.
    CHECK_EQ(xg::suggest_layout(texture(1024, 1024, 1, 0, rgba8)), large_64kb_r_x);
    CHECK_EQ(xg::suggest_layout(texture(64, 64, 1, 0, rgba8)), small_4kb_r_x);
}

// 256x32x8 R16_FLOAT in SW_4KB_S: 8x16x16 bricks (x, z, y, z, y, x, ...), so the 8 slices pad to 16: 0x40000 bytes,
// the size of the game's archive chunk for it (the 2D-slice layout gave 0x20000).
void thick_volume_bricks() {
    const auto layout = xg::compute_layout(texture(256, 32, 1, 0x105, 54, 8, xg::dimension_3d));
    CHECK(layout.has_value());
    if (!layout) return;
    CHECK(layout->planes[0].thick);
    CHECK_EQ(layout->size, 0x40000);
    CHECK_EQ(xg::element_offset(*layout, 0, 0, 0, 1, 0, 0), 2);
    CHECK_EQ(xg::element_offset(*layout, 0, 0, 0, 0, 0, 1), 4);
    CHECK_EQ(xg::element_offset(*layout, 0, 0, 0, 0, 1, 0), 8);
    CHECK_EQ(xg::element_offset(*layout, 0, 0, 0, 8, 0, 0), 0x1000);
    CHECK_EQ(xg::element_offset(*layout, 0, 0, 0, 0, 16, 0), 0x20000);
}

// The rules of docs/xg.md that the five-star tests do not pin down, worked out by hand.
void layout_rules() {
    CHECK(!xg::is_console_tiled(0));
    CHECK(!xg::is_console_tiled(1));
    CHECK(!xg::is_console_tiled(2));
    CHECK(xg::is_console_tiled(0x100));
    CHECK(xg::is_console_tiled(0x102));
    CHECK(xg::is_console_tiled(0x10A));
    CHECK(!xg::compute_layout(texture(64, 64, 1, 0, rgba8)).has_value());
    CHECK(!xg::compute_layout(texture(64, 64, 1, 1, rgba8)).has_value());
    xg::Desc msaa = texture(64, 64, 1, small_4kb_r_x, rgba8);
    msaa.sample_count = 4;
    CHECK(!xg::compute_layout(msaa).has_value());

    // Mode 0: rows padded to 256 bytes, mips on 512 bytes, in order from mip 0.
    if (const auto linear = xg::compute_layout(texture(100, 10, 3, 0x100, rgba8))) {
        const xg::Plane& plane = linear->planes[0];
        CHECK_EQ(plane.block_bytes, 0);
        CHECK_EQ(plane.mips[0].pitch_bytes, 512);
        CHECK_EQ(plane.mips[0].offset, 0);
        CHECK_EQ(plane.mips[1].offset, 5120);
        CHECK_EQ(plane.mips[1].pitch_bytes, 256);
        CHECK_EQ(plane.mips[2].offset, 6656);
        CHECK_EQ(linear->size, 7168);
        CHECK_EQ(texel(*linear, 1, 3, 2), 5120 + 2 * 256 + 3 * 4);
    } else {
        CHECK(false);
    }

    // 12-byte elements have no swizzle: linear even in a tiled mode, rows of 64 elements.
    if (const auto rgb32 = xg::compute_layout(texture(10, 4, 1, small_4kb_r_x, 6))) {
        CHECK_EQ(rgb32->planes[0].block_bytes, 0);
        CHECK_EQ(rgb32->planes[0].mips[0].pitch_bytes, 768);
    } else {
        CHECK(false);
    }

    // A chain of one mip has no tail: a 16x16 BC1 texture starts at offset 0 and fills one 4 KB block.
    if (const auto lone = xg::compute_layout(texture(16, 16, 1, small_4kb_r_x))) {
        CHECK_EQ(lone->planes[0].first_tail_mip, 1);
        CHECK_EQ(lone->size, 0x1000);
        CHECK_EQ(texel(*lone, 0, 0, 0), 0);
    } else {
        CHECK(false);
    }

    // D32_FLOAT_S8X24_UINT: 4-byte depth in 32x32 blocks (2x2 of them), then a one-byte stencil plane in one block.
    if (const auto depth = xg::compute_layout(texture(64, 64, 1, small_4kb_r_x, 20))) {
        CHECK_EQ(depth->plane_count, 2);
        CHECK_EQ(depth->planes[0].size, 0x4000);
        CHECK_EQ(depth->planes[1].offset, 0x4000);
        CHECK_EQ(depth->planes[1].bytes_per_element, 1);
        CHECK_EQ(depth->size, 0x5000);
    } else {
        CHECK(false);
    }

    // 3D in an R mode keeps 2D depth slices: each depth slice is a full chain.
    if (const auto thin = xg::compute_layout(texture(32, 32, 2, small_4kb_r_x, rgba8, 4, xg::dimension_3d))) {
        CHECK(!thin->planes[0].thick);
        CHECK_EQ(thin->size, thin->planes[0].slice_stride * 4);
    } else {
        CHECK(false);
    }

    // Swizzle checks: the 1-byte 64 KB pattern takes y bit 1 before y bit 0, the 256-byte block is a 4 KB prefix.
    const xg::Swizzle one_byte_large = xg::detail::find_swizzle(xg::large_block_bytes, 0);
    CHECK_EQ(one_byte_large.tile_width, 256);
    CHECK_EQ(one_byte_large.tile_height, 256);
    CHECK_EQ(one_byte_large.index(0, 1), 1u << 4);
    CHECK_EQ(one_byte_large.index(0, 2), 1u << 3);
    const xg::Swizzle micro = xg::detail::find_swizzle(xg::micro_block_bytes, 2);
    CHECK_EQ(micro.tile_width, 8);
    CHECK_EQ(micro.tile_height, 8);
}

std::byte pattern_byte(std::uint64_t index, std::uint32_t seed) {
    return static_cast<std::byte>((index * 7) + (index / 251) + (seed * 31));
}

// Tiles every subresource of a desc with its own bytes, then detiles each back and checks it, and checks that each
// element sits at its element_offset inside the layout's size. Overlapping subresources would break the read-back.
void round_trip(const xg::Desc& desc, const char* name) {
    const auto layout = xg::compute_layout(desc);
    if (!layout) {
        std::fprintf(stderr, "%s: no layout\n", name);
        ++failures;
        return;
    }
    std::vector<std::byte> memory(static_cast<std::size_t>(layout->size), std::byte{0xCD});
    const bool volume = desc.dimension == xg::dimension_3d;
    std::uint32_t seed = 0;
    int bad = 0;

    for (int pass = 0; pass < 2; ++pass) {
        seed = 0;
        for (std::uint32_t plane_index = 0; plane_index < layout->plane_count; ++plane_index) {
            const xg::Plane& plane = layout->planes[plane_index];
            const std::uint32_t slices = volume ? 1 : plane.slices;
            for (std::uint32_t slice = 0; slice < slices; ++slice) {
                for (std::uint32_t mip_index = 0; mip_index < plane.mip_count; ++mip_index) {
                    const xg::Mip& mip = plane.mips[mip_index];
                    const std::uint32_t layers = volume ? std::max(1u, plane.slices >> mip_index) : 1u;
                    // A padded pitch, as a D3D12 footprint has.
                    const std::uint64_t row_pitch = std::uint64_t{mip.width_elements} * plane.bytes_per_element + 16;
                    const std::uint64_t depth_pitch = row_pitch * mip.height_elements;
                    std::vector<std::byte> linear(static_cast<std::size_t>(depth_pitch * layers));
                    for (std::size_t i = 0; i < linear.size(); ++i) linear[i] = pattern_byte(i, seed);
                    ++seed;

                    if (pass == 0) {
                        if (!xg::tile_subresource(*layout, plane_index, mip_index, slice, linear.data(), row_pitch,
                                                  depth_pitch, memory)) {
                            std::fprintf(stderr, "%s: tile failed (plane %u, slice %u, mip %u)\n", name, plane_index,
                                         slice, mip_index);
                            ++failures;
                        }
                        continue;
                    }

                    std::vector<std::byte> back(linear.size(), std::byte{0});
                    if (!xg::detile_subresource(*layout, plane_index, mip_index, slice, memory, back.data(),
                                                row_pitch, depth_pitch)) {
                        std::fprintf(stderr, "%s: detile failed (plane %u, slice %u, mip %u)\n", name, plane_index,
                                     slice, mip_index);
                        ++failures;
                        continue;
                    }
                    for (std::uint32_t layer = 0; layer < layers; ++layer) {
                        for (std::uint32_t y = 0; y < mip.height_elements; ++y) {
                            for (std::uint32_t x = 0; x < mip.width_elements; ++x) {
                                const std::size_t at = static_cast<std::size_t>(layer * depth_pitch + y * row_pitch +
                                                                                std::uint64_t{x} * plane.bytes_per_element);
                                const std::uint64_t offset =
                                    xg::element_offset(*layout, plane_index, mip_index, volume ? 0 : slice, x, y,
                                                       volume ? layer : 0);
                                const bool inside = offset + plane.bytes_per_element <= layout->size;
                                bool same = inside;
                                for (std::uint32_t b = 0; b < plane.bytes_per_element; ++b) {
                                    same = same && back[at + b] == linear[at + b] &&
                                           memory[static_cast<std::size_t>(offset) + b] == linear[at + b];
                                }
                                if (!same && bad++ < 4) {
                                    std::fprintf(stderr,
                                                 "%s: plane %u slice %u mip %u layer %u element (%u, %u) at 0x%llx "
                                                 "differs\n",
                                                 name, plane_index, slice, mip_index, layer, x, y,
                                                 static_cast<unsigned long long>(offset));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (bad != 0) {
        std::fprintf(stderr, "%s: %d elements differ\n", name, bad);
        ++failures;
    }
}

// Five-star's D3D12XGpuResourceTests: a 64x64 RGBA8 texture with 2 mips in the suggested mode is tiled by the CPU
// and must read back linear; it fits the two 64 KB pages the test maps.
void rgba8_suggested_round_trip() {
    xg::Desc desc = texture(64, 64, 2, 0, rgba8);
    desc.layout = xg::suggest_layout(desc);
    CHECK_EQ(desc.layout, small_4kb_r_x);
    const auto layout = xg::compute_layout(desc);
    CHECK(layout.has_value());
    if (!layout) return;
    CHECK(layout->size <= 0x20000);
    round_trip(desc, "rgba8 64x64 suggested");
}

void round_trips() {
    // The five-star test textures.
    round_trip(texture(2048, 2048, 7, large_64kb_r_x), "bc1 2048 64kb");
    round_trip(texture(256, 128, 8, small_4kb_r_x), "bc1 256x128 4kb");
    round_trip(texture(256, 32, 1, 0x105, 54, 8, xg::dimension_3d), "r16f volume 4kb_s");

    // Every element size in 256 B, 4 KB and 64 KB blocks and linear, with arrays, odd sizes and full chains.
    const std::uint32_t formats[] = {61 /* R8 */, 54 /* R16F */, rgba8, 10 /* RGBA16F */, 2 /* RGBA32F */,
                                     6 /* RGB32F */, bc1, 98 /* BC7 */};
    const std::uint32_t layouts[] = {0x100, 0x102, 0x106, small_4kb_r_x, 0x10A, large_64kb_r_x};
    for (const std::uint32_t format : formats) {
        for (const std::uint32_t layout : layouts) {
            char name[64];
            std::snprintf(name, sizeof name, "format %u layout 0x%x", format, layout);
            round_trip(texture(100, 37, 0, layout, format, 3), name);
            round_trip(texture(512, 8, 0, layout, format, 1), name);
        }
    }
    // Depth-stencil planes, a 1D array, thin and thick volumes with mips (thick in 4 KB and 64 KB S and Z modes).
    round_trip(texture(80, 48, 0, small_4kb_r_x, 20, 2), "d32s8 array");
    round_trip(texture(64, 64, 3, large_64kb_r_x, 45), "d24s8 64kb");
    round_trip(texture(300, 1, 0, small_4kb_r_x, rgba8, 4, xg::dimension_1d), "rgba8 1d array");
    round_trip(texture(40, 24, 0, small_4kb_r_x, rgba8, 9, xg::dimension_3d), "rgba8 thin volume");
    round_trip(texture(40, 24, 0, 0x105, rgba8, 9, xg::dimension_3d), "rgba8 thick 4kb_s volume");
    round_trip(texture(40, 24, 0, 0x108, 61, 9, xg::dimension_3d), "r8 thick 64kb_z volume");
    round_trip(texture(20, 12, 0, 0x109, 2, 5, xg::dimension_3d), "rgba32f thick 64kb_s volume");
}

void rejects() {
    const auto layout = xg::compute_layout(texture(64, 64, 2, small_4kb_r_x, rgba8, 2));
    CHECK(layout.has_value());
    if (!layout) return;
    std::vector<std::byte> memory(static_cast<std::size_t>(layout->size));
    std::vector<std::byte> linear(64 * 64 * 4);
    CHECK(xg::detile_subresource(*layout, 0, 0, 1, memory, linear.data(), 256, 0));
    CHECK(!xg::detile_subresource(*layout, 1, 0, 0, memory, linear.data(), 256, 0));
    CHECK(!xg::detile_subresource(*layout, 0, 2, 0, memory, linear.data(), 256, 0));
    CHECK(!xg::detile_subresource(*layout, 0, 0, 2, memory, linear.data(), 256, 0));
    CHECK(!xg::detile_subresource(*layout, 0, 0, 0, memory, linear.data(), 255, 0));
    CHECK(!xg::detile_subresource(*layout, 0, 0, 0, std::span<const std::byte>(memory).first(memory.size() - 1),
                                  linear.data(), 256, 0));
    CHECK_EQ(xg::element_offset(*layout, 0, 0, 2, 0, 0, 0), 0);
    CHECK_EQ(xg::element_offset(*layout, 0, 0, 0, 0, 0, 1), 0);
    CHECK_EQ(xg::element_offset(*layout, 1, 0, 0, 0, 0, 0), 0);
}

}  // namespace

int main() {
    large_texture_sizes_and_mip_places();
    large_texture_element_offsets();
    small_texture_size_and_suggestion();
    thick_volume_bricks();
    layout_rules();
    rgba8_suggested_round_trip();
    round_trips();
    rejects();
    if (failures != 0) {
        std::fprintf(stderr, "xg_console_layout_test: %d failures\n", failures);
        return 1;
    }
    std::printf("xg_console_layout_test: all cases passed\n");
    return 0;
}
