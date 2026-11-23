// Value-level replay regression: full and rectangular integer clears must carry integer values, not float bits,
// and normalised clears are rounded to nearest even like D3D (0.5 on 8 bits is 128).
// No Vulkan loader, GPU, shaders or game data. Expected conversions also match the WARP differential GPU tests.
#include "../src/vulkan_replay.h"
#include "../src/uav_clear.h"
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace {
VkClearColorValue captured{};
unsigned calls = 0;
VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL clear(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* color,
    std::uint32_t, const VkImageSubresourceRange*) { captured = *color; ++calls; }
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t n, const VkClearAttachment* a,
    std::uint32_t, const VkClearRect*) { if (n == 1) { captured = a[0].clearValue.color; ++calls; } }

bool replay_color(VkFormat format, const leonida::ClearColor& color, bool rectangle) {
    leonida::CommandList list;
    const leonida::Rect rect{1, 1, 3, 3};
    if (!list.clear_render_target({1, 0, 0, 1}, color,
        rectangle ? std::span<const leonida::Rect>(&rect, 1) : std::span<const leonida::Rect>{}) || !list.close()) return false;
    leonida::ResourceBinding binding{};
    binding.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    binding.format = format;
    binding.width = binding.height = 4;
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.image_view = [](const auto&, const auto&) { return reinterpret_cast<VkImageView>(std::uintptr_t{1}); };
    calls = 0;
    return leonida::replay(list, VK_NULL_HANDLE, [&](auto) { return &binding; }, vk) == leonida::ReplayResult::success && calls == 1;
}

// Integer RTV format with its component range; A2B10G10R10 has a 2-bit alpha.
struct IntegerFormat { VkFormat format; const char* name; double low, high, alpha_high; bool is_signed; };

// Reference conversion measured against WARP: NaN -> 0, clamp to the range, truncate fractions toward zero.
std::uint32_t expected_bits(float input, double low, double high, bool is_signed) {
    const double v = double(input);
    const double r = v != v ? 0.0 : v <= low ? low : v >= high ? high : std::trunc(v);
    return is_signed ? static_cast<std::uint32_t>(static_cast<std::int32_t>(r)) : static_cast<std::uint32_t>(r);
}
} // namespace

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    const leonida::ClearColor cases[] = {
        {1, 2, 3, 4},
        {1.5f, 2.5f, 3.5f, 4.5f},
        {-1, -0.0f, 254.5f, 255.5f},
        {256, 65536, 16777216, 4294967296.0f},
        {nan, inf, -inf, 0.5f},
        {0.499f, 0.501f, 1.499f, 1.501f},
        {4294967040.0f, std::numeric_limits<float>::max(), 255, 0},
        // Component-range boundaries for the 8-, 16-, 10- and 2-bit formats and the signed 32-bit limits.
        {127.5f, 128, -128.5f, -129},
        {32767.5f, 32768, -32768.5f, -32769},
        {65535.5f, 65536, 1023.5f, 1024},
        {3.5f, 4, 2147483520.0f, -2147483648.0f},
    };
    constexpr double u32max = 4294967295.0, s32min = -2147483648.0, s32max = 2147483647.0;
    const IntegerFormat integer_formats[] = {
        {VK_FORMAT_R8_UINT, "R8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R8G8_UINT, "R8G8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R8G8B8A8_UINT, "R8G8B8A8_UINT", 0, 255, 255, false},
        {VK_FORMAT_R16_UINT, "R16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R16G16_UINT, "R16G16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R16G16B16A16_UINT, "R16G16B16A16_UINT", 0, 65535, 65535, false},
        {VK_FORMAT_R32_UINT, "R32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_R32G32_UINT, "R32G32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_R32G32B32A32_UINT, "R32G32B32A32_UINT", 0, u32max, u32max, false},
        {VK_FORMAT_A2B10G10R10_UINT_PACK32, "A2B10G10R10_UINT_PACK32", 0, 1023, 3, false},
        {VK_FORMAT_R8_SINT, "R8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R8G8_SINT, "R8G8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R8G8B8A8_SINT, "R8G8B8A8_SINT", -128, 127, 127, true},
        {VK_FORMAT_R16_SINT, "R16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R16G16_SINT, "R16G16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R16G16B16A16_SINT, "R16G16B16A16_SINT", -32768, 32767, 32767, true},
        {VK_FORMAT_R32_SINT, "R32_SINT", s32min, s32max, s32max, true},
        {VK_FORMAT_R32G32_SINT, "R32G32_SINT", s32min, s32max, s32max, true},
        {VK_FORMAT_R32G32B32A32_SINT, "R32G32B32A32_SINT", s32min, s32max, s32max, true},
    };
    unsigned checks = 0, failed_formats = 0;
    for (const auto& f : integer_formats) {
        bool format_ok = true;
        unsigned case_index = 0;
        for (const auto& c : cases) {
            for (const bool rect : {false, true}) {
                if (!replay_color(f.format, c, rect)) {
                    std::fprintf(stderr, "%s: replay failed (case %u rect %d)\n", f.name, case_index, int(rect));
                    return 1;
                }
                for (unsigned i = 0; i < 4 && format_ok; ++i) {
                    const std::uint32_t expected = expected_bits(c[i], f.low, i == 3 ? f.alpha_high : f.high, f.is_signed);
                    if (captured.uint32[i] != expected) {
                        std::fprintf(stderr, "integer clear %s case %u rect %d channel %u: got 0x%08x, expected 0x%08x\n",
                            f.name, case_index, int(rect), i, captured.uint32[i], expected);
                        format_ok = false;
                    }
                }
                if (!format_ok) break;
                ++checks;
            }
            if (!format_ok) break;
            ++case_index;
        }
        if (!format_ok) ++failed_formats;
    }
    for (const auto& c : cases) {
        for (const VkFormat format : {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT}) {
            for (const bool rect : {false, true}) {
                if (!replay_color(format, c, rect) || std::memcmp(captured.float32, c.data(), sizeof(captured)) != 0) {
                    std::fprintf(stderr, "float clear: format %d rect %d changed the value\n", int(format), int(rect));
                    return 1;
                }
                ++checks;
            }
        }
    }
    // Normalised formats: D3D rounds to nearest even, the clamp and NaN rules are 0..1 and 0, and an exact tie goes to
    // the even neighbour. NVIDIA rounds 0.5 on 8 bits down to 127 by itself, so the value is rounded before Vulkan sees it.
    struct Norm {
        VkFormat format;
        leonida::ClearColor color;
        std::array<std::int32_t, 4> steps;  // what D3D writes, in units of the channel's largest value
        std::array<double, 4> maximum;
    };
    const Norm norms[] = {
        {VK_FORMAT_R8G8B8A8_UNORM, {0.5f, 0.499f, 0.501f, 1.0f}, {128, 127, 128, 255}, {255, 255, 255, 255}},
        {VK_FORMAT_B8G8R8A8_UNORM, {-1, 2, nan, 0.5f}, {0, 255, 0, 128}, {255, 255, 255, 255}},
        {VK_FORMAT_R8G8_UNORM, {0.5f, 1.5f, 0, 0}, {128, 255, 0, 0}, {255, 255, 255, 255}},
        {VK_FORMAT_A2B10G10R10_UNORM_PACK32, {0.5f, 0.5f, 0.5f, 0.5f}, {512, 512, 512, 2}, {1023, 1023, 1023, 3}},
        {VK_FORMAT_R16G16B16A16_UNORM, {0.5f, 0.25f, 0.75f, 1.0f}, {32768, 16384, 49151, 65535},
         {65535, 65535, 65535, 65535}},
        {VK_FORMAT_R8G8B8A8_SRGB, {0.2f, 0.7f, 0.5f, 0.5f}, {0, 0, 0, 128}, {0, 0, 0, 255}},  // colour: the GPU's sRGB curve
        {VK_FORMAT_B8G8R8A8_SRGB, {0.5f, 0.25f, 0.75f, 0.499f}, {0, 0, 0, 127}, {0, 0, 0, 255}},
        {VK_FORMAT_R16_UNORM, {0.5f, 0, 0, 0}, {32768, 0, 0, 0}, {65535, 65535, 65535, 65535}},
        // Signed normalised: -1.0 is -127 (-32767), the scale is 127 (32767) and ties go to the even neighbour.
        {VK_FORMAT_R8G8B8A8_SNORM, {-1, -0.5f, 0.5f, 1}, {-127, -64, 64, 127}, {127, 127, 127, 127}},
        {VK_FORMAT_R8G8_SNORM, {nan, inf, -inf, -0.5f}, {0, 127, -127, -64}, {127, 127, 127, 127}},
        {VK_FORMAT_R16G16B16A16_SNORM, {-2, 2, 0.001f, -0.001f}, {-32767, 32767, 33, -33},
         {32767, 32767, 32767, 32767}},
    };
    for (const auto& n : norms) {
        for (const bool rect : {false, true}) {
            if (!replay_color(n.format, n.color, rect)) return 1;
            for (unsigned i = 0; i < 4; ++i) {
                const float expected = n.maximum[i] != 0.0 ? static_cast<float>(double(n.steps[i]) / n.maximum[i]) : n.color[i];
                if (captured.float32[i] != expected) {
                    std::fprintf(stderr, "normalised clear: format %d rect %d channel %u: got %.9g, expected %.9g\n",
                        int(n.format), int(rect), i, double(captured.float32[i]), double(expected));
                    return 1;
                }
            }
            ++checks;
        }
    }
    // The other integer formats clamp to the component range and truncate toward zero (what WARP writes).
    struct Integer {
        VkFormat format;
        leonida::ClearColor color;
        std::array<std::int32_t, 4> expected;
    };
    const Integer integers[] = {
        {VK_FORMAT_R8_SINT, {-1.5f, -2.5f, 2.5f, 3.5f}, {-1, -2, 2, 3}},
        {VK_FORMAT_R8G8B8A8_SINT, {-129, 128, -32769, 32768}, {-128, 127, -128, 127}},
        {VK_FORMAT_R8G8B8A8_SINT, {2147483648.0f, -2147483904.0f, nan, inf}, {127, -128, 0, 127}},
        {VK_FORMAT_R16_SINT, {-129, 128, -32769, 32768}, {-129, 128, -32768, 32767}},
        {VK_FORMAT_R32_SINT, {-129, 128, -32769, 32768}, {-129, 128, -32769, 32768}},
        {VK_FORMAT_R32_SINT, {2147483648.0f, -2147483904.0f, nan, inf}, {2147483647, -2147483647 - 1, 0, 2147483647}},
        {VK_FORMAT_R32G32B32A32_SINT, {-inf, 0.5f, -0.5f, 100000.0f}, {-2147483647 - 1, 0, 0, 100000}},
        {VK_FORMAT_R8_UINT, {256, 1.5f, -1, nan}, {255, 1, 0, 0}},
        {VK_FORMAT_R16_UINT, {65536, 65535.5f, -0.0f, 2.5f}, {65535, 65535, 0, 2}},
        {VK_FORMAT_R8G8_UINT, {256, 1.5f, -1, nan}, {255, 1, 0, 0}},
        {VK_FORMAT_R16G16_SINT, {-32769, 32768, -1.5f, nan}, {-32768, 32767, -1, 0}},
        {VK_FORMAT_R16G16B16A16_UINT, {65536, 65535.5f, -0.0f, 2.5f}, {65535, 65535, 0, 2}},
        // R10G10B10A2_UINT: each channel clamps to its own width, 10, 10, 10 and 2 bits.
        {VK_FORMAT_A2B10G10R10_UINT_PACK32, {1023.5f, 2000, 1024, 4}, {1023, 1023, 1023, 3}},
        {VK_FORMAT_A2B10G10R10_UINT_PACK32, {5.5f, 1.5f, 1022.5f, 2.5f}, {5, 1, 1022, 2}},
    };
    for (const auto& c : integers) {
        for (const bool rect : {false, true}) {
            if (!replay_color(c.format, c.color, rect)) return 1;
            for (unsigned i = 0; i < 4; ++i) {
                if (captured.int32[i] != c.expected[i]) {
                    std::fprintf(stderr, "integer clear: format %d rect %d channel %u: got %d, expected %d\n",
                        int(c.format), int(rect), i, captured.int32[i], c.expected[i]);
                    return 1;
                }
            }
            ++checks;
        }
    }
    // Ties go to even whatever the thread's floating-point rounding mode (std::nearbyint would follow it: 127.5 is
    // 127 when rounding down). Converted under each mode, checked after the default mode is back.
    {
        const int mode = std::fegetround();
        VkClearColorValue unorm[3]{}, snorm[3]{};
        const int modes[3] = {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};
        for (int m = 0; m < 3; ++m) {
            std::fesetround(modes[m]);
            unorm[m] = leonida::detail::render_target_clear_color(VK_FORMAT_R8G8B8A8_UNORM, {0.5f, 0.5f, 0.5f, 0.5f});
            snorm[m] = leonida::detail::render_target_clear_color(VK_FORMAT_R8G8B8A8_SNORM, {-0.5f, 0.5f, -0.5f, 0.5f});
        }
        std::fesetround(mode);
        for (int m = 0; m < 3; ++m) {
            const long u = std::lround(double(unorm[m].float32[0]) * 255.0);
            const long s0 = std::lround(double(snorm[m].float32[0]) * 127.0);
            const long s1 = std::lround(double(snorm[m].float32[1]) * 127.0);
            if (u != 128 || s0 != -64 || s1 != 64) {
                std::fprintf(stderr, "tie rounding under FP mode %d: UNORM %ld (128), SNORM %ld %ld (-64 64)\n",
                             modes[m], u, s0, s1);
                return 1;
            }
            ++checks;
        }
    }
    // Measured against WARP by d3d12_clear_fuzz_test, where NVIDIA (621.14) converts otherwise when handed the
    // application's float: the packed 16-bit formats (through 8 bits there) and denormals.
    {
        struct Exact {
            VkFormat format;
            unsigned channel;
            float input;
            double code, maximum;  // the expected float is code / maximum
            const char* what;
        };
        const Exact exact[] = {
            {VK_FORMAT_R5G6B5_UNORM_PACK16, 1, 0.373858f, 24, 63, "B5G6R5 green 0.373858 is 24 (NVIDIA: 23)"},
            {VK_FORMAT_R5G6B5_UNORM_PACK16, 2, 0.309818f, 10, 31, "B5G6R5 blue 0.309818 is 10 (NVIDIA: 9)"},
            {VK_FORMAT_A1R5G5B5_UNORM_PACK16, 1, 0.753251f, 23, 31, "B5G5R5A1 green 0.753251 is 23 (NVIDIA: 24)"},
            {VK_FORMAT_A1R5G5B5_UNORM_PACK16, 3, 0.75f, 1, 1, "B5G5R5A1 alpha 0.75 is 1"},
            {VK_FORMAT_A4R4G4B4_UNORM_PACK16, 0, 0.5f, 8, 15, "B4G4R4A4 0.5 is the tie 7.5, to the even 8"},
            {VK_FORMAT_R32_SFLOAT, 0, 1e-40f, 0, 1, "R32_FLOAT flushes a denormal to +0"},
            {VK_FORMAT_R32_SFLOAT, 0, -1e-40f, -0.0, 1, "R32_FLOAT flushes a negative denormal to -0"},
        };
        for (const Exact& e : exact) {
            leonida::ClearColor input{0.0f, 0.0f, 0.0f, 0.0f};
            input[e.channel] = e.input;
            const VkClearColorValue got = leonida::detail::render_target_clear_color(e.format, input);
            const float want = float(e.code / e.maximum);
            if (std::memcmp(&got.float32[e.channel], &want, sizeof want) != 0) {
                std::fprintf(stderr, "%s: got %.9g, want %.9g\n", e.what, got.float32[e.channel], want);
                return 1;
            }
            ++checks;
        }
        // A D16 depth whose float product is a tie: 0.681063533 * 65535 is 44633.5 in float (WARP writes 44634, the
        // even code), 44633.4999... in double.
        {
            const int mode = std::fegetround();
            for (const int m : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO}) {
                std::fesetround(m);
                const float d16 = leonida::detail::quantize_unorm(0.681063533f, 65535.0);
                std::fesetround(mode);
                if (std::lround(double(d16) * 65535.0) != 44634) {
                    std::fprintf(stderr, "D16 tie under FP mode %d: got %.9g, want code 44634\n", m, double(d16) * 65535.0);
                    return 1;
                }
                ++checks;
            }
        }
        if (leonida::depth_clear_value(1e-40f) != 0.0f || leonida::depth_clear_value(0.25f) != 0.25f) {
            std::fprintf(stderr, "depth_clear_value: a denormal depth must clear to 0\n");
            return 1;
        }
        // The packed 16-bit formats have UAV clear layouts: Uint copies the low bits (all ones is 0xffff), Float
        // rounds ties to even (green 0.5 on 6 bits is 31.5, so 32), and the decoded value is code / max.
        {
            leonida::uav_clear::Element element{};
            VkClearColorValue decoded{};
            const bool uint_ok = leonida::uav_clear::encode(VK_FORMAT_R5G6B5_UNORM_PACK16, false, {0xff, 0xff, 0xff, 0}, element) &&
                                 element.size == 2 && element.bytes[0] == 0xff && element.bytes[1] == 0xff;
            const float rgba[4] = {1.0f, 0.5f, 0.0f, 1.0f};
            std::array<std::uint32_t, 4> bits{};
            std::memcpy(bits.data(), rgba, sizeof rgba);
            const bool float_ok = leonida::uav_clear::encode(VK_FORMAT_R5G6B5_UNORM_PACK16, true, bits, element) &&
                                  (element.bytes[0] | (element.bytes[1] << 8)) == ((31u << 11) | (32u << 5)) &&
                                  leonida::uav_clear::decode(VK_FORMAT_R5G6B5_UNORM_PACK16, element, decoded) &&
                                  decoded.float32[1] == float(32.0 / 63.0);
            const bool others = leonida::uav_clear::encode(VK_FORMAT_A1R5G5B5_UNORM_PACK16, false, {1, 2, 3, 1}, element) &&
                                (element.bytes[0] | (element.bytes[1] << 8)) == ((1u << 15) | (1u << 10) | (2u << 5) | 3u) &&
                                leonida::uav_clear::encode(VK_FORMAT_A4R4G4B4_UNORM_PACK16, false, {1, 2, 3, 4}, element) &&
                                (element.bytes[0] | (element.bytes[1] << 8)) == ((4u << 12) | (1u << 8) | (2u << 4) | 3u);
            if (!uint_ok || !float_ok || !others) {
                std::fprintf(stderr, "uav_clear: the packed 16-bit layouts encode wrong (%d %d %d)\n", uint_ok, float_ok, others);
                return 1;
            }
            checks += 3;
        }
        // ClearUnorderedAccessViewFloat rounds ties to even too (WARP clears R8_UNORM 0.5/255 to 0), whatever the
        // FP rounding mode.
        const int mode = std::fegetround();
        for (const int m : {FE_TONEAREST, FE_UPWARD, FE_DOWNWARD}) {
            std::fesetround(m);
            leonida::uav_clear::Element element{};
            const float half_code = 0.5f / 255.0f, minus_one = -1.0f;
            std::array<std::uint32_t, 4> uav{};
            std::memcpy(&uav[0], &half_code, 4);
            std::memcpy(&uav[1], &minus_one, 4);
            const bool unorm = leonida::uav_clear::encode(VK_FORMAT_R8G8B8A8_UNORM, true, uav, element) &&
                               element.bytes[0] == 0;
            const bool snorm = leonida::uav_clear::encode(VK_FORMAT_R8G8B8A8_SNORM, true, uav, element) &&
                               element.bytes[1] == 0x81;
            std::fesetround(mode);
            if (!unorm || !snorm) {
                std::fprintf(stderr, "uav_clear::encode under FP mode %d: not D3D12's rounding\n", m);
                return 1;
            }
            checks += 2;
        }
    }
    if (failed_formats) {
        std::fprintf(stderr, "clear_color_test: %u integer format(s) failed\n", failed_formats);
        return 1;
    }
    std::printf("clear_color_test: %u full/rectangle integer, normalised and float replay checks passed\n", checks);
    return 0;
}
