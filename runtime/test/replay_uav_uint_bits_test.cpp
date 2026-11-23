// ClearUnorderedAccessViewUint on a float, UNORM or SNORM view of a typeless texture writes its bits exactly: the
// replay clears through the family's UINT format (uav_clear::uint_twin), so no float carries the bits to the driver
// (a NaN payload, SNORM 0x80). A texture that isn't typeless, and Float clears, keep the view's own format. Checks
// the recorded Vulkan calls. No Vulkan loader, GPU, shaders or game data.
#include "../src/vulkan_replay.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
VkFormat view_format = VK_FORMAT_UNDEFINED;
VkClearColorValue cleared{};
unsigned attachment_clears = 0, image_clears = 0;

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
    std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue* c,
    std::uint32_t, const VkImageSubresourceRange*) {
    cleared = *c;
    ++image_clears;
}
VKAPI_ATTR void VKAPI_CALL begin(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL end(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL attachments(VkCommandBuffer, std::uint32_t n, const VkClearAttachment* a, std::uint32_t,
    const VkClearRect*) {
    if (n == 1) cleared = a[0].clearValue.color;
    ++attachment_clears;
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool run(VkFormat backing, bool typeless, VkFormat view, bool is_float, const std::array<std::uint32_t, 4>& values) {
    leonida::ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(std::uintptr_t{1});
    b.format = backing;
    b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    b.width = b.height = 4;
    b.depth = b.mip_levels = b.array_layers = b.samples = 1;
    b.typeless = typeless;
    b.color_attachment = true;
    leonida::ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear_color;
    vk.begin_rendering = begin;
    vk.end_rendering = end;
    vk.clear_attachments = attachments;
    vk.image_view = [](const leonida::ResourceBinding& r, const VkImageSubresourceRange&) {
        view_format = r.format;
        return reinterpret_cast<VkImageView>(std::uintptr_t{1});
    };
    leonida::CommandList list;
    const leonida::UavView uav{1, false, std::uint32_t(view), 0, 0, 0, 0, 0, 1};
    if (!list.clear_unordered_access(uav, is_float, values) || !list.close()) return false;
    view_format = VK_FORMAT_UNDEFINED;
    cleared = {};
    attachment_clears = image_clears = 0;
    return leonida::replay(list, VK_NULL_HANDLE, [&](auto) { return &b; }, vk) == leonida::ReplayResult::success;
}

bool same(const std::array<std::uint32_t, 4>& want) {
    for (int i = 0; i < 4; ++i)
        if (cleared.uint32[i] != want[i]) return false;
    return true;
}
}  // namespace

int main() {
    // SNORM 0x80 (-1.0 the other way): through a float it came back 0x81.
    const std::array<std::uint32_t, 4> snorm{0x80, 0x7f, 0x81, 0xff};
    expect(run(VK_FORMAT_R8G8B8A8_UNORM, true, VK_FORMAT_R8G8B8A8_SNORM, false, snorm), "RGBA8 SNORM replays");
    expect(view_format == VK_FORMAT_R8G8B8A8_UINT && attachment_clears == 1 && same(snorm),
           "a typeless RGBA8 texture's SNORM Uint clear goes through R8G8B8A8_UINT with the raw bits");

    // A half NaN with a payload, 0xfffd, and a float NaN pattern.
    const std::array<std::uint32_t, 4> half{0xfffd, 0x7c01, 0x3c00, 0x8000};
    expect(run(VK_FORMAT_R16G16B16A16_SFLOAT, true, VK_FORMAT_R16G16B16A16_SFLOAT, false, half), "RGBA16F replays");
    expect(view_format == VK_FORMAT_R16G16B16A16_UINT && same(half),
           "a typeless RGBA16 texture's FLOAT-view Uint clear keeps the NaN payloads");
    const std::array<std::uint32_t, 4> single{0x7fc12345, 0, 0, 0};
    expect(run(VK_FORMAT_R32_SFLOAT, true, VK_FORMAT_R32_SFLOAT, false, single), "R32F replays");
    expect(view_format == VK_FORMAT_R32_UINT && cleared.uint32[0] == 0x7fc12345u, "R32_FLOAT: the NaN bits as they are");

    // Not typeless: the view's own format, as before (no UINT view of a non-mutable image).
    expect(run(VK_FORMAT_R8G8B8A8_SNORM, false, VK_FORMAT_R8G8B8A8_SNORM, false, snorm), "non-typeless replays");
    expect(view_format != VK_FORMAT_R8G8B8A8_UINT, "a texture that isn't typeless keeps its format");

    // A Float clear keeps the view's format: the value is converted, not copied.
    float one = 1.0f;
    std::uint32_t bits = 0;
    std::memcpy(&bits, &one, 4);
    expect(run(VK_FORMAT_R8G8B8A8_UNORM, true, VK_FORMAT_R8G8B8A8_SNORM, true, {bits, bits, bits, bits}),
           "Float clear replays");
    expect(view_format == VK_FORMAT_R8G8B8A8_SNORM, "a Float clear goes through the view's own format");

    if (failures) return 1;
    std::puts("replay_uav_uint_bits_test: OK");
    return 0;
}
