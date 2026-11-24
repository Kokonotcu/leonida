// The zero fill of a texture's first use is skipped when that use overwrites the whole subresource (I5). A D3D12
// committed texture reads as zeros until written, so the first command that touches a never-used piece (tracked
// UNDEFINED) zero-fills it: vkCmdClearColorImage, or for block-compressed formats a copy from the zero page. When
// that command is a copy over the whole subresource, or a whole-view clear, the fill is overwritten at once: every
// streamed texture paid an extra pass over its memory before its upload.
//
// vkCmdClearColorImage, vkCmdClearDepthStencilImage and the copies are functions that count their calls. Checked:
//   1. CopyResource into a never-used texture: no fill (one clear before);
//   2. CopyTextureRegion from a buffer covering a whole mip of a BC1 texture: no zero-page copy, only the upload;
//   3. a copy of half of a mip: still filled (the other half must read zero);
//   4. a whole-view ClearRenderTargetView: one clear, not two; a rectangle clear: still filled;
//   5. ClearDepthStencilView of depth only on a depth-stencil texture tracked as one: still filled (stencil must
//      read zero); of both: one clear.
// No Vulkan loader, GPU or game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace leonida;

unsigned g_clears = 0, g_depth_clears = 0, g_copies = 0, g_buffer_to_image = 0;

void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags, std::uint32_t,
                        const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*, std::uint32_t,
                        const VkImageMemoryBarrier*) {}
void VKAPI_CALL clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, std::uint32_t,
                            const VkImageSubresourceRange*) {
    ++g_clears;
}
void VKAPI_CALL clear_depth_image(VkCommandBuffer, VkImage, VkImageLayout, const VkClearDepthStencilValue*, std::uint32_t,
                            const VkImageSubresourceRange*) {
    ++g_depth_clears;
}
void VKAPI_CALL copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, std::uint32_t,
                           const VkImageCopy*) {
    ++g_copies;
}
void VKAPI_CALL copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
void VKAPI_CALL copy_buffer_to_image(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                     const VkBufferImageCopy*) {
    ++g_buffer_to_image;
}
void VKAPI_CALL copy_image_to_buffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                     const VkBufferImageCopy*) {}
void VKAPI_CALL fill_buffer(VkCommandBuffer, VkBuffer, VkDeviceSize, VkDeviceSize, std::uint32_t) {}
void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {}
void VKAPI_CALL end_rendering(VkCommandBuffer) {}
void VKAPI_CALL clear_attachments(VkCommandBuffer, std::uint32_t, const VkClearAttachment*, std::uint32_t,
                                  const VkClearRect*) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

ResourceBinding texture(std::uintptr_t handle, VkFormat format, VkImageAspectFlags aspects, std::uint32_t size) {
    ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(handle);
    b.format = format;
    b.aspects = aspects;
    b.width = b.height = size;
    b.color_attachment = aspects == VK_IMAGE_ASPECT_COLOR_BIT;
    b.sampled_in_general = true;
    return b;
}

}  // namespace

int main() {
    const VkCommandBuffer cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x10));
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.clear_color_image = clear_color;
    vk.clear_depth_stencil_image = clear_depth_image;
    vk.copy_image = copy_image;
    vk.copy_buffer = copy_buffer;
    vk.copy_buffer_to_image = copy_buffer_to_image;
    vk.copy_image_to_buffer = copy_image_to_buffer;
    vk.fill_buffer = fill_buffer;
    vk.begin_rendering = begin_rendering;
    vk.end_rendering = end_rendering;
    vk.clear_attachments = clear_attachments;
    vk.image_view = [](const ResourceBinding&, const VkImageSubresourceRange&) {
        return reinterpret_cast<VkImageView>(std::uintptr_t(0x77));
    };
    // A zero page for block-compressed fills (as the device's arena reserve).
    vk.zero_buffer = reinterpret_cast<VkBuffer>(std::uintptr_t(0x99));
    vk.zero_offset = 0;
    vk.zero_size = 64 * 1024;

    ResourceBinding upload{};
    upload.buffer = reinterpret_cast<VkBuffer>(std::uintptr_t(0x50));
    upload.size = 1 << 20;
    upload.va = 0x10000;
    upload.host = nullptr;

    struct Case {
        const char* what;
        unsigned clears, depth_clears, buffer_to_image;  // expected
    };
    const auto run = [&](CommandList& list, const ResourceBinding& a, const ResourceBinding& b, const Case& c) {
        const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
            return id == 1 ? &a : id == 2 ? &b : id == 3 ? &upload : nullptr;
        };
        LayoutTracker tracker;
        tracker.track(1, a, VK_IMAGE_LAYOUT_UNDEFINED);
        tracker.track(2, b, VK_IMAGE_LAYOUT_GENERAL);
        g_clears = g_depth_clears = g_copies = g_buffer_to_image = 0;
        expect(list.close(), c.what);
        PreparedList prepared;
        const ReplayResult r = prepare(list, resolve, vk, prepared);
        if (r != ReplayResult::success) {
            std::fprintf(stderr, "FAIL: %s: prepare returned %d\n", c.what, int(r));
            ++failures;
            return;
        }
        emit(prepared, cb, vk, &tracker);
        std::printf("%s: %u colour clears, %u depth clears, %u buffer-to-image copies\n", c.what, g_clears,
                    g_depth_clears, g_buffer_to_image);
        expect(g_clears == c.clears && g_depth_clears == c.depth_clears && g_buffer_to_image == c.buffer_to_image,
               c.what);
    };

    const ResourceBinding rgba = texture(0x1000, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 64);
    const ResourceBinding rgba_src = texture(0x2000, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 64);
    {
        CommandList list;
        expect(list.copy_resource({1, 2}), "record CopyResource");
        run(list, rgba, rgba_src, {"1 CopyResource into a never-used texture", 0, 0, 0});
    }
    {
        const ResourceBinding bc1 = texture(0x3000, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, VK_IMAGE_ASPECT_COLOR_BIT, 64);
        CommandList list;
        CopyTexture copy{};
        copy.dst = {1, false, 0, 0, 0, 0, 0, 0};
        copy.src = {3, true, 0, 0, 64, 64, 1, 256};  // 16 blocks of 8 bytes a row, pitch-aligned
        expect(list.copy_texture(copy), "record a whole-mip BC1 upload");
        run(list, bc1, rgba_src, {"2 a whole-mip BC1 upload", 0, 0, 1});
    }
    {
        CommandList list;
        CopyTexture copy{};
        copy.dst = {1, false, 0, 0, 0, 0, 0, 0};
        copy.src = {3, true, 0, 0, 32, 64, 1, 256};  // the left half: 32 texels of 4 bytes a row
        expect(list.copy_texture(copy), "record a half-mip upload");
        run(list, rgba, rgba_src, {"3 an upload of half a mip", 1, 0, 1});
    }
    {
        CommandList list;
        expect(list.clear_render_target({1, 0, 0, 1}, {0.5f, 0.5f, 0.5f, 1.0f}), "record a whole-view clear");
        run(list, rgba, rgba_src, {"4 a whole-view ClearRenderTargetView", 1, 0, 0});
        CommandList rects;
        const Rect half{0, 0, 32, 64};
        expect(rects.clear_render_target({1, 0, 0, 1}, {0.5f, 0.5f, 0.5f, 1.0f}, {&half, 1}), "record a rect clear");
        run(rects, rgba, rgba_src, {"4 a rectangle ClearRenderTargetView", 1, 0, 0});
    }
    {
        const ResourceBinding ds = texture(0x4000, VK_FORMAT_D32_SFLOAT_S8_UINT,
                                           VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 64);
        CommandList depth_only;
        expect(depth_only.clear_depth_stencil({1, 0, 0, 1}, leonida::clear_depth, 1.0f, 0), "record a depth-only clear");
        run(depth_only, ds, rgba_src, {"5 ClearDepthStencilView of depth only", 0, 2, 0});
        CommandList both;
        expect(both.clear_depth_stencil({1, 0, 0, 1}, leonida::clear_depth | leonida::clear_stencil, 1.0f, 0), "record a full clear");
        run(both, ds, rgba_src, {"5 ClearDepthStencilView of depth and stencil", 0, 1, 0});
    }

    if (failures) return 1;
    std::printf("replay_first_use_fill_test: OK\n");
    return 0;
}
