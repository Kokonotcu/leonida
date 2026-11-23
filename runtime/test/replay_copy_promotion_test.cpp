// A copy over a texture in common runs through an implicit promotion, and D3D12 decays the texture back to common
// with no barrier: when a copy queue's ExecuteCommandLists ends, and after a read-only promotion on any queue
// ("Implicit State Transitions"). A shader may then sample it, promoted again, through its set-0 descriptor, which
// is written in GENERAL (ResourceBinding::sampled_in_general). Vulkan wants the image in that layout when the shader
// reads it (VUID-vkCmdDraw-None-09600), so such a copy must leave it in GENERAL, not in a transfer layout:
//   1. a never-used texture (tracked UNDEFINED: created in common) copied into from one in common;
//   2. control: a texture moved to COPY_DEST and a source in COPY_SOURCE by explicit barriers keep the transfer layouts;
//   3. a range whose mips disagree (one in common, one in COPY_DEST) is copied in GENERAL, all of it;
//   4. decay_to_common, recorded at the end of a copy queue's lists, moves what an explicit COPY_DEST barrier left in
//      TRANSFER_DST back to GENERAL, and leaves untracked and swapchain images alone.
// vkCmdPipelineBarrier, vkCmdCopyImage and vkCmdClearColorImage are functions that record what they are given. No
// Vulkan loader, GPU or game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace leonida;

std::vector<VkImageMemoryBarrier> g_barriers;
struct Copy {
    VkImage src;
    VkImageLayout src_layout;
    VkImage dst;
    VkImageLayout dst_layout;
};
std::vector<Copy> g_copies;

void VKAPI_CALL record_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                               std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                               std::uint32_t count, const VkImageMemoryBarrier* barriers) {
    for (std::uint32_t i = 0; i < count; ++i) g_barriers.push_back(barriers[i]);
}
void VKAPI_CALL record_copy(VkCommandBuffer, VkImage src, VkImageLayout src_layout, VkImage dst,
                            VkImageLayout dst_layout, std::uint32_t, const VkImageCopy*) {
    g_copies.push_back({src, src_layout, dst, dst_layout});
}
void VKAPI_CALL clear_color(VkCommandBuffer, VkImage, VkImageLayout, const VkClearColorValue*, std::uint32_t,
                            const VkImageSubresourceRange*) {}
void VKAPI_CALL copy_buffer(VkCommandBuffer, VkBuffer, VkBuffer, std::uint32_t, const VkBufferCopy*) {}
void VKAPI_CALL copy_buffer_to_image(VkCommandBuffer, VkBuffer, VkImage, VkImageLayout, std::uint32_t,
                                     const VkBufferImageCopy*) {}
void VKAPI_CALL copy_image_to_buffer(VkCommandBuffer, VkImage, VkImageLayout, VkBuffer, std::uint32_t,
                                     const VkBufferImageCopy*) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

const char* name(VkImageLayout layout) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED: return "UNDEFINED";
    case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL: return "TRANSFER_SRC_OPTIMAL";
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL: return "TRANSFER_DST_OPTIMAL";
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR: return "PRESENT_SRC_KHR";
    default: return "other";
    }
}

void expect_layout(const LayoutTracker& tracker, ResourceId id, std::uint32_t subresource, VkImageLayout want,
                   const char* what) {
    const VkImageLayout got = tracker.layout(id, subresource);
    if (got != want)
        std::fprintf(stderr, "%s: subresource %u tracked as %s, expected %s\n", what, subresource, name(got),
                     name(want));
    expect(got == want, what);
}

ResourceBinding texture(std::uintptr_t handle, std::uint32_t mips) {
    ResourceBinding b{};
    b.image = reinterpret_cast<VkImage>(handle);
    b.format = VK_FORMAT_R8G8B8A8_UNORM;
    b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    b.width = b.height = 16;
    b.mip_levels = mips;
    b.sampled = true;
    b.sampled_in_general = true;
    return b;
}

}  // namespace

int main() {
    namespace s = d3d12_state;
    const VkCommandBuffer cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x10));
    ReplayFunctions vk{};
    vk.pipeline_barrier = record_barrier;
    vk.copy_image = record_copy;
    vk.clear_color_image = clear_color;
    vk.copy_buffer = copy_buffer;
    vk.copy_buffer_to_image = copy_buffer_to_image;
    vk.copy_image_to_buffer = copy_image_to_buffer;

    constexpr ResourceId src_id = 1, dst_id = 2, swap_id = 3, loose_id = 4;
    const auto run = [&](const ResourceBinding& src, const ResourceBinding& dst, LayoutTracker& tracker,
                         CommandList& list, const char* what) {
        const auto resolve = [&](ResourceId r) -> const ResourceBinding* {
            return r == src_id ? &src : r == dst_id ? &dst : nullptr;
        };
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, what);
        g_barriers.clear();
        g_copies.clear();
        emit(prepared, cb, vk, &tracker);
        expect(g_copies.size() == 1, what);
    };

    // 1. Never used: the destination is copied into in GENERAL, and stays there.
    {
        const char* what = "1 a never-used texture copied into from one in common";
        const ResourceBinding src = texture(0x1000, 1), dst = texture(0x2000, 1);
        LayoutTracker tracker;
        tracker.track(src_id, src, VK_IMAGE_LAYOUT_GENERAL);
        tracker.track(dst_id, dst, VK_IMAGE_LAYOUT_UNDEFINED);
        CommandList list;
        expect(list.copy_resource({dst_id, src_id}) && list.close(), what);
        run(src, dst, tracker, list, what);
        if (!g_copies.empty()) {
            const Copy& c = g_copies[0];
            if (c.src_layout != VK_IMAGE_LAYOUT_GENERAL || c.dst_layout != VK_IMAGE_LAYOUT_GENERAL)
                std::fprintf(stderr, "%s: copied %s -> %s, expected GENERAL -> GENERAL\n", what, name(c.src_layout),
                             name(c.dst_layout));
            expect(c.src_layout == VK_IMAGE_LAYOUT_GENERAL && c.dst_layout == VK_IMAGE_LAYOUT_GENERAL, what);
        }
        // The zero fill of the first use clears in TRANSFER_DST; the last barrier on the image ends in GENERAL.
        VkImageLayout last = VK_IMAGE_LAYOUT_UNDEFINED;
        for (const VkImageMemoryBarrier& b : g_barriers)
            if (b.image == dst.image) last = b.newLayout;
        expect(last == VK_IMAGE_LAYOUT_GENERAL, "1 the destination's last barrier ends in GENERAL");
        expect_layout(tracker, src_id, 0, VK_IMAGE_LAYOUT_GENERAL, what);
        expect_layout(tracker, dst_id, 0, VK_IMAGE_LAYOUT_GENERAL, what);
    }

    // 2. Explicit barriers: the copy runs in the transfer layouts the application asked for.
    {
        const char* what = "2 explicit COPY_DEST and COPY_SOURCE keep their layouts";
        const ResourceBinding src = texture(0x1000, 1), dst = texture(0x2000, 1);
        LayoutTracker tracker;
        tracker.track(src_id, src, VK_IMAGE_LAYOUT_GENERAL);
        tracker.track(dst_id, dst, VK_IMAGE_LAYOUT_GENERAL);
        CommandList list;
        expect(list.transition({src_id, all_subresources, s::common, s::copy_source}) &&
                   list.transition({dst_id, all_subresources, s::common, s::copy_dest}) &&
                   list.copy_resource({dst_id, src_id}) && list.close(),
               what);
        run(src, dst, tracker, list, what);
        if (!g_copies.empty())
            expect(g_copies[0].src_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                       g_copies[0].dst_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   what);
        expect_layout(tracker, src_id, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, what);
        expect_layout(tracker, dst_id, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, what);
    }

    // 3. Mips that disagree: mip 0 in common, mip 1 left in COPY_DEST. The copy promotes mip 0, so all of it is
    //    copied in GENERAL; mip 1 is moved there.
    {
        const char* what = "3 a range in common and COPY_DEST is copied in GENERAL";
        const ResourceBinding src = texture(0x1000, 2), dst = texture(0x2000, 2);
        LayoutTracker tracker;
        tracker.track(src_id, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        tracker.track(dst_id, dst, VK_IMAGE_LAYOUT_GENERAL);
        tracker.set(dst_id, {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1}, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        CommandList list;
        expect(list.copy_resource({dst_id, src_id}) && list.close(), what);
        run(src, dst, tracker, list, what);
        if (!g_copies.empty())
            expect(g_copies[0].src_layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL &&
                       g_copies[0].dst_layout == VK_IMAGE_LAYOUT_GENERAL,
                   what);
        bool moved_mip1 = false;
        for (const VkImageMemoryBarrier& b : g_barriers)
            moved_mip1 |= b.image == dst.image && b.subresourceRange.baseMipLevel == 1 &&
                          b.oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && b.newLayout == VK_IMAGE_LAYOUT_GENERAL;
        expect(moved_mip1, "3 mip 1 moved from TRANSFER_DST to GENERAL");
        expect_layout(tracker, dst_id, 0, VK_IMAGE_LAYOUT_GENERAL, what);
        expect_layout(tracker, dst_id, 1, VK_IMAGE_LAYOUT_GENERAL, what);
        expect_layout(tracker, src_id, 0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, what);
    }

    // 4. The decay at the end of a copy queue's lists.
    {
        const char* what = "4 decay_to_common after an explicit COPY_DEST barrier";
        const ResourceBinding src = texture(0x1000, 1), dst = texture(0x2000, 1);
        ResourceBinding swap = texture(0x3000, 1);
        swap.presentable = true;
        const ResourceBinding loose = texture(0x4000, 1);  // not tracked
        LayoutTracker tracker;
        tracker.track(src_id, src, VK_IMAGE_LAYOUT_GENERAL);
        tracker.track(dst_id, dst, VK_IMAGE_LAYOUT_GENERAL);
        tracker.track(swap_id, swap, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
        CommandList list;
        expect(list.transition({dst_id, all_subresources, s::common, s::copy_dest}) &&
                   list.copy_resource({dst_id, src_id}) &&
                   list.transition({swap_id, all_subresources, s::common, s::copy_dest}) &&
                   list.transition({loose_id, all_subresources, s::common, s::copy_dest}) && list.close(),
               what);
        const auto resolve = [&](ResourceId r) -> const ResourceBinding* {
            return r == src_id ? &src : r == dst_id ? &dst : r == swap_id ? &swap : r == loose_id ? &loose : nullptr;
        };
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, what);
        emit(prepared, cb, vk, &tracker);
        expect_layout(tracker, dst_id, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, "4 before the decay");
        const VkImageLayout swap_before = tracker.layout(swap_id, 0);
        expect(swap_before == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, "4 the swapchain image is in TRANSFER_DST");
        g_barriers.clear();
        const CommandList* lists[] = {&list};
        decay_to_common(vk, cb, lists, resolve, &tracker);
        expect(g_barriers.size() == 1, "4 one barrier: the destination");
        if (!g_barriers.empty())
            expect(g_barriers[0].image == dst.image && g_barriers[0].oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
                       g_barriers[0].newLayout == VK_IMAGE_LAYOUT_GENERAL,
                   what);
        expect_layout(tracker, dst_id, 0, VK_IMAGE_LAYOUT_GENERAL, what);
        expect_layout(tracker, src_id, 0, VK_IMAGE_LAYOUT_GENERAL, what);
        expect_layout(tracker, swap_id, 0, swap_before, "4 a swapchain image is left alone");
        // Nothing left to decay: a second pass records nothing.
        g_barriers.clear();
        decay_to_common(vk, cb, lists, resolve, &tracker);
        expect(g_barriers.empty(), "4 a second decay records nothing");
    }

    if (failures) return 1;
    std::printf("replay_copy_promotion_test: OK\n");
    return 0;
}
