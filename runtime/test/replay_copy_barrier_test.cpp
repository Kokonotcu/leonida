// Back-to-back copies get a transfer -> transfer barrier only where one depends on another (I10). D3D12 orders copies
// on a resource that stays in COPY_DEST or COPY_SOURCE without a barrier from the application; Vulkan does not, so the
// replay recorded a transfer -> transfer memory barrier before every copy (one per side), and a list of uploads into
// distinct textures could not overlap on the GPU. Now one barrier starts each run of copies, and another comes only
// before a copy that reads or writes what a copy of the run wrote, or writes what one read.
//
// vkCmdPipelineBarrier is a function that counts the memory barriers it is given. Textures are tracked in the
// transfer layouts already, so no layout transition hides a barrier. Checked:
//   1. 16 copies between 32 distinct textures: one barrier (32 before);
//   2. X -> Y then Y -> Z (read after write): a barrier between them;
//   3. Y -> Z then W -> Y (write after read): a barrier between them;
//   4. X -> Y then X -> Z (both read X): none between them;
//   5. a copy, a transition, a copy: the run restarts after the transition.
// No Vulkan loader, GPU or game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace leonida;

// The memory barriers recorded, and the copies, in order: 'B' a barrier, 'C' a copy.
std::vector<char> g_events;

void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags,
                        std::uint32_t memory_count, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                        std::uint32_t, const VkImageMemoryBarrier*) {
    if (memory_count && (src & VK_PIPELINE_STAGE_TRANSFER_BIT) && (dst & VK_PIPELINE_STAGE_TRANSFER_BIT))
        g_events.push_back('B');
}
void VKAPI_CALL copy_image(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, std::uint32_t,
                           const VkImageCopy*) {
    g_events.push_back('C');
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
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

}  // namespace

int main() {
    namespace s = d3d12_state;
    const VkCommandBuffer cb = reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x10));
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.copy_image = copy_image;
    vk.clear_color_image = clear_color;
    vk.copy_buffer = copy_buffer;
    vk.copy_buffer_to_image = copy_buffer_to_image;
    vk.copy_image_to_buffer = copy_image_to_buffer;

    constexpr ResourceId textures = 40;
    std::vector<ResourceBinding> bindings(textures + 1);
    for (ResourceId id = 1; id <= textures; ++id) {
        ResourceBinding& b = bindings[id];
        b.image = reinterpret_cast<VkImage>(std::uintptr_t(0x1000 * id));
        b.format = VK_FORMAT_R8G8B8A8_UNORM;
        b.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
        b.width = b.height = 16;
    }
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        return id >= 1 && id <= textures ? &bindings[id] : nullptr;
    };
    // Every texture starts in COPY_DEST, already used: no layout transition, no zero fill. A copy source moves to
    // TRANSFER_SRC with an image barrier, which these counts leave out (only memory barriers are counted).
    const auto run = [&](CommandList& list, const char* what) {
        LayoutTracker tracker;
        for (ResourceId id = 1; id <= textures; ++id)
            tracker.track(id, bindings[id], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        expect(list.close(), what);
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, what);
        g_events.clear();
        emit(prepared, cb, vk, &tracker);
        std::printf("%s: %s\n", what, std::string(g_events.begin(), g_events.end()).c_str());
        return std::string(g_events.begin(), g_events.end());
    };
    const auto count = [](const std::string& events, char c) { return std::size_t(std::count(events.begin(), events.end(), c)); };

    {
        CommandList list;
        for (ResourceId i = 0; i < 16; ++i) expect(list.copy_resource({ResourceId(1 + i), ResourceId(17 + i)}), "record");
        const std::string e = run(list, "1 16 copies between distinct textures");
        expect(count(e, 'C') == 16 && count(e, 'B') == 1 && e.front() == 'B', "1 one barrier, first");
    }
    {
        CommandList list;
        expect(list.copy_resource({2, 1}) && list.copy_resource({3, 2}), "record");  // X=1 -> Y=2, Y -> Z=3
        const std::string e = run(list, "2 read after write");
        expect(e == "BCBC", "2 a barrier between them");
    }
    {
        CommandList list;
        expect(list.copy_resource({3, 2}) && list.copy_resource({2, 4}), "record");  // Y=2 -> Z=3, W=4 -> Y
        const std::string e = run(list, "3 write after read");
        expect(e == "BCBC", "3 a barrier between them");
    }
    {
        CommandList list;
        expect(list.copy_resource({2, 1}) && list.copy_resource({3, 1}), "record");  // X=1 -> Y=2, X -> Z=3
        const std::string e = run(list, "4 two reads of one source");
        expect(e == "BCC", "4 none between them");
    }
    {
        CommandList list;
        expect(list.copy_resource({2, 1}) &&
                   list.transition({5, all_subresources, s::copy_dest, s::pixel_shader_resource}) &&
                   list.copy_resource({4, 3}),
               "record");
        const std::string e = run(list, "5 a transition between copies");
        expect(count(e, 'B') == 2 && count(e, 'C') == 2, "5 the run restarts");
    }

    if (failures) return 1;
    std::printf("replay_copy_barrier_test: OK\n");
    return 0;
}
