// What preparing a draw costs (I6): prepare_draw resolved every attachment (resolve, then image_view, which in
// VulkanContext takes views_mutex_ and looks a std::map up) and every vertex and index buffer (buffer_at) for every
// draw, though runs of draws share one DrawViews block (d3d12_com.cpp keeps it while the bound state does not
// change). Now a draw with the same views, pipeline and depth view as the draw before it in the list takes that
// draw's attachments and vertex buffers, and the same index buffer view takes its index buffer.
//
// The Vulkan functions are mocks; image_view and buffer_at count their calls, and image_view does what the
// context's does (a mutex and a map lookup). Checked, for 20000 draws in runs of 50 that share their views:
//   - image_view and buffer_at are called once per run, not once per draw;
//   - every draw still gets the right attachments, area, vertex and index buffers (a run boundary changes them);
// and printed: draws prepared per millisecond (for the record: before the change, every draw looked everything up).
// No GPU or game data.
#include "../src/vulkan_replay.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

namespace {

using namespace leonida;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {}
VKAPI_ATTR void VKAPI_CALL end_rendering(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t,
                                     const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                                const void*) {}
// vkCmdSetViewportWithCount / vkCmdSetScissorWithCount, what the replay binds since the viewport arrays (M4).
VKAPI_ATTR void VKAPI_CALL viewport(VkCommandBuffer, std::uint32_t, const VkViewport*) {}
VKAPI_ATTR void VKAPI_CALL scissor(VkCommandBuffer, std::uint32_t, const VkRect2D*) {}
VKAPI_ATTR void VKAPI_CALL topology(VkCommandBuffer, VkPrimitiveTopology) {}
VKAPI_ATTR void VKAPI_CALL restart(VkCommandBuffer, VkBool32) {}
VKAPI_ATTR void VKAPI_CALL stencil(VkCommandBuffer, VkStencilFaceFlags, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL blend(VkCommandBuffer, const float[4]) {}
VKAPI_ATTR void VKAPI_CALL vertex(VkCommandBuffer, std::uint32_t, std::uint32_t, const VkBuffer*, const VkDeviceSize*,
                                  const VkDeviceSize*, const VkDeviceSize*) {}
VKAPI_ATTR void VKAPI_CALL index(VkCommandBuffer, VkBuffer, VkDeviceSize, VkIndexType) {}
VKAPI_ATTR void VKAPI_CALL draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                        std::uint32_t) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::uintptr_t pso = 100;
constexpr std::uint32_t runs = 400, per_run = 50, draws = runs * per_run;

}  // namespace

int main() {
    // Two render targets of different sizes and one depth buffer per run parity, two vertex buffers.
    std::vector<ResourceBinding> targets;
    for (std::uint32_t i = 0; i < 4; ++i) {
        ResourceBinding b{};
        b.image = handle<VkImage>(1 + i);
        b.format = i == 3 ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
        b.aspects = i == 3 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        b.width = b.height = 64u << (i & 1);
        b.color_attachment = i != 3;
        targets.push_back(b);
    }
    ResourceBinding vb{};
    vb.buffer = handle<VkBuffer>(10);
    vb.size = 1 << 20;
    vb.va = 0x100000;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        return id >= 1 && id <= targets.size() ? &targets[id - 1] : nullptr;
    };

    GraphicsPipeline p;
    p.pipeline = handle<VkPipeline>(pso);
    p.layout = handle<VkPipelineLayout>(1);
    p.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    p.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    p.depth_format = VK_FORMAT_D32_SFLOAT;
    p.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    p.vertex_slots = 0x3;

    std::mutex views_mutex;
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t,
                        VkFormat>,
             VkImageView>
        views;
    std::uint64_t view_calls = 0, buffer_calls = 0;
    ReplayFunctions vk{};
    vk.pipeline_barrier = barrier;
    vk.begin_rendering = begin_rendering;
    vk.end_rendering = end_rendering;
    vk.bind_pipeline = bind_pipeline;
    vk.bind_descriptor_sets = bind_sets;
    vk.push_constants = push;
    vk.set_viewport = viewport;
    vk.set_scissor = scissor;
    vk.set_primitive_topology = topology;
    vk.set_primitive_restart_enable = restart;
    vk.set_stencil_reference = stencil;
    vk.set_blend_constants = blend;
    vk.bind_vertex_buffers2 = vertex;
    vk.bind_index_buffer = index;
    vk.draw = draw;
    vk.draw_indexed = draw_indexed;
    vk.graphics_pipeline = [&](std::uintptr_t id) -> const GraphicsPipeline* { return id == pso ? &p : nullptr; };
    vk.image_view = [&](const ResourceBinding& r, const VkImageSubresourceRange& range) {
        ++view_calls;
        std::lock_guard lock(views_mutex);  // as VulkanContext::view
        const auto key = std::make_tuple(r.image, range.aspectMask, range.baseMipLevel, range.levelCount,
                                         range.baseArrayLayer, range.layerCount, r.format);
        auto [it, added] = views.try_emplace(key, VK_NULL_HANDLE);
        if (added) it->second = handle<VkImageView>(1000 + views.size());
        return it->second;
    };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
        ++buffer_calls;
        if (va < vb.va || va - vb.va >= vb.size) return false;
        buffer = vb.buffer;
        offset = va - vb.va;
        size = std::min<std::uint64_t>(size, vb.size - offset);
        return true;
    };

    // Runs of draws sharing one DrawViews block (as the COM layer keeps it): run r renders to target r % 3 and
    // reads vertex data at an offset of its own; the index buffer view changes every 10 draws.
    CommandList list;
    std::vector<std::shared_ptr<const DrawViews>> blocks;
    for (std::uint32_t r = 0; r < runs; ++r) {
        DrawViews views_block;
        views_block.render_targets = {TargetView{ResourceId(1 + r % 3), 0, 0, 1}};
        views_block.render_target_count = 1;
        views_block.vertex_buffers = {VertexBufferView{0x100000 + r * 256, 4096, 16},
                                      VertexBufferView{0x180000 + r * 256, 4096, 32}};
        blocks.push_back(std::make_shared<const DrawViews>(views_block));
        for (std::uint32_t i = 0; i < per_run; ++i) {
            Draw d{};
            d.pipeline = pso;
            d.views = blocks.back();
            d.depth = TargetView{4, 0, 0, 1};
            d.index = IndexBufferView{0x1C0000 + ((r * per_run + i) / 10) * 64, 1024, 57};
            d.indexed = true;
            d.topology = 4;
            d.viewport = {0, 0, 64, 64, 0, 1};
            d.scissor = {0, 0, 64, 64};
            d.blend_factor = {1, 1, 1, 1};
            d.count = 3;
            d.instances = 1;
            expect(list.draw(d), "record a draw");
        }
    }
    expect(list.close(), "close the list");

    PreparedList prepared;
    double best_ms = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        view_calls = buffer_calls = 0;
        const auto start = std::chrono::steady_clock::now();
        const ReplayResult result = prepare(list, resolve, vk, prepared);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        expect(result == ReplayResult::success, "prepare the list");
        best_ms = std::min(best_ms, ms);
    }
    std::printf("%u draws in runs of %u: %llu image_view calls, %llu buffer_at calls; %.0f draws/ms (best of 5)\n",
                draws, per_run, static_cast<unsigned long long>(view_calls),
                static_cast<unsigned long long>(buffer_calls), draws / best_ms);
    // Two attachments and two vertex buffers per run, one index buffer per 10 draws.
    expect(view_calls <= 2ull * runs, "image_view once per attachment per run");
    expect(buffer_calls <= 2ull * runs + draws / 10 + runs, "buffer_at once per vertex buffer per run and per index view");

    // Every draw got what its own state names.
    expect(prepared.draw_count == draws, "every draw prepared");
    for (std::uint32_t r = 0; r < runs; ++r)
        for (std::uint32_t i = 0; i < per_run; i += per_run - 1) {
            const std::uint32_t n = r * per_run + i;
            const detail::PreparedDraw& d = prepared.draws[n];
            const ResourceBinding& target = targets[r % 3];
            expect(d.colors.size() == 1 && d.colors[0].id == ResourceId(1 + r % 3), "the run's render target");
            expect(d.colors[0].binding.image == target.image && d.colors[0].view != VK_NULL_HANDLE, "its view");
            expect(d.depth.id == 4 && d.depth.view != VK_NULL_HANDLE, "the depth view");
            expect(d.area.width == target.width && d.area.height == target.height,
                   "the area (the smaller of target and depth)");
            expect(d.vertex_buffers.size() == 2 && d.vertex_offsets[0] == r * 256 &&
                       d.vertex_offsets[1] == 0x80000 + r * 256 && d.vertex_strides[1] == 32,
                   "the run's vertex buffers");
            expect(d.index_buffer == vb.buffer && d.index_offset == 0xC0000 + (n / 10) * 64, "the draw's index buffer");
        }

    if (failures) return 1;
    std::printf("replay_draw_prepare_bench_test: OK\n");
    return 0;
}
