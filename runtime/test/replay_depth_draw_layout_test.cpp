// A draw whose depth-stencil view names a texture in DEPTH_READ | PIXEL_SHADER_RESOURCE renders into it in GENERAL.
// D3D12 lets that state test against the texture through a read-only DSV and sample it in the same draw (soft
// particles, decals); with the bindless set 0 its sampled views are written in GENERAL
// (ResourceBinding::sampled_in_general), the layout the state takes (replay_depth_read_layout_test), and Vulkan wants
// the image in the descriptor's layout when a shader reads it (VUID-vkCmdDraw-None-09600). Moving it to
// DEPTH_STENCIL_ATTACHMENT_OPTIMAL for the draw broke that and left the tracker in a write layout. Controls: in
// DEPTH_WRITE, in DEPTH_READ alone, and without set 0 the draw still renders in DEPTH_STENCIL_ATTACHMENT_OPTIMAL.
// The Vulkan functions are mocks that record what they are called with (as in replay_draw_scope_test). No GPU or
// game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using namespace leonida;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

constexpr std::uintptr_t depth_image = 3;
VkImageLayout g_rendered = VK_IMAGE_LAYOUT_MAX_ENUM;  // the depth attachment's layout at vkCmdBeginRendering
std::vector<VkImageMemoryBarrier> g_depth_barriers;    // barriers on the depth image

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t count, const VkImageMemoryBarrier* barriers) {
    for (std::uint32_t i = 0; i < count; ++i)
        if (barriers[i].image == handle<VkImage>(depth_image)) g_depth_barriers.push_back(barriers[i]);
}
VKAPI_ATTR void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo* info) {
    if (info->pDepthAttachment) g_rendered = info->pDepthAttachment->imageLayout;
}
VKAPI_ATTR void VKAPI_CALL end_rendering(VkCommandBuffer) {}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL bind_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout, std::uint32_t, std::uint32_t,
                                     const VkDescriptorSet*, std::uint32_t, const std::uint32_t*) {}
VKAPI_ATTR void VKAPI_CALL push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                                const void*) {}
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

const char* name(VkImageLayout layout) {
    switch (layout) {
    case VK_IMAGE_LAYOUT_GENERAL: return "GENERAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL: return "DEPTH_STENCIL_ATTACHMENT_OPTIMAL";
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL: return "DEPTH_STENCIL_READ_ONLY_OPTIMAL";
    case VK_IMAGE_LAYOUT_MAX_ENUM: return "(no depth attachment)";
    default: return "other";
    }
}

enum : ResourceId { color_id = 1, depth_id, vertex_id };
constexpr std::uintptr_t pso = 100;

}  // namespace

int main() {
    namespace s = d3d12_state;
    ResourceBinding rt{};
    rt.image = handle<VkImage>(1);
    rt.format = VK_FORMAT_R8G8B8A8_UNORM;
    rt.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
    rt.width = rt.height = 64;
    rt.color_attachment = true;
    ResourceBinding vb{};
    vb.buffer = handle<VkBuffer>(4);
    vb.size = 4096;
    vb.va = 0x10000;

    GraphicsPipeline p;
    p.pipeline = handle<VkPipeline>(pso);
    p.layout = handle<VkPipelineLayout>(1);
    p.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    p.bindless_set = handle<VkDescriptorSet>(1);
    p.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
    p.depth_format = VK_FORMAT_D32_SFLOAT;
    p.depth_aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
    p.vertex_slots = 1;

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
    vk.image_view = [](const ResourceBinding& r, const VkImageSubresourceRange& range) {
        return handle<VkImageView>(reinterpret_cast<std::uintptr_t>(r.image) * 16 + range.aspectMask);
    };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
        if (va < vb.va || va - vb.va >= vb.size) return false;
        buffer = vb.buffer;
        offset = va - vb.va;
        size = std::min<std::uint64_t>(size, vb.size - offset);
        return true;
    };

    struct Case {
        const char* what;
        bool sampled_in_general;
        std::uint32_t state;  // the depth texture's state at the draw
        VkImageLayout expected;
    };
    const Case cases[] = {
        {"DEPTH_READ | PIXEL_SHADER_RESOURCE with set 0", true, s::depth_read | s::pixel_shader_resource,
         VK_IMAGE_LAYOUT_GENERAL},
        {"DEPTH_READ | NON_PIXEL_SHADER_RESOURCE with set 0", true, s::depth_read | s::non_pixel_shader_resource,
         VK_IMAGE_LAYOUT_GENERAL},
        {"DEPTH_WRITE with set 0", true, s::depth_write, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
        {"DEPTH_READ with set 0", true, s::depth_read, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
        {"DEPTH_READ | PIXEL_SHADER_RESOURCE without set 0", false, s::depth_read | s::pixel_shader_resource,
         VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL},
    };
    for (const Case& c : cases) {
        ResourceBinding ds{};
        ds.image = handle<VkImage>(depth_image);
        ds.format = VK_FORMAT_D32_SFLOAT;
        ds.aspects = VK_IMAGE_ASPECT_DEPTH_BIT;
        ds.width = ds.height = 64;
        ds.sampled = true;
        ds.sampled_in_general = c.sampled_in_general;
        const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
            return id == color_id ? &rt : id == depth_id ? &ds : id == vertex_id ? &vb : nullptr;
        };
        LayoutTracker tracker;
        tracker.track(color_id, rt, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        tracker.track(depth_id, ds, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        Draw d{};
        d.pipeline = pso;
        DrawViews views;
        views.render_targets = {TargetView{color_id, 0, 0, 1}};
        views.vertex_buffers = {VertexBufferView{0x10000, 1024, 16}};
        d.views = std::make_shared<const DrawViews>(views);
        d.depth = TargetView{depth_id, 0, 0, 1};
        d.index = IndexBufferView{0x10400, 1024, 57};  // R16_UINT (DXGI 57)
        d.indexed = true;
        d.topology = 4;
        d.viewport = {0, 0, 64, 64, 0, 1};
        d.scissor = {0, 0, 64, 64};
        d.blend_factor = {1, 1, 1, 1};
        d.count = 3;
        d.instances = 1;

        CommandList list;
        if (c.state != s::depth_write)
            expect(list.transition({depth_id, all_subresources, s::depth_write, c.state}), c.what);
        expect(list.draw(d) && list.close(), "record the draw");
        PreparedList prepared;
        const ReplayResult prepared_result = prepare(list, resolve, vk, prepared);
        if (prepared_result != ReplayResult::success) {
            std::fprintf(stderr, "FAIL: %s: prepare returned %d\n", c.what, int(prepared_result));
            ++failures;
            continue;
        }
        g_rendered = VK_IMAGE_LAYOUT_MAX_ENUM;
        g_depth_barriers.clear();
        emit(prepared, VK_NULL_HANDLE, vk, &tracker);
        const VkImageLayout state_layout = image_layout(ds, vulkan_access(c.state, ds).layout);
        if (g_rendered != c.expected)
            std::fprintf(stderr, "%s: rendered in %s, expected %s\n", c.what, name(g_rendered), name(c.expected));
        expect(g_rendered == c.expected, c.what);
        expect(tracker.layout(depth_id, 0) == c.expected, c.what);
        // No barrier takes the image from the state's layout to another one, other than the draw's.
        for (const VkImageMemoryBarrier& b : g_depth_barriers)
            expect(b.oldLayout == b.newLayout || b.newLayout == state_layout || b.newLayout == c.expected, c.what);
    }

    if (failures) return 1;
    std::printf("replay_depth_draw_layout_test: OK\n");
    return 0;
}
