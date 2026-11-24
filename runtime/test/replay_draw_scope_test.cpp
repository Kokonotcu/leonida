// Draws share a rendering scope (D3D-324, the review's I1; runtime/src/vulkan_replay.h emit): consecutive draws with the
// same attachments are recorded in one vkCmdBeginRendering scope without an attachment barrier between them, and any
// other command (a barrier, a marker, predication) ends the scope first. ReplayFunctions::merge_draw_scopes = false gives
// every draw its own scope, as before. The Vulkan functions are mocks that record what they are called with (the mock
// setup follows #3346's test). No GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using namespace leonida;

struct Trace {
    bool rendering = false;
    unsigned begins = 0, ends = 0, barriers = 0, barriers_inside = 0, nested = 0, stray_ends = 0;
    unsigned draws = 0, draws_outside = 0, cond_ends_inside = 0;
    std::vector<std::string> events;
};

Trace* g = nullptr;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) {
    ++g->barriers;
    if (g->rendering) ++g->barriers_inside;
    g->events.push_back("barrier");
}
VKAPI_ATTR void VKAPI_CALL begin_rendering(VkCommandBuffer, const VkRenderingInfo*) {
    if (g->rendering) ++g->nested;
    g->rendering = true;
    ++g->begins;
    g->events.push_back("begin");
}
VKAPI_ATTR void VKAPI_CALL end_rendering(VkCommandBuffer) {
    if (!g->rendering) ++g->stray_ends;
    g->rendering = false;
    ++g->ends;
    g->events.push_back("end");
}
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
void count_draw() {
    ++g->draws;
    if (!g->rendering) ++g->draws_outside;
    g->events.push_back("draw");
}
VKAPI_ATTR void VKAPI_CALL draw(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t) {
    count_draw();
}
VKAPI_ATTR void VKAPI_CALL draw_indexed(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t, std::int32_t,
                                        std::uint32_t) {
    count_draw();
}
VKAPI_ATTR void VKAPI_CALL begin_cond(VkCommandBuffer, const VkConditionalRenderingBeginInfoEXT*) {
    g->events.push_back("begin_cond");
}
VKAPI_ATTR void VKAPI_CALL end_cond(VkCommandBuffer) {
    if (g->rendering) ++g->cond_ends_inside;
    g->events.push_back("end_cond");
}

int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

// Every scope ends, none nests, and no barrier or conditional-rendering end is recorded inside one.
void expect_well_formed(const Trace& t, const char* what) {
    if (t.begins != t.ends || t.nested || t.stray_ends || t.barriers_inside || t.draws_outside || t.rendering ||
        t.cond_ends_inside) {
        std::fprintf(stderr, "FAIL: %s: begins %u ends %u nested %u stray %u barriers inside %u draws outside %u\n", what,
                     t.begins, t.ends, t.nested, t.stray_ends, t.barriers_inside, t.draws_outside);
        ++failures;
    }
}

enum : ResourceId { color_a = 1, color_b, depth_id, vertex_id, index_id, predicate_id };
enum : std::uintptr_t { pso_a = 100, pso_stencil };

}  // namespace

int main() {
    const auto image = [](std::uintptr_t n, VkFormat format, VkImageAspectFlags aspects) {
        ResourceBinding b{};
        b.image = handle<VkImage>(n);
        b.format = format;
        b.aspects = aspects;
        b.width = b.height = 64;
        b.color_attachment = aspects == VK_IMAGE_ASPECT_COLOR_BIT;
        return b;
    };
    const ResourceBinding rt_a = image(1, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding rt_b = image(2, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
    const ResourceBinding ds =
        image(3, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    ResourceBinding vb{};
    vb.buffer = handle<VkBuffer>(4);
    vb.size = 4096;
    vb.va = 0x10000;
    ResourceBinding ib = vb;
    ib.buffer = handle<VkBuffer>(5);
    ib.va = 0x20000;
    ResourceBinding pred{};
    pred.buffer = handle<VkBuffer>(6);
    pred.size = 16;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        switch (id) {
        case color_a: return &rt_a;
        case color_b: return &rt_b;
        case depth_id: return &ds;
        case vertex_id: return &vb;
        case index_id: return &ib;
        case predicate_id: return &pred;
        default: return nullptr;
        }
    };
    const auto pipeline = [](std::uintptr_t n, VkImageAspectFlags depth_aspects) {
        GraphicsPipeline p;
        p.pipeline = handle<VkPipeline>(n);
        p.layout = handle<VkPipelineLayout>(1);
        p.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        p.uses_set0 = true;
        p.color_formats = {VK_FORMAT_R8G8B8A8_UNORM};
        p.depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
        p.depth_aspects = depth_aspects;
        p.vertex_slots = 1;
        return p;
    };
    const GraphicsPipeline a = pipeline(pso_a, VK_IMAGE_ASPECT_DEPTH_BIT);
    const GraphicsPipeline with_stencil = pipeline(pso_stencil, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    ReplayFunctions vk{};
    vk.bindless_set = handle<VkDescriptorSet>(1);
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
    vk.graphics_pipeline = [&](std::uintptr_t id) -> const GraphicsPipeline* {
        if (id == pso_a) return &a;
        if (id == pso_stencil) return &with_stencil;
        return nullptr;
    };
    vk.image_view = [](const ResourceBinding& r, const VkImageSubresourceRange& range) {
        return handle<VkImageView>(reinterpret_cast<std::uintptr_t>(r.image) * 16 + range.aspectMask);
    };
    vk.buffer_at = [&](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
        for (const ResourceBinding* r : {&vb, &ib}) {
            if (va < r->va || va - r->va >= r->size) continue;
            buffer = r->buffer;
            offset = va - r->va;
            size = std::min<std::uint64_t>(size, r->size - offset);
            return true;
        }
        return false;
    };
    vk.begin_conditional_rendering = begin_cond;
    vk.end_conditional_rendering = end_cond;
    vk.prepare_predicate = [](VkBuffer source, VkDeviceSize offset, PredicateCapture& out) {
        out = {source, offset, handle<VkBuffer>(7), 0, handle<VkDescriptorSet>(8)};
        return true;
    };
    vk.capture_predicate = [](VkCommandBuffer, const PredicateCapture&) {};

    const auto make_draw = [](std::uintptr_t pso, ResourceId target) {
        Draw d{};
        d.pipeline = pso;
        DrawViews views;
        views.render_targets = {TargetView{target, 0, 0, 1}};
        views.vertex_buffers = {VertexBufferView{0x10000, 1024, 16}};
        d.views = std::make_shared<const DrawViews>(views);
        d.depth = TargetView{depth_id, 0, 0, 1};
        d.index = IndexBufferView{0x20000, 1024, 57};
        d.topology = 4;
        d.viewport = {0, 0, 64, 64, 0, 1};
        d.scissor = {0, 0, 64, 64};
        d.blend_factor = {1, 1, 1, 1};
        d.indexed = true;
        d.count = 3;
        d.instances = 1;
        return d;
    };
    const auto run = [&](CommandList& list, Trace& trace) {
        g = &trace;
        expect(list.close(), "close the list");
        PreparedList prepared;
        expect(prepare(list, resolve, vk, prepared) == ReplayResult::success, "prepare the list");
        emit(prepared, VK_NULL_HANDLE, vk);
    };

    // 1. Eight draws to the same targets: one scope, each attachment's barrier once.
    {
        Trace t;
        CommandList list;
        for (int i = 0; i < 8; ++i) expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        run(list, t);
        expect_well_formed(t, "same targets");
        expect(t.begins == 1 && t.draws == 8, "same targets: eight draws in one scope");
        expect(t.barriers == 2, "same targets: one barrier per attachment, when the scope opens");
    }
    // 2. Another render target, or another depth/stencil attachment set, opens a new scope.
    {
        Trace t;
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw to A");
        expect(list.draw(make_draw(pso_a, color_b)), "record a draw to B");
        expect(list.draw(make_draw(pso_a, color_b)), "record a second draw to B");
        expect(list.draw(make_draw(pso_stencil, color_b)), "record a draw that also attaches stencil");
        run(list, t);
        expect_well_formed(t, "other targets");
        expect(t.begins == 3 && t.draws == 4, "other targets: a scope for A, one for B, one with stencil");
    }
    // 3. A barrier or a marker between draws ends the scope before it.
    {
        Trace t;
        CommandList list;
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        expect(list.uav_barrier({0}), "record a UAV barrier");
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        expect(list.marker(DebugMarker::Kind::set, "between"), "record a marker");
        expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        run(list, t);
        expect_well_formed(t, "commands between draws");
        expect(t.begins == 3, "commands between draws: three scopes");
    }
    // 4. Predicated draws share a scope that ends before conditional rendering does.
    {
        Trace t;
        CommandList list;
        expect(list.set_predication({predicate_id, 0, d3d12_predication::equal_zero}), "record predication");
        expect(list.draw(make_draw(pso_a, color_a)), "record a predicated draw");
        expect(list.draw(make_draw(pso_a, color_a)), "record a predicated draw");
        run(list, t);
        expect_well_formed(t, "predicated draws");
        expect(t.begins == 1, "predicated draws: one scope");
        expect(t.events.size() >= 2 && t.events[t.events.size() - 2] == "end" && t.events.back() == "end_cond",
               "predicated draws: the scope ends before conditional rendering");
    }
    // 5. merge_draw_scopes = false: one scope per draw, as before.
    {
        vk.merge_draw_scopes = false;
        Trace t;
        CommandList list;
        for (int i = 0; i < 4; ++i) expect(list.draw(make_draw(pso_a, color_a)), "record a draw");
        run(list, t);
        expect_well_formed(t, "no merging");
        expect(t.begins == 4 && t.draws == 4, "no merging: a scope per draw");
        vk.merge_draw_scopes = true;
    }

    // 6. LayoutTracker transactions (review I2): a rolled-back submission leaves the tracker as before it, a committed
    //    one keeps its changes, and images tracked or forgotten inside the transaction are undone too.
    {
        LayoutTracker tracker;
        tracker.track(color_a, rt_a, VK_IMAGE_LAYOUT_GENERAL);
        tracker.track(color_b, rt_b, VK_IMAGE_LAYOUT_GENERAL);
        const VkImageSubresourceRange all{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        tracker.begin();
        tracker.set(color_a, all, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        tracker.set(color_a, all, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);  // journaled once: the first state is kept
        tracker.forget(color_b);
        tracker.track(depth_id, ds, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        tracker.rollback();
        expect(tracker.layout(color_a, 0) == VK_IMAGE_LAYOUT_GENERAL, "rollback: a changed layout is restored");
        expect(tracker.tracked(color_b) && tracker.layout(color_b, 0) == VK_IMAGE_LAYOUT_GENERAL,
               "rollback: a forgotten image is back");
        expect(!tracker.tracked(depth_id), "rollback: an image tracked inside the transaction is gone");
        tracker.begin();
        tracker.set(color_a, all, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        tracker.commit();
        tracker.set(color_b, all, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);  // outside a transaction: not journaled
        tracker.rollback();                                               // nothing to undo
        expect(tracker.layout(color_a, 0) == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL &&
                   tracker.layout(color_b, 0) == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               "commit: the changes are kept");
    }

    if (failures) return 1;
    std::puts("draw scopes: same targets share one scope; other targets, commands between draws and predication "
              "end it; merging off gives a scope per draw; tracker transactions roll back and commit");
    return 0;
}
