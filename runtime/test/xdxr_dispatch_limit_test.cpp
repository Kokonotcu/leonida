// XdxrBuilder (src/xdxr_build.h) keeps every vkCmdDispatch within maxComputeWorkGroupCount[0]: a step whose grid
// is larger is recorded as several dispatches, each with the index of its first thread in the push constants
// (xdxr_build.comp pc.base), which together cover the step's threads once. No GPU: a fake vkGetDeviceProcAddr
// records the dispatches and push constants.
#include "../src/xdxr_build.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

int failures = 0;
#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); ++failures; } \
} while (0)

// xdxr_build.comp `Push` byte offsets
constexpr std::size_t off_mode = 48, off_count = 52, off_base = 104;
constexpr std::uint32_t mode_init = 0, mode_triangles = 1, mode_morton = 3, mode_radix_count = 7,
                        mode_radix_scatter = 9, mode_copy_as = 12, mode_kept = 14, mode_kept_index = 16;

struct Dispatch {
    std::uint32_t groups = 0, mode = 0, count = 0, base = 0;
    std::vector<unsigned char> push;
};
struct Recorder {
    std::vector<unsigned char> push;
    std::vector<Dispatch> dispatches;
    unsigned barriers = 0;
} rec;

template <typename T>
T handle(std::uintptr_t n) { return reinterpret_cast<T>(n); }

std::uint32_t word(const std::vector<unsigned char>& p, std::size_t off) {
    std::uint32_t v = 0;
    if (p.size() >= off + 4) std::memcpy(&v, p.data() + off, 4);
    return v;
}

VKAPI_ATTR VkResult VKAPI_CALL create_module(VkDevice, const VkShaderModuleCreateInfo*, const VkAllocationCallbacks*,
                                             VkShaderModule* m) {
    *m = handle<VkShaderModule>(0x10);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_layout(VkDevice, const VkPipelineLayoutCreateInfo*, const VkAllocationCallbacks*,
                                             VkPipelineLayout* l) {
    *l = handle<VkPipelineLayout>(0x20);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*) {}
VKAPI_ATTR VkResult VKAPI_CALL create_pipelines(VkDevice, VkPipelineCache, std::uint32_t,
                                                const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*,
                                                VkPipeline* p) { *p = handle<VkPipeline>(0x30); return VK_SUCCESS; }
VKAPI_ATTR void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*) {}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t offset,
                                          std::uint32_t size, const void* values) {
    if (rec.push.size() < offset + size) rec.push.resize(offset + size);
    std::memcpy(rec.push.data() + offset, values, size);
}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t x, std::uint32_t, std::uint32_t) {
    rec.dispatches.push_back(
        {x, word(rec.push, off_mode), word(rec.push, off_count), word(rec.push, off_base), rec.push});
}
VKAPI_ATTR void VKAPI_CALL pipeline_barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags,
                                            VkDependencyFlags, std::uint32_t, const VkMemoryBarrier*, std::uint32_t,
                                            const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
    ++rec.barriers;
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get_device_proc(VkDevice, const char* name) {
    struct Entry { const char* name; PFN_vkVoidFunction fn; };
    static const Entry entries[] = {
        {"vkCreateShaderModule", reinterpret_cast<PFN_vkVoidFunction>(create_module)},
        {"vkDestroyShaderModule", reinterpret_cast<PFN_vkVoidFunction>(destroy_module)},
        {"vkCreatePipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(create_layout)},
        {"vkDestroyPipelineLayout", reinterpret_cast<PFN_vkVoidFunction>(destroy_layout)},
        {"vkCreateComputePipelines", reinterpret_cast<PFN_vkVoidFunction>(create_pipelines)},
        {"vkDestroyPipeline", reinterpret_cast<PFN_vkVoidFunction>(destroy_pipeline)},
        {"vkCmdBindPipeline", reinterpret_cast<PFN_vkVoidFunction>(bind_pipeline)},
        {"vkCmdPushConstants", reinterpret_cast<PFN_vkVoidFunction>(push_constants)},
        {"vkCmdDispatch", reinterpret_cast<PFN_vkVoidFunction>(dispatch)},
        {"vkCmdPipelineBarrier", reinterpret_cast<PFN_vkVoidFunction>(pipeline_barrier)},
    };
    for (const Entry& e : entries)
        if (std::strcmp(e.name, name) == 0) return e.fn;
    return nullptr;
}

// One step: a run of dispatches whose push constants differ only in pc.base, the first with base 0.
struct Step {
    std::uint32_t mode = 0, count = 0;
    std::uint64_t groups = 0;   // all of its dispatches together
    unsigned dispatches = 0;
};

// Splits the recording into steps and checks each dispatch: at most `limit` groups, and the next dispatch of a step
// starts where the one before it ended (base counts threads, 64 per group).
std::vector<Step> steps(std::uint32_t limit) {
    std::vector<Step> out;
    std::uint64_t next_base = 0;
    for (std::size_t k = 0; k < rec.dispatches.size(); ++k) {
        const Dispatch& d = rec.dispatches[k];
        EXPECT(d.groups <= limit);
        if (d.groups > limit)
            std::fprintf(stderr, "  dispatch %zu (mode %u): %u groups, limit %u\n", k, d.mode, d.groups, limit);
        bool same = !out.empty() && d.base != 0;
        if (same) {
            std::vector<unsigned char> a = d.push, b = rec.dispatches[k - 1].push;
            std::memset(a.data() + off_base, 0, 4);
            std::memset(b.data() + off_base, 0, 4);
            same = a == b;
            EXPECT(same);                                  // a non-zero base continues the step before it
            EXPECT(d.base == next_base);
        }
        if (!same) out.push_back({d.mode, d.count, 0, 0});
        out.back().groups += d.groups;
        ++out.back().dispatches;
        next_base = (d.base + std::uint64_t(d.groups) * 64);
    }
    return out;
}

const Step* find(const std::vector<Step>& s, std::uint32_t mode) {
    for (const Step& x : s)
        if (x.mode == mode) return &x;
    return nullptr;
}

constexpr std::uint32_t vulkan_min_groups = 65535;   // the least maxComputeWorkGroupCount[0] Vulkan allows

}  // namespace

int main() {
    const VkDevice device = handle<VkDevice>(1);
    const VkCommandBuffer cb = handle<VkCommandBuffer>(2);
    leonida::XdxrBuilder builder(get_device_proc, device);   // no limit given: the least Vulkan guarantees
    EXPECT(builder.loaded());
    if (!builder.loaded()) return 1;

    // CopyRaytracingAccelerationStructure into a 64 MiB pool: one thread per word of the room, 262,144 groups.
    {
        rec = {};
        EXPECT(builder.record_copy(cb, 0x10000, 0x20000, std::uint64_t(64) << 20));
        const auto s = steps(vulkan_min_groups);
        EXPECT(s.size() == 1);
        if (!s.empty()) {
            EXPECT(s[0].mode == mode_copy_as);
            EXPECT(s[0].count == (64u << 20) / 4);
            EXPECT(s[0].groups == (64u << 20) / 4 / 64);
            EXPECT(s[0].dispatches == 5);                  // 4 x 65,535 + 4
        }
    }

    // An update into another buffer: MODE_COPY_AS of the whole layout first (from 98,689 triangles the copy passes
    // 65,535 groups; 150,000 take two dispatches).
    {
        rec = {};
        leonida::XdxrLayout layout;
        EXPECT(layout.compute(150000, false));
        leonida::XdxrTriangles t;
        t.vertices = 0x100000;
        t.triangle_count = 150000;
        EXPECT(builder.record_blas(cb, layout, {t}, 0x4000000, 0x8000000, 0x2000000));
        const auto s = steps(vulkan_min_groups);
        const Step* copy = find(s, mode_copy_as);
        EXPECT(copy && copy->count == layout.size / 4 && copy->groups == (layout.size / 4 + 63) / 64);
        EXPECT(copy && copy->dispatches == 2);
        const Step* tris = find(s, mode_triangles);
        EXPECT(tris && tris->groups == (150000 + 63) / 64 && tris->dispatches == 1);
    }

    // A full build of 4,500,000 triangles: MODE_INIT, MODE_TRIANGLES and MODE_MORTON pass 65,535 groups, and the
    // radix count and scatter (one group per 256 keys) use the tile of their group plus pc.base / 64.
    {
        rec = {};
        const std::uint32_t n = 4500000;
        leonida::XdxrLayout layout;
        EXPECT(layout.compute(n, false));
        leonida::XdxrTriangles t;
        t.vertices = 0x100000;
        t.triangle_count = n;
        EXPECT(builder.record_blas(cb, layout, {t}, 0x40000000, 0x80000000));
        const auto s = steps(vulkan_min_groups);
        const Step* init = find(s, mode_init);
        EXPECT(init && init->groups == (std::uint64_t(layout.parent_entries) + 63) / 64 && init->dispatches > 1);
        const Step* tris = find(s, mode_triangles);
        EXPECT(tris && tris->count == n && tris->groups == (n + 63) / 64 && tris->dispatches == 2);
        const Step* morton = find(s, mode_morton);
        EXPECT(morton && morton->groups == (n + 63) / 64);
        unsigned counts = 0, scatters = 0;
        for (const Step& x : s) {
            if (x.mode == mode_radix_count || x.mode == mode_radix_scatter) {
                EXPECT(x.groups == layout.tiles());
                (x.mode == mode_radix_count ? counts : scatters) += 1;
            }
        }
        EXPECT(counts == 8 && scatters == 8);
    }

    // The device's own limit, as the backend passes it: 300,000 triangles have 1,172 radix tiles and 1,172 kept
    // tiles, so with 1,000 groups the per-tile modes split too.
    {
        leonida::XdxrBuilder small(get_device_proc, device, 1000);
        EXPECT(small.loaded() && small.max_groups() == 1000);
        rec = {};
        const std::uint32_t n = 300000;
        leonida::XdxrLayout layout;
        EXPECT(layout.compute(n, false));
        leonida::XdxrTriangles t;
        t.vertices = 0x100000;
        t.triangle_count = n;
        EXPECT(small.record_blas(cb, layout, {t}, 0x40000000, 0x80000000));
        const auto s = steps(1000);
        const Step* tris = find(s, mode_triangles);
        EXPECT(tris && tris->groups == (n + 63) / 64 && tris->dispatches == 5);
        for (const Step& x : s) {
            if (x.mode == mode_radix_count || x.mode == mode_radix_scatter) EXPECT(x.groups == layout.tiles());
            if (x.mode == mode_kept || x.mode == mode_kept_index) EXPECT(x.groups == layout.kept_tiles());
        }
        EXPECT(find(s, mode_kept) && find(s, mode_kept)->dispatches == 2);
        // 0 is not a limit: the least Vulkan allows instead
        small.set_max_groups(0);
        EXPECT(small.max_groups() == vulkan_min_groups);
    }

    if (failures) {
        std::fprintf(stderr, "xdxr_dispatch_limit_test: %d failures\n", failures);
        return 1;
    }
    std::printf("xdxr_dispatch_limit_test: ok\n");
    return 0;
}
