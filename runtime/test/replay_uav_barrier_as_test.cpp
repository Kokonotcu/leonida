// A UAV barrier on a compute queue keeps the acceleration-structure build stage and accesses (B20). With
// VK_KHR_acceleration_structure the replay widens a UAV barrier to the build stage, so a BLAS build followed by a UAV
// barrier orders the build's writes before what reads them; but restrict_to_queue cut every stage and access a compute
// queue's list did not name, and that list lacked the build stage. On async compute the build went unsynchronised.
//
// vkCmdPipelineBarrier is a function that records its stages and memory barrier. Checked: on a compute queue the
// barrier names VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR on both sides and the acceleration-structure
// read and write accesses (fails before); on a copy queue it still names neither (no builds run there); without the
// extension (no build function) a compute queue's barrier names neither. No Vulkan loader, GPU or game data.
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace leonida;

VkPipelineStageFlags g_src = 0, g_dst = 0;
VkAccessFlags g_src_access = 0, g_dst_access = 0;
unsigned g_barriers = 0;

void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags src, VkPipelineStageFlags dst, VkDependencyFlags,
                        std::uint32_t memory_count, const VkMemoryBarrier* memory, std::uint32_t,
                        const VkBufferMemoryBarrier*, std::uint32_t, const VkImageMemoryBarrier*) {
    ++g_barriers;
    g_src = src;
    g_dst = dst;
    g_src_access = memory_count ? memory[0].srcAccessMask : 0;
    g_dst_access = memory_count ? memory[0].dstAccessMask : 0;
}
void VKAPI_CALL build(VkCommandBuffer, std::uint32_t, const VkAccelerationStructureBuildGeometryInfoKHR*,
                      const VkAccelerationStructureBuildRangeInfoKHR* const*) {}

int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

}  // namespace

int main() {
    const auto run = [](VkPipelineStageFlags2 queue_stages, bool acceleration_structures) {
        ReplayFunctions vk{};
        vk.pipeline_barrier = barrier;
        vk.queue_stages = queue_stages;
        if (acceleration_structures) vk.build_acceleration_structures = build;
        CommandList list;
        const bool recorded = list.uav_barrier({0}) && list.close();
        PreparedList prepared;
        const auto resolve = [](ResourceId) -> const ResourceBinding* { return nullptr; };
        const bool ok = recorded && prepare(list, resolve, vk, prepared) == ReplayResult::success;
        g_barriers = 0;
        if (ok) emit(prepared, reinterpret_cast<VkCommandBuffer>(std::uintptr_t(0x10)), vk);
        return ok && g_barriers == 1;
    };
    constexpr VkPipelineStageFlags as_stage = VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    constexpr VkAccessFlags as_access =
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    expect(run(detail::queue_stages_compute, true), "compute queue: one barrier");
    std::printf("compute queue: stages %#x -> %#x, access %#x -> %#x\n", g_src, g_dst, g_src_access, g_dst_access);
    expect((g_src & as_stage) && (g_dst & as_stage), "compute queue: the build stage on both sides");
    expect((g_src_access & VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR) && (g_dst_access & as_access) == as_access,
           "compute queue: the acceleration-structure accesses");

    expect(run(detail::queue_stages_copy, true), "copy queue: one barrier");
    expect(!(g_src & as_stage) && !(g_dst & as_stage) && !(g_dst_access & as_access), "copy queue: no build stage");

    expect(run(detail::queue_stages_compute, false), "compute queue without the extension: one barrier");
    expect(!(g_src & as_stage) && !(g_dst & as_stage) && !(g_dst_access & as_access),
           "compute queue without the extension: no build stage");

    if (failures) return 1;
    std::printf("replay_uav_barrier_as_test: OK\n");
    return 0;
}
