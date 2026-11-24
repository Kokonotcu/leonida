// A counted ExecuteIndirect(DISPATCH) is one count-driven call where the device has generated commands (I7). Vulkan
// has no vkCmdDispatchIndirectCount, so the replay ran a pass that copied the records and zeroed those past the GPU's
// count, then recorded MaxCommandCount vkCmdDispatchIndirect calls: thousands for a large MaxCommandCount, most of
// them empty, every frame. With VK_EXT_device_generated_commands (the path multi-argument signatures already take,
// vulkan_indirect.h) the call becomes a plan with a dispatch-only signature whose sequence count is the GPU's count:
// one vkCmdExecuteGeneratedCommandsEXT. Without the extension, or where it cannot plan the call, nothing changes.
//
// The Vulkan functions and the generated-commands hooks are mocks that count their calls. Checked, for MaxCommandCount
// 4096 with a count buffer: with the hooks, one prepass and one execute and no vkCmdDispatchIndirect; the plan names
// the argument and count buffers, the stride and the count; without them, the counted path as before (one counting
// pass and 4096 indirect dispatches); a call without a count buffer keeps its plain indirect dispatches. The counts
// are printed. No GPU or game data.
#include "../src/vulkan_replay.h"

#include <cstdint>
#include <cstdio>

namespace {

using namespace leonida;

template<class T>
T handle(std::uintptr_t n) {
    return reinterpret_cast<T>(n);
}

unsigned g_dispatch_indirect = 0, g_counting_passes = 0, g_prepasses = 0, g_executes = 0, g_prepares = 0;
IndirectPlan g_last_plan;
IndirectExpander::Request g_last_request;
std::shared_ptr<const IndirectSignature> g_last_signature;  // a copy: the request's pointer dies with the list

VKAPI_ATTR void VKAPI_CALL barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
                                   std::uint32_t, const VkMemoryBarrier*, std::uint32_t, const VkBufferMemoryBarrier*,
                                   std::uint32_t, const VkImageMemoryBarrier*) {}
VKAPI_ATTR void VKAPI_CALL bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline) {}
VKAPI_ATTR void VKAPI_CALL push(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags, std::uint32_t, std::uint32_t,
                                const void*) {}
VKAPI_ATTR void VKAPI_CALL dispatch(VkCommandBuffer, std::uint32_t, std::uint32_t, std::uint32_t) {}
VKAPI_ATTR void VKAPI_CALL dispatch_indirect(VkCommandBuffer, VkBuffer, VkDeviceSize) { ++g_dispatch_indirect; }

int failures = 0;
void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
}

constexpr std::uintptr_t pso = 7;
constexpr std::uint32_t max_count = 4096, stride = 16;

}  // namespace

int main() {
    ResourceBinding args{}, count{};
    args.buffer = handle<VkBuffer>(1);
    args.size = VkDeviceSize(max_count) * stride;
    count.buffer = handle<VkBuffer>(2);
    count.size = 4;
    count.buffer_offset = 256;
    const auto resolve = [&](ResourceId id) -> const ResourceBinding* {
        return id == 1 ? &args : id == 2 ? &count : nullptr;
    };
    ComputePipeline compute{};
    compute.pipeline = handle<VkPipeline>(70);
    compute.layout = handle<VkPipelineLayout>(71);

    const auto base = [&] {
        ReplayFunctions vk{};
        vk.pipeline_barrier = barrier;
        vk.bind_pipeline = bind_pipeline;
        vk.push_constants = push;
        vk.dispatch = dispatch;
        vk.dispatch_indirect = dispatch_indirect;
        vk.compute_pipeline = [&](std::uintptr_t id) -> const ComputePipeline* { return id == pso ? &compute : nullptr; };
        vk.prepare_counted_dispatch = [](VkBuffer, VkDeviceSize, VkBuffer, VkDeviceSize, std::uint32_t,
                                         std::uint32_t, CountedDispatch& out) {
            out.indirect = handle<VkBuffer>(9);
            return true;
        };
        vk.emit_counted_dispatch = [](VkCommandBuffer, const CountedDispatch&) { ++g_counting_passes; };
        return vk;
    };
    const auto with_generated = [&](ReplayFunctions vk) {
        vk.indirect_prepare = [](const IndirectExpander::Request& r, IndirectPlan& plan) {
            ++g_prepares;
            g_last_request = r;
            g_last_signature = r.signature ? *r.signature : nullptr;
            plan = {};
            plan.layout = handle<VkIndirectCommandsLayoutEXT>(80);
            plan.pipeline = r.pipeline;
            plan.max_count = r.max_count;
            plan.count_source = r.count_buffer ? 0x5000 + r.count_offset : 0;
            plan.count_address = r.count_buffer ? 0x6000 : 0;
            g_last_plan = plan;
            return true;
        };
        vk.indirect_prepass = [](VkCommandBuffer, const IndirectPlan&) { ++g_prepasses; };
        vk.indirect_execute = [](VkCommandBuffer, const IndirectPlan&) { ++g_executes; };
        return vk;
    };
    const auto run = [&](const ReplayFunctions& vk, bool counted, const char* what) {
        g_dispatch_indirect = g_counting_passes = g_prepasses = g_executes = g_prepares = 0;
        CommandList list;
        ExecuteIndirect call{};
        call.kind = IndirectKind::dispatch;
        call.stride = stride;
        call.max_count = max_count;
        call.argument_buffer = 1;
        call.argument_offset = 0;
        call.count_buffer = counted ? 2 : 0;
        call.count_offset = 0;
        call.pipeline = pso;
        expect(list.execute_indirect(call) && list.close(), what);
        PreparedList prepared;
        const ReplayResult result = prepare(list, resolve, vk, prepared);
        if (result != ReplayResult::success) {
            std::fprintf(stderr, "FAIL: %s: prepare returned %d\n", what, int(result));
            ++failures;
            return;
        }
        emit(prepared, VK_NULL_HANDLE, vk);
        std::printf("%s: %u vkCmdDispatchIndirect, %u counting passes, %u generated prepasses, %u generated "
                    "executes\n",
                    what, g_dispatch_indirect, g_counting_passes, g_prepasses, g_executes);
    };

    // 1. Counted, with generated commands: one call.
    run(with_generated(base()), true, "1 counted, generated commands");
    expect(g_executes == 1 && g_dispatch_indirect == 0 && g_counting_passes == 0, "1 one generated call");
    expect(g_last_signature && g_last_signature->kind == IndirectKind::dispatch &&
               g_last_signature->stride == stride && g_last_signature->arguments.size() == 1,
           "1 a dispatch-only signature of the call's stride");
    expect(g_last_request.argument_buffer == args.buffer && g_last_request.count_buffer == count.buffer &&
               g_last_request.count_offset == count.buffer_offset && g_last_request.max_count == max_count &&
               g_last_request.pipeline == compute.pipeline,
           "1 the plan names the call's buffers, count and pipeline");

    // 2. Without generated commands: the counted path, as before.
    run(base(), true, "2 counted, no generated commands");
    expect(g_executes == 0 && g_counting_passes == 1 && g_dispatch_indirect == max_count, "2 the counted path");

    // 3. Generated commands that refuse the call: the counted path.
    ReplayFunctions refusing = with_generated(base());
    refusing.indirect_prepare = [](const IndirectExpander::Request&, IndirectPlan&) { return false; };
    run(refusing, true, "3 counted, generated commands refuse");
    expect(g_executes == 0 && g_counting_passes == 1 && g_dispatch_indirect == max_count, "3 the counted path");

    // 4. No count buffer: plain indirect dispatches, untouched.
    run(with_generated(base()), false, "4 uncounted");
    expect(g_executes == 0 && g_prepares == 0 && g_dispatch_indirect == max_count, "4 plain indirect dispatches");

    if (failures) return 1;
    std::printf("replay_counted_dispatch_test: OK\n");
    return 0;
}
