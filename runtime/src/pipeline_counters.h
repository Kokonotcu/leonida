#pragma once

// Pipelines the runtime created and the time it spent inside vkCreate*Pipelines, for the title HUD
// (shims/title_hud.h, LEONIDA_TITLE_HUD): an inline compile shows up there as a hitch. Counted by wrapping the two
// create calls of the runtime's function table (install(), once per process: the first device wins), and only when the
// HUD is on.

#include <vulkan/vulkan.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace leonida::pipeline_counters {

struct Counters {
    std::atomic<bool> installed{false};
    std::atomic<std::uint64_t> graphics{0}, compute{0};  // pipelines made
    std::atomic<std::uint64_t> create_ns{0};             // time inside vkCreateGraphicsPipelines/ComputePipelines
    PFN_vkCreateGraphicsPipelines create_graphics = nullptr;
    PFN_vkCreateComputePipelines create_compute = nullptr;
};

inline Counters& counters() {
    static Counters c;
    return c;
}

inline std::uint64_t now_ns() {
    return std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count());
}

inline void note(std::atomic<std::uint64_t>& made, std::uint32_t count, const VkPipeline* out, std::uint64_t ns) {
    Counters& c = counters();
    c.create_ns.fetch_add(ns, std::memory_order_relaxed);
    std::uint64_t n = 0;
    for (std::uint32_t i = 0; out && i < count; ++i) n += out[i] != VK_NULL_HANDLE;
    made.fetch_add(n, std::memory_order_relaxed);
}

inline VKAPI_ATTR VkResult VKAPI_CALL create_graphics_pipelines(VkDevice device, VkPipelineCache cache,
                                                                std::uint32_t count,
                                                                const VkGraphicsPipelineCreateInfo* infos,
                                                                const VkAllocationCallbacks* callbacks,
                                                                VkPipeline* out) {
    const std::uint64_t start = now_ns();
    const VkResult result = counters().create_graphics(device, cache, count, infos, callbacks, out);
    note(counters().graphics, result >= 0 ? count : 0, out, now_ns() - start);
    return result;
}

inline VKAPI_ATTR VkResult VKAPI_CALL create_compute_pipelines(VkDevice device, VkPipelineCache cache,
                                                               std::uint32_t count,
                                                               const VkComputePipelineCreateInfo* infos,
                                                               const VkAllocationCallbacks* callbacks,
                                                               VkPipeline* out) {
    const std::uint64_t start = now_ns();
    const VkResult result = counters().create_compute(device, cache, count, infos, callbacks, out);
    note(counters().compute, result >= 0 ? count : 0, out, now_ns() - start);
    return result;
}

template<typename Functions>
void install(Functions& vk) {
    const char* hud = std::getenv("LEONIDA_TITLE_HUD");  // as TitleHud::enabled()
    if (!hud || !*hud || *hud == '0') return;
    Counters& c = counters();
    if (!vk.vkCreateGraphicsPipelines || !vk.vkCreateComputePipelines || c.installed.exchange(true)) return;
    c.create_graphics = vk.vkCreateGraphicsPipelines, vk.vkCreateGraphicsPipelines = create_graphics_pipelines;
    c.create_compute = vk.vkCreateComputePipelines, vk.vkCreateComputePipelines = create_compute_pipelines;
}

}  // namespace leonida::pipeline_counters
