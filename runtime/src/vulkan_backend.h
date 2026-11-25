#pragma once
#include "vulkan_gpu_labels.h"

// The execution side of the D3D12 layer, without COM: the per-device Vulkan context (resources, layout tracker,
// image views), queues that execute command lists, and timeline-semaphore fences. The COM objects in
// shims/d3d12_com.cpp are thin wrappers over these. Needs Vulkan 1.2 timeline semaphores; rectangle clears and
// enhanced barriers need the Vulkan 1.3 functions (dynamicRendering, synchronization2) to be enabled, as core
// features or through their KHR extensions on a 1.2 device (vulkan13_features.h).

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include "binary_occlusion_spirv.h"
#include "buffer_va_index.h"
#include "image_memory.h"
#include "occlusion_sum_spirv.h"
#include "pipeline_workers.h"
#include "vulkan_aftermath.h"
#include "queue_holds.h"
#include "retire_tickets.h"
#include "statistics_sum_spirv.h"
#include "vulkan13_features.h"
#include "vulkan_breadcrumbs.h"
#include "vulkan_depth24.h"
#include "vulkan_title_image_pool.h"
#include "vulkan_indirect_dispatch_count.h"
#include "vulkan_indirect_clamp.h"
#include "vulkan_scratch.h"
#include "vulkan_replay.h"
#include "vulkan_title_pages.h"
#include "queue_families.h"
#include "resource_tiling.h"
#include "pipeline_profile.h"
#include "submit_phases.h"
#include "title_va_alias.h"
#include "xdxr_build.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <chrono>
#include <unordered_set>
#include <shared_mutex>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace leonida {

// This thread's time in the device's title-memory mapping (DeviceState::map_title_memory, host imports), in ns: a
// recording that maps title memory holds resource_mutex_ shared meanwhile (ExecuteTiming::title_map_ms). A plain
// integer: no thread_local with a destructor in gta6rt.
inline thread_local std::uint64_t g_title_map_ns = 0;
// Which check of the queue's replay refused a batch ("ExecuteCommandLists failed: unsupported"), by source line,
// the first time each one does. The batch is then not submitted at all.
inline ReplayResult queue_unsupported(int line) {
    static std::atomic<bool> seen[4096];
    if (line < 0 || line >= 4096 || !seen[line].exchange(true, std::memory_order_relaxed))
        std::fprintf(stderr, "[rt] d3d12: queue replay refused a batch at vulkan_backend.h:%d\n", line);
    return ReplayResult::unsupported;
}

// A CPU-written Vulkan-address instance pointer array, kept until the submission that builds with it completes.
// Completed stages are pooled per queue and reused, so a per-frame TLAS build does not allocate memory each frame.
// The same stages carry the title-memory translation table of an XDXR TLAS build (VulkanQueue::stage_xdxr_table).
struct KhrInstanceStage {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    std::uint64_t size = 0;  // bytes available in buffer, so a larger later list can be refused for a new stage
    VkBufferUsageFlags usage = 0;  // a pooled stage is reused only for the same usage
};

// One per-queue query pool of VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, retained on the
// CommandSlot until its submission completes and then recycled to the queue pool.
struct KhrAsQueryPool {
    VkQueryPool pool = VK_NULL_HANDLE;
    std::uint32_t capacity = 0;
    std::uint32_t used = 0;
};

// VK_EXT_debug_utils labels: an instance extension, so a device has them only when its instance enabled it.
struct DebugLabelFunctions {
    PFN_vkCmdInsertDebugUtilsLabelEXT cmd_insert = nullptr;
    PFN_vkCmdBeginDebugUtilsLabelEXT cmd_begin = nullptr;
    PFN_vkCmdEndDebugUtilsLabelEXT cmd_end = nullptr;
    PFN_vkQueueInsertDebugUtilsLabelEXT queue_insert = nullptr;
    PFN_vkQueueBeginDebugUtilsLabelEXT queue_begin = nullptr;
    PFN_vkQueueEndDebugUtilsLabelEXT queue_end = nullptr;

    bool complete() const { return cmd_insert && cmd_begin && cmd_end && queue_insert && queue_begin && queue_end; }
};

struct VulkanDeviceFunctions {
#define LEONIDA_VK_DEVICE_FUNCTIONS(X)                                                                         \
    X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer)         \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkCreateSemaphore)     \
    X(vkDestroySemaphore) X(vkGetSemaphoreCounterValue) X(vkWaitSemaphores) X(vkSignalSemaphore)               \
    X(vkCreateImageView) X(vkDestroyImageView) X(vkCmdPipelineBarrier) X(vkCmdPipelineBarrier2)                \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage) X(vkCmdBeginRendering) X(vkCmdEndRendering)         \
    X(vkCmdClearAttachments) X(vkDeviceWaitIdle) X(vkCreateImage) X(vkDestroyImage)                         \
    X(vkGetImageMemoryRequirements) X(vkBindImageMemory) X(vkCreateBuffer) X(vkDestroyBuffer)                  \
    X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory) X(vkAllocateMemory) X(vkFreeMemory) X(vkMapMemory)  \
    X(vkUnmapMemory) X(vkCmdCopyBuffer) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer) X(vkCmdCopyImage) X(vkCmdBlitImage)  \
    X(vkCmdResolveImage) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer) X(vkCreateSampler) X(vkDestroySampler)        \
    X(vkCmdBeginConditionalRenderingEXT) X(vkCmdEndConditionalRenderingEXT) X(vkCreateQueryPool)               \
    X(vkDestroyQueryPool) X(vkCmdResetQueryPool) X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCmdWriteTimestamp)    \
    X(vkCmdCopyQueryPoolResults) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect) X(vkCmdDrawIndirectCount)    \
    X(vkCmdDrawIndexedIndirectCount) X(vkCmdDispatchIndirect) X(vkGetBufferDeviceAddress)                      \
    X(vkGetAccelerationStructureBuildSizesKHR) X(vkCreateAccelerationStructureKHR)                             \
    X(vkDestroyAccelerationStructureKHR) X(vkGetAccelerationStructureDeviceAddressKHR)                         \
    X(vkCmdBuildAccelerationStructuresKHR) X(vkCmdCopyAccelerationStructureKHR)                                \
    X(vkCmdWriteAccelerationStructuresPropertiesKHR)                                                           \
    X(vkCreateDescriptorSetLayout) X(vkDestroyDescriptorSetLayout) X(vkCreateDescriptorPool)                   \
    X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)                          \
    X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout) X(vkCreateShaderModule) X(vkDestroyShaderModule)      \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets)            \
    X(vkCmdPushConstants) X(vkCmdDispatch)                                                                     \
    X(vkCmdBindVertexBuffers2) X(vkCmdBindIndexBuffer) X(vkCmdSetViewport) X(vkCmdSetScissor)                  \
    X(vkCmdSetViewportWithCount) X(vkCmdSetScissorWithCount)                                                   \
    X(vkCmdSetPrimitiveTopology) X(vkCmdSetPrimitiveRestartEnable) X(vkCmdSetStencilReference)                          \
    X(vkCmdSetBlendConstants) X(vkCmdDraw)                                                                              \
    X(vkCmdDrawIndexed) X(vkCreateGraphicsPipelines) X(vkQueueBindSparse) X(vkGetImageSparseMemoryRequirements)   \
    X(vkCreatePipelineCache) X(vkDestroyPipelineCache) X(vkGetPipelineCacheData) X(vkMergePipelineCaches)   \
    X(vkCreateEvent) X(vkDestroyEvent) X(vkResetEvent) X(vkCmdSetEvent) X(vkCmdWaitEvents) X(vkCmdDrawMeshTasksEXT) \
    X(vkCmdDrawMeshTasksIndirectEXT) X(vkCmdDrawMeshTasksIndirectCountEXT)                                    \
    X(vkCreateRenderPass2) X(vkDestroyRenderPass) X(vkCreateFramebuffer) X(vkDestroyFramebuffer)               \
    X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass)                                                              \
    X(vkCmdBeginQueryIndexedEXT) X(vkCmdEndQueryIndexedEXT) /* optional: VK_EXT_transform_feedback (D3D-302) */  \
    X(vkCmdWriteBufferMarkerAMD) X(vkGetDeviceFaultInfoEXT) /* optional: device-lost diagnostics (LEONIDA_DRED); vkGetPipelineCacheData is above */
#define LEONIDA_VK_MEMBER(name) PFN_##name name = nullptr;
    LEONIDA_VK_DEVICE_FUNCTIONS(LEONIDA_VK_MEMBER)
#undef LEONIDA_VK_MEMBER

    // The Vulkan 1.3 entry points may be null on older devices, the extension ones (conditional rendering, mesh
    // shaders) when the extension is not enabled, and the query functions are optional (a list that uses queries
    // without them is unsupported). The compute and descriptor functions are needed for normalized or summed
    // occlusion results.
    bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
#define LEONIDA_VK_LOAD(name) name = reinterpret_cast<PFN_##name>(leonida::device_proc_or_khr(get, device, #name));
        LEONIDA_VK_DEVICE_FUNCTIONS(LEONIDA_VK_LOAD)
#undef LEONIDA_VK_LOAD
        return vkCreateCommandPool && vkDestroyCommandPool && vkAllocateCommandBuffers && vkResetCommandBuffer &&
               vkBeginCommandBuffer && vkEndCommandBuffer && vkQueueSubmit && vkQueueWaitIdle && vkCreateSemaphore &&
               vkDestroySemaphore && vkGetSemaphoreCounterValue && vkWaitSemaphores && vkSignalSemaphore &&
               vkCreateImageView && vkDestroyImageView && vkCmdPipelineBarrier && vkCmdClearColorImage &&
               vkCmdClearDepthStencilImage && vkCmdClearAttachments && vkDeviceWaitIdle && vkCreateImage &&
               vkDestroyImage && vkGetImageMemoryRequirements && vkBindImageMemory && vkCreateBuffer &&
               vkDestroyBuffer && vkGetBufferMemoryRequirements && vkBindBufferMemory && vkAllocateMemory &&
               vkFreeMemory && vkMapMemory && vkUnmapMemory && vkCmdCopyBuffer && vkCmdCopyBufferToImage &&
               vkCmdCopyImageToBuffer && vkCmdCopyImage && vkCmdResolveImage && vkCmdFillBuffer &&
               vkCmdUpdateBuffer && vkCmdBindPipeline && vkCmdPushConstants && vkCmdDispatch &&
               vkCreateShaderModule && vkDestroyShaderModule && vkCreatePipelineLayout &&
               vkDestroyPipelineLayout && vkCreateComputePipelines && vkDestroyPipeline &&
               vkGetBufferDeviceAddress && vkCmdBindIndexBuffer && vkCmdSetViewport && vkCmdSetScissor &&
               vkCmdSetStencilReference && vkCmdSetBlendConstants && vkCmdDraw && vkCmdDrawIndexed &&
               vkCreateGraphicsPipelines;
    }

    // Graphics pipelines also need these Vulkan 1.3 entry points (dynamic rendering, extended dynamic state).
    bool graphics_complete() const {
        return vkCmdBeginRendering && vkCmdEndRendering && vkCmdBindVertexBuffers2 && vkCmdSetPrimitiveTopology &&
               vkCmdSetPrimitiveRestartEnable && vkCmdSetViewportWithCount && vkCmdSetScissorWithCount;
    }
    // VK_EXT_transform_feedback is enabled on the device: the indexed-query entry points are loaded, so an
    // SO_STATISTICS heap can be a real transform-feedback pool (D3D-302).
    bool transform_feedback = false;
};

// Both kinds of compute resolve use the same storage buffer, descriptor set and pipeline layout.
// The destination needs transfer usage only. Pipelines are created on first use.
class OcclusionResolve {
public:
    OcclusionResolve(const VulkanDeviceFunctions& vk, VkDevice device) : vk_(vk), device_(device) {}
    OcclusionResolve(const OcclusionResolve&) = delete;
    OcclusionResolve& operator=(const OcclusionResolve&) = delete;

    ~OcclusionResolve() {
        if (sum_pipeline_) vk_.vkDestroyPipeline(device_, sum_pipeline_, nullptr);
        if (binary_pipeline_) vk_.vkDestroyPipeline(device_, binary_pipeline_, nullptr);
        if (statistics_pipeline_) vk_.vkDestroyPipeline(device_, statistics_pipeline_, nullptr);
        if (layout_) vk_.vkDestroyPipelineLayout(device_, layout_, nullptr);
        if (pool_) vk_.vkDestroyDescriptorPool(device_, pool_, nullptr);
        if (set_layout_) vk_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
        if (scratch_) vk_.vkDestroyBuffer(device_, scratch_, nullptr);
        if (memory_) vk_.vkFreeMemory(device_, memory_, nullptr);
    }

    bool initialize(const VkPhysicalDeviceMemoryProperties& memory) {
        if (!vk_.vkCreateDescriptorSetLayout || !vk_.vkDestroyDescriptorSetLayout ||
            !vk_.vkCreateDescriptorPool || !vk_.vkDestroyDescriptorPool || !vk_.vkAllocateDescriptorSets ||
            !vk_.vkUpdateDescriptorSets || !vk_.vkCreatePipelineLayout || !vk_.vkDestroyPipelineLayout ||
            !vk_.vkCreateShaderModule || !vk_.vkDestroyShaderModule || !vk_.vkCreateComputePipelines ||
            !vk_.vkDestroyPipeline || !vk_.vkCmdBindPipeline || !vk_.vkCmdBindDescriptorSets ||
            !vk_.vkCmdPushConstants || !vk_.vkCmdDispatch)
            return false;

        VkBufferCreateInfo buffer{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer.size = scratch_size;
        buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vk_.vkCreateBuffer(device_, &buffer, nullptr, &scratch_) != VK_SUCCESS) return false;
        VkMemoryRequirements requirements{};
        vk_.vkGetBufferMemoryRequirements(device_, scratch_, &requirements);
        auto type = memory.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if (type == memory.memoryTypeCount) type = i;
            if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                type = i;
                break;
            }
        }
        if (type == memory.memoryTypeCount) return false;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (vk_.vkAllocateMemory(device_, &allocation, nullptr, &memory_) != VK_SUCCESS ||
            vk_.vkBindBufferMemory(device_, scratch_, memory_, 0) != VK_SUCCESS)
            return false;

        const VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                   VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo set_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        set_layout.bindingCount = 1;
        set_layout.pBindings = &binding;
        if (vk_.vkCreateDescriptorSetLayout(device_, &set_layout, nullptr, &set_layout_) != VK_SUCCESS) return false;
        const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.maxSets = 1;
        pool.poolSizeCount = 1;
        pool.pPoolSizes = &size;
        if (vk_.vkCreateDescriptorPool(device_, &pool, nullptr, &pool_) != VK_SUCCESS) return false;
        VkDescriptorSetAllocateInfo set{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set.descriptorPool = pool_;
        set.descriptorSetCount = 1;
        set.pSetLayouts = &set_layout_;
        if (vk_.vkAllocateDescriptorSets(device_, &set, &set_) != VK_SUCCESS) return false;
        const VkDescriptorBufferInfo storage{scratch_, 0, scratch_size};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = set_;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &storage;
        vk_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

        // The count, and for statistics_sum.comp whether the segments are mesh-pool results.
        const VkPushConstantRange count{VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(std::uint32_t)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &set_layout_;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &count;
        return vk_.vkCreatePipelineLayout(device_, &layout, nullptr, &layout_) == VK_SUCCESS;
    }

    bool prepare(bool needs_sum, bool needs_binary, bool needs_statistics_sum = false,
                 bool needs_sum_binary = false) {
        // The composite (sum then normalize, D3D-252) uses both pipelines.
        needs_sum = needs_sum || needs_sum_binary;
        needs_binary = needs_binary || needs_sum_binary;
        return (!needs_sum || sum_pipeline_ || create_pipeline(occlusion_sum_spirv, sum_pipeline_)) &&
               (!needs_binary || binary_pipeline_ || create_pipeline(binary_occlusion_spirv, binary_pipeline_)) &&
               (!needs_statistics_sum || statistics_pipeline_ ||
                create_pipeline(statistics_sum_spirv, statistics_pipeline_));
    }

    // Adds `count` pipeline-statistics results into one D3D12_QUERY_DATA_PIPELINE_STATISTICS at dst + offset: the
    // segments of a statistics query that was split around helper dispatches (D3D-162) or mesh draws (D3D-291).
    // Segment i is in `mesh_queries` when mesh[i] is set (two counters: fragment and compute; see
    // statistics_sum.comp), otherwise in `queries` (all eleven). A mesh segment's clipping invocations come from
    // `primitives_queries`, the mesh-primitives count of that segment (D3D-MESH-CLIPPING-COUNTER); it is null when
    // the device has no meshShaderQueries, and the field then reads 0.
    void sum_statistics(VkCommandBuffer cb, VkQueryPool queries, VkQueryPool mesh_queries,
                        VkQueryPool primitives_queries, const std::uint8_t* mesh, std::uint32_t first,
                        std::uint32_t count, VkBuffer dst, VkDeviceSize offset) const {
        constexpr VkDeviceSize result = statistics_counters * word_size;
        constexpr VkDeviceSize mesh_slot = 7 * word_size;  // PSInvocations, where the mesh pool's first counter goes
        constexpr VkDeviceSize clipping_slot = 5 * word_size;  // CInvocations, where the mesh-primitives count goes
        begin_scratch_write(cb);
        vk_.vkCmdFillBuffer(cb, scratch_, 0, result, 0);
        bind(cb, statistics_pipeline_);
        while (count) {
            const bool in_mesh = mesh && mesh[first];
            std::uint32_t chunk = 1;
            while (chunk < std::min(count, statistics_capacity) && bool(mesh && mesh[first + chunk]) == in_mesh)
                ++chunk;
            if (in_mesh) {
                // The mesh pool holds only the fragment and compute counters, so the clipping field of a mesh
                // segment reads 0 unless the mesh-primitives count fills it (D3D-MESH-CLIPPING-COUNTER).
                vk_.vkCmdFillBuffer(cb, scratch_, result, chunk * result, 0);
                barrier(cb, transfer, transfer, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            }
            vk_.vkCmdCopyQueryPoolResults(cb, in_mesh ? mesh_queries : queries, first, chunk, scratch_,
                                          result + (in_mesh ? mesh_slot : 0), result,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            if (in_mesh && primitives_queries)
                vk_.vkCmdCopyQueryPoolResults(cb, primitives_queries, first, chunk, scratch_, result + clipping_slot,
                                              result, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer | compute, compute, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    shader_access);
            dispatch(cb, chunk, 1, in_mesh);
            first += chunk;
            count -= chunk;
            if (count) barrier(cb, compute, transfer, shader_access, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        copy_result(cb, dst, offset, result);
    }

    void sum(VkCommandBuffer cb, VkQueryPool queries, std::uint32_t first, std::uint32_t count,
             VkBuffer dst, VkDeviceSize offset) const {
        begin_scratch_write(cb);
        vk_.vkCmdFillBuffer(cb, scratch_, 0, word_size, 0);
        bind(cb, sum_pipeline_);
        while (count) {
            const auto chunk = std::min(count, sum_capacity);
            vk_.vkCmdCopyQueryPoolResults(cb, queries, first, chunk, scratch_, word_size, word_size,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer | compute, compute, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    shader_access);
            dispatch(cb, chunk, 1);
            first += chunk;
            count -= chunk;
            if (count) barrier(cb, compute, transfer, shader_access, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        copy_result(cb, dst, offset, word_size);
    }

    // Sums `count` binary-occlusion segments into one value and normalizes that total in place to 0 or 1: the
    // result of a binary query whose interval spans more than one physical segment (overlapping queries, D3D-252).
    void sum_binary(VkCommandBuffer cb, VkQueryPool queries, std::uint32_t first, std::uint32_t count,
                    VkBuffer dst, VkDeviceSize offset) const {
        begin_scratch_write(cb);
        vk_.vkCmdFillBuffer(cb, scratch_, 0, word_size, 0);
        bind(cb, sum_pipeline_);
        auto tail = first, left = count;
        while (left) {
            const auto chunk = std::min(left, sum_capacity);
            vk_.vkCmdCopyQueryPoolResults(cb, queries, tail, chunk, scratch_, word_size, word_size,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer | compute, compute, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                    shader_access);
            dispatch(cb, chunk, 1);
            tail += chunk;
            left -= chunk;
            if (left) barrier(cb, compute, transfer, shader_access, VK_ACCESS_TRANSFER_WRITE_BIT);
        }
        barrier(cb, compute, compute, VK_ACCESS_SHADER_WRITE_BIT, shader_access);
        bind(cb, binary_pipeline_);
        dispatch(cb, 1, 1);
        copy_result(cb, dst, offset, word_size);
    }

    void normalize(VkCommandBuffer cb, VkQueryPool queries, std::uint32_t first, std::uint32_t count,
                   VkBuffer dst, VkDeviceSize offset) const {
        bind(cb, binary_pipeline_);
        while (count) {
            const auto chunk = std::min(count, binary_capacity);
            const VkDeviceSize bytes = VkDeviceSize(chunk) * word_size;
            begin_scratch_write(cb);
            vk_.vkCmdCopyQueryPoolResults(cb, queries, first, chunk, scratch_, 0, word_size,
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            barrier(cb, transfer, compute, VK_ACCESS_TRANSFER_WRITE_BIT, shader_access);
            dispatch(cb, chunk, (chunk + binary_group_size - 1) / binary_group_size);
            copy_result(cb, dst, offset, bytes);
            first += chunk;
            count -= chunk;
            offset += bytes;
        }
    }

private:
    bool create_pipeline(std::span<const std::uint32_t> code, VkPipeline& out) const {
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.codeSize = code.size_bytes();
        module.pCode = code.data();
        VkShaderModule shader = VK_NULL_HANDLE;
        const aftermath::ShaderName aftermath_name("(runtime) occlusion resolve");
        if (vk_.vkCreateShaderModule(device_, &module, nullptr, &shader) != VK_SUCCESS) return false;
        VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                          VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", nullptr};
        pipeline.layout = layout_;
        VkPipeline handle = VK_NULL_HANDLE;
        const VkResult result = vk_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &handle);
        vk_.vkDestroyShaderModule(device_, shader, nullptr);
        if (result == VK_SUCCESS) out = handle;
        else if (handle) vk_.vkDestroyPipeline(device_, handle, nullptr);
        return result == VK_SUCCESS;
    }

    void bind(VkCommandBuffer cb, VkPipeline pipeline) const {
        vk_.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vk_.vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set_, 0, nullptr);
    }

    void dispatch(VkCommandBuffer cb, std::uint32_t chunk, std::uint32_t groups, bool mesh = false) const {
        const std::uint32_t range[2] = {chunk, mesh ? 1u : 0u};
        vk_.vkCmdPushConstants(cb, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(range), range);
        vk_.vkCmdDispatch(cb, groups, 1, 1);
    }

    void begin_scratch_write(VkCommandBuffer cb) const {
        // A sum or a normalization in an earlier submission may still use this buffer.
        barrier(cb, transfer | compute, transfer,
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | shader_access,
                VK_ACCESS_TRANSFER_WRITE_BIT);
    }

    void copy_result(VkCommandBuffer cb, VkBuffer dst, VkDeviceSize offset, VkDeviceSize bytes) const {
        barrier(cb, compute, transfer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        const VkBufferCopy copy{0, offset, bytes};
        vk_.vkCmdCopyBuffer(cb, scratch_, dst, 1, &copy);
    }

    void barrier(VkCommandBuffer cb, VkPipelineStageFlags src, VkPipelineStageFlags dst,
                 VkAccessFlags before, VkAccessFlags after) const {
        VkBufferMemoryBarrier memory{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        memory.srcAccessMask = before;
        memory.dstAccessMask = after;
        memory.srcQueueFamilyIndex = memory.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memory.buffer = scratch_;
        memory.size = scratch_size;
        vk_.vkCmdPipelineBarrier(cb, src, dst, 0, 0, nullptr, 1, &memory, 0, nullptr);
    }

    static constexpr auto transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
    static constexpr auto compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    static constexpr VkAccessFlags shader_access = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    static constexpr std::uint32_t sum_capacity = 256;  // one more word holds the running sum
    static constexpr std::uint32_t binary_capacity = 1024;
    static constexpr std::uint32_t binary_group_size = 64;  // local_size_x in binary_occlusion.comp
    static constexpr std::uint32_t statistics_counters = 11;  // local_size_x in statistics_sum.comp
    static constexpr std::uint32_t statistics_capacity = 64;  // segments per pass; 11 more words hold the sums
    static constexpr VkDeviceSize word_size = sizeof(std::uint64_t);
    static constexpr VkDeviceSize scratch_size =
        std::max({sum_capacity + 1, binary_capacity, (statistics_capacity + 1) * statistics_counters}) * word_size;
    const VulkanDeviceFunctions& vk_;
    VkDevice device_;
    VkBuffer scratch_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline sum_pipeline_ = VK_NULL_HANDLE;
    VkPipeline binary_pipeline_ = VK_NULL_HANDLE;
    VkPipeline statistics_pipeline_ = VK_NULL_HANDLE;
};

struct SparseResource;
using ImageMemory = ImageMemoryPool<VkDeviceMemory>;

// An object that frees itself once the GPU is done with it (DeferredDeletionQueue::Kind::release): a range of a
// suballocated memory block goes back to its block this way (vulkan_suballocator.h).
struct DeferredRelease {
    void (*release)(DeferredRelease* self) = nullptr;
};

// Zero-allocation O(1) Ring Buffer deferred deletion queue tied to the device timeline semaphore.
class DeferredDeletionQueue {
public:
    enum class Kind : std::uint8_t {
        bindless_slot,
        image_view,
        image,
        sparse_resource,
        pooled_title_image,
        buffer,
        device_memory,
        image_memory,  // an ImageMemory::Range*, deleted: its bytes go back to their block
        acceleration_structure,
        framebuffer,
        release,  // handle: a DeferredRelease*
        pipeline,
    };

    struct Item {
        std::uint64_t ticket = 0;
        std::uint64_t handle = 0;  // for bindless_slot: (uint64_t(dimension) << 32) | index
        Kind kind = Kind::image_view;
    };

    using BindlessReleaseFn = void (*)(void* user, std::uint32_t dimension, std::uint32_t index);

    explicit DeferredDeletionQueue(std::size_t initial_capacity = 256) : pending_(next_pow2(initial_capacity)) {}

    void set_bindless_release(BindlessReleaseFn fn, void* user) {
        std::lock_guard lock(mutex_);
        bindless_release_fn_ = fn;
        bindless_release_user_ = user;
    }

    void push(std::uint64_t retire_ticket, Kind kind, std::uint64_t handle) {
        if (!handle && kind != Kind::bindless_slot) return;
        std::lock_guard lock(mutex_);
        if (count_ == pending_.size()) {
            const std::size_t old_cap = pending_.size();
            std::vector<Item> grown(old_cap * 2);
            const std::size_t first = old_cap - head_;
            std::memcpy(grown.data(), pending_.data() + head_, first * sizeof(Item));
            if (head_ > 0) std::memcpy(grown.data() + first, pending_.data(), head_ * sizeof(Item));
            pending_ = std::move(grown);
            head_ = 0;
            tail_ = count_;
        }
        pending_[tail_] = {retire_ticket, handle, kind};
        tail_ = (tail_ + 1) & (pending_.size() - 1);
        ++count_;
    }

    // Drains all items with ticket <= current_ticket in O(k), invoking vkDestroy* outside the lock.
    // Uses a stack-local batch vector so concurrent drain() calls from multiple queues never race.
    void drain(std::uint64_t current_ticket, VkDevice device, const VulkanDeviceFunctions& vk) {
        BindlessReleaseFn release_fn = nullptr;
        void* release_user = nullptr;
        std::vector<Item> to_destroy;
        {
            std::lock_guard lock(mutex_);
            if (count_ == 0 || pending_[head_].ticket > current_ticket) return;
            release_fn = bindless_release_fn_;
            release_user = bindless_release_user_;
            const std::size_t mask = pending_.size() - 1;
            while (count_ > 0 && pending_[head_].ticket <= current_ticket) {
                to_destroy.push_back(pending_[head_]);
                head_ = (head_ + 1) & mask;
                --count_;
            }
            if (count_ == 0) {
                head_ = 0;
                tail_ = 0;
            }
        }

        for (const Item& item : to_destroy) {
            switch (item.kind) {
            case Kind::bindless_slot:
                if (release_fn) {
                    release_fn(release_user, static_cast<std::uint32_t>(item.handle >> 32),
                               static_cast<std::uint32_t>(item.handle & 0xFFFFFFFFu));
                }
                break;
            case Kind::image_view:
                vk.vkDestroyImageView(device, reinterpret_cast<VkImageView>(item.handle), nullptr);
                break;
            case Kind::image:
                vk.vkDestroyImage(device, reinterpret_cast<VkImage>(item.handle), nullptr);
                break;
            case Kind::pooled_title_image:
                delete reinterpret_cast<PooledTitleImage*>(item.handle);
                break;
            case Kind::sparse_resource:
                // Releases the resource and its page backing together after outstanding submissions.
                delete reinterpret_cast<std::shared_ptr<SparseResource>*>(item.handle);
                break;
            case Kind::buffer:
                vk.vkDestroyBuffer(device, reinterpret_cast<VkBuffer>(item.handle), nullptr);
                break;
            case Kind::device_memory:
                vk.vkFreeMemory(device, reinterpret_cast<VkDeviceMemory>(item.handle), nullptr);
                break;
            case Kind::image_memory:
                delete reinterpret_cast<ImageMemory::Range*>(item.handle);
                break;
            case Kind::acceleration_structure:
                if (vk.vkDestroyAccelerationStructureKHR)
                    vk.vkDestroyAccelerationStructureKHR(
                        device, reinterpret_cast<VkAccelerationStructureKHR>(item.handle), nullptr);
                break;
            case Kind::framebuffer:
                vk.vkDestroyFramebuffer(device, reinterpret_cast<VkFramebuffer>(item.handle), nullptr);
                break;
            case Kind::release:
                if (auto* release = reinterpret_cast<DeferredRelease*>(item.handle); release->release)
                    release->release(release);
                break;
            case Kind::pipeline:
                vk.vkDestroyPipeline(device, reinterpret_cast<VkPipeline>(item.handle), nullptr);
                break;
            }
        }
    }

private:
    static std::size_t next_pow2(std::size_t n) {
        std::size_t p = 16;
        while (p < n) p <<= 1;
        return p;
    }

    std::mutex mutex_;
    std::vector<Item> pending_;
    std::size_t head_ = 0;
    std::size_t tail_ = 0;
    std::size_t count_ = 0;
    BindlessReleaseFn bindless_release_fn_ = nullptr;
    void* bindless_release_user_ = nullptr;
};

// Sets a device's lost flag and says so once, on stderr, at the moment it is first seen. From then on every fence of
// the device reads UINT64_MAX and device calls return DXGI_ERROR_DEVICE_REMOVED, so a title takes its own
// device-removed path (on the retail boot an int3 at game_scarlett_master.exe+0x2474606, reached from its frame wait
// reading a fence's completed value as ~0); without this line that later stop is the first sign of the loss.
// True when this call set the flag. device_lost_reports counts the lines said in this process (for tests).
inline std::atomic<unsigned> device_lost_reports{0};
inline bool set_device_lost(std::atomic<bool>& flag, const char* where) noexcept {
    if (flag.exchange(true, std::memory_order_acq_rel)) return false;
    device_lost_reports.fetch_add(1, std::memory_order_relaxed);
    std::fprintf(stderr,
                 "[rt] vulkan: the device was lost (%s); its fences now read UINT64_MAX and device calls return "
                 "DXGI_ERROR_DEVICE_REMOVED\n",
                 where);
    return true;
}
// LEONIDA_GPU_LATENCY=1 (diagnostic): for each queue kind, how long the GPU took from a submission's vkQueueSubmit to
// the dispatcher seeing it done, and how much of the wall time the queue had work (in-order queue: a submission is
// busy from the later of its submit and the previous one's completion to its own). One line per kind every 10 s.
namespace gpu_latency {
inline bool enabled() {
    static const bool on = [] {
        const char* v = std::getenv("LEONIDA_GPU_LATENCY");
        return v && std::strcmp(v, "1") == 0;
    }();
    return on;
}
struct Stats {
    std::mutex mutex;
    std::uint64_t count = 0;
    double sum_ms = 0, max_ms = 0, busy_ms = 0;
    std::chrono::steady_clock::time_point last_done{}, since = std::chrono::steady_clock::now();
};
inline Stats& stats(int kind) {
    static Stats s[3];
    return s[kind < 0 || kind > 2 ? 0 : kind];
}
inline void done(int kind, std::chrono::steady_clock::time_point submitted) {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    const auto ms = [](clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    Stats& s = stats(kind);
    std::lock_guard lock(s.mutex);
    const double latency = ms(now - submitted);
    ++s.count;
    s.sum_ms += latency;
    s.max_ms = (std::max)(s.max_ms, latency);
    s.busy_ms += ms(now - (std::max)(submitted, s.last_done));
    s.last_done = now;
    if (now - s.since >= std::chrono::seconds(10)) {
        const double wall = ms(now - s.since);
        std::fprintf(stderr, "[rt] gpu latency (%s): %llu submissions in %.0f s, submit to done %.2f ms on average, "
                             "%.1f ms the longest; the queue had work %.1f%% of the time\n",
                     kind == 0 ? "direct" : kind == 1 ? "compute" : "copy",
                     static_cast<unsigned long long>(s.count), wall / 1000.0, s.sum_ms / double(s.count), s.max_ms,
                     100.0 * s.busy_ms / wall);
        s.count = 0;
        s.sum_ms = s.max_ms = s.busy_ms = 0;
        s.since = now;
    }
}
// A held queue operation (Wait before Signal), released: how long it waited on the CPU before it could run.
inline void held(int kind, int op, std::chrono::steady_clock::time_point at) {
    using clock = std::chrono::steady_clock;
    struct Held {
        std::mutex mutex;
        std::uint64_t count[3] = {};
        double sum_ms = 0, max_ms = 0;
        clock::time_point since = clock::now();
    };
    static Held h[3];
    Held& s = h[kind < 0 || kind > 2 ? 0 : kind];
    const auto now = clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - at).count();
    std::lock_guard lock(s.mutex);
    ++s.count[op < 0 || op > 2 ? 0 : op];
    s.sum_ms += ms;
    s.max_ms = (std::max)(s.max_ms, ms);
    if (now - s.since >= std::chrono::seconds(10)) {
        const std::uint64_t all = s.count[0] + s.count[1] + s.count[2];
        std::fprintf(stderr, "[rt] gpu latency (%s): %llu held operations released in %.0f s (%llu batches, %llu "
                             "signals, %llu markers), held %.2f ms on average, %.1f ms the longest\n",
                     kind == 0 ? "direct" : kind == 1 ? "compute" : "copy", static_cast<unsigned long long>(all),
                     std::chrono::duration<double>(now - s.since).count(), static_cast<unsigned long long>(s.count[0]),
                     static_cast<unsigned long long>(s.count[1]), static_cast<unsigned long long>(s.count[2]),
                     all ? s.sum_ms / double(all) : 0.0, s.max_ms);
        s.count[0] = s.count[1] = s.count[2] = 0;
        s.sum_ms = s.max_ms = 0;
        s.since = now;
    }
}
}  // namespace gpu_latency

// Centralized single-thread fence dispatcher using Vulkan 1.2 vkWaitSemaphores(VK_SEMAPHORE_WAIT_ANY_BIT).
// Eliminates per-fence OS threads and 10ms polling: a synthetic timeline semaphore (wakeup_sem_) at index 0
// wakes vkWaitSemaphores immediately whenever a new wait, a released fence, or shutdown arrives while waiting_ == true.
// The last ID3D12CommandQueue::Signal any queue made, for FenceWaitDispatcher::report_stall.
namespace detail_stall {
inline std::atomic<VkSemaphore> last_signal_semaphore{VK_NULL_HANDLE};
inline std::atomic<std::uint64_t> last_signal_value{0};
inline std::atomic<bool> last_signal_held{false};
// LEONIDA_DRED's ring, when configured: a stall names the lists the GPU started and did not finish.
inline std::atomic<BreadcrumbRing*> breadcrumbs{nullptr};
// Per queue (its progress timeline): the last submitted ticket and the recent submissions that carried waits, so a
// stalled GPU queue can be told apart: a wait whose value is never reached, or work that never finishes.
struct WaitRecord {
    std::uint64_t ticket = 0;
    std::vector<std::pair<VkSemaphore, std::uint64_t>> waits;
};
struct QueueRecord {
    std::uint64_t submitted = 0;
    std::deque<WaitRecord> waited;  // newest last, at most 64
};
inline std::mutex queues_mutex;
inline std::map<VkSemaphore, QueueRecord> queues;
inline void note_submit(VkSemaphore progress, std::uint64_t ticket, std::span<const VkSemaphore> semaphores,
                        std::span<const std::uint64_t> values) {
    std::lock_guard lock(queues_mutex);
    QueueRecord& q = queues[progress];
    q.submitted = ticket;
    if (semaphores.empty()) return;
    WaitRecord r{ticket, {}};
    for (std::size_t i = 0; i < semaphores.size() && i < values.size(); ++i) r.waits.emplace_back(semaphores[i], values[i]);
    q.waited.push_back(std::move(r));
    if (q.waited.size() > 64) q.waited.pop_front();
}
}  // namespace detail_stall

class FenceWaitDispatcher {
public:
    struct Entry {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        std::uint64_t value = 0;
        void* event_handle = nullptr;
        std::function<void()> callback;
    };

    // A dispatch runs outside mutex_. Each entry in a ready batch gets its own slot: a callback may re-entrantly
    // call cancel() for a later entry in the same batch, which marks that entry cancelled so it is skipped before
    // it can use the semaphore. A callback running on another thread must still be waited for.
    // A cancelled entry still in pending_ is dispatched by cancel() itself (callback and bookkeeping both run);
    // one already in a ready batch is skipped and only its event is signalled. The held Signal op's count is
    // therefore not decremented for a skipped batch entry. That is safe because only ~Fence cancels a wait, but
    // keep it in mind if cancel() gains other callers.
    struct DispatchSlot {
        VkSemaphore semaphore = VK_NULL_HANDLE;
        std::thread::id owner;
        bool active = false;     // guarded by mutex_
        bool cancelled = false;  // guarded by mutex_
    };

    using SignalEventFn = void (*)(void* event_handle);

    // Longest single vkWaitSemaphores call in run(): bounds how long the thread can sit in an uninterruptible wait.
    static constexpr std::uint64_t kWaitSliceNs = 100'000'000;  // 100 ms

    FenceWaitDispatcher(VkDevice device, const VulkanDeviceFunctions& vk, std::atomic<bool>& device_lost)
        : device_(device), vk_(vk), device_lost_(device_lost) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.initialValue = 0;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        if (vk_.vkCreateSemaphore(device_, &info, nullptr, &wakeup_sem_) != VK_SUCCESS)
            wakeup_sem_ = VK_NULL_HANDLE;
        semaphores_.reserve(64);
        values_.reserve(64);
    }

    ~FenceWaitDispatcher() {
        stop();
        if (wakeup_sem_) vk_.vkDestroySemaphore(device_, wakeup_sem_, nullptr);
    }

    FenceWaitDispatcher(const FenceWaitDispatcher&) = delete;
    FenceWaitDispatcher& operator=(const FenceWaitDispatcher&) = delete;

    void set_device_fault(DeviceFaultDiagnostics* diagnostics) { diagnostics_ = diagnostics; }

    void set_signal_event_fn(SignalEventFn fn) {
        std::lock_guard lock(mutex_);
        signal_event_ = fn;
    }

    void enqueue(VkSemaphore sem, std::uint64_t value, void* event_handle) {
        if (!sem || !event_handle) return;
        SignalEventFn immediate_fn = nullptr;
        {
            std::lock_guard lock(mutex_);
            if (stop_ || device_lost_.load(std::memory_order_acquire)) {
                immediate_fn = signal_event_;
            } else {
                pending_.push_back({sem, value, event_handle, {}});
                if (!thread_.joinable() && wakeup_sem_) thread_ = std::thread([this] { run(); });
                signal_wakeup_locked();
            }
        }
        if (immediate_fn) {
            immediate_fn(event_handle);
            return;
        }
        cv_.notify_one();
    }

    void enqueue_callback(VkSemaphore sem, std::uint64_t value, std::function<void()> cb) {
        if (!sem || !cb) return;
        std::function<void()> immediate_cb;
        {
            std::lock_guard lock(mutex_);
            if (stop_ || device_lost_.load(std::memory_order_acquire)) {
                immediate_cb = std::move(cb);
            } else {
                pending_.push_back({sem, value, nullptr, std::move(cb)});
                if (!thread_.joinable() && wakeup_sem_) thread_ = std::thread([this] { run(); });
                signal_wakeup_locked();
            }
        }
        if (immediate_cb) {
            immediate_cb();
            return;
        }
        cv_.notify_one();
    }

    // Wakes run() when another path marks the device lost so pending waits complete and stop()/cancel() never hang.
    void notify_device_lost() {
        {
            std::lock_guard lock(mutex_);
            signal_wakeup_locked();
        }
        cv_.notify_all();
    }

    // Completes the pending waits for `sem` (a released D3D12 fence counts as reached, as on the PC runtime) and
    // blocks until both a vkWaitSemaphores on `sem` and any callback already dispatched for it have finished, before
    // the caller destroys the VkSemaphore.
    void cancel(VkSemaphore sem) {
        if (!sem) return;
        std::vector<Entry> cancelled;
        SignalEventFn fn = nullptr;
        {
            std::unique_lock lock(mutex_);
            fn = signal_event_;
            std::size_t write = 0;
            for (std::size_t i = 0; i < pending_.size(); ++i) {
                if (pending_[i].semaphore == sem) {
                    cancelled.push_back(std::move(pending_[i]));
                } else {
                    if (write != i) pending_[write] = std::move(pending_[i]);
                    ++write;
                }
            }
            pending_.resize(write);
            // Entries already taken out of pending_ but not yet started belong to a live ready batch. Mark them
            // cancelled so dispatch skips them; this is how a re-entrant cancel() from another callback in the
            // same batch is completed without waiting on the callback that is already on this thread's stack.
            for (const auto& slot : dispatching_) {
                if (slot->semaphore == sem && !slot->active) slot->cancelled = true;
            }
            if (thread_.joinable() && (waiting_ || callback_on_other_thread(sem))) {
                const std::uint64_t target_epoch = epoch_ + 1;
                signal_wakeup_locked();
                epoch_cv_.wait(lock, [this, sem, target_epoch] {
                    return (!waiting_ || epoch_ >= target_epoch) && !callback_on_other_thread(sem);
                });
            }
        }
        dispatch_entries(cancelled, fn);
    }

    void stop() {
        {
            std::lock_guard lock(mutex_);
            if (stop_) return;
            stop_ = true;
            signal_wakeup_locked();
        }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    static void dispatch_entries(std::vector<Entry>& entries, SignalEventFn signal_event) {
        for (Entry& e : entries) {
            if (e.callback) e.callback();
            else if (signal_event && e.event_handle) signal_event(e.event_handle);
        }
    }

    // True only for a callback already running on another thread. A slot that has not started yet is cancelled by
    // cancel() and skipped, so it never uses a destroyed semaphore and must not hold up the cancelling caller.
    bool callback_on_other_thread(VkSemaphore sem) const {
        const std::thread::id me = std::this_thread::get_id();
        return std::any_of(dispatching_.begin(), dispatching_.end(), [&](const auto& d) {
            return d->semaphore == sem && d->active && d->owner != me;
        });
    }

    // mutex_ is held on entry and again on return. Recording every entry before unlocking closes the window where
    // run() has taken them out of pending_ but has not started the callback, yet cancel() can return and let the
    // caller destroy the semaphore. Each entry has its own slot so a cancel() for a later entry in the same batch
    // skips that entry instead of being mistaken for the callback already on the cancelling thread's stack.
    void dispatch_outside_lock(std::unique_lock<std::mutex>& lock, std::vector<Entry>& entries, SignalEventFn fn) {
        const std::thread::id me = std::this_thread::get_id();
        std::vector<std::shared_ptr<DispatchSlot>> slots;
        slots.reserve(entries.size());
        for (const Entry& e : entries) {
            auto slot = std::make_shared<DispatchSlot>();
            slot->semaphore = e.semaphore;
            slot->owner = me;
            slots.push_back(slot);
            dispatching_.push_back(slot);
        }
        lock.unlock();
        for (std::size_t i = 0; i < entries.size(); ++i) {
            Entry& e = entries[i];
            DispatchSlot& slot = *slots[i];
            bool skip = false;
            {
                std::lock_guard guard(mutex_);
                skip = slot.cancelled;
                if (!skip) slot.active = true;
            }
            if (skip) {
                // A re-entrant cancel() removed this wait; still complete its event so no waiter is left behind.
                if (fn && e.event_handle) fn(e.event_handle);
                continue;
            }
            if (e.callback) e.callback();
            else if (fn && e.event_handle) fn(e.event_handle);
            {
                std::lock_guard guard(mutex_);
                slot.active = false;
                const auto it = std::find(dispatching_.begin(), dispatching_.end(), slots[i]);
                if (it != dispatching_.end()) dispatching_.erase(it);
            }
            epoch_cv_.notify_all();
        }
        lock.lock();
        for (const auto& slot : slots) {
            const auto it = std::find(dispatching_.begin(), dispatching_.end(), slot);
            if (it != dispatching_.end()) dispatching_.erase(it);
        }
        epoch_cv_.notify_all();
    }

    // Only signals wakeup_sem_ when run() is actively waiting (waiting_ == true) on wakeup_val_ + 1,
    // eliminating any lost-wakeup race between snapshot construction and vkWaitSemaphores. Keep signaling even if
    // device_lost_ is already set so a blocked vkWaitSemaphores wakes cleanly for cancel()/stop().
    void signal_wakeup_locked() {
        if (!waiting_ || !wakeup_sem_) return;
        const std::uint64_t next = ++wakeup_val_;
        VkSemaphoreSignalInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        si.semaphore = wakeup_sem_;
        si.value = next;
        DeviceFaultDiagnostics::observe(diagnostics_, vk_.vkSignalSemaphore(device_, &si),
                                       "vkSignalSemaphore (fence wakeup)");
    }

    void run() {
        std::unique_lock lock(mutex_);
        while (true) {
            cv_.wait(lock, [this] { return stop_ || !pending_.empty() || device_lost_.load(std::memory_order_acquire); });
            if (device_lost_.load(std::memory_order_acquire)) {
                std::vector<Entry> drained = std::move(pending_);
                pending_.clear();
                dispatch_outside_lock(lock, drained, signal_event_);
                return;
            }
            if (stop_) return;

            semaphores_.clear();
            values_.clear();
            semaphores_.push_back(wakeup_sem_);
            values_.push_back(wakeup_val_ + 1);
            for (const Entry& e : pending_) {
                semaphores_.push_back(e.semaphore);
                values_.push_back(e.value);
            }

            waiting_ = true;
            lock.unlock();

            VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wi.flags = VK_SEMAPHORE_WAIT_ANY_BIT;
            wi.semaphoreCount = static_cast<std::uint32_t>(semaphores_.size());
            wi.pSemaphores = semaphores_.data();
            wi.pValues = values_.data();
            // A bounded slice, not UINT64_MAX. Under GPU paravirtualization (measured in Windows Sandbox on an
            // RTX 5060 Ti) this wait is a kernel wait the OS cannot interrupt: a process that exits while a wait
            // is pending (d3d12_queue_wait_timeout's _Exit, or quitting with a fence outstanding) never finishes
            // exiting, and TerminateProcess cannot end it either. A timeout returns here and the loop re-arms the
            // same wait; completions still wake it at once through the ANY wait.
            const VkResult res = DeviceFaultDiagnostics::observe(
                diagnostics_, vk_.vkWaitSemaphores(device_, &wi, kWaitSliceNs), "vkWaitSemaphores (fence dispatcher)");

            lock.lock();
            waiting_ = false;
            ++epoch_;
            epoch_cv_.notify_all();

            if (res == VK_ERROR_DEVICE_LOST || device_lost_.load(std::memory_order_acquire)) {
                set_device_lost(device_lost_, "vkWaitSemaphores in the fence dispatcher returned VK_ERROR_DEVICE_LOST");
                std::vector<Entry> drained = std::move(pending_);
                pending_.clear();
                dispatch_outside_lock(lock, drained, signal_event_);
                return;
            }
            if (stop_) return;

            std::vector<Entry> ready;
            bool lost = false;
            std::size_t write = 0;
            for (std::size_t i = 0; i < pending_.size(); ++i) {
                std::uint64_t done = 0;
                const VkResult cr = DeviceFaultDiagnostics::observe(
                    diagnostics_, vk_.vkGetSemaphoreCounterValue(device_, pending_[i].semaphore, &done),
                    "vkGetSemaphoreCounterValue (fence dispatcher)");
                if (cr == VK_ERROR_DEVICE_LOST) {
                    lost = true;
                    for (std::size_t j = i; j < pending_.size(); ++j) ready.push_back(std::move(pending_[j]));
                    break;
                }
                if (cr == VK_SUCCESS && done >= pending_[i].value) {
                    ready.push_back(std::move(pending_[i]));
                } else {
                    if (write != i) pending_[write] = std::move(pending_[i]);
                    ++write;
                }
            }
            if (lost) {
                pending_.resize(write);
                for (Entry& e : pending_) ready.push_back(std::move(e));
                pending_.clear();
                set_device_lost(device_lost_,
                                "vkGetSemaphoreCounterValue in the fence dispatcher returned VK_ERROR_DEVICE_LOST");
                dispatch_outside_lock(lock, ready, signal_event_);
                return;
            }
            pending_.resize(write);
            report_stall(ready.empty());
            if (!ready.empty()) {
                dispatch_outside_lock(lock, ready, signal_event_);
                if (stop_) return;
            }
        }
    }

    // A fence wait that has not moved for 10 s is named once per stall, with each pending semaphore's wanted and
    // current value: a value never reached means the GPU queue stopped (or its Signal was never submitted), which
    // a title waiting on its frame fence turns into its own watchdog kill. Called under mutex_.
    void report_stall(bool idle) {
        const auto now = std::chrono::steady_clock::now();
        if (!idle || pending_.empty()) {
            last_progress_ = now;
            stall_reported_ = false;
            return;
        }
        if (stall_reported_ || now - last_progress_ < std::chrono::seconds(10)) return;
        stall_reported_ = true;
        std::fprintf(stderr, "[rt] fence waits: no progress for 10 s, %zu pending:\n", pending_.size());
        for (std::size_t i = 0; i < pending_.size() && i < 16; ++i) {
            std::uint64_t done = 0;
            vk_.vkGetSemaphoreCounterValue(device_, pending_[i].semaphore, &done);
            std::fprintf(stderr, "[rt]   semaphore %p wants %llu, at %llu (%s)\n",
                         reinterpret_cast<void*>(pending_[i].semaphore),
                         static_cast<unsigned long long>(pending_[i].value), static_cast<unsigned long long>(done),
                         pending_[i].callback ? "callback" : "event");
        }
        std::fprintf(stderr, "[rt]   last queue Signal: semaphore %p value %llu%s\n",
                     reinterpret_cast<void*>(detail_stall::last_signal_semaphore.load(std::memory_order_relaxed)),
                     static_cast<unsigned long long>(detail_stall::last_signal_value.load(std::memory_order_relaxed)),
                     detail_stall::last_signal_held.load(std::memory_order_relaxed) ? " (held behind earlier work)" : "");
        std::lock_guard queues_lock(detail_stall::queues_mutex);
        for (const auto& [progress, q] : detail_stall::queues) {
            std::uint64_t done = 0;
            vk_.vkGetSemaphoreCounterValue(device_, progress, &done);
            std::fprintf(stderr, "[rt]   queue timeline %p: submitted %llu, completed %llu\n",
                         reinterpret_cast<void*>(progress), static_cast<unsigned long long>(q.submitted),
                         static_cast<unsigned long long>(done));
            for (const detail_stall::WaitRecord& r : q.waited) {
                if (r.ticket <= done) continue;
                for (const auto& [sem, value] : r.waits) {
                    std::uint64_t at = 0;
                    vk_.vkGetSemaphoreCounterValue(device_, sem, &at);
                    std::fprintf(stderr, "[rt]     submission %llu waits on %p for %llu, at %llu%s\n",
                                 static_cast<unsigned long long>(r.ticket), reinterpret_cast<void*>(sem),
                                 static_cast<unsigned long long>(value), static_cast<unsigned long long>(at),
                                 at < value ? "  <-- not reached" : "");
                }
                break;  // the first incomplete submission with waits
            }
        }
        if (BreadcrumbRing* ring = detail_stall::breadcrumbs.load(std::memory_order_acquire)) ring->report(stderr);
    }

    VkDevice device_;
    const VulkanDeviceFunctions& vk_;
    std::atomic<bool>& device_lost_;
    DeviceFaultDiagnostics* diagnostics_ = nullptr;  // configured before the dispatcher starts
    SignalEventFn signal_event_ = nullptr;
    VkSemaphore wakeup_sem_ = VK_NULL_HANDLE;
    std::uint64_t wakeup_val_ = 0;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::condition_variable epoch_cv_;
    std::vector<Entry> pending_;
    std::vector<std::shared_ptr<DispatchSlot>> dispatching_;  // ready-batch callbacks outside mutex_
    std::vector<VkSemaphore> semaphores_;
    std::vector<std::uint64_t> values_;
    std::uint64_t epoch_ = 0;
    bool waiting_ = false;
    bool stop_ = false;
    std::chrono::steady_clock::time_point last_progress_ = std::chrono::steady_clock::now();  // report_stall
    bool stall_reported_ = false;
    std::thread thread_;
};

// A range of a D3D12 heap allocation. Mappings and in-flight unbinds keep the arena range alive.
struct TileHeapAllocation {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0;
    std::uint32_t memory_type = 0;
};

struct TileMapping {
    std::shared_ptr<const TileHeapAllocation> heap;
    VkDeviceSize offset = 0;
    bool operator==(const TileMapping&) const = default;
};

struct SparseResource {
    SparseResource(VkDevice device, const VulkanDeviceFunctions& vk) : device(device), vk(vk) {}
    SparseResource(const SparseResource&) = delete;
    SparseResource& operator=(const SparseResource&) = delete;
    ~SparseResource() {
        if (image) vk.vkDestroyImage(device, image, nullptr);
        if (buffer) vk.vkDestroyBuffer(device, buffer, nullptr);
    }

    VkDevice device;
    const VulkanDeviceFunctions& vk;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkImage image = VK_NULL_HANDLE;
    VkExtent2D image_extent{};
    std::uint64_t tile_count = 0;
    std::uint32_t memory_types = 0;
    std::map<std::uint64_t, TileMapping> mapped;
};

struct TileRegion {
    std::uint64_t first = 0;
    std::uint64_t count = 0;
};

enum class TileRangeMode { sequential, unmap, skip, reuse };

struct TileRange {
    TileRangeMode mode;
    std::uint64_t first;
    std::uint64_t count;
};

// Per device: what the queues share. Resources are registered by their stable ID (the COM resource pointer).
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4324)  // intentional alignas(64) cache-line padding to prevent false sharing
#endif
struct GraphicsIndirectCapabilities {
    bool first_instance = false;
    bool multiple = false;
    bool count_buffer = false;
    std::uint32_t max_count = 1;
    bool robust_index_view = false;
    bool require_clamp = false;
};

class VulkanContext {
public:
    static constexpr std::uint64_t kRetireWaitSliceNs = 100'000'000;

    // mesh_queries: VK_EXT_mesh_shader's meshShaderQueries feature is enabled on `device`, so a mesh segment's
    // clipping invocations can come from a mesh-primitives query (D3D-MESH-CLIPPING-COUNTER). Set by the device
    // that created the context.
    bool mesh_queries = false;
    // conditional_rendering: VK_EXT_conditional_rendering is enabled on `device` (SetPredication).
    // ray_tracing: VK_KHR_acceleration_structure + VK_KHR_ray_query are enabled on `device`.
    // xdxr: acceleration-structure builds write XDXR BVH4 records (xdxr_build.h) instead; needs bufferDeviceAddress.
    VulkanContext(PFN_vkGetDeviceProcAddr get, VkDevice device, bool separate_depth_stencil_layouts,
                  bool conditional_rendering = false, bool ray_tracing = false, bool xdxr = false,
                  GraphicsIndirectCapabilities indirect = {})
        : device_(device), tracker_(separate_depth_stencil_layouts), ray_tracing_(ray_tracing) {
        loaded_ = vk.load(get, device);
        // VK_EXT_transform_feedback: enabled on the device when its indexed-query entry points resolved (D3D-302).
        vk.transform_feedback = vk.vkCmdBeginQueryIndexedEXT && vk.vkCmdEndQueryIndexedEXT;
        if (loaded_ && gpu_labels_.initialize(device, get)) {
            replay_.gpu_label = [this](VkCommandBuffer cb, const label::Write& write) { gpu_labels_.record(cb, write); };
        }
        replay_.pipeline_barrier = vk.vkCmdPipelineBarrier;
        replay_.clear_color_image = vk.vkCmdClearColorImage;
        replay_.clear_depth_stencil_image = vk.vkCmdClearDepthStencilImage;
        replay_.pipeline_barrier2 = vk.vkCmdPipelineBarrier2;
        replay_.begin_rendering = vk.vkCmdBeginRendering;
        replay_.end_rendering = vk.vkCmdEndRendering;
        replay_.clear_attachments = vk.vkCmdClearAttachments;
        replay_.copy_buffer = vk.vkCmdCopyBuffer;
        replay_.copy_buffer_to_image = vk.vkCmdCopyBufferToImage;
        replay_.copy_image_to_buffer = vk.vkCmdCopyImageToBuffer;
        replay_.copy_image = vk.vkCmdCopyImage;
        replay_.blit_image = vk.vkCmdBlitImage;
        replay_.resolve_image = vk.vkCmdResolveImage;
        replay_.fill_buffer = vk.vkCmdFillBuffer;
        replay_.update_buffer = vk.vkCmdUpdateBuffer;
        replay_.draw_indirect = vk.vkCmdDrawIndirect;
        replay_.draw_indexed_indirect = vk.vkCmdDrawIndexedIndirect;
        replay_.draw_indirect_first_instance = indirect.first_instance;
        replay_.multi_draw_indirect = indirect.multiple;
        replay_.draw_indirect_count_enabled = indirect.count_buffer;
        replay_.max_draw_indirect_count = indirect.max_count;
        replay_.require_indirect_clamp = indirect.require_clamp;
        replay_.draw_indirect_count = vk.vkCmdDrawIndirectCount;
        replay_.draw_indexed_indirect_count = vk.vkCmdDrawIndexedIndirectCount;
        replay_.dispatch_indirect = vk.vkCmdDispatchIndirect;
        if (conditional_rendering) {  // a driver may hand out these pointers even when the extension is off
            replay_.begin_conditional_rendering = vk.vkCmdBeginConditionalRenderingEXT;
            replay_.end_conditional_rendering = vk.vkCmdEndConditionalRenderingEXT;
        }
        replay_.reset_query_pool = vk.vkCmdResetQueryPool;
        replay_.begin_query = vk.vkCmdBeginQuery;
        replay_.end_query = vk.vkCmdEndQuery;
        // VK_EXT_transform_feedback (D3D-302), loaded when the device enabled it: indexed SO_STATISTICS queries.
        if (vk.transform_feedback) {
            replay_.begin_query_indexed = vk.vkCmdBeginQueryIndexedEXT;
            replay_.end_query_indexed = vk.vkCmdEndQueryIndexedEXT;
        }
        replay_.write_timestamp = vk.vkCmdWriteTimestamp;
        replay_.copy_query_pool_results = vk.vkCmdCopyQueryPoolResults;
        replay_.bind_pipeline = vk.vkCmdBindPipeline;
        {
            // D3D-324: LEONIDA_DRAW_SCOPES=0 gives every draw its own rendering scope again (to compare). Read here,
            // once per context, not through a function-local static on a title thread.
#ifdef _WIN32
            char value[4] = {};
            std::size_t length = 0;
            replay_.merge_draw_scopes =
                !(getenv_s(&length, value, sizeof(value), "LEONIDA_DRAW_SCOPES") == 0 && length == 2 && value[0] == '0');
#else
            const char* value = std::getenv("LEONIDA_DRAW_SCOPES");
            replay_.merge_draw_scopes = !(value && value[0] == '0' && value[1] == '\0');
#endif
        }
        replay_.push_constants = vk.vkCmdPushConstants;
        replay_.bind_descriptor_sets = vk.vkCmdBindDescriptorSets;
        replay_.dispatch = vk.vkCmdDispatch;
        replay_.compute_pipeline = [this](std::uintptr_t id) -> const ComputePipeline* {
            const auto it = pipelines_.find(id);
            return it == pipelines_.end() ? nullptr : &it->second;
        };
        replay_.bind_vertex_buffers2 = vk.vkCmdBindVertexBuffers2;
        replay_.bind_index_buffer = vk.vkCmdBindIndexBuffer;
        if (indirect.robust_index_view)
            replay_.bind_robust_index_view = reinterpret_cast<decltype(replay_.bind_robust_index_view)>(
                get(device, "vkCmdBindIndexBuffer2KHR"));
        replay_.set_viewport = vk.vkCmdSetViewport;
        replay_.set_scissor = vk.vkCmdSetScissor;
        replay_.set_viewport_with_count = vk.vkCmdSetViewportWithCount;
        replay_.set_scissor_with_count = vk.vkCmdSetScissorWithCount;
        replay_.set_primitive_topology = vk.vkCmdSetPrimitiveTopology;
        replay_.set_primitive_restart_enable = vk.vkCmdSetPrimitiveRestartEnable;
        replay_.set_stencil_reference = vk.vkCmdSetStencilReference;
        replay_.set_blend_constants = vk.vkCmdSetBlendConstants;
        replay_.draw = vk.vkCmdDraw;
        replay_.draw_mesh_tasks = vk.vkCmdDrawMeshTasksEXT;  // null unless VK_EXT_mesh_shader is enabled
        replay_.draw_mesh_tasks_indirect = vk.vkCmdDrawMeshTasksIndirectEXT;
        replay_.draw_mesh_tasks_indirect_count = vk.vkCmdDrawMeshTasksIndirectCountEXT;
        replay_.draw_indexed = vk.vkCmdDrawIndexed;
        replay_.graphics_pipeline = [this](std::uintptr_t id) -> const GraphicsPipeline* {
            const auto it = graphics_pipelines_.find(id);
            return it == graphics_pipelines_.end() ? nullptr : &it->second;
        };
        // Called with resource_mutex_ held. A binary search of the buffers' VA ranges (buffer_va_index.h), kept in step
        // with resources_ by set_resource_locked and remove_resource.
        // A range that runs past its buffer is cut at the buffer's end (`size` is updated): D3D12 reads past a view's
        // buffer as out of bounds, which is no reason to drop the submission.
        replay_.buffer_at = [this](std::uint64_t va, std::uint64_t& size, VkBuffer& buffer, VkDeviceSize& offset) {
            BufferVaIndex<ResourceId>::Hit hit{};
            // prefers an aliasing buffer that holds all of it; a title address no buffer object covers is bound
            // through the title-memory page table's import of it (set_title_buffer_lookup)
            if (!buffer_vas_.find(va, hit, size)) return title_buffer_at_ && title_buffer_at_(va, size, buffer, offset);
            const ResourceBinding* b = resolve(hit.id);
            if (!b) return false;
            buffer = b->buffer;
            offset = b->buffer_offset + (va - b->va);
            size = std::min<std::uint64_t>(size, b->size - (va - b->va));
            return true;
        };
        // Host reads of small CPU-written inputs. Unlike buffer_at, the whole range must be in one CPU-writable
        // mapping: a partial read would see bytes the application has not written through the D3D12 API. The
        // binding's host pointer is null on DEFAULT and READBACK heaps, whose mapped bytes the GPU can write.
        replay_.host_at = [this](std::uint64_t va, std::uint64_t bytes, const std::uint8_t*& host) {
            BufferVaIndex<ResourceId>::Hit hit{};
            if (!buffer_vas_.find(va, hit, bytes)) return HostReadStatus::absent;
            const ResourceBinding* b = resolve(hit.id);
            if (!b) return HostReadStatus::absent;
            if (!b->host) return HostReadStatus::not_cpu_written;
            const std::uint64_t offset = va - b->va;
            if (offset > b->size || bytes > b->size - offset) return HostReadStatus::absent;
            host = b->host + offset;
            return HostReadStatus::ok;
        };
        if (xdxr && loaded_) {
            xdxr_ = std::make_unique<XdxrBuilder>(get, device);
            if (xdxr_->loaded()) {
                replay_.xdxr_check = [](const BuildAccelerationStructure& b) {
                    XdxrLayout layout;
                    // An update needs ALLOW_UPDATE (D3D12 requires it on the build and the update): a structure
                    // built without it may have paired leaves, which a refit would rewrite as wrong geometry.
                    return xdxr_inputs(b, nullptr, 0) && layout.compute(xdxr_elements(b), b.is_tlas, xdxr_box16(b)) &&
                           b.dst_va && b.scratch_va && (!b.is_tlas || !b.instance_count || b.instances_va) &&
                           (!b.update || (b.build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) != 0);
                };
                replay_.xdxr_record = [this](VkCommandBuffer cb, const BuildAccelerationStructure& b, const XdxrBlasTable&) {
                    // The instances' BLAS addresses are translated by the paged builder, so upstream's table is not used.
                    // Every address is a console VA. The builder translates them itself (XdxrBuilder::set_paged): an
                    // arena VA is heap_base + VA, a title VA (at or above 4 GiB: XboxBvhBuilder's builds, in the
                    // title's own memory) goes through the title page table, as the translated shaders read it.
                    // Before, heap_base + a title VA was a wild device address.
                    xdxr_->set_paged(replay_.heap_base);
                    XdxrLayout layout;
                    layout.compute(xdxr_elements(b), b.is_tlas, xdxr_box16(b));
                    map_title_range(b.dst_va, layout.size);
                    map_title_range(b.scratch_va, layout.scratch);
                    // PERFORM_UPDATE refits the source (the destination itself when no source is given, as D3D12
                    // allows for an in-place update); a full build otherwise.
                    const std::uint64_t from = b.update ? (b.src_va ? b.src_va : b.dst_va) : 0;
                    if (from && from != b.dst_va) map_title_range(from, layout.size);
                    if (b.is_tlas) {
                        map_title_range(b.instances_va, std::uint64_t(b.instance_count) * 64);
                        xdxr_->record_tlas(cb, layout, b.instances_va, b.dst_va, b.scratch_va, 0, from);
                    } else {
                        for (const auto& t : b.triangles) {
                            map_title_range(t.vertex_va, std::uint64_t(t.vertex_count) * t.vertex_stride);
                            map_title_range(t.index_va, std::uint64_t(t.index_count) * (t.index_type == 1 ? 2 : 4));
                            map_title_range(t.transform_va, 48);
                        }
                        std::vector<XdxrTriangles> geometries;
                        xdxr_inputs(b, &geometries, 0);
                        xdxr_->record_blas(cb, layout, geometries, b.dst_va, b.scratch_va, from, xdxr_pair(b));
                    }
                };
                replay_.xdxr_copy = [this](VkCommandBuffer cb, std::uint64_t src, std::uint64_t dst, std::uint64_t bytes) {
                    xdxr_->set_paged(replay_.heap_base);
                    map_title_range(src, bytes);
                    map_title_range(dst, bytes);
                    xdxr_->record_copy(cb, src, dst, bytes);
                };
                replay_.xdxr_postbuild = [this](VkCommandBuffer cb, std::uint64_t src, std::uint64_t dest) {
                    xdxr_->set_paged(replay_.heap_base);
                    map_title_range(src, XdxrLayout::header_size);
                    map_title_range(dest, 8);
                    xdxr_->record_postbuild(cb, src, dest);
                };
            } else {
                xdxr_.reset();
            }
        }
        if (!xdxr_ && ray_tracing_ && vk.vkCmdBuildAccelerationStructuresKHR) {
            replay_.build_acceleration_structures = vk.vkCmdBuildAccelerationStructuresKHR;
            replay_.copy_acceleration_structure = vk.vkCmdCopyAccelerationStructureKHR;
            replay_.write_acceleration_structures_properties = vk.vkCmdWriteAccelerationStructuresPropertiesKHR;
            replay_.resolve_acceleration_structure = [this](std::uint64_t va) {
                const auto it = acceleration_structures_.find(va);
                return it == acceleration_structures_.end() ? VK_NULL_HANDLE : it->second;
            };
        }
        replay_.image_view = [this](const ResourceBinding& r, const VkImageSubresourceRange& range) {
            return view(r, range);
        };
        if (vk.vkCreateRenderPass2 && vk.vkDestroyRenderPass && vk.vkCreateFramebuffer && vk.vkDestroyFramebuffer &&
            vk.vkCmdBeginRenderPass && vk.vkCmdEndRenderPass) {
            replay_.begin_render_pass = vk.vkCmdBeginRenderPass;
            replay_.end_render_pass = vk.vkCmdEndRenderPass;
            replay_.depth_resolve_framebuffer = [this](const ResourceBinding& source, VkImageView from,
                                                       VkImageLayout from_layout, const ResourceBinding& target,
                                                       VkImageView to, VkImageLayout to_layout,
                                                       VkResolveModeFlagBits mode, VkExtent2D extent,
                                                       VkRenderPass& pass) {
                return depth_resolve_framebuffer(source, from, from_layout, target, to, to_layout, mode, extent, pass);
            };
        }
        replay_.query_pool = [this](QueryHeapId id) -> QueryPool* {  // called with resource_mutex_ held
            const auto it = query_pools_.find(id);
            return it == query_pools_.end() ? nullptr : &it->second;
        };
        replay_.separate_depth_stencil_layouts = separate_depth_stencil_layouts;
        if (loaded_) {
            fence_dispatcher_ = std::make_unique<FenceWaitDispatcher>(device_, vk, device_lost_);
        }
    }

    ~VulkanContext() {
        stop_binder();  // every batch handed over is made (or its ticket signalled) before the queues go
        progress_watch_ = {};  // stop before queues, timelines or diagnostic state go away
        gpu_labels_.stop(device_);
        fence_dispatcher_.reset();
        deletion_queue_.drain(UINT64_MAX, device_, vk);
        if (binder_.semaphore) {
            if (vk.vkDeviceWaitIdle) vk.vkDeviceWaitIdle(device_);
            vk.vkDestroySemaphore(device_, binder_.semaphore, nullptr);
            if (binder_.chain) vk.vkDestroySemaphore(device_, binder_.chain, nullptr);
        }
        if (vk.vkDestroySemaphore) {
            for (const auto& [sem, state] : progress_waits_)
                if (state.destroy_when_done) vk.vkDestroySemaphore(device_, sem, nullptr);
        }
        for (const auto& [key, framebuffer] : resolve_framebuffers_)
            vk.vkDestroyFramebuffer(device_, framebuffer, nullptr);
        for (const auto& [key, pass] : resolve_passes_) vk.vkDestroyRenderPass(device_, pass, nullptr);
        for (const auto& [key, view] : views_) vk.vkDestroyImageView(device_, view, nullptr);
        for (const auto& [key, view] : plane_views_) vk.vkDestroyImageView(device_, view, nullptr);
    }

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool loaded() const { return loaded_; }

    void configure_device_fault(PFN_vkGetDeviceProcAddr get, DeviceFaultExtensions enabled) {
        device_fault_ = std::make_unique<DeviceFaultDiagnostics>(get, device_, enabled);
        if (fence_dispatcher_) fence_dispatcher_->set_device_fault(device_fault_.get());
        if (loaded_) progress_watch_ = std::jthread([this](std::stop_token stop) { watch_queue_progress(stop); });
    }

    void add_diagnostic_queue(VkQueue queue) {
        if (device_fault_) device_fault_->add_queue(queue);
    }

    DeviceFaultDiagnostics* device_fault() const { return device_fault_.get(); }

    VkResult observe_result(VkResult result, const char* operation) const noexcept {
        return DeviceFaultDiagnostics::observe(device_fault_.get(), result, operation);
    }
    void configure_predication(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceMemoryProperties& memory,
                               VkDeviceSize max_range, std::uint32_t family) {
        std::unique_lock lock(resource_mutex_);
        predicate_family_ = family;
        predicate_reducer_ = std::make_unique<PredicateReducer>(get, device_, memory, max_range);
    }
    // Configure before submitting work, only when storageBuffer8BitAccess is enabled on the device.
    void configure_predicated_copy(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits) {
        std::unique_lock lock(resource_mutex_);
        predicated_copier_ = std::make_unique<PredicatedBufferCopier>(get, device_, limits);
    }
    // Counts for DISPATCH-only ExecuteIndirect are read by a helper compute shader from GPU memory.
    // Requires Vulkan bufferDeviceAddress, already enabled for the D3D12 address arena.
    void configure_counted_dispatch(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceMemoryProperties& memory,
                                    const VkPhysicalDeviceLimits& limits) {
        std::unique_lock lock(resource_mutex_);
        counted_dispatch_ = std::make_unique<IndirectDispatchCounter>(get, device_, memory, limits);
    }
    // The XDXR builder splits its dispatches at limits.maxComputeWorkGroupCount[0] (65,535, the least Vulkan allows,
    // until this is called). Configure before submitting work.
    void configure_xdxr_dispatch(const VkPhysicalDeviceLimits& limits) {
        std::unique_lock lock(resource_mutex_);
        if (xdxr_) xdxr_->set_max_groups(limits.maxComputeWorkGroupCount[0]);
    }
    // Indexed quad lists are drawn from triangle indices a compute pass writes into the copy scratch
    // (quad_index_expand.h). Call only with bufferDeviceAddress enabled and after an enable_copy_scratch whose chunks
    // have INDEX_BUFFER and SHADER_DEVICE_ADDRESS usage on memory allocated with a device address.
    void configure_quad_indices(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits) {
        std::unique_lock lock(resource_mutex_);
        quad_indices_ = std::make_unique<QuadIndexExpander>(get, device_, limits);

    }

    // Every ExecuteIndirect's arguments are clamped on the GPU into scratch before the indirect command reads them
    // (vulkan_indirect_clamp.h), so a stale or garbage record from a skipped compute pass cannot reset the display
    // driver. Needs bufferDeviceAddress and a compute-capable queue; `allocate` gives the record scratch
    // device-local memory, and must set VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT. LEONIDA_INDIRECT_CLAMP=0 turns the
    // pass off at replay time.
    void enable_indirect_clamp(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceMemoryProperties& memory,
                               const VkPhysicalDeviceLimits& limits, ScratchBuffers::Allocate allocate,
                               std::vector<std::uint32_t> families = {}) {
        std::unique_lock lock(resource_mutex_);
        replay_.require_indirect_clamp = indirect_clamp_enabled();
        if (!replay_.require_indirect_clamp) return;
        indirect_clamp_ = std::make_unique<IndirectArgumentClamp>(get, device_, memory, limits, std::move(allocate),
                                                                 device_fault_.get(), std::move(families));
        if (!indirect_clamp_->loaded()) {
            indirect_clamp_.reset();
            return;
        }
        // The preparation callback needs this submission's counter-slot batch, so it is set per ExecuteCommandLists
        // (execute()); recording the pass has no per-submission state.
        replay_.emit_indirect_clamp = [this](VkCommandBuffer command_buffer, const ClampedIndirect& job) {
            indirect_clamp_->emit(command_buffer, job);
        };
    }
    VkDevice device() const { return device_; }

    // Debug markers become labels from here on; call before any queue or list uses the context. Incomplete sets
    // are ignored, so markers stay no-ops.
    void enable_debug_labels(const DebugLabelFunctions& labels) {
        if (!labels.complete()) return;
        labels_ = labels;
        replay_.insert_label = labels.cmd_insert;
        replay_.begin_label = labels.cmd_begin;
        replay_.end_label = labels.cmd_end;
    }
    bool debug_labels() const { return labels_.complete(); }

    // Registers a resource in its D3D12 initial state (a legacy state; images get the matching layout).
    // The image must already be in that layout, e.g. transitioned from UNDEFINED at creation.
    void add_resource(ResourceId id, const ResourceBinding& binding) {
        std::unique_lock lock(resource_mutex_);
        set_resource_locked(id, binding);
    }
    // Held around every vkQueueBindSparse call (bind_title_pages) and every host-memory import (d3d12_vulkan_device.cpp,
    // import_host): never both at once. With a bind on the bind thread and an import on a recording thread inside the
    // NVIDIA driver together, the import spun at one instruction (nvoglv64+0xa6a4ce) for over 10 s while the bind thread
    // and a staging submit waited on a driver lock, and every queue stalled into the title's watchdog (runs 20, 22:
    // stacks_1). Apart, an import costs 0.05-1 ms (D:\GTA6\analysis\tools\heapimport); in the game each call took 15-30 ms,
    // about one 16-image bind call.
    std::mutex& driver_heavy_mutex() { return driver_heavy_mutex_; }

    // Staging buffers a batched texture-sync copy reads (d3d12_vulkan_device.cpp, flush_staged_batch): resolvable while
    // their list is recorded, without resource_mutex_. Registering them as resources took it exclusively, which waited
    // for every queue's recording and submission (they hold it shared): 140-650 ms per sync, with the queue lock held
    // (run 14, 2026-11-25). Plain buffers with no VA, tracking or layout, so the registry's other duties do not apply.
    // A binding stays at one address until removed (node-based map), as resolve's callers expect while recording.
    void add_transient_binding(ResourceId id, const ResourceBinding& binding) {
        std::lock_guard lock(transient_mutex_);
        transient_bindings_.insert_or_assign(id, binding);
        transient_count_.store(transient_bindings_.size(), std::memory_order_release);
    }
    void remove_transient_binding(ResourceId id) {
        std::lock_guard lock(transient_mutex_);
        transient_bindings_.erase(id);
        transient_count_.store(transient_bindings_.size(), std::memory_order_release);
    }
    // Whether `va` is in a buffer this context knows by VA (the index buffer_at searches).
    bool knows_buffer_va(std::uint64_t va, std::uint64_t bytes = 1) {
        std::shared_lock lock(resource_mutex_);
        return buffer_vas_.covers(va, bytes);
    }
    void add_resource(ResourceId id, const ResourceBinding& binding, VkImageLayout current_layout) {
        std::unique_lock rlock(resource_mutex_);
        set_resource_locked(id, binding);
        if (binding.image) {
            std::lock_guard tlock(tracker_mutex_);
            tracker_.track(id, binding, current_layout);
        }
    }

    // A shader can access a storage image or an initially unbound sparse image before any command names it.
    // The first shader submission moves it to GENERAL, the descriptor's layout, ahead of its lists. An image the
    // tracker has in a defined layout everywhere was initialized already: nothing to do, and no other queue has to
    // wait for a submission (a title creates UAVs over the same textures every frame, in ring heaps). Call with no
    // lock held that comes after tracker_mutex_ (descriptor_mutex, submit_mutex_).
    void note_shader_image(ResourceId id) {
        {
            std::lock_guard tlock(tracker_mutex_);
            if (tracker_.defined(id)) return;
        }
        std::lock_guard lock(storage_mutex_);
        storage_pending_.insert(id);
    }

    // A device-local buffer D3D12 says starts zeroed (D3D-71): the host cannot write it, so the next submission on
    // any queue fills it with zeros ahead of its lists. That submission publishes the same initialisation semaphore as
    // the storage images above, so a list on another queue that uses the buffer first waits for the fill.
    void zero_on_next_submit(ResourceId id, VkBuffer buffer) {
        std::lock_guard lock(storage_mutex_);
        zero_pending_.push_back({id, buffer});
    }

    void add_sparse_resource(ResourceId id, const ResourceBinding& binding, std::shared_ptr<SparseResource> sparse) {
        std::unique_lock lock(resource_mutex_);
        if (!binding.image) {
            auto [it, inserted] = sparse_resources_.try_emplace(id, std::move(sparse));
            try {
                set_resource_locked(id, binding);
            } catch (...) {
                if (inserted) sparse_resources_.erase(it);
                throw;
            }
            return;
        }
        set_resource_locked(id, binding);
        sparse_resources_.emplace(id, std::move(sparse));
        if (binding.image) {
            std::lock_guard tlock(tracker_mutex_);
            tracker_.track(id, binding, VK_IMAGE_LAYOUT_UNDEFINED);
        }
    }

    void remove_resource(ResourceId id) {
        VkImage img = VK_NULL_HANDLE;
        std::shared_ptr<SparseResource> sparse_resource;
        {
            std::unique_lock rlock(resource_mutex_);
            const auto it = resources_.find(id);
            if (it == resources_.end()) return;
            img = it->second.image;
            unindex_buffer(id, it->second);
            resources_.erase(it);
            if (auto sparse = sparse_resources_.find(id); sparse != sparse_resources_.end()) {
                sparse_resource = std::move(sparse->second);
                sparse_resources_.erase(sparse);
            }
        }
        {
            std::lock_guard lock(storage_mutex_);  // a buffer destroyed before any submission is not filled
            storage_pending_.erase(id);
            std::erase_if(zero_pending_, [id](const PendingZero& zero) { return zero.id == id; });
        }
        if (img) {
            std::lock_guard vlock(views_mutex_);
            for (auto f = resolve_framebuffers_.begin(); f != resolve_framebuffers_.end();) {
                if (std::get<0>(f->first) == img || std::get<1>(f->first) == img) {
                    defer_destroy(DeferredDeletionQueue::Kind::framebuffer, f->second);
                    f = resolve_framebuffers_.erase(f);
                } else {
                    ++f;
                }
            }
            for (auto v = views_.begin(); v != views_.end();) {
                if (std::get<0>(v->first) == img) {
                    defer_destroy(DeferredDeletionQueue::Kind::image_view, v->second);
                    v = views_.erase(v);
                } else {
                    ++v;
                }
            }
            for (auto v = plane_views_.lower_bound({img, VkFormat(0)}); v != plane_views_.end() && v->first.first == img;) {
                defer_destroy(DeferredDeletionQueue::Kind::image_view, v->second);
                v = plane_views_.erase(v);
            }
        }
        if (sparse_resource)
            defer_destroy(DeferredDeletionQueue::Kind::sparse_resource,
                          new std::shared_ptr<SparseResource>(std::move(sparse_resource)));
        std::lock_guard tlock(tracker_mutex_);
        tracker_.forget(id);
    }

    // A query heap, by its stable ID (the COM query heap pointer). The pool stays the caller's to destroy, after
    // remove_query_pool.
    void add_query_pool(QueryHeapId id, const QueryPool& pool) {
        std::unique_lock lock(resource_mutex_);
        query_pools_[id] = pool;
    }
    void remove_query_pool(QueryHeapId id) {
        // Lock hierarchy: resource_mutex_ -> tracker_mutex_. Holding tracker_mutex_ before erasing keeps the
        // QueryPool node alive until any in-flight execute() that unlocked resource_mutex_ finishes QueryReplay::commit().
        std::unique_lock lock(resource_mutex_);
        std::lock_guard tlock(tracker_mutex_);
        query_pools_.erase(id);
    }
    // The heap's d3d12_query_heap type, or ~0u for a heap that is not registered.
    std::uint32_t query_heap_type(QueryHeapId id) {
        std::shared_lock lock(resource_mutex_);   // query_pools_ is guarded by resource_mutex_, like add/remove
        const auto it = query_pools_.find(id);
        return it == query_pools_.end() ? ~0u : it->second.heap_type;
    }

    // VkPhysicalDeviceLimits::timestampPeriod (nanoseconds per tick), or 0 when the queue writes no timestamps
    // (timestampValidBits 0). Set by the device before any queue uses it.
    double timestamp_period() const { return timestamp_period_; }
    void set_timestamp_period(double period) { timestamp_period_ = period; }
    // True where a timestamp resolve must be ordered explicitly after the timestamps it copies (MoltenVK on Apple
    // GPUs, docs/d3d12_vulkan.md section 4.6). Set by the device before any queue uses it.
    bool order_timestamp_copies() const { return order_timestamp_copies_; }
    void set_order_timestamp_copies(bool order) { order_timestamp_copies_ = order; }

    // The device address of the GPU address arena: console VAs are offsets from it (docs section 3.3).
    // execute() copies replay_ under a shared resource_mutex_, so writers hold it uniquely.
    // The indirect-draw features enabled on the device (VulkanDeviceInfo::multi_draw_indirect / draw_indirect_count).
    void set_indirect_features(bool multi_draw, bool draw_count, std::uint32_t max_draw_count = UINT32_MAX) {
        replay_.multi_draw_indirect = multi_draw;
        replay_.max_draw_indirect_count = max_draw_count;
        if (!draw_count) {
            replay_.draw_indirect_count = replay_.draw_indexed_indirect_count = nullptr;
            replay_.draw_mesh_tasks_indirect_count = nullptr;
        }
    }

    // `heap_size`: the arena's bytes, below which a VA a GPU pass finds in memory (a TLAS instance's BLAS) is an arena
    // VA; 0 when unknown (such a VA is then taken as one whatever its value).
    void set_heap_base(std::uint64_t heap_base, std::uint64_t heap_size = 0) {
        std::unique_lock lock(resource_mutex_);
        replay_.heap_base = heap_base;
        replay_.heap_size = heap_size;
    }

    // Zeroed memory for unbound vertex and index buffers to read (ReplayFunctions::zero_buffer).
    void set_zero_page(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size) {
        std::unique_lock lock(resource_mutex_);
        replay_.zero_buffer = buffer;
        replay_.zero_offset = offset;
        replay_.zero_size = size;
    }

    void set_depth_range_unrestricted(bool enabled) {
        std::unique_lock lock(resource_mutex_);
        replay_.depth_range_unrestricted = enabled;
    }

    struct GeometryUpload {
        std::uint64_t source{}, size{};
        VkBuffer buffer{};
        VkDeviceSize offset{};
        std::shared_ptr<void> owner;
    };
    using GeometryProvider = std::function<bool(std::uint64_t,std::uint64_t,GeometryUpload&)>;
    void set_geometry_provider(GeometryProvider provider) {
        std::unique_lock lock(resource_mutex_);
        geometry_provider_=std::move(provider);
    }

    // LOCAL: the copy-in fallback for vertex and index views in title memory (ReplayFunctions::title_memory).
    void set_title_memory(std::function<bool(std::uint64_t, std::uint64_t, VkBuffer&, VkDeviceSize&)> copy) {
        std::unique_lock lock(resource_mutex_);
        replay_.title_memory = std::move(copy);
    }

    // Pre-expanded 32-bit triangle indices for non-indexed quad lists (ReplayFunctions::quad_indices).
    void set_quad_indices(VkBuffer buffer, VkDeviceSize offset, std::uint32_t count) {
        std::unique_lock lock(resource_mutex_);
        replay_.quad_indices = buffer;
        replay_.quad_indices_offset = offset;
        replay_.quad_index_count = count;
    }

    // Physical-device COLOR_ATTACHMENT format check for reinterpreted UAV clears.
    void set_color_attachment_format(std::function<bool(VkFormat)> check) {
        std::unique_lock lock(resource_mutex_);
        replay_.color_attachment_format = std::move(check);
    }
    std::uint64_t heap_base() {
        std::shared_lock lock(resource_mutex_);
        return replay_.heap_base;
    }
    // Where buffer_at looks for a title address (at or above 4 GiB) that no buffer object covers: the device's
    // title-memory page table imports. Called with resource_mutex_ held; it must not take it.
    using TitleBufferLookup = std::function<bool(std::uint64_t, std::uint64_t&, VkBuffer&, VkDeviceSize&)>;
    void set_title_buffer_lookup(TitleBufferLookup lookup) {
        std::unique_lock lock(resource_mutex_);
        title_buffer_at_ = std::move(lookup);
    }
    // ReplayFunctions::map_title: called while commands are recorded (resource_mutex_ held); must not take it.
    void set_title_mapper(std::function<void(std::uint64_t)> mapper) {
        std::unique_lock lock(resource_mutex_);
        replay_.map_title = std::move(mapper);
    }
    // Page-table entries for a whole title range (va, bytes) before a command that reads or writes it through the
    // table (the XDXR builder's paged addresses). Called while commands are recorded, like map_title.
    void set_title_range_mapper(std::function<void(std::uint64_t, std::uint64_t)> mapper) {
        std::unique_lock lock(resource_mutex_);
        title_range_mapper_ = std::move(mapper);
    }
    void map_title_range(std::uint64_t va, std::uint64_t bytes) {
        if (va >> 32 && bytes && title_range_mapper_) title_range_mapper_(va, bytes);
    }

    // Pipelines are registered by an opaque id (the PSO object) for as long as the PSO lives.
    // Guarded by pipeline_mutex_ (separate from resource_mutex_) so shader compilation on one queue does not block
    // resource registration or destruction on the render thread.
    void add_pipeline(std::uintptr_t id, const ComputePipeline& pipeline, const char* shader = nullptr) {
        std::unique_lock lock(pipeline_mutex_);
        pipelines_[id] = pipeline;
        if (shader && device_fault_) device_fault_->pipeline_names[id] = {shader};
    }
    void remove_pipeline(std::uintptr_t id) {
        std::unique_lock lock(pipeline_mutex_);
        pipelines_.erase(id);
        graphics_pipelines_.erase(id);
        if (device_fault_) device_fault_->pipeline_names.erase(id);
    }
    void add_graphics_pipeline(std::uintptr_t id, const GraphicsPipeline& pipeline,
                               const CheckpointShaders* shaders = nullptr) {
        std::unique_lock lock(pipeline_mutex_);
        graphics_pipelines_[id] = pipeline;
        if (shaders && device_fault_) device_fault_->pipeline_names[id] = *shaders;
    }
    // Workers for the pipelines a submission needs and that are not built yet. Null (the default): compiled serially on
    // the submitting thread once resource_mutex_ is released. Replacing it waits for submissions that are compiling, and
    // ends the workers it had.
    void set_pipeline_workers(std::unique_ptr<PipelineWorkers> workers) {
        std::unique_lock lock(pipeline_mutex_);
        pipeline_workers_.swap(workers);
    }
    // For the large pipelines of a submission: the memory they may use while they compile together
    // (PipelineCompiles::build), and what is kept while they do (LargeModuleGate::Demand). None (the default): they compile
    // one after another.
    void set_large_compiles(std::function<std::uint64_t()> budget, std::function<std::shared_ptr<void>()> demand) {
        std::unique_lock lock(pipeline_mutex_);
        large_compile_budget_ = std::move(budget);
        large_compile_demand_ = std::move(demand);
    }
    PipelineWorkers::Stats pipeline_worker_stats() {
        std::shared_lock lock(pipeline_mutex_);
        return pipeline_workers_ ? pipeline_workers_->stats() : PipelineWorkers::Stats{};
    }
    // Configure before queue creation, when the selected queue family supports compute.
    void set_occlusion_resolve_memory(const VkPhysicalDeviceMemoryProperties& memory) {
        occlusion_resolve_memory_ = memory;
    }

    // Device-lost diagnostics (vulkan_breadcrumbs.h): per-list breadcrumbs from now on; call before any queue uses
    // the context. The caller decides (LEONIDA_DRED) and says which extensions are enabled: a device function of an
    // extension that is not enabled must not be called, even when the loader returned one. True with GPU
    // breadcrumbs (VK_AMD_buffer_marker); false when the ring keeps only what the CPU knows, because the marker
    // extension is not enabled or no host-visible memory fits.
    // `per_op` (BUILD-1655): a progress marker after every D3D12 operation; false keeps the two around each list
    // (for measuring what the per-op markers cost).
    bool configure_breadcrumbs(const VkPhysicalDeviceMemoryProperties& memory, bool buffer_marker_enabled,
                               bool device_fault_enabled, bool per_op = true) {
        device_fault_enabled_ = device_fault_enabled;
        BreadcrumbFunctions fn;
        fn.create_buffer = vk.vkCreateBuffer;
        fn.destroy_buffer = vk.vkDestroyBuffer;
        fn.requirements = vk.vkGetBufferMemoryRequirements;
        fn.allocate = vk.vkAllocateMemory;
        fn.free = vk.vkFreeMemory;
        fn.bind = vk.vkBindBufferMemory;
        fn.map = vk.vkMapMemory;
        fn.unmap = vk.vkUnmapMemory;
        fn.write_marker = buffer_marker_enabled ? vk.vkCmdWriteBufferMarkerAMD : nullptr;
        auto ring = std::make_unique<BreadcrumbRing>();
        const bool gpu = fn.write_marker && ring->initialize(device_, memory, fn, per_op);
        if (!gpu) {
            ring = std::make_unique<BreadcrumbRing>();  // frees whatever a failed initialize() made
            ring->initialize_cpu_only();
        }
        breadcrumbs_ = std::move(ring);
        detail_stall::breadcrumbs.store(breadcrumbs_.get(), std::memory_order_release);
        return gpu;
    }
    // Where the device-lost report goes (stderr); for tests.
    void set_report_output(std::FILE* out) { report_out_ = out ? out : stderr; }
    BreadcrumbRing* breadcrumbs() const { return breadcrumbs_.get(); }

    // VK_EXT_device_fault after a loss: the faulting addresses and the vendor's description. False when the
    // extension is not enabled, a query fails, or the driver has nothing to say. Vendor binary data is not asked for.
    bool device_fault(std::vector<VkDeviceFaultAddressInfoEXT>& addresses, std::string& description) const {
        addresses.clear();
        description.clear();
        if (!device_fault_enabled_ || !vk.vkGetDeviceFaultInfoEXT) return false;
        VkDeviceFaultCountsEXT counts{VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
        if (vk.vkGetDeviceFaultInfoEXT(device_, &counts, nullptr) != VK_SUCCESS) return false;
        addresses.resize(counts.addressInfoCount);
        std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
        counts.vendorBinarySize = 0;  // no vendor dump
        VkDeviceFaultInfoEXT info{VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT};
        info.pAddressInfos = addresses.empty() ? nullptr : addresses.data();
        info.pVendorInfos = vendor.empty() ? nullptr : vendor.data();
        const VkResult result = vk.vkGetDeviceFaultInfoEXT(device_, &counts, &info);
        if (result != VK_SUCCESS && result != VK_INCOMPLETE) return false;
        addresses.resize(counts.addressInfoCount);
        description.assign(info.description, strnlen(info.description, VK_MAX_DESCRIPTION_SIZE));
        return !addresses.empty() || !description.empty();  // else the driver has nothing to say
    }

    struct HostTextureStats { std::uint64_t copies = 0, waits = 0; };
    HostTextureStats host_texture_stats() {
        std::lock_guard lock(submit_mutex_);
        return host_texture_stats_;
    }

    // CPU writes to a title texture have no application fence. Prevent another queue from
    // claiming a new layout or submitting work between the completion check and the host copy.
    // An unsatisfied console Wait lives in QueueHolds, outside the Vulkan queues.
    template<class Copy>
    VkResult write_host_texture(ResourceId id, std::uint32_t subresource, Copy&& copy) {
        std::lock_guard layout_lock(tracker_mutex_);
        std::lock_guard queue_lock(submit_mutex_);
        if (tracker_.layout(id, subresource) != VK_IMAGE_LAYOUT_GENERAL) return VK_NOT_READY;
        if (image_work_pending_) {
            ++host_texture_stats_.waits;
            const auto result = observe_result(vk.vkDeviceWaitIdle(device_), "vkDeviceWaitIdle (title texture upload)");
            if (result != VK_SUCCESS) return result;
            image_work_pending_ = false;
        }
        if (!copy()) return VK_ERROR_UNKNOWN;
        ++host_texture_stats_.copies;
        return VK_SUCCESS;
    }

    // A complete CPU-side replacement of these color subresources. Serialize with replay and
    // present while recording the old layouts, finishing prior reads and publishing the copy.
    template<class Copy>
    VkResult write_staged_title_image(ResourceId id, const VkImageSubresourceRange& range, Copy&& copy) {
        std::lock_guard layout_lock(tracker_mutex_);
        std::lock_guard queue_lock(submit_mutex_);
        LayoutPieces pieces;
        if (!tracker_.pieces(id, range, pieces) || pieces.empty()) return VK_ERROR_UNKNOWN;
        if (image_work_pending_) {
            const auto r = observe_result(vk.vkDeviceWaitIdle(device_), "vkDeviceWaitIdle (title staging)");
            if (r != VK_SUCCESS) return r;
            image_work_pending_ = false;
        }
        const auto r = observe_result(copy(replay_, pieces), "title staging upload");
        if (r == VK_SUCCESS) {
            for (const auto& p : pieces)
                if (p.layout == VK_IMAGE_LAYOUT_UNDEFINED) tracker_.set(id, p.range, VK_IMAGE_LAYOUT_GENERAL);
        }
        return r;
    }

    // Present and probe submissions have their own fences, outside the queue retirement timelines.
    void note_image_work_locked() { image_work_pending_ = true; }

    // For tests and traces.
    VkImageLayout layout(ResourceId id, std::uint32_t subresource) {
        std::lock_guard lock(tracker_mutex_);
        return tracker_.layout(id, subresource);
    }

    bool has_resource(ResourceId id) {
        std::shared_lock lock(resource_mutex_);
        return resources_.find(id) != resources_.end();
    }

    // The registered buffer whose VA range holds `va`, preferring one that holds [va, va + bytes) as buffer_at does: its
    // ID, first VA and size in bytes. False when none does, or when `acquire(id)` returns false. `acquire` runs with
    // the registry locked, so the resource cannot finish being removed under it: it takes the caller's reference
    // (Resource's destructor removes its entry first, which waits for this lock). Used where a console call names a
    // buffer by VA only (Xbox device slot 70).
    template<class Acquire>
    bool buffer_resource_at(std::uint64_t va, std::uint64_t bytes, ResourceId& id, std::uint64_t& start,
                            std::uint64_t& size, Acquire&& acquire) {
        std::shared_lock lock(resource_mutex_);
        BufferVaIndex<ResourceId>::Hit hit{};
        if (!buffer_vas_.find(va, hit, bytes) || !acquire(hit.id)) return false;
        id = hit.id;
        start = hit.start;
        size = hit.size;
        return true;
    }

    bool ray_tracing_supported() const { return ray_tracing_ && vk.vkCmdBuildAccelerationStructuresKHR != nullptr; }
    bool xdxr_acceleration_structures() const { return xdxr_ != nullptr; }

    // Whether a build's inner nodes are BOX16 (64 bytes, half boxes rounded outward) instead of BOX32: bottom-level
    // builds with MINIMIZE_MEMORY (VK LOW_MEMORY). A top-level build keeps BOX32: it is in world space, where half
    // precision would make boxes metres too large a few kilometres from the origin. The flags of an update must match
    // its build's (D3D12), so prebuild info, the build and its refits agree. WARNING: an update that breaks that rule
    // (MINIMIZE_MEMORY on one and not the other) is refitted with the wrong layout and writes past the structure; it
    // cannot be detected when the command is recorded.
    static bool xdxr_box16(const BuildAccelerationStructure& b) {
        return !b.is_tlas && (b.build_flags & VK_BUILD_ACCELERATION_STRUCTURE_LOW_MEMORY_BIT_KHR) != 0;
    }
    // Whether a bottom-level build pairs sibling triangles into TRI0/TRI1 leaves: not when the structure may be
    // updated (ALLOW_UPDATE), since a refit rewrites the leaves one triangle each and would leave every TRI1 child
    // reading wrong vertices. xdxr_check refuses an update without ALLOW_UPDATE for that reason. Pairing changes
    // neither the layout nor the tree, so prebuild info does not depend on it.
    static bool xdxr_pair(const BuildAccelerationStructure& b) {
        return !b.is_tlas && !b.update && (b.build_flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) == 0;
    }
    // Triangles (BLAS) or instances (TLAS) a build has: what XdxrLayout is computed from, for the build and for
    // GetRaytracingAccelerationStructurePrebuildInfo.
    static std::uint32_t xdxr_elements(const BuildAccelerationStructure& b) {
        if (b.is_tlas) return b.instance_count;
        std::uint64_t n = 0;
        for (const auto& t : b.triangles) n += (t.index_type ? t.index_count : t.vertex_count) / 3;
        return n > 0xffffffffu ? 0xffffffffu : std::uint32_t(n);
    }
    // The geometries as the builder reads them, with `base` added to their VAs; false for an input it cannot read
    // (a vertex format other than R32G32B32_FLOAT or R32G32_FLOAT, an index type other than none/R16/R32).
    static bool xdxr_inputs(const BuildAccelerationStructure& b, std::vector<XdxrTriangles>* out, std::uint64_t base) {
        if (b.is_tlas) return true;
        for (const auto& t : b.triangles) {
            if (t.vertex_format != VK_FORMAT_R32G32B32_SFLOAT && t.vertex_format != VK_FORMAT_R32G32_SFLOAT) return false;
            if (t.index_type > 2) return false;
            const std::uint32_t count = (t.index_type ? t.index_count : t.vertex_count) / 3;
            if (count && !t.vertex_va) return false;
            if (count && t.index_type && !t.index_va) return false;
            if (!out) continue;
            XdxrTriangles g;
            g.vertices = t.vertex_va ? base + t.vertex_va : 0;
            g.vertex_stride = t.vertex_stride;
            g.xy_only = t.vertex_format == VK_FORMAT_R32G32_SFLOAT;
            g.indices = t.index_type && t.index_va ? base + t.index_va : 0;
            g.index_type = t.index_type;
            g.triangle_count = count;
            g.transform = t.transform_va ? base + t.transform_va : 0;
            g.flags = t.flags | (t.opaque ? 1u : 0u);
            out->push_back(g);
        }
        return true;
    }

    void register_acceleration_structure(std::uint64_t va, VkAccelerationStructureKHR as) {
        if (!va || !as) return;
        std::unique_lock lock(resource_mutex_);
        acceleration_structures_[va] = as;
    }

    void unregister_acceleration_structure(std::uint64_t va) {
        if (!va) return;
        std::unique_lock lock(resource_mutex_);
        acceleration_structures_.erase(va);
    }

    VkAccelerationStructureKHR resolve_acceleration_structure(std::uint64_t va) {
        std::shared_lock lock(resource_mutex_);
        const auto it = acceleration_structures_.find(va);
        return it == acceleration_structures_.end() ? VK_NULL_HANDLE : it->second;
    }

    // `prepare` is called at the start of the command buffer of every ExecuteCommandLists, before the lists are
    // written: the device brings the bindless set 0 of the translated shaders up to date there (vulkan_bindless.h)
    // and records what the set needs first. When it fails, nothing of that call is executed.
    void set_transfer_only_family(std::uint32_t family) { transfer_only_family_ = family; }
    void set_graphics_family(std::uint32_t family) { graphics_family_ = family; }
    void set_compute_copy_aspects(std::function<VkImageAspectFlags(VkFormat)> check) {
        replay_.compute_copy_aspects = std::move(check);
    }

    // `before` is called at the start of every submission of command lists (a held batch: when it is released),
    // before any lock here is taken: the device re-copies its slot-133 shadow buffers from the title's pages there.
    // Set once, before the first queue exists.
    void set_before_execute(std::function<void()> before) { before_execute_ = std::move(before); }

    // Table VAs are offsets in this device's arena. Install its scanner before creating any queues.
    // (The per-device scan is from the patch descriptor-table-refresh, part 2, of
    // Sigtryggur-Mani-Gudmundsson/graphics-runtime-patches at d9a0bc1b, GPL-3.0.)
    using TableScanner = std::function<void(std::uint64_t, std::uint32_t)>;
    void set_table_scanner(TableScanner scan) { table_scanner_ = std::move(scan); }
    void set_cpu_texture_publisher(std::function<bool()> f) { cpu_texture_publisher_ = std::move(f); }
    bool publish_cpu_textures() const { return !cpu_texture_publisher_ || cpu_texture_publisher_(); }
    void set_sample_table_resolver(std::function<bool(std::uint64_t,std::uint32_t)> f) { sample_table_resolver_ = std::move(f); }
    bool prepare_sample_table(std::uint64_t address, std::uint32_t count) const {
        return !sample_table_resolver_ || sample_table_resolver_(address,count);
    }
    // False when this context has no scanner of its own.
    bool scan_table(std::uint64_t va, std::uint32_t descriptors) const {
        if (!table_scanner_) return false;
        table_scanner_(va, descriptors);
        return true;
    }

    // `before` is called at the start of every submission of command lists (a held batch: when it is released),
    // before any lock here is taken: the device re-copies its slot-133 shadow buffers from the title's pages there.
    // Set once, before the first queue exists.
    using InputProbe = std::function<void(std::span<const CommandList* const>, std::uint64_t, const char*)>;
    void set_input_probe(InputProbe probe) { input_probe_ = std::move(probe); }
    bool diagnostic_buffer(std::uint64_t va, std::uint64_t& host, std::uint64_t& room) {
        std::shared_lock lock(resource_mutex_);
        BufferVaIndex<ResourceId>::Hit hit{};
        if (!buffer_vas_.find(va, hit)) return false;
        const auto* b = resolve(hit.id);
        if (!b || va - b->va >= b->size) return false;
        const auto offset = va - b->va;
        host = b->host ? reinterpret_cast<std::uint64_t>(b->host + offset) : 0;
        room = b->size - offset;
        return true;
    }
    const char* diagnostic_compute_name(std::uintptr_t id) {
        std::shared_lock lock(pipeline_mutex_);
        if (!device_fault_) return nullptr;
        const auto it = device_fault_->pipeline_names.find(id);
        return it == device_fault_->pipeline_names.end() ? nullptr : it->second[0];
    }

    void set_bindless_prepare(std::function<bool(VkCommandBuffer)> prepare) {
        std::unique_lock lock(resource_mutex_);
        bindless_prepare_ = std::move(prepare);
    }

    // D24_UNORM_S8_UINT stored as D32: copies of its depth plane run a compute pass (vulkan_depth24.h). `allocate`
    // gives the scratch buffers device-local memory. Without this such a copy is refused as unsupported.
    void enable_depth24(PFN_vkGetDeviceProcAddr get, Depth24Converter::Allocate allocate,
                       std::uint64_t max_storage_range = 1u << 27) {
        std::unique_lock lock(resource_mutex_);
        depth24_ = std::make_unique<Depth24Converter>(get, device_, std::move(allocate), max_storage_range,
                                                      device_fault_.get());
        if (!depth24_->loaded()) {
            depth24_.reset();
            return;
        }
        replay_.depth24_prepare = [this](Depth24Job& job, std::vector<Depth24Job>& parts) { return depth24_->prepare(job, parts); };
        replay_.depth24_run = [this](VkCommandBuffer command_buffer, const Depth24Job& job) {
            depth24_->run(command_buffer, job);
        };
    }

    // Copies the replay stages through a buffer (depth <-> colour, vulkan_replay.h) take device-local scratch from
    // `allocate`. Without this such a copy is refused as unsupported. `extra_usage` is added to the chunks' transfer
    // usage, for configure_quad_indices.
    void enable_copy_scratch(PFN_vkGetDeviceProcAddr get, ScratchBuffers::Allocate allocate,
                             VkBufferUsageFlags extra_usage = 0, std::vector<std::uint32_t> families = {}) {
        std::unique_lock lock(resource_mutex_);
        scratch_ = std::make_unique<ScratchBuffers>(get, device_, std::move(allocate), extra_usage, std::move(families),
                                                    device_fault_.get());
        if (!scratch_->loaded()) scratch_.reset();
    }

    // ExecuteIndirect with multi-argument signatures (vulkan_indirect.h) needs VK_EXT_device_generated_commands.
    // `allocate` returns device-local memory with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT. Without this such a call is
    // refused as unsupported.
    void enable_indirect_commands(PFN_vkGetDeviceProcAddr get, const IndirectExpander::Properties& properties,
                                  ScratchBuffers::Allocate allocate) {
        std::unique_lock lock(resource_mutex_);
        indirect_ = std::make_unique<IndirectExpander>(get, device_, properties, std::move(allocate));
        if (!indirect_->loaded()) indirect_.reset();
    }

    // Called after that command buffer's submit: true when it was accepted, false when vkQueueSubmit failed. The
    // set keeps its upload pending in the second case, because the copy never reached the GPU.
    void set_bindless_commit(std::function<void(bool)> commit) {
        std::unique_lock lock(resource_mutex_);
        bindless_commit_ = std::move(commit);
    }

    // The set 0 the commands of this ExecuteCommandLists bind, read after `prepare` succeeded and under the same
    // locks (a flush may have made a new set). Without it, lists with set-0 pipelines are refused (review D10:
    // pipelines keep no set of their own, whose handle a flush could free).
    void set_bindless_current(std::function<VkDescriptorSet()> current) {
        std::unique_lock lock(resource_mutex_);
        bindless_current_ = std::move(current);
    }

    void set_bindless_release(DeferredDeletionQueue::BindlessReleaseFn fn, void* user) {
        deletion_queue_.set_bindless_release(fn, user);
    }

    // The ticket of the latest submission: everything submitted so far is at or below it, and a resource that any
    // of that work may use can be freed once completed_retire_ticket() has reached it. Tickets run over all queues,
    // each recorded with its queue's timeline value, so this waits for every queue's timeline to pass what was
    // submitted to it, and never makes a queue wait.
    std::uint64_t submitted_retire_ticket() const { return submitted_ticket_.load(std::memory_order_acquire); }

    // With submit_mutex() held: each queue's timeline at the latest of its submissions not yet seen complete, one entry
    // per queue. A submit outside the D3D12 queues (the presenter's blit) waits for these to run after all the work
    // submitted so far, whichever VkQueue the title rendered the frame on (review B5).
    // Shared pages for the title's placed textures (vulkan_title_pages.h): made once by the device where sparse
    // residency images can alias (`families`: the queue families that can bind them, as a bit mask).
    void enable_title_pages(const VkPhysicalDeviceMemoryProperties& memory, std::uint32_t families,
                            std::function<VkDeviceMemory(std::uint32_t, VkDeviceSize)> allocate) {
        TitlePages::Functions fn;
        fn.allocate = std::move(allocate);
        fn.free = vk.vkFreeMemory;
        fn.requirements = vk.vkGetImageMemoryRequirements;
        fn.sparse_requirements = vk.vkGetImageSparseMemoryRequirements;
        fn.submitted_ticket = [this] { return submitted_ticket_.load(std::memory_order_acquire); };
        fn.completed_ticket = [this] { return completed_ticket_seen_.load(std::memory_order_acquire); };
        if (!fn.allocate || !fn.free || !fn.requirements || !fn.sparse_requirements || !vk.vkQueueBindSparse || !families)
            return;
        title_pages_ = std::make_unique<TitlePages>(device(), fn, memory);
        title_pages_families_ = families;
    }
    TitlePages* title_pages() const { return title_pages_.get(); }

    // The title pages' binds on a VkQueue of their own, made by a worker thread (flush_title_pages): NVIDIA's
    // vkQueueBindSparse costs about 0.8 ms of CPU per image, and the loading screen first-binds a thousand or more
    // textures at a time (run 03:27: 2,943 in 3.2 s), on the title's submitting thread with every queue lock waiting
    // behind it. Submissions wait for a batch on the GPU (title_pages_semaphore_: wait-before-signal, core with timeline
    // semaphores), so the binds overlap the work already queued. `queue`: in a family with VK_QUEUE_SPARSE_BINDING_BIT
    // and used by nothing else. Without one the binds stay on the submitting queue.
    void set_title_pages_queue(VkQueue queue) {
        if (!queue || !title_pages_ || binder_.thread.joinable()) return;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        if (vk.vkCreateSemaphore(device_, &info, nullptr, &binder_.semaphore) != VK_SUCCESS) {
            binder_.semaphore = VK_NULL_HANDLE;
            return;
        }
        if (vk.vkCreateSemaphore(device_, &info, nullptr, &binder_.chain) != VK_SUCCESS) {
            vk.vkDestroySemaphore(device_, binder_.semaphore, nullptr);
            binder_.semaphore = binder_.chain = VK_NULL_HANDLE;
            return;
        }
        binder_.queue = queue;
        binder_.thread = std::thread([this] { run_binder(); });
    }
    bool title_pages_async() const { return binder_.queue != VK_NULL_HANDLE; }

    // With submit_mutex() held: hands a batch of binds to the worker. It waits for `waits` at `values` (every queue's
    // work submitted so far: a page is reused only once what used it before has run) and signals the binder's timeline
    // at the value returned, which every queue's next submission then waits for. Tickets are handed out under
    // submit_mutex() and the worker takes batches in order, so the timeline only rises.
    //
    // `publish_now`: the batch binds render targets or other images the title writes right away, so every submission
    // from here on waits for it (on NVIDIA such a wait holds the next vkQueueSubmit until the bind is made). Otherwise
    // (streamed textures) the ticket is published once the worker has made the bind: submissions made meanwhile do not
    // wait, a view of a texture whose bind is not made yet reads zeros for those frames (strict residency), the title
    // texture sync copies into it only once it is made (title_bind_pending), and a list that copies into it waits for
    // it first (wait_title_binds).
    std::uint64_t enqueue_title_binds_locked(std::vector<TitlePages::Op> ops, std::vector<VkSemaphore> waits,
                                             std::vector<std::uint64_t> values, bool publish_now) {
        const std::uint64_t ticket = ++binder_.ticket;
        record_submit_locked(binder_.semaphore, ticket);  // deletions of what the binds name wait for it
        {
            std::lock_guard lock(binder_.mutex);
            if (!publish_now) {
                for (const auto& op : ops) binder_.pending[op.id] = ticket;
                binder_.pending_count.store(binder_.pending.size(), std::memory_order_release);
            }
            binder_.jobs.push_back({std::move(ops), std::move(waits), std::move(values), ticket, publish_now});
        }
        binder_.cv.notify_one();
        if (publish_now) publish_title_binds(ticket);
        return ticket;
    }

    // A streamed texture whose bind the worker has not made yet (enqueue_title_binds_locked).
    bool title_bind_pending(std::uintptr_t id) {
        if (!binder_.pending_count.load(std::memory_order_acquire)) return false;
        std::lock_guard lock(binder_.mutex);
        return binder_.pending.count(id) != 0;
    }

    // Before a submission that copies or resolves into title textures: until the binds of those it names are made
    // (and published, so the submission waits for them on the GPU). A copy into an image whose pages are not bound yet
    // writes nothing.
    void wait_title_binds(std::span<const CommandList* const> lists) {
        if (!binder_.pending_count.load(std::memory_order_acquire)) return;
        std::vector<std::uintptr_t> ids;
        for (const CommandList* list : lists)
            for (const auto& command : list->commands()) {
                if (const auto* c = std::get_if<CopyTexture>(&command)) ids.push_back(c->dst.resource);
                else if (const auto* r = std::get_if<CopyResource>(&command)) ids.push_back(r->dst);
                else if (const auto* s = std::get_if<ResolveSubresource>(&command)) ids.push_back(s->dst);
            }
        if (ids.empty()) return;
        std::unique_lock lock(binder_.mutex);
        binder_.done.wait(lock, [&] {
            return binder_.pending.empty() ||
                   std::none_of(ids.begin(), ids.end(), [&](std::uintptr_t id) { return binder_.pending.count(id) != 0; });
        });
    }

    std::vector<std::pair<VkSemaphore, std::uint64_t>> latest_pending_queue_values() const {
        std::vector<std::pair<VkSemaphore, std::uint64_t>> out;
        for (const auto& [semaphore, value] :
             retire_tickets_.pending_through(submitted_ticket_.load(std::memory_order_acquire))) {
            auto it = std::find_if(out.begin(), out.end(), [&](const auto& e) { return e.first == semaphore; });
            if (it == out.end()) out.emplace_back(semaphore, value);
            else if (value > it->second) it->second = value;
        }
        return out;
    }

    template<typename VkHandle>
    void defer_destroy(DeferredDeletionQueue::Kind kind, VkHandle handle) {
        const auto raw = static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(handle));
        if (!raw) return;
        std::lock_guard lock(submit_mutex_);
        const std::uint64_t safe_ticket = submitted_ticket_.load(std::memory_order_acquire) + 1;
        deletion_queue_.push(safe_ticket, kind, raw);
    }

    void defer_destroy(DeferredDeletionQueue::Kind kind, std::uint64_t raw_handle) {
        if (!raw_handle && kind != DeferredDeletionQueue::Kind::bindless_slot) return;
        std::lock_guard lock(submit_mutex_);
        const std::uint64_t safe_ticket = submitted_ticket_.load(std::memory_order_acquire) + 1;
        deletion_queue_.push(safe_ticket, kind, raw_handle);
    }

    // Replaces a graphics PSO's pipeline for every later draw (GraphicsPipeline::Upgrade, the optimised pipeline-library
    // link) and destroys the one it replaces once the work submitted so far is done. Under resource_mutex_ (which every
    // execute holds shared from prepare until just before its submit) and tracker_mutex_ (held from there through the
    // submit): no list is between choosing its pipeline and submitting, so every list either recorded `retired` and is
    // submitted already, which the deferred deletion waits for, or will see `replacement`.
    void replace_graphics_pipeline(std::atomic<VkPipeline>& slot, VkPipeline replacement, VkPipeline retired) {
        std::unique_lock resource_lock(resource_mutex_);
        std::lock_guard tracker_lock(tracker_mutex_);
        slot.store(replacement, std::memory_order_release);
        defer_destroy(DeferredDeletionQueue::Kind::pipeline, retired);
    }

    // Waits (not for the device: for the queues' timelines) until every submission up to `ticket` is done, then
    // returns the completed ticket. For a caller whose list of retired objects has grown past what it will hold.
    std::uint64_t wait_retire_ticket(std::uint64_t ticket) {
        std::vector<VkSemaphore> semaphores;
        std::vector<std::uint64_t> values;
        {
            std::lock_guard lock(submit_mutex_);
            if (is_device_lost()) return 0;
            const auto pending = retire_tickets_.pending_through(ticket);
            for (const auto& [semaphore, value] : pending) {
                std::size_t idx = 0;
                for (; idx < semaphores.size(); ++idx)
                    if (semaphores[idx] == semaphore) break;
                if (idx == semaphores.size()) {
                    semaphores.push_back(semaphore);
                    values.push_back(value);
                } else if (value > values[idx]) {
                    values[idx] = value;
                }
            }
            for (VkSemaphore sem : semaphores) ++progress_waits_[sem].waiters;
        }
        if (!semaphores.empty()) {
            struct WaiterGuard {
                VulkanContext& ctx;
                const std::vector<VkSemaphore>& sems;
                ~WaiterGuard() {
                    std::vector<VkSemaphore> to_destroy;
                    {
                        std::lock_guard lock(ctx.submit_mutex_);
                        for (VkSemaphore sem : sems) {
                            const auto it = ctx.progress_waits_.find(sem);
                            if (it == ctx.progress_waits_.end()) continue;
                            if (--it->second.waiters == 0) {
                                if (it->second.destroy_when_done) to_destroy.push_back(sem);
                                ctx.progress_waits_.erase(it);
                            }
                        }
                    }
                    if (ctx.vk.vkDestroySemaphore) {
                        for (VkSemaphore sem : to_destroy) ctx.vk.vkDestroySemaphore(ctx.device_, sem, nullptr);
                    }
                }
            } guard{*this, semaphores};

            VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait.semaphoreCount = std::uint32_t(semaphores.size());
            wait.pSemaphores = semaphores.data();
            wait.pValues = values.data();
            for (;;) {  // outside submit_mutex_: a signal may need it; bounded slice wakes on device loss
                if (is_device_lost()) return 0;
                const VkResult res = vk.vkWaitSemaphores(device_, &wait, kRetireWaitSliceNs);
                if (res == VK_SUCCESS) break;
                if (res == VK_TIMEOUT) continue;
                if (res == VK_ERROR_DEVICE_LOST) {
                    mark_device_lost();
                    return 0;
                }
                break;
            }
        }
        return completed_retire_ticket();
    }

    // The bindless flush of this ExecuteCommandLists started a new set generation, whose table upload is in its
    // command buffer: called from the `prepare` callback (tracker_mutex_ held), so execute() makes every other
    // queue's next submission wait for this one, which binds the new set.
    void note_bindless_upload() { bindless_upload_pending_ = true; }

    std::uint64_t completed_retire_ticket() {
        bool lost = false;
        std::uint64_t ticket = 0;
        {
            std::lock_guard lock(submit_mutex_);
            if (is_device_lost()) return 0;
            ticket = retire_tickets_.completed([this, &lost](VkSemaphore semaphore, std::uint64_t value) {
                std::uint64_t done = 0;
                const VkResult result = observe_result(vk.vkGetSemaphoreCounterValue(device_, semaphore, &done),
                                                       "vkGetSemaphoreCounterValue (retirement)");
                if (result == VK_ERROR_DEVICE_LOST) lost = true;
                return result == VK_SUCCESS && done >= value;
            });
        }
        // Outside submit_mutex_, as in submit(): notifying the fence dispatcher can run callbacks that submit.
        // A DEVICE_LOST seen here is the first Vulkan operation to discover it; it must not be mistaken for
        // "this ticket has not completed".
        if (lost) mark_device_lost();
        if (!lost) {  // the last completed ticket seen, for readers that may not take submit_mutex_ (TitlePages)
            std::uint64_t seen = completed_ticket_seen_.load(std::memory_order_relaxed);
            while (ticket > seen && !completed_ticket_seen_.compare_exchange_weak(seen, ticket, std::memory_order_acq_rel)) {
            }
        }
        return lost ? 0 : ticket;
    }

    void drain_deleted_resources(std::uint64_t current_ticket) {
        deletion_queue_.drain(current_ticket, device_, vk);
    }

    // True when the VA names a sparse resource: the sparse pass binds its memory, not the arena's mapping, so the
    // host's bytes at that offset are not the resource's. The console's value writes skip those.
    bool is_sparse_va(std::uint64_t va) {
        const std::shared_lock lock(resource_mutex_);
        BufferVaIndex<ResourceId>::Hit hit{};
        if (!buffer_vas_.find(va, hit)) return false;
        const ResourceBinding* const b = resolve(hit.id);
        return b && b->sparse_buffer;
    }

    // Both methods are called with submit_mutex_ held. A queue's own progress timeline is
    // monotonic even when another queue completes first; global tickets are retired in order.
    void record_submit_locked(VkSemaphore progress, std::uint64_t value) {
        const auto ticket = submitted_ticket_.fetch_add(1, std::memory_order_acq_rel) + 1;
        retire_tickets_.record(ticket, progress, value);
        note_image_work_locked();
    }
    void retire_queue_idle_locked(VkSemaphore progress) {
        retire_tickets_.queue_idle(progress);
    }

    VulkanDeviceFunctions vk;

    // Vulkan requires external synchronisation of a VkQueue; every D3D12 queue of the device may share one.
    std::mutex& submit_mutex() { return submit_mutex_; }

    // What the device-lost summary says about each queue (I3): VulkanQueue keeps it, by its progress semaphore.
    void track_queue(VkSemaphore progress, VkQueue queue, std::uint32_t family, const char* kind) {
        std::lock_guard lock(queue_records_mutex_);
        queue_records_.push_back({progress, queue, family, kind, 0, {}});
    }
    void untrack_queue(VkSemaphore progress) {
        std::lock_guard lock(queue_records_mutex_);
        std::erase_if(queue_records_, [&](const QueueRecord& r) { return r.progress == progress; });
        if (lost_submission_ == progress) lost_submission_ = VK_NULL_HANDLE;
    }
    // A submission reached the queue (its progress timeline will reach `value`); `marker`: the last debug marker
    // it carries, if any.
    void note_queue_submitted(VkSemaphore progress, std::uint64_t value, std::string_view marker = {}) {
        std::lock_guard lock(queue_records_mutex_);
        for (QueueRecord& r : queue_records_)
            if (r.progress == progress) {
                r.submitted = value;
                if (!marker.empty()) r.marker.assign(marker);
            }
    }
    void note_queue_marker(VkSemaphore progress, std::string_view marker) {
        std::lock_guard lock(queue_records_mutex_);
        for (QueueRecord& r : queue_records_)
            if (r.progress == progress && !marker.empty()) r.marker.assign(marker);
    }
    // vkQueueSubmit on this queue returned VK_ERROR_DEVICE_LOST.
    void note_lost_submission(VkSemaphore progress) {
        std::lock_guard lock(queue_records_mutex_);
        if (!lost_submission_) lost_submission_ = progress;
    }
    // VK_EXT_device_fault and its deviceFault feature are enabled: the loss report asks it (LEONIDA_DRED or not).
    void set_device_fault_enabled(bool enabled) { device_fault_enabled_ = enabled; }

    // The one line every loss prints.
    std::string device_lost_summary(bool fault, const std::vector<VkDeviceFaultAddressInfoEXT>& addresses,
                                    const std::string& description) const {
        std::string line = "[rt] d3d12: DEVICE LOST:";
        char text[256];
        bool any = false;
        {
            std::lock_guard lock(queue_records_mutex_);
            for (const QueueRecord& r : queue_records_) {
                std::uint64_t done = 0;
                const bool known = vk.vkGetSemaphoreCounterValue &&
                                   vk.vkGetSemaphoreCounterValue(device_, r.progress, &done) == VK_SUCCESS;
                const bool lost_here = r.progress == lost_submission_;
                if (!lost_here && (!r.submitted || (known && done >= r.submitted))) continue;
                std::snprintf(text, sizeof text, "%s %s queue %p (family %u)", any ? ";" : "", r.kind,
                              static_cast<void*>(r.queue), r.family);
                line += text;
                if (lost_here) {
                    std::snprintf(text, sizeof text, " lost its submission %llu at vkQueueSubmit",
                                  static_cast<unsigned long long>(r.submitted + 1));
                    line += text;
                }
                if (r.submitted && (!known || done < r.submitted)) {
                    if (known)
                        std::snprintf(text, sizeof text, "%s submissions %llu..%llu unfinished", lost_here ? "," : "",
                                      static_cast<unsigned long long>(done + 1),
                                      static_cast<unsigned long long>(r.submitted));
                    else
                        std::snprintf(text, sizeof text, "%s %llu submitted, GPU progress unreadable",
                                      lost_here ? "," : "", static_cast<unsigned long long>(r.submitted));
                    line += text;
                }
                if (!r.marker.empty()) line += ", last marker \"" + r.marker + "\"";
                any = true;
            }
        }
        if (!any) line += " no queue had a submission in flight";
        if (fault) {
            line += "; fault \"" + description + "\"";
            if (!addresses.empty()) {
                const VkDeviceFaultAddressInfoEXT& a = addresses.front();
                std::snprintf(text, sizeof text, " at %#llx (type %d%s)", static_cast<unsigned long long>(a.reportedAddress),
                              static_cast<int>(a.addressType), addresses.size() > 1 ? ", more follow with LEONIDA_DRED" : "");
                line += text;
            }
        } else {
            line += device_fault_enabled_ ? "; the driver reports no fault information"
                                          : "; no fault information (VK_EXT_device_fault not enabled)";
        }
        line += breadcrumbs_ ? "; breadcrumbs follow" : "; LEONIDA_DRED=1 adds per-command breadcrumbs";
        return line;
    }

    void mark_device_lost(const char* where = "a Vulkan call returned VK_ERROR_DEVICE_LOST") noexcept {
        set_device_lost(device_lost_, where);
        run_device_lost_hook();  // LEONIDA_AFTERMATH: the GPU crash dump is written before waiters wake
        if (fence_dispatcher_) {
            try {
                fence_dispatcher_->notify_device_lost();
            } catch (...) {
            }
        }
        report_device_lost();
    }
    // The fence waiter sets device_lost_ itself, so the first caller that sees the loss prints the report.
    bool is_device_lost() const noexcept {
        if (!device_lost_.load(std::memory_order_acquire)) return false;
        report_device_lost();
        return true;
    }

    // Once per loss. Always one line (device_lost_summary): which queue lost a submission or still had submissions
    // unfinished, its last debug marker, and VK_EXT_device_fault's first address. Then, when the diagnostics are
    // configured (LEONIDA_DRED), the breadcrumbs and every fault address.
    void report_device_lost() const noexcept {
        if (lost_reported_.exchange(true, std::memory_order_acq_rel)) return;
        std::FILE* const out = report_out_;
        try {
            std::vector<VkDeviceFaultAddressInfoEXT> addresses;
            std::string description;
            const bool fault = device_fault(addresses, description);
            std::fprintf(out, "%s\n", device_lost_summary(fault, addresses, description).c_str());
            if (!breadcrumbs_) return;
            breadcrumbs_->report(out);
            if (!fault) {
                std::fprintf(out, "[rt] d3d12 DRED: no device fault information (VK_EXT_device_fault)\n");
                return;
            }
            std::fprintf(out, "[rt] d3d12 DRED: device fault: %s\n", description.c_str());
            for (const auto& a : addresses)
                std::fprintf(out, "[rt] d3d12 DRED:   address %#llx (precision %#llx, type %d)\n",
                             static_cast<unsigned long long>(a.reportedAddress),
                             static_cast<unsigned long long>(a.addressPrecision), static_cast<int>(a.addressType));
        } catch (...) {
            std::fprintf(out, "[rt] d3d12 DRED: the report failed (out of memory)\n");
        }
    }

    void set_signal_event_fn(FenceWaitDispatcher::SignalEventFn fn) {
        if (fence_dispatcher_) fence_dispatcher_->set_signal_event_fn(fn);
    }
    void enqueue_fence_wait(VkSemaphore sem, std::uint64_t value, void* event_handle) {
        if (fence_dispatcher_) fence_dispatcher_->enqueue(sem, value, event_handle);
    }
    void enqueue_fence_callback(VkSemaphore sem, std::uint64_t value, std::function<void()> cb) {
        if (fence_dispatcher_) fence_dispatcher_->enqueue_callback(sem, value, std::move(cb));
    }
    void cancel_fence_waits(VkSemaphore sem) {
        if (fence_dispatcher_) fence_dispatcher_->cancel(sem);
    }

    // Cross-queue Wait-before-Signal: per fence, the highest value a Signal was submitted for, and the queues that
    // hold batches until one is (queue_holds.h).
    QueueHolds<VkSemaphore>& holds() { return holds_; }

    // A cached view of `range` (every field of it counts) in r.format, made on first use and destroyed with the image
    // (remove_resource). Called from prepare; protected by views_mutex_ so concurrent shared_lock readers are safe.
    VkImageView view(const ResourceBinding& r, const VkImageSubresourceRange& range) {
        std::lock_guard lock(views_mutex_);
        const auto key = std::make_tuple(r.image, range.aspectMask, range.baseMipLevel, range.levelCount,
                                         range.baseArrayLayer, range.layerCount, r.format);
        if (const auto it = views_.find(key); it != views_.end()) return it->second;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = r.image;
        info.viewType = r.line ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_2D_ARRAY;  // 1D images take no 2D view
        info.format = r.format;
        info.subresourceRange = range;
        // Attachment views need only color-attachment usage. Inheriting STORAGE usage from a mutable/extended
        // resource (typeless views such as RGBA8 sRGB, or R32_UINT-backed R9G9B9E5 UAV images) would require unrelated usages.
        VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
        usage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        if (r.typeless || (range.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT && r.color_attachment))
            info.pNext = &usage;
        VkImageView view = VK_NULL_HANDLE;
        if (vk.vkCreateImageView(device_, &info, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
        views_[key] = view;
        return view;
    }

    // A 2D view of mip 0, layer 0 of `r` in r.format, to sample: a plane of a composed present. Made on first use and
    // destroyed with the image (remove_resource), like view(); the presenter made two for every composed present
    // (review I12).
    VkImageView plane_view(const ResourceBinding& r) {
        std::lock_guard lock(views_mutex_);
        const auto key = std::make_pair(r.image, r.format);
        if (const auto it = plane_views_.find(key); it != plane_views_.end()) return it->second;
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = r.image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = r.format;
        info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        if (vk.vkCreateImageView(device_, &info, nullptr, &view) != VK_SUCCESS) return VK_NULL_HANDLE;
        plane_views_[key] = view;
        return view;
    }

private:
    friend class VulkanQueue;
    friend class VulkanPresenter;

    // buffer_at finds a buffer by VA: one with no image, a VkBuffer, a VA and bytes (what the linear search it replaced
    // accepted). Both are called with resource_mutex_ held uniquely.
    static bool addressable(const ResourceBinding& b) { return !b.image && b.buffer && b.va && b.size; }
    void unindex_buffer(ResourceId id, const ResourceBinding& b) {
        if (addressable(b)) buffer_vas_.erase(b.va, id);
    }
    // resources_[id] = binding, with the VA index kept in step (a re-registered ID drops its old range first).
    void set_resource_locked(ResourceId id, const ResourceBinding& binding) {
        auto [it, inserted] = resources_.try_emplace(id, binding);
        if (addressable(binding)) {
            try {
                buffer_vas_.insert(binding.va, binding.size, id);
            } catch (...) {
                if (inserted) resources_.erase(it);
                throw;
            }
        }
        if (!inserted) {
            // A same-VA insert updates its size in place; removing the old key would remove the new range.
            if (addressable(it->second) && (!addressable(binding) || it->second.va != binding.va))
                unindex_buffer(id, it->second);
            it->second = binding;
        }
    }

    const ResourceBinding* resolve(ResourceId id) const {
        const ResourceBinding* found = nullptr;
        if (const auto it = resources_.find(id); it != resources_.end()) {
            found = &it->second;
        } else {
            if (!transient_count_.load(std::memory_order_acquire)) return nullptr;
            std::lock_guard lock(transient_mutex_);
            const auto t = transient_bindings_.find(id);
            if (t == transient_bindings_.end()) return nullptr;
            found = &t->second;
        }
        const auto& binding = *found;
        if (auto* memory = binding.deferred_memory; memory && !memory->ensure()) return nullptr;
        // The caller holds resource_mutex_, so the registered owner's lifetime includes this bind.
        if (binding.ensure_residency && !binding.ensure_residency(binding.residency_owner)) return nullptr;
        if (auto* seen = binding.replay_seen; seen && !seen->load(std::memory_order_relaxed))
            seen->store(true, std::memory_order_relaxed);
        return found;
    }

    // The render pass and framebuffer that resolve one layer of a depth-only multisampled image into one layer of
    // another: views `from` and `to` (from view()), in the layouts given. Used where the dynamic-rendering resolve
    // reads the wrong source layer (D3D-417; vulkan_replay.h says when). Both objects are cached: a pass by what it
    // describes, a framebuffer by its images and views, until remove_resource retires either image.
    VkFramebuffer depth_resolve_framebuffer(const ResourceBinding& source, VkImageView from, VkImageLayout from_layout,
                                            const ResourceBinding& target, VkImageView to, VkImageLayout to_layout,
                                            VkResolveModeFlagBits mode, VkExtent2D extent, VkRenderPass& pass) {
        std::lock_guard lock(views_mutex_);
        const auto pass_key = std::make_tuple(source.format, source.samples, from_layout, to_layout, mode);
        if (const auto it = resolve_passes_.find(pass_key); it != resolve_passes_.end()) {
            pass = it->second;
        } else {
            VkAttachmentDescription2 attachments[2]{};
            for (auto& attachment : attachments) {
                attachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
                attachment.format = source.format;
                attachment.samples = VK_SAMPLE_COUNT_1_BIT;
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;      // the source is read; the destination keeps
                attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;   // what is outside the render area
                attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;  // depth-only formats
                attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            }
            attachments[0].samples = static_cast<VkSampleCountFlagBits>(source.samples);
            attachments[0].initialLayout = attachments[0].finalLayout = from_layout;
            attachments[1].initialLayout = attachments[1].finalLayout = to_layout;
            const VkAttachmentReference2 from_reference{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 0,
                                                        from_layout, VK_IMAGE_ASPECT_DEPTH_BIT};
            const VkAttachmentReference2 to_reference{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 1, to_layout,
                                                      VK_IMAGE_ASPECT_DEPTH_BIT};
            VkSubpassDescriptionDepthStencilResolve resolve{
                VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE};
            resolve.depthResolveMode = mode;
            resolve.stencilResolveMode = VK_RESOLVE_MODE_NONE;
            resolve.pDepthStencilResolveAttachment = &to_reference;
            VkSubpassDescription2 subpass{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
            subpass.pNext = &resolve;
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.pDepthStencilAttachment = &from_reference;
            VkRenderPassCreateInfo2 info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
            info.attachmentCount = 2;
            info.pAttachments = attachments;
            info.subpassCount = 1;
            info.pSubpasses = &subpass;
            if (vk.vkCreateRenderPass2(device_, &info, nullptr, &pass) != VK_SUCCESS) return VK_NULL_HANDLE;
            resolve_passes_[pass_key] = pass;
        }
        const auto key = std::make_tuple(source.image, target.image, from, to, pass);
        if (const auto it = resolve_framebuffers_.find(key); it != resolve_framebuffers_.end()) return it->second;
        const VkImageView views[] = {from, to};
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = pass;
        info.attachmentCount = 2;
        info.pAttachments = views;
        info.width = extent.width;
        info.height = extent.height;
        info.layers = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        if (vk.vkCreateFramebuffer(device_, &info, nullptr, &framebuffer) != VK_SUCCESS) return VK_NULL_HANDLE;
        resolve_framebuffers_[key] = framebuffer;
        return framebuffer;
    }

    std::uint64_t poll_retirement_locked(VkResult& result) {
        return retire_tickets_.completed([&](VkSemaphore semaphore, std::uint64_t value) {
            std::uint64_t done = 0;
            result = observe_result(vk.vkGetSemaphoreCounterValue(device_, semaphore, &done),
                                    "vkGetSemaphoreCounterValue (retirement)");
            return result == VK_SUCCESS && done >= value;
        });
    }

    void watch_queue_progress(std::stop_token stop) {
        using Watch = SubmissionProgressWatch;
        Watch watch;
        std::uint64_t retired = 0;
        std::mutex timer_mutex;
        std::condition_variable_any timer;
        std::unique_lock timer_lock(timer_mutex);
        while (!stop.stop_requested()) {
            timer.wait_for(timer_lock, stop, std::chrono::seconds(1), [] { return false; });
            if (stop.stop_requested() || is_device_lost()) break;
            auto state = Watch::State::submit_lock_busy;
            VkResult result = VK_SUCCESS;
            std::uint64_t submitted;
            {
                // Keep queue timelines alive as in completed_retire_ticket(). A submit or idle wait
                // blocked inside the driver can hold this mutex indefinitely.
                std::unique_lock lock(submit_mutex_, std::try_to_lock);
                submitted = submitted_retire_ticket();
                if (lock.owns_lock()) {
                    if (is_device_lost()) break;
                    retired = poll_retirement_locked(result);
                    state = retired < submitted ? Watch::State::pending : Watch::State::idle;
                }
            }
            if (result == VK_ERROR_DEVICE_LOST) {
                mark_device_lost();
                break;
            }
            // A failed counter query does not measure GPU progress.
            if (result != VK_SUCCESS) state = Watch::State::idle;
            // LEONIDA_HANG_EXIT_S=<s> (for boots): work pending and retirement still for s seconds is a GPU hang
            // the driver may recover from without Vulkan ever reporting the device lost (Xid 109: the title then
            // waits forever, the window frozen). The stall is reported (with the checkpoints under
            // LEONIDA_STALL_CHECKPOINTS=1) and the process ends, so the boot does not sit to its time limit.
            {
                static const long hang_exit_s = [] {
                    const char* v = std::getenv("LEONIDA_HANG_EXIT_S");
                    return v && *v ? std::strtol(v, nullptr, 10) : 0l;
                }();
                static std::uint64_t last_retired = ~std::uint64_t(0);
                static auto still_since = Watch::Clock::now();
                const auto now = Watch::Clock::now();
                if (state != Watch::State::pending || retired != last_retired) {
                    last_retired = retired;
                    still_since = now;
                } else if (hang_exit_s > 0 && now - still_since >= std::chrono::seconds(hang_exit_s)) {
                    std::fprintf(stderr, "[rt] hang watchdog: queue retirement still for %ld s with work pending "
                                         "(submitted ticket %llu, retired %llu): ending the process\n",
                                 hang_exit_s, static_cast<unsigned long long>(submitted),
                                 static_cast<unsigned long long>(retired));
                    device_fault_->report_stall("hang watchdog", submitted, retired);
                    std::fflush(stderr);
                    std::_Exit(0x47);  // 'HANG' (portable immediate process termination)
                }
            }
            if (watch.sample(state, retired, Watch::Clock::now()))
                device_fault_->report_stall(state == Watch::State::pending ? "queue retirement has not advanced"
                                                                          : "could not sample the submit lock",
                                            submitted, retired);
        }
    }

    VkDevice device_;
    bool loaded_ = false;
    bool ray_tracing_ = false;
    std::unique_ptr<XdxrBuilder> xdxr_;   // set: acceleration structures are XDXR BVH4 records
    std::atomic<bool> device_lost_{false};
    std::unique_ptr<BreadcrumbRing> breadcrumbs_;  // device-lost diagnostics, when configured
    mutable std::atomic<bool> lost_reported_{false};
    bool device_fault_enabled_ = false;
    std::FILE* report_out_ = stderr;
    // For the device-lost summary (track_queue). A leaf lock: nothing else is taken under it.
    struct QueueRecord {
        VkSemaphore progress;
        VkQueue queue;
        std::uint32_t family;
        const char* kind;
        std::uint64_t submitted;  // the progress value of the last submission that reached the queue
        std::string marker;       // the last debug marker submitted on it
    };
    mutable std::mutex queue_records_mutex_;
    std::vector<QueueRecord> queue_records_;
    VkSemaphore lost_submission_ = VK_NULL_HANDLE;  // the queue whose vkQueueSubmit returned VK_ERROR_DEVICE_LOST
    alignas(64) std::shared_mutex pipeline_mutex_;  // shared for prepare/compile/emit pipeline lookups, unique for add/remove/workers
    alignas(64) std::shared_mutex resource_mutex_;  // shared for prepare lookups, unique for add/remove
    alignas(64) std::mutex tracker_mutex_;
    alignas(64) std::mutex views_mutex_;
    alignas(64) std::mutex submit_mutex_;
    bool image_work_pending_ = false;  // under submit_mutex_
    HostTextureStats host_texture_stats_;  // under submit_mutex_
    RetireTickets<VkSemaphore> retire_tickets_;
    struct ProgressWaitState {
        std::size_t waiters = 0;
        bool destroy_when_done = false;
    };
    void destroy_or_defer_progress_semaphore(VkSemaphore semaphore) {
        if (!semaphore) return;
        bool destroy_now = true;
        {
            std::lock_guard lock(submit_mutex_);
            const auto it = progress_waits_.find(semaphore);
            if (it != progress_waits_.end() && it->second.waiters > 0) {
                it->second.destroy_when_done = true;
                destroy_now = false;
            }
        }
        if (destroy_now && vk.vkDestroySemaphore) vk.vkDestroySemaphore(device_, semaphore, nullptr);
    }
    std::unordered_map<VkSemaphore, ProgressWaitState> progress_waits_;
    std::atomic<std::uint64_t> submitted_ticket_{0};
    std::atomic<std::uint64_t> completed_ticket_seen_{0};  // completed_retire_ticket's last answer
    DeferredDeletionQueue deletion_queue_;
    std::unique_ptr<DeviceFaultDiagnostics> device_fault_;
    std::jthread progress_watch_;
    std::unique_ptr<FenceWaitDispatcher> fence_dispatcher_;
    QueueHolds<VkSemaphore> holds_;
    std::unordered_map<ResourceId, ResourceBinding> resources_;
    mutable std::mutex transient_mutex_;  // after resource_mutex_ when both are held (resolve)
    std::mutex driver_heavy_mutex_;       // driver_heavy_mutex(): a leaf lock, held only around the driver calls
    std::unordered_map<ResourceId, ResourceBinding> transient_bindings_;  // add_transient_binding
    std::atomic<std::size_t> transient_count_{0};
    BufferVaIndex<ResourceId> buffer_vas_;  // the addressable buffers of resources_, by VA range (buffer_at)
    TitleBufferLookup title_buffer_at_;      // buffer_at's fallback for title addresses (set_title_buffer_lookup)
    std::function<void(std::uint64_t, std::uint64_t)> title_range_mapper_;  // set_title_range_mapper
    std::unordered_map<ResourceId, std::shared_ptr<SparseResource>> sparse_resources_;
    std::unordered_map<std::uintptr_t, ComputePipeline> pipelines_;
    std::unordered_map<std::uintptr_t, GraphicsPipeline> graphics_pipelines_;
    std::unique_ptr<PipelineWorkers> pipeline_workers_;
    std::function<std::uint64_t()> large_compile_budget_;
    std::function<std::shared_ptr<void>()> large_compile_demand_;
    std::unordered_map<std::uint64_t, VkAccelerationStructureKHR> acceleration_structures_;
    LayoutTracker tracker_;
    std::mutex storage_mutex_;  // a leaf lock: guards storage_pending_ and zero_pending_ only
    std::unordered_set<ResourceId> storage_pending_;
    struct PendingZero {
        ResourceId id;
        VkBuffer buffer;
    };
    std::vector<PendingZero> zero_pending_;  // zero_on_next_submit
    // The last submission that initialized storage images (its queue's progress semaphore and ticket) and a counter
    // that says which one: every other queue's next submission waits for it, so no queue uses such an image before
    // the zero-fill that put it in GENERAL has run.
    VkSemaphore storage_init_semaphore_ = VK_NULL_HANDLE;
    std::uint64_t storage_init_value_ = 0;
    std::uint64_t storage_init_version_ = 0;
    // Keyed by everything view() puts in the view: image, aspects, mips (base, count), layers (base, count), format.
    std::map<std::tuple<VkImage, VkImageAspectFlags, std::uint32_t, std::uint32_t, std::uint32_t, std::uint32_t, VkFormat>,
             VkImageView>
        views_;
    std::map<std::pair<VkImage, VkFormat>, VkImageView> plane_views_;  // plane_view(), under views_mutex_
    // depth_resolve_framebuffer's caches, under views_mutex_ like views_.
    std::map<std::tuple<VkFormat, std::uint32_t, VkImageLayout, VkImageLayout, VkResolveModeFlagBits>, VkRenderPass>
        resolve_passes_;
    std::map<std::tuple<VkImage, VkImage, VkImageView, VkImageView, VkRenderPass>, VkFramebuffer> resolve_framebuffers_;
    std::function<void()> before_execute_;
    InputProbe input_probe_;
    TableScanner table_scanner_;
    GeometryProvider geometry_provider_;
    std::function<bool(std::uint64_t,std::uint32_t)> sample_table_resolver_;
    std::function<bool()> cpu_texture_publisher_;
    std::function<bool(VkCommandBuffer)> bindless_prepare_;
    std::function<VkDescriptorSet()> bindless_current_;
    bool bindless_upload_pending_ = false;  // tracker_mutex_
    // The last submission that carried a new generation's table upload (its queue's progress semaphore and value), and
    // a counter that says which one: every other queue's next submission waits for it (storage_mutex_).
    VkSemaphore bindless_upload_semaphore_ = VK_NULL_HANDLE;
    std::uint64_t bindless_upload_value_ = 0;
    std::uint64_t bindless_upload_version_ = 0;
    // The title's textures on shared pages (vulkan_title_pages.h), and the last batch of their binds (the queue whose
    // timeline signalled it and the value), which every other queue's next submission waits for, as for the bindless
    // upload above (storage_mutex_).
    std::unique_ptr<TitlePages> title_pages_;
    std::uint32_t title_pages_families_ = 0;  // queue families with VK_QUEUE_SPARSE_BINDING_BIT, as a mask
    VkSemaphore title_pages_semaphore_ = VK_NULL_HANDLE;
    std::uint64_t title_pages_value_ = 0;
    std::uint64_t title_pages_version_ = 0;
    // Uploads submitted without a host wait (the device's staged texture writes): per staging queue timeline, the
    // latest value; every queue's next submission waits for the ones it has not seen (note_upload). Under
    // storage_mutex_.
    std::vector<std::pair<VkSemaphore, std::uint64_t>> upload_values_;
    std::uint64_t upload_version_ = 0;

    // The title pages' bind worker (set_title_pages_queue).
    struct BindJob {
        std::vector<TitlePages::Op> ops;
        std::vector<VkSemaphore> waits;
        std::vector<std::uint64_t> values;
        std::uint64_t ticket = 0;
        bool published = false;  // published at enqueue (publish_now)
    };
    struct Binder {
        VkQueue queue = VK_NULL_HANDLE;
        VkSemaphore semaphore = VK_NULL_HANDLE;
        VkSemaphore chain = VK_NULL_HANDLE;  // orders the calls of one job (bind_title_pages); worker thread only
        std::uint64_t chain_value = 0;
        std::uint64_t ticket = 0;  // the last handed out (submit_mutex_)
        std::mutex mutex;
        std::condition_variable cv;    // jobs to make
        std::condition_variable done;  // a job made: `pending` shrank
        std::deque<BindJob> jobs;
        // Streamed textures in jobs not made yet, and the job's ticket (title_bind_pending).
        std::unordered_map<std::uintptr_t, std::uint64_t> pending;
        std::atomic<std::size_t> pending_count{0};
        bool stop = false;
        std::thread thread;
    } binder_;

    // Every queue's next submission waits for the binder's timeline at `ticket` (or a later value published already).
    void publish_title_binds(std::uint64_t ticket) {
        std::lock_guard lock(storage_mutex_);
        if (title_pages_semaphore_ == binder_.semaphore && title_pages_value_ >= ticket) return;
        title_pages_semaphore_ = binder_.semaphore;
        title_pages_value_ = ticket;
        ++title_pages_version_;
    }

    // After the worker made (or gave up on) a job: published if it was not, its textures no longer pending.
    void finish_bind_job(const BindJob& job) {
        if (!job.published) publish_title_binds(job.ticket);
        {
            std::lock_guard lock(binder_.mutex);
            for (const auto& op : job.ops) {
                const auto it = binder_.pending.find(op.id);
                if (it != binder_.pending.end() && it->second == job.ticket) binder_.pending.erase(it);
            }
            binder_.pending_count.store(binder_.pending.size(), std::memory_order_release);
        }
        binder_.done.notify_all();
    }

    void run_binder() {
        std::unique_lock lock(binder_.mutex);
        for (;;) {
            binder_.cv.wait(lock, [this] { return binder_.stop || !binder_.jobs.empty(); });
            if (binder_.jobs.empty()) return;  // stopping, every batch made
            BindJob job = std::move(binder_.jobs.front());
            binder_.jobs.pop_front();
            lock.unlock();
            bind_title_pages(job);
            lock.lock();
        }
    }

    // Images per vkQueueBindSparse call. NVIDIA holds a device-wide lock through the call: every other thread's
    // vkQueueSubmit waited behind a 2 s bind of 2,000 textures (run 03:44: a 129x1 texture copy took 841 ms). Chunks of
    // this many keep each call to a fraction of a frame, and the submissions get in between. 128 images were still
    // ~100-150 ms a call (0.8-1.2 ms an image), and the texture sync's copy flush waited 140-380 ms behind them (run 13);
    // 16 keep a call near 15 ms.
    static constexpr std::size_t kBindChunkImages = 16;

    void bind_title_pages(BindJob& job) {
        std::size_t image_binds = 0, opaque_binds = 0, image_info_count = 0, opaque_info_count = 0;
        for (const auto& op : job.ops) {
            image_binds += op.image_binds.size();
            opaque_binds += op.opaque_binds.size();
            image_info_count += !op.image_binds.empty();
            opaque_info_count += !op.opaque_binds.empty();
        }
        const auto t_wait = std::chrono::steady_clock::now();
        VkResult result = is_device_lost() ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
        // The job's waits (every queue's work submitted before it) are met here, on the host, before any bind call:
        // passed to vkQueueBindSparse, NVIDIA met them inside the call, with the device-wide lock held, so every
        // submission waited until the GPU had drained (0.3-4.3 s "slow binds", each holding the title's submitting
        // threads; runs 10-13). The binds themselves then wait for nothing.
        bool waited_on_host = false;
        if (result == VK_SUCCESS && !job.waits.empty()) {
            VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            wait.semaphoreCount = std::uint32_t(job.waits.size());
            wait.pSemaphores = job.waits.data();
            wait.pValues = job.values.data();
            for (;;) {
                const VkResult r = vk.vkWaitSemaphores(device_, &wait, kRetireWaitSliceNs);
                if (r == VK_SUCCESS) {
                    waited_on_host = true;
                    break;
                }
                if (r != VK_TIMEOUT || is_device_lost()) {
                    if (r == VK_ERROR_DEVICE_LOST) result = r;
                    break;  // not met on the host: the first call keeps the waits, as before
                }
            }
        }
        const auto t0 = std::chrono::steady_clock::now();
        const long long wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t0 - t_wait).count();
        // The chunks run in order through the binder's own chain timeline: the first waits for the job's waits (unless
        // met on the host above), each later one for the one before, and the last signals the job's ticket.
        const std::size_t chunks = std::max<std::size_t>(1, (job.ops.size() + kBindChunkImages - 1) / kBindChunkImages);
        for (std::size_t c = 0; c < chunks && result == VK_SUCCESS; ++c) {
            const std::size_t from = c * kBindChunkImages, to = std::min(job.ops.size(), from + kBindChunkImages);
            std::vector<VkSparseImageMemoryBindInfo> image_infos;
            std::vector<VkSparseImageOpaqueMemoryBindInfo> opaque_infos;
            for (std::size_t i = from; i < to; ++i) {
                const auto& op = job.ops[i];
                if (!op.image_binds.empty())
                    image_infos.push_back({op.image, std::uint32_t(op.image_binds.size()), op.image_binds.data()});
                if (!op.opaque_binds.empty())
                    opaque_infos.push_back({op.image, std::uint32_t(op.opaque_binds.size()), op.opaque_binds.data()});
            }
            std::vector<VkSemaphore> waits;
            std::vector<std::uint64_t> wait_values;
            if (c == 0) {
                if (!waited_on_host) {
                    waits = job.waits;
                    wait_values = job.values;
                }
            } else {
                waits.push_back(binder_.chain);
                wait_values.push_back(binder_.chain_value);
            }
            const bool last = c + 1 == chunks;
            const VkSemaphore signal = last ? binder_.semaphore : binder_.chain;
            const std::uint64_t signal_value = last ? job.ticket : ++binder_.chain_value;
            VkTimelineSemaphoreSubmitInfo tickets{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            tickets.waitSemaphoreValueCount = std::uint32_t(wait_values.size());
            tickets.pWaitSemaphoreValues = wait_values.data();
            tickets.signalSemaphoreValueCount = 1;
            tickets.pSignalSemaphoreValues = &signal_value;
            VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &tickets};
            info.waitSemaphoreCount = std::uint32_t(waits.size());
            info.pWaitSemaphores = waits.data();
            info.imageBindCount = std::uint32_t(image_infos.size());
            info.pImageBinds = image_infos.data();
            info.imageOpaqueBindCount = std::uint32_t(opaque_infos.size());
            info.pImageOpaqueBinds = opaque_infos.data();
            info.signalSemaphoreCount = 1;
            info.pSignalSemaphores = &signal;
            std::lock_guard heavy(driver_heavy_mutex_);  // never beside a host import (driver_heavy_mutex)
            result = vk.vkQueueBindSparse(binder_.queue, 1, &info, VK_NULL_HANDLE);
        }
        const std::size_t image_infos_size = image_info_count, opaque_infos_size = opaque_info_count;
        const long long ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        if (result == VK_ERROR_DEVICE_LOST) mark_device_lost();
        if (result != VK_SUCCESS) {
            static std::atomic<bool> told{false};
            if (!told.exchange(true))
                std::fprintf(stderr, "[rt] title pages: vkQueueBindSparse failed (%d); %zu images keep their old binds\n",
                             int(result), job.ops.size());
            // Nothing will signal the ticket on the GPU: signal it from the host once the batch before it is
            // signalled (a timeline only rises), or every queue's next submission waits for it forever.
            VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            const std::uint64_t before = job.ticket - 1;
            wait.semaphoreCount = 1;
            wait.pSemaphores = &binder_.semaphore;
            wait.pValues = &before;
            if (before) vk.vkWaitSemaphores(device_, &wait, UINT64_MAX);
            VkSemaphoreSignalInfo signal{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
            signal.semaphore = binder_.semaphore;
            signal.value = job.ticket;
            vk.vkSignalSemaphore(device_, &signal);
            finish_bind_job(job);
            return;
        }
        finish_bind_job(job);  // published only now: what is submitted from here on waits for a bind already made
        if (ms + wait_ms >= 300) {
            static std::atomic<int> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 200)
                std::fprintf(stderr,
                             "[rt] title pages: slow bind %lld ms on the bind queue (after %lld ms waiting on the host "
                             "for the GPU): %zu images, %zu image binds in %zu infos, %zu opaque binds in %zu infos, %zu "
                             "calls (ticket %llu)\n",
                             ms, wait_ms, job.ops.size(), image_binds, image_infos_size, opaque_binds, opaque_infos_size,
                             chunks, static_cast<unsigned long long>(job.ticket));
        }
    }

    void stop_binder() {
        if (!binder_.thread.joinable()) return;
        {
            std::lock_guard lock(binder_.mutex);
            binder_.stop = true;
        }
        binder_.cv.notify_one();
        binder_.thread.join();
    }

public:
    void note_upload(VkSemaphore semaphore, std::uint64_t value) {
        if (!semaphore || !value) return;
        std::lock_guard lock(storage_mutex_);
        auto it = std::find_if(upload_values_.begin(), upload_values_.end(),
                               [&](const auto& e) { return e.first == semaphore; });
        if (it == upload_values_.end()) upload_values_.emplace_back(semaphore, value);
        else it->second = std::max(it->second, value);
        ++upload_version_;
    }

private:
    std::unique_ptr<Depth24Converter> depth24_;
    std::unique_ptr<ScratchBuffers> scratch_;  // copies staged through a buffer (depth <-> colour)
    std::unique_ptr<IndirectExpander> indirect_;  // multi-argument command signatures
    std::function<void(bool)> bindless_commit_;
    std::unordered_map<QueryHeapId, QueryPool> query_pools_;  // element pointers stay valid across inserts
    double timestamp_period_ = 0;
    bool order_timestamp_copies_ = false;
    VkPhysicalDeviceMemoryProperties occlusion_resolve_memory_{};
    label::Completions gpu_labels_;
    ReplayFunctions replay_;
    std::unique_ptr<PredicateReducer> predicate_reducer_;
    std::unique_ptr<PredicatedBufferCopier> predicated_copier_;
    std::unique_ptr<IndirectDispatchCounter> counted_dispatch_;
    std::unique_ptr<QuadIndexExpander> quad_indices_;
    std::unique_ptr<IndirectArgumentClamp> indirect_clamp_;
    std::uint32_t predicate_family_ = UINT32_MAX;
    std::uint32_t transfer_only_family_ = UINT32_MAX;
    std::uint32_t graphics_family_ = UINT32_MAX;
    DebugLabelFunctions labels_;
};
#ifdef _MSC_VER
#pragma warning(pop)
#endif

// A Vulkan timeline semaphore with D3D12 fence semantics for the host side.
class VulkanFence {
public:
    VulkanFence(VulkanContext& context, std::uint64_t initial) : context_(context) {
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.initialValue = initial;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        if (context.vk.vkCreateSemaphore(context.device(), &info, nullptr, &semaphore_) != VK_SUCCESS)
            semaphore_ = VK_NULL_HANDLE;
        if (semaphore_) context.holds().add_fence(semaphore_, initial);
    }
    ~VulkanFence() {
        if (!semaphore_) return;
        context_.holds().remove_fence(semaphore_);  // batches held on it can never run: dropped and reported
        context_.vk.vkDestroySemaphore(context_.device(), semaphore_, nullptr);
    }
    VulkanFence(const VulkanFence&) = delete;
    VulkanFence& operator=(const VulkanFence&) = delete;

    VkSemaphore semaphore() const { return semaphore_; }

    std::uint64_t completed() const {
        if (context_.is_device_lost()) return UINT64_MAX;
        std::uint64_t value = 0;
        const VkResult result = context_.observe_result(
            context_.vk.vkGetSemaphoreCounterValue(context_.device(), semaphore_, &value), "vkGetSemaphoreCounterValue");
        if (result == VK_ERROR_DEVICE_LOST) {
            context_.mark_device_lost();
            return UINT64_MAX;
        }
        if (result != VK_SUCCESS) return 0;
        if (value == UINT64_MAX) {
            // A lost device can show itself here first: measured on NVIDIA 621.24 and reported on 617.14, the counter
            // query succeeds with UINT64_MAX (and vkWaitSemaphores with a zero timeout succeeds too) before any call
            // has returned VK_ERROR_DEVICE_LOST. vkDeviceWaitIdle is what then says so, and it is also what a driver
            // needs to have returned before vkGetDeviceFaultInfoEXT answers (LEONIDA_DEVICE_FAULT=1). When
            // vkDeviceWaitIdle succeeds, nothing is recorded and the value is returned as the driver gave it.
            VkResult idle = VK_SUCCESS;
            {
                std::lock_guard lock(context_.submit_mutex());
                idle = context_.observe_result(context_.vk.vkDeviceWaitIdle(context_.device()),
                                               "vkDeviceWaitIdle (a fence counter of UINT64_MAX)");
            }
            if (idle == VK_ERROR_DEVICE_LOST)
                context_.mark_device_lost(
                    "a fence's counter read UINT64_MAX and vkDeviceWaitIdle then returned VK_ERROR_DEVICE_LOST");
        }
        return value;
    }

    // ID3D12Fence::Signal from the CPU. Vulkan timelines only increase: a value at or below the current one is
    // ignored (D3D12 would set it), and the caller is told.
    bool signal(std::uint64_t value) {
        // The dispatcher may be testing a failed counter query. Do not call completed() here:
        // it marks the context lost on a query error and prevents the dispatcher from waking.
        if (context_.is_device_lost()) return false;
        std::uint64_t current = 0;
        if (context_.observe_result(context_.vk.vkGetSemaphoreCounterValue(context_.device(), semaphore_, &current),
                                    "vkGetSemaphoreCounterValue") == VK_SUCCESS &&
            value <= current)
            return false;
        VkSemaphoreSignalInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO};
        info.semaphore = semaphore_;
        info.value = value;
        const VkResult result = context_.observe_result(context_.vk.vkSignalSemaphore(context_.device(), &info),
                                                        "vkSignalSemaphore");
        if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
        if (result != VK_SUCCESS) return false;
        context_.holds().signaled(semaphore_, value);  // queues that held a batch for this value run it now
        context_.holds().pump_ready();
        return true;
    }

    // true when the value was reached within the timeout.
    bool wait(std::uint64_t value, std::uint64_t timeout_ns) const {
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &semaphore_;
        info.pValues = &value;
        const VkResult result = context_.observe_result(
            context_.vk.vkWaitSemaphores(context_.device(), &info, timeout_ns), "vkWaitSemaphores");
        if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
        return result == VK_SUCCESS;
    }

private:
    VulkanContext& context_;
    VkSemaphore semaphore_ = VK_NULL_HANDLE;
};

// One D3D12 command queue on one VkQueue. D3D12 lets one thread use a queue at a time per call and the COM wrapper
// serialises its calls; state_mutex_ additionally serialises them against the thread that releases a held batch
// (Wait-before-Signal, queue_holds.h), the only other thread that touches a queue.
//
// A Wait on a fence value no Signal has been submitted for yet holds the next batch (ExecuteCommandLists or Signal)
// on the CPU, with everything the queue is asked to do after it, instead of replaying it against layouts from before
// the work it waits for: Vulkan itself would accept the wait (timeline semaphores allow wait-before-signal), but the
// layout tracker follows replay order. Holding on the CPU also keeps a Wait and the Signal that satisfies it from
// landing on one VkQueue in the wrong order, which would never make progress on the GPU.
class VulkanQueue : private QueueHolds<VkSemaphore>::Holder {
public:
    enum class QueueKind : std::uint8_t { direct, compute, copy };

    // The most command buffers a queue holds at once. At that many, a caller that outruns the GPU waits for the oldest
    // submitted one to finish (before taking any context lock, see wait_for_room) instead of allocating without bound. The
    // cap holds while the GPU is slow too: a wait that times out is retried, in kQueuedWaitNs slices so a device lost
    // meanwhile ends it, and a device lost makes execute() submit nothing.
    static constexpr std::size_t kMaxQueuedCommandBuffers = 64;
    static constexpr std::uint64_t kQueuedWaitNs = 5'000'000'000;

    static constexpr VkPipelineStageFlags2 queue_supported_stages(QueueKind kind) noexcept {
        switch (kind) {
        case QueueKind::copy:
            return VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        case QueueKind::compute:
            return VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        case QueueKind::direct:
        default:
            return VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT;
        }
    }

    VulkanQueue(VulkanContext& context, VkQueue queue, std::uint32_t family, QueueKind kind = QueueKind::direct)
        : context_(context), queue_(queue), kind_(kind), family_(family) {
        auto& vk = context.vk;
        VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool.queueFamilyIndex = family;
        ok_ = vk.vkCreateCommandPool(context.device(), &pool, nullptr, &pool_) == VK_SUCCESS;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &type};
        ok_ = ok_ && vk.vkCreateSemaphore(context.device(), &info, nullptr, &progress_) == VK_SUCCESS;
        try {
            if (progress_)
                context_.track_queue(progress_, queue_, family_,
                                     kind_ == QueueKind::copy      ? "copy"
                                     : kind_ == QueueKind::compute ? "compute"
                                                                   : "direct");
            context_.holds().attach(this);
        } catch (...) {
            // A failed registration leaves no queue to run the destructor for these Vulkan handles.
            if (progress_) vk.vkDestroySemaphore(context.device(), progress_, nullptr);
            if (pool_) vk.vkDestroyCommandPool(context.device(), pool_, nullptr);
            throw;
        }
    }

    ~VulkanQueue() {
        auto& vk = context_.vk;
        // LEONIDA_DRED: a queue that never names itself (the device's staging queue) may later live at this
        // address; it must not report this queue's debug name.
        if (BreadcrumbRing* ring = context_.breadcrumbs_.get()) ring->label_queue(this, {});
        context_.holds().detach(this);  // no thread releases into this queue from here on
        report_held_at_teardown();
        context_.holds().pump_ready();  // presents queued behind what was just dropped can run
        bool lost = false;
        if (queue_) {
            std::lock_guard lock(context_.submit_mutex());
            for (; open_labels_; --open_labels_) context_.labels_.queue_end(queue_);
            const VkResult idle_result = context_.observe_result(vk.vkQueueWaitIdle(queue_), "vkQueueWaitIdle");
            if (idle_result == VK_SUCCESS) {
                if (progress_) context_.retire_queue_idle_locked(progress_);
            } else {
                lost = true;
            }
        }
        if (lost) context_.mark_device_lost();
        if (progress_) context_.untrack_queue(progress_);
        if (progress_) {
            std::lock_guard lock(context_.tracker_mutex_);
            if (context_.depth24_) context_.depth24_->forget(progress_);  // its scratch is idle now
            if (context_.scratch_) context_.scratch_->forget(progress_);
            if (context_.indirect_) context_.indirect_->forget(progress_);
            if (context_.indirect_clamp_) context_.indirect_clamp_->forget(progress_);
            // This queue is idle, so whatever initialisation (storage images, zero-filled buffers) or bindless table
            // upload it published has finished: clear both under tracker_mutex_ + storage_mutex_ so no other queue's
            // submission can wait on a semaphore that is about to go.
            std::lock_guard storage_lock(context_.storage_mutex_);
            if (context_.storage_init_semaphore_ == progress_) context_.storage_init_semaphore_ = VK_NULL_HANDLE;
            if (context_.bindless_upload_semaphore_ == progress_) context_.bindless_upload_semaphore_ = VK_NULL_HANDLE;
            if (context_.title_pages_semaphore_ == progress_) context_.title_pages_semaphore_ = VK_NULL_HANDLE;
        }
        if (pool_) vk.vkDestroyCommandPool(context_.device(), pool_, nullptr);
        if (progress_) context_.destroy_or_defer_progress_semaphore(progress_);
        destroy_khr_stages(khr_stage_pool_);
        destroy_khr_as_queries(khr_as_query_pool_);
        for (auto& slot : buffers_) {
            destroy_khr_stages(slot.khr_instances);
            destroy_khr_as_queries(slot.khr_as_queries);
            free_events_.insert(free_events_.end(), slot.events.begin(), slot.events.end());
        }
        for (VkEvent event : free_events_) vk.vkDestroyEvent(context_.device(), event, nullptr);
    }

    VulkanQueue(const VulkanQueue&) = delete;
    VulkanQueue& operator=(const VulkanQueue&) = delete;

    bool ok() const { return ok_; }

    // Entries kept between executions. Execute only uses the first lists.size(); a larger count is retained storage,
    // not work the next submission will replay (vk_queue_test checks both halves of that contract).
    std::size_t prepared_entries() const { return prepared_.size(); }

    template<typename GetHeap>
    bool update_tiles(ResourceId id, std::span<const TileRegion> regions,
                      std::span<const TileRange> ranges, GetHeap&& get_heap) {
        std::lock_guard state(state_mutex_);  // outermost: held batches are released from other threads
        std::unique_lock lock(context_.resource_mutex_);
        const auto it = context_.sparse_resources_.find(id);
        if (it == context_.sparse_resources_.end()) return false;
        const auto& sparse = it->second;
        for (const auto& region : regions)
            if (!valid_tiles(*sparse, region)) return false;
        const auto heap = get_heap(sparse->image != VK_NULL_HANDLE, sparse->memory_types);

        std::map<std::uint64_t, TileMapping> changes;
        std::size_t region_index = 0;
        std::uint64_t consumed = 0;
        for (const auto& range : ranges) {
            const bool backed = range.mode == TileRangeMode::sequential || range.mode == TileRangeMode::reuse;
            const std::uint64_t heap_count = heap ? heap->size / tile_size_in_bytes : 0;
            const std::uint64_t required = range.mode == TileRangeMode::reuse ? 1 : range.count;
            if (backed && (!heap || range.first > heap_count || required > heap_count - range.first)) return false;
            for (std::uint64_t i = 0; i < range.count;) {
                if (region_index == regions.size()) return false;
                const auto& region = regions[region_index];
                const auto count = std::min(range.count - i, region.count - consumed);
                const auto first = region.first + consumed;
                if (range.mode == TileRangeMode::unmap) {
                    const auto end = first + count;
                    for (auto old = sparse->mapped.lower_bound(first); old != sparse->mapped.end() && old->first < end; ++old)
                        changes[old->first] = {};
                    for (auto change = changes.lower_bound(first); change != changes.end() && change->first < end; ++change)
                        change->second = {};
                } else if (backed) {
                    for (std::uint64_t j = 0; j < count; ++j) {
                        const auto page = range.first + (range.mode == TileRangeMode::sequential ? i + j : 0);
                        changes[first + j] = {heap, heap->offset + page * tile_size_in_bytes};
                    }
                }
                i += count;
                consumed += count;
                if (consumed == region.count) {
                    ++region_index;
                    consumed = 0;
                }
            }
        }
        return region_index == regions.size() && bind_tiles(id, sparse, changes);
    }

    bool copy_tile_mappings(ResourceId dst, std::span<const TileRegion> dst_regions, ResourceId src,
                            std::span<const TileRegion> src_regions) {
        std::lock_guard state(state_mutex_);
        std::unique_lock lock(context_.resource_mutex_);
        const auto source = context_.sparse_resources_.find(src), target = context_.sparse_resources_.find(dst);
        if (source == context_.sparse_resources_.end() || target == context_.sparse_resources_.end() ||
            bool(source->second->image) != bool(target->second->image) || src_regions.size() != dst_regions.size())
            return false;
        // Read the complete source mapping before changing any destination slot, including copies within one resource.
        std::map<std::uint64_t, TileMapping> changes;
        for (std::size_t row = 0; row < src_regions.size(); ++row) {
            const auto from = src_regions[row], to = dst_regions[row];
            if (!valid_tiles(*source->second, from) || !valid_tiles(*target->second, to) || from.count != to.count)
                return false;
            for (std::uint64_t i = 0; i < from.count; ++i) {
                const auto page = source->second->mapped.find(from.first + i);
                changes.emplace(to.first + i, page == source->second->mapped.end() ? TileMapping{} : page->second);
            }
        }
        return bind_tiles(dst, target->second, changes);
    }

    // ExecuteCommandLists: every list is checked before any is written, so a failure submits nothing and leaves
    // the layout tracker as it was. Pending Wait()s are attached to this submission. When one of them (or an
    // earlier held operation) waits for a fence value no Signal has been submitted for, the lists are copied and
    // held instead, and the call succeeds: the batch is replayed by the thread that submits the Signal (or the CPU
    // Signal), which is also where its errors are reported (stderr, and take_held_failure()).
    //
    // `pin`, called only when the batch is held, returns what has to stay alive until the batch has run: the COM layer
    // passes the pipeline states and root signatures the lists named (the application may release them and Reset
    // the lists as soon as the call returns).
    using PinFn = std::function<std::shared_ptr<void>()>;

    ReplayResult execute(std::span<const CommandList* const> lists, const PinFn& pin = {}) {
        auto* diagnostics = context_.device_fault();
        const auto execute_call = diagnostics ? diagnostics->next_execute_call() : 0;
        ReplayResult result = ReplayResult::success;
        bool held = false;
        std::vector<std::shared_ptr<void>> retired;  // destroyed after state_mutex_ is released
        {
            std::lock_guard state(state_mutex_);
            if (!must_hold()) {
                result = execute_now(lists, execute_call);
            } else if (context_.is_device_lost()) {
                result = queue_unsupported(__LINE__);
            } else {
                Deferred op;
                op.kind = Deferred::Kind::batch;
                op.execute_call = execute_call;
                op.lists.reserve(lists.size());
                for (const CommandList* list : lists) op.lists.push_back(*list);
                if (pin) op.pins.push_back(pin());
                op.origins.assign(lists.begin(), lists.end());
                if (const BreadcrumbRing* ring = context_.breadcrumbs_.get())
                    for (const CommandList* list : lists) op.names.push_back(ring->names_of(list, this));
                hold(std::move(op));
                held = true;
            }
            drain_pins(retired);
        }
        retired.clear();
        if (held) context_.holds().pump_ready();  // a Signal may have been submitted since the check
        return result;
    }

    // What made the last ExecuteCommandLists this queue ran (now, or a held one when it was released) fail (I4):
    // the list and the command prepare refused (none: a whole-list or submission failure), and the VkResult of
    // the vkQueueSubmit when that was the failure. result is success when it did not fail.
    struct Failure {
        static constexpr std::size_t none = ~std::size_t(0);
        ReplayResult result = ReplayResult::success;
        std::size_t list = none;
        std::size_t command = none;
        const char* kind = nullptr;  // command_name of that command
        VkResult vk_result = VK_SUCCESS;
        const char* reason = nullptr;  // a whole-submission failure's step (query pools, bindless flush, ...)
    };
    Failure last_failure() {
        std::lock_guard state(state_mutex_);
        return failure_;
    }
    // "list 1, command 7 (CopyTextureRegion)", "the lists as a whole", plus ", vkQueueSubmit returned -4".
    static std::string describe(const Failure& f) {
        char text[160];
        if (f.command != Failure::none)
            std::snprintf(text, sizeof text, "list %zu, command %zu (%s)", f.list, f.command,
                          f.kind ? f.kind : "unknown");
        else if (f.list != Failure::none)
            std::snprintf(text, sizeof text, "list %zu as a whole", f.list);
        else
            std::snprintf(text, sizeof text, "no single command");
        std::string out = text;
        if (f.reason) {
            out += " (";
            out += f.reason;
            out += ")";
        }
        if (f.vk_result != VK_SUCCESS) {
            std::snprintf(text, sizeof text, ", vkQueueSubmit returned %d", static_cast<int>(f.vk_result));
            out += text;
        }
        return out;
    }

    // ID3D12CommandQueue::Signal: the fence reaches `value` after all work submitted so far. Held behind earlier
    // held operations, so the fence does not run ahead of the work it follows. `pin` keeps the fence object alive
    // until the submission that signals it has completed, as D3D12 does: the application may release the fence
    // right after the call, and destroying a semaphore with a signal still pending on the GPU is undefined
    // (VUID-vkDestroySemaphore-semaphore-05149).
    bool signal(VkSemaphore fence, std::uint64_t value, const PinFn& pin = {}) {
        bool ok = true;
        std::vector<std::shared_ptr<void>> retired;
        {
            std::lock_guard state(state_mutex_);
            const bool held = must_hold();
            detail_stall::last_signal_semaphore.store(fence, std::memory_order_relaxed);
            detail_stall::last_signal_value.store(value, std::memory_order_relaxed);
            detail_stall::last_signal_held.store(held, std::memory_order_relaxed);
            if (held) {
                Deferred op;
                op.kind = Deferred::Kind::signal;
                op.fence = fence;
                op.value = value;
                if (pin) op.pins.push_back(pin());
                hold(std::move(op));
            } else {
                ok = submit(VK_NULL_HANDLE, fence, value);
                if (ok && pin) {
                    std::vector<std::shared_ptr<void>> pins{pin()};
                    retire_pins(pins, submitted_);
                }
            }
            drain_pins(retired);
        }
        retired.clear();
        context_.holds().pump_ready();  // this Signal, or the hold above, may have made a held batch ready
        return ok;
    }

    // ID3D12CommandQueue::Wait: later submissions wait for the fence. A wait in a Vulkan batch only holds back that
    // batch, so it is kept and attached to the next submission (held on the CPU if no Signal reaches it yet).
    // `pin` returns what keeps the fence object alive until then (a D3D12 queue holds its fences).
    void wait(VkSemaphore fence, std::uint64_t value, const PinFn& pin = {}) {
        std::vector<std::shared_ptr<void>> retired;  // fences whose submission has completed: released unlocked
        {
            std::lock_guard state(state_mutex_);
            waits_.push_back(fence);
            wait_values_.push_back(value);
            if (pin) wait_pins_.push_back(pin());
            drain_pins(retired);
        }
    }

    // The first failure of a held ExecuteCommandLists since the last call (what the call would have returned had it
    // not been held), or success. A Signal behind a failed batch still signals, as when the failure is not held: the
    // work did not run, the application is told here and on stderr, and nothing waits forever.
    ReplayResult take_held_failure() {
        return static_cast<ReplayResult>(held_failure_.exchange(static_cast<int>(ReplayResult::success)));
    }

    // ID3D12CommandQueue::SetMarker/BeginEvent/EndEvent: queue labels, between the submissions around them. An end
    // with no begin open on this queue is dropped, as Vulkan requires a matching begin. No-ops without the extension.
    void marker(DebugMarker::Kind kind, const char* label) {
        std::vector<std::shared_ptr<void>> retired;
        std::lock_guard state(state_mutex_);
        drain_pins(retired);  // destroyed after the lock (reverse declaration order)
        if (!held_.empty()) {  // behind the held work, so behind it in the queue's labels too
            Deferred op;
            op.kind = Deferred::Kind::marker;
            op.marker_kind = kind;
            op.label = label ? label : "";
            hold(std::move(op));
            return;
        }
        marker_now(kind, label);
    }

    bool idle() {
        std::lock_guard lock(context_.submit_mutex());
        return context_.observe_result(context_.vk.vkQueueWaitIdle(queue_), "vkQueueWaitIdle") == VK_SUCCESS;
    }

    // Waits on the CPU until what this queue has submitted so far is done: its own timeline, without the device's
    // submit lock, so other queues keep submitting, and without waiting for anything submitted after (review D9;
    // idle() waits for the whole VkQueue, which D3D12 queues may share, and blocks every submission meanwhile).
    // Where the last execute_lists spent its time (the device's slow-submission log): waiting for room in the ring,
    // for compiling compute pipelines, preparing the lists, recording them (acquire through vkEndCommandBuffer), and
    // submitting.
    struct ExecuteTiming {
        std::uint32_t room_ms = 0, compute_wait_ms = 0, prepare_ms = 0, record_ms = 0, submit_ms = 0;
        std::uint32_t title_map_ms = 0;  // of record: mapping title memory into the page table (g_title_map_ns)
    };
    ExecuteTiming last_timing() const { return last_timing_; }
    ExecuteTiming last_timing_{};

    // This queue's timeline and the value of its latest submission (VulkanContext::note_upload).
    VkSemaphore progress_semaphore() const { return progress_; }
    std::uint64_t submitted_value() {
        std::lock_guard state(state_mutex_);
        return submitted_;
    }

    bool wait_submitted() {
        std::uint64_t value = 0;
        {
            std::lock_guard state(state_mutex_);
            value = submitted_;
        }
        if (!value) return true;
        VkSemaphoreWaitInfo wait{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        wait.semaphoreCount = 1;
        wait.pSemaphores = &progress_;
        wait.pValues = &value;
        for (;;) {  // bounded slices: a lost device wakes the wait
            if (context_.is_device_lost()) return false;
            const VkResult res =
                context_.vk.vkWaitSemaphores(context_.device(), &wait, VulkanContext::kRetireWaitSliceNs);
            if (res == VK_SUCCESS) return true;
            if (res == VK_TIMEOUT) continue;
            if (res == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
            return false;
        }
    }

private:
    // Completed stages, reused by later ARRAY_OF_POINTERS builds. Only execute_now() touches the pool, and it holds
    // state_mutex_ (including the deferred/held release paths), so no extra lock is needed.
    std::vector<KhrInstanceStage> khr_stage_pool_;
    std::vector<KhrAsQueryPool> khr_as_query_pool_;
    std::vector<TitleVaAliases::Mapping> xdxr_ranges_;  // stage_xdxr_table's snapshot, kept for its capacity

    void destroy_khr_stages(std::vector<KhrInstanceStage>& stages) {
        for (const auto& stage : stages) {
            if (stage.memory) context_.vk.vkFreeMemory(context_.device(), stage.memory, nullptr);
            if (stage.buffer) context_.vk.vkDestroyBuffer(context_.device(), stage.buffer, nullptr);
        }
        stages.clear();
    }

    // A stage returns to the pool only after its submission has completed, so rewriting it cannot race the GPU.
    void recycle_khr_stages(std::vector<KhrInstanceStage>& stages) {
        for (auto& stage : stages) khr_stage_pool_.push_back(std::move(stage));
        stages.clear();
    }

    void destroy_khr_as_queries(std::vector<KhrAsQueryPool>& pools) {
        if (context_.vk.vkDestroyQueryPool) {
            for (const auto& entry : pools) {
                if (entry.pool) context_.vk.vkDestroyQueryPool(context_.device(), entry.pool, nullptr);
            }
        }
        pools.clear();
    }

    void recycle_khr_as_queries(std::vector<KhrAsQueryPool>& pools) {
        for (auto& entry : pools) {
            entry.used = 0;
            khr_as_query_pool_.push_back(entry);
        }
        pools.clear();
    }

    bool allocate_khr_as_queries(std::uint32_t count, std::vector<KhrAsQueryPool>& active, VkQueryPool& out_pool,
                                 std::uint32_t& out_first) {
        if (!count || !context_.vk.vkCreateQueryPool || !context_.vk.vkDestroyQueryPool) return false;
        if (!active.empty()) {
            auto& back = active.back();
            if (back.used <= back.capacity && count <= back.capacity - back.used) {
                out_pool = back.pool;
                out_first = back.used;
                back.used += count;
                return true;
            }
        }
        for (auto it = khr_as_query_pool_.begin(); it != khr_as_query_pool_.end(); ++it) {
            if (it->capacity >= count) {
                KhrAsQueryPool entry = *it;
                khr_as_query_pool_.erase(it);
                entry.used = count;
                active.push_back(entry);
                out_pool = entry.pool;
                out_first = 0;
                return true;
            }
        }
        const std::uint32_t capacity = std::max<std::uint32_t>(count, 16u);
        VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        info.queryCount = capacity;
        VkQueryPool pool = VK_NULL_HANDLE;
        if (context_.vk.vkCreateQueryPool(context_.device(), &info, nullptr, &pool) != VK_SUCCESS || !pool)
            return false;
        active.push_back({pool, capacity, count});
        out_pool = pool;
        out_first = 0;
        return true;
    }

    void destroy_stage(KhrInstanceStage& stage) {
        if (stage.memory) context_.vk.vkFreeMemory(context_.device(), stage.memory, nullptr);
        if (stage.buffer) context_.vk.vkDestroyBuffer(context_.device(), stage.buffer, nullptr);
        stage = {};
    }

    // A host-visible, device-addressable stage of at least `bytes`: a completed one of the same usage from the pool
    // when one is large enough (a per-frame TLAS rebuild then reuses the same buffer and memory instead of making a
    // new vkAllocateMemory call every frame), a new one otherwise. False, with `stage` empty, when none can be made.
    bool acquire_stage(std::uint64_t bytes, VkBufferUsageFlags usage, KhrInstanceStage& stage) {
        const auto& funcs = context_.vk;
        stage = {};
        if (!funcs.vkCreateBuffer || !funcs.vkDestroyBuffer || !funcs.vkGetBufferMemoryRequirements ||
            !funcs.vkAllocateMemory || !funcs.vkFreeMemory || !funcs.vkBindBufferMemory || !bytes)
            return false;
        for (auto it = khr_stage_pool_.begin(); it != khr_stage_pool_.end(); ++it) {
            if (it->size >= bytes && it->usage == usage) {
                stage = std::move(*it);
                khr_stage_pool_.erase(it);
                return true;
            }
        }
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        if (funcs.vkCreateBuffer(context_.device(), &info, nullptr, &stage.buffer) != VK_SUCCESS) {
            stage = {};
            return false;
        }
        VkMemoryRequirements requirements{};
        funcs.vkGetBufferMemoryRequirements(context_.device(), stage.buffer, &requirements);
        const auto& memory = context_.occlusion_resolve_memory_;
        constexpr VkMemoryPropertyFlags host_flags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        std::uint32_t type = memory.memoryTypeCount;
        for (std::uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
            if (!(requirements.memoryTypeBits & (1u << i))) continue;
            if ((memory.memoryTypes[i].propertyFlags & host_flags) == host_flags) {
                type = i;
                break;
            }
        }
        VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &flags};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (type == memory.memoryTypeCount ||
            funcs.vkAllocateMemory(context_.device(), &allocation, nullptr, &stage.memory) != VK_SUCCESS ||
            funcs.vkBindBufferMemory(context_.device(), stage.buffer, stage.memory, 0) != VK_SUCCESS) {
            destroy_stage(stage);
            return false;
        }
        stage.size = bytes;
        stage.usage = usage;
        return true;
    }

    // The table an XDXR TLAS build translates its instances' BLAS addresses with (xdxr_build.comp blas_address):
    // every range of the title's memory the layer reaches through a copy (src/title_va_alias.h), as 24-byte entries
    // {title VA, end, shader VA} in title-VA order, staged like the KHR pointer lists and kept until the submission
    // completes. `count` 0 and `address` 0 when there are no ranges, or the stage could not be made.
    ReplayResult stage_xdxr_table(std::vector<KhrInstanceStage>& stages, VkDeviceAddress& address,
                                  std::uint32_t& count) {
        const auto& funcs = context_.vk;
        address = 0;
        count = 0;
        // Without a table the build still runs: a BLAS in the title's memory is then unreachable (an instance with
        // no BLAS), never read through heap_base + its title address.
        const auto none = [] {
            static std::atomic<bool> logged{false};
            if (!logged.exchange(true))
                std::fprintf(stderr, "[rt] d3d12: could not stage the title-memory table of an XDXR TLAS build; its "
                                     "instances in the title's memory get no BLAS\n");
            return ReplayResult::success;
        };
        // A member, not thread_local: a C++ thread_local with a destructor in gta6rt.exe adds TLS callbacks to the
        // image the loader's TLS index-0 move must not have (pe_loader.cpp reserve_tls_index0).
        auto& ranges = xdxr_ranges_;
        title_va_aliases().mappings(ranges);
        if (ranges.empty()) return ReplayResult::success;
        if (ranges.size() > 0xffffffffull / 24 || !funcs.vkMapMemory || !funcs.vkUnmapMemory ||
            !funcs.vkGetBufferDeviceAddress)
            return none();
        const std::uint64_t bytes = 24ull * ranges.size();
        KhrInstanceStage stage{};
        if (!acquire_stage(bytes, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                           stage))
            return none();
        void* mapped = nullptr;
        if (funcs.vkMapMemory(context_.device(), stage.memory, 0, bytes, 0, &mapped) != VK_SUCCESS) {
            destroy_stage(stage);
            return none();
        }
        auto* out = static_cast<std::uint8_t*>(mapped);
        for (const auto& range : ranges) {
            const std::uint64_t entry[3] = {range.title_va, range.end, range.shader_va};
            std::memcpy(out, entry, sizeof(entry));
            out += sizeof(entry);
        }
        funcs.vkUnmapMemory(context_.device(), stage.memory);
        if (!stage.address) {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = stage.buffer;
            stage.address = funcs.vkGetBufferDeviceAddress(context_.device(), &info);
        }
        if (!stage.address) {
            destroy_stage(stage);
            return none();
        }
        address = stage.address;
        count = std::uint32_t(ranges.size());
        stages.push_back(std::move(stage));
        return ReplayResult::success;
    }

    // A KHR TLAS built with D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS holds console VAs in its pointer list; Vulkan
    // reads those 8-byte entries as device addresses. Stage a translated copy in our own host-visible buffer instead
    // of editing the guest's input. The caller keeps the stage until the submission that builds with it completes.
    ReplayResult stage_khr_instances(const ReplayFunctions& vk, const BuildAccelerationStructure& build,
                                     std::vector<KhrInstanceStage>& stages, VkDeviceAddress& address) {
        const auto& funcs = context_.vk;
        if (!funcs.vkCreateBuffer || !funcs.vkDestroyBuffer || !funcs.vkGetBufferMemoryRequirements ||
            !funcs.vkAllocateMemory || !funcs.vkFreeMemory || !funcs.vkBindBufferMemory || !funcs.vkMapMemory ||
            !funcs.vkUnmapMemory || !funcs.vkGetBufferDeviceAddress || !vk.host_at || !vk.buffer_at ||
            !build.instance_count)
            return queue_unsupported(__LINE__);
        const std::uint64_t bytes = 8ull * build.instance_count;
        const std::uint8_t* guest = nullptr;
        const HostReadStatus status = vk.host_at(build.instances_va, bytes, guest);
        if (status == HostReadStatus::not_cpu_written) return queue_unsupported(__LINE__);
        if (status != HostReadStatus::ok) return ReplayResult::invalid_copy;

        KhrInstanceStage stage{};
        const auto discard = [&] { destroy_stage(stage); };
        if (!acquire_stage(bytes,
                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                           stage))
            return queue_unsupported(__LINE__);

        void* mapped = nullptr;
        if (funcs.vkMapMemory(context_.device(), stage.memory, 0, bytes, 0, &mapped) != VK_SUCCESS) {
            discard();
            return queue_unsupported(__LINE__);
        }
        // Validate and translate in one pass straight into the staging buffer; no second vector of addresses.
        for (std::uint32_t i = 0; i < build.instance_count; ++i) {
            std::uint64_t va = 0;
            std::memcpy(&va, guest + 8ull * i, sizeof(va));
            VkBuffer buffer = VK_NULL_HANDLE;
            VkDeviceSize offset = 0;
            std::uint64_t room = 64;  // one D3D12_RAYTRACING_INSTANCE_DESC
            if (!va || !vk.buffer_at(va, room, buffer, offset) || room < 64) {
                funcs.vkUnmapMemory(context_.device(), stage.memory);
                discard();
                return ReplayResult::invalid_copy;
            }
            // heap_base + va wraps for a buffer below the arena (an UPLOAD buffer's VA is its device address minus the
            // heap base), as every other VA-to-address conversion in the layer does; it must not be refused.
            const std::uint64_t device_address = vk.heap_base + va;
            std::memcpy(static_cast<std::uint8_t*>(mapped) + 8ull * i, &device_address, sizeof(device_address));
        }
        funcs.vkUnmapMemory(context_.device(), stage.memory);
        VkBufferDeviceAddressInfo device_address{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        device_address.buffer = stage.buffer;
        stage.address = funcs.vkGetBufferDeviceAddress(context_.device(), &device_address);
        if (!stage.address) {
            discard();
            return queue_unsupported(__LINE__);
        }
        stages.push_back(std::move(stage));
        address = stages.back().address;
        return ReplayResult::success;
    }

    // `execute_call`: the ExecuteCommandLists call's number for the device-fault diagnostics (0: none). For a held
    // batch (LEONIDA_DRED): `identity`, the application's lists its copies stand for (their addresses are the list
    // ids), and `held_names`, their debug names when the batch was held.
    // execute_lists, with what made it fail kept in failure_ (I4). state_mutex_ held.
    ReplayResult execute_now(std::span<const CommandList* const> lists, std::uint64_t execute_call,
                             std::span<const CommandList* const> identity = {},
                             std::span<const BreadcrumbRing::Names> held_names = {}) {
        failure_ = {};
        last_submit_result_ = VK_SUCCESS;
        const ReplayResult result = execute_lists(lists, execute_call, identity, held_names);
        if (result != ReplayResult::success) {
            failure_.result = result;
            failure_.vk_result = last_submit_result_;
        }
        return result;
    }

    // Before the lists are prepared: every compute pipeline their dispatches use that is still being compiled
    // (LEONIDA_ASYNC_PSO, LEONIDA_LAZY_COMPUTE) is started and waited for, within the budget below, instead of its
    // dispatches being left out at once: a left-out dispatch is missing work, not a missing frame. A pipeline still
    // not ready when the budget runs out has its dispatches left out, as before. LEONIDA_COMPUTE_WAIT=0: no waiting.
    void wait_for_pending_compute(std::span<const CommandList* const> lists) {
        static const bool enabled = [] {
            const char* v = std::getenv("LEONIDA_COMPUTE_WAIT");
            return !(v && v[0] == '0');
        }();
        if (!enabled) return;
        static std::mutex given_up_mutex;
        static std::unordered_set<std::uintptr_t> given_up;
        std::vector<std::uintptr_t> ids;
        {
            std::shared_lock lock(context_.pipeline_mutex_);  // the pipeline table's lock (upstream)
            for (const CommandList* list : lists)
                for (const auto& command : list->commands()) {
                    std::uintptr_t id = 0;
                    if (const auto* d = std::get_if<Dispatch>(&command)) id = d->pipeline;
                    else if (const auto* x = std::get_if<ExecuteIndirect>(&command);
                             x && x->kind == IndirectKind::dispatch)
                        id = x->pipeline;
                    if (!id) continue;
                    const auto it = context_.pipelines_.find(id);
                    if (it == context_.pipelines_.end() || !it->second.pending || it->second.stub) continue;
                    if (std::find(ids.begin(), ids.end(), id) != ids.end()) continue;
                    {
                        std::lock_guard g(given_up_mutex);
                        if (given_up.count(id)) continue;
                    }
                    request_compute(it->second);
                    ids.push_back(id);
                }
        }
        if (ids.empty()) return;
        // A budget, not a promise: waits that add up hold the title's render threads, and a loading-screen frame that
        // took over 30 s ended the run in the title's watchdog (uncached wave64-emulated compiles after the corpus was
        // re-translated: 23 s of waits in 15 s). Each submission waits at most 250 ms, and all of them together at
        // most 1.5 s in any 15 s window; past that a dispatch is left out (its compile goes on, and the pipeline cache
        // checkpoint keeps it for the next run). With 1 s and 4 s, the loading screen's slow-submission log
        // (2026-11-25) showed one ~1000 ms wait after another, each holding the queue lock and every thread behind it.
        using clock = std::chrono::steady_clock;
        static std::mutex budget_mutex;
        static clock::time_point window_start{};
        static clock::duration window_spent{};
        const auto start = clock::now();
        clock::duration allowed{};
        {
            std::lock_guard g(budget_mutex);
            if (start - window_start > std::chrono::seconds(15)) {
                window_start = start;
                window_spent = {};
            }
            const clock::duration left = std::chrono::milliseconds(1500) - window_spent;
            allowed = std::min<clock::duration>(std::chrono::milliseconds(250),
                                                left > clock::duration{} ? left : clock::duration{});
        }
        const auto deadline = start + allowed;
        std::size_t left_out = 0;
        for (const std::uintptr_t id : ids) {
            for (;;) {
                bool pending = false;
                {
                    std::shared_lock lock(context_.pipeline_mutex_);  // the pipeline table's lock (upstream)
                    const auto it = context_.pipelines_.find(id);
                    pending = it != context_.pipelines_.end() && it->second.pending;
                }
                if (!pending) break;
                if (clock::now() >= deadline) {
                    ++left_out;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        const clock::duration spent = clock::now() - start;
        {
            std::lock_guard g(budget_mutex);
            window_spent += spent;
        }
        static std::atomic<int> told{0};
        const double ms = std::chrono::duration<double, std::milli>(spent).count();
        if (told.fetch_add(1, std::memory_order_relaxed) < 16 || ms > 500)
            std::fprintf(stderr, "[rt] compute wait: %zu compiling pipeline(s), %.0f ms, %zu still compiling (left out)\n",
                         ids.size(), ms, left_out);
    }

    ReplayResult execute_lists(std::span<const CommandList* const> lists, std::uint64_t execute_call,
                               std::span<const CommandList* const> identity,
                               std::span<const BreadcrumbRing::Names> held_names) {
        using phase_clock = std::chrono::steady_clock;
        const auto phase_start = phase_clock::now();
        last_timing_ = {};
        const auto phase_ms = [](phase_clock::time_point a, phase_clock::time_point b) {
            return static_cast<std::uint32_t>(std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count());
        };
        {
            const submit_phases::Scope waited(submit_phases::wait_room);
            if (context_.is_device_lost() || !wait_for_room()) return queue_unsupported(__LINE__);
        }
        if (context_.before_execute_) context_.before_execute_();
        // The rest of this call is the replay phase; the submission is counted when it returns, however it returns.
        struct Replayed {
            std::int64_t start = submit_phases::now_ns();
            ~Replayed() {
                submit_phases::add(submit_phases::replay, std::uint64_t(submit_phases::now_ns() - start));
                submit_phases::submitted();
            }
        } replayed;
        if (context_.input_probe_)
            context_.input_probe_(lists, execute_call, kind_ == QueueKind::direct ? "direct" :
                                                      kind_ == QueueKind::compute ? "compute" : "copy");
        const auto room = phase_clock::now();
        wait_for_pending_compute(lists);
        last_timing_.room_ms = phase_ms(phase_start, room);
        last_timing_.compute_wait_ms = phase_ms(room, phase_clock::now());
        const auto prepare_start = phase_clock::now();
        context_.wait_title_binds(lists);  // copies into streamed textures whose bind is not made yet (in "prepare")
        if (kind_ != QueueKind::direct)
            for (const auto* list : lists)
                for (const auto& command : list->commands())
                    if (std::holds_alternative<ClearRenderTarget>(command) ||
                        std::holds_alternative<ClearDepthStencil>(command))
                        return queue_unsupported(__LINE__);
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        struct KhrGuard {
            VulkanQueue* queue;
            std::vector<KhrInstanceStage> stages;
            std::vector<KhrAsQueryPool> as_queries;
            ~KhrGuard() {
                queue->recycle_khr_stages(stages);
                queue->recycle_khr_as_queries(as_queries);
            }
        } khr_guard{this};
        std::function<void(bool)> bindless_commit;
        // The title memory the commands' root arguments name (detail::map_title_user_data, called again while
        // recording) is mapped here, before resource_mutex_: its host imports took up to 3.7 s of one submission's
        // recording (run 16, 2026-11-25), with the mutex held shared, so every resource creation (which takes it
        // exclusively) and then every other queue's recording waited behind it. The addresses are in the recorded
        // commands themselves; the record-time calls then find them mapped.
        if (context_.replay_.map_title) {
            std::unordered_set<std::uint64_t> seen;
            const auto map_pairs = [&](const auto& user_data) {
                for (std::size_t i = 0; i + 1 < user_data.size(); ++i) {
                    const std::uint64_t va = std::uint64_t(user_data[i]) | (std::uint64_t(user_data[i + 1]) << 32);
                    if (va >> 32 == 0 || va >> 47) continue;
                    if (seen.insert(va).second) context_.replay_.map_title(va);
                }
            };
            for (const CommandList* list : lists)
                for (const auto& command : list->commands()) {
                    if (const auto* draw = std::get_if<Draw>(&command)) {
                        map_pairs(draw->user_data);
                    } else if (const auto* dispatch = std::get_if<Dispatch>(&command)) {
                        map_pairs(dispatch->user_data);
                    } else if (const auto* ind = std::get_if<ExecuteIndirect>(&command);
                               ind && ind->kind == IndirectKind::dispatch && ind->pipeline) {
                        map_pairs(ind->user_data);
                    }
                }
        }
        // Lock hierarchy: state_mutex_ -> pipeline_mutex_ -> resource_mutex_ -> tracker_mutex_ -> submit_mutex_.
        // Shared pipeline_lock keeps the registered pipelines (add/remove_pipeline take it uniquely) alive through
        // prepare and recording (upstream's; our pipeline states compile in the background, so it is never held
        // across a compile here). Shared read-lock allows concurrent prepare() across queues; tracker_mutex_ is
        // acquired lazily during prepare() only if a list actually converts a depth24_as_float copy, and otherwise right
        // before bindless_prepare_/emit() through submit() so bindless flushes, layout emits, and submissions stay in
        // order.
        std::shared_lock pipeline_lock(context_.pipeline_mutex_);
        std::shared_lock resource_lock(context_.resource_mutex_);
        std::unique_lock tracker_lock(context_.tracker_mutex_, std::defer_lock);
        // Declared after tracker_lock so any early return after depth24_->prepare() retires uncommitted scratch/pools
        // with ticket 0 while tracker_lock is still held.
        struct Depth24Guard {
            Depth24Converter* d24 = nullptr;
            VkSemaphore progress = VK_NULL_HANDLE;
            bool active = false;
            ~Depth24Guard() {
                if (active && d24) d24->retire(progress, 0);
            }
        } depth24_guard{context_.depth24_.get(), progress_, false};
        struct ScratchGuard {  // the same for the copy scratch
            ScratchBuffers* scratch = nullptr;
            VkSemaphore progress = VK_NULL_HANDLE;
            bool active = false;
            ~ScratchGuard() {
                if (active && scratch) scratch->retire(progress, 0);
            }
        } scratch_guard{context_.scratch_.get(), progress_, false};
        struct IndirectGuard {  // and for the generated-commands scratch
            IndirectExpander* indirect = nullptr;
            VkSemaphore progress = VK_NULL_HANDLE;
            bool active = false;
            ~IndirectGuard() {
                if (active && indirect) indirect->retire(progress, 0);
            }
        } indirect_guard{context_.indirect_.get(), progress_, false};
        struct ClampGuard {  // and for the ExecuteIndirect clamp's record scratch
            IndirectArgumentClamp* clamp = nullptr;
            VkSemaphore progress = VK_NULL_HANDLE;
            bool active = false;
            ~ClampGuard() {
                if (active && clamp) clamp->retire(progress, 0);
            }
        } clamp_guard{context_.indirect_clamp_.get(), progress_, false};
        const bool transfer_only = family_ == context_.transfer_only_family_;
        if (!transfer_only) bindless_commit = context_.bindless_commit_;
        // Kept between executions: prepare refills each list's entries in the storage the last execution left, where
        // making them fresh for every command was about half the layer's CPU cost per command (replay_cpu_bench).
        auto& prepared = prepared_;
        // Keep the storage from a larger batch. resize() down would destroy the unused entries and their vectors, so
        // a queue that sees 8 lists and then 1 in every frame would rebuild them on each execution. The loops below
        // use only the first lists.size() entries, so stale entries are never prepared, queried or emitted.
        if (prepared.size() < lists.size()) prepared.resize(lists.size());
        ReplayFunctions replay = context_.replay_;
        // Provisional, so that prepare refuses set-0 pipelines on a device without one; re-read after the flush below.
        // The resource lock held from here keeps bindless_current_ in place.
        replay.bindless_set = context_.bindless_current_ ? context_.bindless_current_() : VK_NULL_HANDLE;
        unsigned mesh_draws = 0;
        if (pipeline_profile::enabled()) replay.mesh_draw_count = &mesh_draws;
        std::vector<VulkanContext::GeometryUpload> geometry;
        if (context_.geometry_provider_) {
            replay.title_memory = [&](std::uint64_t va, std::uint64_t bytes, VkBuffer& buffer, VkDeviceSize& offset) {
                if (!va || !bytes || bytes>UINT64_MAX-va) return false;
                for (const auto& upload : geometry) {
                    if (va < upload.source || va-upload.source > upload.size || bytes > upload.size-(va-upload.source)) continue;
                    buffer=upload.buffer; offset=upload.offset+(va-upload.source);
                    return true;
                }
                VulkanContext::GeometryUpload upload;
                if (!context_.geometry_provider_(va,bytes,upload)) return false;
                buffer=upload.buffer;offset=upload.offset;
                geometry.push_back(std::move(upload));
                return true;
            };
        }
        replay.graphics_family = family_ == context_.graphics_family_;
        replay.queue_stages = kind_ == QueueKind::copy      ? detail::queue_stages_copy
                              : kind_ == QueueKind::compute ? detail::queue_stages_compute
                                                            : 0;
        replay.prepare_khr_instances = [this, &replay, &khr_guard](const BuildAccelerationStructure& build,
                                                                   VkDeviceAddress& address) {
            return stage_khr_instances(replay, build, khr_guard.stages, address);
        };
        replay.prepare_xdxr_table = [this, &khr_guard](VkDeviceAddress& address, std::uint32_t& count) {
            return stage_xdxr_table(khr_guard.stages, address, count);
        };
        if (replay.write_acceleration_structures_properties) {
            replay.prepare_khr_postbuild = [this, &khr_guard](std::uint32_t count, VkQueryPool& pool,
                                                              std::uint32_t& first) {
                return allocate_khr_as_queries(count, khr_guard.as_queries, pool, first);
            };
        }
        std::uint64_t captures = 0;
        for (const auto* list : lists)
            for (const auto& command : list->commands())
                if (const auto* p = std::get_if<Predication>(&command); p && p->buffer) ++captures;
        std::unique_ptr<PredicateReducer::Batch> batch;
        std::unique_ptr<PredicatedBufferCopier::Batch> copy_batch;
        std::unique_ptr<IndirectDispatchCounter::Batch> counted_batch;
        std::unique_ptr<IndirectArgumentClamp::Batch> clamp_batch;
        if (captures) {
            if (kind_ == QueueKind::copy || !context_.predicate_reducer_ || family_ != context_.predicate_family_)
                return queue_unsupported(__LINE__);
            // The helper's lazy pipeline is shared by queues. Serialize preparation with the tracker lock,
            // retaining main's resource -> tracker -> submit lock order through emit and submission.
            if (!tracker_lock.owns_lock()) tracker_lock.lock();
            batch = context_.predicate_reducer_->batch(captures);
            if (!batch) return queue_unsupported(__LINE__);
            replay.prepare_predicate = [&](VkBuffer buffer, VkDeviceSize offset, PredicateCapture& out) {
                return batch->prepare(buffer, offset, out);
            };
            replay.capture_predicate = [&](VkCommandBuffer cb, const PredicateCapture& p) {
                context_.predicate_reducer_->capture(cb, p);
            };
            if (context_.predicated_copier_) {
                copy_batch = context_.predicated_copier_->batch();
                replay.prepare_predicated_copy = [&](VkBuffer source, VkBuffer destination, VkDeviceSize src,
                                                     VkDeviceSize dst, VkDeviceSize size,
                                                     std::vector<PredicatedCopyDispatch>& out) {
                    return copy_batch->prepare(source, destination, src, dst, size, out);
                };
                replay.emit_predicated_copy = [&](VkCommandBuffer cb, const std::vector<PredicatedCopyDispatch>& jobs) {
                    context_.predicated_copier_->emit(cb, jobs);
                };
            }
        }
        if (context_.scratch_) {
            replay.copy_scratch = [this, &tracker_lock, &scratch_guard](VkDeviceSize bytes, VkBuffer& buffer,
                                                                        VkDeviceSize& offset) {
                if (!tracker_lock.owns_lock()) tracker_lock.lock();
                if (!scratch_guard.active) {
                    context_.scratch_->collect();
                    scratch_guard.active = true;
                }
                return context_.scratch_->get(bytes, buffer, offset);
            };
        }
        if (kind_ != QueueKind::copy && context_.indirect_clamp_) {
            // The record scratch belongs to this submission: collected here, retired below with its ticket. The
            // counter slots are read back once that ticket completes (the batch's destructor).
            replay.prepare_indirect_clamp = [this, &tracker_lock, &clamp_batch, &clamp_guard](
                                                VkBuffer arguments, VkDeviceSize argument_offset, VkBuffer count,
                                                VkDeviceSize count_offset, std::uint32_t kind, std::uint32_t stride,
                                                std::uint32_t max_count, ClampedIndirect& out) {
                // Pipeline creation is lazy and shared across queues; serialize it with other replay preparation.
                if (!tracker_lock.owns_lock()) tracker_lock.lock();
                if (!clamp_batch) clamp_batch = context_.indirect_clamp_->batch();
                if (!clamp_guard.active) {
                    context_.indirect_clamp_->collect();
                    clamp_guard.active = true;
                }
                return clamp_batch->prepare(arguments, argument_offset, count, count_offset, kind, stride, max_count,
                                            out);
            };
        }
        if (kind_ != QueueKind::copy && context_.counted_dispatch_) {
            replay.prepare_counted_dispatch = [this, &tracker_lock, &counted_batch](
                VkBuffer args, VkDeviceSize args_offset, VkBuffer count, VkDeviceSize count_offset,
                std::uint32_t stride, std::uint32_t maximum, CountedDispatch& out) {
                // Pipeline creation is lazy and shared across queues; serialize it with other replay preparation.
                if (!tracker_lock.owns_lock()) tracker_lock.lock();
                if (!counted_batch) counted_batch = context_.counted_dispatch_->batch();
                return counted_batch->prepare(args, args_offset, count, count_offset, stride, maximum, out);
            };
            replay.emit_counted_dispatch = [this](VkCommandBuffer cb, const CountedDispatch& job) {
                context_.counted_dispatch_->emit(cb, job);
            };
        }
        if (kind_ != QueueKind::copy && context_.indirect_) {
            replay.indirect_prepare = [this, &tracker_lock, &indirect_guard](const IndirectExpander::Request& request,
                                                                             IndirectPlan& plan) {
                if (!tracker_lock.owns_lock()) tracker_lock.lock();
                if (!indirect_guard.active) {
                    context_.indirect_->collect();
                    indirect_guard.active = true;
                }
                return context_.indirect_->prepare(request, plan);
            };
            replay.indirect_prepass = [this](VkCommandBuffer cb, const IndirectPlan& plan) {
                context_.indirect_->emit_prepass(cb, plan);
            };
            replay.indirect_execute = [this](VkCommandBuffer cb, const IndirectPlan& plan) {
                context_.indirect_->emit_execute(cb, plan);
            };
        }
        if (kind_ != QueueKind::copy && context_.quad_indices_ && context_.scratch_) {
            replay.prepare_quad_indices = [this, &tracker_lock](
                VkBuffer indices, VkDeviceSize offset, VkIndexType type, std::uint64_t readable, std::uint32_t quads,
                VkBuffer scratch, VkDeviceSize scratch_offset, QuadIndexExpansion& out) {
                // Pipeline creation is lazy and shared across queues; serialize it with other replay preparation.
                if (!tracker_lock.owns_lock()) tracker_lock.lock();
                return context_.quad_indices_->prepare(indices, offset, type, readable, quads, scratch,
                                                       scratch_offset, out);
            };
            replay.emit_quad_indices = [this](VkCommandBuffer cb, const QuadIndexExpansion& job) {
                context_.quad_indices_->emit(cb, job);
            };
        }
        if (context_.depth24_ && !transfer_only) {
            replay.depth24_prepare = [this, &tracker_lock, &depth24_guard](Depth24Job& job,
                                                                           std::vector<Depth24Job>& parts) {
                if (!tracker_lock.owns_lock()) tracker_lock.lock();
                if (!depth24_guard.active) {
                    context_.depth24_->collect();
                    depth24_guard.active = true;
                }
                return context_.depth24_->prepare(job, parts);
            };
        }
        const auto& occlusion_memory = context_.occlusion_resolve_memory_;
        const bool can_resolve_occlusion = occlusion_memory.memoryTypeCount != 0;
        if (can_resolve_occlusion) {
            replay.sum_occlusion = [this](VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                          std::uint32_t count, VkBuffer dst, VkDeviceSize offset) {
                occlusion_resolve_->sum(cb, pool, first, count, dst, offset);
            };
            replay.resolve_binary_occlusion = [this](VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                                      std::uint32_t count, VkBuffer dst, VkDeviceSize offset) {
                occlusion_resolve_->normalize(cb, pool, first, count, dst, offset);
            };
            replay.sum_statistics = [this](VkCommandBuffer cb, VkQueryPool pool, VkQueryPool mesh_pool,
                                           VkQueryPool primitives_pool, const std::uint8_t* mesh,
                                           std::uint32_t first, std::uint32_t count, VkBuffer dst,
                                           VkDeviceSize offset) {
                occlusion_resolve_->sum_statistics(cb, pool, mesh_pool, primitives_pool, mesh, first, count, dst,
                                                  offset);
            };
            replay.sum_binary_occlusion = [this](VkCommandBuffer cb, VkQueryPool pool, std::uint32_t first,
                                                 std::uint32_t count, VkBuffer dst, VkDeviceSize offset) {
                occlusion_resolve_->sum_binary(cb, pool, first, count, dst, offset);
            };
        }
        const auto resolve = [this](ResourceId id) { return context_.resolve(id); };
        std::uint32_t segments = 0, statistics_segments = 0;
        bool has_queries = false;
        // With workers, the pipelines the lists need and that are not built yet are compiled together below instead of
        // one after another in prepare(). resource_lock is held until they are all built, which keeps the pipeline
        // states that own them alive (remove_pipeline takes it uniquely).
        PipelineCompiles compiles;
        if (context_.pipeline_workers_) replay.compiles = &compiles;
        ReplayResult prepare_result = ReplayResult::success;
        for (std::size_t i = 0; i < lists.size(); ++i) {
            const auto result = prepare(*lists[i], resolve, replay, prepared[i], can_resolve_occlusion);
            // A failure stops preparing but still goes through build_pipelines below, which finishes the compiles the
            // earlier lists started before the result is returned.
            if (result != ReplayResult::success) {
                failure_.list = i;
                failure_.command = prepared[i].failed_command;
                const auto commands = lists[i]->commands();
                if (failure_.command < commands.size()) failure_.kind = command_name(commands[failure_.command]);
                prepare_result = result;
                break;
            }
            if (prepared[i].segment_count > UINT32_MAX - segments ||
                prepared[i].statistics_segment_count > UINT32_MAX - statistics_segments) {
                failure_.reason = "query segment count overflow";
                prepare_result = queue_unsupported(__LINE__);
                break;
            }
            segments += prepared[i].segment_count;
            statistics_segments += prepared[i].statistics_segment_count;
            has_queries = has_queries || !prepared[i].queries.empty();
        }
        prepare_result = build_pipelines(
            compiles, std::span(prepared).first(lists.size()), prepare_result,
            [this](std::size_t count, const std::function<void(std::size_t)>& job) {
                context_.pipeline_workers_->run(count, job);
            });
        if (prepare_result != ReplayResult::success) return prepare_result;
        const auto segment_pool = [this](VkQueryType type, std::uint32_t count,
                                         std::shared_ptr<OcclusionSegmentPool>& out) {
            const auto& vk = context_.vk;
            if (!vk.vkCreateQueryPool || !vk.vkDestroyQueryPool) return false;
            out = std::make_shared<OcclusionSegmentPool>();
            out->device = context_.device();
            out->destroy = vk.vkDestroyQueryPool;
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = type;
            info.queryCount = count;
            // The counters of a D3D12 pipeline-statistics heap, in D3D12_QUERY_DATA_PIPELINE_STATISTICS order.
            if (type == VK_QUERY_TYPE_PIPELINE_STATISTICS) info.pipelineStatistics = 0x7ff;
            return vk.vkCreateQueryPool(out->device, &info, nullptr, &out->pool) == VK_SUCCESS;
        };
        std::shared_ptr<OcclusionSegmentPool> physical, statistics;
        if (segments && !segment_pool(VK_QUERY_TYPE_OCCLUSION, segments, physical)) {
            failure_.reason = "occlusion query pool";
            return ReplayResult::unsupported;
        }
        // Without the compute helper a split query could not be summed: leave statistics queries whole, as before
        // (QueryReplay::plan refuses a list whose statistics query holds a mesh draw then).
        if (statistics_segments && can_resolve_occlusion) {
            if (!segment_pool(VK_QUERY_TYPE_PIPELINE_STATISTICS, statistics_segments, statistics)) {
                failure_.reason = "statistics query pool";
                return ReplayResult::unsupported;
            }
            for (const auto& list : prepared)
                statistics->mesh.insert(statistics->mesh.end(), list.statistics_mesh.begin(),
                                        list.statistics_mesh.end());
            if (std::find(statistics->mesh.begin(), statistics->mesh.end(), 1) == statistics->mesh.end()) {
                statistics->mesh.clear();
            } else {
                // The counters a mesh draw allows (VUID-vkCmdDrawMeshTasksEXT-pipelineStatistics-07076): the
                // fragment and compute invocations, and no other (D3D-291).
                VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                info.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
                info.queryCount = statistics_segments;
                info.pipelineStatistics = kMeshStatisticsFlags;
                if (context_.vk.vkCreateQueryPool(statistics->device, &info, nullptr, &statistics->mesh_pool) !=
                    VK_SUCCESS)
                    return queue_unsupported(__LINE__);
                // The mesh-primitives count of a mesh segment (D3D-MESH-CLIPPING-COUNTER): a query of its own type,
                // which the pipeline-statistics pool cannot hold. The device needs the meshShaderQueries feature,
                // or the clipping field of a mesh segment stays 0.
                if (context_.mesh_queries) {
                    VkQueryPoolCreateInfo primitives{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
                    primitives.queryType = VK_QUERY_TYPE_MESH_PRIMITIVES_GENERATED_EXT;
                    primitives.queryCount = statistics_segments;
                    if (context_.vk.vkCreateQueryPool(statistics->device, &primitives, nullptr,
                                                      &statistics->primitives_pool) != VK_SUCCESS)
                        return queue_unsupported(__LINE__);
                }
            }
        }
        // QueryReplay::plan() reads QueryPool::ended / occlusion and QueryReplay::commit() writes them after
        // resource_lock is unlocked. Hold tracker_mutex_ across plan -> emit -> submit -> commit so concurrent queues
        // see submission-order query state and remove_query_pool() waits until commit finishes.
        if (has_queries && !tracker_lock.owns_lock()) tracker_lock.lock();
        QueryReplay queries;
        std::uint32_t base = 0, statistics_base = 0;
        for (std::size_t i = 0; i < lists.size(); ++i) {
            auto& list = prepared[i];
            const auto result = queries.plan(list, physical, base, can_resolve_occlusion, statistics, statistics_base);
            if (result != ReplayResult::success) {
                failure_.list = i;
                failure_.reason = "query plan";
                return result;
            }
            base += list.segment_count;
            statistics_base += list.statistics_segment_count;
        }
        if (queries.needs_sum || queries.needs_binary || queries.needs_statistics_sum || queries.needs_sum_binary) {
            if (!occlusion_resolve_) {
                auto helper = std::make_unique<OcclusionResolve>(context_.vk, context_.device());
                if (!helper->initialize(occlusion_memory)) {
                    failure_.reason = "occlusion resolve helper";
                    return ReplayResult::unsupported;
                }
                occlusion_resolve_ = std::move(helper);
            }
            if (!occlusion_resolve_->prepare(queries.needs_sum, queries.needs_binary, queries.needs_statistics_sum,
                                             queries.needs_sum_binary)) {
                failure_.reason = "occlusion resolve prepare";
                return ReplayResult::unsupported;
            }
        }

        last_timing_.prepare_ms = phase_ms(prepare_start, phase_clock::now());
        const auto record_start = phase_clock::now();
        const std::uint64_t title_map_before = g_title_map_ns;
        CommandSlot* slot = acquire();
        if (!slot) {
            failure_.reason = "no command slot";
            return ReplayResult::unsupported;
        }
        slot->khr_instances = std::move(khr_guard.stages);  // freed when this slot is reused
        khr_guard.stages.clear();
        slot->khr_as_queries = std::move(khr_guard.as_queries);
        khr_guard.as_queries.clear();
        command_buffer = slot->buffer;
        // MoltenVK writes a timestamp at the end of a Metal encoder and resolves it with a blit that nothing orders
        // after that write, so a copy in the same or the next command buffer reads the old value (docs section 4.6).
        // An event set and waited for in this command buffer orders it after this submission's timestamps; a wait for
        // this queue's previous submission orders it after earlier ones still running.
        bool ordered_timestamps = false;
        const auto& events = context_.vk;
        if (context_.order_timestamp_copies() && events.vkCreateEvent && events.vkDestroyEvent && events.vkResetEvent &&
            events.vkCmdSetEvent && events.vkCmdWaitEvents)
            replay.order_timestamp_copy = [this, slot, &ordered_timestamps](VkCommandBuffer cb) {
                const VkEvent event = take_event();
                if (!event) return;
                slot->events.push_back(event);
                const auto& vk = context_.vk;
                vk.vkCmdSetEvent(cb, event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
                vk.vkCmdWaitEvents(cb, 1, &event, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, nullptr, 0, nullptr, 0, nullptr);
                ordered_timestamps = true;
            };
        const auto discard = [&] {
            if (context_.vk.vkResetCommandBuffer(command_buffer, 0) == VK_SUCCESS) slot->ticket = 0;
        };
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        context_.gpu_labels_.begin_record(command_buffer);
        if (context_.observe_result(context_.vk.vkBeginCommandBuffer(command_buffer, &begin),
                                    "vkBeginCommandBuffer") != VK_SUCCESS) {
            discard();
            failure_.reason = "vkBeginCommandBuffer";
            return queue_unsupported(__LINE__);
        }
        // D3D12 ends every ExecuteCommandLists with all its work done and its writes visible to the next one on the
        // queue (implicit state decay relies on it), and a list may then touch the same memory without a barrier:
        // two submissions copying into one buffer, say. Vulkan orders submissions on a queue only by what their
        // barriers say, so this one starts with a full memory barrier against everything submitted before it
        // (synchronization validation: WRITE_AFTER_WRITE between vk_queue_test's back-to-back copies).
        if (context_.vk.vkCmdPipelineBarrier) {  // a mocked context may leave it out
            VkMemoryBarrier full{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            full.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            full.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &full, 0, nullptr, 0, nullptr);
        }
        if (!tracker_lock.owns_lock()) tracker_lock.lock();
        if (!transfer_only && context_.bindless_prepare_ && !context_.bindless_prepare_(command_buffer)) {
            // What the failed flush did record (the fallback images' first initialization, say) still goes out with
            // this submission, and other queues wait for it as for any other.
            const bool uploaded = std::exchange(context_.bindless_upload_pending_, false);
            const bool ended = context_.observe_result(context_.vk.vkEndCommandBuffer(command_buffer),
                                                      "vkEndCommandBuffer") == VK_SUCCESS;
            resource_lock.unlock();
            pipeline_lock.unlock();
            const bool submitted = ended && submit(command_buffer, VK_NULL_HANDLE, 0);
            if (!submitted) discard();
            if (submitted && uploaded) publish_bindless_upload();
            if (depth24_guard.active) {
                context_.depth24_->retire(progress_, 0);
                depth24_guard.active = false;
            }
            if (scratch_guard.active) {
                context_.scratch_->retire(progress_, 0);
                scratch_guard.active = false;
            }
            if (indirect_guard.active) {
                context_.indirect_->retire(progress_, 0);
                indirect_guard.active = false;
            }
            if (clamp_guard.active) {
                context_.indirect_clamp_->retire(progress_, 0);
                clamp_guard.active = false;
            }
            if (bindless_commit) bindless_commit(submitted);
            tracker_lock.unlock();
            failure_.reason = "bindless table flush";
            return queue_unsupported(__LINE__);
        }
        replay.bindless_set = context_.bindless_current_ ? context_.bindless_current_() : VK_NULL_HANDLE;
        const bool bindless_uploaded = std::exchange(context_.bindless_upload_pending_, false);
        // Layout changes go into the tracker itself, journalled (review I2: no copy of every tracked image), and are
        // undone unless the submission reaches the queue (tracker_mutex_ is held from here to the commit, so nothing
        // else sees them meanwhile). The transaction also rolls back on every early return below.
        auto& next_tracker = context_.tracker_;
        next_tracker.begin();
        struct TrackerTransaction {
            LayoutTracker& tracker;
            bool done = false;
            ~TrackerTransaction() {
                if (!done) tracker.rollback();
            }
        } tracker_transaction{next_tracker};
        // Bindless images that may already be visible to shaders are initialized on a submission that runs shaders.
        // initialize() zeroes committed storage images and only transitions sparse images.
        bool runs_shaders = false;
        const DebugMarker* last_marker = nullptr;  // named by the device-lost summary
        for (const CommandList* list : lists)
            for (const auto& command : list->commands()) {
                runs_shaders |= std::holds_alternative<Dispatch>(command) || draw_command(command).state ||
                                std::holds_alternative<ExecuteIndirect>(command);
                if (const auto* m = std::get_if<DebugMarker>(&command); m && m->kind != DebugMarker::Kind::end)
                    last_marker = m;
            }
        submit_marker_ = last_marker ? last_marker->label : std::string();
        std::vector<ResourceId> storage_images;
        std::vector<VulkanContext::PendingZero> zero_buffers;
        {
            std::lock_guard lock(context_.storage_mutex_);
            if (context_.storage_init_semaphore_ && context_.storage_init_semaphore_ != progress_ &&
                context_.storage_init_version_ != storage_init_seen_) {
                waits_.push_back(context_.storage_init_semaphore_);
                wait_values_.push_back(context_.storage_init_value_);
                storage_init_seen_ = context_.storage_init_version_;
            }
            // The set this submission binds may be a new generation whose table another queue's submission uploads,
            // or the same one with entries that submission adds in place: wait for it, as for the storage images
            // (the table's buffers are used by one queue family only: every D3D12 queue of the device is made in
            // info.queue_family).
            if (context_.bindless_upload_semaphore_ && context_.bindless_upload_semaphore_ != progress_ &&
                context_.bindless_upload_version_ != bindless_upload_seen_) {
                waits_.push_back(context_.bindless_upload_semaphore_);
                wait_values_.push_back(context_.bindless_upload_value_);
            }
            bindless_upload_seen_ = context_.bindless_upload_version_;
            // The title's textures on shared pages: the last batch of binds another queue made (flush_title_pages).
            if (context_.title_pages_semaphore_ && context_.title_pages_semaphore_ != progress_ &&
                context_.title_pages_version_ != title_pages_seen_) {
                waits_.push_back(context_.title_pages_semaphore_);
                wait_values_.push_back(context_.title_pages_value_);
            }
            title_pages_seen_ = context_.title_pages_version_;
            // Uploads made on the staging queues without a host wait (note_upload): this submission may read them.
            if (context_.upload_version_ != upload_seen_) {
                for (const auto& [semaphore, value] : context_.upload_values_)
                    if (semaphore != progress_) {
                        waits_.push_back(semaphore);
                        wait_values_.push_back(value);
                    }
                upload_seen_ = context_.upload_version_;
            }
            if (kind_ != QueueKind::copy && runs_shaders)
                storage_images.assign(context_.storage_pending_.begin(), context_.storage_pending_.end());
            if (!storage_images.empty()) context_.storage_pending_.clear();
            zero_buffers.swap(context_.zero_pending_);  // any queue can fill a buffer
        }
        for (const auto& zero : zero_buffers)
            context_.vk.vkCmdFillBuffer(command_buffer, zero.buffer, 0, VK_WHOLE_SIZE, 0);
        if (!zero_buffers.empty()) {
            VkMemoryBarrier filled{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            filled.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            filled.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            context_.vk.vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &filled, 0, nullptr, 0, nullptr);
        }
        // Only an image this submission actually initializes (one still UNDEFINED) makes other queues wait for it.
        bool storage_initialized = false;
        for (const ResourceId id : storage_images) {
            const ResourceBinding* image = context_.resolve(id);
            if (!image || !image->image) continue;
            const VkImageSubresourceRange range{image->aspects, 0, image->mip_levels, 0, image->array_layers};
            auto pieces = detail::current(&next_tracker, id, range, VK_IMAGE_LAYOUT_UNDEFINED);
            storage_initialized |= detail::initialize(replay, command_buffer, *image, id, pieces,
                                                      VK_IMAGE_LAYOUT_GENERAL, &next_tracker);
        }
        bool wrote_readback = false;
        VkPipelineStageFlags2 readback_stages = 0;
        std::uint32_t open_labels = 0;  // events may span the lists of one ExecuteCommandLists
        BreadcrumbRing* const breadcrumbs = context_.breadcrumbs_.get();  // LEONIDA_DRED only
        std::vector<BreadcrumbRing::Mark> marks;
        // Device-fault diagnostics (upstream): per-command checkpoints for the lists of a numbered call.
        CheckpointSubmission diagnostics;
        PFN_vkCmdSetCheckpointNV checkpoint = nullptr;
        if (execute_call) {
            auto& fault = *context_.device_fault();
            checkpoint = fault.checkpoint_command();
            const char* queue_kind = kind_ == QueueKind::direct ? "direct" :
                                     kind_ == QueueKind::compute ? "compute" : "copy";
            diagnostics = fault.begin_submission(execute_call, queue_, queue_kind, lists.size());
        }
        for (std::size_t i = 0; i < lists.size(); ++i) {
            const auto& list = prepared[i];
            if (breadcrumbs) {
                marks.push_back(breadcrumbs->mark(*lists[i], this, i < identity.size() ? identity[i] : lists[i],
                                                  i < held_names.size() ? &held_names[i] : nullptr));
                breadcrumbs->record_begin(command_buffer, marks.back());
            }
            OpMarkers op_markers;
            if (breadcrumbs && breadcrumbs->op_markers_for(marks.back())) {
                const BreadcrumbRing::Mark& m = marks.back();
                op_markers = {breadcrumbs->op_marker_function(), breadcrumbs->marker_buffer(),
                              breadcrumbs->progress_offset(m), m.first_op, m.positions.data(), m.positions.size()};
            }
            const OpMarkers* const markers = op_markers.write ? &op_markers : nullptr;
            if (execute_call && i < diagnostics.lists.size()) {
                const CheckpointRecorder recorder{checkpoint, diagnostics, context_.device_fault()->pipeline_names, i};
                emit_commands<true>(list, command_buffer, replay, &next_tracker, &open_labels, &recorder, markers);
            } else {
                emit(list, command_buffer, replay, &next_tracker, &open_labels, markers);
            }
            if (breadcrumbs) breadcrumbs->record_end(command_buffer, marks.back());
            wrote_readback |= list.wrote_readback;
            readback_stages |= list.readback_src_stages;
        }
        close_labels(command_buffer, replay, open_labels);
        if (kind_ == QueueKind::copy) decay_to_common(replay, command_buffer, lists, resolve, &next_tracker);
        // Only flush GPU writes to the host domain when this batch actually wrote to a readback buffer.
        if (wrote_readback) {
            const VkPipelineStageFlags2 allowed = queue_supported_stages(kind_);
            VkPipelineStageFlags2 src_stages = readback_stages & allowed;
            if (!src_stages) src_stages = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            const bool copy_only =
                (src_stages & ~(VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT)) == 0;
            const VkAccessFlags2 src_access =
                copy_only ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_MEMORY_WRITE_BIT;

            if (context_.vk.vkCmdPipelineBarrier2) {
                VkMemoryBarrier2 host{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                host.srcStageMask = src_stages;
                host.srcAccessMask = src_access;
                host.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
                host.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
                VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers = &host;
                context_.vk.vkCmdPipelineBarrier2(command_buffer, &dep);
            } else {
                const VkPipelineStageFlags legacy_src =
                    copy_only ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host.srcAccessMask = copy_only ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_MEMORY_WRITE_BIT;
                host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                context_.vk.vkCmdPipelineBarrier(command_buffer, legacy_src, VK_PIPELINE_STAGE_HOST_BIT,
                                                 0, 1, &host, 0, nullptr, 0, nullptr);
            }
        }
        const bool ended = context_.observe_result(context_.vk.vkEndCommandBuffer(command_buffer),
                                                  "vkEndCommandBuffer") == VK_SUCCESS;
        last_timing_.record_ms = phase_ms(record_start, phase_clock::now());
        last_timing_.title_map_ms = static_cast<std::uint32_t>((g_title_map_ns - title_map_before) / 1000000);
        const struct SubmitTimer {
            VulkanQueue& q;
            phase_clock::time_point start = phase_clock::now();
            ~SubmitTimer() {
                q.last_timing_.submit_ms = static_cast<std::uint32_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(phase_clock::now() - start).count());
            }
        } submit_timer{*this};
        resource_lock.unlock();  // release resource_mutex_ and pipeline_mutex_ before vkQueueSubmit while keeping tracker_lock held
        pipeline_lock.unlock();
        const bool wait_previous = ordered_timestamps && submitted_;
        if (wait_previous) {
            waits_.push_back(progress_);
            wait_values_.push_back(submitted_);
        }
        const bool submitted = ended && submit(command_buffer, VK_NULL_HANDLE, 0,
                                               execute_call ? &diagnostics : nullptr);
        // A refused vkQueueSubmit is named by its VkResult in describe() (", vkQueueSubmit returned -4"); the step is
        // named only when no VkResult does.
        if (!submitted)
            failure_.reason = !ended ? "vkEndCommandBuffer" : last_submit_result_ == VK_SUCCESS ? "vkQueueSubmit" : nullptr;
        if (mesh_draws)
            std::fprintf(stderr, "[rt] pso-mesh unix_ms=%lld draws=%u submitted=%d\n",
                         pipeline_profile::unix_ms(), mesh_draws, int(submitted));
        if (!submitted && (!storage_images.empty() || !zero_buffers.empty())) {
            // not initialized after all: try again with the next submission
            std::lock_guard lock(context_.storage_mutex_);
            context_.storage_pending_.insert(storage_images.begin(), storage_images.end());
            context_.zero_pending_.insert(context_.zero_pending_.end(), zero_buffers.begin(), zero_buffers.end());
        }
        if (submitted && (storage_initialized || !zero_buffers.empty())) {
            std::lock_guard lock(context_.storage_mutex_);
            context_.storage_init_semaphore_ = progress_;
            context_.storage_init_value_ = submitted_;
            ++context_.storage_init_version_;
            storage_init_seen_ = context_.storage_init_version_;
        }
        if (submitted && bindless_uploaded) publish_bindless_upload();
        if (wait_previous && !submitted && !waits_.empty()) {  // a failed submit keeps the waits for the next one
            waits_.pop_back();
            wait_values_.pop_back();
        }
        if (breadcrumbs) breadcrumbs->submitted(std::move(marks), submitted);  // "submitted" is CPU knowledge only
        if (submitted) {
            next_tracker.commit();
            tracker_transaction.done = true;
            slot->predicate = std::move(batch);
            slot->predicated_copy = std::move(copy_batch);
            slot->counted_dispatch = std::move(counted_batch);
            slot->indirect_clamp = std::move(clamp_batch);
            queries.commit();
            slot->segments = std::move(queries.retained);
            slot->geometry = std::move(geometry);
        } else {
            next_tracker.rollback();  // here, under tracker_mutex_ (unlocked below, before the guard's scope ends)
            tracker_transaction.done = true;
            discard();
        }
        if (depth24_guard.active) {
            context_.depth24_->retire(progress_, submitted ? submitted_ : 0);
            depth24_guard.active = false;
        }
        if (scratch_guard.active) {
            context_.scratch_->retire(progress_, submitted ? submitted_ : 0);
            scratch_guard.active = false;
        }
        if (indirect_guard.active) {
            context_.indirect_->retire(progress_, submitted ? submitted_ : 0);
            indirect_guard.active = false;
        }
        if (clamp_guard.active) {
            context_.indirect_clamp_->retire(progress_, submitted ? submitted_ : 0);
            clamp_guard.active = false;
        }
        // Under tracker_mutex_, so no other flush comes between this submit and the set's note of whether it reached
        // the queue (it keeps a single upload obligation).
        if (bindless_commit) bindless_commit(submitted);
        tracker_lock.unlock();
        return submitted ? ReplayResult::success : queue_unsupported(__LINE__);
    }

    void marker_now(DebugMarker::Kind kind, const char* label) {
        if (kind != DebugMarker::Kind::end && label && progress_) context_.note_queue_marker(progress_, label);
        const DebugLabelFunctions& f = context_.labels_;
        if (!f.complete()) return;
        if (kind == DebugMarker::Kind::end && !open_labels_) return;
        VkDebugUtilsLabelEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
        info.pLabelName = label ? label : "";
        std::lock_guard lock(context_.submit_mutex());
        if (kind == DebugMarker::Kind::set) {
            f.queue_insert(queue_, &info);
        } else if (kind == DebugMarker::Kind::begin) {
            f.queue_begin(queue_, &info);
            ++open_labels_;
        } else {
            f.queue_end(queue_);
            --open_labels_;
        }
    }

    // ---- Wait-before-Signal: operations held on the CPU (queue_holds.h) ----

    // One held operation, with the waits that were pending when it was asked for.
    struct Deferred {
        enum class Kind : std::uint8_t { batch, signal, marker };
        Kind kind = Kind::batch;
        std::uint64_t seq = 0;  // QueueHolds::next_seq(): the order operations were held in, across queues
        std::uint64_t execute_call = 0;  // original ExecuteCommandLists call for a held batch
        std::vector<VkSemaphore> waits;
        std::vector<std::uint64_t> wait_values;
        std::vector<CommandList> lists;  // batch: copies, so the application may Reset its lists after the call
        std::vector<const CommandList*> origins;  // batch: the lists as passed, as LEONIDA_DRED's list ids only
        std::vector<BreadcrumbRing::Names> names;  // batch, LEONIDA_DRED: the debug names when it was held
        VkSemaphore fence = VK_NULL_HANDLE;  // signal
        std::uint64_t value = 0;
        DebugMarker::Kind marker_kind = DebugMarker::Kind::set;  // marker
        std::string label;
        // Alive until the operation has run: the fences it waits on and signals, the pipeline states and root
        // signatures of the lists (the application may release them right after the call).
        std::vector<std::shared_ptr<void>> pins;
        std::vector<std::shared_ptr<void>> wait_pins;  // the fences of `waits`
        std::chrono::steady_clock::time_point held_at{};  // LEONIDA_GPU_LATENCY: when it was held
    };

    bool waits_pending(const std::vector<VkSemaphore>& semaphores, const std::vector<std::uint64_t>& values) const {
        for (std::size_t i = 0; i < semaphores.size(); ++i)
            if (context_.holds().pending(semaphores[i], values[i])) return true;
        return false;
    }

    // The next batch or Signal has to be held: something is held ahead of it (queue order), or a Wait it carries
    // has no submitted Signal yet. Called with state_mutex_ held.
    bool must_hold() const { return !held_.empty() || waits_pending(waits_, wait_values_); }

    // The waits are consumed: submitted as `ticket` (0: dropped on device loss). What the application may release
    // (the fences they named) is kept until that submission has completed on the GPU, as a D3D12 queue does:
    // destroying a semaphore a pending submission uses is a validation error. Retired pins are destroyed by the
    // caller after state_mutex_ is released (drain_pins): the last release of a fence waits for a running release,
    // which must not be waiting for this lock.
    void clear_waits(std::uint64_t ticket = 0) {
        waits_.clear();
        wait_values_.clear();
        retire_pins(wait_pins_, ticket);
    }

    // The submission just made (submitted_) carries a write to the bindless set's table or fallback images: every
    // other queue's next submission waits for it (execute_now). tracker_mutex_ held.
    void publish_bindless_upload() {
        std::lock_guard lock(context_.storage_mutex_);
        context_.bindless_upload_semaphore_ = progress_;
        context_.bindless_upload_value_ = submitted_;
        ++context_.bindless_upload_version_;
        bindless_upload_seen_ = context_.bindless_upload_version_;
    }

    void retire_pins(std::vector<std::shared_ptr<void>>& pins, std::uint64_t ticket) {
        for (auto& pin : pins) {
            if (ticket) pin_queue_.push_back({ticket, std::move(pin)});
            else retired_pins_.push_back(std::move(pin));
        }
        pins.clear();
    }

    // Moves the pins whose submission has completed (and the ones retired outright) to `out`. state_mutex_ held.
    void drain_pins(std::vector<std::shared_ptr<void>>& out) {
        out.swap(retired_pins_);
        if (pin_queue_.empty()) return;
        const std::uint64_t done = completed();
        while (!pin_queue_.empty() && pin_queue_.front().ticket <= done) {
            out.push_back(std::move(pin_queue_.front().pin));
            pin_queue_.pop_front();
        }
    }

    // Appends `op`, which takes the pending waits with it (a marker has none). Called with state_mutex_ held.
    void hold(Deferred&& op) {
        op.seq = context_.holds().next_seq();
        if (op.kind != Deferred::Kind::marker) {
            op.waits = std::move(waits_);
            op.wait_values = std::move(wait_values_);
            op.wait_pins = std::move(wait_pins_);
            waits_.clear();
            wait_values_.clear();
            wait_pins_.clear();
        }
        {
            // Held work per queue at powers of two, with what the oldest held operation waits for: work that is held
            // forever never reaches replay (retail boots 50-58: draw lists executed, none prepared).
            const std::size_t n = held_.size() + 1;
            if ((n & (n - 1)) == 0 && n >= 4) {
                const Deferred& oldest = held_.empty() ? op : held_.front();
                std::fprintf(stderr, "[rt] d3d12: queue %p holds %zu operations; the oldest waits on %zu fence(s), "
                                     "first value %llu\n",
                             static_cast<void*>(this), n, oldest.waits.size(),
                             oldest.wait_values.empty() ? 0ull
                                                        : static_cast<unsigned long long>(oldest.wait_values.front()));
            }
        }
        if (gpu_latency::enabled()) op.held_at = std::chrono::steady_clock::now();
        held_.push_back(std::move(op));
        context_.holds().held_changed(1);
    }

    // Runs one held operation as the call would have, with its own waits and none of the ones queued since.
    // A submission that fails leaves its waits for the next one, as when the call is not held (a transient
    // vkQueueSubmit failure is retried by the application's next call), ahead of the waits asked for since.
    void run(Deferred& op) {
        auto later_waits = std::move(waits_);
        auto later_values = std::move(wait_values_);
        auto later_pins = std::move(wait_pins_);
        waits_ = std::move(op.waits);
        wait_values_ = std::move(op.wait_values);
        wait_pins_ = std::move(op.wait_pins);  // the fences stay alive until the submission has consumed their waits
        switch (op.kind) {
        case Deferred::Kind::batch: {
            std::vector<const CommandList*> lists;
            for (const CommandList& list : op.lists) lists.push_back(&list);
            // The pipeline states and root signatures of the lists are in op.pins (see execute()): they live until
            // the batch has been recorded and then, with the fences, until its submission has completed (below).
            const ReplayResult result = execute_now(lists, op.execute_call, op.origins, op.names);
            if (result != ReplayResult::success) {
                int none = static_cast<int>(ReplayResult::success);
                held_failure_.compare_exchange_strong(none, static_cast<int>(result));
                std::fprintf(stderr, "[rt] d3d12: a held ExecuteCommandLists failed when it was released (result %d, "
                                     "%s); nothing was submitted for it, and a Signal behind it still signals\n",
                             static_cast<int>(result), describe(failure_).c_str());
            }
            break;
        }
        case Deferred::Kind::signal:
            if (!submit(VK_NULL_HANDLE, op.fence, op.value))
                std::fprintf(stderr, "[rt] d3d12: a held Signal failed when it was released\n");
            break;
        case Deferred::Kind::marker:
            marker_now(op.marker_kind, op.label.c_str());
            break;
        }
        retire_pins(op.pins, submitted_);  // the fence it signals, the pipeline states: until its submission is done
        waits_.insert(waits_.end(), later_waits.begin(), later_waits.end());
        wait_values_.insert(wait_values_.end(), later_values.begin(), later_values.end());
        for (auto& pin : later_pins) wait_pins_.push_back(std::move(pin));
    }

    bool ready(const Deferred& op) const { return !waits_pending(op.waits, op.wait_values); }

    // Holder: called by the thread whose Signal made a held operation ready (never with a queue lock held).
    bool release_ready() override {
        std::vector<std::shared_ptr<void>> retired;
        bool released = false;
        {
            std::lock_guard state(state_mutex_);
            while (!held_.empty() && (context_.is_device_lost() || ready(held_.front()))) {
                Deferred op = std::move(held_.front());
                held_.pop_front();
                context_.holds().held_changed(-1);
                if (gpu_latency::enabled()) gpu_latency::held(int(kind_), int(op.kind), op.held_at);
                if (!context_.is_device_lost()) run(op);
                released = true;
            }
            drain_pins(retired);
        }
        return released;
    }

    // Holder: the fence is released, so what waits on it can never run.
    void discard_waiting_on(VkSemaphore semaphore) override {
        std::vector<Deferred> dropped;
        {
            std::lock_guard state(state_mutex_);
            for (auto it = held_.begin(); it != held_.end();) {
                if (std::find(it->waits.begin(), it->waits.end(), semaphore) == it->waits.end() &&
                    !(it->kind == Deferred::Kind::signal && it->fence == semaphore)) {
                    ++it;
                    continue;
                }
                std::fprintf(stderr, "[rt] d3d12: a fence was released while a queue still held work that names it; "
                                     "that work is dropped\n");
                dropped.push_back(std::move(*it));
                it = held_.erase(it);
                context_.holds().held_changed(-1);
            }
        }
    }

    // Holder: the sequence number of the oldest held operation.
    std::uint64_t oldest_held_seq() override {
        std::lock_guard state(state_mutex_);
        return held_.empty() ? 0 : held_.front().seq;
    }

    // A queue destroyed with held work. Reported, not waited for: nothing was submitted for it, so the VkQueue is
    // idle and the destructor goes on. The fences of held Signals will never reach their values, so what is waiting
    // for them on the CPU (SetEventOnCompletion, as a released fence completes them) is woken, and the batches
    // that wait for them on other queues stay held (and are reported when those queues are destroyed).
    void report_held_at_teardown() {
        std::deque<Deferred> dropped;
        {
            std::lock_guard state(state_mutex_);
            if (held_.empty()) return;
            std::uint64_t value = 0;
            for (const Deferred& op : held_)
                for (std::size_t i = 0; i < op.waits.size() && !value; ++i)
                    if (context_.holds().pending(op.waits[i], op.wait_values[i])) value = op.wait_values[i];
            std::size_t signals = 0;
            for (const Deferred& op : held_) signals += op.kind == Deferred::Kind::signal;
            std::fprintf(stderr, "[rt] d3d12: queue destroyed with %zu held operation(s) never submitted: waits for a "
                                 "fence value (%llu) that no Signal reached; the fence was never signaled%s\n",
                         held_.size(), static_cast<unsigned long long>(value),
                         signals ? "; CPU waits on the fences of its held Signals are completed" : "");
            for (const Deferred& op : held_)
                if (op.kind == Deferred::Kind::signal) context_.cancel_fence_waits(op.fence);
            context_.holds().held_changed(-static_cast<std::int64_t>(held_.size()));
            dropped.swap(held_);
        }
        // The pins (fences, pipeline states) are released here, outside the lock.
    }

    struct CommandSlot {
        VkCommandBuffer buffer;
        std::uint64_t ticket;
        std::vector<std::shared_ptr<const OcclusionSegmentPool>> segments;
        std::vector<VkEvent> events;  // order_timestamp_copy's, reset and reused once the slot completes
        std::unique_ptr<PredicatedBufferCopier::Batch> predicated_copy;
        std::unique_ptr<IndirectDispatchCounter::Batch> counted_dispatch;
        std::unique_ptr<IndirectArgumentClamp::Batch> indirect_clamp;  // counter slots, read once this slot completes
        std::unique_ptr<PredicateReducer::Batch> predicate; // retained until this slot completes
        std::vector<KhrInstanceStage> khr_instances;  // KHR ARRAY_OF_POINTERS staging, same lifetime
        std::vector<KhrAsQueryPool> khr_as_queries;   // KHR postbuild COMPACTED_SIZE query pools, same lifetime
        std::vector<VulkanContext::GeometryUpload> geometry;
    };

    struct PendingBindings {
        std::uint64_t ticket = 0;
        // Vulkan needs the resource until the bind completes and the previous allocation until it is unmapped.
        std::vector<std::shared_ptr<const TileHeapAllocation>> old_heaps;
        std::shared_ptr<SparseResource> resource;
    };

    static bool valid_tiles(const SparseResource& resource, TileRegion region) {
        return region.count && region.first < resource.tile_count && region.count <= resource.tile_count - region.first;
    }

    bool initialize_sparse_image_layout(ResourceId id) {
        std::lock_guard lock(context_.tracker_mutex_);
        if (context_.tracker_.layout(id, 0) != VK_IMAGE_LAYOUT_UNDEFINED) return true;
        auto* slot = acquire();
        if (!slot) return false;
        const VkCommandBuffer command_buffer = slot->buffer;
        auto& vk = context_.vk;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        context_.gpu_labels_.begin_record(command_buffer);
        if (vk.vkBeginCommandBuffer(command_buffer, &begin) != VK_SUCCESS) {
            slot->ticket = 0;
            return false;
        }
        // Transition while the image is still unbound. Doing this after binding could discard another image's
        // contents in an aliased page. Subsequent uses keep the image in GENERAL.
        constexpr VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        detail::image_barrier(context_.replay_, command_buffer, context_.resources_.at(id), range,
                              VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
                              VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        if (vk.vkEndCommandBuffer(command_buffer) != VK_SUCCESS || !submit(command_buffer, VK_NULL_HANDLE, 0)) {
            slot->ticket = 0;
            return false;
        }
        context_.tracker_.set(id, range, VK_IMAGE_LAYOUT_GENERAL);
        return true;
    }

  public:
    // Makes the binds the title's shared pages queued (vulkan_title_pages.h) in one vkQueueBindSparse on this queue,
    // before the title's next lists and copies: after every queue's work submitted so far (a page is reused only once
    // what used it before has run) and before the next submission of every queue (title_pages_semaphore_, waited for
    // in execute_lists). Images bound for the first time are moved to GENERAL over their whole range first, while
    // still unbound, so no later first use can discard what the pages they share hold. Called by a D3D12 queue's
    // ExecuteCommandLists before it copies the title's textures; a queue of a family without sparse binding leaves the
    // binds to the next one that has it.
    bool flush_title_pages() {
        TitlePages* pages = context_.title_pages_.get();
        if (!pages || !pages->pending() || !(context_.title_pages_families_ & (1u << family_))) return true;
        std::lock_guard state(state_mutex_);
        if (context_.is_device_lost()) return false;
        std::vector<TitlePages::Op> ops = pages->take();
        if (ops.empty()) return true;
        auto& vk = context_.vk;
        // The application's waits belong to its next batch, not to these internal submissions.
        auto saved_waits = std::exchange(waits_, {});
        auto saved_values = std::exchange(wait_values_, {});
        const auto restore = [&] {
            waits_ = std::move(saved_waits);
            wait_values_ = std::move(saved_values);
        };
        // A slow flush (over 300 ms, on the submitting thread) says where it went: the first-bind layout
        // initialisation or the bind call, and how many images and binds it carried.
        const auto t0 = std::chrono::steady_clock::now();
        std::size_t first_count = 0;
        {
            std::vector<ResourceId> first;
            for (const auto& op : ops)
                if (op.first) first.push_back(op.id);
            first_count = first.size();
            if (!first.empty() && !initialize_title_images(first)) {
                restore();
                return false;
            }
        }
        const auto t1 = std::chrono::steady_clock::now();
        if (context_.title_pages_async()) {
            // The bind worker makes them (VulkanContext::set_title_pages_queue): after every queue's work submitted so
            // far, this queue's included; this queue's next submission waits for them like every other queue's
            // (title_pages_seen_ stays behind: the binds are on another VkQueue now). Render targets (whole-image
            // ops) go in a batch of their own that submissions wait for at once; streamed textures in one published
            // when made (enqueue_title_binds_locked).
            std::vector<TitlePages::Op> urgent, streamed;
            for (auto& op : ops) (op.full ? urgent : streamed).push_back(std::move(op));
            std::lock_guard lock(context_.submit_mutex());
            // No waits for the queues' work: what a bind changes is either memory the title says is free now (a
            // texture placed over another's range; the title released or no longer uses it, as on the console) or a
            // page of our pool, which is taken again only once the GPU is past the work submitted before it was
            // given back (TitlePages quarantine). Waiting for "every queue's work submitted so far" deadlocked: that
            // work can wait (wait-before-signal across queues) on submissions made later, which wait on these binds
            // (run 28: binder -> compute queue -> direct queue -> binder).
            std::vector<VkSemaphore> semaphores;
            std::vector<std::uint64_t> values;
            if (!urgent.empty()) context_.enqueue_title_binds_locked(std::move(urgent), semaphores, values, true);
            if (!streamed.empty())
                context_.enqueue_title_binds_locked(std::move(streamed), std::move(semaphores), std::move(values), false);
            restore();
            return true;
        }
        std::vector<VkSparseImageMemoryBindInfo> image_infos;
        std::vector<VkSparseImageOpaqueMemoryBindInfo> opaque_infos;
        for (const auto& op : ops) {
            if (!op.image_binds.empty())
                image_infos.push_back({op.image, std::uint32_t(op.image_binds.size()), op.image_binds.data()});
            if (!op.opaque_binds.empty())
                opaque_infos.push_back({op.image, std::uint32_t(op.opaque_binds.size()), op.opaque_binds.data()});
        }
        const std::uint64_t ticket = submitted_ + 1;
        bool ok = false;
        {
            std::lock_guard lock(context_.submit_mutex());
            std::vector<VkSemaphore> semaphores;
            std::vector<std::uint64_t> values;
            for (const auto& [semaphore, value] : context_.latest_pending_queue_values()) {
                semaphores.push_back(semaphore);
                values.push_back(value);
            }
            if (submitted_ && std::find(semaphores.begin(), semaphores.end(), progress_) == semaphores.end()) {
                semaphores.push_back(progress_);
                values.push_back(submitted_);
            }
            VkTimelineSemaphoreSubmitInfo tickets{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            tickets.waitSemaphoreValueCount = std::uint32_t(values.size());
            tickets.pWaitSemaphoreValues = values.data();
            tickets.signalSemaphoreValueCount = 1;
            tickets.pSignalSemaphoreValues = &ticket;
            VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &tickets};
            info.waitSemaphoreCount = std::uint32_t(semaphores.size());
            info.pWaitSemaphores = semaphores.data();
            info.imageBindCount = std::uint32_t(image_infos.size());
            info.pImageBinds = image_infos.data();
            info.imageOpaqueBindCount = std::uint32_t(opaque_infos.size());
            info.pImageOpaqueBinds = opaque_infos.data();
            info.signalSemaphoreCount = 1;
            info.pSignalSemaphores = &progress_;
            const auto t2 = std::chrono::steady_clock::now();
            const VkResult result = vk.vkQueueBindSparse(queue_, 1, &info, VK_NULL_HANDLE);
            if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
            ok = result == VK_SUCCESS;
            if (ok) context_.record_submit_locked(progress_, ticket);
            const auto t3 = std::chrono::steady_clock::now();
            const auto ms = [](auto a, auto b) {
                return static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count());
            };
            if (ms(t0, t3) >= 300) {
                static std::atomic<int> told{0};
                if (told.fetch_add(1, std::memory_order_relaxed) < 200) {
                    std::size_t image_binds = 0, opaque_binds = 0;
                    for (const auto& i : image_infos) image_binds += i.bindCount;
                    for (const auto& o : opaque_infos) opaque_binds += o.bindCount;
                    std::fprintf(stderr,
                                 "[rt] title pages: slow flush %lld ms: %zu images (%zu first), %zu image binds in %zu "
                                 "infos, %zu opaque binds in %zu infos; first-bind init %lld ms, submit lock %lld ms, "
                                 "vkQueueBindSparse %lld ms\n",
                                 ms(t0, t3), ops.size(), first_count, image_binds, image_infos.size(), opaque_binds,
                                 opaque_infos.size(), ms(t0, t1), ms(t1, t2), ms(t2, t3));
                }
            }
        }
        restore();
        if (!ok) {
            static bool told = false;
            if (!std::exchange(told, true))
                std::fprintf(stderr, "[rt] title pages: vkQueueBindSparse failed; %zu images keep their old binds\n",
                             ops.size());
            return false;
        }
        submitted_ = last_sparse_ = ticket;
        std::lock_guard lock(context_.storage_mutex_);
        context_.title_pages_semaphore_ = progress_;
        context_.title_pages_value_ = ticket;
        ++context_.title_pages_version_;
        title_pages_seen_ = context_.title_pages_version_;
        return true;
    }

  private:
    // Every subresource of these never-bound images from UNDEFINED to GENERAL, in one submission (see
    // flush_title_pages).
    bool initialize_title_images(const std::vector<ResourceId>& ids) {
        // Lock order: resource_mutex_ (the bindings are read) -> tracker_mutex_.
        std::shared_lock resources(context_.resource_mutex_);
        std::lock_guard lock(context_.tracker_mutex_);
        auto* slot = acquire();
        if (!slot) return false;
        const VkCommandBuffer command_buffer = slot->buffer;
        auto& vk = context_.vk;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vk.vkBeginCommandBuffer(command_buffer, &begin) != VK_SUCCESS) {
            slot->ticket = 0;
            return false;
        }
        std::vector<std::pair<ResourceId, VkImageSubresourceRange>> done;
        for (const ResourceId id : ids) {
            const auto it = context_.resources_.find(id);
            if (it == context_.resources_.end() || !it->second.image) continue;
            const ResourceBinding& b = it->second;
            const VkImageSubresourceRange range{b.aspects, 0, std::max(1u, b.mip_levels), 0,
                                                b.volume ? 1u : std::max(1u, b.array_layers)};
            detail::image_barrier(context_.replay_, command_buffer, b, range, VK_IMAGE_LAYOUT_UNDEFINED,
                                  VK_IMAGE_LAYOUT_GENERAL, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                  VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
            done.emplace_back(id, range);
        }
        if (vk.vkEndCommandBuffer(command_buffer) != VK_SUCCESS || !submit(command_buffer, VK_NULL_HANDLE, 0)) {
            slot->ticket = 0;
            return false;
        }
        for (const auto& [id, range] : done) context_.tracker_.set(id, range, VK_IMAGE_LAYOUT_GENERAL);
        return true;
    }

  public:
    bool bind_tiles(ResourceId id, const std::shared_ptr<SparseResource>& resource,
                    std::map<std::uint64_t, TileMapping>& changes) {
        auto& vk = context_.vk;
        if (!vk.vkQueueBindSparse || context_.is_device_lost()) return false;
        PendingBindings pending{0, {}, resource};
        std::vector<VkSparseMemoryBind> binds;
        std::vector<VkSparseImageMemoryBind> image_binds;
        for (const auto& [index, page] : changes) {
            const auto old = resource->mapped.find(index);
            if (old != resource->mapped.end()) {
                if (old->second == page) continue;
                pending.old_heaps.push_back(old->second.heap);
            } else if (!page.heap) {
                continue;
            }
            if (page.heap && !(resource->memory_types & (1u << page.heap->memory_type))) return false;
            const VkDeviceMemory memory = page.heap ? page.heap->memory : VK_NULL_HANDLE;
            if (resource->image) {
                constexpr auto shape = element_tile_shape(false, 4);
                const auto extent = resource->image_extent;
                const auto columns = 1 + (extent.width - 1) / shape.width;
                const auto x = std::uint32_t(index % columns) * shape.width;
                const auto y = std::uint32_t(index / columns) * shape.height;
                image_binds.push_back({
                    {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0}, {std::int32_t(x), std::int32_t(y), 0},
                    {std::min(shape.width, extent.width - x), std::min(shape.height, extent.height - y), 1},
                    memory, page.offset, 0});
                continue;
            }
            const VkDeviceSize resource_offset = index * tile_size_in_bytes;
            if (!binds.empty()) {
                auto& previous = binds.back();
                if (previous.resourceOffset + previous.size == resource_offset && previous.memory == memory &&
                    (!memory || previous.memoryOffset + previous.size == page.offset)) {
                    previous.size += tile_size_in_bytes;
                    continue;
                }
            }
            binds.push_back({resource_offset, tile_size_in_bytes, memory, page.offset, 0});
        }
        if (binds.empty() && image_binds.empty()) return true;
        if (!image_binds.empty() && !initialize_sparse_image_layout(id)) return false;
        const auto ticket = submitted_ + 1;
        pending.ticket = ticket;
        collect_bindings(completed());
        auto semaphores = waits_;
        auto values = wait_values_;
        // Sparse binds are not ordered against work on the same Vulkan queue by submission order alone.
        if (submitted_) {
            semaphores.push_back(progress_);
            values.push_back(submitted_);
        }
        VkTimelineSemaphoreSubmitInfo tickets{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        tickets.waitSemaphoreValueCount = std::uint32_t(values.size());
        tickets.pWaitSemaphoreValues = values.data();
        tickets.signalSemaphoreValueCount = 1;
        tickets.pSignalSemaphoreValues = &ticket;
        const VkSparseBufferMemoryBindInfo buffer_bind{resource->buffer, std::uint32_t(binds.size()), binds.data()};
        const VkSparseImageMemoryBindInfo image_bind{
            resource->image, std::uint32_t(image_binds.size()), image_binds.data()};
        VkBindSparseInfo info{VK_STRUCTURE_TYPE_BIND_SPARSE_INFO, &tickets};
        info.waitSemaphoreCount = std::uint32_t(semaphores.size());
        info.pWaitSemaphores = semaphores.data();
        if (!binds.empty()) {
            info.bufferBindCount = 1;
            info.pBufferBinds = &buffer_bind;
        }
        if (!image_binds.empty()) {
            info.imageBindCount = 1;
            info.pImageBinds = &image_bind;
        }
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &progress_;
        pending_bindings_.push_back(std::move(pending));
        {
            std::lock_guard lock(context_.submit_mutex());
            const auto result = context_.observe_result(vk.vkQueueBindSparse(queue_, 1, &info, VK_NULL_HANDLE),
                                                        "vkQueueBindSparse");
            if (result != VK_SUCCESS) {
                if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
                pending_bindings_.pop_back();
                return false;
            }
            context_.record_submit_locked(progress_, ticket);
        }
        submitted_ = last_sparse_ = ticket;
        clear_waits(ticket);
        for (auto it = changes.begin(); it != changes.end();) {
            resource->mapped.erase(it->first);
            if (it->second.heap) ++it;
            else it = changes.erase(it);
        }
        resource->mapped.merge(changes);
        return true;
    }

    std::uint64_t completed() const {
        if (context_.is_device_lost()) return 0;
        std::uint64_t done = 0;
        const VkResult result = context_.observe_result(
            context_.vk.vkGetSemaphoreCounterValue(context_.device(), progress_, &done), "vkGetSemaphoreCounterValue");
        if (result == VK_ERROR_DEVICE_LOST) context_.mark_device_lost();
        // A failed timeline query cannot be used as a resource-retirement or command-buffer reuse ticket.
        return result == VK_SUCCESS ? done : 0;
    }

    void collect_bindings(std::uint64_t done) {
        while (!pending_bindings_.empty() && pending_bindings_.front().ticket <= done)
            pending_bindings_.pop_front();
        if (last_sparse_ <= done) last_sparse_ = 0;
    }

    // Releases what every finished command buffer holds; the first of them, or null when none has finished.
    CommandSlot* recycle_finished() {
        const auto done = completed();
        if (context_.is_device_lost()) return nullptr;
        collect_bindings(done);
        CommandSlot* reuse = nullptr;
        for (auto& slot : buffers_) {
            if (slot.ticket > done) continue;
            slot.segments.clear();
            slot.geometry.clear();
            recycle_khr_stages(slot.khr_instances);
            recycle_khr_as_queries(slot.khr_as_queries);
            for (VkEvent event : slot.events)
                if (context_.vk.vkResetEvent(context_.device(), event) == VK_SUCCESS) free_events_.push_back(event);
                else context_.vk.vkDestroyEvent(context_.device(), event, nullptr);
            slot.events.clear();
            if (!reuse) reuse = &slot;
        }
        return reuse;
    }

    // A queue at the cap with nothing finished waits here for its oldest submission. execute_now() calls it before it
    // takes resource_mutex_ or tracker_mutex_: the submission waited for may itself wait on another queue's work, and
    // that queue needs those locks to submit it. One execute runs at a time on a queue, so buffers_ is this caller's.
    // A timeout only means the GPU is slow: wait again rather than allocate past the cap. False when the device is lost
    // (now or while waiting), so the caller submits nothing.
    bool wait_for_room() {
        for (;;) {
            if (context_.is_device_lost()) return false;
            if (buffers_.size() < kMaxQueuedCommandBuffers) return true;
            const auto done = completed();
            for (const auto& slot : buffers_)
                if (slot.ticket <= done) return true;
            switch (wait_oldest_submitted()) {
            case VK_SUCCESS:
                return true;
            case VK_TIMEOUT:
                continue;
            case VK_ERROR_DEVICE_LOST:
                context_.mark_device_lost();
                return false;
            default:
                return true;  // nothing submitted to wait for (or a host error): no wait can make room, allocate
            }
        }
    }

    // Waits up to kQueuedWaitNs for the oldest submitted command buffer. VK_INCOMPLETE when none is submitted yet.
    VkResult wait_oldest_submitted() {
        std::uint64_t oldest = UINT64_MAX;
        for (const auto& slot : buffers_)
            if (slot.ticket < oldest) oldest = slot.ticket;
        if (oldest == UINT64_MAX) return VK_INCOMPLETE;  // all still being recorded
        VkSemaphoreWaitInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        info.semaphoreCount = 1;
        info.pSemaphores = &progress_;
        info.pValues = &oldest;
        return context_.observe_result(context_.vk.vkWaitSemaphores(context_.device(), &info, kQueuedWaitNs),
                                       "vkWaitSemaphores (queue progress)");
    }

    CommandSlot* acquire() {
        if (context_.is_device_lost()) return nullptr;
        context_.drain_deleted_resources(context_.completed_retire_ticket());
        CommandSlot* reuse = recycle_finished();
        if (reuse) {
            if (context_.vk.vkResetCommandBuffer(reuse->buffer, 0) != VK_SUCCESS) return nullptr;
            reuse->predicate.reset();
            reuse->predicated_copy.reset();
            reuse->counted_dispatch.reset();
            reuse->indirect_clamp.reset();
            reuse->ticket = UINT64_MAX;  // in use until submitted
            return reuse;
        }
        VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        info.commandPool = pool_;
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        if (context_.vk.vkAllocateCommandBuffers(context_.device(), &info, &command_buffer) != VK_SUCCESS)
            return nullptr;
        buffers_.push_back({command_buffer, UINT64_MAX});
        return &buffers_.back();
    }

    // An unsignaled event for order_timestamp_copy, or null. MoltenVK decides when the command buffer is submitted
    // whether a set or wait does anything, from the event's state then, so each use gets an event of its own.
    VkEvent take_event() {
        if (!free_events_.empty()) {
            const VkEvent event = free_events_.back();
            free_events_.pop_back();
            return event;
        }
        VkEventCreateInfo info{VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
        VkEvent event = VK_NULL_HANDLE;
        if (context_.vk.vkCreateEvent(context_.device(), &info, nullptr, &event) != VK_SUCCESS) return VK_NULL_HANDLE;
        return event;
    }

    static void pass_trace(const CheckpointSubmission& d) {
        static const std::uint64_t every = [] {
            const char* v = std::getenv("LEONIDA_PASS_TRACE");
            return v && *v ? std::strtoull(v, nullptr, 10) : 0ull;
        }();
        if (!every) return;
        const std::uint64_t present = submit_phases::totals().present_serial.load(std::memory_order_relaxed);
        if (present % every != 1 % every) return;
        std::string line;
        const char* last = nullptr;
        unsigned run = 0;
        const auto flush = [&] {
            if (!last) return;
            char text[96];
            std::snprintf(text, sizeof(text), run > 1 ? " %s x%u" : " %s", last, run);
            line += text;
        };
        for (const auto& x : d.dispatches) {
            const char* name = x.shader ? x.shader : "-";
            if (const char* slash = std::strrchr(name, '/')) name = slash + 1;
            if (last && std::strcmp(last, name) == 0) {
                ++run;
                continue;
            }
            flush();
            last = name;
            run = 1;
        }
        flush();
        std::fprintf(stderr, "[rt] passes: present %llu execute %llu (%s) %zu dispatches%s:%s\n",
                     static_cast<unsigned long long>(present), static_cast<unsigned long long>(d.execute_call),
                     d.queue_kind ? d.queue_kind : "?", d.dispatch_count,
                     d.dispatch_count > d.dispatches.size() ? " (last 64)" : "", line.empty() ? " none" : line.c_str());
    }

    bool submit(VkCommandBuffer command_buffer, VkSemaphore fence, std::uint64_t value,
                CheckpointSubmission* diagnostics = nullptr) {
        if (context_.is_device_lost()) return false;
        const std::uint64_t ticket = submitted_ + 1;
        if (last_sparse_) {
            collect_bindings(completed());
            if (context_.is_device_lost()) return false;
        }
        std::span<const VkSemaphore> semaphores = waits_;
        std::span<const std::uint64_t> values = wait_values_;
        std::vector<VkSemaphore> sparse_waits;
        std::vector<std::uint64_t> sparse_values;
        if (last_sparse_) {
            sparse_waits = waits_;
            sparse_values = wait_values_;
            sparse_waits.push_back(progress_);
            sparse_values.push_back(last_sparse_);
            semaphores = sparse_waits;
            values = sparse_values;
        }
        const std::vector<VkPipelineStageFlags> stages(semaphores.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);        VkResult res = VK_SUCCESS;
        {
            std::lock_guard lock(context_.submit_mutex());
            VkSemaphore signals[2]{progress_, VK_NULL_HANDLE};
            std::uint64_t signal_values[2]{ticket, 0};
            std::uint32_t signal_count = 1;
            // Decided under the submit lock, so the order of the checks is the order of the submissions. A Signal
            // that is not above what was already submitted for this fence (a held Signal(F, 2) released after an
            // unheld Signal(F, 3)) leaves the Vulkan signal out: the timeline cannot go down, and waits for 2 pass.
            // The value counts as submitted only after the submit succeeded (below).
            const bool signals_fence = fence && context_.holds().would_signal(fence, value);
            if (signals_fence) {
                signals[signal_count] = fence;
                signal_values[signal_count] = value;
                ++signal_count;
            }
            VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline.waitSemaphoreValueCount = std::uint32_t(values.size());
            timeline.pWaitSemaphoreValues = values.data();
            timeline.signalSemaphoreValueCount = signal_count;
            timeline.pSignalSemaphoreValues = signal_values;
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO, &timeline};
            submit.waitSemaphoreCount = std::uint32_t(semaphores.size());
            submit.pWaitSemaphores = semaphores.data();
            submit.pWaitDstStageMask = stages.data();
            submit.commandBufferCount = command_buffer ? 1 : 0;
            submit.pCommandBuffers = &command_buffer;
            submit.signalSemaphoreCount = signal_count;
            submit.pSignalSemaphores = signals;
            // LEONIDA_PASS_TRACE=<n> (with LEONIDA_DEVICE_FAULT=1): in the frame after every n-th present, each
            // submission logs its compute programs in order (repeats folded), for the frame's pass order.
            if (diagnostics) pass_trace(*diagnostics);
            // Publish before the driver can execute a marker or a concurrent fence wait can see the loss.
            if (diagnostics) context_.device_fault()->publish(std::move(*diagnostics));
            const auto label_fence=context_.gpu_labels_.submit_fence(command_buffer);
            res = context_.observe_result(context_.vk.vkQueueSubmit(queue_, 1, &submit, label_fence),
                                          "vkQueueSubmit");
            context_.gpu_labels_.submitted(command_buffer,res==VK_SUCCESS);
            if (res == VK_SUCCESS) {
                detail_stall::note_submit(progress_, ticket, semaphores, values);
                context_.record_submit_locked(progress_, ticket);
                if (signals_fence) context_.holds().signaled(fence, value);
            }
        }
        last_submit_result_ = res;
        if (res == VK_SUCCESS) context_.note_queue_submitted(progress_, ticket, std::exchange(submit_marker_, {}));
        if (res == VK_ERROR_DEVICE_LOST) {
            context_.note_lost_submission(progress_);
            context_.mark_device_lost();
            clear_waits();
            return false;
        }
        const bool ok = (res == VK_SUCCESS);
        if (ok && gpu_latency::enabled()) {
            // LEONIDA_GPU_LATENCY=1: when the GPU finished this submission, measured from the submit.
            const auto kind = int(kind_);
            const auto at = std::chrono::steady_clock::now();
            context_.enqueue_fence_callback(progress_, ticket, [kind, at] { gpu_latency::done(kind, at); });
        }
        if (ok) {
            submitted_ = ticket;
            clear_waits(ticket);
            for (auto& slot : buffers_)
                if (slot.buffer == command_buffer) slot.ticket = ticket;
        }
        return ok;
    }

    VulkanContext& context_;
    VkQueue queue_;
    QueueKind kind_ = QueueKind::direct;
    std::string submit_marker_;  // the last debug marker of the lists the next submit() carries (device-lost summary)
    VkResult last_submit_result_ = VK_SUCCESS;  // what the last vkQueueSubmit returned
    Failure failure_;  // the last execute_now's (state_mutex_)
    std::uint32_t family_;
    VkCommandPool pool_ = VK_NULL_HANDLE;
    VkSemaphore progress_ = VK_NULL_HANDLE;  // our own timeline: which command buffers are done
    std::uint64_t submitted_ = 0;
    std::vector<CommandSlot> buffers_;
    std::vector<VkEvent> free_events_;  // unsignaled, for order_timestamp_copy
    std::unique_ptr<OcclusionResolve> occlusion_resolve_;
    std::uint64_t storage_init_seen_ = 0;  // the storage_init_version_ this queue has waited for
    std::uint64_t bindless_upload_seen_ = 0;  // the bindless_upload_version_ this queue has waited for
    std::uint64_t title_pages_seen_ = 0;      // the title_pages_version_ this queue has waited for
    std::uint64_t upload_seen_ = 0;           // the upload_version_ this queue has waited for
    std::uint64_t last_sparse_ = 0;
    std::deque<PendingBindings> pending_bindings_;
    std::vector<VkSemaphore> waits_;
    std::vector<std::uint64_t> wait_values_;
    // execute's working storage. Per queue like buffers_: one execute at a time on a queue, as acquire() and
    // submit() already assume; other queues prepare concurrently with their own.
    std::vector<PreparedList> prepared_;
    std::uint32_t open_labels_ = 0;  // queue labels begun and not ended
    bool ok_ = false;
    // Wait-before-Signal. state_mutex_ is outermost (before pipeline_mutex_ -> resource_mutex_ -> tracker_mutex_ ->
    // submit_mutex_) and
    // guards the members above that execute, signal, wait and marker touch.
    std::mutex state_mutex_;
    std::deque<Deferred> held_;
    std::vector<std::shared_ptr<void>> wait_pins_;     // keep the fences of waits_ alive
    std::vector<std::shared_ptr<void>> retired_pins_;  // released by the caller once state_mutex_ is dropped
    struct TicketedPin {
        std::uint64_t ticket;
        std::shared_ptr<void> pin;
    };
    std::deque<TicketedPin> pin_queue_;  // released once progress_ reaches the ticket
    std::atomic<int> held_failure_{0};                 // a ReplayResult: the first held batch that failed
};

} // namespace leonida
