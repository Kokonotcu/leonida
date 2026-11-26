// docs/aot_pipelines.md step 3: graphics pipelines linked from VK_EXT_graphics_pipeline_library parts, for devices
// without VK_EXT_shader_object. A pipeline is split into its four parts:
//   - vertex input: the recipe's vertex bindings, attributes, divisors and topology class (cheap, made per PSO);
//   - pre-rasterization shaders: the vertex (and geometry) shader, the layout, and the rasterization state that cannot
//     be dynamic without VK_EXT_extended_dynamic_state3 (depth clamp, polygon mode, conservative mode);
//   - fragment shader: the pixel shader and the layout (and, for a shader that runs per sample, the multisample state);
//   - fragment output: the blend state, the multisample state and the attachment formats (cheap, made per PSO).
// Everything else the monolithic pipeline bakes in is dynamic here (core Vulkan 1.3 state: cull mode, front face,
// depth bias, rasterizer discard, line width, the depth and stencil tests), so the two shader parts depend on the
// shader alone and are shared by every PSO that uses it: `gta6rt --prebuild-shaders` compiles one per translated
// shader into the pipeline cache, and a PSO is a fast link (no link-time optimisation) of four parts. After its first
// draw the same parts are linked again with link-time optimisation in the background and that pipeline replaces the
// fast one; it goes into the pipeline cache, so the next run's PSO takes it at creation (Link).
// The runtime and the prebuild make their create infos through LibraryCreateInfo below, so a part the prebuild put in
// the pipeline cache is found there again (the cache is keyed by the whole create info).
#pragma once

#include "pso_recipe.h"

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace leonida::pso {

enum class Part : std::uint32_t {
    vertex_input = VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
    pre_raster = VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
    fragment = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
    output = VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT,
};

// The dynamic states of each part (a part names only its own). Together they are the monolithic path's dynamic
// states (GraphicsCreateInfo) plus what that path bakes in from the rasterization and depth-stencil state; the replay
// sets those at every draw (LibraryState::emit).
// `viewport_with_count`: Recipe::viewport_with_count, the pre-rasterization part's viewport and scissor states.
inline std::span<const VkDynamicState> library_dynamic_states(Part part, bool vertex_bindings,
                                                              bool viewport_with_count = true) {
    static constexpr VkDynamicState vertex_input[] = {VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY,
                                                      VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE,
                                                      VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE};
    static constexpr VkDynamicState pre_raster[] = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_LINE_WIDTH};
    static constexpr VkDynamicState pre_raster_one_viewport[] = {
        VK_DYNAMIC_STATE_VIEWPORT,          VK_DYNAMIC_STATE_SCISSOR,    VK_DYNAMIC_STATE_CULL_MODE,
        VK_DYNAMIC_STATE_FRONT_FACE,        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_LINE_WIDTH};
    static constexpr VkDynamicState fragment[] = {
        VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,   VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_BOUNDS,  VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE,
        VK_DYNAMIC_STATE_STENCIL_OP,          VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE};
    static constexpr VkDynamicState output[] = {VK_DYNAMIC_STATE_BLEND_CONSTANTS};
    switch (part) {
    case Part::vertex_input:
        // As GraphicsCreateInfo: no stride state without vertex bindings (VUID-vkCmdDraw-pStrides-04913).
        return {vertex_input, std::size(vertex_input) - (vertex_bindings ? 0 : 1)};
    case Part::pre_raster:
        if (!viewport_with_count) return pre_raster_one_viewport;
        return pre_raster;
    case Part::fragment: return fragment;
    case Part::output: return output;
    }
    return {};
}

// A pixel shader whose library must carry the multisample state (VUID-VkGraphicsPipelineCreateInfo-Input-06632): one
// that runs per sample, because an input is decorated Sample or it reads SampleId / SamplePosition. Its library is
// then keyed by the recipe's multisample state as well; any other pixel shader's library has none, so it is shared
// whatever the sample count.
inline bool spirv_runs_per_sample(const std::vector<std::uint32_t>& words) {
    constexpr std::uint32_t op_decorate = 71, decoration_builtin = 11, decoration_sample = 17;
    constexpr std::uint32_t builtin_sample_id = 18, builtin_sample_position = 19;
    for (std::size_t i = 5; i < words.size();) {
        const std::uint32_t count = words[i] >> 16, op = words[i] & 0xffff;
        if (!count || i + count > words.size()) break;
        if (op == op_decorate && count >= 3) {
            if (words[i + 2] == decoration_sample) return true;
            if (words[i + 2] == decoration_builtin && count >= 4 &&
                (words[i + 3] == builtin_sample_id || words[i + 3] == builtin_sample_position))
                return true;
        }
        i += count;
    }
    return false;
}

// The identity of a shared part: the stages it compiles and the state baked into it. Two PSOs with the same key share
// the library.
inline std::uint64_t library_key(Part part, const Recipe& r, bool per_sample = false) {
    std::uint64_t h = fnv1a(&part, sizeof(part));
    const auto mix = [&](const auto& value) { h = fnv1a(&value, sizeof(value), h); };
    for (const Stage& s : r.stages) {
        const bool fragment_stage = s.stage == VK_SHADER_STAGE_FRAGMENT_BIT;
        if (fragment_stage != (part == Part::fragment)) continue;
        mix(s.stage);
        h = fnv1a(s.name.data(), s.name.size(), h);
        mix(s.flags);
        mix(s.required_subgroup);
        mix(s.hash);
    }
    mix(r.push_stages);
    mix(r.push_size);
    mix(r.bindless);
    if (part == Part::pre_raster) {
        mix(r.depth_clamp);
        mix(r.polygon_mode);
        mix(r.conservative);
        if (!r.viewport_with_count) mix(r.viewport_with_count);  // keys of the usual dynamic-count part unchanged
    } else if (part == Part::fragment && per_sample) {
        mix(r.samples);
        mix(r.sample_shading);
        mix(r.min_sample_shading);
        mix(r.sample_mask);
        mix(r.alpha_to_coverage);
        mix(r.alpha_to_one);
    }
    return h;
}

// The identity of a vertex input or fragment output part: every recipe field LibraryCreateInfo reads for that part, as
// bytes (compared whole, not hashed, so two parts are shared only when their create infos are equal). These parts
// carry no shader; they were made per PSO (and per variant for the output part) until each turned out to cost the
// NVIDIA driver ~7.5 KB of host memory (617.42: 20,000 identical vertex input parts took 144 MiB of private bytes,
// D:\GTA6\analysis\tools\vkobj_cost.cpp), and the title's PSOs share a few hundred vertex layouts and target setups.
inline std::string state_part_key(Part part, const Recipe& r) {
    std::string key;
    const auto put = [&key](const void* data, std::size_t bytes) {
        key.append(static_cast<const char*>(data), bytes);
    };
    const auto value = [&put](const auto& v) { put(&v, sizeof(v)); };
    const auto array = [&put, &value](const auto& items) {
        value(items.size());
        if (!items.empty()) put(items.data(), items.size() * sizeof(items[0]));
    };
    value(part);
    if (part == Part::vertex_input) {
        array(r.bindings);
        array(r.attributes);
        array(r.divisors);
        value(r.topology);
        value(r.primitive_restart);
    } else if (part == Part::output) {
        array(r.colors);
        value(r.depth_format);
        value(r.stencil_format);
        array(r.blends);
        value(r.logic_op_enable);
        value(r.logic_op);
        value(r.samples);
        value(r.sample_shading);
        value(r.min_sample_shading);
        value(r.sample_mask);
        value(r.alpha_to_coverage);
        value(r.alpha_to_one);
    }
    return key;
}

// PSOs get link-time-optimised pipelines (from the pipeline cache at creation, else linked on the device's background
// worker after the first draw), and the parts are created retaining their link-time-optimisation information for
// that. LEONIDA_GPL_OPTIMIZE=0: fast links only, and parts that retain nothing.
//
// With LEONIDA_DEFER_GRAPHICS=1 it is off unless LEONIDA_GPL_OPTIMIZE=1 asks for it. A deferred PSO's pipelines are
// all built through its variants (vulkan_replay.h takes the variant path for every draw of a deferred PSO and never
// reaches the optimised link's request), so the optimised link never ran, while every shared part still kept its
// retained data for the device's life: the NVIDIA driver holds a copy of the part's SPIR-V and its own intermediate
// code in its 1 GiB heaps. Measured 100 s into a retail boot (2026-11-25, arena_scan): 144 SPIR-V copies (~262 MiB)
// in those heaps with retention, 2 without, and the heaps grow with every new shader the title draws with.
inline bool gpl_optimize() {
    static const bool on = [] {
        const char* value = std::getenv("LEONIDA_GPL_OPTIMIZE");
        const char* defer = std::getenv("LEONIDA_DEFER_GRAPHICS");
        const bool asked_off = value && value[0] == '0' && value[1] == 0;
        const bool asked_on = value && value[0] == '1' && value[1] == 0;
        const bool deferred = defer && defer[0] == '1';
        if (asked_off)
            std::fprintf(stderr, "[rt] d3d12: LEONIDA_GPL_OPTIMIZE=0: pipeline-library PSOs keep their fast links only\n");
        else if (deferred && !asked_on)
            std::fprintf(stderr, "[rt] d3d12: deferred graphics PSOs (LEONIDA_DEFER_GRAPHICS=1) keep their fast links "
                                 "only; their parts retain no link-time-optimisation data (LEONIDA_GPL_OPTIMIZE=1 "
                                 "keeps it)\n");
        return !asked_off && (!deferred || asked_on);
    }();
    return on;
}

// One part's create info. modules[i] is the module for recipe.stages[i]; the part takes the stages it compiles
// (vertex and geometry for pre_raster, the pixel shader for fragment). `per_sample`: a fragment part whose shader runs
// per sample carries the recipe's multisample state (spirv_runs_per_sample). The state a shared part does not depend
// on is left at fixed values, so every PSO asks for the very same create info. Not copyable: `info` points at members.
class LibraryCreateInfo {
public:
    LibraryCreateInfo(Part part, const Recipe& r, const std::vector<VkShaderModule>& modules, VkPipelineLayout layout,
                      bool per_sample = false) {
        library_.flags = VkGraphicsPipelineLibraryFlagsEXT(part);
        info.pNext = &library_;
        // Retained only when the same parts are linked again with link-time optimisation (gpl_optimize).
        info.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR |
                     (gpl_optimize() ? VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT : 0);
        const std::span<const VkDynamicState> dynamic =
            library_dynamic_states(part, !r.bindings.empty(), r.viewport_with_count);
        dynamic_.dynamicStateCount = std::uint32_t(dynamic.size());
        dynamic_.pDynamicStates = dynamic.data();
        info.pDynamicState = &dynamic_;
        library_.pNext = &rendering_;  // viewMask 0 for every part; the formats only for the fragment output part

        if (part == Part::pre_raster || part == Part::fragment) {
            info.layout = layout;
            for (std::size_t i = 0; i < r.stages.size(); ++i) {
                const bool fragment_stage = r.stages[i].stage == VK_SHADER_STAGE_FRAGMENT_BIT;
                if (fragment_stage != (part == Part::fragment)) continue;
                VkPipelineShaderStageCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
                s.stage = VkShaderStageFlagBits(r.stages[i].stage);
                s.flags = r.stages[i].flags;
                s.module = i < modules.size() ? modules[i] : VK_NULL_HANDLE;
                s.pName = "main";
                stages_.push_back(s);
                sizes_.push_back({VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, nullptr,
                                  r.stages[i].required_subgroup});
            }
            for (std::size_t i = 0; i < stages_.size(); ++i)
                if (sizes_[i].requiredSubgroupSize) stages_[i].pNext = &sizes_[i];
            info.stageCount = std::uint32_t(stages_.size());
            info.pStages = stages_.empty() ? nullptr : stages_.data();
        }
        switch (part) {
        case Part::vertex_input:
            bindings_ = r.bindings;
            attributes_ = r.attributes;
            divisors_ = r.divisors;
            if (!divisors_.empty()) {
                divisor_state_.vertexBindingDivisorCount = std::uint32_t(divisors_.size());
                divisor_state_.pVertexBindingDivisors = divisors_.data();
                vertex_input_.pNext = &divisor_state_;
            }
            vertex_input_.vertexBindingDescriptionCount = std::uint32_t(bindings_.size());
            vertex_input_.pVertexBindingDescriptions = bindings_.empty() ? nullptr : bindings_.data();
            vertex_input_.vertexAttributeDescriptionCount = std::uint32_t(attributes_.size());
            vertex_input_.pVertexAttributeDescriptions = attributes_.empty() ? nullptr : attributes_.data();
            assembly_.topology = VkPrimitiveTopology(r.topology);
            assembly_.primitiveRestartEnable = r.primitive_restart;
            info.pVertexInputState = &vertex_input_;
            info.pInputAssemblyState = &assembly_;
            break;
        case Part::pre_raster:
            // Dynamic here: discard, cull mode, front face, depth bias and line width (fixed values below).
            raster_.depthClampEnable = r.depth_clamp;
            raster_.polygonMode = VkPolygonMode(r.polygon_mode);
            raster_.cullMode = VK_CULL_MODE_NONE;
            raster_.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster_.lineWidth = 1.0f;
            if (r.conservative) {
                conservative_.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
                raster_.pNext = &conservative_;
            }
            viewport_.viewportCount = r.viewport_with_count ? 0 : 1;  // as GraphicsCreateInfo
            viewport_.scissorCount = r.viewport_with_count ? 0 : 1;
            info.pRasterizationState = &raster_;
            info.pViewportState = &viewport_;
            break;
        case Part::fragment:
            // Every depth and stencil field is dynamic; the struct is still required without
            // VK_EXT_extended_dynamic_state3 (VUID-VkGraphicsPipelineCreateInfo-renderPass-09035).
            info.pDepthStencilState = &depth_;
            if (per_sample) {
                fill_multisample(r);
                info.pMultisampleState = &multisample_;
            }
            break;
        case Part::output:
            for (std::uint32_t f : r.colors) colors_.push_back(VkFormat(f));
            rendering_.colorAttachmentCount = std::uint32_t(colors_.size());
            rendering_.pColorAttachmentFormats = colors_.empty() ? nullptr : colors_.data();
            rendering_.depthAttachmentFormat = VkFormat(r.depth_format);
            rendering_.stencilAttachmentFormat = VkFormat(r.stencil_format);
            blends_ = r.blends;
            blend_.logicOpEnable = r.logic_op_enable;
            blend_.logicOp = VkLogicOp(r.logic_op);
            blend_.attachmentCount = std::uint32_t(blends_.size());
            blend_.pAttachments = blends_.empty() ? nullptr : blends_.data();
            fill_multisample(r);
            info.pColorBlendState = &blend_;
            info.pMultisampleState = &multisample_;
            break;
        }
    }
    LibraryCreateInfo(const LibraryCreateInfo&) = delete;
    LibraryCreateInfo& operator=(const LibraryCreateInfo&) = delete;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};

private:
    void fill_multisample(const Recipe& r) {
        sample_mask_ = r.sample_mask;
        multisample_.rasterizationSamples = VkSampleCountFlagBits(r.samples);
        multisample_.sampleShadingEnable = r.sample_shading;
        multisample_.minSampleShading = r.min_sample_shading;
        multisample_.pSampleMask = &sample_mask_;
        multisample_.alphaToCoverageEnable = r.alpha_to_coverage;
        multisample_.alphaToOneEnable = r.alpha_to_one;
    }

    VkGraphicsPipelineLibraryCreateInfoEXT library_{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
    VkPipelineRenderingCreateInfo rendering_{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkPipelineDynamicStateCreateInfo dynamic_{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    std::vector<VkPipelineShaderStageCreateInfo> stages_;
    std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> sizes_;
    std::vector<VkFormat> colors_;
    std::vector<VkVertexInputBindingDescription> bindings_;
    std::vector<VkVertexInputAttributeDescription> attributes_;
    std::vector<VkVertexInputBindingDivisorDescriptionEXT> divisors_;
    VkPipelineVertexInputDivisorStateCreateInfoEXT divisor_state_{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO_EXT};
    VkPipelineVertexInputStateCreateInfo vertex_input_{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly_{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    VkPipelineViewportStateCreateInfo viewport_{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    VkPipelineRasterizationStateCreateInfo raster_{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    VkPipelineRasterizationConservativeStateCreateInfoEXT conservative_{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT};
    VkPipelineDepthStencilStateCreateInfo depth_{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkSampleMask sample_mask_ = ~0u;
    VkPipelineMultisampleStateCreateInfo multisample_{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    std::vector<VkPipelineColorBlendAttachmentState> blends_;
    VkPipelineColorBlendStateCreateInfo blend_{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
};

// How a PSO's parts are linked:
//   fast       no link-time optimisation: cheap with graphicsPipelineLibraryFastLinking, the pipeline a PSO starts with;
//   optimized  with VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT: the whole-pipeline-speed one, linked on the
//              device's background worker after the fast one's first draw, and kept by the pipeline cache;
//   cached     optimized, but only if the pipeline cache has it (VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
//              which answers VK_PIPELINE_COMPILE_REQUIRED otherwise): a later run's PSO takes it at creation.
enum class Link { fast, optimized, cached };

// The pipeline linked from four parts. `libraries` holds the vertex input, pre-rasterization, fragment shader and
// fragment output parts.
class LinkCreateInfo {
public:
    LinkCreateInfo(const std::array<VkPipeline, 4>& libraries, VkPipelineLayout layout, Link link = Link::fast)
        : libraries_(libraries) {
        link_.libraryCount = std::uint32_t(libraries_.size());
        link_.pLibraries = libraries_.data();
        info.pNext = &link_;
        info.layout = layout;
        if (link != Link::fast) info.flags |= VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT;
        if (link == Link::cached) info.flags |= VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT;
    }
    LinkCreateInfo(const LinkCreateInfo&) = delete;
    LinkCreateInfo& operator=(const LinkCreateInfo&) = delete;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};

private:
    std::array<VkPipeline, 4> libraries_;
    VkPipelineLibraryCreateInfoKHR link_{VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR};
};

// The state a linked pipeline takes at the draw that the monolithic one bakes in (library_dynamic_states), from the
// recipe. Viewport, scissor, topology, primitive restart, strides, stencil reference and blend constants stay the
// replay's, as for the monolithic path.
struct LibraryState {
    struct Functions {
        PFN_vkCmdSetCullMode vkCmdSetCullMode = nullptr;
        PFN_vkCmdSetFrontFace vkCmdSetFrontFace = nullptr;
        PFN_vkCmdSetDepthBiasEnable vkCmdSetDepthBiasEnable = nullptr;
        PFN_vkCmdSetDepthBias vkCmdSetDepthBias = nullptr;
        PFN_vkCmdSetRasterizerDiscardEnable vkCmdSetRasterizerDiscardEnable = nullptr;
        PFN_vkCmdSetLineWidth vkCmdSetLineWidth = nullptr;
        PFN_vkCmdSetDepthTestEnable vkCmdSetDepthTestEnable = nullptr;
        PFN_vkCmdSetDepthWriteEnable vkCmdSetDepthWriteEnable = nullptr;
        PFN_vkCmdSetDepthCompareOp vkCmdSetDepthCompareOp = nullptr;
        PFN_vkCmdSetDepthBoundsTestEnable vkCmdSetDepthBoundsTestEnable = nullptr;
        PFN_vkCmdSetDepthBounds vkCmdSetDepthBounds = nullptr;
        PFN_vkCmdSetStencilTestEnable vkCmdSetStencilTestEnable = nullptr;
        PFN_vkCmdSetStencilOp vkCmdSetStencilOp = nullptr;
        PFN_vkCmdSetStencilCompareMask vkCmdSetStencilCompareMask = nullptr;
        PFN_vkCmdSetStencilWriteMask vkCmdSetStencilWriteMask = nullptr;

        // The Vulkan 1.3 names, else the VK_EXT_extended_dynamic_state(2) ones of a 1.2 device.
        bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
            bool all = true;
            const auto one = [&](auto& fn, const char* name, const char* ext) {
                auto p = get(device, name);
                if (!p && ext) p = get(device, ext);
                fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(p);
                all = all && fn;
            };
            one(vkCmdSetCullMode, "vkCmdSetCullMode", "vkCmdSetCullModeEXT");
            one(vkCmdSetFrontFace, "vkCmdSetFrontFace", "vkCmdSetFrontFaceEXT");
            one(vkCmdSetDepthBiasEnable, "vkCmdSetDepthBiasEnable", "vkCmdSetDepthBiasEnableEXT");
            one(vkCmdSetDepthBias, "vkCmdSetDepthBias", nullptr);
            one(vkCmdSetRasterizerDiscardEnable, "vkCmdSetRasterizerDiscardEnable", "vkCmdSetRasterizerDiscardEnableEXT");
            one(vkCmdSetLineWidth, "vkCmdSetLineWidth", nullptr);
            one(vkCmdSetDepthTestEnable, "vkCmdSetDepthTestEnable", "vkCmdSetDepthTestEnableEXT");
            one(vkCmdSetDepthWriteEnable, "vkCmdSetDepthWriteEnable", "vkCmdSetDepthWriteEnableEXT");
            one(vkCmdSetDepthCompareOp, "vkCmdSetDepthCompareOp", "vkCmdSetDepthCompareOpEXT");
            one(vkCmdSetDepthBoundsTestEnable, "vkCmdSetDepthBoundsTestEnable", "vkCmdSetDepthBoundsTestEnableEXT");
            one(vkCmdSetDepthBounds, "vkCmdSetDepthBounds", nullptr);
            one(vkCmdSetStencilTestEnable, "vkCmdSetStencilTestEnable", "vkCmdSetStencilTestEnableEXT");
            one(vkCmdSetStencilOp, "vkCmdSetStencilOp", "vkCmdSetStencilOpEXT");
            one(vkCmdSetStencilCompareMask, "vkCmdSetStencilCompareMask", nullptr);
            one(vkCmdSetStencilWriteMask, "vkCmdSetStencilWriteMask", nullptr);
            return all;
        }
    };

    bool rasterizer_discard = false, depth_bias = false;
    VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
    VkFrontFace front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    float bias_constant = 0, bias_clamp = 0, bias_slope = 0, line_width = 1.0f;
    bool depth_test = false, depth_write = false, depth_bounds = false, stencil_test = false;
    VkCompareOp depth_compare = VK_COMPARE_OP_NEVER;
    VkStencilOpState front{}, back{};
    float min_depth_bounds = 0, max_depth_bounds = 1;

    explicit LibraryState(const Recipe& r)
        : rasterizer_discard(r.rasterizer_discard), depth_bias(r.depth_bias), cull_mode(r.cull_mode),
          front_face(VkFrontFace(r.front_face)), bias_constant(r.bias_constant), bias_clamp(r.bias_clamp),
          bias_slope(r.bias_slope), line_width(r.line_width), depth_test(r.depth_test), depth_write(r.depth_write),
          depth_bounds(r.depth_bounds), stencil_test(r.stencil_test), depth_compare(VkCompareOp(r.depth_compare)),
          front(r.front), back(r.back), min_depth_bounds(r.min_depth_bounds), max_depth_bounds(r.max_depth_bounds) {}

    void emit(const Functions& fn, VkCommandBuffer cb) const {
        fn.vkCmdSetRasterizerDiscardEnable(cb, rasterizer_discard);
        fn.vkCmdSetCullMode(cb, cull_mode);
        fn.vkCmdSetFrontFace(cb, front_face);
        fn.vkCmdSetDepthBiasEnable(cb, depth_bias);
        fn.vkCmdSetDepthBias(cb, bias_constant, bias_clamp, bias_slope);
        fn.vkCmdSetLineWidth(cb, line_width);
        fn.vkCmdSetDepthTestEnable(cb, depth_test);
        fn.vkCmdSetDepthWriteEnable(cb, depth_write);
        fn.vkCmdSetDepthCompareOp(cb, depth_compare);
        fn.vkCmdSetDepthBoundsTestEnable(cb, depth_bounds);
        fn.vkCmdSetDepthBounds(cb, min_depth_bounds, max_depth_bounds);
        fn.vkCmdSetStencilTestEnable(cb, stencil_test);
        fn.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, front.failOp, front.passOp, front.depthFailOp,
                             front.compareOp);
        fn.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, back.failOp, back.passOp, back.depthFailOp, back.compareOp);
        fn.vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, front.compareMask);
        fn.vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, back.compareMask);
        fn.vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, front.writeMask);
        fn.vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, back.writeMask);
    }
};

}  // namespace leonida::pso
