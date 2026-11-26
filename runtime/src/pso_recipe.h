// A graphics pipeline's complete Vulkan description, as plain data: what the runtime records for every pipeline it
// builds (pso_recipes.jsonl beside gta6rt.exe) and what the build's pipelines stage replays ahead of time, so the
// driver finds the same pipelines already compiled in pipeline_cache.bin and the game compiles nothing at run time.
//
// The runtime and the replay make their VkGraphicsPipelineCreateInfo through the one GraphicsCreateInfo below, so the
// two cannot drift apart: a pipeline cache is keyed by the whole create info, and one field that differs is a miss.
// Shader code is named (the translator index's entry name) and hashed, never stored: a recipe holds no game data.
#pragma once

#include "shader_library.h"

#include <vulkan/vulkan_core.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace leonida::pso {

inline constexpr int recipe_version = 1;

// FNV-1a 64 over bytes; the stage hash is over the SPIR-V words exactly as the pipeline was given them.
inline std::uint64_t fnv1a(const void* data, std::size_t size, std::uint64_t h = 0xcbf29ce484222325ull) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}
inline std::uint64_t spirv_hash(const std::vector<std::uint32_t>& words) {
    return fnv1a(words.data(), words.size() * sizeof(std::uint32_t));
}

struct Stage {
    std::uint32_t stage = 0;              // VkShaderStageFlagBits
    std::string name;                     // translator index entry
    std::uint32_t flags = 0;              // VkPipelineShaderStageCreateFlags
    std::uint32_t required_subgroup = 0;  // VkPipelineShaderStageRequiredSubgroupSizeCreateInfo, 0 = none
    std::uint64_t hash = 0;               // spirv_hash of the words the stage was created from
};

// One VkDescriptorSetLayoutBinding plus its binding flags: the set-0 (bindless) layout is recorded with the recipes so
// the replay creates it byte for byte, whatever device limits sized it.
struct SetBinding {
    std::uint32_t binding = 0, type = 0, count = 0, stages = 0, flags = 0;
    bool operator==(const SetBinding&) const = default;
};
struct SetLayoutDesc {
    std::uint32_t create_flags = 0;
    std::vector<SetBinding> bindings;
    bool operator==(const SetLayoutDesc&) const = default;
};

struct Recipe {
    std::vector<Stage> stages;
    bool mesh = false;
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDivisorDescriptionEXT> divisors;
    std::uint32_t topology = 0;
    bool primitive_restart = false;
    // VkPipelineRasterizationStateCreateInfo
    bool depth_clamp = false, rasterizer_discard = false, depth_bias = false;
    std::uint32_t polygon_mode = 0, cull_mode = 0, front_face = 0;
    float bias_constant = 0, bias_clamp = 0, bias_slope = 0, line_width = 1.0f;
    bool conservative = false;
    // VkPipelineMultisampleStateCreateInfo
    std::uint32_t samples = VK_SAMPLE_COUNT_1_BIT;
    bool sample_shading = false, alpha_to_coverage = false, alpha_to_one = false;
    float min_sample_shading = 0;
    std::uint32_t sample_mask = ~0u;
    // VkPipelineDepthStencilStateCreateInfo
    bool depth_test = false, depth_write = false, depth_bounds = false, stencil_test = false;
    std::uint32_t depth_compare = 0;
    VkStencilOpState front{}, back{};
    float min_depth_bounds = 0, max_depth_bounds = 1;
    // VkPipelineColorBlendStateCreateInfo
    bool logic_op_enable = false;
    std::uint32_t logic_op = 0;
    float blend_constants[4]{};
    std::vector<VkPipelineColorBlendAttachmentState> blends;
    // VkPipelineRenderingCreateInfo, with the unbound slots already VK_FORMAT_UNDEFINED
    std::vector<std::uint32_t> colors;
    std::uint32_t depth_format = 0, stencil_format = 0;
    // The pipeline layout: one push-constant range over push_stages, and set 0 when bindless
    std::uint32_t push_stages = 0, push_size = 0;
    bool bindless = false;
    // The viewport and scissor counts are dynamic (VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT / SCISSOR_WITH_COUNT,
    // counts 0): the device has multiViewport, so a draw binds 1..16 for SV_ViewportArrayIndex
    // (leonida::GraphicsPipeline::viewport_with_count). Else VIEWPORT / SCISSOR with one each. A recipe written
    // without the field had dynamic counts.
    bool viewport_with_count = true;
};

// ---------------------------------------------------------------- the create info both sides use

// Owns every array a VkGraphicsPipelineCreateInfo points into. Not copyable or movable: `info` points at members.
class GraphicsCreateInfo {
public:
    // modules[i] is the module for recipe.stages[i].
    GraphicsCreateInfo(const Recipe& r, const std::vector<VkShaderModule>& modules, VkPipelineLayout layout) {
        stages_.resize(r.stages.size());
        sizes_.resize(r.stages.size());
        for (std::size_t i = 0; i < r.stages.size(); ++i) {
            VkPipelineShaderStageCreateInfo& s = stages_[i];
            s = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            s.stage = VkShaderStageFlagBits(r.stages[i].stage);
            s.flags = r.stages[i].flags;
            s.module = i < modules.size() ? modules[i] : VK_NULL_HANDLE;
            s.pName = "main";
            if (r.stages[i].required_subgroup) {
                sizes_[i] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
                sizes_[i].requiredSubgroupSize = r.stages[i].required_subgroup;
                s.pNext = &sizes_[i];
            }
        }
        for (std::uint32_t f : r.colors) colors_.push_back(VkFormat(f));
        rendering_.colorAttachmentCount = std::uint32_t(colors_.size());
        rendering_.pColorAttachmentFormats = colors_.empty() ? nullptr : colors_.data();
        rendering_.depthAttachmentFormat = VkFormat(r.depth_format);
        rendering_.stencilAttachmentFormat = VkFormat(r.stencil_format);

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

        // With dynamic counts (VIEWPORT_WITH_COUNT, SCISSOR_WITH_COUNT) a draw sets one, or the viewports
        // RSSetViewports bound for SV_ViewportArrayIndex (VUID-VkPipelineViewportStateCreateInfo-viewportCount-04135).
        viewport_.viewportCount = r.viewport_with_count ? 0 : 1;
        viewport_.scissorCount = r.viewport_with_count ? 0 : 1;

        raster_.depthClampEnable = r.depth_clamp;
        raster_.rasterizerDiscardEnable = r.rasterizer_discard;
        raster_.polygonMode = VkPolygonMode(r.polygon_mode);
        raster_.cullMode = r.cull_mode;
        raster_.frontFace = VkFrontFace(r.front_face);
        raster_.depthBiasEnable = r.depth_bias;
        raster_.depthBiasConstantFactor = r.bias_constant;
        raster_.depthBiasClamp = r.bias_clamp;
        raster_.depthBiasSlopeFactor = r.bias_slope;
        raster_.lineWidth = r.line_width;
        if (r.conservative) {
            conservative_.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
            conservative_.extraPrimitiveOverestimationSize = 0.0f;
            raster_.pNext = &conservative_;
        }

        sample_mask_ = r.sample_mask;
        multisample_.rasterizationSamples = VkSampleCountFlagBits(r.samples);
        multisample_.sampleShadingEnable = r.sample_shading;
        multisample_.minSampleShading = r.min_sample_shading;
        multisample_.pSampleMask = &sample_mask_;
        multisample_.alphaToCoverageEnable = r.alpha_to_coverage;
        multisample_.alphaToOneEnable = r.alpha_to_one;

        depth_.depthTestEnable = r.depth_test;
        depth_.depthWriteEnable = r.depth_write;
        depth_.depthCompareOp = VkCompareOp(r.depth_compare);
        depth_.depthBoundsTestEnable = r.depth_bounds;
        depth_.stencilTestEnable = r.stencil_test;
        depth_.front = r.front;
        depth_.back = r.back;
        depth_.minDepthBounds = r.min_depth_bounds;
        depth_.maxDepthBounds = r.max_depth_bounds;

        blends_ = r.blends;
        blend_.logicOpEnable = r.logic_op_enable;
        blend_.logicOp = VkLogicOp(r.logic_op);
        blend_.attachmentCount = std::uint32_t(blends_.size());
        blend_.pAttachments = blends_.empty() ? nullptr : blends_.data();
        std::memcpy(blend_.blendConstants, r.blend_constants, sizeof(blend_.blendConstants));

        // The stride state is last: a pipeline without vertex bindings (a full-screen pass that builds its vertices
        // from SV_VertexID) leaves it out, since no vkCmdBindVertexBuffers2 would ever set it
        // (VUID-vkCmdDraw-pStrides-04913). A mesh pipeline takes neither the topology nor the strides as dynamic
        // state (VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-07065 and -07066).
        std::uint32_t dynamic_count = 0;
        dynamic_states_[dynamic_count++] =
            r.viewport_with_count ? VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT : VK_DYNAMIC_STATE_VIEWPORT;
        dynamic_states_[dynamic_count++] =
            r.viewport_with_count ? VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT : VK_DYNAMIC_STATE_SCISSOR;
        if (!r.mesh) {
            dynamic_states_[dynamic_count++] = VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY;
            dynamic_states_[dynamic_count++] = VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE;
        }
        dynamic_states_[dynamic_count++] = VK_DYNAMIC_STATE_STENCIL_REFERENCE;
        dynamic_states_[dynamic_count++] = VK_DYNAMIC_STATE_BLEND_CONSTANTS;
        // A depth-bounds pipeline (DEPTH_STENCIL1/2) takes its bounds from OMSetDepthBounds at each draw.
        if (r.depth_bounds) dynamic_states_[dynamic_count++] = VK_DYNAMIC_STATE_DEPTH_BOUNDS;
        if (!r.mesh && !r.bindings.empty())
            dynamic_states_[dynamic_count++] = VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE;
        dynamic_.dynamicStateCount = dynamic_count;
        dynamic_.pDynamicStates = dynamic_states_;

        info.pNext = &rendering_;
        info.stageCount = std::uint32_t(stages_.size());
        info.pStages = stages_.data();
        info.pVertexInputState = r.mesh ? nullptr : &vertex_input_;  // ignored with a mesh stage; null for clarity
        info.pInputAssemblyState = r.mesh ? nullptr : &assembly_;
        info.pViewportState = &viewport_;
        info.pRasterizationState = &raster_;
        info.pMultisampleState = &multisample_;
        info.pDepthStencilState = &depth_;
        info.pColorBlendState = &blend_;
        info.pDynamicState = &dynamic_;
        info.layout = layout;
    }
    GraphicsCreateInfo(const GraphicsCreateInfo&) = delete;
    GraphicsCreateInfo& operator=(const GraphicsCreateInfo&) = delete;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};

private:
    std::vector<VkPipelineShaderStageCreateInfo> stages_;
    std::vector<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo> sizes_;
    std::vector<VkFormat> colors_;
    VkPipelineRenderingCreateInfo rendering_{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
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
    VkSampleMask sample_mask_ = ~0u;
    VkPipelineMultisampleStateCreateInfo multisample_{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    VkPipelineDepthStencilStateCreateInfo depth_{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    std::vector<VkPipelineColorBlendAttachmentState> blends_;
    VkPipelineColorBlendStateCreateInfo blend_{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkDynamicState dynamic_states_[8]{};
    VkPipelineDynamicStateCreateInfo dynamic_{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
};

// The layout create info for a recipe, given the set-0 layout handle (ignored unless recipe.bindless).
struct LayoutCreateInfo {
    LayoutCreateInfo(const Recipe& r, VkDescriptorSetLayout set0) : set0_(set0) {
        push_ = {r.push_stages, 0, r.push_size};
        info.pushConstantRangeCount = 1;
        info.pPushConstantRanges = &push_;
        if (r.bindless) {
            info.setLayoutCount = 1;
            info.pSetLayouts = &set0_;
        }
    }
    LayoutCreateInfo(const LayoutCreateInfo&) = delete;
    LayoutCreateInfo& operator=(const LayoutCreateInfo&) = delete;
    VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};

private:
    VkPushConstantRange push_{};
    VkDescriptorSetLayout set0_ = VK_NULL_HANDLE;
};

// ---------------------------------------------------------------- one line of JSON per recipe

namespace detail {

class Writer {
public:
    void key(const char* k) {
        sep();
        out += '"';
        out += k;
        out += "\":";
        fresh = true;
    }
    void num(std::uint64_t v) { sep(); out += std::to_string(v); }
    void flt(float v) {
        sep();
        char b[32];
        std::snprintf(b, sizeof b, "%.9g", double(v));
        out += b;
    }
    void boolean(bool v) { sep(); out += v ? "true" : "false"; }
    void str(std::string_view s) {
        sep();
        out += '"';
        for (char c : s) {
            if (c == '"' || c == '\\') out += '\\';
            if (static_cast<unsigned char>(c) < 0x20) {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", c);
                out += b;
                continue;
            }
            out += c;
        }
        out += '"';
    }
    void hex(std::uint64_t v) {
        char b[24];
        std::snprintf(b, sizeof b, "%016" PRIx64, v);
        str(b);
    }
    void open(char c) { sep(); out += c; fresh = true; }
    void close(char c) { out += c; fresh = false; }
    std::string out;

private:
    void sep() {
        if (!fresh && !out.empty()) out += ',';
        fresh = false;
    }
    bool fresh = true;
};

inline void write_stencil(Writer& w, const char* k, const VkStencilOpState& s) {
    w.key(k);
    w.open('[');
    for (std::uint32_t v : {std::uint32_t(s.failOp), std::uint32_t(s.passOp), std::uint32_t(s.depthFailOp),
                            std::uint32_t(s.compareOp), s.compareMask, s.writeMask, s.reference})
        w.num(v);
    w.close(']');
}

inline const Json* member(const Json& o, std::string_view k, Json::Kind kind) {
    const Json* v = o.find(k);
    return v && v->kind == kind ? v : nullptr;
}
inline bool get_u32(const Json& o, std::string_view k, std::uint32_t& out) {
    const Json* v = member(o, k, Json::Kind::number);
    if (!v || v->number < 0 || v->number > 4294967295.0) return false;
    out = std::uint32_t(v->number);
    return true;
}
inline bool get_f32(const Json& o, std::string_view k, float& out) {
    const Json* v = member(o, k, Json::Kind::number);
    if (!v) return false;
    out = float(v->number);
    return true;
}
inline bool get_bool(const Json& o, std::string_view k, bool& out) {
    const Json* v = member(o, k, Json::Kind::boolean);
    if (!v) return false;
    out = v->boolean;
    return true;
}
inline bool u32_array(const Json* a, std::vector<std::uint32_t>& out, std::size_t expect = 0) {
    if (!a || a->kind != Json::Kind::array || (expect && a->array.size() != expect)) return false;
    out.clear();
    for (const Json& e : a->array) {
        if (e.kind != Json::Kind::number || e.number < 0 || e.number > 4294967295.0) return false;
        out.push_back(std::uint32_t(e.number));
    }
    return true;
}
inline bool read_stencil(const Json& o, std::string_view k, VkStencilOpState& s) {
    std::vector<std::uint32_t> v;
    if (!u32_array(o.find(k), v, 7)) return false;
    s = {VkStencilOp(v[0]), VkStencilOp(v[1]), VkStencilOp(v[2]), VkCompareOp(v[3]), v[4], v[5], v[6]};
    return true;
}
inline bool parse_hex64(const Json* v, std::uint64_t& out) {
    if (!v || v->kind != Json::Kind::string || v->string.size() != 16) return false;
    out = 0;
    for (char c : v->string) {
        out <<= 4;
        if (c >= '0' && c <= '9') out |= std::uint64_t(c - '0');
        else if (c >= 'a' && c <= 'f') out |= std::uint64_t(c - 'a' + 10);
        else return false;
    }
    return true;
}

}  // namespace detail

inline std::string to_json(const Recipe& r) {
    detail::Writer w;
    w.open('{');
    w.key("v"); w.num(recipe_version);
    w.key("stages"); w.open('[');
    for (const Stage& s : r.stages) {
        w.open('{');
        w.key("stage"); w.num(s.stage);
        w.key("name"); w.str(s.name);
        w.key("flags"); w.num(s.flags);
        w.key("subgroup"); w.num(s.required_subgroup);
        w.key("hash"); w.hex(s.hash);
        w.close('}');
    }
    w.close(']');
    w.key("mesh"); w.boolean(r.mesh);
    w.key("bindings"); w.open('[');
    for (const auto& b : r.bindings) {
        w.open('['); w.num(b.binding); w.num(b.stride); w.num(b.inputRate); w.close(']');
    }
    w.close(']');
    w.key("attributes"); w.open('[');
    for (const auto& a : r.attributes) {
        w.open('['); w.num(a.location); w.num(a.binding); w.num(a.format); w.num(a.offset); w.close(']');
    }
    w.close(']');
    w.key("divisors"); w.open('[');
    for (const auto& d : r.divisors) {
        w.open('['); w.num(d.binding); w.num(d.divisor); w.close(']');
    }
    w.close(']');
    w.key("topology"); w.num(r.topology);
    w.key("restart"); w.boolean(r.primitive_restart);
    w.key("depth_clamp"); w.boolean(r.depth_clamp);
    w.key("discard"); w.boolean(r.rasterizer_discard);
    w.key("polygon"); w.num(r.polygon_mode);
    w.key("cull"); w.num(r.cull_mode);
    w.key("front_face"); w.num(r.front_face);
    w.key("depth_bias"); w.boolean(r.depth_bias);
    w.key("bias"); w.open('['); w.flt(r.bias_constant); w.flt(r.bias_clamp); w.flt(r.bias_slope); w.close(']');
    w.key("line_width"); w.flt(r.line_width);
    w.key("conservative"); w.boolean(r.conservative);
    w.key("samples"); w.num(r.samples);
    w.key("sample_shading"); w.boolean(r.sample_shading);
    w.key("min_sample_shading"); w.flt(r.min_sample_shading);
    w.key("alpha_to_coverage"); w.boolean(r.alpha_to_coverage);
    w.key("alpha_to_one"); w.boolean(r.alpha_to_one);
    w.key("sample_mask"); w.num(r.sample_mask);
    w.key("depth_test"); w.boolean(r.depth_test);
    w.key("depth_write"); w.boolean(r.depth_write);
    w.key("depth_compare"); w.num(r.depth_compare);
    w.key("depth_bounds"); w.boolean(r.depth_bounds);
    w.key("stencil_test"); w.boolean(r.stencil_test);
    detail::write_stencil(w, "front", r.front);
    detail::write_stencil(w, "back", r.back);
    w.key("depth_bounds_range"); w.open('['); w.flt(r.min_depth_bounds); w.flt(r.max_depth_bounds); w.close(']');
    w.key("logic_op_enable"); w.boolean(r.logic_op_enable);
    w.key("logic_op"); w.num(r.logic_op);
    w.key("blend_constants"); w.open('[');
    for (float c : r.blend_constants) w.flt(c);
    w.close(']');
    w.key("blends"); w.open('[');
    for (const auto& b : r.blends) {
        w.open('[');
        for (std::uint32_t v : {std::uint32_t(b.blendEnable), std::uint32_t(b.srcColorBlendFactor),
                                std::uint32_t(b.dstColorBlendFactor), std::uint32_t(b.colorBlendOp),
                                std::uint32_t(b.srcAlphaBlendFactor), std::uint32_t(b.dstAlphaBlendFactor),
                                std::uint32_t(b.alphaBlendOp), std::uint32_t(b.colorWriteMask)})
            w.num(v);
        w.close(']');
    }
    w.close(']');
    w.key("colors"); w.open('[');
    for (std::uint32_t f : r.colors) w.num(f);
    w.close(']');
    w.key("depth_format"); w.num(r.depth_format);
    w.key("stencil_format"); w.num(r.stencil_format);
    w.key("push_stages"); w.num(r.push_stages);
    w.key("push_size"); w.num(r.push_size);
    w.key("bindless"); w.boolean(r.bindless);
    w.key("viewport_with_count"); w.boolean(r.viewport_with_count);
    w.close('}');
    return w.out;
}

inline bool from_json(const Json& o, Recipe& r, std::string* error = nullptr) {
    using namespace detail;
    auto fail = [&](const char* what) {
        if (error) *error = what;
        return false;
    };
    if (o.kind != Json::Kind::object) return fail("not an object");
    std::uint32_t version = 0;
    if (!get_u32(o, "v", version) || version != recipe_version) return fail("unknown recipe version");
    r = Recipe{};
    const Json* stages = member(o, "stages", Json::Kind::array);
    if (!stages || stages->array.empty()) return fail("stages");
    for (const Json& s : stages->array) {
        Stage st;
        const Json* name = member(s, "name", Json::Kind::string);
        if (!name || !get_u32(s, "stage", st.stage) || !get_u32(s, "flags", st.flags) ||
            !get_u32(s, "subgroup", st.required_subgroup) || !parse_hex64(s.find("hash"), st.hash))
            return fail("stage");
        st.name = name->string;
        r.stages.push_back(std::move(st));
    }
    if (!get_bool(o, "mesh", r.mesh)) return fail("mesh");
    const Json* bindings = member(o, "bindings", Json::Kind::array);
    const Json* attributes = member(o, "attributes", Json::Kind::array);
    const Json* divisors = member(o, "divisors", Json::Kind::array);
    if (!bindings || !attributes || !divisors) return fail("vertex input");
    std::vector<std::uint32_t> v;
    for (const Json& b : bindings->array) {
        if (!u32_array(&b, v, 3)) return fail("binding");
        r.bindings.push_back({v[0], v[1], VkVertexInputRate(v[2])});
    }
    for (const Json& a : attributes->array) {
        if (!u32_array(&a, v, 4)) return fail("attribute");
        r.attributes.push_back({v[0], v[1], VkFormat(v[2]), v[3]});
    }
    for (const Json& d : divisors->array) {
        if (!u32_array(&d, v, 2)) return fail("divisor");
        r.divisors.push_back({v[0], v[1]});
    }
    if (!get_u32(o, "topology", r.topology) || !get_bool(o, "restart", r.primitive_restart) ||
        !get_bool(o, "depth_clamp", r.depth_clamp) || !get_bool(o, "discard", r.rasterizer_discard) ||
        !get_u32(o, "polygon", r.polygon_mode) || !get_u32(o, "cull", r.cull_mode) ||
        !get_u32(o, "front_face", r.front_face) || !get_bool(o, "depth_bias", r.depth_bias) ||
        !get_f32(o, "line_width", r.line_width) || !get_bool(o, "conservative", r.conservative))
        return fail("raster");
    const Json* bias = member(o, "bias", Json::Kind::array);
    if (!bias || bias->array.size() != 3) return fail("bias");
    for (const Json& e : bias->array)
        if (e.kind != Json::Kind::number) return fail("bias");
    r.bias_constant = float(bias->array[0].number);
    r.bias_clamp = float(bias->array[1].number);
    r.bias_slope = float(bias->array[2].number);
    if (!get_u32(o, "samples", r.samples) || !get_bool(o, "sample_shading", r.sample_shading) ||
        !get_f32(o, "min_sample_shading", r.min_sample_shading) ||
        !get_bool(o, "alpha_to_coverage", r.alpha_to_coverage) || !get_bool(o, "alpha_to_one", r.alpha_to_one) ||
        !get_u32(o, "sample_mask", r.sample_mask))
        return fail("multisample");
    if (!get_bool(o, "depth_test", r.depth_test) || !get_bool(o, "depth_write", r.depth_write) ||
        !get_u32(o, "depth_compare", r.depth_compare) || !get_bool(o, "depth_bounds", r.depth_bounds) ||
        !get_bool(o, "stencil_test", r.stencil_test) || !read_stencil(o, "front", r.front) ||
        !read_stencil(o, "back", r.back))
        return fail("depth");
    const Json* range = member(o, "depth_bounds_range", Json::Kind::array);
    if (!range || range->array.size() != 2 || range->array[0].kind != Json::Kind::number ||
        range->array[1].kind != Json::Kind::number)
        return fail("depth bounds");
    r.min_depth_bounds = float(range->array[0].number);
    r.max_depth_bounds = float(range->array[1].number);
    if (!get_bool(o, "logic_op_enable", r.logic_op_enable) || !get_u32(o, "logic_op", r.logic_op)) return fail("logic op");
    const Json* constants = member(o, "blend_constants", Json::Kind::array);
    if (!constants || constants->array.size() != 4) return fail("blend constants");
    for (std::size_t i = 0; i < 4; ++i) {
        if (constants->array[i].kind != Json::Kind::number) return fail("blend constants");
        r.blend_constants[i] = float(constants->array[i].number);
    }
    const Json* blends = member(o, "blends", Json::Kind::array);
    if (!blends) return fail("blends");
    for (const Json& b : blends->array) {
        if (!u32_array(&b, v, 8)) return fail("blend");
        r.blends.push_back({VkBool32(v[0]), VkBlendFactor(v[1]), VkBlendFactor(v[2]), VkBlendOp(v[3]),
                            VkBlendFactor(v[4]), VkBlendFactor(v[5]), VkBlendOp(v[6]), VkColorComponentFlags(v[7])});
    }
    if (!u32_array(o.find("colors"), r.colors)) return fail("colors");
    if (!get_u32(o, "depth_format", r.depth_format) || !get_u32(o, "stencil_format", r.stencil_format) ||
        !get_u32(o, "push_stages", r.push_stages) || !get_u32(o, "push_size", r.push_size) ||
        !get_bool(o, "bindless", r.bindless))
        return fail("layout");
    if (o.find("viewport_with_count") && !get_bool(o, "viewport_with_count", r.viewport_with_count))
        return fail("viewport_with_count");
    return true;
}

inline bool from_json(std::string_view line, Recipe& r, std::string* error = nullptr) {
    Json root;
    if (!::leonida::JsonReader(line).parse(root)) {
        if (error) *error = "not JSON";
        return false;
    }
    return from_json(root, r, error);
}

inline std::string to_json(const SetLayoutDesc& d) {
    detail::Writer w;
    w.open('{');
    w.key("v"); w.num(recipe_version);
    w.key("set0_flags"); w.num(d.create_flags);
    w.key("set0"); w.open('[');
    for (const SetBinding& b : d.bindings) {
        w.open('['); w.num(b.binding); w.num(b.type); w.num(b.count); w.num(b.stages); w.num(b.flags); w.close(']');
    }
    w.close(']');
    w.close('}');
    return w.out;
}

inline bool from_json(const Json& o, SetLayoutDesc& d) {
    std::uint32_t version = 0;
    const Json* set = detail::member(o, "set0", Json::Kind::array);
    if (!detail::get_u32(o, "v", version) || version != recipe_version || !set ||
        !detail::get_u32(o, "set0_flags", d.create_flags))
        return false;
    d.bindings.clear();
    std::vector<std::uint32_t> v;
    for (const Json& b : set->array) {
        if (!detail::u32_array(&b, v, 5)) return false;
        d.bindings.push_back({v[0], v[1], v[2], v[3], v[4]});
    }
    return true;
}

// The recipe's identity: two recipes with the same line are the same pipeline.
inline std::uint64_t key(const Recipe& r) {
    const std::string line = to_json(r);
    return fnv1a(line.data(), line.size());
}

// ---------------------------------------------------------------- the recipes file
//
// pso_recipes.jsonl: line 1 is the set-0 layout (to_json(SetLayoutDesc)), every other line one recipe. A file whose
// set-0 line differs from this device's is started again (its pipelines could not be rebuilt the same way), and a
// recipe already in the file is not written twice. Lines are appended and flushed one at a time, so a run that ends
// abruptly loses at most the line being written; the reader skips a partial last line.

struct RecipesFile {
    SetLayoutDesc set0;
    std::vector<Recipe> recipes;
    std::size_t bad_lines = 0;
};

inline bool read_recipes(const std::filesystem::path& path, RecipesFile& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::string line;
    bool header = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        Json root;
        if (!::leonida::JsonReader(line).parse(root) || root.kind != Json::Kind::object) {
            ++out.bad_lines;
            continue;
        }
        if (!header) {
            if (!from_json(root, out.set0)) return false;
            header = true;
            continue;
        }
        Recipe r;
        if (from_json(root, r)) out.recipes.push_back(std::move(r));
        else ++out.bad_lines;
    }
    return header;
}

class Recorder {
public:
    // Opens (or starts again) the file for this device's set-0 layout. A recorder that could not open its file
    // records nothing and says so once.
    Recorder(std::filesystem::path path, const SetLayoutDesc& set0) : path_(std::move(path)) {
        RecipesFile existing;
        const bool same = read_recipes(path_, existing) && existing.set0 == set0;
        if (same)
            for (const Recipe& r : existing.recipes) seen_.insert(key(r));
        out_.open(path_, same ? std::ios::binary | std::ios::app : std::ios::binary | std::ios::trunc);
        if (!out_) {
            std::fprintf(stderr, "[rt] pso: cannot write %s; pipelines are not recorded\n", path_.string().c_str());
            return;
        }
        if (!same) {
            out_ << to_json(set0) << '\n';
            out_.flush();
        }
        ok_ = true;
    }

    // True when the recipe was new and has been written.
    bool record(const Recipe& r) {
        if (!ok_) return false;
        const std::string line = to_json(r);
        const std::uint64_t k = fnv1a(line.data(), line.size());
        std::lock_guard lock(mutex_);
        if (!seen_.insert(k).second) return false;
        out_ << line << '\n';
        out_.flush();
        return bool(out_);
    }

    bool ok() const { return ok_; }
    std::size_t size() const {
        std::lock_guard lock(mutex_);
        return seen_.size();
    }

private:
    std::filesystem::path path_;
    std::ofstream out_;
    mutable std::mutex mutex_;
    std::unordered_set<std::uint64_t> seen_;
    bool ok_ = false;
};

}  // namespace leonida::pso
