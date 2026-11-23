// Shader objects (VK_EXT_shader_object): every translated shader compiled once, on its own, and its driver binary kept
// on disk, so a game run creates its shaders from machine code and compiles nothing (docs/aot_pipelines.md, step 2).
//
// Each shader is independent of the pipeline that uses it: one push-constant range over every graphics stage (and
// one over compute) and one set-0 layout for all of them, so a binary made at build time fits any PSO the game
// creates later. All fixed-function state comes from the PSO's recipe (pso_recipe.h) as dynamic state at the draw.
//
// Binaries are stored as <dir>/<stage>/<name>.bin with a 48-byte header: magic, version, the driver's
// shaderBinaryUUID and shaderBinaryVersion, the SPIR-V hash and the subgroup request they were made with. A binary
// whose header does not match this driver or this SPIR-V is ignored and the shader is made from SPIR-V instead.
#pragma once

#include "pso_recipe.h"

#include <vulkan/vulkan_core.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace leonida::shader_objects {

#define LEONIDA_SHADER_OBJECT_FUNCTIONS(X)                                                                         \
    X(vkCreateShadersEXT) X(vkDestroyShaderEXT) X(vkGetShaderBinaryDataEXT) X(vkCmdBindShadersEXT)                \
    X(vkCmdSetRasterizerDiscardEnable) X(vkCmdSetCullMode) X(vkCmdSetFrontFace) X(vkCmdSetDepthBiasEnable)        \
    X(vkCmdSetDepthBias) X(vkCmdSetPolygonModeEXT) X(vkCmdSetRasterizationSamplesEXT) X(vkCmdSetSampleMaskEXT)    \
    X(vkCmdSetAlphaToCoverageEnableEXT) X(vkCmdSetAlphaToOneEnableEXT) X(vkCmdSetDepthTestEnable)                 \
    X(vkCmdSetDepthWriteEnable) X(vkCmdSetDepthCompareOp) X(vkCmdSetDepthBoundsTestEnable) X(vkCmdSetDepthBounds) \
    X(vkCmdSetStencilTestEnable) X(vkCmdSetStencilOp) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask)   \
    X(vkCmdSetLogicOpEnableEXT) X(vkCmdSetLogicOpEXT) X(vkCmdSetColorBlendEnableEXT)                              \
    X(vkCmdSetColorBlendEquationEXT) X(vkCmdSetColorWriteMaskEXT) X(vkCmdSetVertexInputEXT)                       \
    X(vkCmdSetDepthClampEnableEXT) X(vkCmdSetLineWidth) X(vkCmdSetViewportWithCount) X(vkCmdSetScissorWithCount)  \
    X(vkCmdSetPrimitiveTopology) X(vkCmdSetPrimitiveRestartEnable) X(vkCmdSetBlendConstants)

// Optional: only when the device enabled VK_EXT_conservative_rasterization.
#define LEONIDA_SHADER_OBJECT_OPTIONAL(X) \
    X(vkCmdSetConservativeRasterizationModeEXT) X(vkCmdSetExtraPrimitiveOverestimationSizeEXT)

struct Functions {
#define LEONIDA_SO_MEMBER(name) PFN_##name name = nullptr;
    LEONIDA_SHADER_OBJECT_FUNCTIONS(LEONIDA_SO_MEMBER)
    LEONIDA_SHADER_OBJECT_OPTIONAL(LEONIDA_SO_MEMBER)
#undef LEONIDA_SO_MEMBER

    // True when every required entry point resolved (the device enabled shaderObject and extended dynamic state 3).
    bool load(PFN_vkGetDeviceProcAddr get, VkDevice device) {
        bool ok = true;
#define LEONIDA_SO_LOAD(name)                                                                         \
    name = reinterpret_cast<PFN_##name>(get(device, #name));                                          \
    ok = ok && name;
        LEONIDA_SHADER_OBJECT_FUNCTIONS(LEONIDA_SO_LOAD)
#undef LEONIDA_SO_LOAD
#define LEONIDA_SO_LOAD_OPTIONAL(name) name = reinterpret_cast<PFN_##name>(get(device, #name));
        LEONIDA_SHADER_OBJECT_OPTIONAL(LEONIDA_SO_LOAD_OPTIONAL)
#undef LEONIDA_SO_LOAD_OPTIONAL
        return ok;
    }
};

// The stages a shader object may hand over to: every graphics stage after it that the device has. Shared by the build
// and the runtime, because nextStage is part of the binary's identity.
inline VkShaderStageFlags next_stages(VkShaderStageFlagBits stage, bool geometry) {
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT:
            return VK_SHADER_STAGE_FRAGMENT_BIT | (geometry ? VK_SHADER_STAGE_GEOMETRY_BIT : 0);
        case VK_SHADER_STAGE_GEOMETRY_BIT:
        case VK_SHADER_STAGE_MESH_BIT_EXT:
            return VK_SHADER_STAGE_FRAGMENT_BIT;
        default:
            return 0;
    }
}

// A pipeline stage's create flags as shader-object create flags (the two subgroup bits are the only ones the runtime
// sets), plus NO_TASK_SHADER for a mesh shader: the game has no task shaders (VUID-VkShaderCreateInfoEXT-flags-08414).
inline VkShaderCreateFlagsEXT shader_flags(VkShaderStageFlagBits stage, VkPipelineShaderStageCreateFlags pipeline_flags) {
    VkShaderCreateFlagsEXT flags = 0;
    if (pipeline_flags & VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT)
        flags |= VK_SHADER_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT_EXT;
    if (pipeline_flags & VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT)
        flags |= VK_SHADER_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT;
    if (stage == VK_SHADER_STAGE_MESH_BIT_EXT) flags |= VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT;
    return flags;
}

// The one push-constant range every graphics shader object declares, and the one for compute.
inline VkShaderStageFlags graphics_push_stages(bool geometry, bool mesh) {
    return VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | (geometry ? VK_SHADER_STAGE_GEOMETRY_BIT : 0) |
           (mesh ? VK_SHADER_STAGE_MESH_BIT_EXT : 0);
}

// ---------------------------------------------------------------- stored binaries

inline constexpr std::uint32_t binary_magic = 0x4F53454C;  // "LESO"
inline constexpr std::uint32_t binary_format = 1;

struct BinaryHeader {
    std::uint32_t magic = binary_magic;
    std::uint32_t format = binary_format;
    std::uint8_t uuid[VK_UUID_SIZE]{};
    std::uint32_t driver_version = 0;  // shaderBinaryVersion
    std::uint32_t subgroup = 0;        // required subgroup size the shader was made with (0: none)
    std::uint64_t spirv_hash = 0;
};
static_assert(sizeof(BinaryHeader) == 40);

struct DriverIdentity {
    std::uint8_t uuid[VK_UUID_SIZE]{};
    std::uint32_t version = 0;
    bool operator==(const DriverIdentity& o) const {
        return version == o.version && std::memcmp(uuid, o.uuid, sizeof uuid) == 0;
    }
};

inline const char* stage_folder(VkShaderStageFlagBits stage) {
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT: return "vs";
        case VK_SHADER_STAGE_GEOMETRY_BIT: return "gs";
        case VK_SHADER_STAGE_FRAGMENT_BIT: return "ps";
        case VK_SHADER_STAGE_MESH_BIT_EXT: return "ms";
        case VK_SHADER_STAGE_COMPUTE_BIT: return "cs";
        default: return "other";
    }
}

inline std::filesystem::path binary_path(const std::filesystem::path& dir, VkShaderStageFlagBits stage,
                                         const std::string& name) {
    return dir / stage_folder(stage) / (name + ".bin");
}

// The stored machine code for (stage, name), or empty when there is none for this driver, SPIR-V and subgroup.
inline std::vector<std::uint8_t> read_binary(const std::filesystem::path& path, const DriverIdentity& driver,
                                             std::uint64_t spirv_hash, std::uint32_t subgroup) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    BinaryHeader h;
    if (!in.read(reinterpret_cast<char*>(&h), sizeof h)) return {};
    if (h.magic != binary_magic || h.format != binary_format || h.driver_version != driver.version ||
        std::memcmp(h.uuid, driver.uuid, VK_UUID_SIZE) != 0 || h.spirv_hash != spirv_hash || h.subgroup != subgroup)
        return {};
    std::vector<std::uint8_t> code((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return code;
}

// Writes through a staged file and a rename, so a reader never sees half a binary.
inline bool write_binary(const std::filesystem::path& path, const DriverIdentity& driver, std::uint64_t spirv_hash,
                         std::uint32_t subgroup, const std::vector<std::uint8_t>& code) {
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    BinaryHeader h;
    std::memcpy(h.uuid, driver.uuid, VK_UUID_SIZE);
    h.driver_version = driver.version;
    h.subgroup = subgroup;
    h.spirv_hash = spirv_hash;
    auto staged = path;
    staged += ".part";
    {
        std::ofstream out(staged, std::ios::binary | std::ios::trunc);
        if (!out) return false;
        out.write(reinterpret_cast<const char*>(&h), sizeof h);
        out.write(reinterpret_cast<const char*>(code.data()), std::streamsize(code.size()));
        if (!out) return false;
    }
    std::filesystem::rename(staged, path, ec);
    if (ec) {
        std::filesystem::remove(staged, ec);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- making shaders

struct ShaderSource {
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_VERTEX_BIT;
    std::string name;
    const std::vector<std::uint32_t>* spirv = nullptr;  // used when no binary fits
    std::uint64_t spirv_hash = 0;
    std::uint32_t subgroup = 0;                         // VkShaderRequiredSubgroupSizeCreateInfoEXT, 0: none
    VkShaderCreateFlagsEXT flags = 0;                   // e.g. REQUIRE_FULL_SUBGROUPS / ALLOW_VARYING_SUBGROUP_SIZE
};

struct Layout {
    VkDescriptorSetLayout set0 = VK_NULL_HANDLE;  // null: no set 0
    VkPushConstantRange push{};
};

// Fills a VkShaderCreateInfoEXT; `size_info` must outlive the call that uses `info`.
inline void fill_create_info(const ShaderSource& s, const Layout& layout, bool geometry, const void* code,
                             std::size_t code_size, VkShaderCodeTypeEXT code_type,
                             VkShaderRequiredSubgroupSizeCreateInfoEXT& size_info, VkShaderCreateInfoEXT& info) {
    info = {VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT};
    info.flags = s.flags;
    info.stage = s.stage;
    info.nextStage = next_stages(s.stage, geometry);
    info.codeType = code_type;
    info.codeSize = code_size;
    info.pCode = code;
    info.pName = "main";
    info.setLayoutCount = layout.set0 ? 1u : 0u;
    info.pSetLayouts = layout.set0 ? &layout.set0 : nullptr;
    info.pushConstantRangeCount = 1;
    info.pPushConstantRanges = &layout.push;
    if (s.subgroup) {
        size_info = {VK_STRUCTURE_TYPE_SHADER_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
        size_info.requiredSubgroupSize = s.subgroup;
        info.pNext = &size_info;
    }
}

enum class Origin { binary, spirv, failed };

// One shader object, from the stored binary when it fits, else from SPIR-V.
inline VkShaderEXT create_shader(const Functions& fn, VkDevice device, const ShaderSource& s, const Layout& layout,
                                 bool geometry, const std::filesystem::path& binaries, const DriverIdentity& driver,
                                 Origin* origin = nullptr) {
    VkShaderRequiredSubgroupSizeCreateInfoEXT size_info{};
    VkShaderCreateInfoEXT info{};
    VkShaderEXT shader = VK_NULL_HANDLE;
    if (!binaries.empty()) {
        const std::vector<std::uint8_t> code =
            read_binary(binary_path(binaries, s.stage, s.name), driver, s.spirv_hash, s.subgroup);
        if (!code.empty()) {
            fill_create_info(s, layout, geometry, code.data(), code.size(), VK_SHADER_CODE_TYPE_BINARY_EXT, size_info,
                             info);
            if (fn.vkCreateShadersEXT(device, 1, &info, nullptr, &shader) == VK_SUCCESS && shader) {
                if (origin) *origin = Origin::binary;
                return shader;
            }
            shader = VK_NULL_HANDLE;  // VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT: fall through to SPIR-V
        }
    }
    if (!s.spirv || s.spirv->empty()) {
        if (origin) *origin = Origin::failed;
        return VK_NULL_HANDLE;
    }
    fill_create_info(s, layout, geometry, s.spirv->data(), s.spirv->size() * sizeof(std::uint32_t),
                     VK_SHADER_CODE_TYPE_SPIRV_EXT, size_info, info);
    if (fn.vkCreateShadersEXT(device, 1, &info, nullptr, &shader) != VK_SUCCESS) shader = VK_NULL_HANDLE;
    if (origin) *origin = shader ? Origin::spirv : Origin::failed;
    return shader;
}

// The driver's machine code for a shader object, for the build to store.
inline std::vector<std::uint8_t> binary_of(const Functions& fn, VkDevice device, VkShaderEXT shader) {
    std::size_t size = 0;
    if (fn.vkGetShaderBinaryDataEXT(device, shader, &size, nullptr) != VK_SUCCESS || !size) return {};
    std::vector<std::uint8_t> code(size);
    if (fn.vkGetShaderBinaryDataEXT(device, shader, &size, code.data()) != VK_SUCCESS) return {};
    code.resize(size);
    return code;
}

// ---------------------------------------------------------------- the PSO's state at the draw

// Everything a pipeline built from `r` would have baked in, as dynamic state. `strides[slot]` (slot < stride_count)
// is the stride the draw binds for that vertex-buffer slot, which the pipeline path also takes as dynamic state; a slot
// past it keeps the recipe's. The primitive restart enable and topology stay the replay's (already dynamic there).
inline void emit_state(const Functions& fn, VkCommandBuffer cb, const pso::Recipe& r, bool conservative_supported,
                       const VkDeviceSize* strides = nullptr, std::uint32_t stride_count = 0) {
    fn.vkCmdSetRasterizerDiscardEnable(cb, r.rasterizer_discard);
    fn.vkCmdSetCullMode(cb, r.cull_mode);
    fn.vkCmdSetFrontFace(cb, VkFrontFace(r.front_face));
    fn.vkCmdSetDepthBiasEnable(cb, r.depth_bias);
    fn.vkCmdSetDepthBias(cb, r.bias_constant, r.bias_clamp, r.bias_slope);
    fn.vkCmdSetPolygonModeEXT(cb, VkPolygonMode(r.polygon_mode));
    fn.vkCmdSetDepthClampEnableEXT(cb, r.depth_clamp);
    fn.vkCmdSetLineWidth(cb, r.line_width);
    // With VK_EXT_conservative_rasterization enabled a shader-object draw must have set the mode (and, while it is
    // OVERESTIMATE, the extra size); the pipeline path's 0.0 extra size is kept.
    if (conservative_supported && fn.vkCmdSetConservativeRasterizationModeEXT) {
        fn.vkCmdSetConservativeRasterizationModeEXT(cb, r.conservative
                                                            ? VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT
                                                            : VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);
        if (fn.vkCmdSetExtraPrimitiveOverestimationSizeEXT) fn.vkCmdSetExtraPrimitiveOverestimationSizeEXT(cb, 0.0f);
    }

    fn.vkCmdSetRasterizationSamplesEXT(cb, VkSampleCountFlagBits(r.samples));
    const VkSampleMask mask[2] = {r.sample_mask, ~0u};
    fn.vkCmdSetSampleMaskEXT(cb, VkSampleCountFlagBits(r.samples), mask);
    fn.vkCmdSetAlphaToCoverageEnableEXT(cb, r.alpha_to_coverage);
    fn.vkCmdSetAlphaToOneEnableEXT(cb, r.alpha_to_one);

    fn.vkCmdSetDepthTestEnable(cb, r.depth_test);
    fn.vkCmdSetDepthWriteEnable(cb, r.depth_write);
    fn.vkCmdSetDepthCompareOp(cb, VkCompareOp(r.depth_compare));
    fn.vkCmdSetDepthBoundsTestEnable(cb, r.depth_bounds);
    fn.vkCmdSetDepthBounds(cb, r.min_depth_bounds, r.max_depth_bounds);
    fn.vkCmdSetStencilTestEnable(cb, r.stencil_test);
    fn.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, r.front.failOp, r.front.passOp, r.front.depthFailOp,
                         r.front.compareOp);
    fn.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, r.back.failOp, r.back.passOp, r.back.depthFailOp,
                         r.back.compareOp);
    fn.vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, r.front.compareMask);
    fn.vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, r.back.compareMask);
    fn.vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, r.front.writeMask);
    fn.vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, r.back.writeMask);
    // The stencil reference stays the command list's OMSetStencilRef (already dynamic in the pipeline path).

    fn.vkCmdSetLogicOpEnableEXT(cb, r.logic_op_enable);
    fn.vkCmdSetLogicOpEXT(cb, VkLogicOp(r.logic_op));
    if (const std::uint32_t n = std::uint32_t(r.blends.size())) {
        std::vector<VkBool32> enable(n);
        std::vector<VkColorBlendEquationEXT> eq(n);
        std::vector<VkColorComponentFlags> write(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            const VkPipelineColorBlendAttachmentState& b = r.blends[i];
            enable[i] = b.blendEnable;
            eq[i] = {b.srcColorBlendFactor, b.dstColorBlendFactor, b.colorBlendOp,
                     b.srcAlphaBlendFactor, b.dstAlphaBlendFactor, b.alphaBlendOp};
            write[i] = b.colorWriteMask;
        }
        fn.vkCmdSetColorBlendEnableEXT(cb, 0, n, enable.data());
        fn.vkCmdSetColorBlendEquationEXT(cb, 0, n, eq.data());
        fn.vkCmdSetColorWriteMaskEXT(cb, 0, n, write.data());
    }

    if (!r.mesh) {
        // The draw's strides (the pipeline path's VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE), the recipe's for a
        // slot the draw did not report.
        std::vector<VkVertexInputBindingDescription2EXT> bindings;
        std::vector<VkVertexInputAttributeDescription2EXT> attributes;
        for (const VkVertexInputBindingDescription& b : r.bindings) {
            std::uint32_t divisor = 1;
            for (const auto& d : r.divisors)
                if (d.binding == b.binding) divisor = d.divisor;
            const std::uint32_t stride =
                strides && b.binding < stride_count ? std::uint32_t(strides[b.binding]) : b.stride;
            bindings.push_back({VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, nullptr, b.binding,
                                stride, b.inputRate, divisor});
        }
        for (const VkVertexInputAttributeDescription& a : r.attributes)
            attributes.push_back({VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, nullptr, a.location,
                                  a.binding, a.format, a.offset});
        fn.vkCmdSetVertexInputEXT(cb, std::uint32_t(bindings.size()), bindings.data(),
                                  std::uint32_t(attributes.size()), attributes.data());
    }
}

// The shader objects of one PSO, by stage. Every graphics stage the device has gets a binding at each draw: the PSO's
// shader or VK_NULL_HANDLE (a stage the device enabled but the PSO leaves out must be bound to null explicitly).
struct GraphicsShaders {
    VkShaderEXT vertex = VK_NULL_HANDLE, geometry = VK_NULL_HANDLE, fragment = VK_NULL_HANDLE, mesh = VK_NULL_HANDLE;
};

struct DeviceStages {
    bool geometry = false, tessellation = false, mesh = false, task = false;
};

inline void bind(const Functions& fn, VkCommandBuffer cb, const GraphicsShaders& s, const DeviceStages& device) {
    VkShaderStageFlagBits stages[7];
    VkShaderEXT shaders[7];
    std::uint32_t n = 0;
    const auto add = [&](VkShaderStageFlagBits stage, VkShaderEXT shader) {
        stages[n] = stage;
        shaders[n] = shader;
        ++n;
    };
    add(VK_SHADER_STAGE_VERTEX_BIT, s.vertex);
    add(VK_SHADER_STAGE_FRAGMENT_BIT, s.fragment);
    if (device.geometry) add(VK_SHADER_STAGE_GEOMETRY_BIT, s.geometry);
    if (device.tessellation) {
        add(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, VK_NULL_HANDLE);
        add(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_NULL_HANDLE);
    }
    if (device.mesh) add(VK_SHADER_STAGE_MESH_BIT_EXT, s.mesh);
    if (device.task) add(VK_SHADER_STAGE_TASK_BIT_EXT, VK_NULL_HANDLE);
    fn.vkCmdBindShadersEXT(cb, n, stages, shaders);
}

}  // namespace leonida::shader_objects
