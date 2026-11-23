// pso_recipe.h: a recipe survives its JSON line unchanged, its key follows every field, and GraphicsCreateInfo puts
// each field where the runtime's own pipeline build did. No device needed.
#include "../src/pso_recipe.h"

#include <cstdio>
#include <cstdlib>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL: %s\n", what);
        ++failures;
    }
}

leonida::pso::Recipe sample() {
    leonida::pso::Recipe r;
    r.stages.push_back({VK_SHADER_STAGE_VERTEX_BIT, "vs_0001", 0, 0, 0x0123456789abcdefull});
    r.stages.push_back({VK_SHADER_STAGE_FRAGMENT_BIT, "ps_\"q\"\\x", 2, 64, 0xfedcba9876543210ull});
    r.bindings.push_back({0, 32, VK_VERTEX_INPUT_RATE_VERTEX});
    r.bindings.push_back({1, 16, VK_VERTEX_INPUT_RATE_INSTANCE});
    r.attributes.push_back({0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0});
    r.attributes.push_back({3, 1, VK_FORMAT_R8G8B8A8_UNORM, 12});
    r.divisors.push_back({1, 4});
    r.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    r.primitive_restart = true;
    r.depth_clamp = true;
    r.cull_mode = VK_CULL_MODE_BACK_BIT;
    r.front_face = VK_FRONT_FACE_CLOCKWISE;
    r.depth_bias = true;
    r.bias_constant = -3.0f;
    r.bias_clamp = 0.125f;
    r.bias_slope = 1.0f / 3.0f;
    r.conservative = true;
    r.samples = VK_SAMPLE_COUNT_4_BIT;
    r.sample_shading = true;
    r.min_sample_shading = 0.25f;
    r.alpha_to_coverage = true;
    r.sample_mask = 0x0000000Bu;
    r.depth_test = true;
    r.depth_write = true;
    r.depth_compare = VK_COMPARE_OP_GREATER_OR_EQUAL;
    r.stencil_test = true;
    r.front = {VK_STENCIL_OP_KEEP, VK_STENCIL_OP_REPLACE, VK_STENCIL_OP_INCREMENT_AND_CLAMP, VK_COMPARE_OP_EQUAL,
               0xFF, 0x0F, 7};
    r.back = {VK_STENCIL_OP_ZERO, VK_STENCIL_OP_INVERT, VK_STENCIL_OP_KEEP, VK_COMPARE_OP_ALWAYS, 0x80, 0xFF, 0};
    r.min_depth_bounds = 0.1f;
    r.max_depth_bounds = 0.9f;
    r.logic_op_enable = true;
    r.logic_op = VK_LOGIC_OP_XOR;
    r.blend_constants[0] = 0.5f;
    r.blend_constants[3] = 1.0f;
    r.blends.push_back({VK_TRUE, VK_BLEND_FACTOR_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, VK_BLEND_OP_ADD,
                        VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_MAX, 0x7});
    r.blends.push_back({VK_FALSE, VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, VK_BLEND_FACTOR_ONE,
                        VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD, 0xF});
    r.colors = {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_UNDEFINED};
    r.depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    r.stencil_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    r.push_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    r.push_size = 144;
    r.bindless = true;
    return r;
}

}  // namespace

int main() {
    using namespace leonida::pso;
    const Recipe a = sample();
    const std::string line = to_json(a);
    check(line.find('\n') == std::string::npos, "one line");
    Recipe b;
    std::string error;
    check(from_json(line, b, &error), ("parse: " + error).c_str());
    check(to_json(b) == line, "round trip gives the same line");
    check(key(a) == key(b), "same key after a round trip");
    check(b.stages[1].name == "ps_\"q\"\\x" && b.stages[1].hash == 0xfedcba9876543210ull, "stage name and hash");
    check(b.bias_slope == a.bias_slope && b.bias_clamp == 0.125f, "floats survive exactly");
    check(b.front.reference == 7 && b.back.compareMask == 0x80, "stencil states");
    check(b.divisors.size() == 1 && b.divisors[0].divisor == 4, "divisors");

    Recipe c = a;
    c.blends[0].colorWriteMask = 0xF;
    check(key(c) != key(a), "a blend field changes the key");
    c = a;
    c.stages[0].hash ^= 1;
    check(key(c) != key(a), "the shader hash changes the key");

    check(!from_json(std::string_view("{\"v\":99}"), b), "an unknown version is refused");
    check(!from_json(std::string_view("not json"), b), "garbage is refused");
    std::string cut = line.substr(0, line.size() / 2);
    check(!from_json(cut, b), "a truncated line is refused");

    // GraphicsCreateInfo: every field lands where the runtime's build() put it.
    std::vector<VkShaderModule> modules(2, VK_NULL_HANDLE);
    GraphicsCreateInfo ci(a, modules, VK_NULL_HANDLE);
    const VkGraphicsPipelineCreateInfo& i = ci.info;
    check(i.stageCount == 2 && i.pStages[1].stage == VK_SHADER_STAGE_FRAGMENT_BIT && i.pStages[1].flags == 2,
          "stages");
    const auto* size = static_cast<const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*>(i.pStages[1].pNext);
    check(size && size->requiredSubgroupSize == 64 && !i.pStages[0].pNext, "required subgroup size only where set");
    const auto* rendering = static_cast<const VkPipelineRenderingCreateInfo*>(i.pNext);
    check(rendering && rendering->colorAttachmentCount == 2 &&
              rendering->pColorAttachmentFormats[1] == VK_FORMAT_UNDEFINED &&
              rendering->stencilAttachmentFormat == VK_FORMAT_D32_SFLOAT_S8_UINT,
          "rendering formats");
    check(i.pVertexInputState->vertexAttributeDescriptionCount == 2 && i.pVertexInputState->pNext, "vertex input");
    check(i.pInputAssemblyState->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP &&
              i.pInputAssemblyState->primitiveRestartEnable,
          "input assembly");
    check(i.pRasterizationState->pNext && i.pRasterizationState->depthBiasSlopeFactor == a.bias_slope,
          "raster with conservative chained");
    check(*i.pMultisampleState->pSampleMask == 0xB && i.pMultisampleState->rasterizationSamples == 4, "multisample");
    check(i.pDepthStencilState->front.writeMask == 0x0F && i.pDepthStencilState->maxDepthBounds == 0.9f, "depth");
    check(i.pColorBlendState->attachmentCount == 2 && i.pColorBlendState->logicOp == VK_LOGIC_OP_XOR &&
              i.pColorBlendState->blendConstants[0] == 0.5f,
          "blend");
    check(i.pDynamicState->dynamicStateCount == 7, "seven dynamic states with vertex bindings");
    Recipe no_vb = a;
    no_vb.bindings.clear();
    no_vb.attributes.clear();
    no_vb.divisors.clear();
    GraphicsCreateInfo ci2(no_vb, modules, VK_NULL_HANDLE);
    check(ci2.info.pDynamicState->dynamicStateCount == 6 && !ci2.info.pVertexInputState->pNext,
          "no stride state without bindings");
    Recipe mesh = no_vb;
    mesh.mesh = true;
    GraphicsCreateInfo ci3(mesh, modules, VK_NULL_HANDLE);
    check(ci3.info.pDynamicState->dynamicStateCount == 4 && !ci3.info.pVertexInputState &&
              !ci3.info.pInputAssemblyState,
          "mesh pipelines take no vertex input");

    LayoutCreateInfo layout(a, reinterpret_cast<VkDescriptorSetLayout>(std::uintptr_t(1)));
    check(layout.info.setLayoutCount == 1 && layout.info.pPushConstantRanges->size == 144, "layout with set 0");
    Recipe plain = a;
    plain.bindless = false;
    LayoutCreateInfo layout2(plain, VK_NULL_HANDLE);
    check(layout2.info.setLayoutCount == 0, "layout without set 0");

    SetLayoutDesc d;
    d.create_flags = 2;
    d.bindings.push_back({0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 65536, VK_SHADER_STAGE_ALL, 0xD});
    SetLayoutDesc e;
    leonida::Json root;
    check(leonida::JsonReader(to_json(d)).parse(root) && from_json(root, e) && e == d, "set layout round trip");

    // The recipes file: header, dedup, reopen with the same layout keeps it, another layout starts it again.
    const auto path = std::filesystem::temp_directory_path() / "leonida_pso_recipe_test.jsonl";
    std::filesystem::remove(path);
    {
        Recorder rec(path, d);
        check(rec.ok(), "recorder opens");
        check(rec.record(a), "first recipe written");
        check(!rec.record(a), "same recipe not written twice");
        check(rec.record(no_vb), "second recipe written");
    }
    {
        Recorder rec(path, d);
        check(rec.size() == 2 && !rec.record(a), "reopened recorder knows what the file holds");
    }
    {
        std::ofstream torn(path, std::ios::binary | std::ios::app);
        torn << "{\"v\":1,\"stages\":[";  // a run that died mid-line
    }
    RecipesFile file;
    check(read_recipes(path, file) && file.set0 == d && file.recipes.size() == 2 && file.bad_lines == 1,
          "reader keeps whole lines and skips a torn one");
    check(file.recipes.size() == 2 && key(file.recipes[0]) == key(a), "recipes read back in order");
    SetLayoutDesc other = d;
    other.bindings[0].count = 1024;
    {
        Recorder rec(path, other);
        check(rec.size() == 0, "another set-0 layout starts the file again");
    }
    RecipesFile again;
    check(read_recipes(path, again) && again.set0 == other && again.recipes.empty(), "restarted file is empty");
    std::filesystem::remove(path);

    if (failures) return 1;
    std::printf("pso_recipe_test: all checks passed\n");
    return 0;
}
