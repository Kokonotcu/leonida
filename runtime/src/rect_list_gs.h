#pragma once

// Rect lists (AMD primitive type 0x11, D3D topology value 17 on the console): three vertices name three corners of
// an axis-aligned rectangle and the hardware draws all of it. Vulkan has no such topology, so a rect-list draw takes a
// pipeline variant with this geometry shader: it emits the given triangle and the second half of the rectangle, whose
// fourth corner is a + b - c for the position and every attribute, c being the corner between the two given edges
// (the vertex opposite the longest edge). The rule is GTA6-BootTest's tools/rect_gs.hlsl (ExtremelyAlexander),
// written here as SPIR-V so the runtime makes one module per vertex-output count without a shader compiler.
//
// The translator's vertex shaders write gl_Position and vec4 outputs at locations 0..n-1 with no interpolation
// decorations (tools/rdna2spirv.py); the GS reads and writes exactly those, so it matches the VS before it and the PS
// after it. Attributes are added and subtracted as floats, raw integer bits included, as the rule does.

#include <cstdint>
#include <cstring>
#include <vector>

namespace leonida {

// The number of user output locations (highest location + 1) of a vertex shader's SPIR-V: OpDecorate Location on
// variables of storage class Output. 0 when it has none or the words are not SPIR-V.
inline std::uint32_t vertex_output_locations(const std::vector<std::uint32_t>& words) {
    if (words.size() < 5 || words[0] != 0x07230203u) return 0;
    std::vector<std::uint32_t> location_of;  // id -> location + 1
    std::vector<bool> output;                // id -> an Output variable
    const std::uint32_t bound = words[3];
    if (bound > (1u << 22)) return 0;
    location_of.assign(bound, 0);
    output.assign(bound, false);
    for (std::size_t i = 5; i < words.size();) {
        const std::uint32_t count = words[i] >> 16, op = words[i] & 0xFFFF;
        if (!count || i + count > words.size()) break;
        if (op == 71 && count >= 4 && words[i + 2] == 30 && words[i + 1] < bound)  // OpDecorate id Location n
            location_of[words[i + 1]] = words[i + 3] + 1;
        if (op == 59 && count >= 4 && words[i + 3] == 3 && words[i + 2] < bound)  // OpVariable type id Output
            output[words[i + 2]] = true;
        i += count;
    }
    std::uint32_t n = 0;
    for (std::uint32_t id = 0; id < bound; ++id)
        if (output[id] && location_of[id] > n) n = location_of[id];
    return n;
}

// The rect-list geometry shader for a vertex shader with `outputs` vec4 output locations.
inline std::vector<std::uint32_t> rect_list_geometry_shader(std::uint32_t outputs) {
    std::vector<std::uint32_t> w;
    std::uint32_t next = 1;
    const auto id = [&] { return next++; };
    const auto op = [&](std::uint32_t opcode, std::initializer_list<std::uint32_t> operands) {
        w.push_back(std::uint32_t(operands.size() + 1) << 16 | opcode);
        w.insert(w.end(), operands.begin(), operands.end());
    };
    // Ids.
    const std::uint32_t t_void = id(), t_fn = id(), t_float = id(), t_v4 = id(), t_v2 = id(), t_uint = id(),
                        t_int = id(), t_bool = id(), c3 = id(), ci0 = id(), ci1 = id(), ci2 = id(), ci3 = id(),
                        f1 = id(), t_pv = id(), t_arr_pv = id(), p_in_arr_pv = id(), p_out_pv = id(),
                        t_arr_v4 = id(), p_in_arr_v4 = id(), p_out_v4 = id(), p_in_v4 = id(), v_in_pv = id(),
                        v_out_pv = id(), fn_main = id();
    std::vector<std::uint32_t> v_in(outputs), v_out(outputs);
    for (std::uint32_t i = 0; i < outputs; ++i) v_in[i] = id(), v_out[i] = id();

    // Header (bound patched at the end), capabilities, memory model, entry point, modes.
    w = {0x07230203u, 0x00010500u, 0u, 0u, 0u};
    op(17, {2});  // OpCapability Geometry
    op(14, {0, 1});  // OpMemoryModel Logical GLSL450
    {
        std::vector<std::uint32_t> e = {3, fn_main, 0x6E69616Du, 0u, v_in_pv, v_out_pv};  // Geometry "main"
        e.insert(e.end(), v_in.begin(), v_in.end());
        e.insert(e.end(), v_out.begin(), v_out.end());
        w.push_back(std::uint32_t(e.size() + 1) << 16 | 15);  // OpEntryPoint
        w.insert(w.end(), e.begin(), e.end());
    }
    op(16, {fn_main, 22});      // Triangles
    op(16, {fn_main, 29});      // OutputTriangleStrip
    op(16, {fn_main, 26, 6});   // OutputVertices 6
    op(16, {fn_main, 0, 1});    // Invocations 1
    // Decorations.
    op(71, {t_pv, 2});             // Block
    op(72, {t_pv, 0, 11, 0});      // member 0 BuiltIn Position
    for (std::uint32_t i = 0; i < outputs; ++i) {
        op(71, {v_in[i], 30, i});  // Location i
        op(71, {v_out[i], 30, i});
    }
    // Types and constants.
    op(19, {t_void});
    op(33, {t_fn, t_void});
    op(22, {t_float, 32});
    op(23, {t_v4, t_float, 4});
    op(23, {t_v2, t_float, 2});
    op(21, {t_uint, 32, 0});
    op(21, {t_int, 32, 1});
    op(20, {t_bool});
    op(43, {t_uint, c3, 3});
    op(43, {t_int, ci0, 0});
    op(43, {t_int, ci1, 1});
    op(43, {t_int, ci2, 2});
    op(43, {t_int, ci3, 3});
    std::uint32_t one;
    const float onef = 1.0f;
    std::memcpy(&one, &onef, 4);
    op(43, {t_float, f1, one});
    op(30, {t_pv, t_v4});
    op(28, {t_arr_pv, t_pv, c3});
    op(32, {p_in_arr_pv, 1, t_arr_pv});
    op(32, {p_out_pv, 3, t_pv});
    op(28, {t_arr_v4, t_v4, c3});
    op(32, {p_in_arr_v4, 1, t_arr_v4});
    op(32, {p_out_v4, 3, t_v4});
    op(32, {p_in_v4, 1, t_v4});
    op(59, {p_in_arr_pv, v_in_pv, 1});
    op(59, {p_out_pv, v_out_pv, 3});
    for (std::uint32_t i = 0; i < outputs; ++i) {
        op(59, {p_in_arr_v4, v_in[i], 1});
        op(59, {p_out_v4, v_out[i], 3});
    }
    // main.
    op(54, {t_void, fn_main, 0, t_fn});
    op(248, {id()});
    const std::uint32_t corner_idx[3] = {ci0, ci1, ci2};
    // Positions of the three vertices, and their screen xy (xy / w).
    std::uint32_t pos[3], xy[3];
    for (int k = 0; k < 3; ++k) {
        const std::uint32_t ptr = id(), v = id(), sw = id(), wv = id(), inv = id(), q = id();
        op(65, {p_in_v4, ptr, v_in_pv, corner_idx[k], ci0});  // OpAccessChain in_pv[k].gl_Position
        op(61, {t_v4, v, ptr});
        op(79, {t_v2, sw, v, v, 0, 1});                        // OpVectorShuffle xy
        op(81, {t_float, wv, v, 3});                           // OpCompositeExtract w
        op(136, {t_float, inv, f1, wv});                       // 1 / w
        op(142, {t_v2, q, sw, inv});                           // OpVectorTimesScalar
        pos[k] = v;
        xy[k] = q;
    }
    const auto edge = [&](int a, int b) {
        const std::uint32_t d = id(), e = id();
        op(131, {t_v2, d, xy[a], xy[b]});  // OpFSub
        op(148, {t_float, e, d, d});       // OpDot
        return e;
    };
    const std::uint32_t e01 = edge(0, 1), e12 = edge(1, 2), e20 = edge(2, 0);
    // Corner 0 when e12 is the longest edge, corner 1 when e20 is (and not corner 0), corner 2 otherwise.
    const std::uint32_t g1 = id(), g2 = id(), c0 = id(), g3 = id(), nc0 = id(), c1 = id();
    op(190, {t_bool, g1, e12, e01});  // OpFOrdGreaterThanEqual
    op(190, {t_bool, g2, e12, e20});
    op(167, {t_bool, c0, g1, g2});    // OpLogicalAnd
    op(190, {t_bool, g3, e20, e01});
    op(168, {t_bool, nc0, c0});       // OpLogicalNot
    op(167, {t_bool, c1, nc0, g3});
    // For a value set (v0, v1, v2): the first edge vertex a, the fourth corner d and the second edge vertex b.
    struct Quad { std::uint32_t v[3], a, d, b; };
    const auto quad = [&](const std::uint32_t v[3]) {
        Quad r{{v[0], v[1], v[2]}, 0, 0, 0};
        std::uint32_t s[3];  // s[k] = sum of the other two minus v[k]
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t sum = id(), dk = id();
            op(129, {t_v4, sum, v[(k + 1) % 3], v[(k + 2) % 3]});  // OpFAdd
            op(131, {t_v4, dk, sum, v[k]});                          // OpFSub
            s[k] = dk;
        }
        const std::uint32_t d12 = id(), a12 = id(), b12 = id();
        op(169, {t_v4, d12, c1, s[1], s[2]});  // OpSelect (scalar condition, SPIR-V 1.4+)
        r.d = id();
        op(169, {t_v4, r.d, c0, s[0], d12});
        op(169, {t_v4, a12, c1, v[2], v[0]});  // a = corner + 1
        r.a = id();
        op(169, {t_v4, r.a, c0, v[1], a12});
        op(169, {t_v4, b12, c1, v[0], v[1]});  // b = corner + 2
        r.b = id();
        op(169, {t_v4, r.b, c0, v[2], b12});
        return r;
    };
    const Quad qp = quad(pos);
    std::vector<Quad> qa;
    for (std::uint32_t i = 0; i < outputs; ++i) {
        std::uint32_t v[3];
        for (int k = 0; k < 3; ++k) {
            const std::uint32_t ptr = id(), val = id();
            op(65, {p_in_v4, ptr, v_in[i], corner_idx[k]});
            op(61, {t_v4, val, ptr});
            v[k] = val;
        }
        qa.push_back(quad(v));
    }
    const auto emit = [&](std::uint32_t p, auto&& attribute) {
        const std::uint32_t ptr = id();
        op(65, {p_out_v4, ptr, v_out_pv, ci0});  // out_pv.gl_Position
        op(62, {ptr, p});                         // OpStore
        for (std::uint32_t i = 0; i < outputs; ++i) op(62, {v_out[i], attribute(i)});
        op(218, {});  // OpEmitVertex
    };
    for (int k = 0; k < 3; ++k) emit(qp.v[k], [&](std::uint32_t i) { return qa[i].v[k]; });
    op(219, {});  // OpEndPrimitive
    emit(qp.a, [&](std::uint32_t i) { return qa[i].a; });
    emit(qp.d, [&](std::uint32_t i) { return qa[i].d; });
    emit(qp.b, [&](std::uint32_t i) { return qa[i].b; });
    op(219, {});
    op(253, {});  // OpReturn
    op(56, {});   // OpFunctionEnd
    (void)ci3;
    w[3] = next;
    return w;
}

}  // namespace leonida
