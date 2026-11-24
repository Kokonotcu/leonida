"""SV_ViewportArrayIndex (M4, cascaded shadow maps): the stage that writes it gets BuiltIn ViewportIndex.

- A vertex shader writes it in the pos1 misc vector, z[19:16] (OSG1 system value 5): ShaderViewportIndex.
- A legacy geometry shader writes it as a uint in one component of its output register: each emit also stores that
  component to ViewportIndex, under MultiViewport (the capability a geometry shader declares it with), and the
  register stays a Location output for the pixel shader.
Standard library only; no GPU, no LLVM, no game files.
"""
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402

GS_TRI = os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm")


def signature(elements):
    """An ISG1/OSG1 payload of unnamed elements (system value, component type, register, mask)."""
    return struct.pack("<II", len(elements), 8) + b"".join(
        struct.pack("<6I2B6x", 0, 0, 0, sv, ctype, reg, mask, mask) for sv, ctype, reg, mask in elements)


def container(chunks):
    """A DXBC container of (fourcc, payload) chunks."""
    header = 32 + 4 * len(chunks)
    offsets, body = [], b""
    for fourcc, payload in chunks:
        offsets.append(header + len(body))
        body += fourcc + struct.pack("<I", len(payload)) + payload
    return b"DXBC" + bytes(16) + struct.pack("<3I", 1, header + len(body), len(chunks)) + \
        b"".join(struct.pack("<I", o) for o in offsets) + body


def translate(name, lines, dxbc):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(dxbc)
        src = os.path.join(tmp, "disasm", f"{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in lines))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def gs_tri_lines():
    with open(GS_TRI, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith(";")]


class ViewportIndexTests(unittest.TestCase):
    def test_vertex_shader_misc_vector(self):
        osg1 = signature([(1, 3, 0, 0xF), (5, 1, 1, 0x1)])
        spv = translate("vs_vpindex_sem", [
            "v_mov_b32_e32 v1, 0",
            "v_mov_b32_e32 v2, 1.0",
            "v_lshlrev_b32_e32 v3, 16, v3",
            "exp pos0 v1, v1, v1, v2",
            "exp pos1 v1, v1, v3, v1 done",
            "s_endpgm",
        ], container([(b"OSG1", osg1)]))
        self.assertIn("OpCapability ShaderViewportIndex", spv)
        self.assertNotIn("OpCapability MultiViewport", spv)
        self.assertIn("OpDecorate %out_viewport BuiltIn ViewportIndex", spv)
        self.assertRegex(spv, r"OpStore %out_viewport ")

    def test_geometry_shader_component(self):
        isg1 = signature([(1, 3, 0, 0xF), (0, 3, 1, 0xF)])
        # TEXCOORD0.xyz and SV_ViewportArrayIndex in .w of register 1, as fxc packs them.
        osg1 = signature([(1, 3, 0, 0xF), (0, 3, 1, 0x7), (5, 1, 1, 0x8)])
        spv = translate("gs_vpindex_sem", gs_tri_lines(), container([(b"ISG1", isg1), (b"OSG1", osg1)]))
        self.assertIn("OpCapability MultiViewport", spv)
        self.assertNotIn("OpCapability ShaderViewportIndex", spv)
        self.assertIn("OpDecorate %out_viewport BuiltIn ViewportIndex", spv)
        self.assertIn("OpDecorate %gs_out1 Location 1", spv)
        # Every emit: component 3 of register 1, as an int, to ViewportIndex.
        emits = spv.count("OpEmitVertex")
        self.assertEqual(emits, 3)
        stores = re.findall(r"(%\S+) = OpCompositeExtract %float \S+ 3\n\s*(%\S+) = OpBitcast %int \1\n"
                            r"\s*OpStore %out_viewport \2", spv)
        self.assertEqual(len(stores), emits)
        self.assertIn("%out_viewport", re.search(r"OpEntryPoint Geometry .*", spv).group(0))

    def test_geometry_shader_without_it(self):
        sig = signature([(1, 3, 0, 0xF), (0, 3, 1, 0xF)])
        spv = translate("gs_novp_sem", gs_tri_lines(), container([(b"ISG1", sig), (b"OSG1", sig)]))
        self.assertNotIn("ViewportIndex", spv)
        self.assertNotIn("MultiViewport", spv)


if __name__ == "__main__":
    unittest.main()
