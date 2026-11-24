"""W-range 3D UAVs (docs/design.md, "W-range 3D UAVs"): a 3D `image_load` / `image_load_mip` reads the sampled 3D
array, which views the whole volume, so it adds the T#'s first slice (BASE_ARRAY, word 4 [28:16]) to z when the T# is
a sliced view (ARRAY_PITCH, word 5 [3:0], = 1). A 3D `image_store` goes through the storage view, which the runtime
makes sliced, so its z is used as it is. Standard library only; no GPU, no LLVM, no game files.
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

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
FSSM = bytearray(0x148)
struct.pack_into("<4I", FSSM, 0x138, 0x88, 32, 1, 1)  # 4 user SGPRs, 32x1x1
FSSM = bytes(FSSM)


def translate_cs(body):
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("disasm", "fssm"):
            os.mkdir(os.path.join(tmp, d))
        name = "cs_img_3d_sliced"
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", f"{name}.fssm"), "wb") as f:
            f.write(FSSM)
        src = os.path.join(tmp, "disasm", f"{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(src)


def constants(spv):
    """uint constant ids by value."""
    out = {}
    for m in re.finditer(r"^\s*(%\S+) = OpConstant %uint (\d+)\s*$", spv, re.M):
        out.setdefault(int(m.group(2)), set()).add(m.group(1))
    return out


def slice_offsets(spv):
    """How many times the module extracts ARRAY_PITCH (bits 0..3) and BASE_ARRAY (bits 16..28) of a T# word."""
    c = constants(spv)
    zero, four, sixteen, thirteen = (c.get(v, set()) for v in (0, 4, 16, 13))
    pitch = base = 0
    for m in re.finditer(r"OpBitFieldUExtract %uint \S+ (\S+) (\S+)", spv):
        off, cnt = m.groups()
        pitch += off in zero and cnt in four
        base += off in sixteen and cnt in thirteen
    return pitch, base


class Image3dSlicedTests(unittest.TestCase):
    def test_load_adds_the_first_slice_of_a_sliced_tsharp(self):
        spv = translate_cs([
            "s_load_dwordx8 s[8:15], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "image_load v[4:7], v[0:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_3D unorm",
            "s_waitcnt vmcnt(0)",
        ])
        pitch, base = slice_offsets(spv)
        self.assertEqual((pitch, base), (1, 1))
        self.assertRegex(spv, r"OpSelect %uint \S+ \S+ \S+")
        self.assertRegex(spv, r"OpIAdd %int ")

    def test_load_mip_adds_the_first_slice(self):
        spv = translate_cs([
            "s_load_dwordx8 s[8:15], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "image_load_mip v[4:7], v[0:3], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_3D unorm",
            "s_waitcnt vmcnt(0)",
        ])
        self.assertEqual(slice_offsets(spv), (1, 1))

    def test_store_and_2d_load_do_not(self):
        store = translate_cs([
            "s_load_dwordx8 s[8:15], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "image_store v[4:7], v[0:2], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_3D unorm",
        ])
        self.assertEqual(slice_offsets(store), (0, 0))
        load2d = translate_cs([
            "s_load_dwordx8 s[8:15], s[0:1], 0x0",
            "s_waitcnt lgkmcnt(0)",
            "image_load v[4:7], v[0:1], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm",
            "s_waitcnt vmcnt(0)",
        ])
        self.assertEqual(slice_offsets(load2d), (0, 0))


if __name__ == "__main__":
    unittest.main()
