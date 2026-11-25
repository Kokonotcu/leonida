"""Semantic and structural tests for GPU virtual-address page-table translation in tools/rdna2spirv.py.

When PAGE_TABLE is enabled (default), Gen.host_addr maps:
  * VA < 4 GiB  -> heap_base + VA (direct offset in the GPU address arena)
  * VA >= 4 GiB -> two-level lookup at heap_base + PAGE_TABLE_DIR (dir[(va >> 32) & 0xffff] gives
    1-based chunk number; chunk[(va >> 16) & 0xffff] gives 64 KiB page device address; low 16 bits added),
    falling back to heap_base + PAGE_TABLE_SCRATCH + (va & 0xffff) when unmapped.

Also verifies:
  * --no-page-table CLI option disables page-table lookups cleanly.
  * Inactive or out-of-bounds buffer_load, multi-dword buffer_store, and s_buffer_load skip host_addr
    (avoiding wasted page-table memory reads on OOB/inactive lanes).
  * Generated SPIR-V passes spirv-as / spirv-val when available.

Standard library only; no game files.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import lane_masks  # noqa: E402
import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import support  # noqa: E402


def _assemble_and_validate(spvasm: str) -> None:
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_path = os.path.join(tmp, "s.spvasm")
        spv_path = os.path.join(tmp, "s.spv")
        with open(asm_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        subprocess.run([spirv_as, "--target-env", "vulkan1.3", asm_path, "-o", spv_path],
                       check=True, capture_output=True, text=True)
        subprocess.run([spirv_val, "--target-env", "vulkan1.3", spv_path],
                       check=True, capture_output=True, text=True)


class ReadLog(dict):
    def __init__(self, *args):
        super().__init__(*args)
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)


class PageTableTests(unittest.TestCase):
    def test_arena_va_below_4gib_reads_directly_without_table_entries(self):
        spv = support.translate_lines("ps_pt_arena", [
            "s_mov_b32 s4, 0x2000",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 0x100",
            "s_mov_b32 s7, 0x31016fac",
            "buffer_load_dword v2, off, s[4:7], 0 offset:8",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ], page_table=True)
        _assemble_and_validate(spv)
        got = support.mrt0_words(spv, memory={0x2008: 0x12345678})
        self.assertEqual(got, [0x12345678] * 4)

    def test_high_va_uses_two_level_page_table_and_scratch_fallback(self):
        # VA = 0x0003_0002_0008 -> chunk_dir_idx = 3, page_idx = 2, page_off = 8.
        # Unmapped VA = 0x0004_0001_000C -> falls back to PAGE_TABLE_SCRATCH + 0x0C.
        chunk_table_base = rdna2spirv.PAGE_TABLE_DIR + rdna2spirv.PAGE_TABLE_DIR_BYTES
        # Chunk 1-based slot = 2 -> chunk offset = (2 - 1) * PAGE_TABLE_CHUNK_BYTES.
        entry_addr = chunk_table_base + rdna2spirv.PAGE_TABLE_CHUNK_BYTES + (2 * 8)
        target_host_page = 0x00000010_00050000
        mem = {
            rdna2spirv.PAGE_TABLE_DIR + 3 * 4: 2,
            entry_addr: target_host_page & 0xFFFFFFFF,
            entry_addr + 4: target_host_page >> 32,
            target_host_page + 8: 0xAABBCCDD,
            rdna2spirv.PAGE_TABLE_SCRATCH + 0x0C: 0x55667788,
            rdna2spirv.PAGE_TABLE_SCRATCH + 0x10: 0x99887766,
        }
        spv = support.translate_lines("ps_pt_lookup", [
            "s_mov_b32 s4, 0x00020008",
            "s_mov_b32 s5, 0x0003",
            "s_mov_b32 s6, 0x100",
            "s_mov_b32 s7, 0x31016fac",
            "s_mov_b32 s8, 0x0001000C",
            "s_mov_b32 s9, 0x0004",
            "s_mov_b32 s10, 0x100",
            "s_mov_b32 s11, 0x31016fac",
            # Wild 64-bit SMEM pointer with bits [63:48] non-zero (0x0001_0003_0002_0010): must not alias dir[3].
            "s_mov_b32 s12, 0x00020010",
            "s_mov_b32 s13, 0x00010003",
            "buffer_load_dword v2, off, s[4:7], 0",
            "buffer_load_dword v3, off, s[8:11], 0",
            "s_load_dword s14, s[12:13], 0",
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "v_mov_b32 v4, s14",
            "exp mrt0 v2, v3, v4, v3 done vm",
            "s_endpgm",
        ], page_table=True)
        _assemble_and_validate(spv)
        got = support.mrt0_words(spv, memory=mem)
        self.assertEqual(got, [0xAABBCCDD, 0x55667788, 0x99887766, 0x55667788])

    def test_oob_buffer_load_and_store_skip_page_table_lookups(self):
        spv = support.translate_lines("ps_pt_oob_skip", [
            "s_mov_b32 s4, 0x2000",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 4",
            "s_mov_b32 s7, 0x31016fac",
            "v_mov_b32 v1, 0x10",
            "v_mov_b32 v4, 0x11111111",
            "v_mov_b32 v5, 0x22222222",
            "buffer_load_dwordx4 v[8:11], v1, s[4:7], 0 offen",
            "buffer_store_dwordx2 v[4:5], v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v8, v9, v10, v11 done vm",
            "s_endpgm",
        ], page_table=True)
        _assemble_and_validate(spv)
        mem = ReadLog({})
        got = support.mrt0_words(spv, memory=mem)
        self.assertEqual(got, [0, 0, 0, 0])
        self.assertEqual(mem.reads, [])

    def test_multi_dword_partial_edge_buffer_load_and_store_with_page_table(self):
        # Straddles NR=8: dword 0 at +0 (in), dword 1 at +4 (in), dword 2 at +8 (OOB), dword 3 at +12 (OOB).
        spv = support.translate_lines("ps_pt_partial", [
            "s_mov_b32 s4, 0x2000",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 8",
            "s_mov_b32 s7, 0x31016fac",
            "buffer_load_dwordx4 v[8:11], off, s[4:7], 0",
            "v_mov_b32 v2, 0xCAFE0001",
            "v_mov_b32 v3, 0xCAFE0002",
            "buffer_store_dwordx2 v[2:3], off, s[4:7], 0 offset:4",
            "s_waitcnt vmcnt(0)",
            "exp mrt0 v8, v9, v10, v11 done vm",
            "s_endpgm",
        ], page_table=True)
        _assemble_and_validate(spv)
        mem = {0x2000: 0x11111111, 0x2004: 0x22222222, 0x2008: 0x33333333, 0x200C: 0x44444444}
        got = support.mrt0_words(spv, memory=mem)
        self.assertEqual(got, [0x11111111, 0x22222222, 0, 0])
        self.assertEqual(mem[0x2004], 0xCAFE0001)
        self.assertEqual(mem[0x2008], 0x33333333)

    def test_s_buffer_load_static_oob_emits_no_page_table_loads(self):
        g = rdna2spirv.Gen("ps")
        g.sgpr_imm[4] = 0x2000
        g.sgpr_imm[5] = 0x00040000
        g.sgpr_imm[6] = 4   # limit = 4 * 4 = 16 bytes
        g.sgpr_imm[7] = 0x31000FAC
        ins = parse_line("s_buffer_load_dword s0, s[4:7], 0x40")
        prev = rdna2spirv.PAGE_TABLE
        rdna2spirv.PAGE_TABLE = True
        try:
            g.scalar_load(ins["op"], ins["args"], ins["mods"])
        finally:
            rdna2spirv.PAGE_TABLE = prev
        text = "\n".join(g.body)
        self.assertNotIn("Aligned 4", text)
        self.assertNotIn("OpConvertUToPtr", text)

    def test_no_page_table_cli_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            dis = os.path.join(tmp, "disasm")
            out = os.path.join(tmp, "out")
            os.mkdir(dis)
            s_path = os.path.join(dis, "ps_cli_pt.s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cli test\n\ts_mov_b32 s4, 0x2000\n\ts_mov_b32 s5, 0\n"
                        "\ts_load_dword s0, s[4:5], 0\n\ts_waitcnt lgkmcnt(0)\n"
                        "\tv_mov_b32 v0, s0\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n")
            with open(os.path.join(tmp, "ps_cli_pt.dxbc"), "wb") as f:
                f.write(support.STUB_DXBC)
            proc = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS_DIR, "rdna2spirv.py"),
                 dis, "--out", out, "--assembler", "python", "--no-validate", "--no-page-table"],
                capture_output=True, text=True, check=True,
            )
            self.assertEqual(proc.returncode, 0)
            with open(os.path.join(out, "ps_cli_pt.spvasm"), encoding="utf-8") as f:
                spvasm = f.read()
            self.assertNotIn("%cl30000", spvasm)
            self.assertNotIn("%cl70000", spvasm)


class TranslatorAuditRegressionTests(unittest.TestCase):
    def test_dpp_5op_co_ci_shuffles_src0_instead_of_carry_dst(self):
        spv = support.translate_lines("ps_dpp_co_ci", [
            "s_mov_b32 vcc_lo, 0",
            "v_mov_b32 v1, 10",
            "v_mov_b32 v3, 20",
            "v_add_co_ci_u32_dpp v2, vcc_lo, v1, v3, vcc_lo row_shr:1 row_mask:0xf bank_mask:0xf",
            "exp mrt0 v2, v2, v2, v2 done vm",
            "s_endpgm",
        ])
        _assemble_and_validate(spv)

    def test_mov64_scalar_vcc_preserves_high_dword_in_wave32(self):
        g = rdna2spirv.Gen("cs")
        g.wave64 = False
        for line in (
            "s_mov_b32 s2, 0x11223344",
            "s_mov_b32 s3, 0x55667788",
            "s_mov_b64 vcc, s[2:3]",
            "s_mov_b64 s[4:5], vcc",
        ):
            g.insn(parse_line(line))
        body = "\n".join(g.body)
        self.assertIn("OpStore %vcc", body)
        self.assertIn("OpStore %vcc_hi", body)
        self.assertIn("OpLoad %uint %vcc_hi", body)

    def test_lane_masks_3op_co_ci_marks_implicit_vcc_not_src0(self):
        state = {}
        ins = parse_line("v_add_co_ci_u32_e32 v2, s4, v3")
        lane_masks.step(state, ins, wave64=False)
        self.assertNotIn(4, state)
        self.assertIn("vcc", state)

    def test_image_bvh64_guards_host_addr_with_active_check(self):
        g = rdna2spirv.Gen("cs")
        ins = parse_line("image_bvh64_intersect_ray v[4:7], v[8:19], s[8:11]")
        g.image_bvh(ins["op"], ins["args"], ins["mods"])
        body = "\n".join(g.body)
        branch_pos = body.find("OpBranchConditional")
        host_load_pos = body.find("OpConvertUToPtr %_ptr_PSB_arr")
        self.assertGreaterEqual(branch_pos, 0)
        self.assertGreater(host_load_pos, branch_pos)


if __name__ == "__main__":
    unittest.main()
