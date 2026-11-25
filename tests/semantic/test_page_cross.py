"""Multi-dword memory accesses that straddle a 64 KiB title page (Gen.page_span / span_dword_ptr).

host_addr translates an Xbox VA through the runtime's two-level page table (heap_base + 0x30000: directory of 1-based
chunk numbers, one per 4 GiB; chunks at heap_base + 0x70000, 65536 u64 device addresses of 64 KiB pages; unmapped
pages read and write the scratch page at heap_base + 0x10000). The title maps its physical pages scattered, so the VA
page after a mapped one is usually not the next host page. Until 2026-11-25 a dwordx2/x3/x4 access translated only its
first dword and added 4k for dword k: past the page boundary loads read the wrong host memory and stores overwrote it.
Here two adjacent VA pages are mapped to host pages 0x30000 apart, in reverse order, and each access starts 8 bytes
before the boundary, so dwords 0-1 are in the first page and 2-3 in the second. The words the old translation used
(the first host page's next 8 bytes) hold decoys that must be neither read nor written.

Covers global_load/store_dwordx4, buffer_load/store_dwordx4 (raw, OOB_SELECT 3, and the num_records bound cutting the
access inside the second page), s_load_dwordx4, s_buffer_load_dwordx4, buffer_load_format_xyzw / buffer_store_format_xyzw
(V# format 77, 32_32_32_32 float), the 4 GiB-below path (dir[0] tagged chunk), an unmapped second page (scratch page),
an access wholly inside one page, a 16-bit load at byte 3 of the last dword of a page, and spirv-val. Standard library
only; no game files.
"""
import os
import shutil
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402  (adds LEONIDA_TOOLS_DIR / tools to sys.path)
import rdna2spirv  # noqa: E402

M32 = 0xFFFFFFFF
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

HEAP = 0x00000010_00000000                  # heap_base (push constant dwords 32..33)
DIR = HEAP + rdna2spirv.PAGE_TABLE_DIR
CHUNKS = DIR + rdna2spirv.PAGE_TABLE_DIR_BYTES
SCRATCH = HEAP + rdna2spirv.PAGE_TABLE_SCRATCH

VA = 0x00000005_1234FFF8                     # 8 bytes before the end of VA page 0x5_1234
PA_A = 0x00000077_00050000                   # host page of VA page 0x5_1234
PA_B = 0x00000077_00020000                   # host page of VA page 0x5_1235 (not PA_A + 0x10000)
LOW_VA = 0x0002FFF8                          # the same layout below 4 GiB (dir[0] = tag | chunk)

WORDS = (0x11111111, 0x22222222, 0x33333333, 0x44444444)
DECOY = (0xDEC0DE01, 0xDEC0DE02)
SENT = 0x5E5E5E5E


def pc():
    return [0] * 32 + [HEAP & M32, HEAP >> 32, 0, 0]


def map_page(mem, va, pa, slot=1, low=False):
    """Entry of the 64 KiB page holding `va` -> host `pa` in chunk `slot` (1-based), directory entry included."""
    d = (va >> 32) & 0xFFFF
    mem[DIR + 4 * d] = (rdna2spirv.PAGE_TABLE_LOW_TAG | slot) if low else slot
    e = CHUNKS + (slot - 1) * rdna2spirv.PAGE_TABLE_CHUNK_BYTES + ((va >> 16) & 0xFFFF) * 8
    mem[e] = pa & M32
    mem[e + 4] = pa >> 32


def table(va=VA, second=True, low=False):
    """Page table for `va`'s page (-> PA_A) and, when `second`, the next one (-> PA_B); the first host page's next
    8 bytes are decoys (what a first-page-only translation used for dwords 2-3)."""
    mem = {}
    map_page(mem, va, PA_A, low=low)
    if second:
        map_page(mem, va + 0x10000, PA_B, low=low)
    mem[PA_A + 0x10000] = DECOY[0]
    mem[PA_A + 0x10004] = DECOY[1]
    return mem


def seeded(va=VA, second=True, low=False):
    mem = table(va, second, low)
    off = va & 0xFFFF
    mem[PA_A + off] = WORDS[0]
    mem[PA_A + off + 4] = WORDS[1]
    base_b = PA_B if second else SCRATCH
    mem[base_b] = WORDS[2]
    mem[base_b + 4] = WORDS[3]
    return mem


def run_ps(lines, mem):
    spv = support.translate_lines("ps_page_cross", [*lines, "exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"])
    mem = dict(mem)
    words = support.mrt0_words(spv, push_constants=pc(), memory=mem)
    return words, mem, spv


def va_regs(va, lo="v4", hi="v5"):
    return [f"v_mov_b32_e32 {lo}, {va & M32:#x}", f"v_mov_b32_e32 {hi}, {va >> 32:#x}"]


def vsharp(base, records=0x100, word3=0x30000000):
    """s[0:3]: a raw V# (stride 0) at `base`, OOB_SELECT 3 by default."""
    return [f"s_mov_b32 s0, {base & M32:#x}", f"s_mov_b32 s1, {(base >> 32) & 0xFFFF:#x}",
            f"s_mov_b32 s2, {records:#x}", f"s_mov_b32 s3, {word3:#x}"]


def stores(va=VA, second=True):
    """(host address, payload word) of the four dwords of a store at `va`."""
    off = va & 0xFFFF
    base_b = PA_B if second else SCRATCH
    return [(PA_A + off, 0xA0000000), (PA_A + off + 4, 0xA0000001), (base_b, 0xA0000002), (base_b + 4, 0xA0000003)]


PAYLOAD = [f"v_mov_b32_e32 v{6 + k}, {0xA0000000 + k:#x}" for k in range(4)]


class PageCrossLoadTests(unittest.TestCase):
    def test_global_load_dwordx4(self):
        words, _, _ = run_ps([*va_regs(VA), "global_load_dwordx4 v[10:13], v[4:5], off", "s_waitcnt vmcnt(0)"],
                             seeded())
        self.assertEqual(words, list(WORDS))

    def test_global_load_dwordx4_saddr_offset(self):
        base = VA - 0x20
        words, _, _ = run_ps([f"s_mov_b32 s20, {base & M32:#x}", f"s_mov_b32 s21, {base >> 32:#x}",
                              "v_mov_b32_e32 v1, 0x10", "global_load_dwordx4 v[10:13], v1, s[20:21] offset:16",
                              "s_waitcnt vmcnt(0)"], seeded())
        self.assertEqual(words, list(WORDS))

    def test_buffer_load_dwordx4(self):
        words, _, _ = run_ps([*vsharp(VA - 8), "buffer_load_dwordx4 v[10:13], off, s[0:3], 0 offset:8",
                              "s_waitcnt vmcnt(0)"], seeded())
        self.assertEqual(words, list(WORDS))

    def test_buffer_load_dwordx4_num_records_inside_second_page(self):
        # num_records ends after dword 2: dword 3 is out of range (0), dword 2 still comes from the second page.
        words, _, _ = run_ps([*vsharp(VA - 8, records=8 + 12), "v_mov_b32_e32 v13, 0x77",
                              "buffer_load_dwordx4 v[10:13], off, s[0:3], 0 offset:8", "s_waitcnt vmcnt(0)"], seeded())
        self.assertEqual(words, [WORDS[0], WORDS[1], WORDS[2], 0])

    def test_s_load_dwordx4(self):
        words, _, _ = run_ps([f"s_mov_b32 s0, {(VA - 8) & M32:#x}", f"s_mov_b32 s1, {VA >> 32:#x}",
                              "s_load_dwordx4 s[8:11], s[0:1], 0x8", "s_waitcnt lgkmcnt(0)",
                              *(f"v_mov_b32_e32 v{10 + k}, s{8 + k}" for k in range(4))], seeded())
        self.assertEqual(words, list(WORDS))

    def test_s_buffer_load_dwordx4(self):
        words, _, _ = run_ps([*vsharp(VA - 8), "s_buffer_load_dwordx4 s[8:11], s[0:3], 0x8", "s_waitcnt lgkmcnt(0)",
                              *(f"v_mov_b32_e32 v{10 + k}, s{8 + k}" for k in range(4))], seeded())
        self.assertEqual(words, list(WORDS))

    def test_s_buffer_load_dwordx4_runtime_offset(self):
        # an SGPR offset the translator cannot fold: the run-time bound path (one selection, per-dword fallback)
        mem = seeded()
        mem[PA_A + 0xFFF0] = 8                      # s12 = 8, loaded: unknown at translate time
        words, _, _ = run_ps([*vsharp(VA - 8), "s_load_dword s12, s[0:1], 0x0", "s_waitcnt lgkmcnt(0)",
                              "s_buffer_load_dwordx4 s[8:11], s[0:3], s12", "s_waitcnt lgkmcnt(0)",
                              *(f"v_mov_b32_e32 v{10 + k}, s{8 + k}" for k in range(4))], mem)
        self.assertEqual(words, list(WORDS))

    def test_buffer_load_format_xyzw(self):
        # V# format 77 (32_32_32_32 FLOAT, word3 bits 18:12) with DST_SEL XYZW (0xfac): the words come back as stored
        word3 = 0x30000000 | (77 << 12) | 0xFAC
        words, _, _ = run_ps([*vsharp(VA - 8, word3=word3), "buffer_load_format_xyzw v[10:13], off, s[0:3], 0 offset:8",
                              "s_waitcnt vmcnt(0)"], seeded())
        self.assertEqual(words, list(WORDS))

    def test_low_va_tagged_chunk(self):
        words, _, _ = run_ps([*va_regs(LOW_VA), "global_load_dwordx4 v[10:13], v[4:5], off", "s_waitcnt vmcnt(0)"],
                             seeded(LOW_VA, low=True))
        self.assertEqual(words, list(WORDS))

    def test_unmapped_second_page_reads_scratch(self):
        words, _, _ = run_ps([*va_regs(VA), "global_load_dwordx4 v[10:13], v[4:5], off", "s_waitcnt vmcnt(0)"],
                             seeded(second=False))
        self.assertEqual(words, list(WORDS))

    def test_inside_one_page_unchanged(self):
        va = VA - 0x100
        mem = table()
        for k in range(4):
            mem[PA_A + (va & 0xFFFF) + 4 * k] = WORDS[k]
        words, _, _ = run_ps([*va_regs(va), "global_load_dwordx4 v[10:13], v[4:5], off", "s_waitcnt vmcnt(0)"], mem)
        self.assertEqual(words, list(WORDS))

    def test_ushort_at_byte_3_of_last_dword(self):
        # buffer_load_ushort and global_load_ushort at VA page end - 1: the low byte is the first page's last byte,
        # the high byte the second page's first.
        va = VA + 7                                 # 0x...FFFF
        mem = table()
        mem[PA_A + 0xFFFC] = 0xCD000000
        mem[PA_B] = 0x000000AB
        for line in ("buffer_load_ushort v10, off, s[0:3], 0 offset:15", "global_load_ushort v10, v[4:5], off"):
            with self.subTest(line=line):
                words, _, _ = run_ps([*vsharp(VA - 8), *va_regs(va), "v_mov_b32_e32 v11, 0", "v_mov_b32_e32 v12, 0",
                                      "v_mov_b32_e32 v13, 0", line, "s_waitcnt vmcnt(0)"], mem)
                self.assertEqual(words[0], 0xABCD)


class PageCrossStoreTests(unittest.TestCase):
    def check(self, mem, va=VA, second=True):
        for addr, want in stores(va, second):
            self.assertEqual(mem.get(addr), want, hex(addr))
        self.assertEqual(mem[PA_A + 0x10000], DECOY[0])
        self.assertEqual(mem[PA_A + 0x10004], DECOY[1])

    def test_global_store_dwordx4(self):
        _, mem, _ = run_ps([*va_regs(VA), *PAYLOAD, "global_store_dwordx4 v[4:5], v[6:9], off"], table())
        self.check(mem)

    def test_buffer_store_dwordx4(self):
        _, mem, _ = run_ps([*vsharp(VA - 8), *PAYLOAD, "buffer_store_dwordx4 v[6:9], off, s[0:3], 0 offset:8"],
                           table())
        self.check(mem)

    def test_buffer_store_dwordx4_num_records_inside_second_page(self):
        m = table()
        m[PA_B + 4] = SENT
        _, mem, _ = run_ps([*vsharp(VA - 8, records=8 + 12), *PAYLOAD,
                            "buffer_store_dwordx4 v[6:9], off, s[0:3], 0 offset:8"], m)
        self.assertEqual([mem.get(a) for a, _ in stores()[:3]], [w for _, w in stores()[:3]])
        self.assertEqual(mem[PA_B + 4], SENT)                   # dword 3 is past num_records: not written
        self.assertEqual(mem[PA_A + 0x10000], DECOY[0])

    def test_buffer_store_format_xyzw(self):
        word3 = 0x30000000 | (77 << 12) | 0xFAC
        _, mem, _ = run_ps([*vsharp(VA - 8, word3=word3), *PAYLOAD,
                            "buffer_store_format_xyzw v[6:9], off, s[0:3], 0 offset:8"], table())
        self.check(mem)

    def test_16bit_format_store_at_page_end(self):
        # store_small: a 2-byte element at VA page end - 1 (odd, which the translator stores bytewise) puts its high
        # byte in the second page; the V# format (11, 16 UINT) and the tbuffer one (MTBUF format 11) both
        m = table()
        m[PA_A + 0xFFFC] = 0x00123456
        m[PA_B] = 0x77777700
        for word3, line in ((0x30000000 | (11 << 12) | 0xFAC, "buffer_store_format_x v6, off, s[0:3], 0 offset:15"),
                            (0x30000000 | (20 << 12) | 0xFAC,
                             "tbuffer_store_format_x v6, off, s[0:3], format:[BUF_FMT_16_UINT] offset:15")):
            with self.subTest(line=line):
                _, mem, _ = run_ps([*vsharp(VA - 8, word3=word3), "v_mov_b32_e32 v6, 0xabcd", line], m)
                self.assertEqual(mem[PA_A + 0xFFFC], 0xCD123456)
                self.assertEqual(mem[PA_B], 0x777777AB)
                self.assertEqual(mem[PA_A + 0x10000], DECOY[0])

    def test_low_va_and_unmapped_second_page(self):
        with self.subTest("low VA"):
            _, mem, _ = run_ps([*va_regs(LOW_VA), *PAYLOAD, "global_store_dwordx4 v[4:5], v[6:9], off"],
                               table(LOW_VA, low=True))
            self.check(mem, LOW_VA)
        with self.subTest("unmapped second page"):
            _, mem, _ = run_ps([*va_regs(VA), *PAYLOAD, "global_store_dwordx4 v[4:5], v[6:9], off"],
                               table(second=False))
            self.check(mem, VA, second=False)

    def test_single_dword_has_one_table_walk(self):
        # a single-dword access cannot cross: one translation (three table loads), a dwordx2 two
        def table_loads(op):
            spv = support.translate_lines("ps_page_cross_cost", [*va_regs(VA), "v_mov_b32_e32 v6, 1",
                                                                 "v_mov_b32_e32 v7, 2", op,
                                                                 "exp mrt0 v6, v6, v6, v6 done vm", "s_endpgm"])
            return spv.count("OpConvertUToPtr %_ptr_PSB_uint")
        self.assertEqual(table_loads("global_store_dword v[4:5], v6, off"), 3)
        self.assertEqual(table_loads("global_store_dwordx2 v[4:5], v[6:7], off"), 6)

    @unittest.skipUnless(SPIRV_AS and SPIRV_VAL, "requires spirv-as and spirv-val")
    def test_spirv_val(self):
        programs = {
            "global": [*va_regs(VA), *PAYLOAD, "global_load_dwordx4 v[10:13], v[4:5], off", "s_waitcnt vmcnt(0)",
                       "global_store_dwordx4 v[4:5], v[6:9], off"],
            "buffer": [*vsharp(VA - 8), *PAYLOAD, "v_mov_b32_e32 v1, 4",
                       "buffer_load_dwordx4 v[10:13], v1, s[0:3], 0 offen", "s_waitcnt vmcnt(0)",
                       "buffer_store_dwordx4 v[6:9], v1, s[0:3], 0 offen", "buffer_load_ushort v10, v1, s[0:3], 0 offen",
                       "s_waitcnt vmcnt(0)", "buffer_load_format_xyzw v[10:13], v1, s[0:3], 0 offen",
                       "s_waitcnt vmcnt(0)", "buffer_store_format_xyzw v[6:9], v1, s[0:3], 0 offen",
                       "buffer_store_format_xy v[6:7], v1, s[0:3], 0 offen"],
            "scalar": [*vsharp(VA - 8), "s_load_dwordx8 s[8:15], s[0:1], 0x8",
                       "s_buffer_load_dwordx4 s[16:19], s[0:3], s20", "s_waitcnt lgkmcnt(0)",
                       "v_mov_b32_e32 v10, s8", "v_mov_b32_e32 v11, s19"],
        }
        for name, lines in programs.items():
            with self.subTest(name=name):
                spv = support.translate_lines("ps_page_cross_val",
                                              [*lines, "exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"])
                assembled = subprocess.run([SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", "-"],
                                           input=spv.encode("utf-8"), capture_output=True)
                self.assertEqual(assembled.returncode, 0, assembled.stderr.decode("utf-8", "replace"))
                validated = subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.2", "-"], input=assembled.stdout,
                                           capture_output=True)
                self.assertEqual(validated.returncode, 0, validated.stderr.decode("utf-8", "replace"))


if __name__ == "__main__":
    unittest.main()
