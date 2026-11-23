"""runtime/test/shaders/build.py: the PSV0 container of a mesh shader with a "psv0:" header.

build.py used to skip every ms_ shader (it made no PSV0, which the translator reads a mesh shader's thread count and
output counts from). A mesh shader whose header has "; psv0: THREADS MAX_VERTICES MAX_PRIMITIVES" now gets one, so a
layer test (d3d12_m5_mesh_test) can use a translated mesh shader. The container must be the one
scripts/run_gpu_tests.py builds for its mesh fixtures (make_dxbc + make_ms_psv0), byte for byte.

Standard library only; no llvm-mc, SPIR-V tools, GPU or game data.

    python -I tests/test_shader_build_mesh.py
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "runtime", "test", "shaders", "build.py")
GPU_TESTS = os.path.join(ROOT, "scripts", "run_gpu_tests.py")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MeshContainerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = load(BUILD, "shaders_build_mesh_under_test")
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        cls.gpu = load(GPU_TESTS, "run_gpu_tests_mesh_under_test")

    def test_container_matches_run_gpu_tests(self):
        for threads, verts, prims in ((128, 128, 128), (128, 64, 64), (32, 3, 1)):
            want = self.gpu.make_dxbc([(b"PSV0", self.gpu.make_ms_psv0(threads, verts, prims))])
            self.assertEqual(self.build.ms_container(threads, verts, prims), want)

    def test_psv0_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ms_x.asm"
            path.write_text("; a mesh shader\n; psv0: 128 64 32\n; fssm-at: 0x10\ns_endpgm\n", encoding="utf-8")
            self.assertEqual(self.build.asm_psv0(path), (128, 64, 32))
            path.write_text("; a mesh shader\ns_endpgm\n; psv0: 128 64 32\n", encoding="utf-8")
            self.assertIsNone(self.build.asm_psv0(path))  # only the header counts

    def test_the_m5_mesh_shader_is_built_and_the_old_fixtures_are_not(self):
        here = os.path.dirname(BUILD)
        self.assertEqual(self.build.asm_psv0(os.path.join(here, "ms_m5.asm")), (128, 128, 128))
        for name in ("ms_tri", "ms_v5_workgroup", "ms_clip"):  # run_gpu_tests.py makes their containers
            self.assertIsNone(self.build.asm_psv0(os.path.join(here, name + ".asm")))

    def test_run_gpu_tests_reads_the_same_header(self):
        # run_gpu_tests.py translates every .asm in the folder itself; without the header's PSV0 ms_m5 fails there.
        here = os.path.dirname(BUILD)
        for name in ("ms_m5", "ms_tri"):
            path = os.path.join(here, name + ".asm")
            self.assertEqual(self.gpu.asm_psv0(path), self.build.asm_psv0(path))


if __name__ == "__main__":
    unittest.main()
