; M5 mesh shader (NGG, wave64, 128 threads = two waves), written with the corpus's mesh idioms (docs/design.md, "Mesh
; shaders"): the group index from v5 (v_readfirstlane_b32, then >> 7, as all 9,282 corpus mesh shaders do), the
; thread id from the wave index in s3[27:24] and mbcnt, GS_ALLOC_REQ with m0 = prims << 12 | verts (sent by every
; thread with the same m0, as ms_tri.asm does; the corpus sends it from thread 0), and one vertex and one primitive
; export per thread. Run: d3d12_m5_mesh_test (found by the SHA-256 of its FSSM program).
; psv0: 128 128 128
; Each group draws one quad, 4 vertices and 2 triangles, into cell k of an 8x8 grid on a 64x64 target:
;   L = v5 >> 7, the linear group index (x + X * (y + Y * z) of the DispatchMesh grid, as the translator numbers it)
;   k = L + s0 (s0, a root constant: the first cell);   column k & 7, row k >> 3 (cells k >= 64 are off the target)
;   thread t < 4: vertex t at corner (t & 1, t >> 1) of the cell, pos0 = (col / 4 - 1, 1 - row / 4, 0, 1), and
;   param0 = (s1, s2, 0, 1.0), the colour (root constants, floats);   t = 0, 1: triangles (0, 1, 2) and (2, 1, 3)
; The user data are read from s0 as the layer packs root constants today (docs/d3d12_vulkan.md, section 3.5; the corpus
; reads mesh user data from s16, a base the layer does not apply yet). s3 is not user data here: it is the merged wave
; info, so a fourth root dword is out of a mesh shader's reach under that packing. Every value is exact (quarters).
; fssm-at: 0x10
; fssm-data: 0x354d534d 0x20534d20
v_mbcnt_lo_u32_b32 v0, -1, 0
v_mbcnt_hi_u32_b32 v0, -1, v0
s_bfe_u32 s8, s3, 0x40018
v_lshl_add_u32 v0, s8, 6, v0
v_readfirstlane_b32 s9, v5
s_lshr_b32 s9, s9, 7
s_add_u32 s9, s9, s0
s_and_b32 s10, s9, 7
s_lshr_b32 s11, s9, 3
v_and_b32_e32 v1, 1, v0
v_lshrrev_b32_e32 v2, 1, v0
v_and_b32_e32 v2, 1, v2
v_add_nc_u32_e32 v1, s10, v1
v_add_nc_u32_e32 v2, s11, v2
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_mul_f32_e32 v1, 0x3e800000, v1
v_add_f32_e32 v1, -1.0, v1
v_mul_f32_e32 v2, 0xbe800000, v2
v_add_f32_e32 v2, 1.0, v2
v_mov_b32_e32 v3, s1
v_mov_b32_e32 v4, s2
v_mov_b32_e32 v6, 0
v_mov_b32_e32 v7, 1.0
v_mov_b32_e32 v8, 0
v_mov_b32_e32 v10, 0x300402
v_cmp_eq_u32_e32 vcc_lo, 1, v0
v_cndmask_b32_e32 v9, 0x200400, v10, vcc_lo
s_mov_b32 m0, 0x2004
s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)
v_cmpx_gt_u32_e32 4, v0
exp pos0 v1, v2, v8, v7 done
exp param0 v3, v4, v6, v7
v_cmpx_gt_u32_e32 2, v0
exp prim v9, off, off, off done
s_endpgm
