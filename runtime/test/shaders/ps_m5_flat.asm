; M5 pixel shader for the multi-threaded lists: a flat colour from root constants (the root signature of vs_m3.asm,
; [0], eight 32-bit constants). Run: d3d12_m5_world_test.
; User data: s13, s14, s15 = c0, c1, c2 (floats, k/255; the constants start at s8).   mrt0 = (c0, c1, c2, 1.0)
; fssm-at: 0x10
; fssm-data: 0x35534650 0x20535020
v_mov_b32_e32 v2, s13
v_mov_b32_e32 v3, s14
v_mov_b32_e32 v4, s15
v_mov_b32_e32 v5, 1.0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
