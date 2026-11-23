; Multi-argument ExecuteIndirect pixel shader: a flat colour from root constants (the root signature of vs_ma.asm, [0], five
; 32-bit constants). Run: d3d12_indirect_multiarg_test.
; User data: s10, s11, s12 = the colour (floats, k/255; the constants start at s8).   mrt0 = (c0, c1, c2, 1.0)
; fssm-at: 0x10
; fssm-data: 0x414d5350 0x20535020
v_mov_b32_e32 v2, s10
v_mov_b32_e32 v3, s11
v_mov_b32_e32 v4, s12
v_mov_b32_e32 v5, 1.0
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
