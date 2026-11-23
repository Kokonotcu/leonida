; M5 pixel shader for the mesh draws: mrt0 = param0, the colour ms_m5.asm passes (constant over each quad, so the
; interpolation returns it exactly). Run: d3d12_m5_mesh_test.
; fssm-at: 0x10
; fssm-data: 0x48534d50 0x20535020
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
v_interp_p1_f32_e32 v4, v0, attr0.z
v_interp_p2_f32_e32 v4, v1, attr0.z
v_interp_p1_f32_e32 v5, v0, attr0.w
v_interp_p2_f32_e32 v5, v1, attr0.w
exp mrt0 v2, v3, v4, v5 done vm
s_endpgm
