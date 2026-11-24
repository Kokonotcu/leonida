; M4 vertex shader: SV_ViewportArrayIndex (cascaded shadow maps pick a cascade's viewport this way). Run:
; d3d12_viewports_test (found by the SHA-256 of its FSSM program, as vs_m3 is). build.py gives it an OSG1 of
; SV_Position (register 0) and SV_ViewportArrayIndex (register 1, .x), so the misc vector goes to pos1.
; v0 = VertexIndex 0..3 (a triangle strip), v3 = InstanceID: a quad over the whole viewport, clip (-1..1, -1..1),
;   x = 2u - 1, y = 2v - 1 with u = v0 & 1, v = (v0 >> 1) & 1;   z 0, w 1
; pos1 = the misc vector: z[19:16] = viewport index = InstanceID (GFX9+, Mesa ac_nir_export_position).
; The pixel shader is ps_m5_flat (s13..s15 = the colour; the root signature has eight constants from s8).
; fssm-at: 0x10
; fssm-data: 0x50565356 0x58444e49
v_and_b32_e32 v1, 1, v0
v_bfe_u32 v2, v0, 1, 1
v_cvt_f32_u32_e32 v1, v1
v_cvt_f32_u32_e32 v2, v2
v_add_f32_e32 v1, v1, v1
v_add_f32_e32 v2, v2, v2
v_add_f32_e32 v1, -1.0, v1
v_add_f32_e32 v2, -1.0, v2
v_mov_b32_e32 v5, 0
v_mov_b32_e32 v6, 1.0
v_lshlrev_b32_e32 v7, 16, v3
exp pos0 v1, v2, v5, v6
exp pos1 v5, v5, v7, v5 done
s_endpgm
