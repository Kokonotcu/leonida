; M5 pixel shader for the bindless-table pressure draws: samples the texture whose T# sits at a byte offset in the
; SRV table, the offset coming from a root constant. Run: d3d12_m5_world_test.
; User data from the root signature ([0] a table of SRVs, [1] a table of one sampler, [2] one 32-bit constant), one SGPR
; per table from s8 (docs/d3d12_vulkan.md section 3.5); the shader zeroes each pointer's high word:
;   s8 = 32-bit console VA of the SRV table's first slot;   s9 = 32-bit console VA of the sampler table's slot
;   s10 = byte offset of the slot (index * descriptor increment size)
; attr0.xy = (u, v), 0..1 over the quad.   mrt0 = the texel under the pixel
; fssm-at: 0x10
; fssm-data: 0x35535450 0x20535020
s_mov_b32 s20, s8
s_mov_b32 s21, 0
s_mov_b32 s22, s9
s_mov_b32 s23, 0
s_load_dwordx8 s[24:31], s[20:21], s10
s_load_dwordx4 s[32:35], s[22:23], 0x0
s_waitcnt lgkmcnt(0)
v_interp_p1_f32_e32 v2, v0, attr0.x
v_interp_p2_f32_e32 v2, v1, attr0.x
v_interp_p1_f32_e32 v3, v0, attr0.y
v_interp_p2_f32_e32 v3, v1, attr0.y
image_sample v[4:7], v[2:3], s[24:31], s[32:35] dmask:0xf dim:SQ_RSRC_IMG_2D
s_waitcnt vmcnt(0)
exp mrt0 v4, v5, v6, v7 done vm
s_endpgm
