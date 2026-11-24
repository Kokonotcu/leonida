; group: 64,1,1
; rsrc2: 0x8088
; fssm-at: 0x150
; fssm-data: 0x44335655 0x44414f4c
; 3D UAV load through its T# into a raw buffer. Run: d3d12_uav3d_test (a compute PSO found by the SHA-256 of its
; synthetic FSSM header). D3D12 build: LEONIDA_USER_DATA_SHIFT=4, as cs_uav3d_store.asm.
; User data: [0] two 32-bit constants (unused), [1] a descriptor table at s2 (s3 zero); group index s4.
;   slot 0 (+0x00): T# of an 8x8x8 R32_UINT 3D UAV, the whole volume or a W range of it (read)
;   slot 1 (+0x20): V#, raw buffer (written)
; g = 64 * group + lane: x = g & 7, y = (g >> 3) & 7, z = g >> 6 (view-relative)
;   out[g] = uav(x, y, z): image_load reads the sampled 3D array, which views the whole volume, so the translator adds
;   a W range's first slice (docs/design.md, "W-range 3D UAVs")
s_load_dwordx8 s[8:15], s[2:3], 0x0
s_load_dwordx4 s[16:19], s[2:3], 0x20
s_lshl_b32 s5, s4, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s5, v0
v_and_b32_e32 v2, 7, v1
v_bfe_u32 v3, v1, 3, 3
v_lshrrev_b32_e32 v4, 6, v1
image_load v5, v[2:4], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v6, 2, v1
buffer_store_dword v5, v6, s[16:19], 0 offen
s_endpgm
