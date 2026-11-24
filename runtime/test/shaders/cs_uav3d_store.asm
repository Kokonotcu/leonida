; group: 64,1,1
; rsrc2: 0x8088
; fssm-at: 0x150
; fssm-data: 0x44335655 0x52545357
; 3D UAV store (M5: volumetric fog and clouds write RWTexture3D). Run: d3d12_uav3d_test (a compute PSO found by the
; SHA-256 of its synthetic FSSM header). D3D12 build: runtime/CMakeLists.txt builds this with LEONIDA_USER_DATA_SHIFT=4
; (the console compute layout starts the root arguments at s4, D3D-448), so every SGPR named here is 4 higher there.
; User data from the root signature: [0] two 32-bit constants, [1] a descriptor table. RSRC2 0x8088 = four user
; SGPRs and TGID_X, so the group index is s4:
;   s0 = tag   s1 = unused   s2 = the table's 32-bit VA (s3, the next user dword, is zero)
;   slot 0 (+0x00): T# of an 8x8x8 R32_UINT 3D UAV, the whole volume or a W range of it (written)
; g = 64 * group + lane: x = g & 7, y = (g >> 3) & 7, z = g >> 6 (view-relative, as D3D12 addresses a W range)
;   uav(x, y, z) = tag + (z << 8) + (y << 4) + x
s_load_dwordx8 s[8:15], s[2:3], 0x0
s_lshl_b32 s5, s4, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s5, v0
v_and_b32_e32 v2, 7, v1
v_bfe_u32 v3, v1, 3, 3
v_lshrrev_b32_e32 v4, 6, v1
v_lshlrev_b32_e32 v5, 8, v4
v_lshlrev_b32_e32 v6, 4, v3
v_add_nc_u32_e32 v5, v5, v6
v_add_nc_u32_e32 v5, v5, v2
v_add_nc_u32_e32 v5, s0, v5
image_store v5, v[2:4], s[8:15] dmask:0x1 dim:SQ_RSRC_IMG_3D unorm
s_endpgm
