; group: 64,1,1
; rsrc2: 0x80be
; fssm-at: 0x150
; fssm-data: 0x4c495053 0x31534320
; M10: compute root arguments past the user SGPRs, read from the console's spill array. Run: d3d12_root_spill_test.
; Written for the D3D12 compute layout directly (no LEONIDA_USER_DATA_SHIFT): RSRC2 0x80be = 31 user SGPRs and
; TGID_X, so the group index is s31. Root signature: [0] 27 constants = s4..s30, [1] a root UAV, [2] 3 constants.
; [1] needs two SGPRs and only s31 is left, so it and [2] go to the spill array behind the 40-bit pointer s[0:1]
; (docs/d3d12_vulkan.md section 7: packed in root order, no alignment, a constant 4 bytes, a root descriptor 8):
;   +0x00 the root UAV's 64-bit VA   +0x08 c0   +0x0c c1   +0x10 c2
; g = 64 * group + lane:   uav[g] = (((g + s4) ^ c0) + c1 - c2) + s30
s_load_dwordx2 s[40:41], s[0:1], 0x0
s_load_dwordx2 s[44:45], s[0:1], 0x8
s_load_dword s46, s[0:1], 0x10
s_lshl_b32 s32, s31, 6
s_waitcnt lgkmcnt(0)
v_add_nc_u32_e32 v1, s32, v0
v_add_nc_u32_e32 v2, s4, v1
v_xor_b32_e32 v2, s44, v2
v_add_nc_u32_e32 v2, s45, v2
v_subrev_nc_u32_e32 v2, s46, v2
v_add_nc_u32_e32 v2, s30, v2
v_lshlrev_b32_e32 v3, 2, v1
global_store_dword v3, v2, s[40:41]
s_endpgm
