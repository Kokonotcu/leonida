; Multi-argument ExecuteIndirect vertex shader: a vertex from the vertex buffer (POSITION, register 0, two floats) moved
; by a per-record offset in root constants. Run: d3d12_indirect_multiarg_test (found by the SHA-256 of its FSSM program).
; User data, from the root signature [0], five 32-bit constants (floats) = s8..s12 (the console layout starts a vertex
; shader's root arguments at s8, docs/d3d12_vulkan.md section 3.5):
;   s8, s9 = clip-space x, y added to the position;   s10, s11, s12 = the colour, read by the pixel shader (ps_ma.asm)
; The vertex fetch is the driver's: the translator replaces the call by the Vulkan vertex input of the input signature
; (build.py writes an ISG1 with POSITION at register 0), whose x and y arrive in v4 and v5 (v(4 + 4 * register + component)).
; Every position and offset is a multiple of 1/16, so each operation is exact.
; fssm-at: 0x10
; fssm-data: 0x414d5356 0x20535620
s_swappc_b64 s[30:31], s[6:7]
v_add_f32_e32 v1, s8, v4
v_add_f32_e32 v2, s9, v5
v_mov_b32_e32 v3, 0
v_mov_b32_e32 v10, 1.0
exp pos0 v1, v2, v3, v10 done
s_endpgm
