	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx1030"
	.section	.text._Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii,#alloc,#execinstr
	.protected	_Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii ; -- Begin function _Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii
	.globl	_Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii
	.p2align	8
	.type	_Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii,@function
_Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii: ; @_Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii
; %bb.0:
	s_clause 0x1
	s_load_dwordx2 s[2:3], s[4:5], 0x5c
	s_load_dwordx2 s[0:1], s[4:5], 0x68
	s_ashr_i32 s9, s6, 31
	s_mov_b32 s17, exec_lo
	s_add_i32 s12, s6, s9
	s_waitcnt lgkmcnt(0)
	s_ashr_i32 s11, s3, 31
	s_add_i32 s7, s3, s11
	s_xor_b32 s7, s7, s11
	v_cvt_f32_u32_e32 v1, s7
	s_sub_i32 s10, 0, s7
	v_rcp_iflag_f32_e32 v1, v1
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_readfirstlane_b32 s8, v1
	s_mul_i32 s10, s10, s8
	s_mul_hi_u32 s13, s8, s10
	s_xor_b32 s10, s12, s9
	s_add_i32 s8, s8, s13
	s_mul_hi_u32 s8, s10, s8
	s_mul_i32 s12, s8, s7
	s_add_i32 s13, s8, 1
	s_sub_i32 s12, s10, s12
	s_sub_i32 s14, s12, s7
	s_cmp_ge_u32 s12, s7
	s_cselect_b32 s8, s13, s8
	s_cselect_b32 s12, s14, s12
	s_add_i32 s13, s8, 1
	s_cmp_ge_u32 s12, s7
	s_mul_i32 s7, s3, s2
	s_cselect_b32 s16, s13, s8
	s_ashr_i32 s8, s2, 31
	s_ashr_i32 s12, s7, 31
	s_add_i32 s2, s2, s8
	s_add_i32 s7, s7, s12
	s_xor_b32 s13, s2, s8
	s_xor_b32 s2, s7, s12
	v_cvt_f32_u32_e32 v1, s13
	v_cvt_f32_u32_e32 v2, s2
	s_mul_i32 s7, s1, s0
	s_mov_b32 s8, 0
	v_rcp_iflag_f32_e32 v1, v1
	v_rcp_iflag_f32_e32 v2, v2
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_mul_f32_e32 v2, 0x4f7ffffe, v2
	v_cvt_u32_f32_e32 v1, v1
	v_cvt_u32_f32_e32 v2, v2
	v_readfirstlane_b32 s15, v1
	v_readfirstlane_b32 s14, v2
	v_cmpx_gt_i32_e64 s7, v0
	s_cbranch_execz .LBB0_3
; %bb.1:                                ; %.lr.ph132
	s_xor_b32 s11, s9, s11
	s_sub_i32 s17, 0, s13
	s_xor_b32 s16, s16, s11
	s_mul_i32 s17, s17, s15
	s_sub_i32 s11, s16, s11
	s_mul_hi_u32 s17, s15, s17
	s_ashr_i32 s16, s11, 31
	s_add_i32 s15, s15, s17
	s_add_i32 s19, s11, s16
	s_sub_i32 s18, 0, s2
	s_xor_b32 s19, s19, s16
	s_mul_i32 s18, s18, s14
	s_mul_hi_u32 s15, s19, s15
	s_mul_hi_u32 s17, s14, s18
	s_mul_i32 s15, s15, s13
	s_mul_i32 s11, s11, s3
	s_sub_i32 s15, s19, s15
	s_add_i32 s14, s14, s17
	s_sub_i32 s17, s6, s11
	s_sub_i32 s6, s15, s13
	s_cmp_ge_u32 s15, s13
	s_mul_hi_u32 s11, s10, s14
	s_cselect_b32 s6, s6, s15
	s_mul_hi_i32 s0, s1, s0
	s_sub_i32 s14, s6, s13
	s_cmp_ge_u32 s6, s13
	s_mul_i32 s13, s11, s2
	s_cselect_b32 s6, s14, s6
	s_load_dword s14, s[4:5], 0x90
	s_xor_b32 s6, s6, s16
	s_sub_i32 s10, s10, s13
	s_xor_b32 s9, s9, s12
	s_sub_i32 s6, s6, s16
	s_add_i32 s12, s11, 1
	s_sub_i32 s13, s10, s2
	s_cmp_ge_u32 s10, s2
	s_cselect_b32 s11, s12, s11
	s_cselect_b32 s10, s13, s10
	s_add_i32 s12, s11, 1
	s_cmp_ge_u32 s10, s2
	s_cselect_b32 s2, s12, s11
	s_xor_b32 s2, s2, s9
	s_sub_i32 s2, s2, s9
	s_clause 0x1
	s_load_dword s9, s[4:5], 0xa4
	s_load_dwordx2 s[10:11], s[4:5], 0x10
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s14, s14, s3
	s_mul_hi_i32 s4, s2, s3
	s_mul_i32 s2, s2, s3
	s_ashr_i32 s3, s17, 31
	s_add_u32 s2, s2, s17
	s_mul_hi_i32 s5, s6, s14
	s_mul_i32 s6, s6, s14
	s_addc_u32 s4, s4, s3
	s_ashr_i32 s3, s1, 31
	s_add_u32 s2, s2, s6
	s_addc_u32 s4, s4, s5
	s_mul_hi_u32 s5, s7, s2
	s_mul_i32 s4, s7, s4
	s_mul_i32 s0, s0, s2
	s_add_i32 s4, s5, s4
	s_add_i32 s5, s4, s0
	s_mul_i32 s4, s7, s2
	s_mov_b32 s2, s1
	s_lshl_b64 s[4:5], s[4:5], 2
	s_add_u32 s4, s10, s4
	s_addc_u32 s5, s11, s5
	s_add_i32 s0, s1, s3
	s_and_b32 s9, s9, 0xffff
	s_xor_b32 s6, s0, s3
	s_sub_i32 s1, 0, s1
	v_cvt_f32_u32_e32 v1, s6
	s_sub_i32 s0, 0, s6
	v_rcp_iflag_f32_e32 v1, v1
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v2, v1
	v_mul_lo_u32 v1, s0, v2
	v_mul_hi_u32 v3, v2, v1
	v_cvt_f32_i32_e32 v1, s17
	v_add_nc_u32_e32 v2, v2, v3
	s_inst_prefetch 0x1
	.p2align	6
.LBB0_2:                                ; %._crit_edge129
                                        ; =>This Inner Loop Header: Depth=1
	v_ashrrev_i32_e32 v3, 31, v0
	v_add_nc_u32_e32 v4, v0, v3
	v_xor_b32_e32 v4, v4, v3
	v_xor_b32_e32 v3, s3, v3
	v_mul_hi_u32 v5, v4, v2
	v_mul_lo_u32 v6, v5, s6
	v_add_nc_u32_e32 v7, 1, v5
	v_sub_nc_u32_e32 v4, v4, v6
	v_subrev_nc_u32_e32 v6, s6, v4
	v_cmp_le_u32_e32 vcc_lo, s6, v4
	v_cndmask_b32_e32 v5, v5, v7, vcc_lo
	v_cndmask_b32_e32 v4, v4, v6, vcc_lo
	v_add_nc_u32_e32 v6, 1, v5
	v_cmp_le_u32_e32 vcc_lo, s6, v4
	v_cndmask_b32_e32 v4, v5, v6, vcc_lo
	v_xor_b32_e32 v4, v4, v3
	v_sub_nc_u32_e32 v5, v4, v3
	v_mad_u64_u32 v[3:4], null, s1, v5, v[0:1]
	v_add_nc_u32_e32 v0, s9, v0
	v_cmp_le_i32_e32 vcc_lo, s7, v0
	v_ashrrev_i32_e32 v4, 31, v3
	s_or_b32 s8, vcc_lo, s8
	v_mad_i64_i32 v[3:4], null, v5, s2, v[3:4]
	v_lshlrev_b64 v[3:4], 2, v[3:4]
	v_add_co_u32 v3, s0, s4, v3
	v_add_co_ci_u32_e64 v4, s0, s5, v4, s0
	global_store_dword v[3:4], v1, off
	s_andn2_b32 exec_lo, exec_lo, s8
	s_cbranch_execnz .LBB0_2
.LBB0_3:                                ; %Flow152
	s_inst_prefetch 0x2
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 408
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 8
		.amdhsa_next_free_sgpr 20
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii,#alloc,#execinstr
.Lfunc_end0:
	.size	_Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii, .Lfunc_end0-_Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 760
; NumSgprs: 22
; NumVgprs: 8
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 8
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z7add_oneIfiEvPT_T0_S2_S2_,#alloc,#execinstr
	.protected	_Z7add_oneIfiEvPT_T0_S2_S2_ ; -- Begin function _Z7add_oneIfiEvPT_T0_S2_S2_
	.globl	_Z7add_oneIfiEvPT_T0_S2_S2_
	.p2align	8
	.type	_Z7add_oneIfiEvPT_T0_S2_S2_,@function
_Z7add_oneIfiEvPT_T0_S2_S2_:            ; @_Z7add_oneIfiEvPT_T0_S2_S2_
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x8
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s3, s1, s0
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB1_5
; %bb.1:                                ; %.lr.ph
	s_clause 0x1
	s_load_dwordx2 s[8:9], s[4:5], 0x0
	s_load_dword s7, s[4:5], 0x24
	s_mul_i32 s0, s3, s6
	v_cvt_f32_u32_e32 v2, s6
	s_ashr_i32 s1, s0, 31
	s_mov_b32 s6, 0
	s_lshl_b64 s[4:5], s[0:1], 2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s1, s8, s4
	s_addc_u32 s4, s9, s5
	s_and_b32 s5, s7, 0xffff
	s_branch .LBB1_3
	.p2align	6
.LBB1_2:                                ;   in Loop: Header=BB1_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s7
	v_add_nc_u32_e32 v0, s5, v0
	v_cmp_le_i32_e32 vcc_lo, s3, v0
	s_or_b32 s6, vcc_lo, s6
	s_andn2_b32 exec_lo, exec_lo, s6
	s_cbranch_execz .LBB1_5
.LBB1_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, s0, v0
	s_mov_b32 s7, exec_lo
	v_cmpx_gt_i32_e64 s2, v1
	s_cbranch_execz .LBB1_2
; %bb.4:                                ;   in Loop: Header=BB1_3 Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[3:4], 2, v[0:1]
	v_add_co_u32 v3, vcc_lo, s1, v3
	v_add_co_ci_u32_e32 v4, vcc_lo, s4, v4, vcc_lo
	global_store_dword v[3:4], v2, off
	s_branch .LBB1_2
.LBB1_5:                                ; %Flow21
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z7add_oneIfiEvPT_T0_S2_S2_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 280
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 5
		.amdhsa_next_free_sgpr 10
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z7add_oneIfiEvPT_T0_S2_S2_,#alloc,#execinstr
.Lfunc_end1:
	.size	_Z7add_oneIfiEvPT_T0_S2_S2_, .Lfunc_end1-_Z7add_oneIfiEvPT_T0_S2_S2_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 184
; NumSgprs: 12
; NumVgprs: 5
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 12
; NumVGPRsForWavesPerEU: 5
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_,#alloc,#execinstr
	.protected	_Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_ ; -- Begin function _Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_
	.globl	_Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_
	.p2align	8
	.type	_Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_,@function
_Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_: ; @_Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s3, s1, s0
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB2_5
; %bb.1:                                ; %.lr.ph
	s_clause 0x2
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	s_load_dwordx2 s[12:13], s[4:5], 0x10
	s_load_dword s16, s[4:5], 0x34
	s_mul_i32 s0, s3, s6
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[14:15], s[0:1], 2
	s_waitcnt lgkmcnt(0)
	s_add_u32 s1, s8, s14
	s_addc_u32 s4, s9, s15
	s_add_u32 s5, s10, s14
	s_addc_u32 s6, s11, s15
	s_add_u32 s7, s12, s14
	s_addc_u32 s8, s13, s15
	s_and_b32 s9, s16, 0xffff
	s_mov_b32 s10, 0
	s_inst_prefetch 0x1
	s_branch .LBB2_3
	.p2align	6
.LBB2_2:                                ;   in Loop: Header=BB2_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v0, s9, v0
	v_cmp_le_i32_e32 vcc_lo, s3, v0
	s_or_b32 s10, vcc_lo, s10
	s_andn2_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB2_5
.LBB2_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, s0, v0
	s_mov_b32 s11, exec_lo
	v_cmpx_gt_i32_e64 s2, v1
	s_cbranch_execz .LBB2_2
; %bb.4:                                ;   in Loop: Header=BB2_3 Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	v_add_co_u32 v3, vcc_lo, s1, v1
	v_add_co_ci_u32_e32 v4, vcc_lo, s4, v2, vcc_lo
	v_add_co_u32 v5, vcc_lo, s5, v1
	v_add_co_ci_u32_e32 v6, vcc_lo, s6, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, s7, v1
	global_load_dword v3, v[3:4], off
	global_load_dword v4, v[5:6], off
	v_add_co_ci_u32_e32 v2, vcc_lo, s8, v2, vcc_lo
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v3, v3, v4
	global_store_dword v[1:2], v3, off
	s_branch .LBB2_2
.LBB2_5:                                ; %Flow32
	s_inst_prefetch 0x2
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 17
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_,#alloc,#execinstr
.Lfunc_end2:
	.size	_Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_, .Lfunc_end2-_Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 268
; NumSgprs: 19
; NumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 19
; NumVGPRsForWavesPerEU: 7
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.text._Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_,#alloc,#execinstr
	.protected	_Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_ ; -- Begin function _Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_
	.globl	_Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_
	.p2align	8
	.type	_Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_,@function
_Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_: ; @_Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_
; %bb.0:
	s_load_dwordx4 s[0:3], s[4:5], 0x18
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s3, s1, s0
	s_mov_b32 s0, exec_lo
	v_cmpx_gt_i32_e64 s3, v0
	s_cbranch_execz .LBB3_5
; %bb.1:                                ; %.lr.ph
	s_clause 0x3
	s_load_dword s0, s[4:5], 0x28
	s_load_dwordx4 s[8:11], s[4:5], 0x0
	s_load_dwordx2 s[12:13], s[4:5], 0x10
	s_load_dword s16, s[4:5], 0x34
	s_not_b32 s1, s6
	s_waitcnt lgkmcnt(0)
	s_add_i32 s0, s0, s1
	s_mul_i32 s0, s0, s3
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[14:15], s[0:1], 2
	s_add_u32 s1, s8, s14
	s_addc_u32 s4, s9, s15
	s_add_u32 s5, s10, s14
	s_addc_u32 s6, s11, s15
	s_add_u32 s7, s12, s14
	s_addc_u32 s8, s13, s15
	s_and_b32 s9, s16, 0xffff
	s_mov_b32 s10, 0
	s_inst_prefetch 0x1
	s_branch .LBB3_3
	.p2align	6
.LBB3_2:                                ;   in Loop: Header=BB3_3 Depth=1
	s_or_b32 exec_lo, exec_lo, s11
	v_add_nc_u32_e32 v0, s9, v0
	v_cmp_le_i32_e32 vcc_lo, s3, v0
	s_or_b32 s10, vcc_lo, s10
	s_andn2_b32 exec_lo, exec_lo, s10
	s_cbranch_execz .LBB3_5
.LBB3_3:                                ; =>This Inner Loop Header: Depth=1
	v_add_nc_u32_e32 v1, s0, v0
	s_mov_b32 s11, exec_lo
	v_cmpx_gt_i32_e64 s2, v1
	s_cbranch_execz .LBB3_2
; %bb.4:                                ;   in Loop: Header=BB3_3 Depth=1
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshlrev_b64 v[1:2], 2, v[0:1]
	v_add_co_u32 v3, vcc_lo, s1, v1
	v_add_co_ci_u32_e32 v4, vcc_lo, s4, v2, vcc_lo
	v_add_co_u32 v5, vcc_lo, s5, v1
	v_add_co_ci_u32_e32 v6, vcc_lo, s6, v2, vcc_lo
	v_add_co_u32 v1, vcc_lo, s7, v1
	global_load_dword v3, v[3:4], off
	global_load_dword v4, v[5:6], off
	v_add_co_ci_u32_e32 v2, vcc_lo, s8, v2, vcc_lo
	s_waitcnt vmcnt(0)
	v_add_f32_e32 v3, v3, v4
	global_store_dword v[1:2], v3, off
	s_branch .LBB3_2
.LBB3_5:                                ; %Flow32
	s_inst_prefetch 0x2
	s_endpgm
	.section	.rodata,#alloc
	.p2align	6, 0x0
	.amdhsa_kernel _Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 296
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_wavefront_size32 1
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 7
		.amdhsa_next_free_sgpr 17
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_workgroup_processor_mode 1
		.amdhsa_memory_ordered 1
		.amdhsa_forward_progress 0
		.amdhsa_shared_vgpr_count 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_,#alloc,#execinstr
.Lfunc_end3:
	.size	_Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_, .Lfunc_end3-_Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_
                                        ; -- End function
	.section	.AMDGPU.csdata
; Kernel info:
; codeLenInByte = 284
; NumSgprs: 19
; NumVgprs: 7
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 19
; NumVGPRsForWavesPerEU: 7
; Occupancy: 16
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.p2alignl 6, 3214868480
	.fill 48, 4, 3214868480
	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE ; @_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
	.type	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE
_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI15__HIP_ThreadIdxE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE ; @_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
	.type	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE
_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI14__HIP_BlockDimE1xE, 1

	.protected	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE ; @_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE
	.type	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE,@object
	.section	.rodata._ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE,#alloc
	.weak	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE
_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE:
	.zero	1
	.size	_ZN17__HIP_CoordinatesI13__HIP_GridDimE1xE, 1

	.ident	"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.2 24012 af27734ed982b52a9f1be0f035ac91726fc697e4)"
	.section	".note.GNU-stack"
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           20
        .value_kind:     by_value
      - .offset:         44
        .size:           20
        .value_kind:     by_value
      - .offset:         64
        .size:           20
        .value_kind:     by_value
      - .offset:         84
        .size:           4
        .value_kind:     by_value
      - .offset:         88
        .size:           4
        .value_kind:     by_value
      - .offset:         92
        .size:           4
        .value_kind:     by_value
      - .offset:         96
        .size:           4
        .value_kind:     by_value
      - .offset:         100
        .size:           4
        .value_kind:     by_value
      - .offset:         104
        .size:           4
        .value_kind:     by_value
      - .offset:         108
        .size:           4
        .value_kind:     by_value
      - .offset:         112
        .size:           4
        .value_kind:     by_value
      - .offset:         116
        .size:           4
        .value_kind:     by_value
      - .offset:         120
        .size:           4
        .value_kind:     by_value
      - .offset:         124
        .size:           4
        .value_kind:     by_value
      - .offset:         128
        .size:           4
        .value_kind:     by_value
      - .offset:         132
        .size:           4
        .value_kind:     by_value
      - .offset:         136
        .size:           4
        .value_kind:     by_value
      - .offset:         140
        .size:           4
        .value_kind:     by_value
      - .offset:         144
        .size:           4
        .value_kind:     by_value
      - .offset:         152
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         156
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         160
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         164
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         166
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         168
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         170
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         172
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         174
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         192
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         200
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         208
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         216
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 408
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z19naive_conv_fwd_nchwILb1EfffEvPKT0_S2_PT2_St5arrayIiLm5EES6_S6_iiiiiiiiiiiiiiii.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     8
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  write_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .offset:         8
        .size:           4
        .value_kind:     by_value
      - .offset:         12
        .size:           4
        .value_kind:     by_value
      - .offset:         16
        .size:           4
        .value_kind:     by_value
      - .offset:         24
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         28
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         36
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         38
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         40
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         42
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         44
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         46
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         64
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         88
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 280
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z7add_oneIfiEvPT_T0_S2_S2_
    .private_segment_fixed_size: 0
    .sgpr_count:     12
    .sgpr_spill_count: 0
    .symbol:         _Z7add_oneIfiEvPT_T0_S2_S2_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     5
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_
    .private_segment_fixed_size: 0
    .sgpr_count:     19
    .sgpr_spill_count: 0
    .symbol:         _Z10vector_addILb0EfiEvPKT0_S2_PS0_T1_S4_S4_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
  - .args:
      - .actual_access:  read_only
        .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  read_only
        .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .actual_access:  write_only
        .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .offset:         24
        .size:           4
        .value_kind:     by_value
      - .offset:         28
        .size:           4
        .value_kind:     by_value
      - .offset:         32
        .size:           4
        .value_kind:     by_value
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         44
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         48
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         52
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         54
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         56
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         58
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         60
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         62
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         96
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         104
        .size:           2
        .value_kind:     hidden_grid_dims
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 296
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 1024
    .name:           _Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_
    .private_segment_fixed_size: 0
    .sgpr_count:     19
    .sgpr_spill_count: 0
    .symbol:         _Z10vector_addILb1EfiEvPKT0_S2_PS0_T1_S4_S4_.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 32
    .workgroup_processor_mode: 1
amdhsa.target:   amdgcn-amd-amdhsa--gfx1030
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
