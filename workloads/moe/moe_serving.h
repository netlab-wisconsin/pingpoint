#pragma once

// =============================================================================================
// moe-serving (E1) configuration constants.
// =============================================================================================

#define FAST 1
#define DEBUG_LEVEL 1
#define TARGET_BLOCKDIM_X (1024)

// Per-XCD CU budget for the TARGET kernel. The target always runs on the top
// TARGET_CUS_PER_XCD CUs of each XCD (constant, independent of the ping), and a ping reserves its
// bpx (or 1, latency) CUs in the low slots on top of that. Launch grid = (TARGET_CUS_PER_XCD +
// bpx) * XCD_NUM, so the target's CU count never shrinks with bpx. Requires
// TARGET_CUS_PER_XCD + max(bpx) <= CU_NUM (16 + 16 <= 36).
#define TARGET_CUS_PER_XCD 16

// Model / regime knobs
#define D_MODEL     4096
#define N_EXPERT    8
#define PREFILL_T   8192   // many tokens/expert  -> compute-bound
#define DECODE_T    8    // few  tokens/expert  -> memory-bound

// Serving-loop knobs (overridable via argv)
#define N_PASSES_DEFAULT 51
#define WARMUP_PASSES    1

// Baseline / Ping plan enables
#define DISABLE_BASELINE 0
// Guarded so a build can override via -DDISABLE_K{1,2}_PLANS=1 (see scripts/).
#ifndef DISABLE_K1_PLANS
#define DISABLE_K1_PLANS 0
#endif
#ifndef DISABLE_K2_PLANS
#define DISABLE_K2_PLANS 0
#endif

// Ping lengths: must outlast the target (see moe_serving.cpp header note).
#define PING_ITERS_K1 (1 << 23)
#define PING_ITERS_K2 (1 << 20)

// Ping plan selection: 1 = build only the customized plans in the #else blocks of
// moe_serving.cpp (default); 0 = build the full cross-product of src_xcd x dst_hbm
// (latency) and src_xcd x dst_hbm x #active-CU (bandwidth).
#define PPNT_PLAN_SELECTED_ONLY 1

// Bandwidth ping #active CUs (bpx) swept in the full (non-selected) construction.
inline constexpr int BW_ACTIVE_CUS[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

// Swept ping rates (Option-A duty cycle) for the latency/bandwidth conditions.
inline constexpr double PING_RATES[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
