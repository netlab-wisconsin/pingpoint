#pragma once

// =============================================================================================
// moe-serving (E1) configuration constants.
// =============================================================================================

#define FAST 1
#define DEBUG_LEVEL 1
#define TARGET_BLOCKDIM_X (1024)

// Model / regime knobs
#define D_MODEL     4096
#define N_EXPERT    8
#define PREFILL_T   8192   // many tokens/expert  -> compute-bound
#define DECODE_T    256    // few  tokens/expert  -> memory-bound

// Serving-loop knobs (overridable via argv)
#define N_PASSES_DEFAULT 60
#define WARMUP_PASSES    10
#define BPX_DEFAULT      8

// Ping lengths: must outlast the target (see moe_serving.cpp header note).
#define PING_ITERS_K1 (1 << 23)
#define PING_ITERS_K2 (1 << 20)

// Plan indices
enum { PLAN_NULL = 0, PLAN_LAT = 1, PLAN_BW = 2, N_PLAN_KINDS = 3 };

// Swept ping rates (Option-A duty cycle) for the latency/bandwidth conditions.
inline constexpr double PING_RATES[] = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
