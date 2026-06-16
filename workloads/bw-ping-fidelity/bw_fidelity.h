#pragma once

// bw-ping-fidelity (E2): infer a hidden target's achieved bandwidth from a sweep of
// co-running bandwidth pings, then evaluate the inference against target-reported bandwidth.

#define BLOCKDIM_X 1024
#define WAVE_SIZE 64

#define SRC_XCD 0
#define DST_HBM 0

// Concurrent target and ping CU budgets on SRC_XCD.
#define N_TARGET_CUS 16
inline constexpr int BW_ACTIVE_CUS[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };

// Time bucketing.
#define N_WINDOWS 64
#define WINDOW_US 50

// Streaming data: four arrays per role, home-identified and restricted to DST_HBM.
#define N_DATAS 8
#define N_PROBE_DATAS 4
#define K2_N_PAGES 1024

// Target load levels.
#define N_LEVELS 8
#define THR_MAX_CYCLES 2000

#define LOAD_STAIRCASE 0
#define LOAD_RANDOM 1
#ifndef TARGET_LOAD_PATTERN
#define TARGET_LOAD_PATTERN LOAD_STAIRCASE
#endif
#define RANDOM_SEED 0x5eed1234u

// High-confidence inference requires observable probe suppression and enough solo probe reach.
// This rejects weak pings that can slow down without filling the path residual headroom.
#define MIN_PROBE_LOSS_FRAC 0.20
#define MIN_PROBE_CAPACITY_FRAC 0.55
