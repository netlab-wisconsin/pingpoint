#pragma once

// =============================================================================================
// bw-ping-fidelity (E2): validate the bandwidth probe's headroom/saturation estimate against a
// calibrated, time-varying memory-bandwidth source.
//
//   Both the calibrated TARGET (load source) and the bandwidth PROBE (ping) stream from the same
//   path (SRC_XCD -> DST_HBM), from *different* data arrays, co-running in one launch. Time is
//   bucketed into N_WINDOWS fixed-duration windows (WINDOW_US each); in each window both report
//   their achieved bandwidth. The TARGET's intensity follows a per-window throttle schedule; the
//   PROBE runs unthrottled at `bpx` CUs. We replay the schedule while sweeping bpx=1..16.
//   Path peak is read inline from the window(s) where the target is idle (probe owns the path).
// =============================================================================================

#define BLOCKDIM_X (1024)

#define SRC_XCD  0
#define DST_HBM  0

// CU budget on SRC_XCD: target uses N_TARGET_CUS; probe uses bpx more (disjoint). Both <= CU_NUM.
#define N_TARGET_CUS 16

// Time bucketing
#define N_WINDOWS 64
#define WINDOW_US 50          // wall-clock duration of each window (microseconds)

// Hot-loop batching: stream this many iterations between clock reads / window advances, so the
// cycle-counter read and window bookkeeping are amortized out of the per-iteration path (the loads
// themselves carry no timing). Must be small enough that STREAM_BATCH iterations take << WINDOW_US
// of wall-clock (otherwise a batch could straddle/skip a window); 8 is safe at this load.
#define STREAM_BATCH 8

// Streaming data: K2-style home-identified arrays. PROBE reads the first N_PROBE_DATAS arrays,
// TARGET reads the rest; per-role per-HBM working set must exceed the LLC (256MB) to hit HBM.
#define N_DATAS       8       // 4 probe + 4 target
#define N_PROBE_DATAS 4
#define K2_N_PAGES    1024    // pages per array (x PAGE_SIZE=2MB) => 2GB/array; ~256MB/array on DST_HBM

// Target steady per-iteration throttle: clock CYCLES of delay between streamed iterations.
// Staircase of N_LEVELS plateaus, delay THR_MAX_CYCLES (lowest target BW) -> 0 (full intensity).
// Must be < window_cycles (= WINDOW_US * clock_MHz) so a single iteration never spans a window.
#define N_LEVELS       8
#define THR_MAX_CYCLES 2000

// Bandwidth probe #active CUs (bpx) swept.
inline constexpr int BW_ACTIVE_CUS[] = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 };
