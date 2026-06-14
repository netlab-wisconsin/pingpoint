#pragma once

// MI300X topology and launch shape.
#define XCD_NUM 8
#define CU_NUM 36
#define BLOCKDIM_X 1024
#define WAVE_SIZE 64
#define TARGET_CUS_PER_XCD 16

// The bandwidth ping is issued from one XCD toward one home-identified HBM stack.
#define SRC_XCD 0
#define DST_HBM 0
#define PAGE_SIZE (2 * 1024 * 1024)
#define CHUNK_SIZE (2 * 1024)
#define N_PROBE_DATAS 4
#define PROBE_N_PAGES 1024
#define PROBE_CHUNKS_PER_DATA 65536
#define STOP_CHECK_ITERS 8

// Synthetic FFN1 dimensions. Decode streams each expert's weights once. Prefill reuses the
// same weights across many tokens and is therefore substantially more compute-heavy.
#define D_MODEL 4096
#define N_EXPERT 8
#define HIDDEN_MULT 4
#define PREFILL_T 1024
#define DECODE_T N_EXPERT
#define PREFILL_ROWS_PER_THREAD 4

#define DEFAULT_MEASURED_REQUESTS 20
#define WARMUP_PASSES 1

// Balanced condition order avoids correlating ping strength or sampling rate with run time.
inline constexpr int BW_ACTIVE_CUS[] = {1, 16, 4, 15, 8, 14, 10, 13, 11, 12};
inline constexpr int RATE_TENTHS[] = {1, 10, 2, 9, 3, 8, 4, 7, 5, 6};
