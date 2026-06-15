#pragma once

// E3: adaptive two-tier monitoring for a path-localized decode-FFN anomaly.

#define BLOCKDIM_X 1024
#define WAVE_SIZE 64

#define TARGET_CUS_PER_XCD 16
#define LATENCY_CUS_PER_XCD 1
#define MAX_BW_CUS_PER_XCD 16

#define N_EXPERT 8
#define TOTAL_TOKENS 64
#define BALANCED_TOKENS_PER_EXPERT 8
#define HOT_EXPERT 0
#define HOT_EXPERT_TOKENS 57
#define COLD_EXPERT_TOKENS 1

// One target block iteration streams 64 KiB. Work scales with routed token count.
#define TARGET_BLOCK_ITERS_PER_TOKEN 128

// Two identical anomaly encounters separated by a stable recovery interval.
#define DEFAULT_REQUESTS 64
#define DEFAULT_FIRST_ANOMALY_START 16
#define DEFAULT_FIRST_ANOMALY_LENGTH 16
#define DEFAULT_SECOND_ANOMALY_START 48
#define DEFAULT_SECOND_ANOMALY_LENGTH 16
#define DEFAULT_CALIBRATION_REQUESTS 30
#define DEFAULT_WARMUP_REQUESTS 3

// Detector/controller settings.
#define DETECT_STREAK 2
#define RESET_STREAK 3
#define INFORMATIVE_BPX_COUNT 7
#define FIRST_ANOMALY_PING_COUNT (INFORMATIVE_BPX_COUNT * 2)
#define SECOND_ANOMALY_BPX 10
#define SECOND_ANOMALY_PING_COUNT 4
#define SECOND_ANOMALY_INTERVAL 4
#define MAX_BW_BPX MAX_BW_CUS_PER_XCD
#define MIN_PROBE_LOSS_FRAC 0.20
#define MIN_PROBE_PEAK_FRAC 0.55

// Home-identified data sizes. Each streaming array is 1 GiB. The pointer-chain allocation
// contains approximately 32 MiB of latency-ping nodes per HBM path.
#define N_STREAM_DATAS 8
#define N_PROBE_DATAS 4
#define STREAM_N_PAGES 512
#define LAT_CHAIN_LINES_PER_HBM (1 << 18)

#define SOLO_BW_WINDOW_US 1000
#define SOLO_BW_REPEATS 3
#define STOP_CHECK_ITERS 8

struct PathData {
    uint64_t* probe[4];
    uint64_t* target[4];
    size_t n_probe_chunks;
    size_t n_target_chunks;
    int64_t* latency_start;
};

struct LatencyStats {
    uint64_t samples;
    uint64_t sum_cycles;
    uint64_t sum_sq_cycles;
    uint64_t max_cycles;
};
