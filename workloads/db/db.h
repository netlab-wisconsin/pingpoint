#pragma once

#include <vector>
#include <random>
#include <cassert>
#include <cstddef>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <climits>
#include <hip/hip_cooperative_groups.h>

using namespace std;
namespace db_cg = cooperative_groups;

// Floats in one 2KB chunk
#define DB_ENTRY_DIM (CHUNK_SIZE / (int)sizeof(float))  // 512

// Number of 2KB chunks in the hot table (8MB >> L2 size of 4MB per XCD).
// Each hot query accesses a *different* chunk (q % N_HOT_CHUNKS), so the
// working set across all blocks on one XCD is up to 8MB and forces HBM reads.
// Rule of thumb: need Q * 0.9 / XCD_NUM > N_HOT_CHUNKS/2 for >50% L2 miss rate
// → with N_HOT_CHUNKS=4096 and Q=32768: 3686 hot queries/XCD > 2048. ✓
#define N_HOT_CHUNKS 4096

// Single XCD used by the focused VecScan experiment.
#ifndef DB_VECSCAN_ACTIVE_XCD
#define DB_VECSCAN_ACTIVE_XCD 2
#endif

// HBM used for VecScan hot candidate lists before migration.
#ifndef DB_VECSCAN_ORIGINAL_HBM
#define DB_VECSCAN_ORIGINAL_HBM 4
#endif

// HBM used for VecScan hot candidate lists after emulated migration.
#ifndef DB_VECSCAN_MIGRATED_HBM
#define DB_VECSCAN_MIGRATED_HBM 2
#endif

// Best-effort VecScan engines used by db_be.cpp.
#ifndef DB_BE_ACTIVE_XCD
#define DB_BE_ACTIVE_XCD 3
#endif

#ifndef DB_BE_HBM0
#define DB_BE_HBM0 4
#endif

#ifndef DB_BE_HBM1
#define DB_BE_HBM1 5
#endif

#if DB_VECSCAN_ACTIVE_XCD < 0 || DB_VECSCAN_ACTIVE_XCD >= XCD_NUM
#error "DB_VECSCAN_ACTIVE_XCD must be in [0, XCD_NUM)"
#endif

#if DB_VECSCAN_ORIGINAL_HBM < 0 || DB_VECSCAN_ORIGINAL_HBM >= HBM_NUM
#error "DB_VECSCAN_ORIGINAL_HBM must be in [0, HBM_NUM)"
#endif

#if DB_VECSCAN_MIGRATED_HBM < 0 || DB_VECSCAN_MIGRATED_HBM >= HBM_NUM
#error "DB_VECSCAN_MIGRATED_HBM must be in [0, HBM_NUM)"
#endif

#if DB_BE_ACTIVE_XCD < 0 || DB_BE_ACTIVE_XCD >= XCD_NUM
#error "DB_BE_ACTIVE_XCD must be in [0, XCD_NUM)"
#endif

#if DB_BE_HBM0 < 0 || DB_BE_HBM0 >= HBM_NUM
#error "DB_BE_HBM0 must be in [0, HBM_NUM)"
#endif

#if DB_BE_HBM1 < 0 || DB_BE_HBM1 >= HBM_NUM
#error "DB_BE_HBM1 must be in [0, HBM_NUM)"
#endif

// Fraction (0-100) of queries that target the hot table
#define HOT_FRAC_PCT 90

// Synthetic IVF-like ANN shape.  The hot corpus is stored as N_HOT_CHUNKS
// vector chunks, split into centroid posting lists of DB_LIST_SIZE vectors.
#ifndef DB_LIST_SIZE
#define DB_LIST_SIZE 64
#endif

#ifndef DB_CANDIDATES_PER_QUERY
#define DB_CANDIDATES_PER_QUERY 64
#endif

#define DB_NPROBE ((DB_CANDIDATES_PER_QUERY + DB_LIST_SIZE - 1) / DB_LIST_SIZE)
#define DB_HOT_CENTROIDS ((N_HOT_CHUNKS + DB_LIST_SIZE - 1) / DB_LIST_SIZE)

// Separate DB-owned placement buffer for VecScan.
#define DB_VECSCAN_DATA_SIZE (256 * 1024 * 1024)

// Logging
#ifndef DEBUG_DB
#define DEBUG_DB 0
#endif

// Verbose home-identification logging (matches DEBUG_K1_HOME / DEBUG_K2_HOME)
#ifndef DEBUG_DB_HOME
#define DEBUG_DB_HOME 0
#endif

#ifndef P_WORKERS
#define P_WORKERS 8
#endif

#ifndef P_ACTIVE_XCD
#define P_ACTIVE_XCD 2
#endif

#ifndef P_TARGET_CC
#define P_TARGET_CC 2
#endif

#ifndef P_ARRIVAL_QPS
#define P_ARRIVAL_QPS 500000
#endif

#ifndef BE_WORKERS_PER_XCD
#define BE_WORKERS_PER_XCD 8
#endif

#ifndef BE_BATCH_CHUNKS
#define BE_BATCH_CHUNKS 64
#endif

// 1: BE batch base hashed per (xcd, batch) — XCDs scan uncorrelated windows
//    so BE traffic actually hits HBM (default).
// 0: sequential laps, phase-shifted per XCD; convoys MALL-hit across XCDs.
#ifndef BE_HASHED_BATCHES
#define BE_HASHED_BATCHES 1
#endif

// BE target-CC policy:
// 0: all BE batches target P_TARGET_CC (policy b, unmanaged co-location)
// 1: each batch targets a random CC (policy c, random migration)
// 2: each batch targets a uniform random non-P CC (policy e, guided migration)
#ifndef BE_TARGET_POLICY
#define BE_TARGET_POLICY 0
#endif

// Policy c bias: probability (0-100) that a random batch lands on P's CC;
// the remainder is uniform over the other CCs. 25 = uniform over 4 CCs,
// 100 = equivalent to policy b.
#ifndef BE_RANDOM_CC0_PCT
#define BE_RANDOM_CC0_PCT 25
#endif

// Policy d: random BE throttling. Percentage (0-100) of batches each worker
// skips, idling for BE_THROTTLE_IDLE_CYCLES instead (so skipped batches cost
// the time a real batch would have, removing work rather than moving it).
#ifndef BE_THROTTLE_PCT
#define BE_THROTTLE_PCT 0
#endif

// Idle duration of a throttled batch; default ~ the time one 64-chunk batch
// takes at HBM-bound BE rates (~9us at 2.1GHz).
#ifndef BE_THROTTLE_IDLE_CYCLES
#define BE_THROTTLE_IDLE_CYCLES 18000
#endif

#ifndef P_MEASURE_LOAD_LATENCY
#define P_MEASURE_LOAD_LATENCY 0
#endif

#ifndef BE_MEASURE_LOAD_LATENCY
#define BE_MEASURE_LOAD_LATENCY 0
#endif

#ifndef DB_USE_GLOBAL_BARRIER
#define DB_USE_GLOBAL_BARRIER 0
#endif

#define P_BLOCKDIM_X DB_ENTRY_DIM
#define BE_BLOCKDIM_X DB_ENTRY_DIM

#if P_WORKERS <= 0 || P_WORKERS >= CU_NUM
#error "P_WORKERS must be in [1, CU_NUM) so later cooperative kernels can co-run"
#endif

#if BE_WORKERS_PER_XCD <= 0 || BE_WORKERS_PER_XCD > CU_NUM
#error "BE_WORKERS_PER_XCD must be in [1, CU_NUM]"
#endif

#if BE_BATCH_CHUNKS < 4 || (BE_BATCH_CHUNKS % 4) != 0
#error "BE_BATCH_CHUNKS must be a positive multiple of 4"
#endif

// BE dwordx4 path: 512 threads = 4 chunks x 128 float4 lanes per iteration.
static_assert(BE_BLOCKDIM_X == DB_ENTRY_DIM && DB_ENTRY_DIM % 4 == 0,
              "BE dwordx4 loads require BE_BLOCKDIM_X == DB_ENTRY_DIM, a multiple of 4");

#if BE_HASHED_BATCHES != 0 && BE_HASHED_BATCHES != 1
#error "BE_HASHED_BATCHES must be 0 or 1"
#endif

#if BE_TARGET_POLICY < 0 || BE_TARGET_POLICY > 2
#error "BE_TARGET_POLICY must be 0, 1, or 2"
#endif

#if BE_TARGET_POLICY != 0 && !BE_HASHED_BATCHES
#error "BE_TARGET_POLICY=1/2 (random/guided CC) requires BE_HASHED_BATCHES=1"
#endif

#if BE_RANDOM_CC0_PCT < 0 || BE_RANDOM_CC0_PCT > 100
#error "BE_RANDOM_CC0_PCT must be in [0, 100]"
#endif

#if BE_THROTTLE_PCT < 0 || BE_THROTTLE_PCT > 100
#error "BE_THROTTLE_PCT must be in [0, 100]"
#endif

#if BE_THROTTLE_IDLE_CYCLES <= 0
#error "BE_THROTTLE_IDLE_CYCLES must be positive"
#endif

#if P_MEASURE_LOAD_LATENCY != 0 && P_MEASURE_LOAD_LATENCY != 1
#error "P_MEASURE_LOAD_LATENCY must be 0 or 1"
#endif

#if BE_MEASURE_LOAD_LATENCY != 0 && BE_MEASURE_LOAD_LATENCY != 1
#error "BE_MEASURE_LOAD_LATENCY must be 0 or 1"
#endif

#if DB_USE_GLOBAL_BARRIER != 0 && DB_USE_GLOBAL_BARRIER != 1
#error "DB_USE_GLOBAL_BARRIER must be 0 or 1"
#endif

#if P_ACTIVE_XCD < 0 || P_ACTIVE_XCD >= XCD_NUM
#error "P_ACTIVE_XCD must be in [0, XCD_NUM)"
#endif

#if P_TARGET_CC < 0 || P_TARGET_CC >= CC_NUM
#error "P_TARGET_CC must be in [0, CC_NUM)"
#endif

#if P_ARRIVAL_QPS <= 0
#error "P_ARRIVAL_QPS must be positive"
#endif

static void db_fill_random(float *v, size_t n, float scale = 1.0f)
{
    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
}

namespace db {

__device__ int d_home_barrier_count = 0;
__device__ int d_home_barrier_sense = 0;

__device__ __forceinline__
void home_global_barrier()
{
    __shared__ int local_sense;
    if (threadIdx.x == 0) {
        const int old_sense = d_home_barrier_sense;
        local_sense = !old_sense;

        const int arrived = atomicAdd(&d_home_barrier_count, 1);
        if (arrived == gridDim.x - 1) {
            d_home_barrier_count = 0;
            __threadfence();
            d_home_barrier_sense = local_sense;
        }
    }
    __syncthreads();
    while (d_home_barrier_sense != local_sense) {}
    __syncthreads();
}

__global__ void identify_home(void *data, size_t size, uint32_t *d_cycles, int n_chunks)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd;

    if (tid == 0) {
        #if DEBUG_DB_HOME
        printf("bid %d: tbid_in_xcd %d\n", bid, tbid_in_xcd);
        #endif
    }

    assert(n_tbs_in_xcd * XCD_NUM == gridDim.x);
    assert(n_tbs_in_xcd == 1);
    assert(blockDim.x == 128);

#if !DB_USE_GLOBAL_BARRIER
    db_cg::grid_group grid = db_cg::this_grid();
#endif
    uint32_t xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    #if DEBUG_DB_HOME
    {
        uint32_t cu_id, se_id;
        asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
        asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
        if (tid == 0)
            printf("bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", bid, xcc_id, se_id, cu_id);
    }
    #endif

    uint4 *data_u4 = (uint4 *)data;
    const size_t inner_size = 64 * 1024 * 1024;
    const size_t n_outer = size / inner_size;
    const size_t n_inner = inner_size / (sizeof(uint4) * blockDim.x);
    assert(size % inner_size == 0);
    assert((size_t)n_chunks == n_outer * n_inner * n_tbs_in_xcd);

    if (bid == 0 && tid == 0) {
        #if DEBUG_DB_HOME
        printf("n_outer: %zu, n_inner: %zu, n_chunks: %d\n", n_outer, n_inner, n_chunks);
        printf("working set per xcd: %.2f MB\n", size / (1024.0 * 1024.0));
        #endif
    }

    for (size_t i = 0; i < n_outer; i++) {
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (tid == 0) {
                #if DEBUG_DB_HOME >= 2
                printf("[warmup] (outer:%zu, inner:%zu, bid:%d) accessing data_u4[%zu..%zu]\n",
                       i, j, bid, index, index + blockDim.x - 1);
                #endif
            }
            asm volatile(
                "flat_load_dwordx4 v[0:3], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data_u4[index])
                : "memory", "v0", "v1", "v2", "v3"
            );
        }
#if DB_USE_GLOBAL_BARRIER
        home_global_barrier();
#else
        grid.sync();
#endif

        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            uint32_t start = __builtin_readcyclecounter();
            asm volatile(
                "flat_load_dwordx4 v[0:3], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data_u4[index])
                : "memory", "v0", "v1", "v2", "v3"
            );
            uint32_t end = __builtin_readcyclecounter();
            if (tid == 0) {
                size_t chunk = (i * n_inner + j) * n_tbs_in_xcd + tbid_in_xcd;
                d_cycles[(size_t)xcc_id * n_chunks + chunk] = end - start;
                #if DEBUG_DB_HOME >= 2
                printf("xcc %u chunk[%zu]: %u cycles\n", xcc_id, chunk, end - start);
                #endif
            }
#if DB_USE_GLOBAL_BARRIER
            home_global_barrier();
#else
            grid.sync();
#endif
        }
    }
}

static int home_identification(
    char *d_data,
    size_t data_size,
    size_t n_chunks,
    vector<uint32_t> &chunk_home_xcd,
    vector<vector<uint64_t>> &chunks_per_cc)
{
    uint32_t *d_cycles = nullptr;
    gpuErrchk(hipMalloc(&d_cycles, sizeof(uint32_t) * XCD_NUM * n_chunks));

    int n_chunks_i = (int)n_chunks;
    void *kernel_args[] = {
        (void *)&d_data,
        (void *)&data_size,
        (void *)&d_cycles,
        (void *)&n_chunks_i
    };
#if DB_USE_GLOBAL_BARRIER
    gpuErrchk(hipLaunchKernel(
        (void *)identify_home, dim3(XCD_NUM), dim3(128), kernel_args, 0, 0));
#else
    gpuErrchk(hipLaunchCooperativeKernel(
        (void *)identify_home, dim3(XCD_NUM), dim3(128), kernel_args, 0, 0));
#endif
    gpuErrchk(hipDeviceSynchronize());

    vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(n_chunks));
    for (int x = 0; x < XCD_NUM; x++) {
        gpuErrchk(hipMemcpy(h_cycles[x].data(), &d_cycles[(size_t)x * n_chunks],
                            sizeof(uint32_t) * n_chunks, hipMemcpyDeviceToHost));
    }

    const int xcd_per_cc = XCD_NUM / CC_NUM;
    chunk_home_xcd.assign(n_chunks, UINT32_MAX);
    chunks_per_cc.assign(CC_NUM, vector<uint64_t>());
    vector<size_t> chunks_per_xcd(XCD_NUM, 0);
    for (size_t k = 0; k < n_chunks; k++) {
        uint32_t min_cycles = UINT32_MAX;
        int min_xcc = -1;
        for (int x = 0; x < XCD_NUM; x++) {
            if (h_cycles[x][k] < min_cycles) {
                min_cycles = h_cycles[x][k];
                min_xcc = x;
            }
        }
        chunk_home_xcd[k] = (uint32_t)min_xcc;
        chunks_per_xcd[min_xcc]++;
        chunks_per_cc[min_xcc / xcd_per_cc].push_back(
            reinterpret_cast<uint64_t>(d_data + k * CHUNK_SIZE));
        #if DEBUG_DB_HOME
        printf("chunk[%zu]: cycles per xcd [", k);
        for (int x = 0; x < XCD_NUM; x++)
            printf("%s%u", x ? " " : "", h_cycles[x][k]);
        printf("] -> home xcd %d (cc%d, %u cycles)\n",
               min_xcc, min_xcc / xcd_per_cc, min_cycles);
        #endif
    }
    #if DEBUG_DB_HOME
    for (int x = 0; x < XCD_NUM; x++)
        printf("xcd %d: n_chunks %zu\n", x, chunks_per_xcd[x]);
    for (int c = 0; c < CC_NUM; c++)
        printf("cc %d: n_chunks %zu\n", c, chunks_per_cc[c].size());
    printf("\n");
    #endif

    gpuErrchk(hipFree(d_cycles));
    return 0;
}

} // namespace db

// ============================================================
// Kernel 1: VecScan — IVF-like vector similarity search
//
// Scenario: a GPU vector DB (e.g. pgvector, Milvus on GPU) has
// a "hot" embedding table in HBM4 that every XCD must fetch to
// answer the majority of ANN queries.
//
// Queries are generated from centroids.  HOT_FRAC_PCT% route to a hot
// centroid list stored as 2KB slabs in a chosen HBM; the remaining queries
// route to cold centroid lists in `cold_entries`.  The kernel scans the
// first-class candidates for that centroid and returns top-1.
// ============================================================
struct VecScanArgs {
    const uint64_t *__restrict__ hot_chunk_ptrs; // N_HOT_CHUNKS device addresses in target HBM
    const float    *__restrict__ cold_entries;   // N_COLD * DB_ENTRY_DIM floats
    const float    *__restrict__ query_vecs;     // Q * DB_ENTRY_DIM floats
    const int      *__restrict__ candidate_ids;  // Q * n_candidates encoded vector ids
    float          *__restrict__ results;        // Q floats (dot-product scores)
    int            *__restrict__ result_ids;     // Q encoded vector ids
    int Q;
    int N_COLD;
    int n_candidates;
    // Per-block timing (nullptr to disable); indexed by physical block id
    uint64_t *__restrict__ block_start_clk;
    uint64_t *__restrict__ block_end_clk;
    int      *__restrict__ block_n_queries;
    // Per-query latency (nullptr to disable); indexed by query id
    uint64_t *__restrict__ query_latencies;
    uint32_t *__restrict__ query_bid;
    // Bitmask of XCDs that execute queries (0 = all XCDs active)
    int active_xcd_mask;
};

struct VecScanTargetFn {
    __device__ __forceinline__
    void operator()(const VecScanArgs *__restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX,
                    int n_ppnt_tbs_in_xcd) const
    {
        uint32_t xcd_lane;
        asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcd_lane));

        if (a->active_xcd_mask != 0 && !((a->active_xcd_mask >> xcd_lane) & 1))
            return;
#if DEBUG_DB
        printf("VecScanTargetFn: bid=%d tid=%d xcd_lane=%d\n", bid, tid, xcd_lane);
#endif

        int n_tbs_in_xcd    = gridDimX / XCD_NUM;
        int logical_bid     = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_sz = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;

        // Shared memory for block-level reduction (blockDimX <= 1024)
        __shared__ float shmem[1024];

        if (tid == 0 && a->block_start_clk)
            a->block_start_clk[bid] = __builtin_readcyclecounter();
        int n_q = 0;

        // Each block iterates over queries in a strided grid loop
        for (int q = logical_bid; q < a->Q; q += logical_grid_sz) {
            n_q++;
            // All threads read the start clock (no divergence); used by tid==0 below.
            uint64_t q_start = __builtin_readcyclecounter();

            float best_score = -FLT_MAX;
            int best_id = -1;
            for (int ci = 0; ci < a->n_candidates; ci++) {
                int cand_id = a->candidate_ids[(size_t)q * a->n_candidates + ci];
                const float *target;
                if (cand_id < N_HOT_CHUNKS) {
                    target = reinterpret_cast<const float *>(a->hot_chunk_ptrs[cand_id]);
                } else {
                    int cold_id = cand_id - N_HOT_CHUNKS;
                    target = a->cold_entries + (size_t)(cold_id % a->N_COLD) * DB_ENTRY_DIM;
                }

                // Each thread accumulates its slice of one candidate dot product.
                // With blockDimX=1024 and DB_ENTRY_DIM=512, threads 512..1023 contribute 0.
                float partial = 0.0f;
                for (int d = tid; d < DB_ENTRY_DIM; d += blockDimX)
                    partial += a->query_vecs[(size_t)q * DB_ENTRY_DIM + d] * target[d];

                shmem[tid] = partial;
                __syncthreads();
                for (int s = blockDimX >> 1; s > 0; s >>= 1) {
                    if (tid < s) shmem[tid] += shmem[tid + s];
                    __syncthreads();
                }
                if (tid == 0 && shmem[0] > best_score) {
                    best_score = shmem[0];
                    best_id = cand_id;
                }
                __syncthreads();
            }
            if (tid == 0) {
                a->results[q] = best_score;
                if (a->result_ids) a->result_ids[q] = best_id;
                // Write-once: the target fn is called once per ping spec in the
                // n_plan loop; only the first call (lowest bpx) writes each query
                // so bid reflects the block that naturally owns the query.
                if (a->query_latencies && a->query_latencies[q] == 0) {
                    a->query_latencies[q] = __builtin_readcyclecounter() - q_start;
                    a->query_bid[q]       = (uint32_t)bid;
                }
            }
        }

        if (tid == 0 && a->block_end_clk) {
            a->block_end_clk[bid]   = __builtin_readcyclecounter();
            a->block_n_queries[bid] = n_q;
        }
    }
};

// Persistent priority embedding scan. Requests arrive at a fixed rate and
// traverse every chunk in the target CC uniformly.
struct PriorityEngineArgs {
    const uint64_t *__restrict__ chunk_ptrs;
    float          *__restrict__ worker_sinks;
    size_t n_chunks;
    int priority_xcd;
    int worker_limit;
    uint64_t arrival_interval_cycles;
    volatile int *__restrict__ stop;
    unsigned long long *__restrict__ start_cycle;
    unsigned long long *__restrict__ next_request;
    unsigned long long *__restrict__ completed_requests;
    unsigned long long *__restrict__ worker_slots_claimed;
    uint64_t *__restrict__ queue_latencies;
    uint64_t *__restrict__ service_latencies;
    size_t metrics_capacity;
    unsigned long long *__restrict__ load_cycle_sums;
    unsigned long long *__restrict__ load_counts;
};

__global__ void priority_engine(PriorityEngineArgs *a)
{
    const int tid = threadIdx.x;
    uint32_t xcd_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcd_id));

    if ((int)xcd_id != a->priority_xcd)
        return;

    __shared__ int active_worker;
    __shared__ int worker_slot;
    __shared__ int should_stop;
    __shared__ unsigned long long request_id;
    __shared__ unsigned long long arrival_cycle;
    __shared__ float reduction[P_BLOCKDIM_X];
    unsigned long long thread_load_cycle_sum = 0;
    unsigned long long thread_load_count = 0;
    unsigned long long worker_completed = 0;

    if (tid == 0) {
        unsigned long long slot = atomicAdd(a->worker_slots_claimed, 1ULL);
        worker_slot = (int)slot;
        active_worker = slot < (unsigned long long)a->worker_limit;
    }
    __syncthreads();
    if (!active_worker)
        return;

    if (tid == 0) {
        unsigned long long now = __builtin_readcyclecounter();
        unsigned long long proposed_start = now + 2 * a->arrival_interval_cycles;
        atomicCAS(a->start_cycle, 0ULL, proposed_start);
    }
    __syncthreads();

    // Static round-robin dispatch: worker w owns requests w, w+W, w+2W, ...
    // Dispatch needs no shared state, so per-request overhead is one clock spin;
    // a backlogged worker sees arrival in the past and serves immediately.
    while (true) {
        if (tid == 0) {
            should_stop = 0;
            const unsigned long long next =
                worker_completed * (unsigned long long)a->worker_limit +
                (unsigned long long)worker_slot;
            const unsigned long long arrival =
                *a->start_cycle + next * a->arrival_interval_cycles;
            while (true) {
                if (*a->stop) {
                    should_stop = 1;
                    break;
                }
                if (__builtin_readcyclecounter() >= arrival) {
                    request_id = next;
                    arrival_cycle = arrival;
                    break;
                }
            }
        }
        __syncthreads();
        if (should_stop)
            break;

        const size_t chunk_index =
            (size_t)(request_id % (unsigned long long)a->n_chunks);
        const float *target =
            reinterpret_cast<const float *>(a->chunk_ptrs[chunk_index]);
        const unsigned long long service_start = __builtin_readcyclecounter();
        float partial = 0.0f;
        for (int d = tid; d < DB_ENTRY_DIM; d += blockDim.x) {
#if P_MEASURE_LOAD_LATENCY
            const unsigned long long load_start = __builtin_readcyclecounter();
            const float value = target[d];
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            const unsigned long long load_end = __builtin_readcyclecounter();
            thread_load_cycle_sum += load_end - load_start;
            thread_load_count++;
            partial += value;
#else
            partial += target[d];
#endif
        }

        reduction[tid] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s)
                reduction[tid] += reduction[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            const unsigned long long completion = __builtin_readcyclecounter();
            a->worker_sinks[worker_slot] = reduction[0];
            if (a->metrics_capacity > 0) {
                const size_t slot = (size_t)(request_id % a->metrics_capacity);
                a->queue_latencies[slot] = completion - arrival_cycle;
                a->service_latencies[slot] = completion - service_start;
            }
            worker_completed++;
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(a->completed_requests, worker_completed);

#if P_MEASURE_LOAD_LATENCY
    const size_t load_slot = (size_t)worker_slot * blockDim.x + tid;
    a->load_cycle_sums[load_slot] = thread_load_cycle_sum;
    a->load_counts[load_slot] = thread_load_count;
#endif
}

// Closed-loop best-effort embedding scan. Every enabled XCD sweeps the complete
// chunk list, while workers within an XCD process disjoint batches. Each worker
// issues a batch of independent chunk loads before one reduction and counter
// update to make BE bandwidth-intensive.
struct BestEffortEngineArgs {
    const uint64_t *__restrict__ chunk_ptrs_per_cc[CC_NUM];
    size_t n_chunks_per_cc[CC_NUM];
    float          *__restrict__ worker_sinks;
    int worker_limit_per_xcd;
    uint32_t active_xcd_mask;
    volatile int *__restrict__ stop;
    unsigned long long *__restrict__ completed_chunks_per_xcd;
    unsigned long long *__restrict__ worker_slots_claimed_per_xcd;
    unsigned long long *__restrict__ load_cycle_sums;
    unsigned long long *__restrict__ load_counts;
};

// Policy d: decides whether a BE batch is skipped.  Tweak granularity here
// the same way as be_pick_target_cc (per-batch is stationary; per-worker or
// per-epoch variants hash worker_slot or batch_id / epoch_len instead).
__device__ __forceinline__
bool be_batch_throttled(uint32_t xcd_id, unsigned long long batch_id)
{
#if BE_THROTTLE_PCT == 0
    (void)xcd_id; (void)batch_id;
    return false;
#else
    unsigned long long h =
        (batch_id * 0xd1b54a32d192ed03ULL) ^ ((unsigned long long)xcd_id << 32);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    return (int)(h % 100ULL) < BE_THROTTLE_PCT;
#endif
}

// Picks the CC a BE batch reads from.  Tweak granularity/distribution here
// (per-batch is stationary; per-worker would be `hash(xcd_id, worker_slot)`,
// per-epoch `hash(xcd_id, batch_id / epoch_len)`).
__device__ __forceinline__
int be_pick_target_cc(uint32_t xcd_id, unsigned long long batch_id)
{
#if BE_TARGET_POLICY == 0
    (void)xcd_id; (void)batch_id;
    return P_TARGET_CC;
#else
    unsigned long long h =
        (batch_id * 0x9e3779b97f4a7c15ULL) ^ ((unsigned long long)xcd_id << 32);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
#if BE_TARGET_POLICY == 1
    // Policy c: biased random per (xcd, batch).  P's CC with probability
    // BE_RANDOM_CC0_PCT, else uniform over the other CCs.
    if ((int)(h % 100ULL) < BE_RANDOM_CC0_PCT)
        return P_TARGET_CC;
#endif
    // Policy e (and policy c's non-P remainder): uniform over non-P CCs.
    const int other = (int)((h >> 8) % (unsigned long long)(CC_NUM - 1));
    return other + (other >= P_TARGET_CC ? 1 : 0);
#endif
}

__global__ void best_effort_engine(BestEffortEngineArgs *a)
{
    const int tid = threadIdx.x;
    uint32_t xcd_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcd_id));

    if (xcd_id >= XCD_NUM || !(a->active_xcd_mask & (1u << xcd_id)))
        return;

    __shared__ int worker_slot;
    __shared__ int active_worker;
    __shared__ int should_stop;
    __shared__ unsigned long long iteration;
    __shared__ float reduction[BE_BLOCKDIM_X];
    unsigned long long thread_load_cycle_sum = 0;
    unsigned long long thread_load_count = 0;

    if (tid == 0) {
        const unsigned long long slot =
            atomicAdd(&a->worker_slots_claimed_per_xcd[xcd_id], 1ULL);
        worker_slot = (int)slot;
        active_worker = slot < (unsigned long long)a->worker_limit_per_xcd;
        iteration = 0;
    }
    __syncthreads();
    if (!active_worker)
        return;

    while (true) {
        if (tid == 0)
            should_stop = *a->stop;
        __syncthreads();
        if (should_stop)
            break;

        const unsigned long long batch_id =
            iteration * (unsigned long long)a->worker_limit_per_xcd +
            (unsigned long long)worker_slot;
        if (be_batch_throttled(xcd_id, batch_id)) {
            // Skipped batch: idle for one batch's duration; no loads, no
            // completed-chunk credit (BE GB/s honestly shows the sacrifice).
            if (tid == 0) {
                const unsigned long long idle_start = __builtin_readcyclecounter();
                while (__builtin_readcyclecounter() - idle_start <
                           (unsigned long long)BE_THROTTLE_IDLE_CYCLES &&
                       !*a->stop) {}
                iteration++;
            }
            __syncthreads();
            continue;
        }
        const int target_cc = be_pick_target_cc(xcd_id, batch_id);
        const uint64_t *__restrict__ chunk_ptrs = a->chunk_ptrs_per_cc[target_cc];
        const size_t n_chunks = a->n_chunks_per_cc[target_cc];
#if BE_HASHED_BATCHES
        // Hashed batch placement: scatter the batch base per (xcd, batch) so
        // XCDs scan uncorrelated windows.  Sequential laps convoy across XCDs
        // and turn the scan into MALL hits instead of HBM traffic.  Batches
        // stay 64-chunk sequential.
        const unsigned long long mix =
            batch_id * 0x9e3779b97f4a7c15ULL ^ ((unsigned long long)xcd_id << 32);
        const size_t batch_base =
            (size_t)((mix * 2654435761ULL) % (unsigned long long)n_chunks);
#else   // Sequential laps: each XCD sweeps the full chunk list in order,
        // phase-shifted by 1/XCD_NUM of the list.
        const size_t xcd_phase =
            (size_t)xcd_id * (n_chunks / (size_t)XCD_NUM);
        const size_t batch_base =
            (xcd_phase +
             (size_t)((batch_id * (unsigned long long)BE_BATCH_CHUNKS) %
                      (unsigned long long)n_chunks)) %
            n_chunks;
#endif

        // dwordx4 loads: each thread reads one float4 (16B); a 512-thread block
        // covers 4 full 2KB chunks per iteration (128 float4 lanes per chunk).
        const int chunk_in_quad = tid >> 7;
        const int lane = tid & 127;

        float accum = 0.0f;
#pragma unroll 4
        for (int chunk = 0; chunk < BE_BATCH_CHUNKS; chunk += 4) {
            const size_t index =
                (batch_base + (size_t)chunk + (size_t)chunk_in_quad) % n_chunks;
            const float4 *target =
                reinterpret_cast<const float4 *>(chunk_ptrs[index]);
#if BE_MEASURE_LOAD_LATENCY
            const unsigned long long load_start = __builtin_readcyclecounter();
            const float4 value = target[lane];
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
            const unsigned long long load_end = __builtin_readcyclecounter();
            thread_load_cycle_sum += load_end - load_start;
            thread_load_count++;
#else
            const float4 value = target[lane];
#endif
            accum += value.x + value.y + value.z + value.w;
        }
        const float partial = accum;

        reduction[tid] = partial;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s)
                reduction[tid] += reduction[tid + s];
            __syncthreads();
        }

        if (tid == 0) {
            const size_t sink_index =
                (size_t)xcd_id * a->worker_limit_per_xcd + worker_slot;
            a->worker_sinks[sink_index] = reduction[0];
            atomicAdd(&a->completed_chunks_per_xcd[xcd_id],
                      (unsigned long long)BE_BATCH_CHUNKS);
            iteration++;
        }
        __syncthreads();
    }

#if BE_MEASURE_LOAD_LATENCY
    const size_t load_slot =
        ((size_t)xcd_id * a->worker_limit_per_xcd + worker_slot) * blockDim.x + tid;
    a->load_cycle_sums[load_slot] = thread_load_cycle_sum;
    a->load_counts[load_slot] = thread_load_count;
#endif
}
