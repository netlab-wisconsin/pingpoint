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

    assert(n_tbs_in_xcd * XCD_NUM == gridDim.x);
    assert(n_tbs_in_xcd == 1);
    assert(blockDim.x == 128);

#if !DB_USE_GLOBAL_BARRIER
    db_cg::grid_group grid = db_cg::this_grid();
#endif
    uint32_t xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    uint4 *data_u4 = (uint4 *)data;
    const size_t inner_size = 64 * 1024 * 1024;
    const size_t n_outer = size / inner_size;
    const size_t n_inner = inner_size / (sizeof(uint4) * blockDim.x);
    assert(size % inner_size == 0);
    assert((size_t)n_chunks == n_outer * n_inner * n_tbs_in_xcd);

    for (size_t i = 0; i < n_outer; i++) {
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
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
        chunks_per_cc[min_xcc / xcd_per_cc].push_back(
            reinterpret_cast<uint64_t>(d_data + k * CHUNK_SIZE));
    }

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

    while (true) {
        if (tid == 0) {
            should_stop = 0;
            while (true) {
                if (*a->stop) {
                    should_stop = 1;
                    break;
                }

                const unsigned long long now = __builtin_readcyclecounter();
                const unsigned long long start = *a->start_cycle;
                if (now < start)
                    continue;

                const unsigned long long arrived =
                    1ULL + (now - start) / a->arrival_interval_cycles;
                unsigned long long next = *a->next_request;
                if (next >= arrived)
                    continue;

                if (atomicCAS(a->next_request, next, next + 1ULL) == next) {
                    request_id = next;
                    arrival_cycle = start + next * a->arrival_interval_cycles;
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
            atomicAdd(a->completed_requests, 1ULL);
        }
        __syncthreads();
    }

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
    const uint64_t *__restrict__ chunk_ptrs;
    float          *__restrict__ worker_sinks;
    size_t n_chunks;
    int worker_limit_per_xcd;
    uint32_t active_xcd_mask;
    volatile int *__restrict__ stop;
    unsigned long long *__restrict__ completed_chunks_per_xcd;
    unsigned long long *__restrict__ worker_slots_claimed_per_xcd;
    unsigned long long *__restrict__ load_cycle_sums;
    unsigned long long *__restrict__ load_counts;
};

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
        const size_t xcd_phase =
            (size_t)xcd_id * (a->n_chunks / (size_t)XCD_NUM);
        const size_t batch_base =
            (xcd_phase +
             (size_t)((batch_id * (unsigned long long)BE_BATCH_CHUNKS) %
                      (unsigned long long)a->n_chunks)) %
            a->n_chunks;

        float accum0 = 0.0f;
        float accum1 = 0.0f;
        float accum2 = 0.0f;
        float accum3 = 0.0f;
        for (int chunk = 0; chunk < BE_BATCH_CHUNKS; chunk += 4) {
            const size_t index0 = (batch_base + (size_t)chunk + 0) % a->n_chunks;
            const size_t index1 = (batch_base + (size_t)chunk + 1) % a->n_chunks;
            const size_t index2 = (batch_base + (size_t)chunk + 2) % a->n_chunks;
            const size_t index3 = (batch_base + (size_t)chunk + 3) % a->n_chunks;
            const float *target0 =
                reinterpret_cast<const float *>(a->chunk_ptrs[index0]);
            const float *target1 =
                reinterpret_cast<const float *>(a->chunk_ptrs[index1]);
            const float *target2 =
                reinterpret_cast<const float *>(a->chunk_ptrs[index2]);
            const float *target3 =
                reinterpret_cast<const float *>(a->chunk_ptrs[index3]);
            for (int d = tid; d < DB_ENTRY_DIM; d += blockDim.x) {
#if BE_MEASURE_LOAD_LATENCY
                unsigned long long load_start = __builtin_readcyclecounter();
                const float value0 = target0[d];
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                unsigned long long load_end = __builtin_readcyclecounter();
                thread_load_cycle_sum += load_end - load_start;

                load_start = __builtin_readcyclecounter();
                const float value1 = target1[d];
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                load_end = __builtin_readcyclecounter();
                thread_load_cycle_sum += load_end - load_start;

                load_start = __builtin_readcyclecounter();
                const float value2 = target2[d];
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                load_end = __builtin_readcyclecounter();
                thread_load_cycle_sum += load_end - load_start;

                load_start = __builtin_readcyclecounter();
                const float value3 = target3[d];
                asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
                load_end = __builtin_readcyclecounter();
                thread_load_cycle_sum += load_end - load_start;
                thread_load_count += 4;

                accum0 += value0;
                accum1 += value1;
                accum2 += value2;
                accum3 += value3;
#else
                accum0 += target0[d];
                accum1 += target1[d];
                accum2 += target2[d];
                accum3 += target3[d];
#endif
            }
        }
        const float partial = accum0 + accum1 + accum2 + accum3;

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
