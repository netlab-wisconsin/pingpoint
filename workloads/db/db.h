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

// "Bad" placement: hot table lives in HBM4, far from XCD0 and XCD1
#define HOT_HBM 4

// "Good" placement: hot table lives in HBM0, local to XCD0
#define LOCAL_HBM 0

// HBM used by XCD2/XCD3 in the focused experiment
#define XCD23_HBM 6

// Fraction (0-100) of queries that target the hot table
#define HOT_FRAC_PCT 90

// Synthetic IVF-like ANN shape.  The hot corpus is stored as N_HOT_CHUNKS
// vector chunks, split into centroid posting lists of DB_LIST_SIZE vectors.
#define DB_LIST_SIZE 64
#define DB_CANDIDATES_PER_QUERY DB_LIST_SIZE
#define DB_HOT_CENTROIDS (N_HOT_CHUNKS / DB_LIST_SIZE)

// Separate DB-owned placement buffer for VecScan.
#define DB_VECSCAN_DATA_SIZE (256 * 1024 * 1024)

static void db_fill_random(float *v, size_t n, float scale = 1.0f)
{
    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
}

namespace db {

__global__ void identify_home(void *data, size_t size, uint32_t *d_cycles, int n_chunks)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd;

    assert(n_tbs_in_xcd * XCD_NUM == gridDim.x);
    assert(n_tbs_in_xcd == 1);
    assert(blockDim.x == 128);

    db_cg::grid_group grid = db_cg::this_grid();
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
        grid.sync();

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
            grid.sync();
        }
    }
}

static int home_identification(
    char *d_data,
    size_t data_size,
    size_t n_chunks,
    vector<uint32_t> &chunk_home_xcd,
    vector<vector<uint64_t>> &chunks_per_hbm)
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
    gpuErrchk(hipLaunchCooperativeKernel(
        (void *)identify_home, dim3(XCD_NUM), dim3(128), kernel_args, 0, 0));
    gpuErrchk(hipDeviceSynchronize());

    vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(n_chunks));
    for (int x = 0; x < XCD_NUM; x++) {
        gpuErrchk(hipMemcpy(h_cycles[x].data(), &d_cycles[(size_t)x * n_chunks],
                            sizeof(uint32_t) * n_chunks, hipMemcpyDeviceToHost));
    }

    chunk_home_xcd.assign(n_chunks, UINT32_MAX);
    chunks_per_hbm.assign(HBM_NUM, vector<uint64_t>());
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
        chunks_per_hbm[min_xcc].push_back(
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
    const uint64_t *__restrict__ hot_chunk_ptrs; // N_HOT_CHUNKS device addresses in HOT_HBM
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
        int xcd_lane        = bid % XCD_NUM;
        if (a->active_xcd_mask != 0 && !((a->active_xcd_mask >> xcd_lane) & 1))
            return;

        int n_tbs_in_xcd    = gridDimX / XCD_NUM;
        int logical_bid     = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_sz = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;

        // Shared memory for block-level reduction (blockDimX <= 1024)
        __shared__ float shmem[1024];

        if (tid == 0 && a->block_start_clk)
            a->block_start_clk[bid] = __builtin_readcyclecounter();
        int n_q = 0;

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
