#pragma once

#include <vector>
#include <random>
#include <cassert>
#include <cstddef>
#include <algorithm>

using namespace std;

// Floats in one 2KB chunk
#define DB_ENTRY_DIM (CHUNK_SIZE / (int)sizeof(float))  // 512

// Number of 2KB chunks in the hot table (8MB >> L2 size of 4MB per XCD).
// Each hot query accesses a *different* chunk (q % N_HOT_CHUNKS), so the
// working set across all blocks on one XCD is up to 8MB and forces HBM reads.
// Rule of thumb: need Q * 0.9 / XCD_NUM > N_HOT_CHUNKS/2 for >50% L2 miss rate
// → with N_HOT_CHUNKS=4096 and Q=32768: 3686 hot queries/XCD > 2048. ✓
#define N_HOT_CHUNKS 4096

// Dimension of the hot aggregator row
#define DB_AGG_DIM 64

// "Bad" placement: hot table lives in HBM4, far from XCD0 and XCD1
#define HOT_HBM 4

// "Good" placement: hot table lives in HBM0, local to XCD0
#define LOCAL_HBM 0

// Fraction (0-100) of queries that target the hot table
#define HOT_FRAC_PCT 90

static void db_fill_random(float *v, size_t n, float scale = 1.0f)
{
    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
}

// ============================================================
// Kernel 1: VecScan — vector similarity search
//
// Scenario: a GPU vector DB (e.g. pgvector, Milvus on GPU) has
// a "hot" embedding table in HBM4 that every XCD must fetch to
// answer the majority of ANN queries.
//
// HOT_FRAC_PCT% of queries index into `hot_chunk_ptrs` (N_HOT_CHUNKS
// 2KB slabs physically in HOT_HBM, total 8MB >> L2=4MB).  Each hot
// query picks chunk  q % N_HOT_CHUNKS, so the working set across all
// blocks on one XCD spans the full 8MB and forces HBM4 reads.
// Cold queries read from `cold_entries` (distributed HBMs).
// ============================================================
struct VecScanArgs {
    const uint64_t *__restrict__ hot_chunk_ptrs; // N_HOT_CHUNKS device addresses in HOT_HBM
    const float    *__restrict__ cold_entries;   // N_COLD * DB_ENTRY_DIM floats
    const float    *__restrict__ query_vecs;     // Q * DB_ENTRY_DIM floats
    float          *__restrict__ results;        // Q floats (dot-product scores)
    int Q;
    int N_COLD;
    int hot_frac_pct;
};

struct VecScanTargetFn {
    __device__ __forceinline__
    void operator()(const VecScanArgs *__restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX,
                    int n_ppnt_tbs_in_xcd) const
    {
        int n_tbs_in_xcd    = gridDimX / XCD_NUM;
        int logical_bid     = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_sz = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;

        // Shared memory for block-level reduction (blockDimX <= 1024)
        __shared__ float shmem[1024];

        for (int q = logical_bid; q < a->Q; q += logical_grid_sz) {
            bool is_hot = (q % 100) < a->hot_frac_pct;
            const float *target;
            if (is_hot) {
                // Each hot query lands on a different 2KB chunk within the 8MB hot table.
                // The L2 (4MB/XCD) cannot hold the full table, so reads go to HBM.
                target = reinterpret_cast<const float *>(
                    a->hot_chunk_ptrs[q % N_HOT_CHUNKS]);
            } else {
                target = a->cold_entries + (size_t)(q % a->N_COLD) * DB_ENTRY_DIM;
            }

            // Each thread accumulates its slice of the dot product.
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
            if (tid == 0) a->results[q] = shmem[0];
        }
    }
};

// ============================================================
// Kernel 2: AtomicAgg — hot-key aggregation update
//
// Scenario: a GPU analytics engine runs GROUP BY / COUNT(*) over
// a skewed dataset.  The "hot group" accumulator is spread across
// N_HOT_CHUNKS 2KB slabs in HOT_HBM (8MB total).  Each aggregation
// round n writes to a different slab (n % N_HOT_CHUNKS), so the
// write working set also exceeds L2 and forces HBM4 traffic.
// ============================================================
struct AtomicAggArgs {
    const float    *__restrict__ input_vals;    // N * DB_AGG_DIM floats (per-round local data)
    const uint64_t *__restrict__ hot_agg_ptrs;  // N_HOT_CHUNKS device addresses in HOT_HBM
    int N; // number of aggregation rounds
};

struct AtomicAggTargetFn {
    __device__ __forceinline__
    void operator()(const AtomicAggArgs *__restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX,
                    int n_ppnt_tbs_in_xcd) const
    {
        int n_tbs_in_xcd    = gridDimX / XCD_NUM;
        int logical_bid     = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_sz = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;

        __shared__ float local_agg[DB_AGG_DIM];

        for (int n = logical_bid; n < a->N; n += logical_grid_sz) {
            // Each round targets a different slab in the 8MB hot table.
            float *hot_agg = reinterpret_cast<float *>(
                a->hot_agg_ptrs[n % N_HOT_CHUNKS]);

            // Load one input row into shared memory
            for (int d = tid; d < DB_AGG_DIM; d += blockDimX)
                local_agg[d] = a->input_vals[(size_t)n * DB_AGG_DIM + d];
            __syncthreads();

            // Atomic accumulate into the slab in HOT_HBM
            for (int d = tid; d < DB_AGG_DIM; d += blockDimX)
                atomicAdd(&hot_agg[d], local_agg[d]);
        }
    }
};
