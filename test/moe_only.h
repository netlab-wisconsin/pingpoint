#pragma once

#include <vector>
#include <random>
#include <cassert>
#include <cstddef>
#include <algorithm>

#define IMBALANCED_DISTRIBUTION 0

#if IMBALANCED_DISTRIBUTION
#define HOT_EXPERT_ID 0 // index of hot expert
#define HOT_EXPERT_FRACTION 0.9 // fraction of tokens assigned to the hot expert
#endif

#define DEBUG_MOE_KERNEL 0

using namespace std;

static void fill_random(float* v, size_t n, float scale = 0.01f)
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (size_t i = 0; i < n; ++i) {
        v[i] = dist(rng);
    }
}

static void fill_expert_ids(int* ids, size_t n, int E)
{
    std::mt19937 rng(456);

#if IMBALANCED_DISTRIBUTION
    assert(E > 0);
    assert(HOT_EXPERT_ID < E);
    std::vector<double> p(E, (1.0 - HOT_EXPERT_FRACTION) / (double)(E - 1));
    p[HOT_EXPERT_ID] = HOT_EXPERT_FRACTION;
    std::discrete_distribution<int> dist(p.begin(), p.end());
#else
    std::uniform_int_distribution<int> dist(0, E - 1);
#endif

    for (size_t i = 0; i < n; ++i) {
        ids[i] = dist(rng) % E;   // the %E is redundant in uniform mode but kept to match your original
    }
}

#if IMBALANCED_DISTRIBUTION
// deterministic version of fill_expert_ids with fixed 90% to hot expert
static void fill_expert_ids_fixed(int* ids, size_t T, int E,
                                  int hot_expert = HOT_EXPERT_ID,
                                  double hot_frac = HOT_EXPERT_FRACTION)
{
    // Preconditions
    if (!ids || T == 0 || E <= 0) return;
    if (hot_expert < 0 || hot_expert >= E) hot_expert = 0;
    hot_frac = std::clamp(hot_frac, 0.0, 1.0);

    // Count assignment
    size_t hot_cnt = static_cast<size_t>(hot_frac * static_cast<double>(T));
    if (hot_cnt > T) hot_cnt = T;

    size_t rem = T - hot_cnt;

    // Fill hot expert
    for (size_t i = 0; i < hot_cnt; ++i) ids[i] = hot_expert;

    // Fill remaining experts uniformly over the leftover tokens
    if (E == 1) return;  // all tokens already assigned to expert 0

    size_t n_other = static_cast<size_t>(E - 1);
    size_t per = rem / n_other;
    size_t extra = rem % n_other;   // first 'extra' experts get one more token

    size_t idx = hot_cnt;
    for (int e = 0; e < E; ++e) {
        if (e == hot_expert) continue;
        size_t cnt = per + (extra ? 1 : 0);
        if (extra) --extra;

        for (size_t k = 0; k < cnt; ++k) ids[idx++] = e;
    }
}
#else
// just a placeholder of fill_expert_ids_fixed
// without the hot_expert and hot_frac params
static void fill_expert_ids_fixed(int*, size_t, int){
    // call error in case of accidental usage
    fprintf(stderr, "Error: fill_expert_ids_fixed should not be called in uniform distribution mode.\n");
    exit(1);
}
#endif


// =============================================================================================
// ATTENTION KERNEL 1: QKV Projection (X -> QKV)
// =============================================================================================
struct AttnQKVArgs {
    const float* __restrict__ X;    // [T, d]
    const float* __restrict__ Wqkv; // [d, 3*d]
    float* __restrict__ QKV;        // [T, 3*d]
    int T, d;
};

struct AttnQKVTargetFn {
    __device__ __forceinline__
    void operator()(const AttnQKVArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const 
    {
        // 1. PPNT Grid Setup
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM;

        // 2. Dimensions
        int out_dim = 3 * a->d; // Q, K, V concatenated
        int block_x = 32; 
        int block_y = 32;
        
        int grid_dim_x = (out_dim + block_x - 1) / block_x;
        int grid_dim_y = (a->T    + block_y - 1) / block_y;
        int total_tiles = grid_dim_x * grid_dim_y;

        // 3. Grid-Stride Loop
        for (int curr_tile = logical_bid; curr_tile < total_tiles; curr_tile += logical_grid_size) {
            
            int by = curr_tile / grid_dim_x; // Token chunk
            int bx = curr_tile % grid_dim_x; // Hidden chunk

            int ty = tid / block_x; 
            int tx = tid % block_x;

            int j = bx * block_x + tx; // output dim (0 to 3d)
            int t = by * block_y + ty; // token (0 to T)
            
            if (t >= a->T || j >= out_dim) continue;

            // Perform GEMM: QKV[t, j] = Sum( X[t, k] * W[k, j] )
            const float* x_row = a->X + (size_t)t * a->d;
            float acc = 0.f;
            
            // Iterate over input dim 'd'
            for (int k = 0; k < a->d; k++) {
                // W is [d, 3d] row-major
                acc += x_row[k] * a->Wqkv[(size_t)k * out_dim + j];
            }
            a->QKV[(size_t)t * out_dim + j] = acc;
        }
    }
};


// =============================================================================================
// ATTENTION KERNEL 2: Output Projection (Ctx -> Out)
// =============================================================================================
// Simplified: Takes QKV buffer (treating it as 'Context') and projects back to d.
struct AttnOutArgs {
    const float* __restrict__ In;  // [T, 3*d] (Simulated Context)
    const float* __restrict__ Wo;  // [3*d, d] (Simulated Output Weight to reduce dim)
    float* __restrict__ Out;       // [T, d]
    int T, d;
};

struct AttnOutTargetFn {
    __device__ __forceinline__
    void operator()(const AttnOutArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM;

        int in_dim = 3 * a->d; 
        int out_dim = a->d;
        
        int block_x = 32; 
        int block_y = 32;
        int grid_dim_x = (out_dim + block_x - 1) / block_x;
        int grid_dim_y = (a->T    + block_y - 1) / block_y;
        int total_tiles = grid_dim_x * grid_dim_y;

        for (int curr_tile = logical_bid; curr_tile < total_tiles; curr_tile += logical_grid_size) {
            
            int by = curr_tile / grid_dim_x; 
            int bx = curr_tile % grid_dim_x;

            int ty = tid / block_x; 
            int tx = tid % block_x;

            int j = bx * block_x + tx; 
            int t = by * block_y + ty; 
            
            if (t >= a->T || j >= out_dim) continue;

            const float* in_row = a->In + (size_t)t * in_dim;
            float acc = 0.f;
            
            // Project 3d -> d
            for (int k = 0; k < in_dim; k++) {
                acc += in_row[k] * a->Wo[(size_t)k * out_dim + j];
            }
            a->Out[(size_t)t * out_dim + j] = acc;
        }
    }
};

// =============================================================================================
// SCATTER
// =============================================================================================

struct ScatterArgs {
    const float* X;            // [T, d]
    const int*   expert_id;    // [T]
    float*       Xexp;         // [E*cap, d]
    int*         pos;          // [T]
    int*         counters;     // [E] (must be zeroed before launch)
    int          T, d, E, cap;
};

__device__ __forceinline__
void moe_scatter_body(int t, const ScatterArgs* __restrict__ a) {
    if (t >= a->T) return;

    int e = a->expert_id[t];
    if (e < 0 || e >= a->E) return;

    int local = atomicAdd(&a->counters[e], 1);
    if (local >= a->cap) {
        a->pos[t] = -1;
        return;
    }

    int slot = e * a->cap + local;
    a->pos[t] = slot;

    const float* src = a->X + (size_t)t * a->d;
    float* dst = a->Xexp + (size_t)slot * a->d;
    for (int j = 0; j < a->d; j++) dst[j] = src[j];
}

struct ScatterTargetFn {
    __device__ __forceinline__
    void operator()(const ScatterArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const 
    {
        // 1. PPNT Logic Setup
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);

        // Calculate logical grid size (Total physical blocks - Reserved blocks)
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM; 

        // 2. Parallelize over Tokens (T)
        // Global stride = total number of threads in the logical grid
        int total_worker_threads = logical_grid_size * blockDimX;
        
        // Start index for this specific thread
        int start_t = logical_bid * blockDimX + tid;

        // 3. Grid-Stride Loop
        // Iterate until all tokens T are processed
        for (int t = start_t; t < a->T; t += total_worker_threads) {
            moe_scatter_body(t, a);
        }
    }
};


// =============================================================================================
// GATHER
// =============================================================================================

struct GatherArgs {
    const float* __restrict__ Src; // Input: [E*cap, d] (was Yexp)
    const int* __restrict__ pos;   // Input: [T]
    float* __restrict__ Dst;       // Output: [T, d] (was Y)
    int T, d, cap;
};

struct GatherTargetFn {
    __device__ __forceinline__
    void operator()(const GatherArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const 
    {
        // 1. PPNT Logic setup
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM;

        // 2. Parallelize over Tokens (T)
        // Global stride = total number of threads in the logical grid
        int total_worker_threads = logical_grid_size * blockDimX;
        
        // Start index for this specific thread
        int start_t = logical_bid * blockDimX + tid;

        // 3. Grid-Stride Loop
        for (int t = start_t; t < a->T; t += total_worker_threads) {
            
            int slot = a->pos[t];
            float* dst_ptr = a->Dst + (size_t)t * a->d;

            if (slot < 0) {
                // Token was dropped or padding; zero output
                for (int j = 0; j < a->d; j++) {
                    dst_ptr[j] = 0.f;
                }
            } else {
                // Gather copy
                // Src is [E * cap, d]
                const float* src_ptr = a->Src + (size_t)slot * a->d;
                for (int j = 0; j < a->d; j++) {
                    dst_ptr[j] = src_ptr[j];
                }
            }
        }
    }
};


// =============================================================================================
// GEMM 1 (FFN Up Projection)
// =============================================================================================

struct Gemm1Args {
    const float* __restrict__ Xexp; // [E*cap, d]
    const float* __restrict__ W1;   // [E, d, hidden]
    float* __restrict__ Tmp;        // [E*cap, hidden]
    const int* __restrict__ cnt;    // [E]
    int E, cap, d, hidden;
};

struct Gemm1TargetFn {
    __device__ __forceinline__
    void operator()(const Gemm1Args* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM;

        int block_x = 32; int block_y = 32;
        int grid_dim_x = (a->hidden + block_x - 1) / block_x;
        int grid_dim_y = (a->cap    + block_y - 1) / block_y;
        int total_tiles = grid_dim_x * grid_dim_y * a->E;

        for (int curr_tile = logical_bid; curr_tile < total_tiles; curr_tile += logical_grid_size) {
            
            int slice = grid_dim_x * grid_dim_y;
            int e   = curr_tile / slice;          
            int rem = curr_tile % slice;
            int by  = rem / grid_dim_x;             
            int bx  = rem % grid_dim_x;             

            int ty = tid / block_x; 
            int tx = tid % block_x;

            int j = bx * block_x + tx; 
            int t = by * block_y + ty; 
            
            if (t >= a->cap || j >= a->hidden) continue;

            // Use size_t for global offset calculation to avoid overflow
            size_t global_idx = (size_t)(e * a->cap + t) * (size_t)a->hidden + j;

            int ne = a->cnt[e];
            if (t >= ne) {
                a->Tmp[global_idx] = 0.f;
                continue;
            }

            // Fix Input Indexing: cast to size_t
            // x_offset = (e * cap + t) * d
            const float* x = a->Xexp + (size_t)(e * a->cap + t) * (size_t)a->d;
            
            // Fix Weight Indexing: cast to size_t
            // w_offset = e * d * hidden
            const float* w = a->W1 + (size_t)e * (size_t)a->d * (size_t)a->hidden;
            
            float acc = 0.f;
            for (int k = 0; k < a->d; k++) {
                acc += x[k] * w[(size_t)k * a->hidden + j];
            }
            a->Tmp[global_idx] = acc;
        }
    }
};


// =============================================================================================
// GEMM 2 (FFN Down Projection)
// =============================================================================================

struct Gemm2Args {
    const float* __restrict__ Tmp; // [E*cap, hidden]
    const float* __restrict__ W2;  // [E, hidden, d]
    float* __restrict__ Yexp;      // [E*cap, d]
    const int* __restrict__ cnt;   // [E]
    int E, cap, d, hidden;
};

struct Gemm2TargetFn {
    __device__ __forceinline__
    void operator()(const Gemm2Args* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM;

        int block_x = 32; int block_y = 32;
        int grid_dim_x = (a->d   + block_x - 1) / block_x; 
        int grid_dim_y = (a->cap + block_y - 1) / block_y;
        int total_tiles = grid_dim_x * grid_dim_y * a->E;

        for (int curr_tile = logical_bid; curr_tile < total_tiles; curr_tile += logical_grid_size) {
            
            int slice = grid_dim_x * grid_dim_y;
            int e   = curr_tile / slice; 
            int rem = curr_tile % slice;
            int by  = rem / grid_dim_x; 
            int bx  = rem % grid_dim_x;

            int ty = tid / block_x; 
            int tx = tid % block_x;

            int j = bx * block_x + tx; 
            int t = by * block_y + ty; 
            
            if (t >= a->cap || j >= a->d) continue;

            size_t global_idx = (size_t)(e * a->cap + t) * (size_t)a->d + j;

            int ne = a->cnt[e];
            if (t >= ne) {
                a->Yexp[global_idx] = 0.f;
                continue;
            }

            // Fix Input (Tmp) Indexing
            const float* tmp = a->Tmp + (size_t)(e * a->cap + t) * (size_t)a->hidden;
            
            // Fix Weight Indexing
            const float* w = a->W2 + (size_t)e * (size_t)a->hidden * (size_t)a->d;
            
            float acc = 0.f;
            for (int k = 0; k < a->hidden; k++) {
                acc += tmp[k] * w[(size_t)k * a->d + j];
            }
            a->Yexp[global_idx] = acc;
        }
    }
};


// =============================================================================================
// ReLU (Element-wise)
// =============================================================================================

struct ReluArgs {
    float* __restrict__ Tmp;
    const int* __restrict__ cnt;
    int E, cap, hidden;
};

struct ReluTargetFn {
    __device__ __forceinline__
    void operator()(const ReluArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM;

        // Use size_t for total elements
        size_t total_elems = (size_t)a->E * (size_t)a->cap * (size_t)a->hidden;
        size_t stride = (size_t)logical_grid_size * blockDimX;
        size_t start_idx = (size_t)logical_bid * blockDimX + tid;

        for (size_t idx = start_idx; idx < total_elems; idx += stride) {
            
            // Reconstruct indices carefully
            // e = idx / (cap * hidden)
            // t = (idx / hidden) % cap
            size_t cap_hidden = (size_t)a->cap * a->hidden;
            int e = idx / cap_hidden;
            int t = (idx / a->hidden) % a->cap;

            if (t >= a->cnt[e]) continue;

            float val = a->Tmp[idx];
            a->Tmp[idx] = val > 0.f ? val : 0.f;
        }
    }
};