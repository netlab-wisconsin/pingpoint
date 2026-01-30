#pragma once

#include <vector>
#include <random>
#include <cassert>

#define IMBALANCED_DISTRIBUTION 1

#if IMBALANCED_DISTRIBUTION
#define HOT_EXPERT_ID 0 // index of hot expert
#define HOT_EXPERT_FRACTION 0.9 // fraction of tokens assigned to the hot expert
#endif

#define DEBUG_MOE_KERNEL 0

using namespace std;

static void fill_random(vector<float> &v, float scale = 0.01f)
{
    mt19937 rng(123);
    uniform_real_distribution<float> dist(-scale, scale);
    for (auto &x : v)
        x = dist(rng);
}

static void fill_expert_ids(vector<int> &ids, int E)
{
    mt19937 rng(456);

#if IMBALANCED_DISTRIBUTION
    /* imbalanced token distribution */
    assert(HOT_EXPERT_ID < E);
    vector<double> p(E, (1.0 - HOT_EXPERT_FRACTION) / (double)(E - 1));
    p[HOT_EXPERT_ID] = HOT_EXPERT_FRACTION;
    discrete_distribution<int> dist(p.begin(), p.end());
#else
    /* uniform token distribution */
    uniform_int_distribution<int> dist(0, E - 1);
#endif
    for (auto &x : ids)
        x = dist(rng) % E;
}


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
                    int bid, int tid, int gridDimX, int blockDimX) const 
    {
        // 1. PPNT Grid Setup
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD);
        int logical_grid_size = (n_tbs_in_xcd - 1) * XCD_NUM; 

        // 2. Dimensions
        int out_dim = 3 * a->d; // Q, K, V concatenated
        int block_x = 32; 
        int block_y = 32;
        
        int grid_dim_x = (out_dim + block_x - 1) / block_x;
        int grid_dim_y = (a->T    + block_y - 1) / block_y;
        int total_tiles = grid_dim_x * grid_dim_y;

        // 3. Grid-Stride Loop
#if DEBUG_MOE_KERNEL
        uint64_t loop_start = __builtin_readcyclecounter();
#endif
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
#if DEBUG_MOE_KERNEL
        uint64_t loop_end = __builtin_readcyclecounter();
        if (tid == 0) {
            printf("AttnQKV loop cycles: %lu\n", loop_end - loop_start);
        }
#endif
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
                    int bid, int tid, int gridDimX, int blockDimX) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD);
        int logical_grid_size = (n_tbs_in_xcd - 1) * XCD_NUM; 

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
// DISPATCH
// =============================================================================================

struct DispatchArgs {
    const float* X;            // [T, d]
    const int*   expert_id;    // [T]
    float*       Xexp;         // [E*cap, d]
    int*         pos;          // [T]
    int*         counters;     // [E] (must be zeroed before launch)
    int          T, d, E, cap;
};

__device__ __forceinline__
void moe_dispatch_body(int t, const DispatchArgs* __restrict__ a) {
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

struct DispatchTargetFn {
    __device__ __forceinline__
    void operator()(const DispatchArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int tbid_in_xcd  = (bid / XCD_NUM) % n_tbs_in_xcd;

        // Skip profiling TBs (those do not execute target)
        if (tbid_in_xcd == PPNT_TBID_IN_XCD) return;

        int logical_bid = ppnt::physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD);

        /* Insert target kernel below */
        
        // Token index based on logical block id
        int t = logical_bid * blockDimX + tid;
        moe_dispatch_body(t, a);
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
                    int bid, int tid, int gridDimX, int blockDimX) const 
    {
        // 1. PPNT Logic setup
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD);
        int logical_grid_size = (n_tbs_in_xcd - 1) * XCD_NUM; 

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
                    int bid, int tid, int gridDimX, int blockDimX) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD);
        int logical_grid_size = (n_tbs_in_xcd - 1) * XCD_NUM; 

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
                    int bid, int tid, int gridDimX, int blockDimX) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD);
        int logical_grid_size = (n_tbs_in_xcd - 1) * XCD_NUM; 

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
                    int bid, int tid, int gridDimX, int blockDimX) const 
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD);
        int logical_grid_size = (n_tbs_in_xcd - 1) * XCD_NUM; 

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