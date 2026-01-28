#pragma once

#include <vector>
#include <random>
#include <cassert>

#define IMBALANCED_DISTRIBUTION 1

#if IMBALANCED_DISTRIBUTION
#define HOT_EXPERT_ID 0 // index of hot expert
#define HOT_EXPERT_FRACTION 0.9 // fraction of tokens assigned to the hot expert
#endif

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
// GEMM
// =============================================================================================