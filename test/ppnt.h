#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "main.h"
#include "k1.h"
#include "k2.h"

namespace cg = cooperative_groups;

#define PPNT_TBID_IN_XCD 0 // ppnt thread block id within xcd
#define DEBUG_PPNT 1

namespace ppnt {

enum class PingKind : uint8_t { Latency=0, Bandwidth=1 };

struct PingSpec {
    uint16_t ping_id;
    PingKind kind;
    uint16_t src_xcd;
    uint16_t dst_hbm;
    size_t iters;
    size_t data_bytes;
    // only data0 is used for k1 (latency ping)
    // for k2 (bandwidth ping), we use data0 ~ data3
    // cast to int64_t for k1
    uint64_t *data0; 
    uint64_t *data1; 
    uint64_t *data2; 
    uint64_t *data3;
    // only for k2
    float *sink;
};

struct PingOut {
    uint16_t ping_id;
    PingKind kind;
    uint16_t src_xcd;
    uint16_t dst_hbm;
    size_t iters;
    uint64_t *iterClk; 
};

__device__ __forceinline__
int physical_to_logical_bid_skip_one(int bid, int n_tbs_in_xcd, int ppnt_tbid_in_xcd) {
    // bid layout assumed interleaved by XCD: bid % XCD_NUM gives "xcd lane"
    // tbid_in_xcd is computed as (bid / XCD_NUM) % n_tbs_in_xcd in your code.

    int xcd_lane   = bid % XCD_NUM;
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd;

    // If this is the profiling TB, caller should not request a logical id.
    // Compute how many target TBs are before this TB within the same XCD lane:
    int before = tbid_in_xcd;
    if (tbid_in_xcd > ppnt_tbid_in_xcd) before -= 1; // skip the reserved profiling slot

    int target_tbs_per_xcd = n_tbs_in_xcd - 1;
    return xcd_lane * target_tbs_per_xcd + before;
}

template <typename TargetFn, typename TargetArgs>
__global__ void fused_kernel(TargetFn target_fn, const TargetArgs* __restrict__ targs,
                             const PingSpec* __restrict__ plan, const size_t N_plan,
                             PingOut* __restrict__ out) 
{
    cg::grid_group grid = cg::this_grid();

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    int n_tbs_in_xcd = (gridDim.x / XCD_NUM); // number of thread blocks in each xcd
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd; // thread block id within xcd

    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    if (tbid_in_xcd == PPNT_TBID_IN_XCD) { // TODO: bw will require multi tbs for profile
        /* Run profiling */

#if DEBUG_PPNT
        if (tid == 0) {
            printf("[profile] bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", \
                bid, xcc_id, se_id, cu_id);
        }
#endif

        for (int i = 0; i < N_plan; i++) {
            // Synchronize the coop grid before each ping/replay (#1)
            grid.sync();

            const PingSpec& spec = plan[i];
#if DEBUG_PPNT
            if (bid == 0 && tid == 0) {
                printf("spec[%d]: ping_id=%d, kind=%d, src_xcd=%d, dst_hbm=%d, iters=%zu, data_bytes=%zu, data0=%p\n", \
                    i, spec.ping_id, (int)spec.kind, spec.src_xcd, spec.dst_hbm, spec.iters, spec.data_bytes, spec.data0);
            }
#endif

            if (spec.src_xcd != xcc_id) continue; // not my ping
            if (spec.kind == PingKind::Latency) {
                if (tid != 0) continue; // only thread 0 runs latency ping
#if DEBUG_PPNT
                printf("(bid:%d,tid:%d) running k1 ping (id: %d)\n", bid, tid, spec.ping_id);
#endif
                k1::k<k1::dtype>(
                    (k1::dtype*)spec.data0, nullptr, spec.iters,
                    /* Pingout */
                    out[i].iterClk
                );
                // TODO: write results
            } else if (spec.kind == PingKind::Bandwidth) {
#if DEBUG_PPNT
                if (tid == 0) printf("(bid:%d) running k1 ping (id: %d)\n", bid, spec.ping_id);
#endif
                const int k2_chunk_size = CHUNK_SIZE;
                k2::k(
                    (uint64_t*)spec.data0, (uint64_t*)spec.data1,
                    (uint64_t*)spec.data2, (uint64_t*)spec.data3,
                    spec.sink, (spec.data_bytes / k2_chunk_size),
                    k2_chunk_size, spec.iters,
                    /* Pingout */
                    out[i].iterClk
                );
                // TODO: write results
            } else { 
                // unknown ping kind
#if DEBUG_PPNT
                if (tid == 0) printf("(bid:%d) unknown ping kind %d (id: %d)\n", bid, (int)spec.kind, spec.ping_id);
                return;
#endif
            }
            
        }

    } else {
        /* Run target fn */
        
#if DEBUG_PPNT
        if (tid == 0) {
            printf("[target] bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", \
                bid, xcc_id, se_id, cu_id);
        }
#endif

#if 0
        target_fn(targs, bid, tid, gridDim.x, blockDim.x);
#else
        // Target kernel replay N_plan times for profiling...
        for (int i = 0; i < N_plan; i++) {
            // Synchronize the coop grid before each ping/replay (#1)
            grid.sync();
            
            target_fn(targs, bid, tid, gridDim.x, blockDim.x);
            // target_fn(targs, 
            //           physical_to_logical_bid_skip_one(bid, n_tbs_in_xcd, PPNT_TBID_IN_XCD), 
            //           tid, 
            //           gridDim.x - XCD_NUM, // adjusted gridDim.x
            //           blockDim.x);

        }
#endif
    }
}

} // namespace ppnt