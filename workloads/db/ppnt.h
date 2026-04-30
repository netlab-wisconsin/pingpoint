#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <cassert>

#include "main.h"
#include "k1.h"
#include "k2.h"

namespace cg = cooperative_groups;

#define DEBUG_PPNT_LEVEL 0

namespace ppnt {

struct TargetArgsT {};
struct TargetFnT { 
    __device__ __forceinline__
    void operator()(const TargetArgsT*, int, int, int, int, int){}
};

enum class PingKind : uint8_t { Latency=0, Bandwidth=1 };

struct PingSpec {
    uint16_t ping_id;
    PingKind kind;
    uint16_t src_xcd;
    uint16_t dst_hbm;
    size_t iters;
    size_t bpx; // number of "ping" prof blocks per xcd
    // only for k1
    k1::dtype *data;
    k1::dtype *dummy; // to avoid compiler optimization
    // only for k2
    uint64_t *data0; 
    uint64_t *data1; 
    uint64_t *data2; 
    uint64_t *data3;
    float *sink;
    size_t data_bytes; // unused, but keep for later
};

struct PingOut {
    uint64_t *iterClk; 
};

__device__ __forceinline__
int physical_to_logical_bid(int bid, int n_tbs_in_xcd, int ppnt_bpx) {
    // 1. Identify XCD lane and local index
    int xcd_lane    = bid % XCD_NUM;
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd;

    // 2. Compute Logical ID
    // Since the first `ppnt_bpx` blocks are reserved, the logical ID for the target
    // starts at 0 when tbid_in_xcd == ppnt_bpx.
    // (Caller must guarantee tbid_in_xcd >= ppnt_bpx)
    int logical_tbid_local = tbid_in_xcd - ppnt_bpx;

    // 3. Compute Global Linear ID
    // Each XCD contributes (n - ppnt_bpx) blocks to the logical grid.
    int target_tbs_per_xcd = n_tbs_in_xcd - ppnt_bpx;

    return xcd_lane * target_tbs_per_xcd + logical_tbid_local;
}

template <typename TargetFn, typename TargetArgs>
__global__ void fused_kernel(TargetFn target_fn, const TargetArgs* __restrict__ targs,
                             const PingSpec* __restrict__ plan, const size_t n_plan,
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

    if (n_plan == 0) {
        // No profiling plan, directly run target fn
        target_fn(targs, bid, tid, gridDim.x, blockDim.x, 0);
        return;
    }

    for (int i = 0; i < n_plan; i++) {
        // Synchronize the coop grid before each ping/replay (#1)
        grid.sync();

        const PingSpec& spec = plan[i];
#if DEBUG_PPNT_LEVEL >= 1
        if (bid == 0 && tid == 0) {
            printf("spec[%d]: ping_id=%d, kind=%d, src_xcd=%d, dst_hbm=%d, iters=%zu, bpx=%zu\n", \
                i, spec.ping_id, (int)spec.kind, spec.src_xcd, spec.dst_hbm, spec.iters, spec.bpx);
        }
#endif

        if (tbid_in_xcd < spec.bpx) {
            /* Run profiling */

#if DEBUG_PPNT_LEVEL >= 2
            if (tid == 0) {
                printf("[profile] bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", \
                    bid, xcc_id, se_id, cu_id);
            }
#endif

            if (spec.src_xcd != xcc_id) continue; // not my ping

            if (spec.kind == PingKind::Latency) 
            {
                if (tid != 0) continue; // only thread 0 runs latency ping
                k1::k<k1::dtype>(
                    spec.data, spec.dummy, spec.iters,
                    /* Pingout */
                    out[i].iterClk
                );
            } 
            else if (spec.kind == PingKind::Bandwidth) 
            {
                const int k2_chunk_size = CHUNK_SIZE;
                k2::k(
                    spec.data0, spec.data1,
                    spec.data2, spec.data3,
                    spec.sink, (spec.data_bytes / k2_chunk_size),
                    k2_chunk_size, spec.iters,
                    /* Pingout */
                    out[i].iterClk
                );
            } 
            else 
            { 
                // unknown ping kind
#if DEBUG_PPNT_LEVEL >= 2
                if (tid == 0) printf("unknown ping kind\n");
                return;
#endif
            }

        } else {
            /* Run target fn */
            target_fn(targs, bid, tid, gridDim.x, blockDim.x, spec.bpx);
        }

    }


}

void parse_pingouts(PingSpec* d_plan, PingOut* d_out, const size_t n_plan, const size_t blockD_x, const size_t clock) {

    if (n_plan == 0) return; // No profiling plan, skip

    // Check PingOut results
    for (size_t i = 0; i < n_plan; i++) {
        PingSpec p;
        PingOut o;
        gpuErrchk(hipMemcpy(&p, &d_plan[i], sizeof(PingSpec), hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(&o, &d_out[i], sizeof(PingOut), hipMemcpyDeviceToHost));

        // Copy per-iteration clock data to host
        size_t n = p.iters * p.bpx;
        vector<uint64_t> h_iterClk(n);
#if DEBUG_PPNT_LEVEL >= 2
        cout << "[PPNT] Parsing pingout id=" << i << " (kind=" 
             << ((p.kind == ppnt::PingKind::Latency) ? "Latency" : "Bandwidth") << ") "
             << p.iters << " iters " << p.bpx << " bpx"
             << "\n"
             << flush;
#endif
        gpuErrchk(hipMemcpy(h_iterClk.data(), o.iterClk, sizeof(uint64_t) * n, hipMemcpyDeviceToHost));

        // Compute average cycles per iteration
        MeasurementSeries cycles;
        for (size_t it = 0; it < n; it++) {
            cycles.add(h_iterClk[it]);
        }
        
        // 1e3 as clock in MHz
        const double dt_mean = cycles.value(); const double ns_mean = dt_mean / (double)clock * 1e3;
        const double dt_50  = cycles.getPercentile(0.5); const double ns_50  = dt_50  / (double)clock * 1e3;
        const double dt_90  = cycles.getPercentile(0.9); const double ns_90  = dt_90  / (double)clock * 1e3;
        const double dt_99  = cycles.getPercentile(0.99); const double ns_99  = dt_99  / (double)clock * 1e3;

        // Compute bandwidth for Bandwidth pings
        string bw_str = "N/A"; // in GB/s
        if (p.kind == ppnt::PingKind::Bandwidth) {
            const size_t k2_n_datas = 4; // k2 uses 4 data pointers
            size_t bytes_per_block = blockD_x * k2_n_datas * 16;
            size_t iter_bytes = bytes_per_block * p.bpx; // Account for multiple blocks (bpx) running in parallel
            double bw_gbps = ((double)iter_bytes / (ns_mean)) ; // GB/s
            bw_str = to_string(bw_gbps);
        }

        // Report
        cout << "[PPNT] Ping id=" << setw(2) << i << " "
                << "kind=" << ((p.kind == ppnt::PingKind::Latency) ? "Latency" : "Bandwidth") << " "
                << "src_xcd=" << setw(1) << p.src_xcd << " "
                << "dst_hbm=" << setw(1) << p.dst_hbm << " "
                << "iters=" << setw(8) << p.iters << " "
                << "bpx=" << setw(2) << p.bpx << " "
                << fixed << setprecision(1)
                << "mean_ns=" << setw(3) << ns_mean << " "
                << "p50_ns=" << setw(3) << ns_50 << " "
                << "p90_ns=" << setw(3) << ns_90 << " "
                << "p99_ns=" << setw(3) << ns_99 << " "
                << "GBps=" << setw(3) << bw_str << " "
                << "\n" << flush;
    }
}

} // namespace ppnt