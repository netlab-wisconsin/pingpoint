#pragma once

#include "ppnt.h"

namespace ppnt {

// Extended PingOut with per-plan MoE start/end device clocks.
struct PingOut2 {
    uint64_t *iterClk;
    uint64_t *moe_start_clk;  // [1] atomicMin across target blocks
    uint64_t *moe_end_clk;    // [1] atomicMax across target blocks
};

// fused_kernel variant that records MoE start/end clocks per plan.
// Identical to ppnt::fused_kernel except the target-block branch is timed.
template <typename TargetFn, typename TargetArgs>
__global__ void fused_kernel2(TargetFn target_fn, const TargetArgs* __restrict__ targs,
                               const PingSpec* __restrict__ plan, const size_t n_plan,
                               PingOut2* __restrict__ out)
{
    cg::grid_group grid = cg::this_grid();

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    const int tbid_in_xcd  = (bid / XCD_NUM) % n_tbs_in_xcd;

    uint32_t xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    if (n_plan == 0) {
        target_fn(targs, bid, tid, gridDim.x, blockDim.x, 0);
        return;
    }

    for (int i = 0; i < (int)n_plan; i++) {
        grid.sync();
        const PingSpec& spec = plan[i];

        if (tbid_in_xcd < (int)spec.bpx) {
            if (spec.src_xcd != xcc_id) continue;

            if (spec.kind == PingKind::Latency) {
                if (tid != 0) continue;
                k1::k<k1::dtype>(spec.data, spec.dummy, spec.iters, out[i].iterClk);
            } else if (spec.kind == PingKind::Bandwidth) {
                k2::k(spec.data0, spec.data1, spec.data2, spec.data3,
                      spec.sink, (spec.data_bytes / CHUNK_SIZE),
                      CHUNK_SIZE, spec.iters, out[i].iterClk);
            }
        } else {
            const uint64_t clk_s = __builtin_readcyclecounter();
            target_fn(targs, bid, tid, gridDim.x, blockDim.x, spec.bpx);
            const uint64_t clk_e = __builtin_readcyclecounter();
            if (tid == 0) {
                atomicMin((unsigned long long*)out[i].moe_start_clk, (unsigned long long)clk_s);
                atomicMax((unsigned long long*)out[i].moe_end_clk,   (unsigned long long)clk_e);
            }
        }
    }
}

// Reset moe clock buffers to sentinel values before each co-run kernel launch.
// Call this before every fused_kernel2 invocation to get clean per-plan timing.
inline void init_moe_clks(const vector<PingOut2>& h_out, hipStream_t stream) {
    static const uint64_t max_val  = UINT64_MAX;
    static const uint64_t zero_val = 0;
    for (const auto& o : h_out) {
        gpuErrchk(hipMemcpyAsync(o.moe_start_clk, &max_val,  sizeof(uint64_t), hipMemcpyHostToDevice, stream));
        gpuErrchk(hipMemcpyAsync(o.moe_end_clk,   &zero_val, sizeof(uint64_t), hipMemcpyHostToDevice, stream));
    }
}

// Feature 1: log MoE kernel throughput for a solo (no-ping) run.
inline void log_solo_throughput(const char* kernel_name, float elapsed_ms, size_t bytes) {
    const double gbps = (double)bytes / (elapsed_ms * 1e-3) / 1e9;
    printf("[SOLO]  %-20s  %.3f ms  %.2f GB/s\n", kernel_name, elapsed_ms, gbps);
}

// Feature 2: log throughput for the full fused-kernel wall-time in co-run mode.
inline void log_corun_throughput(const char* kernel_name, float elapsed_ms, size_t bytes, const PingSpec* spec = nullptr) {
    const double gbps = (double)bytes / (elapsed_ms * 1e-3) / 1e9;
    if (spec) {
        printf("[CORUN] %-20s  ping_id=%-3d src_xcd=%d dst_hbm=%d bpx=%-2zu  %.3f ms  %.2f GB/s\n",
               kernel_name, spec->ping_id, spec->src_xcd, spec->dst_hbm, spec->bpx, elapsed_ms, gbps);
    } else {
        printf("[CORUN] %-20s  %.3f ms  %.2f GB/s\n", kernel_name, elapsed_ms, gbps);
    }
}

} // namespace ppnt
