// E1: target FFN impact from bounded-lifetime bandwidth pings.
//
// Each measured request has a paired no-ping baseline with the same grid shape. Target blocks
// always occupy local slots [0, TARGET_CUS_PER_XCD); ping blocks occupy the following bpx slots.
// A ping runs only until the last target block completes. Sampling rate k/10 is generated exactly
// and evenly with floor((i + 1) * k / 10) > floor(i * k / 10).

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "../../mem_bench/gpu-clock.cuh"

#include "../moe/main.h"
#include "../moe/k2.h"
#include "overhead_moe.h"

using namespace std;
namespace cg = cooperative_groups;

static_assert(BLOCKDIM_X % WAVE_SIZE == 0);
static_assert(TARGET_CUS_PER_XCD + 16 <= CU_NUM);

static constexpr int WAVES_PER_BLOCK = BLOCKDIM_X / WAVE_SIZE;
static constexpr int N_TARGET_BLOCKS = TARGET_CUS_PER_XCD * XCD_NUM;
static constexpr int BYTES_PER_THREAD_ITER = N_PROBE_DATAS * 16;

struct FfnArgs {
    const float* x;
    const float* w;
    float* out;
    int tokens;
    int experts;
    int cap;
    int d;
    int hidden;
};

__device__ __forceinline__
int target_logical_bid(int bid, int local_slot) {
    return (bid % XCD_NUM) * TARGET_CUS_PER_XCD + local_slot;
}

__device__ __forceinline__
void run_prefill_ffn(const FfnArgs* __restrict__ a, int logical_bid, int tid) {
    constexpr int tile_x = 32;
    constexpr int tile_y = 32;
    constexpr int rows_per_thread = PREFILL_ROWS_PER_THREAD;
    constexpr int tokens_per_tile = tile_y * rows_per_thread;
    const int grid_x = (a->hidden + tile_x - 1) / tile_x;
    const int grid_y = (a->cap + tokens_per_tile - 1) / tokens_per_tile;
    const int total_tiles = grid_x * grid_y * a->experts;

    for (int tile = logical_bid; tile < total_tiles; tile += N_TARGET_BLOCKS) {
        const int expert_slice = grid_x * grid_y;
        const int expert = tile / expert_slice;
        const int rem = tile % expert_slice;
        const int by = rem / grid_x;
        const int bx = rem % grid_x;
        const int token = by * tokens_per_tile + (tid / tile_x) * rows_per_thread;
        const int col = bx * tile_x + tid % tile_x;
        if (token + rows_per_thread - 1 >= a->cap || col >= a->hidden) continue;

        const float* x0 = a->x + (size_t)(expert * a->cap + token + 0) * a->d;
        const float* x1 = a->x + (size_t)(expert * a->cap + token + 1) * a->d;
        const float* x2 = a->x + (size_t)(expert * a->cap + token + 2) * a->d;
        const float* x3 = a->x + (size_t)(expert * a->cap + token + 3) * a->d;
        const float* w = a->w + (size_t)expert * a->d * a->hidden;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int k = 0; k < a->d; k++) {
            const float weight = w[(size_t)k * a->hidden + col];
            acc0 += x0[k] * weight;
            acc1 += x1[k] * weight;
            acc2 += x2[k] * weight;
            acc3 += x3[k] * weight;
        }
        a->out[(size_t)(expert * a->cap + token + 0) * a->hidden + col] = acc0;
        a->out[(size_t)(expert * a->cap + token + 1) * a->hidden + col] = acc1;
        a->out[(size_t)(expert * a->cap + token + 2) * a->hidden + col] = acc2;
        a->out[(size_t)(expert * a->cap + token + 3) * a->hidden + col] = acc3;
    }
}

__device__ __forceinline__
void run_decode_ffn(const FfnArgs* __restrict__ a, int logical_bid, int tid) {
    const size_t total_outputs = (size_t)a->experts * a->hidden;
    const size_t total_threads = (size_t)N_TARGET_BLOCKS * BLOCKDIM_X;
    const size_t start = (size_t)logical_bid * BLOCKDIM_X + tid;

    for (size_t oid = start; oid < total_outputs; oid += total_threads) {
        const int expert = (int)(oid / a->hidden);
        const int col = (int)(oid % a->hidden);
        const float* x = a->x + (size_t)expert * a->d;
        const float* w = a->w + (size_t)expert * a->d * a->hidden;
        float acc = 0.0f;
        for (int k = 0; k < a->d; k++)
            acc += x[k] * w[(size_t)k * a->hidden + col];
        a->out[(size_t)expert * a->hidden + col] = acc;
    }
}

template <bool Decode>
__global__ void overhead_kernel(
    const FfnArgs* __restrict__ ffn,
    uint64_t* p0, uint64_t* p1, uint64_t* p2, uint64_t* p3, size_t n_probe_chunks,
    int bpx, int ping_enabled,
    int* __restrict__ target_done, int* __restrict__ stop_ping,
    uint64_t* __restrict__ target_max_cycles,
    uint64_t* __restrict__ ping_wave_iters,
    float* __restrict__ sink)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const bool wave_leader = (tid % WAVE_SIZE) == 0;
    const int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    const int local_slot = (bid / XCD_NUM) % n_tbs_in_xcd;

    uint32_t xcc_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    grid.sync();

    if (local_slot < TARGET_CUS_PER_XCD) {
        const uint64_t start = __builtin_readcyclecounter();
        const int logical_bid = target_logical_bid(bid, local_slot);
        if constexpr (Decode)
            run_decode_ffn(ffn, logical_bid, tid);
        else
            run_prefill_ffn(ffn, logical_bid, tid);
        __syncthreads();
        const uint64_t end = __builtin_readcyclecounter();

        if (tid == 0) {
            atomicMax((unsigned long long*)target_max_cycles,
                      (unsigned long long)(end - start));
            const int completed = atomicAdd(target_done, 1);
            if (completed == N_TARGET_BLOCKS - 1)
                atomicExch(stop_ping, 1);
        }
        return;
    }

    const int ping_slot = local_slot - TARGET_CUS_PER_XCD;
    if (!ping_enabled || ping_slot >= bpx || xcc_id != SRC_XCD) return;

    const int chunk_elems = CHUNK_SIZE / 16;
    const int groups = BLOCKDIM_X / chunk_elems;
    const int group = tid / chunk_elems;
    const int offset = tid % chunk_elems;
    float sink0 = 0.0f, sink1 = 0.0f, sink2 = 0.0f, sink3 = 0.0f;
    uint64_t iter = 0;

    for (;;) {
        if ((iter % STOP_CHECK_ITERS) == 0 && *(volatile int*)stop_ping) break;
        const size_t ci = ((iter * (size_t)bpx + ping_slot) * groups + group) % n_probe_chunks;
        float4* a0 = reinterpret_cast<float4*>(p0[ci]);
        float4* a1 = reinterpret_cast<float4*>(p1[ci]);
        float4* a2 = reinterpret_cast<float4*>(p2[ci]);
        float4* a3 = reinterpret_cast<float4*>(p3[ci]);
        float4 r0, r1, r2, r3;
        asm volatile(
            "flat_load_dwordx4 %[R0], %[A0]\n\t"
            "flat_load_dwordx4 %[R1], %[A1]\n\t"
            "flat_load_dwordx4 %[R2], %[A2]\n\t"
            "flat_load_dwordx4 %[R3], %[A3]\n\t"
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            : [R0]"=&v"(r0), [R1]"=&v"(r1), [R2]"=&v"(r2), [R3]"=&v"(r3)
            : [A0]"v"(&a0[offset]), [A1]"v"(&a1[offset]),
              [A2]"v"(&a2[offset]), [A3]"v"(&a3[offset])
            : "memory");
        sink0 += r0.x + r1.x + r2.x + r3.x;
        sink1 += r0.y + r1.y + r2.y + r3.y;
        sink2 += r0.z + r1.z + r2.z + r3.z;
        sink3 += r0.w + r1.w + r2.w + r3.w;
        iter++;
    }

    if (wave_leader) {
        ping_wave_iters[ping_slot * WAVES_PER_BLOCK + wave] = iter;
        sink[ping_slot * WAVES_PER_BLOCK + wave] = sink0 + sink1 + sink2 + sink3;
    }
}

struct Stats {
    vector<double> values;

    void add(double v) { values.push_back(v); }
    double mean() const {
        if (values.empty()) return NAN;
        return accumulate(values.begin(), values.end(), 0.0) / values.size();
    }
    double percentile(double p) const {
        if (values.empty()) return NAN;
        vector<double> sorted = values;
        sort(sorted.begin(), sorted.end());
        const size_t idx = (size_t)ceil(p * sorted.size()) - 1;
        return sorted[min(idx, sorted.size() - 1)];
    }
};

struct PassResult {
    double target_ns;
    double wall_ns;
    double ping_gbps;
    uint64_t ping_bytes;
};

struct ConditionStats {
    Stats baseline_target, baseline_wall;
    Stats workload_target, workload_wall;
    Stats pinged_target, pinged_wall;
    Stats unpinged_target, unpinged_wall;
    uint64_t ping_bytes = 0;
    double ping_time_ns = 0.0;
    int n_ping = 0;
};

static double slowdown_pct(double value, double baseline) {
    return (value / baseline - 1.0) * 100.0;
}

int main(int argc, char** argv) {
    const string requested_regime = argc > 1 ? argv[1] : "both";
    const int n_measured = argc > 2 ? atoi(argv[2]) : DEFAULT_MEASURED_REQUESTS;
    if (requested_regime != "prefill" && requested_regime != "decode" &&
        requested_regime != "both") {
        fprintf(stderr, "usage: %s [prefill|decode|both] [measured_requests_multiple_of_10]\n",
                argv[0]);
        return 1;
    }
    if (n_measured <= 0 || n_measured % 10 != 0) {
        fprintf(stderr, "measured_requests must be a positive multiple of 10\n");
        return 1;
    }
    const int only_bpx = getenv("OVERHEAD_BPX") ? atoi(getenv("OVERHEAD_BPX")) : 0;
    const int only_rate_tenths =
        getenv("OVERHEAD_RATE_TENTHS") ? atoi(getenv("OVERHEAD_RATE_TENTHS")) : 0;

    const unsigned int clock_mhz = getGPUClock();

    const long long probe_data_size = (long long)PROBE_N_PAGES * PAGE_SIZE;
    const size_t n_chunks = probe_data_size / CHUNK_SIZE;
    vector<char*> d_probe_data(N_PROBE_DATAS);
    for (int i = 0; i < N_PROBE_DATAS; i++) {
        gpuErrchk(hipMalloc((void**)&d_probe_data[i], probe_data_size + 0x1000));
        d_probe_data[i] =
            (char*)(((uintptr_t)d_probe_data[i] & ~(uintptr_t)0x0fff) + 0x1000);
    }

    vector<vector<int>> h_home(N_PROBE_DATAS, vector<int>(n_chunks));
    vector<vector<size_t>> h_xcd_chunks_size(
        N_PROBE_DATAS, vector<size_t>(XCD_NUM, 0));
    if (k2::home_identification(d_probe_data, probe_data_size, n_chunks, N_PROBE_DATAS,
                                h_home, h_xcd_chunks_size) == -1)
        return -1;

    vector<uint64_t*> d_probe_chunks(N_PROBE_DATAS);
    vector<size_t> home_chunks(N_PROBE_DATAS);
    for (int i = 0; i < N_PROBE_DATAS; i++) {
        vector<uint64_t> selected;
        selected.reserve(h_xcd_chunks_size[i][DST_HBM]);
        for (size_t chunk = 0; chunk < n_chunks; chunk++)
            if (h_home[i][chunk] == DST_HBM)
                selected.push_back((uint64_t)d_probe_data[i] + chunk * CHUNK_SIZE);
        home_chunks[i] = selected.size();
        if (selected.size() < PROBE_CHUNKS_PER_DATA) {
            fprintf(stderr, "data %d has only %zu home chunks; need at least %d\n",
                    i, selected.size(), PROBE_CHUNKS_PER_DATA);
            return 1;
        }
        gpuErrchk(hipMalloc((void**)&d_probe_chunks[i],
                            sizeof(uint64_t) * PROBE_CHUNKS_PER_DATA));
        gpuErrchk(hipMemcpy(d_probe_chunks[i], selected.data(),
                            sizeof(uint64_t) * PROBE_CHUNKS_PER_DATA,
                            hipMemcpyHostToDevice));
    }

    uint64_t* p0 = d_probe_chunks[0];
    uint64_t* p1 = d_probe_chunks[1];
    uint64_t* p2 = d_probe_chunks[2];
    uint64_t* p3 = d_probe_chunks[3];
    const size_t n_probe_chunks = PROBE_CHUNKS_PER_DATA;
    const size_t min_home_chunks = *min_element(home_chunks.begin(), home_chunks.end());

    const int d = D_MODEL;
    const int hidden = HIDDEN_MULT * d;
    const int max_cap = PREFILL_T / N_EXPERT;
    const size_t x_elems = (size_t)N_EXPERT * max_cap * d;
    const size_t w_elems = (size_t)N_EXPERT * d * hidden;
    const size_t out_elems = (size_t)N_EXPERT * max_cap * hidden;
    float *d_x = nullptr, *d_w = nullptr, *d_out = nullptr;
    gpuErrchk(hipMalloc(&d_x, sizeof(float) * x_elems));
    gpuErrchk(hipMalloc(&d_w, sizeof(float) * w_elems));
    gpuErrchk(hipMalloc(&d_out, sizeof(float) * out_elems));
    gpuErrchk(hipMemset(d_x, 1, sizeof(float) * x_elems));
    gpuErrchk(hipMemset(d_w, 1, sizeof(float) * w_elems));

    FfnArgs* d_ffn = nullptr;
    int *d_target_done = nullptr, *d_stop_ping = nullptr;
    uint64_t *d_target_max_cycles = nullptr, *d_ping_wave_iters = nullptr;
    float* d_sink = nullptr;
    const size_t ping_count_elems = 16 * WAVES_PER_BLOCK;
    gpuErrchk(hipMalloc(&d_ffn, sizeof(FfnArgs)));
    gpuErrchk(hipMalloc(&d_target_done, sizeof(int)));
    gpuErrchk(hipMalloc(&d_stop_ping, sizeof(int)));
    gpuErrchk(hipMalloc(&d_target_max_cycles, sizeof(uint64_t)));
    gpuErrchk(hipMalloc(&d_ping_wave_iters, sizeof(uint64_t) * ping_count_elems));
    gpuErrchk(hipMalloc(&d_sink, sizeof(float) * ping_count_elems));

    hipStream_t stream;
    hipEvent_t ev_start, ev_end;
    gpuErrchk(hipStreamCreate(&stream));
    gpuErrchk(hipEventCreate(&ev_start));
    gpuErrchk(hipEventCreate(&ev_end));
    vector<uint64_t> h_ping_wave_iters(ping_count_elems);

    auto run_regime = [&](const string& regime) {
        const bool decode = regime == "decode";
        const int tokens = decode ? DECODE_T : PREFILL_T;
        const int cap = tokens / N_EXPERT;
        FfnArgs h_ffn{d_x, d_w, d_out, tokens, N_EXPERT, cap, d, hidden};
        gpuErrchk(hipMemcpy(d_ffn, &h_ffn, sizeof(FfnArgs), hipMemcpyHostToDevice));

        auto run_pass = [&](int bpx, bool ping_enabled) -> PassResult {
            gpuErrchk(hipMemsetAsync(d_target_done, 0, sizeof(int), stream));
            gpuErrchk(hipMemsetAsync(d_stop_ping, 0, sizeof(int), stream));
            gpuErrchk(hipMemsetAsync(d_target_max_cycles, 0, sizeof(uint64_t), stream));
            gpuErrchk(hipMemsetAsync(d_ping_wave_iters, 0,
                                     sizeof(uint64_t) * ping_count_elems, stream));

            int bpx_arg = bpx;
            int ping_arg = ping_enabled ? 1 : 0;
            void* args[] = {
                &d_ffn, &p0, &p1, &p2, &p3, (void*)&n_probe_chunks,
                &bpx_arg, &ping_arg, &d_target_done, &d_stop_ping,
                &d_target_max_cycles, &d_ping_wave_iters, &d_sink
            };
            const int grid_blocks = (TARGET_CUS_PER_XCD + bpx) * XCD_NUM;
            gpuErrchk(hipEventRecord(ev_start, stream));
            if (decode) {
                gpuErrchk(hipLaunchCooperativeKernel((void*)overhead_kernel<true>,
                                                     dim3(grid_blocks), dim3(BLOCKDIM_X),
                                                     args, 0, stream));
            } else {
                gpuErrchk(hipLaunchCooperativeKernel((void*)overhead_kernel<false>,
                                                     dim3(grid_blocks), dim3(BLOCKDIM_X),
                                                     args, 0, stream));
            }
            gpuErrchk(hipEventRecord(ev_end, stream));
            gpuErrchk(hipStreamSynchronize(stream));

            float wall_ms = 0.0f;
            uint64_t target_cycles = 0;
            gpuErrchk(hipEventElapsedTime(&wall_ms, ev_start, ev_end));
            gpuErrchk(hipMemcpy(&target_cycles, d_target_max_cycles, sizeof(uint64_t),
                                hipMemcpyDeviceToHost));
            gpuErrchk(hipMemcpy(h_ping_wave_iters.data(), d_ping_wave_iters,
                                sizeof(uint64_t) * ping_count_elems, hipMemcpyDeviceToHost));

            uint64_t total_wave_iters = 0;
            for (int slot = 0; slot < bpx; slot++)
                for (int wave = 0; wave < WAVES_PER_BLOCK; wave++)
                    total_wave_iters += h_ping_wave_iters[slot * WAVES_PER_BLOCK + wave];
            const uint64_t ping_bytes =
                total_wave_iters * (uint64_t)WAVE_SIZE * BYTES_PER_THREAD_ITER;
            const double target_ns = (double)target_cycles / clock_mhz * 1e3;
            const double ping_gbps = target_ns > 0.0 ? (double)ping_bytes / target_ns : 0.0;
            return {target_ns, (double)wall_ms * 1e6, ping_gbps, ping_bytes};
        };

        printf("[overhead-moe] regime=%s tokens=%d experts=%d d=%d hidden=%d target_cus_per_xcd=%d "
               "src_xcd=%d dst_hbm=%d measured_requests=%d warmup=%d n_probe_chunks=%zu "
               "min_home_chunks=%zu clock_mhz=%u\n",
               regime.c_str(), tokens, N_EXPERT, d, hidden, TARGET_CUS_PER_XCD, SRC_XCD,
               DST_HBM, n_measured, WARMUP_PASSES, n_probe_chunks, min_home_chunks, clock_mhz);
        printf("sample,regime,bpx,requested_rate,request_idx,ping,baseline_target_ns,"
               "target_ns,baseline_wall_ns,wall_ns,ping_gbps,ping_bytes\n");
        printf("summary,regime,bpx,requested_rate,actual_rate,n,n_ping,n_no_ping,"
               "baseline_target_ns,pinged_target_ns,pinged_target_slowdown_pct,"
               "unpinged_target_ns,workload_target_ns,workload_target_slowdown_pct,"
               "workload_p99_target_ns,baseline_wall_ns,pinged_wall_ns,workload_wall_ns,"
               "workload_wall_slowdown_pct,workload_p99_wall_ns,ping_gbps,"
               "ping_bytes_per_pinged_request\n");

        for (int rate_tenths : RATE_TENTHS) {
            if (only_rate_tenths != 0 && rate_tenths != only_rate_tenths) continue;
            for (int bpx : BW_ACTIVE_CUS) {
                if (only_bpx != 0 && bpx != only_bpx) continue;
                for (int w = 0; w < WARMUP_PASSES; w++) {
                    run_pass(bpx, false);
                    run_pass(bpx, true);
                }

                ConditionStats stats;
                for (int i = 0; i < n_measured; i++) {
                    const bool ping =
                        ((i + 1) * rate_tenths / 10) > (i * rate_tenths / 10);
                    PassResult baseline, workload;
                    if ((i & 1) == 0) {
                        baseline = run_pass(bpx, false);
                        workload = run_pass(bpx, ping);
                    } else {
                        workload = run_pass(bpx, ping);
                        baseline = run_pass(bpx, false);
                    }

                    stats.baseline_target.add(baseline.target_ns);
                    stats.baseline_wall.add(baseline.wall_ns);
                    stats.workload_target.add(workload.target_ns);
                    stats.workload_wall.add(workload.wall_ns);
                    if (ping) {
                        stats.pinged_target.add(workload.target_ns);
                        stats.pinged_wall.add(workload.wall_ns);
                        stats.ping_bytes += workload.ping_bytes;
                        stats.ping_time_ns += workload.target_ns;
                        stats.n_ping++;
                    } else {
                        stats.unpinged_target.add(workload.target_ns);
                        stats.unpinged_wall.add(workload.wall_ns);
                    }
                    printf("sample,%s,%d,%.1f,%d,%d,%.1f,%.1f,%.1f,%.1f,%.2f,%llu\n",
                           regime.c_str(), bpx, rate_tenths / 10.0, i, ping ? 1 : 0,
                           baseline.target_ns, workload.target_ns, baseline.wall_ns,
                           workload.wall_ns, workload.ping_gbps,
                           (unsigned long long)workload.ping_bytes);
                }

                const double baseline_target = stats.baseline_target.mean();
                const double workload_target = stats.workload_target.mean();
                const double baseline_wall = stats.baseline_wall.mean();
                const double workload_wall = stats.workload_wall.mean();
                const double ping_gbps = stats.ping_time_ns > 0.0
                    ? (double)stats.ping_bytes / stats.ping_time_ns : 0.0;
                const double ping_bytes_per_request = stats.n_ping > 0
                    ? (double)stats.ping_bytes / stats.n_ping : 0.0;
                printf("summary,%s,%d,%.1f,%.3f,%d,%d,%d,%.1f,%.1f,%.2f,%.1f,%.1f,"
                       "%.2f,%.1f,%.1f,%.1f,%.1f,%.2f,%.1f,%.2f,%.0f\n",
                       regime.c_str(), bpx, rate_tenths / 10.0,
                       (double)stats.n_ping / n_measured, n_measured, stats.n_ping,
                       n_measured - stats.n_ping, baseline_target, stats.pinged_target.mean(),
                       slowdown_pct(stats.pinged_target.mean(), baseline_target),
                       stats.unpinged_target.mean(), workload_target,
                       slowdown_pct(workload_target, baseline_target),
                       stats.workload_target.percentile(0.99), baseline_wall,
                       stats.pinged_wall.mean(), workload_wall,
                       slowdown_pct(workload_wall, baseline_wall),
                       stats.workload_wall.percentile(0.99), ping_gbps,
                       ping_bytes_per_request);
                fflush(stdout);
            }
        }
    };

    if (requested_regime == "prefill" || requested_regime == "both")
        run_regime("prefill");
    if (requested_regime == "decode" || requested_regime == "both")
        run_regime("decode");
    return 0;
}
