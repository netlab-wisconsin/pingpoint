// bw-ping-fidelity (E2)
//
// The estimator sees only path capacity, solo-ping calibration, and concurrent ping bandwidth.
// Target counters and target-only passes are evaluation-only ground truth.

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "../../mem_bench/gpu-clock.cuh"

#include "main.h"
#include "k2.h"
#include "bw_fidelity.h"

using namespace std;
namespace cg = cooperative_groups;

static_assert(BLOCKDIM_X % WAVE_SIZE == 0);
static constexpr int WAVES_PER_BLOCK = BLOCKDIM_X / WAVE_SIZE;
static constexpr int BYTES_PER_THREAD_ITER = 4 * 16;
static constexpr int MAX_PROBE_CUS = CU_NUM;

__global__ void bw_fidelity_kernel(
    uint64_t* p0, uint64_t* p1, uint64_t* p2, uint64_t* p3, size_t n_probe_chunks,
    uint64_t* t0a, uint64_t* t1, uint64_t* t2, uint64_t* t3, size_t n_target_chunks,
    int n_target_slots, int bpx, int target_enabled, int probe_enabled, int target_throttle,
    int W, uint64_t window_cycles, const int* __restrict__ thr_cyc,
    uint64_t* __restrict__ probe_cnt,
    uint64_t* __restrict__ target_cnt,
    float* __restrict__ sink)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int wave = tid / WAVE_SIZE;
    const bool wave_leader = (tid % WAVE_SIZE) == 0;
    const int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    const int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd;

    uint32_t xcc_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    grid.sync();
    const uint64_t t0 = __builtin_readcyclecounter();
    if (xcc_id != SRC_XCD) return;

    const bool is_target = tbid_in_xcd < n_target_slots;
    const int rlocal = is_target ? tbid_in_xcd : tbid_in_xcd - n_target_slots;
    if ((!is_target && rlocal >= bpx) ||
        (is_target && !target_enabled) || (!is_target && !probe_enabled)) {
        while (__builtin_readcyclecounter() - t0 < (uint64_t)W * window_cycles) { }
        return;
    }

    uint64_t *c0, *c1, *c2, *c3;
    size_t nch;
    if (is_target) {
        c0 = t0a; c1 = t1; c2 = t2; c3 = t3; nch = n_target_chunks;
    } else {
        c0 = p0; c1 = p1; c2 = p2; c3 = p3; nch = n_probe_chunks;
    }
    uint64_t* outcnt = is_target ? target_cnt : probe_cnt;

    const int chunk_elems = CHUNK_SIZE / 16;
    const int groups = BLOCKDIM_X / chunk_elems;
    const int g = tid / chunk_elems;
    const int off = tid % chunk_elems;

    float sink0 = 0.f, sink1 = 0.f, sink2 = 0.f, sink3 = 0.f;
    int cur_w = 0;
    int thr = is_target && target_throttle ? thr_cyc[0] : 0;
    uint64_t cnt = 0, iter = 0;

    for (;;) {
        const uint64_t now = __builtin_readcyclecounter();
        const int w = (int)((now - t0) / window_cycles);
        if (w >= W) break;
        if (w != cur_w) {
            if (wave_leader)
                outcnt[(rlocal * W + cur_w) * WAVES_PER_BLOCK + wave] = cnt;
            cur_w = w;
            cnt = 0;
            thr = is_target && target_throttle ? thr_cyc[w] : 0;
        }

        const size_t base = (iter * (size_t)n_tbs_in_xcd + tbid_in_xcd) * groups + g;
        const size_t ci = base % nch;
        float4* a0 = reinterpret_cast<float4*>(c0[ci]);
        float4* a1 = reinterpret_cast<float4*>(c1[ci]);
        float4* a2 = reinterpret_cast<float4*>(c2[ci]);
        float4* a3 = reinterpret_cast<float4*>(c3[ci]);
        float4 r0, r1, r2, r3;
        asm volatile(
            "flat_load_dwordx4 %[R0], %[A0]\n\t"
            "flat_load_dwordx4 %[R1], %[A1]\n\t"
            "flat_load_dwordx4 %[R2], %[A2]\n\t"
            "flat_load_dwordx4 %[R3], %[A3]\n\t"
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            : [R0]"=&v"(r0), [R1]"=&v"(r1), [R2]"=&v"(r2), [R3]"=&v"(r3)
            : [A0]"v"(&a0[off]), [A1]"v"(&a1[off]), [A2]"v"(&a2[off]), [A3]"v"(&a3[off])
            : "memory");
        sink0 += r0.x + r1.x + r2.x + r3.x;
        sink1 += r0.y + r1.y + r2.y + r3.y;
        sink2 += r0.z + r1.z + r2.z + r3.z;
        sink3 += r0.w + r1.w + r2.w + r3.w;
        cnt++;
        iter++;

        if (thr > 0) {
            const uint64_t st = __builtin_readcyclecounter();
            while (__builtin_readcyclecounter() - st < (uint64_t)thr) { }
        }
    }
    if (wave_leader) {
        outcnt[(rlocal * W + cur_w) * WAVES_PER_BLOCK + wave] = cnt;
        sink[bid * WAVES_PER_BLOCK + wave] = sink0 + sink1 + sink2 + sink3;
    }
}

static void hbm_chunks(const vector<uint64_t*>& d_chunks_per_hbm,
                       const vector<vector<size_t>>& h_offsets,
                       const vector<vector<size_t>>& h_xcd_sz,
                       int i, uint64_t** ptr_out, size_t* cnt_out) {
    *ptr_out = d_chunks_per_hbm[i] + h_offsets[i][DST_HBM];
    *cnt_out = h_xcd_sz[i][DST_HBM];
}

static double median(vector<double> values) {
    if (values.empty()) return 0.0;
    const size_t mid = values.size() / 2;
    nth_element(values.begin(), values.begin() + mid, values.end());
    double result = values[mid];
    if (values.size() % 2 == 0) {
        nth_element(values.begin(), values.begin() + mid - 1, values.end());
        result = 0.5 * (result + values[mid - 1]);
    }
    return result;
}

static vector<int> make_target_schedule() {
    vector<int> schedule(N_WINDOWS);
    for (int w = 0; w < N_WINDOWS; w++) {
        const int level = w * N_LEVELS / N_WINDOWS;
        schedule[w] = THR_MAX_CYCLES * (N_LEVELS - 1 - level) / (N_LEVELS - 1);
    }
#if TARGET_LOAD_PATTERN == LOAD_RANDOM
    mt19937 rng(RANDOM_SEED);
    shuffle(schedule.begin(), schedule.end(), rng);
#elif TARGET_LOAD_PATTERN != LOAD_STAIRCASE
#error "Unknown TARGET_LOAD_PATTERN"
#endif
    return schedule;
}

static const char* target_pattern_name() {
#if TARGET_LOAD_PATTERN == LOAD_RANDOM
    return "random";
#else
    return "staircase";
#endif
}

struct PassResult {
    vector<double> probe;
    vector<double> target;
};

struct ErrorStats {
    double abs_sum = 0.0;
    double sq_sum = 0.0;
    double rel_sum = 0.0;
    double bias_sum = 0.0;
    int n = 0;

    void add(double estimate, double truth) {
        const double error = estimate - truth;
        abs_sum += fabs(error);
        sq_sum += error * error;
        if (truth > 1e-9) rel_sum += fabs(error) / truth * 100.0;
        bias_sum += error;
        n++;
    }
};

int main() {
    const unsigned int clock = getGPUClock();
    const uint64_t window_cycles = (uint64_t)WINDOW_US * (uint64_t)clock;
    const double window_sec = (double)WINDOW_US * 1e-6;
    const double bytes_per_wave_iter = (double)WAVE_SIZE * BYTES_PER_THREAD_ITER;

    const int n_datas = N_DATAS;
    const long long data_size = (long long)K2_N_PAGES * PAGE_SIZE;
    const int chunk_size = CHUNK_SIZE;
    const size_t n_chunks = data_size / chunk_size;

    vector<char*> d_data(n_datas);
    for (int i = 0; i < n_datas; i++) {
        gpuErrchk(hipMalloc((void**)&d_data[i], data_size + 0x1000));
        d_data[i] = (char*)(((uintptr_t)d_data[i] & ~(uintptr_t)0x0fff) + 0x1000);
    }
    vector<vector<int>> h_home(n_datas, vector<int>(n_chunks));
    vector<vector<size_t>> h_xcd_sz(n_datas, vector<size_t>(XCD_NUM, 0));
    if (k2::home_identification(d_data, data_size, n_chunks, n_datas, h_home, h_xcd_sz) == -1)
        return 1;

    vector<uint64_t*> d_chunks_per_hbm(n_datas);
    vector<vector<size_t>> h_offsets(n_datas, vector<size_t>(XCD_NUM));
    for (int i = 0; i < n_datas; i++) {
        vector<vector<uint64_t>> xcd_chunks(XCD_NUM);
        for (size_t k = 0; k < n_chunks; k++)
            xcd_chunks[h_home[i][k]].push_back((uint64_t)d_data[i] + k * chunk_size);
        gpuErrchk(hipMalloc((void**)&d_chunks_per_hbm[i], sizeof(uint64_t) * n_chunks));
        size_t off = 0;
        for (int x = 0; x < XCD_NUM; x++) {
            gpuErrchk(hipMemcpy(&d_chunks_per_hbm[i][off], xcd_chunks[x].data(),
                                sizeof(uint64_t) * xcd_chunks[x].size(), hipMemcpyHostToDevice));
            h_offsets[i][x] = off;
            off += xcd_chunks[x].size();
        }
    }

    uint64_t *p0, *p1, *p2, *p3, *t0a, *t1, *t2, *t3;
    size_t np[4], nt[4];
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 0, &p0, &np[0]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 1, &p1, &np[1]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 2, &p2, &np[2]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 3, &p3, &np[3]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 4, &t0a, &nt[0]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 5, &t1, &nt[1]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 6, &t2, &nt[2]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 7, &t3, &nt[3]);
    const size_t n_probe_chunks = *min_element(np, np + 4);
    const size_t n_target_chunks = *min_element(nt, nt + 4);

    const vector<int> h_thr = make_target_schedule();
    int* d_thr = nullptr;
    gpuErrchk(hipMalloc(&d_thr, sizeof(int) * N_WINDOWS));
    gpuErrchk(hipMemcpy(d_thr, h_thr.data(), sizeof(int) * N_WINDOWS, hipMemcpyHostToDevice));

    const size_t probe_count_elems = (size_t)MAX_PROBE_CUS * N_WINDOWS * WAVES_PER_BLOCK;
    const size_t target_count_elems = (size_t)N_TARGET_CUS * N_WINDOWS * WAVES_PER_BLOCK;
    const int max_blocks = CU_NUM * XCD_NUM;
    uint64_t *d_probe_cnt = nullptr, *d_target_cnt = nullptr;
    float* d_sink = nullptr;
    gpuErrchk(hipMalloc(&d_probe_cnt, sizeof(uint64_t) * probe_count_elems));
    gpuErrchk(hipMalloc(&d_target_cnt, sizeof(uint64_t) * target_count_elems));
    gpuErrchk(hipMalloc(&d_sink, sizeof(float) * max_blocks * WAVES_PER_BLOCK));

    hipStream_t stream;
    gpuErrchk(hipStreamCreate(&stream));
    vector<uint64_t> h_probe(probe_count_elems), h_target(target_count_elems);

    auto run_pass = [&](int n_target_slots, int bpx, bool target_enabled,
                        bool probe_enabled, bool target_throttle, bool capture) -> PassResult {
        assert(n_target_slots + bpx <= CU_NUM);
        gpuErrchk(hipMemsetAsync(d_probe_cnt, 0, sizeof(uint64_t) * probe_count_elems, stream));
        gpuErrchk(hipMemsetAsync(d_target_cnt, 0, sizeof(uint64_t) * target_count_elems, stream));

        int nts = n_target_slots, bpxv = bpx;
        int te = target_enabled ? 1 : 0, pe = probe_enabled ? 1 : 0;
        int tt = target_throttle ? 1 : 0;
        int W = N_WINDOWS;
        uint64_t wc = window_cycles;
        const int grid_blocks = (n_target_slots + bpx) * XCD_NUM;
        void* kargs[] = {
            &p0, &p1, &p2, &p3, (void*)&n_probe_chunks,
            &t0a, &t1, &t2, &t3, (void*)&n_target_chunks,
            &nts, &bpxv, &te, &pe, &tt, &W, &wc, &d_thr, &d_probe_cnt, &d_target_cnt, &d_sink
        };
        gpuErrchk(hipLaunchCooperativeKernel((void*)bw_fidelity_kernel,
                                             dim3(grid_blocks), dim3(BLOCKDIM_X),
                                             kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));

        PassResult result;
        if (!capture) return result;
        gpuErrchk(hipMemcpy(h_probe.data(), d_probe_cnt, sizeof(uint64_t) * probe_count_elems,
                            hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(h_target.data(), d_target_cnt, sizeof(uint64_t) * target_count_elems,
                            hipMemcpyDeviceToHost));
        result.probe.assign(N_WINDOWS, 0.0);
        result.target.assign(N_WINDOWS, 0.0);
        for (int w = 0; w < N_WINDOWS; w++) {
            uint64_t pi = 0, ti = 0;
            for (int r = 0; r < bpx; r++)
                for (int wave = 0; wave < WAVES_PER_BLOCK; wave++)
                    pi += h_probe[(r * N_WINDOWS + w) * WAVES_PER_BLOCK + wave];
            for (int r = 0; r < n_target_slots; r++)
                for (int wave = 0; wave < WAVES_PER_BLOCK; wave++)
                    ti += h_target[(r * N_WINDOWS + w) * WAVES_PER_BLOCK + wave];
            result.probe[w] = (double)pi * bytes_per_wave_iter / window_sec / 1e9;
            result.target[w] = (double)ti * bytes_per_wave_iter / window_sec / 1e9;
        }
        return result;
    };

    auto measured_pass = [&](int n_target_slots, int bpx, bool target_enabled,
                             bool probe_enabled, bool target_throttle) -> PassResult {
        run_pass(n_target_slots, bpx, target_enabled, probe_enabled, target_throttle, false);
        return run_pass(n_target_slots, bpx, target_enabled, probe_enabled, target_throttle, true);
    };

    printf("[bw-fidelity] pattern=%s src_xcd=%d dst_hbm=%d target_cus=%d windows=%d "
           "window_us=%d n_probe_chunks=%zu n_target_chunks=%zu clock=%uMHz\n",
           target_pattern_name(), SRC_XCD, DST_HBM, N_TARGET_CUS, N_WINDOWS, WINDOW_US,
           n_probe_chunks, n_target_chunks, clock);

    constexpr int N_BPX = sizeof(BW_ACTIVE_CUS) / sizeof(BW_ACTIVE_CUS[0]);
    const int capacity_bpx = BW_ACTIVE_CUS[N_BPX - 1];
    const PassResult capacity_pass = measured_pass(N_TARGET_CUS, capacity_bpx, true, true, false);
    vector<double> capacity_samples(N_WINDOWS);
    for (int w = 0; w < N_WINDOWS; w++)
        capacity_samples[w] = capacity_pass.target[w] + capacity_pass.probe[w];
    const double capacity_gbps = median(capacity_samples);
    printf("[capacity] calibration=dual_streamers target_cus=%d probe_cus=%d capacity_gbps=%.2f\n",
           N_TARGET_CUS, capacity_bpx, capacity_gbps);

    vector<double> solo_probe(N_BPX);
    vector<PassResult> solo_target(N_BPX), concurrent(N_BPX);
    for (int bi = 0; bi < N_BPX; bi++) {
        const int bpx = BW_ACTIVE_CUS[bi];
        const PassResult probe_only = measured_pass(N_TARGET_CUS, bpx, false, true, true);
        solo_probe[bi] = median(probe_only.probe);
        solo_target[bi] = measured_pass(N_TARGET_CUS, bpx, true, false, true);
        concurrent[bi] = measured_pass(N_TARGET_CUS, bpx, true, true, true);
        printf("[calibration] bpx=%d solo_probe_gbps=%.2f\n", bpx, solo_probe[bi]);
        fflush(stdout);
    }

    printf("bpx,window,thr_cyc,capacity_gbps,solo_probe_gbps,probe_gbps,"
           "probe_loss_pct,informative,solo_target_gbps,target_gbps,target_est_gbps,"
           "abs_err_gbps,rel_err_pct,sum_gbps\n");
    vector<ErrorStats> by_bpx(N_BPX), by_bpx_informative(N_BPX);
    ErrorStats overall, informative_overall;
    for (int bi = 0; bi < N_BPX; bi++) {
        const int bpx = BW_ACTIVE_CUS[bi];
        for (int w = 0; w < N_WINDOWS; w++) {
            const double truth = concurrent[bi].target[w];
            const double estimate = max(0.0, capacity_gbps - concurrent[bi].probe[w]);
            const double loss = max(0.0, 1.0 - concurrent[bi].probe[w] / solo_probe[bi]);
            const bool informative = loss >= MIN_PROBE_LOSS_FRAC &&
                                     solo_probe[bi] >= capacity_gbps * MIN_PROBE_CAPACITY_FRAC;
            const double abs_err = fabs(estimate - truth);
            const double rel_err = truth > 1e-9 ? abs_err / truth * 100.0 : 0.0;
            printf("%d,%d,%d,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                   bpx, w, h_thr[w], capacity_gbps, solo_probe[bi], concurrent[bi].probe[w],
                   loss * 100.0, informative ? 1 : 0, solo_target[bi].target[w], truth,
                   estimate, abs_err, rel_err, truth + concurrent[bi].probe[w]);
            by_bpx[bi].add(estimate, truth);
            overall.add(estimate, truth);
            if (informative) {
                by_bpx_informative[bi].add(estimate, truth);
                informative_overall.add(estimate, truth);
            }
        }
    }

    printf("summary_bpx,bpx,all_mae_gbps,all_mape_pct,informative_n,informative_coverage_pct,"
           "informative_mae_gbps,informative_rmse_gbps,informative_mape_pct,informative_bias_gbps\n");
    for (int bi = 0; bi < N_BPX; bi++) {
        const ErrorStats& a = by_bpx[bi];
        const ErrorStats& c = by_bpx_informative[bi];
        const double cn = c.n > 0 ? c.n : 1;
        printf("summary_bpx,%d,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n", BW_ACTIVE_CUS[bi],
               a.abs_sum / a.n, a.rel_sum / a.n, c.n, 100.0 * c.n / a.n,
               c.n ? c.abs_sum / cn : 0.0, c.n ? sqrt(c.sq_sum / cn) : 0.0,
               c.n ? c.rel_sum / cn : 0.0, c.n ? c.bias_sum / cn : 0.0);
    }
    const double in = informative_overall.n > 0 ? informative_overall.n : 1;
    printf("summary_overall,all_mae_gbps,all_rmse_gbps,all_mape_pct,all_bias_gbps,"
           "informative_n,informative_coverage_pct,informative_mae_gbps,informative_rmse_gbps,"
           "informative_mape_pct,informative_bias_gbps\n");
    printf("summary_overall,%.2f,%.2f,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.2f,%.2f\n",
           overall.abs_sum / overall.n, sqrt(overall.sq_sum / overall.n),
           overall.rel_sum / overall.n, overall.bias_sum / overall.n,
           informative_overall.n, 100.0 * informative_overall.n / overall.n,
           informative_overall.n ? informative_overall.abs_sum / in : 0.0,
           informative_overall.n ? sqrt(informative_overall.sq_sum / in) : 0.0,
           informative_overall.n ? informative_overall.rel_sum / in : 0.0,
           informative_overall.n ? informative_overall.bias_sum / in : 0.0);
    return 0;
}
