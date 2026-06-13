// =============================================================================================
// moe-serving (E1): FFN1 (expert up-projection GEMM) as the target kernel, co-run with a
// PingPoint ping, to measure target throughput vs. ping rate.
//
//   Target  : Gemm1 (FFN1) over E experts. Regime is set by tokens-per-expert:
//               prefill -> large cap  (compute-bound)
//               decode  -> small cap  (memory-bound weight streaming)
//   Ping    : one Latency (k1) or one Bandwidth (k2) ping, src_xcd 0 -> dst_hbm 0.
//   Rate    : Option-A duty cycle over a serving loop of N passes, deterministic stride.
//               a fraction r of passes co-run the ping (fused_kernel2, bpx>0),
//               the rest run ping-free (a null plan with bpx=0, full grid).
//   Metric  : per-pass target time = device-clock span of the target blocks
//             (moe_start_clk/moe_end_clk in PingOut2), i.e. contention-isolated from any
//             ping tail. Throughput = B_ffn1 / mean target time.
//
// Output: CSV lines  regime,kind,rate,n_meas,mean_ns,p99_ns,gbps,mean_wall_ns
//
// NOTE on ping length: a co-run ping must outlast the target so the target runs fully
// contended for its whole span. PING_ITERS_* default high for that; if a bandwidth pass
// reports mean_ns ~ the baseline (no slowdown), the ping finished early -> raise PING_ITERS_K2.
// (mean_wall_ns >> mean_ns on rate=1 bandwidth indicates the ping safely outlasted the target.)
// =============================================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cinttypes>

#include "../../mem_bench/gpu-clock.cuh"
#include "../../mem_bench/MeasurementSeries.hpp"

#include "main.h"
#include "ppnt.h"
#include "ppnt_only.h"
#include "moe_only.h"   // for Gemm1Args / Gemm1TargetFn (and fill_random)
#include "k1.h"
#include "k2.h"

#include "moe_serving.h"

using namespace std;

namespace cg = cooperative_groups;

// Fused target+ping kernel for the serving sweep.
// Records the target's runtime as the MAX over target blocks of (end-start) measured WITHIN a
// single block, i.e. one clock domain -> immune to cross-XCD cycle-counter skew, and excludes any
// ping tail. `bpx==0` => no ping, full grid is target (baseline pass).
template <typename TargetFn, typename TargetArgs>
__global__ void fused_serving_kernel(TargetFn target_fn, const TargetArgs* __restrict__ targs,
                                     const ppnt::PingSpec* __restrict__ plan,
                                     uint64_t* __restrict__ target_dur /*[1], atomicMax of cycles*/,
                                     uint64_t* __restrict__ ping_iterclk)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    const int tbid_in_xcd  = (bid / XCD_NUM) % n_tbs_in_xcd;

    uint32_t xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    const ppnt::PingSpec spec = *plan;
    grid.sync(); // align ping and target start

    if (tbid_in_xcd < (int)spec.bpx) {
        // reserved ping block; only blocks on the ping's source XCD do work
        if (spec.src_xcd != xcc_id) return;
        if (spec.kind == ppnt::PingKind::Latency) {
            if (tid != 0) return;
            k1::k<k1::dtype>(spec.data, spec.dummy, spec.iters, ping_iterclk);
        } else {
            k2::k(spec.data0, spec.data1, spec.data2, spec.data3,
                  spec.sink, (spec.data_bytes / CHUNK_SIZE), CHUNK_SIZE, spec.iters, ping_iterclk);
        }
    } else {
        const uint64_t s = __builtin_readcyclecounter();
        target_fn(targs, bid, tid, gridDim.x, blockDim.x, (int)spec.bpx);
        const uint64_t e = __builtin_readcyclecounter();
        if (tid == 0) atomicMax((unsigned long long*)target_dur, (unsigned long long)(e - s));
    }
}

int main(int argc, char **argv) {

    // argv: [1]=regime(prefill|decode)  [2]=N passes  [3]=bpx
    string regime = (argc > 1) ? argv[1] : "prefill";
    const int N        = (argc > 2) ? atoi(argv[2]) : N_PASSES_DEFAULT;
    const int bpx_bw   = (argc > 3) ? atoi(argv[3]) : BPX_DEFAULT;
    const bool is_decode = (regime == "decode");
    const int T = is_decode ? DECODE_T : PREFILL_T;

    unsigned int clock = getGPUClock();

    // =============================================================================================
    // K1 SETUP (latency ping pointer-chase buffers, one chain per HBM)  [FAST path]
    // =============================================================================================
    vector<k1::dtype*> k1_dbuf_start_ptrs_per_hbm(HBM_NUM, nullptr);
    k1::dtype *k1_dummy_buf = nullptr;
    {
        k1::dtype *dbuf_base = nullptr;
        gpuErrchk(hipMallocManaged(&k1_dummy_buf, sizeof(k1::dtype)));
        k1_dummy_buf[0] = 0;

        const size_t LEN = (1 << 16); // 8MB per HBM
        const size_t multiplicative_factor = XCD_NUM * 2;
        const size_t cl_bytes = 128;
        const size_t cl_size = cl_bytes / sizeof(k1::dtype);
        const size_t skip_factor = 1;
        const size_t n_dtype_dbuf = multiplicative_factor * skip_factor * cl_size * LEN;
        const size_t n_cl_dbuf = n_dtype_dbuf / (cl_size * skip_factor);
        gpuErrchk(hipMalloc(&dbuf_base, n_dtype_dbuf * sizeof(k1::dtype)));

        vector<uint32_t> dtype_home_xcd(n_dtype_dbuf, (uint32_t)-1);
        vector<vector<uint32_t>> hbm_dtypes(HBM_NUM);
        vector<uint32_t> cl_home_xcd(n_cl_dbuf, (uint32_t)-1);
        vector<vector<uint32_t>> hbm_cls(HBM_NUM);

        if (k1::home_identification(dbuf_base, n_dtype_dbuf, n_cl_dbuf, cl_size, cl_bytes, skip_factor,
                                    dtype_home_xcd, cl_home_xcd, hbm_dtypes, hbm_cls) == -1)
            return -1;

        vector<k1::dtype> buf(n_dtype_dbuf, 0);
        random_device rd; mt19937 g(rd());
        for (int v = 0; v < HBM_NUM; v++) {
            vector<uint32_t> seq(LEN);
            for (size_t i = 0; i < LEN; i++) seq[i] = hbm_cls[v][i];
            shuffle(seq.begin(), seq.end(), g);
            for (int cl_lane = 0; cl_lane < (int)cl_size; cl_lane++) {
                for (size_t i = 0; i < LEN; i++) {
                    uint32_t cur_cl = seq[i], next_cl = seq[(i + 1) % LEN];
                    size_t cur_elem  = ((size_t)cur_cl  * cl_size + cl_lane) * skip_factor;
                    size_t next_elem = ((size_t)next_cl * cl_size + cl_lane) * skip_factor;
                    if (cur_elem >= n_dtype_dbuf || next_elem >= n_dtype_dbuf) {
                        cerr << "BUG: k1 elem OOB\n"; return 1;
                    }
                    buf[cur_elem] = (k1::dtype)((uintptr_t)dbuf_base + next_elem * sizeof(k1::dtype));
                }
            }
            size_t start_elem = ((size_t)seq[0] * cl_size + 0) * skip_factor;
            k1_dbuf_start_ptrs_per_hbm[v] = dbuf_base + start_elem;
        }
        gpuErrchk(hipMemcpy(dbuf_base, buf.data(), n_dtype_dbuf * sizeof(k1::dtype), hipMemcpyHostToDevice));
        gpuErrchk(hipDeviceSynchronize());
    }

    // =============================================================================================
    // K2 SETUP (bandwidth ping streaming chunks, grouped per HBM)  [FAST path]
    // =============================================================================================
    const int k2_n_datas = 4;
    vector<uint64_t*> k2_d_chunks_per_hbm(k2_n_datas);
    vector<vector<size_t>> k2_h_offsets(k2_n_datas, vector<size_t>(HBM_NUM));
    vector<size_t> k2_min_num_chunks_over_n_datas(HBM_NUM);
    {
        const long long k2_n_pages = 512;            // 1GB per data (FAST)
        const int k2_page_size = PAGE_SIZE;
        const long long k2_data_size = (k2_n_pages * k2_page_size);
        const int k2_chunk_size = CHUNK_SIZE;
        const size_t k2_n_chunks = k2_data_size / k2_chunk_size;

        vector<char*> k2_d_data(k2_n_datas);
        for (int i = 0; i < k2_n_datas; i++) {
            gpuErrchk(hipMalloc((void**)&k2_d_data[i], k2_data_size + 0x1000));
            k2_d_data[i] = (char*)(((uintptr_t)k2_d_data[i] & ~(0x0FFF)) + 0x1000);
        }
        vector<vector<int>> k2_h_home(k2_n_datas, vector<int>(k2_n_chunks));
        vector<vector<size_t>> k2_h_xcd_chunks_size(k2_n_datas, vector<size_t>(XCD_NUM, 0));
        if (k2::home_identification(k2_d_data, k2_data_size, k2_n_chunks, k2_n_datas,
                                    k2_h_home, k2_h_xcd_chunks_size) == -1)
            return -1;

        vector<vector<vector<uint64_t>>> k2_xcd_chunks(k2_n_datas, vector<vector<uint64_t>>(XCD_NUM));
        for (int i = 0; i < k2_n_datas; i++)
            for (size_t k = 0; k < k2_n_chunks; k++)
                k2_xcd_chunks[i][k2_h_home[i][k]].push_back((uint64_t)k2_d_data[i] + k * k2_chunk_size);

        for (int i = 0; i < k2_n_datas; i++) {
            gpuErrchk(hipMalloc((void**)&k2_d_chunks_per_hbm[i], sizeof(uint64_t) * k2_n_chunks));
            size_t off = 0;
            for (int x = 0; x < XCD_NUM; x++) {
                size_t nc = k2_h_xcd_chunks_size[i][x];
                gpuErrchk(hipMemcpy(&k2_d_chunks_per_hbm[i][off], k2_xcd_chunks[i][x].data(),
                                    sizeof(uint64_t) * nc, hipMemcpyHostToDevice));
                k2_h_offsets[i][x] = off; off += nc;
            }
        }
        fill(k2_min_num_chunks_over_n_datas.begin(), k2_min_num_chunks_over_n_datas.end(), SIZE_MAX);
        for (int i = 0; i < k2_n_datas; i++)
            for (int x = 0; x < XCD_NUM; x++)
                k2_min_num_chunks_over_n_datas[x] =
                    min(k2_min_num_chunks_over_n_datas[x], k2_h_xcd_chunks_size[i][x]);
    }

    // =============================================================================================
    // GRID CONFIG (persistent cooperative grid: 1 block / CU, rounded to XCD_NUM)
    // =============================================================================================
    int physical_grid_size;
    {
        const int num_sms = CU_NUM * XCD_NUM;
        const int target_physical_blocks = 1 * num_sms;
        physical_grid_size = (target_physical_blocks / XCD_NUM) * XCD_NUM;
        if (physical_grid_size < XCD_NUM) physical_grid_size = XCD_NUM;
    }
    const int dst_hbm = 0, src_xcd = 0;

    // =============================================================================================
    // BUILD 3 PLANS: null (bpx=0), latency (bpx=1), bandwidth (bpx=bpx_bw). All src0 -> hbm0.
    // =============================================================================================
    vector<ppnt::PingSpec> h_plan(N_PLAN_KINDS);
    vector<ppnt::PingOut2> h_out2(N_PLAN_KINDS);
    ppnt::PingSpec* d_plan[N_PLAN_KINDS];
    ppnt::PingOut2* d_out2[N_PLAN_KINDS];

    auto bw_data_bytes = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[dst_hbm];
    float* bw_sink = nullptr;
    gpuErrchk(hipMalloc(&bw_sink, sizeof(float) * (size_t)TARGET_BLOCKDIM_X * physical_grid_size));

    for (int k = 0; k < N_PLAN_KINDS; k++) {
        ppnt::PingSpec p{};
        p.ping_id = k; p.src_xcd = src_xcd; p.dst_hbm = dst_hbm;
        if (k == PLAN_NULL) {                 // no ping: full grid, records target span
            p.kind = ppnt::PingKind::Bandwidth; p.bpx = 0; p.iters = 1;
        } else if (k == PLAN_LAT) {           // latency ping
            p.kind = ppnt::PingKind::Latency;  p.bpx = 1; p.iters = PING_ITERS_K1;
            p.data = k1_dbuf_start_ptrs_per_hbm[dst_hbm]; p.dummy = k1_dummy_buf;
        } else {                              // bandwidth ping
            p.kind = ppnt::PingKind::Bandwidth; p.bpx = bpx_bw; p.iters = PING_ITERS_K2;
            p.data_bytes = bw_data_bytes;
            p.data0 = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][dst_hbm];
            p.data1 = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][dst_hbm];
            p.data2 = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][dst_hbm];
            p.data3 = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][dst_hbm];
            p.sink  = bw_sink;
        }
        h_plan[k] = p;

        ppnt::PingOut2 o2{};
        gpuErrchk(hipMalloc(&o2.iterClk, sizeof(uint64_t) * max((size_t)1, p.iters * p.bpx)));
        gpuErrchk(hipMalloc(&o2.moe_start_clk, sizeof(uint64_t)));
        gpuErrchk(hipMalloc(&o2.moe_end_clk,   sizeof(uint64_t)));
        h_out2[k] = o2;

        gpuErrchk(hipMalloc(&d_plan[k], sizeof(ppnt::PingSpec)));
        gpuErrchk(hipMalloc(&d_out2[k], sizeof(ppnt::PingOut2)));
        gpuErrchk(hipMemcpy(d_plan[k], &h_plan[k], sizeof(ppnt::PingSpec), hipMemcpyHostToDevice));
        gpuErrchk(hipMemcpy(d_out2[k], &h_out2[k], sizeof(ppnt::PingOut2), hipMemcpyHostToDevice));
    }

    // =============================================================================================
    // FFN1 (Gemm1) TARGET DATA
    // =============================================================================================
    const int d = D_MODEL, E = N_EXPERT, hidden = 4 * d;
    const int cap = (T + E - 1) / E + 64;

    // uniform token distribution across experts (E1; flip to hot-expert for E3)
    vector<int> h_cnt(E, 0);
    { int base = T / E, rem = T - base * E;
      for (int e = 0; e < E; e++) h_cnt[e] = base + (e < rem ? 1 : 0); }

    float *d_Xexp = nullptr, *d_W1 = nullptr, *d_Tmp = nullptr; int *d_cnt = nullptr;
    gpuErrchk(hipMalloc(&d_Xexp, sizeof(float) * (size_t)E * cap * d));
    gpuErrchk(hipMalloc(&d_W1,   sizeof(float) * (size_t)E * d * hidden));
    gpuErrchk(hipMalloc(&d_Tmp,  sizeof(float) * (size_t)E * cap * hidden));
    gpuErrchk(hipMalloc(&d_cnt,  sizeof(int)   * (size_t)E));
    {
        float *h_W1 = (float*)malloc(sizeof(float) * (size_t)E * d * hidden);
        fill_random(h_W1, (size_t)E * d * hidden);
        gpuErrchk(hipMemcpy(d_W1, h_W1, sizeof(float) * (size_t)E * d * hidden, hipMemcpyHostToDevice));
        free(h_W1);
        float *h_X = (float*)malloc(sizeof(float) * (size_t)E * cap * d);
        fill_random(h_X, (size_t)E * cap * d);
        gpuErrchk(hipMemcpy(d_Xexp, h_X, sizeof(float) * (size_t)E * cap * d, hipMemcpyHostToDevice));
        free(h_X);
        gpuErrchk(hipMemcpy(d_cnt, h_cnt.data(), sizeof(int) * E, hipMemcpyHostToDevice));
    }

    Gemm1Args h_args = { d_Xexp, d_W1, d_Tmp, d_cnt, E, cap, d, hidden };
    Gemm1Args* d_args = nullptr;
    gpuErrchk(hipMalloc(&d_args, sizeof(Gemm1Args)));
    gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(Gemm1Args), hipMemcpyHostToDevice));
    Gemm1TargetFn fn{};

    // bytes touched by FFN1 (read Xexp + read W1 + write Tmp); effective tokens = sum cnt = T
    const size_t B_ffn1 = sizeof(float) * ((size_t)T * d + (size_t)E * d * hidden + (size_t)T * hidden);

    hipStream_t stream; gpuErrchk(hipStreamCreate(&stream));
    uint64_t* d_target_dur = nullptr; gpuErrchk(hipMalloc(&d_target_dur, sizeof(uint64_t)));

    // one fused co-run pass; returns max-per-block target duration (ns) and wall time (ns)
    auto run_pass = [&](int plan_idx, double* wall_ns_out) -> double {
        static const uint64_t zero = 0;
        gpuErrchk(hipMemcpy(d_target_dur, &zero, sizeof(uint64_t), hipMemcpyHostToDevice));

        uint64_t* iterclk = h_out2[plan_idx].iterClk;
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan[plan_idx],
                          (void*)&d_target_dur, (void*)&iterclk };

        hipEvent_t ev_s, ev_e;
        gpuErrchk(hipEventCreate(&ev_s)); gpuErrchk(hipEventCreate(&ev_e));
        gpuErrchk(hipEventRecord(ev_s, stream));
        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)fused_serving_kernel<Gemm1TargetFn, Gemm1Args>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipEventRecord(ev_e, stream));
        gpuErrchk(hipStreamSynchronize(stream));

        float wall_ms = 0; gpuErrchk(hipEventElapsedTime(&wall_ms, ev_s, ev_e));
        *wall_ns_out = (double)wall_ms * 1e6;
        gpuErrchk(hipEventDestroy(ev_s)); gpuErrchk(hipEventDestroy(ev_e));

        uint64_t dur = 0;
        gpuErrchk(hipMemcpy(&dur, d_target_dur, sizeof(uint64_t), hipMemcpyDeviceToHost));
        return (double)dur / (double)clock * 1e3; // cycles / MHz * 1e3 = ns
    };

    // =============================================================================================
    // SERVING-LOOP SWEEP
    // =============================================================================================
    printf("[MoE-serving] regime=%s T=%d d=%d hidden=%d E=%d cap=%d | grid=%d bpx_bw=%d N=%d warmup=%d\n",
           regime.c_str(), T, d, hidden, E, cap, physical_grid_size, bpx_bw, N, WARMUP_PASSES);
    printf("regime,kind,rate,n_meas,mean_ns,p99_ns,gbps,mean_wall_ns\n");

    struct Cond { const char* kind; double rate; int plan_idx; };
    vector<Cond> conds;
    conds.push_back({ "baseline", 0.0, -1 });
    for (double r : PING_RATES) conds.push_back({ "latency",   r, PLAN_LAT });
    for (double r : PING_RATES) conds.push_back({ "bandwidth", r, PLAN_BW  });

    for (const Cond& c : conds) {
        const int period = (c.rate > 0.0) ? (int)lround(1.0 / c.rate) : 0;
        MeasurementSeries span, wall;
        for (int i = 0; i < N; i++) {
            bool is_ping = (c.plan_idx >= 0) && (period > 0) && (i % period == 0);
            int plan_idx = is_ping ? c.plan_idx : PLAN_NULL;
            double w = 0.0;
            double ns = run_pass(plan_idx, &w);
            if (i >= WARMUP_PASSES) { span.add(ns); wall.add(w); }
        }
        double mean_ns = span.value();
        double p99_ns  = span.getPercentile(0.99);
        double gbps    = (double)B_ffn1 / mean_ns; // bytes/ns = GB/s
        printf("%s,%s,%.2f,%d,%.1f,%.1f,%.2f,%.1f\n",
               regime.c_str(), c.kind, c.rate, (int)(N - WARMUP_PASSES),
               mean_ns, p99_ns, gbps, wall.value());
        fflush(stdout);
    }

    return 0;
}
