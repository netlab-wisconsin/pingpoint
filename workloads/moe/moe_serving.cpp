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

// Decode-path FFN1 target: single token/expert GEMV (cap=1, cnt[e]=1). Unlike Gemm1 (which tiles
// 32x32 over hidden x token and wastes 31/32 thread-rows when cap<32), this parallelizes over the
// OUTPUT columns (E x hidden), one thread per output reducing over d. Each thread streams a distinct
// column of W1[e] with coalesced, high-MLP loads (reuse ~1) -> memory-bandwidth-bound, as real
// autoregressive decode is. Uses the same Gemm1Args/layout (token = row 0 of each expert).
struct DecodeGemvTargetFn {
    __device__ __forceinline__
    void operator()(const Gemm1Args* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const
    {
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        int logical_grid_size = (n_tbs_in_xcd - (int)n_ppnt_tbs_in_xcd) * XCD_NUM;

        const size_t total_outputs = (size_t)a->E * a->hidden;          // 1 token/expert
        const size_t total_threads = (size_t)logical_grid_size * blockDimX;
        const size_t start = (size_t)logical_bid * blockDimX + tid;

        for (size_t oid = start; oid < total_outputs; oid += total_threads) {
            const int e = (int)(oid / a->hidden);
            const int j = (int)(oid % a->hidden);
            const float* x = a->Xexp + (size_t)(e * a->cap) * a->d;     // token row 0 of expert e
            const float* w = a->W1   + (size_t)e * a->d * a->hidden;
            float acc = 0.f;
            for (int k = 0; k < a->d; k++) acc += x[k] * w[(size_t)k * a->hidden + j];
            a->Tmp[(size_t)(e * a->cap) * a->hidden + j] = acc;
        }
    }
};

int main(int argc, char **argv) {

    // argv: [1]=regime(prefill|decode)  [2]=N passes
    string regime = (argc > 1) ? argv[1] : "prefill";
    const int N        = (argc > 2) ? atoi(argv[2]) : N_PASSES_DEFAULT;
    const bool is_decode = (regime == "decode");
    const int T = is_decode ? DECODE_T : PREFILL_T;

    // Optional sharding of the bandwidth plan set across processes/GPUs: this process runs only
    // its contiguous slice of the built bandwidth plans (MOE_BW_SHARD of MOE_BW_NSHARD). Unset =>
    // single shard (all bandwidth plans). Latency + baseline conditions are not sharded.
    const char* ns_env = getenv("MOE_BW_NSHARD");
    const char* sh_env = getenv("MOE_BW_SHARD");
    const int bw_nshard = (ns_env && atoi(ns_env) > 0) ? atoi(ns_env) : 1;
    const int bw_shard  = sh_env ? atoi(sh_env) : 0;
    if (bw_shard < 0 || bw_shard >= bw_nshard) {
        fprintf(stderr, "MOE_BW_SHARD=%d out of range [0,%d)\n", bw_shard, bw_nshard); return 1;
    }

    unsigned int clock = getGPUClock();

    // =============================================================================================
    // K1 SETUP (latency ping pointer-chase buffers, one chain per HBM)  [FAST path]
    // =============================================================================================
    vector<k1::dtype*> k1_dbuf_start_ptrs_per_hbm(HBM_NUM, nullptr);
    k1::dtype *k1_dummy_buf = nullptr;
#if !(DISABLE_K1_PLANS)
    {
        k1::dtype *dbuf_base = nullptr;
        gpuErrchk(hipMallocManaged(&k1_dummy_buf, sizeof(k1::dtype)));
        k1_dummy_buf[0] = 0;

#if (FAST)
        const size_t LEN = (1 << 16); // 8MB per HBM (cache-resident; fast debug)
        const size_t multiplicative_factor = XCD_NUM * 2;
#else
        const size_t LEN = (1 << 22); // 512MB per HBM (exceeds LLC -> hits real HBM)
        const size_t multiplicative_factor = XCD_NUM * 1;
#endif
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
#endif // !(DISABLE_K1_PLANS)

    // =============================================================================================
    // K2 SETUP (bandwidth ping streaming chunks, grouped per HBM)  [FAST path]
    // =============================================================================================
    const int k2_n_datas = 4;
    vector<uint64_t*> k2_d_chunks_per_hbm(k2_n_datas);
    vector<vector<size_t>> k2_h_offsets(k2_n_datas, vector<size_t>(HBM_NUM));
    vector<size_t> k2_min_num_chunks_over_n_datas(HBM_NUM);
#if !(DISABLE_K2_PLANS)
    {
#if (FAST)
        const long long k2_n_pages = 512;            // 1GB per data (cache-resident; fast debug)
#else
        const long long k2_n_pages = (128 << 6);     // 16GB per data (exceeds LLC -> streams HBM)
#endif
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
#endif // !(DISABLE_K2_PLANS)

    // =============================================================================================
    // GRID CONFIG (persistent cooperative grid, 1 block / CU)
    //   target = top TARGET_CUS_PER_XCD CUs/XCD (fixed); ping = bpx low CUs/XCD on top.
    //   per-pass grid = (TARGET_CUS_PER_XCD + bpx) * XCD_NUM (computed in run_pass).
    // =============================================================================================
    const int max_grid_blocks = CU_NUM * XCD_NUM; // upper bound (target + max ping); for sink size
    // =============================================================================================
    // BUILD PING PLANS
    //   index 0 = null plan (bpx=0): no ping, full grid, used for non-ping passes.
    //   then latency plans (src_xcd x dst_hbm), then bandwidth plans (src x hbm x #active-CU).
    //   PPNT_PLAN_SELECTED_ONLY=1 -> only the customized plans in the #else blocks below.
    // =============================================================================================
    vector<ppnt::PingSpec>  h_plan;   // host-side, kept for src/hbm/bpx lookup
    vector<ppnt::PingSpec*> d_plan;   // per-plan device copies
    size_t max_iterclk = 1;

    float* bw_sink = nullptr;
    gpuErrchk(hipMalloc(&bw_sink, sizeof(float) * (size_t)TARGET_BLOCKDIM_X * max_grid_blocks));

    auto add_plan = [&](ppnt::PingSpec p) -> int {
        int idx = (int)h_plan.size();
        p.ping_id = idx;
        max_iterclk = max(max_iterclk, p.iters * max((size_t)1, p.bpx));
        h_plan.push_back(p);
        ppnt::PingSpec* dp = nullptr;
        gpuErrchk(hipMalloc(&dp, sizeof(ppnt::PingSpec)));
        gpuErrchk(hipMemcpy(dp, &h_plan.back(), sizeof(ppnt::PingSpec), hipMemcpyHostToDevice));
        d_plan.push_back(dp);
        return idx;
    };
    auto make_lat = [&](int src, int hbm) {
        ppnt::PingSpec p{};
        p.kind = ppnt::PingKind::Latency; p.src_xcd = src; p.dst_hbm = hbm;
        p.bpx = 1; p.iters = PING_ITERS_K1;
        p.data = k1_dbuf_start_ptrs_per_hbm[hbm]; p.dummy = k1_dummy_buf;
        return p;
    };
    auto make_bw = [&](int src, int hbm, int n_cu) {
        ppnt::PingSpec p{};
        p.kind = ppnt::PingKind::Bandwidth; p.src_xcd = src; p.dst_hbm = hbm;
        p.bpx = (size_t)n_cu; p.iters = PING_ITERS_K2;
        p.data_bytes = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[hbm];
        p.data0 = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][hbm];
        p.data1 = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][hbm];
        p.data2 = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][hbm];
        p.data3 = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][hbm];
        p.sink  = bw_sink;
        return p;
    };

    // index 0: null plan (no ping)
    { ppnt::PingSpec p{}; p.kind = ppnt::PingKind::Bandwidth; p.src_xcd = 0; p.dst_hbm = 0;
      p.bpx = 0; p.iters = 1; add_plan(p); }
    const int NULL_IDX = 0;

    vector<int> lat_idx, bw_idx;

#if !(DISABLE_K1_PLANS)
#if !(PPNT_PLAN_SELECTED_ONLY)
    for (int x = 0; x < XCD_NUM; x++)
        for (int v = 0; v < HBM_NUM; v++)
            lat_idx.push_back(add_plan(make_lat(x, v)));
#else
    // --- customize latency (src_xcd, dst_hbm) pairs here ---
    // src_xcd 0 -> HBM 0,2,4,6 (one per cache complex)
    for (int v = 0; v < HBM_NUM; v += 2)
        lat_idx.push_back(add_plan(make_lat(0, v)));
#endif
#endif

#if !(DISABLE_K2_PLANS)
#if !(PPNT_PLAN_SELECTED_ONLY)
    for (int x = 0; x < XCD_NUM; x++)
        for (int v = 0; v < HBM_NUM; v++)
            for (int n_cu : BW_ACTIVE_CUS)
                bw_idx.push_back(add_plan(make_bw(x, v, n_cu)));
#else
    // --- customize bandwidth (src_xcd, dst_hbm, #active-CU) triples here ---
    // src_xcd 0 -> HBM 0,2,4,6 (one per cache complex) x every #active-CU in BW_ACTIVE_CUS (1..16)
    for (int v = 0; v < HBM_NUM; v += 2)
        for (int n_cu : BW_ACTIVE_CUS)
            bw_idx.push_back(add_plan(make_bw(0, v, n_cu)));
#endif
#endif

    // target + ping must fit within one XCD's CUs
    for (const auto& p : h_plan)
        if (TARGET_CUS_PER_XCD + (int)p.bpx > CU_NUM) {
            fprintf(stderr, "TARGET_CUS_PER_XCD(%d) + bpx(%zu) > CU_NUM(%d)\n",
                    TARGET_CUS_PER_XCD, p.bpx, CU_NUM);
            return 1;
        }

    // Single shared iterClk scratch. The ping kernels (k1/k2) write per-iteration cycle counts
    // into this buffer, but moe-serving never reads it (our metric is the target's clock span,
    // d_target_dur, not the ping's own timing). Since passes run one plan at a time and are
    // stream-synchronized, all plans can safely overwrite one buffer sized to the max iters*bpx,
    // and it is never copied back to the host.
    //
    // NOTE: if we later want the ping's own latency/bandwidth (as moe.cpp reports via
    // parse_pingouts), revert to a per-plan iterClk buffer (a shared one is clobbered every pass)
    // and add the device->host copy + parse.
    uint64_t* d_iterclk_scratch = nullptr;
    gpuErrchk(hipMalloc(&d_iterclk_scratch, sizeof(uint64_t) * max_iterclk));

    // =============================================================================================
    // FFN1 (Gemm1) TARGET DATA
    // =============================================================================================
    const int d = D_MODEL, E = N_EXPERT, hidden = 4 * d;
    // decode: cap tracks tokens/expert (no floor) so weights stream with ~no reuse (memory-bound);
    // prefill: +64 headroom as before.
    const int cap = (T + E - 1) / E + (is_decode ? 0 : 64);

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
    // prefill -> token-tiled GEMM (compute-bound); decode -> single-token GEMV (memory-bound)
    Gemm1TargetFn       fn_prefill{};
    DecodeGemvTargetFn  fn_decode{};

    // bytes touched by FFN1 (read Xexp + read W1 + write Tmp); effective tokens = sum cnt = T
    const size_t B_ffn1 = sizeof(float) * ((size_t)T * d + (size_t)E * d * hidden + (size_t)T * hidden);

    hipStream_t stream; gpuErrchk(hipStreamCreate(&stream));
    uint64_t* d_target_dur = nullptr; gpuErrchk(hipMalloc(&d_target_dur, sizeof(uint64_t)));

    // one fused co-run pass; returns max-per-block target duration (ns) and wall time (ns)
    auto run_pass = [&](int plan_idx, double* wall_ns_out) -> double {
        static const uint64_t zero = 0;
        gpuErrchk(hipMemcpy(d_target_dur, &zero, sizeof(uint64_t), hipMemcpyHostToDevice));

        // grid = target (TARGET_CUS_PER_XCD) + ping (bpx) blocks per XCD; target count is fixed.
        const int bpx = (int)h_plan[plan_idx].bpx;
        const int grid_blocks = (TARGET_CUS_PER_XCD + bpx) * XCD_NUM;

        void* kargs_prefill[] = { (void*)&fn_prefill, (void*)&d_args, (void*)&d_plan[plan_idx],
                                  (void*)&d_target_dur, (void*)&d_iterclk_scratch };
        void* kargs_decode[]  = { (void*)&fn_decode,  (void*)&d_args, (void*)&d_plan[plan_idx],
                                  (void*)&d_target_dur, (void*)&d_iterclk_scratch };

        hipEvent_t ev_s, ev_e;
        gpuErrchk(hipEventCreate(&ev_s)); gpuErrchk(hipEventCreate(&ev_e));
        gpuErrchk(hipEventRecord(ev_s, stream));
        if (is_decode) {
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)fused_serving_kernel<DecodeGemvTargetFn, Gemm1Args>,
                dim3(grid_blocks), dim3(TARGET_BLOCKDIM_X), kargs_decode, 0, stream));
        } else {
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)fused_serving_kernel<Gemm1TargetFn, Gemm1Args>,
                dim3(grid_blocks), dim3(TARGET_BLOCKDIM_X), kargs_prefill, 0, stream));
        }
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
    printf("[MoE-serving] regime=%s T=%d d=%d hidden=%d E=%d cap=%d | target_cu/xcd=%d N=%d warmup=%d | "
           "n_lat=%zu n_bw=%zu (shard %d/%d) n_rates=%zu\n",
           regime.c_str(), T, d, hidden, E, cap, TARGET_CUS_PER_XCD, N, WARMUP_PASSES,
           lat_idx.size(), bw_idx.size(), bw_shard, bw_nshard,
           sizeof(PING_RATES) / sizeof(PING_RATES[0]));
    printf("regime,kind,src_xcd,dst_hbm,bpx,rate,n_meas,mean_ns,p99_ns,gbps,mean_wall_ns\n");

    struct Cond { const char* kind; double rate; int plan_idx; };
    vector<Cond> conds;
#if !(DISABLE_BASELINE)
    conds.push_back({ "baseline", 0.0, -1 });
#endif
    for (int idx : lat_idx) for (double r : PING_RATES) conds.push_back({ "latency", r, idx });
    // this process's contiguous slice of the bandwidth plans
    const size_t bw_lo = bw_idx.size() * (size_t)bw_shard       / (size_t)bw_nshard;
    const size_t bw_hi = bw_idx.size() * (size_t)(bw_shard + 1) / (size_t)bw_nshard;
    for (size_t j = bw_lo; j < bw_hi; j++)
        for (double r : PING_RATES) conds.push_back({ "bandwidth", r, bw_idx[j] });

    for (const Cond& c : conds) {
        const int period = (c.rate > 0.0) ? (int)lround(1.0 / c.rate) : 0;
        MeasurementSeries span, wall;
        for (int i = 0; i < N; i++) {
            bool is_ping = (c.plan_idx >= 0) && (period > 0) && (i % period == 0);
            int plan_idx = is_ping ? c.plan_idx : NULL_IDX;
            double w = 0.0;
            double ns = run_pass(plan_idx, &w);
            if (i >= WARMUP_PASSES) { span.add(ns); wall.add(w); }
        }
        double mean_ns = span.value();
        double p99_ns  = span.getPercentile(0.99);
        double gbps    = (double)B_ffn1 / mean_ns; // bytes/ns = GB/s
        int    src = -1, hbm = -1; size_t bpx = 0;
        if (c.plan_idx >= 0) { src = h_plan[c.plan_idx].src_xcd; hbm = h_plan[c.plan_idx].dst_hbm; bpx = h_plan[c.plan_idx].bpx; }
        printf("%s,%s,%d,%d,%zu,%.2f,%d,%.1f,%.1f,%.2f,%.1f\n",
               regime.c_str(), c.kind, src, hbm, bpx, c.rate, (int)(N - WARMUP_PASSES),
               mean_ns, p99_ns, gbps, wall.value());
        fflush(stdout);
    }

    return 0;
}
