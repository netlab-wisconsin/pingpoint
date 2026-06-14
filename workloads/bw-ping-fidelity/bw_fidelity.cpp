// =============================================================================================
// bw-ping-fidelity (E2): see bw_fidelity.h for the experiment description.
//
// Output: CSV  bpx,window,thr_cyc,target_gbps,probe_gbps,sum_gbps
//   - one block per row of (bpx, window); replay the same target schedule across bpx=1..16.
//   - the window(s) where thr_cyc is largest (target near-idle) give the path peak (probe owns it).
//   - fidelity check (offline): probe_gbps ~= peak - target_gbps, and how that tightens with bpx.
// =============================================================================================

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cassert>

#include "../../mem_bench/gpu-clock.cuh"

#include "main.h"
#include "k2.h"
#include "bw_fidelity.h"

using namespace std;
namespace cg = cooperative_groups;

// bytes a block streams per iteration: each thread issues 4 independent float4 loads (64B).
static constexpr int BYTES_PER_THREAD_ITER = 4 * 16;

// =============================================================================================
// Co-run kernel: TARGET (throttled streamer) + PROBE (unthrottled bw ping), both on SRC_XCD->DST_HBM.
// Time is bucketed by absolute cycle; each role records per-window iteration counts (thread 0).
// =============================================================================================
__global__ void bw_fidelity_kernel(
    uint64_t* p0, uint64_t* p1, uint64_t* p2, uint64_t* p3, size_t n_probe_chunks,
    uint64_t* t0a, uint64_t* t1, uint64_t* t2, uint64_t* t3, size_t n_target_chunks,
    int n_target_cus, int bpx, int W, uint64_t window_cycles,
    const int* __restrict__ thr_cyc,
    uint64_t* __restrict__ probe_cnt,   // [bpx * W]
    uint64_t* __restrict__ target_cnt,  // [n_target_cus * W]
    float* __restrict__ sink)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    const int tbid_in_xcd  = (bid / XCD_NUM) % n_tbs_in_xcd;

    uint32_t xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    grid.sync();                                   // align starts across blocks
    const uint64_t t0 = __builtin_readcyclecounter();
    if (xcc_id != SRC_XCD) return;                 // only the source XCD works

    int is_target, rlocal;
    if (tbid_in_xcd < n_target_cus)               { is_target = 1; rlocal = tbid_in_xcd; }
    else if (tbid_in_xcd < n_target_cus + bpx)    { is_target = 0; rlocal = tbid_in_xcd - n_target_cus; }
    else return;                                   // unused reserved block

    uint64_t *c0, *c1, *c2, *c3; size_t nch;
    if (is_target) { c0 = t0a; c1 = t1; c2 = t2; c3 = t3; nch = n_target_chunks; }
    else           { c0 = p0;  c1 = p1;  c2 = p2;  c3 = p3;  nch = n_probe_chunks; }
    uint64_t* outcnt = is_target ? target_cnt : probe_cnt;

    const int chunk_elems = CHUNK_SIZE / 16;          // float4 lanes per chunk (128)
    const int groups = BLOCKDIM_X / chunk_elems;      // chunks read per array per iter (8)
    const int g   = tid / chunk_elems;                // which chunk this thread reads
    const int off = tid % chunk_elems;                // lane within that chunk

    float s = 0.f;
    int cur_w = 0;
    int thr = is_target ? thr_cyc[0] : 0;             // steady per-iter delay (cycles)
    uint64_t cnt = 0, iter = 0;

    for (;;) {
        const uint64_t now = __builtin_readcyclecounter();
        const int w = (int)((now - t0) / window_cycles);
        if (w >= W) break;
        if (w != cur_w) {
            if (tid == 0) outcnt[rlocal * W + cur_w] = cnt;
            cur_w = w; cnt = 0;
            thr = is_target ? thr_cyc[w] : 0;
        }

        // one iteration: each thread issues 4 independent float4 loads (64B) from distinct chunks
        const size_t base = (iter * (size_t)n_tbs_in_xcd + tbid_in_xcd) * groups + g;
        const size_t ci = base % nch;
        float4 a0 = reinterpret_cast<float4*>(c0[ci])[off];
        float4 a1 = reinterpret_cast<float4*>(c1[ci])[off];
        float4 a2 = reinterpret_cast<float4*>(c2[ci])[off];
        float4 a3 = reinterpret_cast<float4*>(c3[ci])[off];
        s += a0.x + a1.x + a2.x + a3.x;
        cnt++; iter++;

        // steady throttle: wait `thr` cycles between iterations (target only; bounded < window)
        if (thr > 0) {
            const uint64_t st = __builtin_readcyclecounter();
            while (__builtin_readcyclecounter() - st < (uint64_t)thr) { }
        }
    }
    if (tid == 0) outcnt[rlocal * W + cur_w] = cnt;
    if (tid == 0 && s == 1.2345678e30f) sink[bid] = s;   // keep loads live
}

// helper: device chunk-pointer array + count for data array `i` restricted to DST_HBM
static void hbm_chunks(const vector<uint64_t*>& d_chunks_per_hbm,
                       const vector<vector<size_t>>& h_offsets,
                       const vector<vector<size_t>>& h_xcd_sz,
                       int i, uint64_t** ptr_out, size_t* cnt_out) {
    *ptr_out = d_chunks_per_hbm[i] + h_offsets[i][DST_HBM];
    *cnt_out = h_xcd_sz[i][DST_HBM];
}

int main(int argc, char** argv) {
    unsigned int clock = getGPUClock();                 // MHz
    const uint64_t window_cycles = (uint64_t)WINDOW_US * (uint64_t)clock;
    const double   window_sec    = (double)WINDOW_US * 1e-6;

    // ---------------------------------------------------------------------------------------------
    // K2 data setup: N_DATAS arrays, home-identified per HBM (reuse ../moe/k2.h).
    // ---------------------------------------------------------------------------------------------
    const int n_datas = N_DATAS;
    const long long data_size = (long long)K2_N_PAGES * PAGE_SIZE;
    const int chunk_size = CHUNK_SIZE;
    const size_t n_chunks = data_size / chunk_size;

    vector<char*> d_data(n_datas);
    for (int i = 0; i < n_datas; i++) {
        gpuErrchk(hipMalloc((void**)&d_data[i], data_size + 0x1000));
        d_data[i] = (char*)(((uintptr_t)d_data[i] & ~(0x0FFF)) + 0x1000);
    }
    vector<vector<int>>    h_home(n_datas, vector<int>(n_chunks));
    vector<vector<size_t>> h_xcd_sz(n_datas, vector<size_t>(XCD_NUM, 0));
    if (k2::home_identification(d_data, data_size, n_chunks, n_datas, h_home, h_xcd_sz) == -1)
        return -1;

    // group per-xcd chunk pointers, lay out contiguously per data with per-xcd offsets
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
            h_offsets[i][x] = off; off += xcd_chunks[x].size();
        }
    }

    // probe = arrays [0,N_PROBE_DATAS), target = the next 4
    uint64_t *p0,*p1,*p2,*p3,*t0a,*t1,*t2,*t3; size_t np[4], nt[4];
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 0, &p0, &np[0]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 1, &p1, &np[1]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 2, &p2, &np[2]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 3, &p3, &np[3]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 4, &t0a, &nt[0]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 5, &t1, &nt[1]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 6, &t2, &nt[2]);
    hbm_chunks(d_chunks_per_hbm, h_offsets, h_xcd_sz, 7, &t3, &nt[3]);
    const size_t n_probe_chunks  = *min_element(np, np + 4);
    const size_t n_target_chunks = *min_element(nt, nt + 4);

    // ---------------------------------------------------------------------------------------------
    // Target throttle staircase: N_LEVELS plateaus, delay THR_MAX_CYCLES -> 0 cycles
    // (lowest target BW -> full intensity), so target BW ramps up across the windows.
    // ---------------------------------------------------------------------------------------------
    vector<int> h_thr(N_WINDOWS);
    for (int w = 0; w < N_WINDOWS; w++) {
        int level = w * N_LEVELS / N_WINDOWS;                          // 0..N_LEVELS-1
        h_thr[w] = THR_MAX_CYCLES * (N_LEVELS - 1 - level) / (N_LEVELS - 1);
    }
    int* d_thr = nullptr;
    gpuErrchk(hipMalloc(&d_thr, sizeof(int) * N_WINDOWS));
    gpuErrchk(hipMemcpy(d_thr, h_thr.data(), sizeof(int) * N_WINDOWS, hipMemcpyHostToDevice));

    // output counters + dce sink
    const int max_blocks = (N_TARGET_CUS + 16) * XCD_NUM;
    uint64_t *d_probe_cnt, *d_target_cnt; float* d_sink;
    gpuErrchk(hipMalloc(&d_probe_cnt,  sizeof(uint64_t) * 16 * N_WINDOWS));
    gpuErrchk(hipMalloc(&d_target_cnt, sizeof(uint64_t) * N_TARGET_CUS * N_WINDOWS));
    gpuErrchk(hipMalloc(&d_sink, sizeof(float) * max_blocks));

    printf("[bw-fidelity] src_xcd=%d dst_hbm=%d target_cus=%d windows=%d window_us=%d "
           "n_probe_chunks=%zu n_target_chunks=%zu clock=%uMHz\n",
           SRC_XCD, DST_HBM, N_TARGET_CUS, N_WINDOWS, WINDOW_US,
           n_probe_chunks, n_target_chunks, clock);
    printf("bpx,window,thr_cyc,target_gbps,probe_gbps,sum_gbps\n");

    hipStream_t stream; gpuErrchk(hipStreamCreate(&stream));
    vector<uint64_t> h_probe(16 * N_WINDOWS), h_target(N_TARGET_CUS * N_WINDOWS);

    for (int bpx : BW_ACTIVE_CUS) {
        assert(N_TARGET_CUS + bpx <= CU_NUM);
        gpuErrchk(hipMemsetAsync(d_probe_cnt,  0, sizeof(uint64_t) * 16 * N_WINDOWS, stream));
        gpuErrchk(hipMemsetAsync(d_target_cnt, 0, sizeof(uint64_t) * N_TARGET_CUS * N_WINDOWS, stream));

        const int grid_blocks = (N_TARGET_CUS + bpx) * XCD_NUM;
        int ntc = N_TARGET_CUS, bpxv = bpx, W = N_WINDOWS;
        uint64_t wc = window_cycles;
        void* kargs[] = { &p0,&p1,&p2,&p3,(void*)&n_probe_chunks,
                          &t0a,&t1,&t2,&t3,(void*)&n_target_chunks,
                          &ntc,&bpxv,&W,&wc,&d_thr,&d_probe_cnt,&d_target_cnt,&d_sink };
        gpuErrchk(hipLaunchCooperativeKernel((void*)bw_fidelity_kernel,
                  dim3(grid_blocks), dim3(BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));

        gpuErrchk(hipMemcpy(h_probe.data(),  d_probe_cnt,  sizeof(uint64_t) * 16 * N_WINDOWS, hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(h_target.data(), d_target_cnt, sizeof(uint64_t) * N_TARGET_CUS * N_WINDOWS, hipMemcpyDeviceToHost));

        const double bytes_per_block_iter = (double)BLOCKDIM_X * BYTES_PER_THREAD_ITER;
        for (int w = 0; w < N_WINDOWS; w++) {
            uint64_t ti = 0, pi = 0;
            for (int r = 0; r < N_TARGET_CUS; r++) ti += h_target[r * N_WINDOWS + w];
            for (int r = 0; r < bpx; r++)          pi += h_probe[r * N_WINDOWS + w];
            double tgt_gbps = (double)ti * bytes_per_block_iter / window_sec / 1e9;
            double prb_gbps = (double)pi * bytes_per_block_iter / window_sec / 1e9;
            printf("%d,%d,%d,%.2f,%.2f,%.2f\n", bpx, w, h_thr[w],
                   tgt_gbps, prb_gbps, tgt_gbps + prb_gbps);
        }
        fflush(stdout);
    }
    return 0;
}
