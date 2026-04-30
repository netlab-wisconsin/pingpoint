#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cinttypes>

#include "../../mem_bench/gpu-clock.cuh"
#include "../../mem_bench/MeasurementSeries.hpp"

#include "main.h"
#include "ppnt.h"
#include "k1.h"
#include "k2.h"

#include "db.h"

// ---------------------------------------------------------------------------
// Verbosity: 0=quiet, 1=PPNT pings, 2+=misc debug
// ---------------------------------------------------------------------------
#define DEBUG_LEVEL 1

// FAST=1 uses smaller buffers (faster run, less accurate) for development.
#define FAST 1

// Generate all (src_xcd x dst_hbm) ping combos.  Set to 1 to limit to a
// focused subset that best illustrates the hot-entry congestion story.
#define PPNT_PLAN_SELECTED_ONLY 0

#define DISABLE_K1_PLANS 0
#define DISABLE_K2_PLANS 0
// Keep K2 data setup for HBM placement, but disable K2 ping generation.
#define ENABLE_BW_PINGS 0

// Which DB stages to co-profile with pings
#define PPNT_PROFILE_VECSCAN_HOT   1   // VecScan, hot entry in HOT_HBM  (bad placement)
#define PPNT_PROFILE_ATOMICAGG     1   // AtomicAgg, hot agg in HOT_HBM
#define PPNT_PROFILE_VECSCAN_LOCAL 1   // VecScan, hot entry in LOCAL_HBM (after migration)

#define TARGET_BLOCKDIM_X 1024

using namespace std;

// Print per-XCD QPS distribution derived from per-block wall-clock timing.
// ns_per_query percentiles are converted to QPS (higher = better).
// p50/p90/p99 QPS mean "X% of blocks achieve at least this throughput."
static void parse_vecscan_qps(
    const uint64_t *d_start, const uint64_t *d_end, const int *d_n_queries,
    int physical_grid_size, int n_tbs_in_xcd, unsigned int clock_mhz)
{
    vector<uint64_t> h_start(physical_grid_size);
    vector<uint64_t> h_end(physical_grid_size);
    vector<int>      h_nq(physical_grid_size);
    gpuErrchk(hipMemcpy(h_start.data(), d_start,     sizeof(uint64_t) * physical_grid_size, hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_end.data(),   d_end,       sizeof(uint64_t) * physical_grid_size, hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_nq.data(),    d_n_queries, sizeof(int)      * physical_grid_size, hipMemcpyDeviceToHost));

    printf("[QPS per XCD — higher is better]\n");
    for (int x = 0; x < XCD_NUM; x++) {
        int count = 0;
        MeasurementSeries ms;
        // Round-robin scheduling: XCD x owns blocks where bid % XCD_NUM == x
        for (int b = x; b < physical_grid_size; b += XCD_NUM) {
            if (h_nq[b] <= 0) continue;
            double cpq   = (double)(h_end[b] - h_start[b]) / h_nq[b];
            double ns_pq = cpq / (double)clock_mhz * 1e3;
            ms.add(ns_pq);
            count++;
        }
        if (count == 0) continue;
        // Percentiles of ns/query → QPS.  p50/p90/p99 are from the slow tail,
        // so 1/p90_ns = throughput achieved by the slowest-10% blocks, etc.
        double mean_ns = ms.value();
        double p50_ns  = ms.getPercentile(0.50);
        double p90_ns  = ms.getPercentile(0.90);
        double p99_ns  = ms.getPercentile(0.99);
        printf("  XCD%d (%2d blocks): mean=%9.0f  p50=%9.0f  p90=%9.0f  p99=%9.0f  QPS\n",
               x, count,
               1e9 / mean_ns, 1e9 / p50_ns, 1e9 / p90_ns, 1e9 / p99_ns);
    }
}

// Print per-XCD tail query latency (ns) from per-query wall-clock measurements.
// q_start is read before the hot-chunk load; q_end is read after the final syncthreads.
// Each query's latency therefore includes its full HBM stall time, un-amortized.
static void parse_query_tail_latency(
    const uint64_t *d_latencies, const uint32_t *d_bids,
    int Q, int n_tbs_in_xcd, unsigned int clock_mhz)
{
    vector<uint64_t> h_lat(Q);
    vector<uint32_t> h_bid(Q);
    gpuErrchk(hipMemcpy(h_lat.data(), d_latencies, sizeof(uint64_t) * Q, hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(h_bid.data(), d_bids,      sizeof(uint32_t) * Q, hipMemcpyDeviceToHost));

    vector<MeasurementSeries> per_xcd(XCD_NUM);
    MeasurementSeries all_ms;
    for (int q = 0; q < Q; q++) {
        if (h_lat[q] == 0) continue; // unwritten slot (ping block owns this query)
        double ns = h_lat[q] / (double)clock_mhz * 1e3;
        int x = (int)(h_bid[q] % XCD_NUM); // round-robin: XCD = bid % XCD_NUM
        all_ms.add(ns);
        if (x >= 0 && x < XCD_NUM) per_xcd[x].add(ns);
    }

    auto print_row = [](const char *lbl, MeasurementSeries &ms) {
        printf("  %-6s p50=%7.1f  p90=%7.1f  p99=%7.1f  p99.9=%7.1f  p99.99=%7.1f  ns\n",
               lbl,
               ms.getPercentile(0.500),  ms.getPercentile(0.900),
               ms.getPercentile(0.990),  ms.getPercentile(0.999),
               ms.getPercentile(0.9999));
    };

    printf("[Query tail latency (ns) — lower is better]\n");
    print_row("ALL:", all_ms);
    for (int x = 0; x < XCD_NUM; x++) {
        char lbl[8]; snprintf(lbl, sizeof(lbl), "XCD%d:", x);
        print_row(lbl, per_xcd[x]);
    }
}

static float launch_fused_and_time(
    void *kernel, int physical_grid_size, void **kargs, hipStream_t stream)
{
    hipEvent_t ev_s, ev_e;
    gpuErrchk(hipEventCreate(&ev_s));
    gpuErrchk(hipEventCreate(&ev_e));
    gpuErrchk(hipEventRecord(ev_s, stream));
    gpuErrchk(hipLaunchCooperativeKernel(
        kernel, dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
    gpuErrchk(hipEventRecord(ev_e, stream));
    gpuErrchk(hipStreamSynchronize(stream));
    float ms = 0.0f;
    gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
    gpuErrchk(hipEventDestroy(ev_s));
    gpuErrchk(hipEventDestroy(ev_e));
    return ms;
}

static void log_tput_impact(
    const char *tag, double n_ops, const char *unit, float solo_ms, float corun_ms, size_t n_replays)
{
    if (n_replays == 0) n_replays = 1;
    const double corun_per_ping_ms = corun_ms / (double)n_replays;
    const double solo_tput  = n_ops / (solo_ms  * 1e-3);
    const double corun_tput = n_ops / (corun_per_ping_ms * 1e-3);
    const double drop_pct = (solo_tput > 0.0) ? (100.0 * (1.0 - corun_tput / solo_tput)) : 0.0;
    const double slowdown = (solo_ms > 0.0f) ? (corun_per_ping_ms / solo_ms) : 0.0;
    printf("[DB THROUGHPUT] %s: solo=%.2f ms (%.0f %s), co-run-total=%.2f ms over %zu pings, "
           "co-run-per-ping=%.2f ms (%.0f %s), drop=%.2f%%, slowdown=%.2fx\n",
           tag, solo_ms, solo_tput, unit, corun_ms, n_replays, corun_per_ping_ms, corun_tput, unit, drop_pct, slowdown);
}

int main(int argc, char **argv) {

    // @june: gpu clock set to 2100 as mi3008x node reports incorrect clock values
    unsigned int clock = 2100;
    // unsigned int clock = getGPUClock();

    // =========================================================================
    // K1 SETUP  (pointer-chase latency data, one chain per HBM)
    // =========================================================================

    vector<k1::dtype *> k1_dbuf_start_ptrs_per_hbm(HBM_NUM, nullptr);
    k1::dtype *k1_dummy_buf;
    size_t k1_profile_iters = -1;
    {
#if !(DISABLE_K1_PLANS)
        k1::dtype *dbuf_base = nullptr;
        gpuErrchk(hipMallocManaged(&k1_dummy_buf, sizeof(k1::dtype)));
        k1_dummy_buf[0] = 0;

#if FAST
        const size_t LEN = (1 << 16);
        const size_t multiplicative_factor = XCD_NUM * 2;
        k1_profile_iters = 10000;
#else
        const size_t LEN = (1 << 22);
        const size_t multiplicative_factor = XCD_NUM * 1;
        k1_profile_iters = max(LEN, (size_t)10000);
#endif

        const size_t cl_bytes = 128;
        const size_t cl_size  = cl_bytes / sizeof(k1::dtype);
        const size_t skip_factor = 1;
        const size_t n_dtype_dbuf = multiplicative_factor * skip_factor * cl_size * LEN;
        const size_t n_cl_dbuf    = n_dtype_dbuf / (cl_size * skip_factor);
        gpuErrchk(hipMalloc(&dbuf_base, n_dtype_dbuf * sizeof(k1::dtype)));

        vector<uint32_t> dtype_home_xcd(n_dtype_dbuf, (uint32_t)-1);
        vector<vector<uint32_t>> hbm_dtypes(HBM_NUM);
        vector<uint32_t> cl_home_xcd(n_cl_dbuf, (uint32_t)-1);
        vector<vector<uint32_t>> hbm_cls(HBM_NUM);

        if (k1::home_identification(
                dbuf_base, n_dtype_dbuf, n_cl_dbuf, cl_size, cl_bytes, skip_factor,
                dtype_home_xcd, cl_home_xcd, hbm_dtypes, hbm_cls) == -1)
            return -1;

#if DEBUG_LEVEL >= 1
        for (int v = 0; v < HBM_NUM; v++) {
            string level = hbm_dtypes[v].size() * sizeof(k1::dtype) > L2_SIZE
                ? (hbm_dtypes[v].size() * sizeof(k1::dtype) > LLC_SIZE ? "hbm" : "llc") : "l2";
            cout << "K1 pinned data: hbm" << v << " "
                 << hbm_dtypes[v].size() * sizeof(k1::dtype) / (1024 * 1024)
                 << "MB at " << level << "\n" << flush;
        }
#endif

        vector<k1::dtype> buf(n_dtype_dbuf, 0);
        random_device rd;
        mt19937 g(rd());

        for (int v = 0; v < HBM_NUM; v++) {
            vector<uint32_t> seq(LEN);
            for (size_t i = 0; i < LEN; i++) seq[i] = hbm_cls[v][i];
            shuffle(seq.begin(), seq.end(), g);

            for (int cl_lane = 0; cl_lane < (int)cl_size; cl_lane++) {
                for (size_t i = 0; i < LEN; i++) {
                    uint32_t cur_cl  = seq[i];
                    uint32_t next_cl = seq[(i + 1) % LEN];
                    size_t cur_elem  = ((size_t)cur_cl  * cl_size + cl_lane) * skip_factor;
                    size_t next_elem = ((size_t)next_cl * cl_size + cl_lane) * skip_factor;
                    if (cur_elem >= n_dtype_dbuf || next_elem >= n_dtype_dbuf) {
                        cerr << "BUG: elem OOB\n";
                        return 1;
                    }
                    buf[cur_elem] = (k1::dtype)((uintptr_t)dbuf_base + next_elem * sizeof(k1::dtype));
                }
            }
            size_t start_elem = ((size_t)seq[0] * cl_size + 0) * skip_factor;
            k1_dbuf_start_ptrs_per_hbm[v] = dbuf_base + start_elem;
        }

        gpuErrchk(hipMemcpy(dbuf_base, buf.data(), n_dtype_dbuf * sizeof(k1::dtype), hipMemcpyHostToDevice));
        gpuErrchk(hipDeviceSynchronize());
#endif // !DISABLE_K1_PLANS
    }

    // =========================================================================
    // K2 SETUP  (streaming bandwidth data, 4 independent streams)
    // =========================================================================

    const int k2_n_datas = 4;
    vector<uint64_t *> k2_d_chunks_per_hbm(k2_n_datas);
    vector<vector<size_t>> k2_h_offsets(k2_n_datas, vector<size_t>(HBM_NUM));
    vector<size_t> k2_min_num_chunks_over_n_datas(HBM_NUM);
    size_t k2_profile_iters = -1;
    {
#if !(DISABLE_K2_PLANS)
#if FAST
        const long long k2_n_pages = 512;
#else
        const long long k2_n_pages = (128 << 6);
#endif
        const int k2_page_size   = PAGE_SIZE;
        const long long k2_data_size  = k2_n_pages * k2_page_size;
        const int k2_chunk_size  = CHUNK_SIZE;
        const size_t k2_n_chunks = k2_data_size / k2_chunk_size;

        vector<char *> k2_d_data(k2_n_datas);
        for (int i = 0; i < k2_n_datas; i++) {
            gpuErrchk(hipMalloc((void **)&k2_d_data[i], k2_data_size + 0x1000));
            k2_d_data[i] = (char *)(((uintptr_t)k2_d_data[i] & ~(0x0FFF)) + 0x1000);
        }

        vector<vector<int>> k2_h_home(k2_n_datas, vector<int>(k2_n_chunks));
        vector<vector<size_t>> k2_h_xcd_chunks_size(k2_n_datas, vector<size_t>(XCD_NUM, 0));

        if (k2::home_identification(k2_d_data, k2_data_size, k2_n_chunks, k2_n_datas,
                                    k2_h_home, k2_h_xcd_chunks_size) == -1)
            return -1;

        vector<vector<vector<uint64_t>>> k2_xcd_chunks(k2_n_datas,
            vector<vector<uint64_t>>(XCD_NUM));
        for (int i = 0; i < k2_n_datas; i++)
            for (size_t k = 0; k < k2_n_chunks; k++)
                k2_xcd_chunks[i][k2_h_home[i][k]].push_back(
                    reinterpret_cast<uint64_t>(k2_d_data[i]) + k * k2_chunk_size);

        for (int i = 0; i < k2_n_datas; i++) {
            gpuErrchk(hipMalloc((void **)&k2_d_chunks_per_hbm[i], sizeof(uint64_t) * k2_n_chunks));
            size_t _offset = 0;
            for (int x = 0; x < XCD_NUM; x++) {
                size_t _n = k2_h_xcd_chunks_size[i][x];
                gpuErrchk(hipMemcpy(&k2_d_chunks_per_hbm[i][_offset],
                                    k2_xcd_chunks[i][x].data(),
                                    sizeof(uint64_t) * _n, hipMemcpyHostToDevice));
                k2_h_offsets[i][x] = _offset;
                _offset += _n;
            }
        }

        fill(k2_min_num_chunks_over_n_datas.begin(), k2_min_num_chunks_over_n_datas.end(), SIZE_MAX);
        for (int i = 0; i < k2_n_datas; i++)
            for (int x = 0; x < XCD_NUM; x++)
                k2_min_num_chunks_over_n_datas[x] =
                    min(k2_min_num_chunks_over_n_datas[x], k2_h_xcd_chunks_size[i][x]);

#if DEBUG_LEVEL >= 1
        for (int v = 0; v < HBM_NUM; v++) {
            size_t tot = 0;
            for (int i = 0; i < k2_n_datas; i++) tot += k2_h_xcd_chunks_size[i][v];
            size_t tot_bytes = tot * k2_chunk_size;
            string level = tot_bytes > L2_SIZE
                ? (tot_bytes > LLC_SIZE ? "hbm" : "llc") : "l2";
            cout << "K2 pinned data: hbm" << v << " "
                 << tot_bytes / (1024 * 1024) << "MB at " << level << "\n" << flush;
        }
#endif

#if FAST
        k2_profile_iters = 10000;
#else
        k2_profile_iters = max(*min_element(k2_min_num_chunks_over_n_datas.begin(),
                                            k2_min_num_chunks_over_n_datas.end()),
                               (size_t)10000);
#endif
#endif // !DISABLE_K2_PLANS
    }

    // =========================================================================
    // PPNT SETUP  (ping plan generation)
    // =========================================================================

    vector<ppnt::PingSpec> h_plan;
    vector<ppnt::PingOut>  h_out;
    ppnt::PingSpec *d_plan = nullptr;
    ppnt::PingOut  *d_out  = nullptr;

#if !(DISABLE_K1_PLANS)
    // Latency plans -------------------------------------------------------
#if !(PPNT_PLAN_SELECTED_ONLY)
    for (int x = 0; x < XCD_NUM; x++) {
        for (int v = 0; v < HBM_NUM; v++) {
            ppnt::PingSpec p{};
            p.ping_id = (int)h_plan.size();
            p.kind    = ppnt::PingKind::Latency;
            p.src_xcd = x;
            p.dst_hbm = v;
            p.iters   = k1_profile_iters;
            p.bpx     = 1;
            p.data    = k1_dbuf_start_ptrs_per_hbm[v];
            p.dummy   = k1_dummy_buf;
            h_plan.push_back(p);
            ppnt::PingOut o{};
            gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
            h_out.push_back(o);
        }
    }
#else
    // Focused set: two "readers" (XCD0, XCD1) and one "owner" (XCD4) vs
    // the hot HBM (4) and a baseline local HBM (0).
    for (int x : {0, 1, 4}) {
        for (int v : {0, 4}) {
            ppnt::PingSpec p{};
            p.ping_id = (int)h_plan.size();
            p.kind    = ppnt::PingKind::Latency;
            p.src_xcd = x;
            p.dst_hbm = v;
            p.iters   = k1_profile_iters;
            p.bpx     = 1;
            p.data    = k1_dbuf_start_ptrs_per_hbm[v];
            p.dummy   = k1_dummy_buf;
            h_plan.push_back(p);
            ppnt::PingOut o{};
            gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
            h_out.push_back(o);
        }
    }
#endif
#endif // !DISABLE_K1_PLANS

#if !(DISABLE_K2_PLANS) && (ENABLE_BW_PINGS)
    // Bandwidth plans ------------------------------------------------------
#if !(PPNT_PLAN_SELECTED_ONLY)
    for (int x = 0; x < XCD_NUM; x++) {
        for (int v = 0; v < HBM_NUM; v++) {
            for (int bpx : {1, 2, 4, 8, 16}) {
                ppnt::PingSpec p{};
                p.ping_id    = (int)h_plan.size();
                p.kind       = ppnt::PingKind::Bandwidth;
                p.src_xcd    = x;
                p.dst_hbm    = v;
                p.iters      = k2_profile_iters;
                p.bpx        = bpx;
                p.data_bytes = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[v];
                p.data0      = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][v];
                p.data1      = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][v];
                p.data2      = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][v];
                p.data3      = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][v];
                gpuErrchk(hipMalloc(&p.sink, sizeof(float) * TARGET_BLOCKDIM_X * XCD_NUM));
                h_plan.push_back(p);
                ppnt::PingOut o{};
                gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
                h_out.push_back(o);
            }
        }
    }
#else
    // Focused: same 6 pairs as K1, bpx=4
    for (int x : {0, 1, 4}) {
        for (int v : {0, 4}) {
            ppnt::PingSpec p{};
            p.ping_id    = (int)h_plan.size();
            p.kind       = ppnt::PingKind::Bandwidth;
            p.src_xcd    = x;
            p.dst_hbm    = v;
            p.iters      = k2_profile_iters;
            p.bpx        = 4;
            p.data_bytes = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[v];
            p.data0      = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][v];
            p.data1      = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][v];
            p.data2      = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][v];
            p.data3      = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][v];
            gpuErrchk(hipMalloc(&p.sink, sizeof(float) * TARGET_BLOCKDIM_X * XCD_NUM));
            h_plan.push_back(p);
            ppnt::PingOut o{};
            gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
            h_out.push_back(o);
        }
    }
#endif
#endif // !DISABLE_K2_PLANS && ENABLE_BW_PINGS

    size_t n_plan = h_plan.size();
    if (n_plan > 0) {
        gpuErrchk(hipMalloc(&d_plan, sizeof(ppnt::PingSpec) * n_plan));
        gpuErrchk(hipMalloc(&d_out,  sizeof(ppnt::PingOut)  * n_plan));
        gpuErrchk(hipMemcpy(d_plan, h_plan.data(), sizeof(ppnt::PingSpec) * n_plan, hipMemcpyHostToDevice));
        gpuErrchk(hipMemcpy(d_out,  h_out.data(),  sizeof(ppnt::PingOut)  * n_plan, hipMemcpyHostToDevice));
    }

    // =========================================================================
    // DB SETUP
    // =========================================================================

    // Query count (argv[1]).  Must satisfy Q*0.9/XCD_NUM > N_HOT_CHUNKS/2 for
    // the hot-table working set to exceed L2 (4MB/XCD) and produce HBM misses.
    // With N_HOT_CHUNKS=4096: need Q > 18205.  Default is 32768.
    const int Q      = (argc > 1) ? atoi(argv[1]) : (1 << 19); // 524288
    const int N_COLD = max(Q, 1024);
    const int N_AGG  = N_COLD; // aggregation rounds

    printf("[DB] Q=%d N_COLD=%d DB_ENTRY_DIM=%d DB_AGG_DIM=%d HOT_HBM=%d LOCAL_HBM=%d\n",
           Q, N_COLD, DB_ENTRY_DIM, DB_AGG_DIM, HOT_HBM, LOCAL_HBM);

    hipStream_t stream;
    gpuErrchk(hipStreamCreate(&stream));

    // Cooperative grid: 1 block per CU, first bpx blocks per XCD reserved for pings
    int physical_grid_size;
    {
        const int num_sms = CU_NUM * XCD_NUM;
        const int max_blocks_per_sm = 1;
        const int target_phys = max_blocks_per_sm * num_sms;
        physical_grid_size = (target_phys / XCD_NUM) * XCD_NUM;
        if (physical_grid_size < XCD_NUM) physical_grid_size = XCD_NUM;
        int logical_blocks = physical_grid_size - XCD_NUM;
        printf("[DB] Cooperative Grid: %d physical blocks (%d logical workers)\n",
               physical_grid_size, logical_blocks);
    }

    // ------------------------------------------------------------------
    // Wire up hot-table and local-table chunk pointer arrays.
    //
    // k2_d_chunks_per_hbm[0] is an already-allocated device array of
    // uint64_t where each element is the device address of a 2KB slab
    // physically in the HBM identified by home_identification.
    // k2_h_offsets[0][hbm] is the start index for that HBM's slabs.
    //
    // Layout within HOT_HBM's section:
    //   [0 .. N_HOT_CHUNKS-1]          → VecScan hot-table reads
    //   [N_HOT_CHUNKS .. 2*N_HOT_CHUNKS-1] → AtomicAgg hot-table writes
    // ------------------------------------------------------------------
#if !(DISABLE_K2_PLANS)
    assert(k2_min_num_chunks_over_n_datas[HOT_HBM]   >= 2 * N_HOT_CHUNKS &&
           "need 2*N_HOT_CHUNKS chunks in HOT_HBM");
    assert(k2_min_num_chunks_over_n_datas[LOCAL_HBM]  >= N_HOT_CHUNKS &&
           "need N_HOT_CHUNKS chunks in LOCAL_HBM");
    assert(k2_min_num_chunks_over_n_datas[XCD23_HBM]  >= N_HOT_CHUNKS &&
           "need N_HOT_CHUNKS chunks in XCD23_HBM");

    // VecScan hot path: N_HOT_CHUNKS slabs in HOT_HBM (8MB total)
    uint64_t *d_hot_chunk_ptrs   = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][HOT_HBM];
    // AtomicAgg hot path: next N_HOT_CHUNKS slabs in HOT_HBM (non-overlapping)
    uint64_t *d_hot_agg_ptrs     = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][HOT_HBM]
                                   + N_HOT_CHUNKS;
    // VecScan migration path: N_HOT_CHUNKS slabs in LOCAL_HBM (8MB total)
    uint64_t *d_local_chunk_ptrs = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][LOCAL_HBM];
    // VecScan XCD2/XCD3 focused experiment: N_HOT_CHUNKS slabs in XCD23_HBM
    uint64_t *d_hbm6_chunk_ptrs  = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][XCD23_HBM];
#else
    printf("[DB] WARNING: K2 disabled; hot table is not pinned to HOT_HBM.\n");
    // Without K2, allocate a stub pointer array on the device (no HBM guarantee).
    uint64_t *d_hot_chunk_ptrs   = nullptr;
    uint64_t *d_hot_agg_ptrs     = nullptr;
    uint64_t *d_local_chunk_ptrs = nullptr;
    uint64_t *d_hbm6_chunk_ptrs  = nullptr;
    // (In practice, re-enable K2 to get meaningful HBM-placement results.)
#endif

    // Allocate and fill query vectors, cold entries, aggregation inputs
    float *d_query_vecs  = nullptr;
    float *d_cold_entries = nullptr;
    float *d_results      = nullptr;
    float *d_agg_inputs   = nullptr;

    gpuErrchk(hipMalloc(&d_query_vecs,   sizeof(float) * (size_t)Q      * DB_ENTRY_DIM));
    gpuErrchk(hipMalloc(&d_cold_entries, sizeof(float) * (size_t)N_COLD * DB_ENTRY_DIM));
    gpuErrchk(hipMalloc(&d_results,      sizeof(float) * Q));
    gpuErrchk(hipMalloc(&d_agg_inputs,   sizeof(float) * (size_t)N_AGG  * DB_AGG_DIM));

    {
        float *h = new float[(size_t)Q * DB_ENTRY_DIM];
        db_fill_random(h, (size_t)Q * DB_ENTRY_DIM);
        gpuErrchk(hipMemcpy(d_query_vecs, h, sizeof(float) * Q * DB_ENTRY_DIM, hipMemcpyHostToDevice));
        delete[] h;
    }
    {
        float *h = new float[(size_t)N_COLD * DB_ENTRY_DIM];
        db_fill_random(h, (size_t)N_COLD * DB_ENTRY_DIM);
        gpuErrchk(hipMemcpy(d_cold_entries, h, sizeof(float) * N_COLD * DB_ENTRY_DIM, hipMemcpyHostToDevice));
        delete[] h;
    }
    {
        float *h = new float[(size_t)N_AGG * DB_AGG_DIM];
        db_fill_random(h, (size_t)N_AGG * DB_AGG_DIM);
        gpuErrchk(hipMemcpy(d_agg_inputs, h, sizeof(float) * N_AGG * DB_AGG_DIM, hipMemcpyHostToDevice));
        delete[] h;
    }

    gpuErrchk(hipStreamSynchronize(stream));

    // =========================================================================
    // PPNT EXECUTION WITH DB KERNELS
    // =========================================================================

    // 0. PPNT-only baseline — pings run without any DB kernel
    {
        printf("\n\nPPNT ONLY BASELINE\n");
        ppnt::TargetArgsT h_args{};
        ppnt::TargetArgsT *d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(ppnt::TargetArgsT)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(ppnt::TargetArgsT), hipMemcpyHostToDevice));
        ppnt::TargetFnT fn{};
        size_t _n_plan = n_plan;
        void *kargs[] = {(void *)&fn, (void *)&d_args, (void *)&d_plan,
                         (void *)&_n_plan, (void *)&d_out};
        gpuErrchk(hipLaunchCooperativeKernel(
            (void *)ppnt::fused_kernel<ppnt::TargetFnT, ppnt::TargetArgsT>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
        gpuErrchk(hipFree(d_args));
    }

    // 1. VecScan — XCD2 and XCD3 only, 90% hot queries to HBM6
    //    All other XCDs skip the query loop entirely (active_xcd_mask gates them out).
    {
        printf("\n\nVECSCAN (XCD2+XCD3 only, hot entry in HBM%d)\n", XCD23_HBM);
        uint64_t *d_blk_start = nullptr, *d_blk_end = nullptr;
        int      *d_blk_nq    = nullptr;
        gpuErrchk(hipMalloc(&d_blk_start, sizeof(uint64_t) * physical_grid_size));
        gpuErrchk(hipMalloc(&d_blk_end,   sizeof(uint64_t) * physical_grid_size));
        gpuErrchk(hipMalloc(&d_blk_nq,    sizeof(int)      * physical_grid_size));
        gpuErrchk(hipMemset(d_blk_nq, 0,  sizeof(int)      * physical_grid_size));
        uint64_t *d_q_lat = nullptr; uint32_t *d_q_bid = nullptr;
        gpuErrchk(hipMalloc(&d_q_lat, sizeof(uint64_t) * Q));
        gpuErrchk(hipMemset(d_q_lat, 0, sizeof(uint64_t) * Q));
        gpuErrchk(hipMalloc(&d_q_bid, sizeof(uint32_t) * Q));
        const int xcd23_mask = (1 << 2) | (1 << 3);
        VecScanArgs h_args = {d_hbm6_chunk_ptrs, d_cold_entries, d_query_vecs, d_results,
                              Q, N_COLD, HOT_FRAC_PCT,
                              d_blk_start, d_blk_end, d_blk_nq, d_q_lat, d_q_bid,
                              xcd23_mask};
        VecScanArgs *d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(VecScanArgs)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(VecScanArgs), hipMemcpyHostToDevice));
        VecScanTargetFn fn{};
        size_t _n_plan = n_plan;
        void *kargs[] = {(void *)&fn, (void *)&d_args, (void *)&d_plan,
                         (void *)&_n_plan, (void *)&d_out};
        size_t n_plan_solo = 0;
        void *kargs_solo[] = {(void *)&fn, (void *)&d_args, (void *)&d_plan,
                              (void *)&n_plan_solo, (void *)&d_out};
        float solo_ms = launch_fused_and_time(
            (void *)ppnt::fused_kernel<VecScanTargetFn, VecScanArgs>,
            physical_grid_size, kargs_solo, stream);
        gpuErrchk(hipMemsetAsync(d_blk_start, 0, sizeof(uint64_t) * physical_grid_size, stream));
        gpuErrchk(hipMemsetAsync(d_blk_end,   0, sizeof(uint64_t) * physical_grid_size, stream));
        gpuErrchk(hipMemsetAsync(d_blk_nq,    0, sizeof(int)      * physical_grid_size, stream));
        gpuErrchk(hipMemsetAsync(d_q_lat,     0, sizeof(uint64_t) * Q, stream));
        gpuErrchk(hipMemsetAsync(d_q_bid,     0, sizeof(uint32_t) * Q, stream));
        float corun_ms = launch_fused_and_time(
            (void *)ppnt::fused_kernel<VecScanTargetFn, VecScanArgs>,
            physical_grid_size, kargs, stream);
        log_tput_impact("VECSCAN XCD23@HBM6", (double)Q, "QPS", solo_ms, corun_ms, _n_plan);
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
        parse_vecscan_qps(d_blk_start, d_blk_end, d_blk_nq,
                          physical_grid_size, physical_grid_size / XCD_NUM, clock);
        parse_query_tail_latency(d_q_lat, d_q_bid, Q, physical_grid_size / XCD_NUM, clock);
        gpuErrchk(hipFree(d_blk_start)); gpuErrchk(hipFree(d_blk_end));
        gpuErrchk(hipFree(d_blk_nq));   gpuErrchk(hipFree(d_q_lat));
        gpuErrchk(hipFree(d_q_bid));     gpuErrchk(hipFree(d_args));
    }

    // 2. AtomicAgg — hot aggregator in HOT_HBM (bad placement)
    //    Demonstrates: all XCDs write updates to the same row in a remote HBM,
    //    creating a write-congestion hotspot on the XCD→HBM4 path.
    {
        printf("\n\nATOMICOAGG (hot_agg in HBM%d — BAD placement)\n", HOT_HBM);
        AtomicAggArgs h_args = {d_agg_inputs, d_hot_agg_ptrs, N_AGG};
        AtomicAggArgs *d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(AtomicAggArgs)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(AtomicAggArgs), hipMemcpyHostToDevice));
        AtomicAggTargetFn fn{};
        size_t _n_plan = PPNT_PROFILE_ATOMICAGG ? n_plan : 0;
        void *kargs[] = {(void *)&fn, (void *)&d_args, (void *)&d_plan,
                         (void *)&_n_plan, (void *)&d_out};
        size_t n_plan_solo = 0;
        void *kargs_solo[] = {(void *)&fn, (void *)&d_args, (void *)&d_plan,
                              (void *)&n_plan_solo, (void *)&d_out};
        float solo_ms = launch_fused_and_time(
            (void *)ppnt::fused_kernel<AtomicAggTargetFn, AtomicAggArgs>,
            physical_grid_size, kargs_solo, stream);
        float corun_ms = launch_fused_and_time(
            (void *)ppnt::fused_kernel<AtomicAggTargetFn, AtomicAggArgs>,
            physical_grid_size, kargs, stream);
        log_tput_impact("ATOMICAGG hot@HBM4", (double)N_AGG, "updates/s", solo_ms, corun_ms, _n_plan);
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
        gpuErrchk(hipFree(d_args));
    }

    gpuErrchk(hipStreamDestroy(stream));
    return 0;
}
