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

#define DISABLE_K1_PLANS 1
#define DISABLE_K2_PLANS 1

// Which DB stages to co-profile with pings
#define PPNT_PROFILE_VECSCAN_HOT   1   // VecScan, hot entry in HOT_HBM  (bad placement)
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

static void normalize_vec(float *v)
{
    double ss = 0.0;
    for (int d = 0; d < DB_ENTRY_DIM; d++) ss += (double)v[d] * (double)v[d];
    double inv = (ss > 0.0) ? (1.0 / sqrt(ss)) : 1.0;
    for (int d = 0; d < DB_ENTRY_DIM; d++) v[d] = (float)(v[d] * inv);
}

static void fill_noisy_centroid_vec(
    float *dst, const vector<float> &centroids, int centroid_id,
    mt19937 &rng, normal_distribution<float> &noise_dist)
{
    const float *centroid = centroids.data() + (size_t)centroid_id * DB_ENTRY_DIM;
    for (int d = 0; d < DB_ENTRY_DIM; d++)
        dst[d] = centroid[d] + noise_dist(rng);
    normalize_vec(dst);
}

static uint32_t db_hash_u32(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static int db_gcd(int a, int b)
{
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int db_sample_list_offset(int q, int centroid_id, int ci, int list_size)
{
    if (list_size <= 1) return 0;

    uint32_t seed = db_hash_u32((uint32_t)q ^ ((uint32_t)centroid_id * 0x9e3779b9U));
    int start = (int)(seed % (uint32_t)list_size);
    int stride = (int)(db_hash_u32(seed ^ 0x85ebca6bU) % (uint32_t)list_size);
    if (stride == 0) stride = 1;
    while (db_gcd(stride, list_size) != 1) {
        stride++;
        if (stride == list_size) stride = 1;
    }
    return (start + ci * stride) % list_size;
}

static int db_probe_centroid(int routed_centroid, int probe, int n_centroids)
{
    if (n_centroids <= 1) return 0;
    if (probe == 0) return routed_centroid;

    int radius = (probe + 1) / 2;
    int delta = (probe & 1) ? radius : -radius;
    int c = (routed_centroid + delta) % n_centroids;
    return c < 0 ? c + n_centroids : c;
}

static int db_make_candidate_id(
    int q, int routed_global_centroid, int routed_local_centroid,
    int n_local_centroids, int n_entries, int ci, int encoded_base)
{
    int list_base = routed_local_centroid * DB_LIST_SIZE;
    int list_size = min(DB_LIST_SIZE, n_entries - list_base);

    if (list_size >= DB_CANDIDATES_PER_QUERY) {
        int offset = db_sample_list_offset(q, routed_global_centroid, ci, list_size);
        return encoded_base + list_base + offset;
    }

    int nprobe = min(DB_NPROBE, n_local_centroids);
    int per_probe = (DB_CANDIDATES_PER_QUERY + nprobe - 1) / nprobe;
    int probe = min(ci / per_probe, nprobe - 1);
    int within_probe = ci % per_probe;
    int probed_local_centroid = db_probe_centroid(routed_local_centroid, probe,
                                                  n_local_centroids);
    int probed_global_centroid = routed_global_centroid
        + (probed_local_centroid - routed_local_centroid);
    list_base = probed_local_centroid * DB_LIST_SIZE;
    list_size = min(DB_LIST_SIZE, n_entries - list_base);
    int offset = db_sample_list_offset(q, probed_global_centroid, within_probe,
                                       list_size);
    return encoded_base + ((list_base + offset) % n_entries);
}

static void build_synthetic_ann_dataset(
    int Q, int N_COLD, const uint64_t *d_hot_chunk_ptrs,
    float *d_query_vecs, float *d_cold_entries, int *d_candidate_ids)
{
    const int hot_centroids = DB_HOT_CENTROIDS;
    const int cold_centroids = max(1, (N_COLD + DB_LIST_SIZE - 1) / DB_LIST_SIZE);
    const int total_centroids = hot_centroids + cold_centroids;

    mt19937 rng(789);
    uniform_real_distribution<float> centroid_dist(-1.0f, 1.0f);
    normal_distribution<float> entry_noise(0.0f, 0.05f);
    normal_distribution<float> query_noise(0.0f, 0.08f);

    vector<float> centroids((size_t)total_centroids * DB_ENTRY_DIM);
    for (int c = 0; c < total_centroids; c++) {
        float *centroid = centroids.data() + (size_t)c * DB_ENTRY_DIM;
        for (int d = 0; d < DB_ENTRY_DIM; d++)
            centroid[d] = centroid_dist(rng);
        normalize_vec(centroid);
    }

    vector<float> h_hot_entries((size_t)N_HOT_CHUNKS * DB_ENTRY_DIM);
    for (int i = 0; i < N_HOT_CHUNKS; i++) {
        int centroid_id = i / DB_LIST_SIZE;
        fill_noisy_centroid_vec(h_hot_entries.data() + (size_t)i * DB_ENTRY_DIM,
                                centroids, centroid_id, rng, entry_noise);
    }

    vector<uint64_t> h_hot_ptrs(N_HOT_CHUNKS);
    gpuErrchk(hipMemcpy(h_hot_ptrs.data(), d_hot_chunk_ptrs,
                        sizeof(uint64_t) * N_HOT_CHUNKS, hipMemcpyDeviceToHost));
    for (int i = 0; i < N_HOT_CHUNKS; i++) {
        gpuErrchk(hipMemcpy(reinterpret_cast<void *>(h_hot_ptrs[i]),
                            h_hot_entries.data() + (size_t)i * DB_ENTRY_DIM,
                            sizeof(float) * DB_ENTRY_DIM, hipMemcpyHostToDevice));
    }

    vector<float> h_cold_entries((size_t)N_COLD * DB_ENTRY_DIM);
    for (int i = 0; i < N_COLD; i++) {
        int centroid_id = hot_centroids + (i / DB_LIST_SIZE) % cold_centroids;
        fill_noisy_centroid_vec(h_cold_entries.data() + (size_t)i * DB_ENTRY_DIM,
                                centroids, centroid_id, rng, entry_noise);
    }
    gpuErrchk(hipMemcpy(d_cold_entries, h_cold_entries.data(),
                        sizeof(float) * (size_t)N_COLD * DB_ENTRY_DIM,
                        hipMemcpyHostToDevice));

    vector<double> hot_weights(hot_centroids);
    for (int c = 0; c < hot_centroids; c++)
        hot_weights[c] = 1.0 / pow((double)c + 1.0, 0.7);
    discrete_distribution<int> hot_centroid_dist(hot_weights.begin(), hot_weights.end());
    uniform_int_distribution<int> cold_centroid_dist(0, cold_centroids - 1);
    uniform_int_distribution<int> pct_dist(0, 99);

    vector<float> h_query_vecs((size_t)Q * DB_ENTRY_DIM);
    vector<int> h_candidate_ids((size_t)Q * DB_CANDIDATES_PER_QUERY);
    int hot_queries = 0;
    for (int q = 0; q < Q; q++) {
        bool is_hot = pct_dist(rng) < HOT_FRAC_PCT;
        int centroid_id;
        if (is_hot) {
            hot_queries++;
            int hot_c = hot_centroid_dist(rng);
            centroid_id = hot_c;
            for (int ci = 0; ci < DB_CANDIDATES_PER_QUERY; ci++)
                h_candidate_ids[(size_t)q * DB_CANDIDATES_PER_QUERY + ci] =
                    db_make_candidate_id(q, centroid_id, hot_c, hot_centroids,
                                         N_HOT_CHUNKS, ci, 0);
        } else {
            int cold_c = cold_centroid_dist(rng);
            centroid_id = hot_centroids + cold_c;
            for (int ci = 0; ci < DB_CANDIDATES_PER_QUERY; ci++)
                h_candidate_ids[(size_t)q * DB_CANDIDATES_PER_QUERY + ci] =
                    db_make_candidate_id(q, centroid_id, cold_c, cold_centroids,
                                         N_COLD, ci, N_HOT_CHUNKS);
        }
        fill_noisy_centroid_vec(h_query_vecs.data() + (size_t)q * DB_ENTRY_DIM,
                                centroids, centroid_id, rng, query_noise);
    }

    gpuErrchk(hipMemcpy(d_query_vecs, h_query_vecs.data(),
                        sizeof(float) * (size_t)Q * DB_ENTRY_DIM,
                        hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(d_candidate_ids, h_candidate_ids.data(),
                        sizeof(int) * (size_t)Q * DB_CANDIDATES_PER_QUERY,
                        hipMemcpyHostToDevice));

    printf("[DB ANN] hot_vectors=%d cold_vectors=%d centroids: hot=%d cold=%d "
           "list_size=%d candidates/query=%d nprobe=%d hot_queries=%d/%d\n",
           N_HOT_CHUNKS, N_COLD, hot_centroids, cold_centroids,
           DB_LIST_SIZE, DB_CANDIDATES_PER_QUERY, DB_NPROBE, hot_queries, Q);
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

#if !(DISABLE_K2_PLANS)
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
#endif // !DISABLE_K2_PLANS

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

    // Query count (argv[1]).  The corpus is split explicitly into hot vectors
    // stored in HBM-classified 2KB chunks and a separate cold vector table.
    const int Q       = (argc > 1) ? atoi(argv[1]) : (1 << 19); // 524288
    const int N_HOT   = N_HOT_CHUNKS;
    const int N_COLD  = max(Q, 1024);
    const int N_TOTAL = N_HOT + N_COLD;

    printf("[DB] Q=%d N_TOTAL=%d N_HOT=%d N_COLD=%d DB_ENTRY_DIM=%d HOT_HBM=%d LOCAL_HBM=%d\n",
           Q, N_TOTAL, N_HOT, N_COLD, DB_ENTRY_DIM, HOT_HBM, LOCAL_HBM);

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
    // VecScan owns its embedding placement buffer.  This keeps the ANN data
    // independent from K2 ping buffers and works even when K2 plans are off.
    // ------------------------------------------------------------------
    char *db_vecscan_raw = nullptr;
    gpuErrchk(hipMalloc(&db_vecscan_raw, DB_VECSCAN_DATA_SIZE + 0x1000));
    char *db_vecscan_data = (char *)(((uintptr_t)db_vecscan_raw & ~(uintptr_t)0x0FFF) + 0x1000);
    const size_t db_vecscan_n_chunks = DB_VECSCAN_DATA_SIZE / CHUNK_SIZE;

    vector<uint32_t> db_chunk_home_xcd;
    vector<vector<uint64_t>> db_chunks_per_hbm;
    if (db::home_identification(db_vecscan_data, DB_VECSCAN_DATA_SIZE,
                                db_vecscan_n_chunks, db_chunk_home_xcd,
                                db_chunks_per_hbm) == -1)
        return -1;

    for (int v : {HOT_HBM, LOCAL_HBM, XCD23_HBM}) {
        if (db_chunks_per_hbm[v].size() < N_HOT_CHUNKS) {
            fprintf(stderr, "[DB] HBM%d has only %zu VecScan chunks; need %d\n",
                    v, db_chunks_per_hbm[v].size(), N_HOT_CHUNKS);
            return 1;
        }
    }

#if DEBUG_LEVEL >= 1
    for (int v = 0; v < HBM_NUM; v++) {
        size_t bytes = db_chunks_per_hbm[v].size() * CHUNK_SIZE;
        string level = bytes > L2_SIZE ? (bytes > LLC_SIZE ? "hbm" : "llc") : "l2";
        cout << "DB VecScan pinned data: hbm" << v << " "
             << bytes / (1024 * 1024) << "MB at " << level << "\n" << flush;
    }
#endif

    auto make_db_chunk_ptrs = [](const vector<uint64_t> &chunks) {
        uint64_t *d_ptrs = nullptr;
        gpuErrchk(hipMalloc(&d_ptrs, sizeof(uint64_t) * N_HOT_CHUNKS));
        gpuErrchk(hipMemcpy(d_ptrs, chunks.data(), sizeof(uint64_t) * N_HOT_CHUNKS,
                            hipMemcpyHostToDevice));
        return d_ptrs;
    };

    uint64_t *d_hot_chunk_ptrs   = make_db_chunk_ptrs(db_chunks_per_hbm[HOT_HBM]);
    uint64_t *d_local_chunk_ptrs = make_db_chunk_ptrs(db_chunks_per_hbm[LOCAL_HBM]);
    uint64_t *d_hbm6_chunk_ptrs  = make_db_chunk_ptrs(db_chunks_per_hbm[XCD23_HBM]);

    // Allocate and fill synthetic IVF-like ANN corpus, queries, and candidate lists.
    float *d_query_vecs  = nullptr;
    float *d_cold_entries = nullptr;
    float *d_results      = nullptr;
    int   *d_candidate_ids = nullptr;
    int   *d_result_ids    = nullptr;

    gpuErrchk(hipMalloc(&d_query_vecs,   sizeof(float) * (size_t)Q      * DB_ENTRY_DIM));
    gpuErrchk(hipMalloc(&d_cold_entries, sizeof(float) * (size_t)N_COLD * DB_ENTRY_DIM));
    gpuErrchk(hipMalloc(&d_results,      sizeof(float) * Q));
    gpuErrchk(hipMalloc(&d_candidate_ids,
                        sizeof(int) * (size_t)Q * DB_CANDIDATES_PER_QUERY));
    gpuErrchk(hipMalloc(&d_result_ids,   sizeof(int)   * Q));

    build_synthetic_ann_dataset(Q, N_COLD, d_hbm6_chunk_ptrs,
                                d_query_vecs, d_cold_entries, d_candidate_ids);
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

    // 1. VecScan — XCD2 and XCD3 only, 90% hot queries scan HBM6 candidate lists
    //    All other XCDs skip the query loop entirely (active_xcd_mask gates them out).
    {
        printf("\n\nVECSCAN (XCD2+XCD3 only, hot candidate lists in HBM%d)\n", XCD23_HBM);
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
        VecScanArgs h_args = {d_hbm6_chunk_ptrs, d_cold_entries, d_query_vecs,
                              d_candidate_ids, d_results, d_result_ids,
                              Q, N_COLD, DB_CANDIDATES_PER_QUERY,
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

    gpuErrchk(hipFree(d_query_vecs));
    gpuErrchk(hipFree(d_cold_entries));
    gpuErrchk(hipFree(d_results));
    gpuErrchk(hipFree(d_candidate_ids));
    gpuErrchk(hipFree(d_result_ids));
    gpuErrchk(hipFree(d_hot_chunk_ptrs));
    gpuErrchk(hipFree(d_local_chunk_ptrs));
    gpuErrchk(hipFree(d_hbm6_chunk_ptrs));
    gpuErrchk(hipFree(db_vecscan_raw));
    gpuErrchk(hipStreamDestroy(stream));
    return 0;
}
