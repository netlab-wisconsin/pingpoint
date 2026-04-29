#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../mem_bench/MeasurementSeries.hpp"

#include "../test/main.h"
#include "../test/ppnt.h"
#include "../test/k1.h"
#include "../test/k2.h"
#include "db_kernels.h"

using namespace std;

namespace {

constexpr int TARGET_BLOCKDIM_X = 1024;

struct PingStats {
    double mean_ns = 0.0;
    double p90_ns = 0.0;
    double p99_ns = 0.0;
    double gbps = 0.0;
};

const char* kind_to_cstr(ppnt::PingKind kind) {
    return (kind == ppnt::PingKind::Latency) ? "lat" : "bw";
}

unsigned int get_device_clock_mhz() {
    hipDeviceProp_t props{};
    if (hipGetDeviceProperties(&props, 0) == hipSuccess && props.clockRate > 0) {
        return (unsigned int)(props.clockRate / 1000);
    }
    return 1800;
}

PingStats collect_ping_stats(ppnt::PingSpec* d_plan_one, ppnt::PingOut* d_out_one, unsigned int clock_mhz) {
    ppnt::PingSpec p{};
    ppnt::PingOut o{};
    gpuErrchk(hipMemcpy(&p, d_plan_one, sizeof(ppnt::PingSpec), hipMemcpyDeviceToHost));
    gpuErrchk(hipMemcpy(&o, d_out_one, sizeof(ppnt::PingOut), hipMemcpyDeviceToHost));

    const size_t n = p.iters * p.bpx;
    vector<uint64_t> h_iterClk(n);
    gpuErrchk(hipMemcpy(h_iterClk.data(), o.iterClk, sizeof(uint64_t) * n, hipMemcpyDeviceToHost));

    MeasurementSeries cycles;
    for (size_t i = 0; i < n; i++) cycles.add(h_iterClk[i]);

    PingStats s;
    s.mean_ns = cycles.value() / (double)clock_mhz * 1e3;
    s.p90_ns = cycles.getPercentile(0.9) / (double)clock_mhz * 1e3;
    s.p99_ns = cycles.getPercentile(0.99) / (double)clock_mhz * 1e3;
    if (p.kind == ppnt::PingKind::Bandwidth) {
        const size_t bytes_per_block = TARGET_BLOCKDIM_X * 4 * 16;
        const size_t iter_bytes = bytes_per_block * p.bpx;
        s.gbps = (double)iter_bytes / s.mean_ns;
    }
    return s;
}

template <typename TargetFn, typename TargetArgs>
float launch_fused(TargetFn fn,
                   TargetArgs* d_args,
                   ppnt::PingSpec* d_plan,
                   size_t n_plan,
                   ppnt::PingOut* d_out,
                   int physical_grid_size,
                   hipStream_t stream) {
    hipEvent_t ev_s, ev_e;
    gpuErrchk(hipEventCreate(&ev_s));
    gpuErrchk(hipEventCreate(&ev_e));

    void* kargs[] = {
        (void*)&fn,
        (void*)&d_args,
        (void*)&d_plan,
        (void*)&n_plan,
        (void*)&d_out
    };

    gpuErrchk(hipEventRecord(ev_s, stream));
    gpuErrchk(hipLaunchCooperativeKernel(
        (void*)ppnt::fused_kernel<TargetFn, TargetArgs>,
        dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
    gpuErrchk(hipEventRecord(ev_e, stream));
    gpuErrchk(hipStreamSynchronize(stream));

    float ms = 0.f;
    gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
    gpuErrchk(hipEventDestroy(ev_s));
    gpuErrchk(hipEventDestroy(ev_e));
    return ms;
}

template <typename TargetFn, typename TargetArgs>
void run_kernel_with_ping_localization(const char* kernel_name,
                                       TargetFn fn,
                                       TargetArgs* d_args,
                                       ppnt::PingSpec* d_plan,
                                       ppnt::PingOut* d_out,
                                       const vector<ppnt::PingSpec>& h_plan,
                                       int physical_grid_size,
                                       hipStream_t stream,
                                       unsigned int clock_mhz) {
    cout << "\n[KERNEL] " << kernel_name << "\n";
    cout << "  ping_id kind src dst bpx | ping_only(ns) corun(ns) ratio | kernel_ms\n";

    for (size_t pi = 0; pi < h_plan.size(); ++pi) {
        ppnt::PingSpec* d_plan_one = d_plan + pi;
        ppnt::PingOut* d_out_one = d_out + pi;
        const size_t one = 1;

        ppnt::TargetFnT empty_fn{};
        ppnt::TargetArgsT* d_empty = nullptr;
        ppnt::TargetArgsT h_empty{};
        gpuErrchk(hipMalloc(&d_empty, sizeof(ppnt::TargetArgsT)));
        gpuErrchk(hipMemcpyAsync(d_empty, &h_empty, sizeof(ppnt::TargetArgsT), hipMemcpyHostToDevice, stream));
        (void)launch_fused(empty_fn, d_empty, d_plan_one, one, d_out_one, physical_grid_size, stream);
        PingStats solo_ping = collect_ping_stats(d_plan_one, d_out_one, clock_mhz);
        gpuErrchk(hipFree(d_empty));

        const float corun_ms = launch_fused(fn, d_args, d_plan_one, one, d_out_one, physical_grid_size, stream);
        PingStats corun_ping = collect_ping_stats(d_plan_one, d_out_one, clock_mhz);

        const ppnt::PingSpec& p = h_plan[pi];
        const double ratio = (solo_ping.mean_ns > 0.0) ? (corun_ping.mean_ns / solo_ping.mean_ns) : 0.0;
        cout << "  " << setw(6) << p.ping_id
             << " " << setw(4) << kind_to_cstr(p.kind)
             << " " << setw(3) << p.src_xcd
             << " " << setw(3) << p.dst_hbm
             << " " << setw(3) << p.bpx
             << " | " << fixed << setprecision(2) << setw(10) << solo_ping.mean_ns
             << " " << setw(9) << corun_ping.mean_ns
             << " " << setw(6) << ratio
             << " | " << setw(8) << corun_ms
             << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const unsigned int clock_mhz = get_device_clock_mhz();
    hipStream_t stream;
    gpuErrchk(hipStreamCreate(&stream));

    const size_t db_len = (argc > 1) ? strtoull(argv[1], nullptr, 10) : (size_t)XCD_NUM * (128 / sizeof(k1::dtype)) * (1 << 14);
    const int query_n = (argc > 2) ? atoi(argv[2]) : (1 << 20);
    const int vec_rows = (argc > 3) ? atoi(argv[3]) : 8192;
    const int vec_dim = (argc > 4) ? atoi(argv[4]) : 128;

    const int hot_hbm = 4;
    const int near_hbm = 0;
    const int reader_xcd = 0;
    const int writer_xcd = 1;

    const int num_sms = CU_NUM * XCD_NUM;
    const int physical_grid_size = (num_sms / XCD_NUM) * XCD_NUM;

    cout << "[SETUP] db_len=" << db_len
         << " query_n=" << query_n
         << " vec_rows=" << vec_rows
         << " vec_dim=" << vec_dim << "\n";

    // K1 setup: per-HBM pointer-chasing sources for latency pings.
    vector<k1::dtype*> k1_dbuf_start_ptrs_per_hbm(HBM_NUM, nullptr);
    vector<k1::dtype*> k1_hbm_bufs(HBM_NUM, nullptr);
    k1::dtype* k1_dummy_buf = nullptr;
    const size_t k1_profile_iters = 8000;
    const size_t cl_size = 128 / sizeof(k1::dtype);
    const size_t len_per_hbm = (1 << 15);
    const size_t n_dtype_per_hbm = cl_size * len_per_hbm;

    gpuErrchk(hipMallocManaged(&k1_dummy_buf, sizeof(k1::dtype)));
    k1_dummy_buf[0] = 0;

    mt19937 g(42);
    for (int h = 0; h < HBM_NUM; h++) {
        gpuErrchk(hipMalloc(&k1_hbm_bufs[h], sizeof(k1::dtype) * n_dtype_per_hbm));
        vector<k1::dtype> host_cycle(n_dtype_per_hbm, 0);
        vector<size_t> cl_idx(len_per_hbm);
        for (size_t i = 0; i < len_per_hbm; i++) cl_idx[i] = i;
        shuffle(cl_idx.begin(), cl_idx.end(), g);

        for (size_t lane = 0; lane < cl_size; lane++) {
            for (size_t i = 0; i < len_per_hbm; i++) {
                const size_t cur_cl = cl_idx[i];
                const size_t next_cl = cl_idx[(i + 1) % len_per_hbm];
                const size_t cur_elem = cur_cl * cl_size + lane;
                const size_t next_elem = next_cl * cl_size + lane;
                host_cycle[cur_elem] = (k1::dtype)((uintptr_t)k1_hbm_bufs[h] + next_elem * sizeof(k1::dtype));
            }
        }
        gpuErrchk(hipMemcpy(k1_hbm_bufs[h], host_cycle.data(), sizeof(k1::dtype) * n_dtype_per_hbm, hipMemcpyHostToDevice));
        k1_dbuf_start_ptrs_per_hbm[h] = k1_hbm_bufs[h];
    }

    // DB table (query entries): logical partition by HBM index for migration experiments.
    k1::dtype* d_table = nullptr;
    gpuErrchk(hipMalloc(&d_table, db_len * sizeof(k1::dtype)));
    vector<k1::dtype> h_table(db_len);
    for (size_t i = 0; i < db_len; i++) h_table[i] = (k1::dtype)(i & 1023);
    gpuErrchk(hipMemcpy(d_table, h_table.data(), db_len * sizeof(k1::dtype), hipMemcpyHostToDevice));

    const size_t per_hbm_entries = max((size_t)1, db_len / HBM_NUM);
    auto pick_idx = [&](int hbm, size_t pos) -> int {
        const size_t base = (size_t)hbm * per_hbm_entries;
        const size_t idx = min(base + pos, db_len - 1);
        return (int)idx;
    };

    const int hot_idx_remote = pick_idx(hot_hbm, per_hbm_entries / 4);
    const int hot_idx_remote2 = pick_idx(hot_hbm, per_hbm_entries / 4 + 1);
    const int hot_idx_local = pick_idx(near_hbm, per_hbm_entries / 4);
    const int hot_idx_local2 = pick_idx(near_hbm, per_hbm_entries / 4 + 1);

    cout << "[HOT-ENTRY] remote(HBM" << hot_hbm << ") idx=" << hot_idx_remote
         << " local(HBM" << near_hbm << ") idx=" << hot_idx_local << "\n";

    // K2 setup: chunk pointers for bandwidth pings.
    const int k2_n_datas = 4;
    const size_t k2_data_size = 64ull * 1024ull * 1024ull;
    const size_t k2_n_chunks = k2_data_size / CHUNK_SIZE;
    const size_t k2_profile_iters = 4000;

    vector<char*> k2_d_data(k2_n_datas);
    for (int i = 0; i < k2_n_datas; i++) {
        gpuErrchk(hipMalloc((void**)&k2_d_data[i], k2_data_size + 0x1000));
        k2_d_data[i] = (char*)(((uintptr_t)k2_d_data[i] & ~(0x0FFF)) + 0x1000);
    }

    vector<uint64_t*> k2_d_chunks_per_hbm(k2_n_datas);
    vector<vector<size_t>> k2_h_offsets(k2_n_datas, vector<size_t>(HBM_NUM, 0));
    vector<size_t> k2_min_num_chunks_over_n_datas(HBM_NUM, 0);

    for (int i = 0; i < k2_n_datas; i++) {
        vector<uint64_t> all_chunks(k2_n_chunks);
        for (size_t k = 0; k < k2_n_chunks; k++) all_chunks[k] = (uint64_t)((uintptr_t)k2_d_data[i] + k * CHUNK_SIZE);
        gpuErrchk(hipMalloc((void**)&k2_d_chunks_per_hbm[i], sizeof(uint64_t) * k2_n_chunks));
        gpuErrchk(hipMemcpy(k2_d_chunks_per_hbm[i], all_chunks.data(), sizeof(uint64_t) * k2_n_chunks, hipMemcpyHostToDevice));
    }
    const size_t chunks_per_hbm = k2_n_chunks / HBM_NUM;
    for (int x = 0; x < HBM_NUM; x++) {
        for (int i = 0; i < k2_n_datas; i++) {
            k2_h_offsets[i][x] = (size_t)x * chunks_per_hbm;
        }
        k2_min_num_chunks_over_n_datas[x] = chunks_per_hbm;
    }

    // Ping plan: focus on hot-entry remote path and migrated local path.
    vector<ppnt::PingSpec> h_plan;
    vector<ppnt::PingOut> h_out;
    ppnt::PingSpec* d_plan = nullptr;
    ppnt::PingOut* d_out = nullptr;

    auto add_latency_ping = [&](int src, int dst) {
        ppnt::PingSpec p{};
        p.ping_id = (uint16_t)h_plan.size();
        p.kind = ppnt::PingKind::Latency;
        p.src_xcd = src;
        p.dst_hbm = dst;
        p.iters = k1_profile_iters;
        p.bpx = 1;
        p.data = k1_dbuf_start_ptrs_per_hbm[dst];
        p.dummy = k1_dummy_buf;
        h_plan.push_back(p);

        ppnt::PingOut o{};
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);
    };

    auto add_bw_ping = [&](int src, int dst, int bpx) {
        ppnt::PingSpec p{};
        p.ping_id = (uint16_t)h_plan.size();
        p.kind = ppnt::PingKind::Bandwidth;
        p.src_xcd = src;
        p.dst_hbm = dst;
        p.iters = k2_profile_iters;
        p.bpx = bpx;
        p.data_bytes = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[dst];
        p.data0 = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][dst];
        p.data1 = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][dst];
        p.data2 = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][dst];
        p.data3 = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][dst];
        gpuErrchk(hipMalloc(&p.sink, sizeof(float) * TARGET_BLOCKDIM_X * XCD_NUM));
        h_plan.push_back(p);

        ppnt::PingOut o{};
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);
    };

    for (int src : {0, 1}) {
        add_latency_ping(src, hot_hbm);
        add_latency_ping(src, near_hbm);
        add_bw_ping(src, hot_hbm, 4);
        add_bw_ping(src, near_hbm, 4);
    }

    const size_t n_plan = h_plan.size();
    gpuErrchk(hipMalloc(&d_plan, sizeof(ppnt::PingSpec) * n_plan));
    gpuErrchk(hipMalloc(&d_out, sizeof(ppnt::PingOut) * n_plan));
    gpuErrchk(hipMemcpy(d_plan, h_plan.data(), sizeof(ppnt::PingSpec) * n_plan, hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(d_out, h_out.data(), sizeof(ppnt::PingOut) * n_plan, hipMemcpyHostToDevice));

    // Shared buffers for DB kernels.
    const int sink_len = physical_grid_size * TARGET_BLOCKDIM_X;
    k1::dtype* d_sink = nullptr;
    gpuErrchk(hipMalloc(&d_sink, sizeof(k1::dtype) * sink_len));

    // Point lookup inputs.
    vector<int> h_query_indices(query_n);
    for (int i = 0; i < query_n; i++) {
        h_query_indices[i] = (i % 10 == 0) ? hot_idx_local : hot_idx_remote;
    }
    int* d_query_indices = nullptr;
    k1::dtype* d_query_out = nullptr;
    gpuErrchk(hipMalloc(&d_query_indices, sizeof(int) * query_n));
    gpuErrchk(hipMalloc(&d_query_out, sizeof(k1::dtype) * query_n));
    gpuErrchk(hipMemcpy(d_query_indices, h_query_indices.data(), sizeof(int) * query_n, hipMemcpyHostToDevice));

    // Vector distance inputs.
    vector<float> h_vectors((size_t)vec_rows * vec_dim);
    vector<float> h_query(vec_dim);
    mt19937 rng(7);
    uniform_real_distribution<float> dist(-1.f, 1.f);
    for (float& x : h_vectors) x = dist(rng);
    for (float& x : h_query) x = dist(rng);
    float *d_vectors = nullptr, *d_query = nullptr, *d_dists = nullptr;
    gpuErrchk(hipMalloc(&d_vectors, sizeof(float) * h_vectors.size()));
    gpuErrchk(hipMalloc(&d_query, sizeof(float) * h_query.size()));
    gpuErrchk(hipMalloc(&d_dists, sizeof(float) * (size_t)vec_rows));
    gpuErrchk(hipMemcpy(d_vectors, h_vectors.data(), sizeof(float) * h_vectors.size(), hipMemcpyHostToDevice));
    gpuErrchk(hipMemcpy(d_query, h_query.data(), sizeof(float) * h_query.size(), hipMemcpyHostToDevice));

    // Range filter output.
    unsigned int* d_count = nullptr;
    gpuErrchk(hipMalloc(&d_count, sizeof(unsigned int)));

    // Kernel 1: Hot entry fixed at HBM4 (remote).
    {
        dbq::HotReadUpdateArgs h_args{
            d_table, (int)db_len, hot_idx_remote, hot_idx_remote2,
            reader_xcd, writer_xcd, 1 << 22, 1, d_sink
        };
        dbq::HotReadUpdateArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(h_args)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(h_args), hipMemcpyHostToDevice));
        run_kernel_with_ping_localization("HotReadUpdate hot@HBM4", dbq::HotReadUpdateTargetFn{}, d_args,
                                          d_plan, d_out, h_plan, physical_grid_size, stream, clock_mhz);
        gpuErrchk(hipFree(d_args));
    }

    // Kernel 2: Simulated migration to HBM0.
    {
        dbq::HotReadUpdateArgs h_args{
            d_table, (int)db_len, hot_idx_local, hot_idx_local2,
            reader_xcd, writer_xcd, 1 << 22, 1, d_sink
        };
        dbq::HotReadUpdateArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(h_args)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(h_args), hipMemcpyHostToDevice));
        run_kernel_with_ping_localization("HotReadUpdate hot@HBM0", dbq::HotReadUpdateTargetFn{}, d_args,
                                          d_plan, d_out, h_plan, physical_grid_size, stream, clock_mhz);
        gpuErrchk(hipFree(d_args));
    }

    // Kernel 3: Point lookup (query engine index probe).
    {
        dbq::PointLookupArgs h_args{d_table, d_query_indices, query_n, d_query_out};
        dbq::PointLookupArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(h_args)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(h_args), hipMemcpyHostToDevice));
        run_kernel_with_ping_localization("PointLookup", dbq::PointLookupTargetFn{}, d_args,
                                          d_plan, d_out, h_plan, physical_grid_size, stream, clock_mhz);
        gpuErrchk(hipFree(d_args));
    }

    // Kernel 4: Vector DB nearest-neighbor primitive (distance pass).
    {
        dbq::VectorDistanceArgs h_args{d_vectors, d_query, d_dists, vec_rows, vec_dim};
        dbq::VectorDistanceArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(h_args)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(h_args), hipMemcpyHostToDevice));
        run_kernel_with_ping_localization("VectorDistance", dbq::VectorDistanceTargetFn{}, d_args,
                                          d_plan, d_out, h_plan, physical_grid_size, stream, clock_mhz);
        gpuErrchk(hipFree(d_args));
    }

    // Kernel 5: SQL-style range filter.
    {
        gpuErrchk(hipMemset(d_count, 0, sizeof(unsigned int)));
        dbq::RangeFilterArgs h_args{d_table, (int)db_len, 100, 300, d_count};
        dbq::RangeFilterArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(h_args)));
        gpuErrchk(hipMemcpy(d_args, &h_args, sizeof(h_args), hipMemcpyHostToDevice));
        run_kernel_with_ping_localization("RangeFilter", dbq::RangeFilterTargetFn{}, d_args,
                                          d_plan, d_out, h_plan, physical_grid_size, stream, clock_mhz);
        gpuErrchk(hipFree(d_args));
    }

    cout << "\n[NOTE] Compare ping ratio(remote HBM4) vs ratio(local HBM0) above for congestion localization.\n";
    return 0;
}
