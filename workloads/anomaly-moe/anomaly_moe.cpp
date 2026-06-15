// E3: adaptive two-tier monitoring of a path-localized decode-FFN anomaly.
//
// Eight decode experts are explicitly mapped one-to-one onto the eight XCDs. Each expert streams
// weight chunks home-identified to its corresponding HBM path. The request timeline keeps total
// tokens fixed while twice routing 57/64 tokens to one expert. Always-on pointer-chase latency
// pings detect each anomaly. The first diagnosis sweeps the informative bpx set twice; the second
// diagnoses with bpx=10 every fourth request.

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "../../mem_bench/gpu-clock.cuh"

#include "../moe/k1.h"
#include "../moe/k2.h"
#include "../moe/main.h"
#include "anomaly_moe.h"

using namespace std;
namespace cg = cooperative_groups;

static_assert(N_EXPERT == XCD_NUM);
static_assert(BLOCKDIM_X % WAVE_SIZE == 0);
static_assert(TARGET_CUS_PER_XCD + LATENCY_CUS_PER_XCD + MAX_BW_CUS_PER_XCD <= CU_NUM);

static constexpr int BYTES_PER_THREAD_ITER = 4 * 16;
static constexpr int GRID_SLOTS_PER_XCD =
    TARGET_CUS_PER_XCD + LATENCY_CUS_PER_XCD + MAX_BW_CUS_PER_XCD;
static constexpr int INFORMATIVE_BPX_ORDER[INFORMATIVE_BPX_COUNT] = {10, 16, 11, 15, 12, 14, 13};

__device__ __forceinline__
void stream_iteration(const uint64_t* const ptrs[4], size_t n_chunks,
                      size_t iter, int lane_slot, int n_lane_slots,
                      float& sink0, float& sink1, float& sink2, float& sink3) {
    const int tid = threadIdx.x;
    const int chunk_elems = CHUNK_SIZE / 16;
    const int groups = BLOCKDIM_X / chunk_elems;
    const int group = tid / chunk_elems;
    const int offset = tid % chunk_elems;
    const size_t ci = ((iter * (size_t)n_lane_slots + lane_slot) * groups + group) % n_chunks;
    float4* a0 = reinterpret_cast<float4*>(ptrs[0][ci]);
    float4* a1 = reinterpret_cast<float4*>(ptrs[1][ci]);
    float4* a2 = reinterpret_cast<float4*>(ptrs[2][ci]);
    float4* a3 = reinterpret_cast<float4*>(ptrs[3][ci]);
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
}

__global__ void anomaly_kernel(
    const PathData* __restrict__ paths,
    const int* __restrict__ expert_tokens,
    int latency_enabled, int bw_mode, int bw_path, int bw_bpx,
    int* __restrict__ target_done, int* __restrict__ stop_pings,
    uint64_t* __restrict__ target_max_cycles,
    uint64_t* __restrict__ target_xcd_cycles,
    LatencyStats* __restrict__ latency_stats,
    uint64_t* __restrict__ bw_block_iters,
    float* __restrict__ sink)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int local_slot = (bid / XCD_NUM) % GRID_SLOTS_PER_XCD;

    uint32_t xcc_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    const int path = (int)xcc_id;

    grid.sync();

    if (local_slot < TARGET_CUS_PER_XCD) {
        const uint64_t start = __builtin_readcyclecounter();
        const int token_count = expert_tokens[path];
        const size_t total_block_iters =
            (size_t)token_count * TARGET_BLOCK_ITERS_PER_TOKEN;
        const uint64_t* ptrs[4] = {
            paths[path].target[0], paths[path].target[1],
            paths[path].target[2], paths[path].target[3]
        };
        float sink0 = 0.0f, sink1 = 0.0f, sink2 = 0.0f, sink3 = 0.0f;
        for (size_t iter = local_slot; iter < total_block_iters;
             iter += TARGET_CUS_PER_XCD) {
            stream_iteration(ptrs, paths[path].n_target_chunks, iter, local_slot,
                             TARGET_CUS_PER_XCD, sink0, sink1, sink2, sink3);
        }
        __syncthreads();
        const uint64_t end = __builtin_readcyclecounter();
        if (tid == 0) {
            atomicMax((unsigned long long*)&target_xcd_cycles[path],
                      (unsigned long long)(end - start));
            atomicMax((unsigned long long*)target_max_cycles,
                      (unsigned long long)(end - start));
            const int completed = atomicAdd(target_done, 1);
            if (completed == TARGET_CUS_PER_XCD * XCD_NUM - 1)
                atomicExch(stop_pings, 1);
        }
        if (tid == 0)
            sink[bid] = sink0 + sink1 + sink2 + sink3;
        return;
    }

    if (local_slot == TARGET_CUS_PER_XCD) {
        if (!latency_enabled || tid != 0) return;
        int64_t* idx = paths[path].latency_start;
        uint64_t n = 0, sum = 0, sum_sq = 0, max_cycles = 0;
        while (!*(volatile int*)stop_pings) {
            const uint64_t start = __builtin_readcyclecounter();
            asm volatile("" ::: "memory");
            idx = (int64_t*)*idx;
            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t" ::: "memory");
            const uint64_t cycles = __builtin_readcyclecounter() - start;
            n++;
            sum += cycles;
            sum_sq += cycles * cycles;
            max_cycles = max(max_cycles, cycles);
        }
        latency_stats[path] = {n, sum, sum_sq, max_cycles};
        if ((uintptr_t)idx == 1) sink[bid] = 1.0f;
        return;
    }

    const int bw_slot = local_slot - TARGET_CUS_PER_XCD - LATENCY_CUS_PER_XCD;
    const bool active_path = bw_mode == 2 || (bw_mode == 1 && path == bw_path);
    if (!active_path || bw_slot >= bw_bpx) return;

    const uint64_t* ptrs[4] = {
        paths[path].probe[0], paths[path].probe[1],
        paths[path].probe[2], paths[path].probe[3]
    };
    float sink0 = 0.0f, sink1 = 0.0f, sink2 = 0.0f, sink3 = 0.0f;
    uint64_t iter = 0;
    while (true) {
        if ((iter % STOP_CHECK_ITERS) == 0 && *(volatile int*)stop_pings) break;
        stream_iteration(ptrs, paths[path].n_probe_chunks, iter, bw_slot, bw_bpx,
                         sink0, sink1, sink2, sink3);
        iter++;
    }
    if (tid == 0) {
        bw_block_iters[path * MAX_BW_CUS_PER_XCD + bw_slot] = iter;
        sink[bid] = sink0 + sink1 + sink2 + sink3;
    }
}

__global__ void solo_bw_kernel(
    const PathData* __restrict__ paths, int selected_path, int bpx, uint64_t duration_cycles,
    uint64_t* __restrict__ bw_block_iters, float* __restrict__ sink)
{
    cg::grid_group grid = cg::this_grid();
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int slot = (bid / XCD_NUM) % MAX_BW_CUS_PER_XCD;
    uint32_t xcc_id;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    const int path = (int)xcc_id;
    grid.sync();
    const uint64_t start = __builtin_readcyclecounter();
    if (path != selected_path || slot >= bpx) return;

    const uint64_t* ptrs[4] = {
        paths[path].probe[0], paths[path].probe[1],
        paths[path].probe[2], paths[path].probe[3]
    };
    float sink0 = 0.0f, sink1 = 0.0f, sink2 = 0.0f, sink3 = 0.0f;
    uint64_t iter = 0;
    while (__builtin_readcyclecounter() - start < duration_cycles) {
        stream_iteration(ptrs, paths[path].n_probe_chunks, iter, slot, bpx,
                         sink0, sink1, sink2, sink3);
        iter++;
    }
    if (tid == 0) {
        bw_block_iters[path * MAX_BW_CUS_PER_XCD + slot] = iter;
        sink[bid] = sink0 + sink1 + sink2 + sink3;
    }
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

static double percentile(vector<double> values, double p) {
    if (values.empty()) return 0.0;
    sort(values.begin(), values.end());
    const size_t index = min(values.size() - 1, (size_t)ceil(p * values.size()) - 1);
    return values[index];
}

struct PassResult {
    double target_ns = 0.0;
    double wall_ns = 0.0;
    vector<double> target_xcd_ns;
    vector<double> latency_mean_ns;
    vector<double> latency_cv;
    vector<double> latency_max_ns;
    vector<double> bw_gbps;
};

struct Detector {
    vector<double> baseline_mean_ns;
    double threshold = 0.0;
    double reset_threshold = 0.0;

    pair<double, int> score(const PassResult& result) const {
        vector<double> ratios(XCD_NUM);
        for (int x = 0; x < XCD_NUM; x++)
            ratios[x] = baseline_mean_ns[x] > 0.0
                ? result.latency_mean_ns[x] / baseline_mean_ns[x] : 0.0;
        const double center = median(ratios);
        int selected = 0;
        for (int x = 1; x < XCD_NUM; x++)
            if (ratios[x] > ratios[selected]) selected = x;
        return {ratios[selected] - center, selected};
    }
};

struct RequestRecord {
    string policy;
    int request = 0;
    string phase;
    int anomaly_id = 0;
    bool anomaly = false;
    bool detector_positive = false;
    bool triggered = false;
    bool bw_active = false;
    int encounter = 0;
    int diagnosis_step = -1;
    int bw_bpx = 0;
    int selected_path = -1;
    double score = 0.0;
    PassResult pass;
    vector<double> probe_loss_pct;
    vector<int> informative;
};

struct Controller {
    int detect_streak = 0;
    int reset_streak = 0;
    int selected_path = -1;
    int encounter = 0;
    int diagnosis_step = 0;
    int probe_index = 0;
    bool latched = false;

    struct Decision {
        bool active = false;
        int path = -1;
        int bpx = 0;
        int encounter = 0;
        int diagnosis_step = -1;
    };

    Decision decide() {
        Decision decision;
        if (!latched) return decision;

        decision.path = selected_path;
        decision.encounter = encounter;
        const int ping_limit =
            encounter == 1 ? FIRST_ANOMALY_PING_COUNT : SECOND_ANOMALY_PING_COUNT;
        if (probe_index >= ping_limit) return decision;
        decision.diagnosis_step = diagnosis_step;
        decision.active = encounter == 1 || diagnosis_step % SECOND_ANOMALY_INTERVAL == 0;
        diagnosis_step++;
        if (decision.active) {
            decision.bpx = encounter == 1
                ? INFORMATIVE_BPX_ORDER[probe_index % INFORMATIVE_BPX_COUNT]
                : SECOND_ANOMALY_BPX;
            probe_index++;
        }
        return decision;
    }

    bool update(double score, int path, const Detector& detector) {
        bool triggered = false;
        if (!latched) {
            if (score >= detector.threshold) {
                detect_streak++;
                if (detect_streak >= DETECT_STREAK) {
                    latched = true;
                    selected_path = path;
                    encounter++;
                    diagnosis_step = 0;
                    probe_index = 0;
                    reset_streak = 0;
                    triggered = true;
                }
            } else {
                detect_streak = 0;
            }
        } else {
            if (score <= detector.reset_threshold) {
                reset_streak++;
                if (reset_streak >= RESET_STREAK) {
                    latched = false;
                    selected_path = -1;
                    detect_streak = 0;
                    reset_streak = 0;
                }
            } else {
                reset_streak = 0;
            }
        }
        return triggered;
    }
};

int main(int argc, char** argv) {
    const int n_requests = argc > 1 ? atoi(argv[1]) : DEFAULT_REQUESTS;
    const int first_start = argc > 2 ? atoi(argv[2]) : DEFAULT_FIRST_ANOMALY_START;
    const int first_length = argc > 3 ? atoi(argv[3]) : DEFAULT_FIRST_ANOMALY_LENGTH;
    const int second_start = argc > 4 ? atoi(argv[4]) : DEFAULT_SECOND_ANOMALY_START;
    const int second_length = argc > 5 ? atoi(argv[5]) : DEFAULT_SECOND_ANOMALY_LENGTH;
    const int calibration_requests = getenv("ANOMALY_CALIBRATION_REQUESTS")
        ? atoi(getenv("ANOMALY_CALIBRATION_REQUESTS")) : DEFAULT_CALIBRATION_REQUESTS;
    if (n_requests <= 0 || first_start < 0 || first_length <= 0 ||
        second_start < first_start + first_length || second_length <= 0 ||
        second_start + second_length > n_requests ||
        calibration_requests < 5) {
        fprintf(stderr, "usage: %s [requests] [first_start] [first_length] "
                        "[second_start] [second_length]\n", argv[0]);
        fprintf(stderr, "calibration requests must be >= 5\n");
        return 1;
    }

    const unsigned int clock_mhz = getGPUClock();
    const double bytes_per_block_iter = (double)BLOCKDIM_X * BYTES_PER_THREAD_ITER;

    // Streaming data setup. Reuse E1/E2 home identification unchanged.
    const long long stream_data_size = (long long)STREAM_N_PAGES * PAGE_SIZE;
    const size_t n_chunks = stream_data_size / CHUNK_SIZE;
    vector<char*> d_stream_data(N_STREAM_DATAS);
    for (int i = 0; i < N_STREAM_DATAS; i++) {
        gpuErrchk(hipMalloc((void**)&d_stream_data[i], stream_data_size + 0x1000));
        d_stream_data[i] =
            (char*)(((uintptr_t)d_stream_data[i] & ~(uintptr_t)0x0fff) + 0x1000);
    }
    vector<vector<int>> h_home(N_STREAM_DATAS, vector<int>(n_chunks));
    vector<vector<size_t>> h_xcd_chunks_size(N_STREAM_DATAS, vector<size_t>(XCD_NUM, 0));
    if (k2::home_identification(d_stream_data, stream_data_size, n_chunks, N_STREAM_DATAS,
                                h_home, h_xcd_chunks_size) == -1)
        return 1;

    vector<uint64_t*> d_chunks_per_hbm(N_STREAM_DATAS);
    vector<vector<size_t>> h_offsets(N_STREAM_DATAS, vector<size_t>(XCD_NUM));
    for (int i = 0; i < N_STREAM_DATAS; i++) {
        vector<vector<uint64_t>> by_hbm(XCD_NUM);
        for (size_t chunk = 0; chunk < n_chunks; chunk++)
            by_hbm[h_home[i][chunk]].push_back((uint64_t)d_stream_data[i] + chunk * CHUNK_SIZE);
        gpuErrchk(hipMalloc((void**)&d_chunks_per_hbm[i], sizeof(uint64_t) * n_chunks));
        size_t offset = 0;
        for (int x = 0; x < XCD_NUM; x++) {
            h_offsets[i][x] = offset;
            gpuErrchk(hipMemcpy(&d_chunks_per_hbm[i][offset], by_hbm[x].data(),
                                sizeof(uint64_t) * by_hbm[x].size(), hipMemcpyHostToDevice));
            offset += by_hbm[x].size();
        }
    }

    // Latency pointer chains. The existing k1 classifier is used unchanged.
    const size_t latency_len = LAT_CHAIN_LINES_PER_HBM;
    const size_t cl_bytes = 128;
    const size_t cl_size = cl_bytes / sizeof(k1::dtype);
    const size_t skip_factor = 1;
    const size_t n_latency_dtypes = (size_t)XCD_NUM * cl_size * latency_len;
    const size_t n_latency_cls = n_latency_dtypes / cl_size;
    k1::dtype* d_latency_base = nullptr;
    gpuErrchk(hipMalloc(&d_latency_base, n_latency_dtypes * sizeof(k1::dtype)));
    vector<uint32_t> dtype_home(n_latency_dtypes, (uint32_t)-1);
    vector<uint32_t> cl_home(n_latency_cls, (uint32_t)-1);
    vector<vector<uint32_t>> hbm_dtypes(XCD_NUM), hbm_cls(XCD_NUM);
    if (k1::home_identification(d_latency_base, n_latency_dtypes, n_latency_cls, cl_size,
                                cl_bytes, skip_factor, dtype_home, cl_home,
                                hbm_dtypes, hbm_cls) == -1)
        return 1;
    size_t usable_latency_len = latency_len;
    for (int x = 0; x < XCD_NUM; x++)
        usable_latency_len = min(usable_latency_len, hbm_cls[x].size());
    if (usable_latency_len == 0) {
        fprintf(stderr, "home identification produced an empty latency path\n");
        return 1;
    }
    vector<k1::dtype> h_latency(n_latency_dtypes, 0);
    vector<k1::dtype*> latency_starts(XCD_NUM, nullptr);
    mt19937 latency_rng(0xE3001234u);
    for (int x = 0; x < XCD_NUM; x++) {
        vector<uint32_t> sequence(hbm_cls[x].begin(), hbm_cls[x].begin() + usable_latency_len);
        shuffle(sequence.begin(), sequence.end(), latency_rng);
        for (size_t i = 0; i < usable_latency_len; i++) {
            const size_t current = (size_t)sequence[i] * cl_size;
            const size_t next = (size_t)sequence[(i + 1) % usable_latency_len] * cl_size;
            h_latency[current] = (k1::dtype)((uintptr_t)d_latency_base + next * sizeof(k1::dtype));
        }
        latency_starts[x] = d_latency_base + (size_t)sequence[0] * cl_size;
    }
    gpuErrchk(hipMemcpy(d_latency_base, h_latency.data(),
                        sizeof(k1::dtype) * n_latency_dtypes, hipMemcpyHostToDevice));

    vector<PathData> h_paths(XCD_NUM);
    for (int x = 0; x < XCD_NUM; x++) {
        h_paths[x].n_probe_chunks = SIZE_MAX;
        h_paths[x].n_target_chunks = SIZE_MAX;
        for (int i = 0; i < 4; i++) {
            h_paths[x].probe[i] = d_chunks_per_hbm[i] + h_offsets[i][x];
            h_paths[x].target[i] = d_chunks_per_hbm[i + 4] + h_offsets[i + 4][x];
            h_paths[x].n_probe_chunks =
                min(h_paths[x].n_probe_chunks, h_xcd_chunks_size[i][x]);
            h_paths[x].n_target_chunks =
                min(h_paths[x].n_target_chunks, h_xcd_chunks_size[i + 4][x]);
        }
        h_paths[x].latency_start = latency_starts[x];
    }
    PathData* d_paths = nullptr;
    gpuErrchk(hipMalloc(&d_paths, sizeof(PathData) * XCD_NUM));
    gpuErrchk(hipMemcpy(d_paths, h_paths.data(), sizeof(PathData) * XCD_NUM,
                        hipMemcpyHostToDevice));

    int *d_expert_tokens = nullptr, *d_target_done = nullptr, *d_stop_pings = nullptr;
    uint64_t *d_target_max_cycles = nullptr, *d_target_xcd_cycles = nullptr;
    LatencyStats* d_latency_stats = nullptr;
    uint64_t* d_bw_block_iters = nullptr;
    float* d_sink = nullptr;
    const size_t bw_count_elems = XCD_NUM * MAX_BW_CUS_PER_XCD;
    gpuErrchk(hipMalloc(&d_expert_tokens, sizeof(int) * XCD_NUM));
    gpuErrchk(hipMalloc(&d_target_done, sizeof(int)));
    gpuErrchk(hipMalloc(&d_stop_pings, sizeof(int)));
    gpuErrchk(hipMalloc(&d_target_max_cycles, sizeof(uint64_t)));
    gpuErrchk(hipMalloc(&d_target_xcd_cycles, sizeof(uint64_t) * XCD_NUM));
    gpuErrchk(hipMalloc(&d_latency_stats, sizeof(LatencyStats) * XCD_NUM));
    gpuErrchk(hipMalloc(&d_bw_block_iters, sizeof(uint64_t) * bw_count_elems));
    gpuErrchk(hipMalloc(&d_sink, sizeof(float) * GRID_SLOTS_PER_XCD * XCD_NUM));

    hipStream_t stream;
    hipEvent_t event_start, event_end;
    gpuErrchk(hipStreamCreate(&stream));
    gpuErrchk(hipEventCreate(&event_start));
    gpuErrchk(hipEventCreate(&event_end));
    vector<uint64_t> h_target_xcd_cycles(XCD_NUM), h_bw_block_iters(bw_count_elems);
    vector<LatencyStats> h_latency_stats(XCD_NUM);

    auto run_pass = [&](const vector<int>& tokens, bool latency_enabled,
                        int bw_mode, int bw_path, int active_bpx) -> PassResult {
        gpuErrchk(hipMemcpyAsync(d_expert_tokens, tokens.data(), sizeof(int) * XCD_NUM,
                                 hipMemcpyHostToDevice, stream));
        gpuErrchk(hipMemsetAsync(d_target_done, 0, sizeof(int), stream));
        gpuErrchk(hipMemsetAsync(d_stop_pings, 0, sizeof(int), stream));
        gpuErrchk(hipMemsetAsync(d_target_max_cycles, 0, sizeof(uint64_t), stream));
        gpuErrchk(hipMemsetAsync(d_target_xcd_cycles, 0, sizeof(uint64_t) * XCD_NUM, stream));
        gpuErrchk(hipMemsetAsync(d_latency_stats, 0, sizeof(LatencyStats) * XCD_NUM, stream));
        gpuErrchk(hipMemsetAsync(d_bw_block_iters, 0,
                                 sizeof(uint64_t) * bw_count_elems, stream));
        int lat = latency_enabled ? 1 : 0;
        int bmode = bw_mode, bpath = bw_path, bpx = active_bpx;
        void* args[] = {
            &d_paths, &d_expert_tokens, &lat, &bmode, &bpath, &bpx,
            &d_target_done, &d_stop_pings, &d_target_max_cycles, &d_target_xcd_cycles,
            &d_latency_stats, &d_bw_block_iters, &d_sink
        };
        gpuErrchk(hipEventRecord(event_start, stream));
        gpuErrchk(hipLaunchCooperativeKernel((void*)anomaly_kernel,
                                             dim3(GRID_SLOTS_PER_XCD * XCD_NUM),
                                             dim3(BLOCKDIM_X), args, 0, stream));
        gpuErrchk(hipEventRecord(event_end, stream));
        gpuErrchk(hipStreamSynchronize(stream));

        float wall_ms = 0.0f;
        gpuErrchk(hipEventElapsedTime(&wall_ms, event_start, event_end));
        uint64_t target_cycles = 0;
        gpuErrchk(hipMemcpy(&target_cycles, d_target_max_cycles, sizeof(uint64_t),
                            hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(h_target_xcd_cycles.data(), d_target_xcd_cycles,
                            sizeof(uint64_t) * XCD_NUM, hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(h_latency_stats.data(), d_latency_stats,
                            sizeof(LatencyStats) * XCD_NUM, hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(h_bw_block_iters.data(), d_bw_block_iters,
                            sizeof(uint64_t) * bw_count_elems, hipMemcpyDeviceToHost));

        PassResult result;
        result.target_ns = (double)target_cycles / clock_mhz * 1e3;
        result.wall_ns = (double)wall_ms * 1e6;
        result.target_xcd_ns.resize(XCD_NUM);
        result.latency_mean_ns.assign(XCD_NUM, 0.0);
        result.latency_cv.assign(XCD_NUM, 0.0);
        result.latency_max_ns.assign(XCD_NUM, 0.0);
        result.bw_gbps.assign(XCD_NUM, 0.0);
        for (int x = 0; x < XCD_NUM; x++) {
            result.target_xcd_ns[x] = (double)h_target_xcd_cycles[x] / clock_mhz * 1e3;
            const LatencyStats& stat = h_latency_stats[x];
            if (stat.samples > 0) {
                const double mean_cycles = (double)stat.sum_cycles / stat.samples;
                const double mean_sq_cycles = (double)stat.sum_sq_cycles / stat.samples;
                const double variance = max(0.0, mean_sq_cycles - mean_cycles * mean_cycles);
                result.latency_mean_ns[x] = mean_cycles / clock_mhz * 1e3;
                result.latency_cv[x] = mean_cycles > 0.0 ? sqrt(variance) / mean_cycles : 0.0;
                result.latency_max_ns[x] = (double)stat.max_cycles / clock_mhz * 1e3;
            }
            uint64_t block_iters = 0;
            for (int slot = 0; slot < active_bpx; slot++)
                block_iters += h_bw_block_iters[x * MAX_BW_CUS_PER_XCD + slot];
            result.bw_gbps[x] = result.target_ns > 0.0
                ? block_iters * bytes_per_block_iter / result.target_ns : 0.0;
        }
        return result;
    };

    auto solo_bw = [&](int selected_path, int bpx) -> double {
        vector<double> samples;
        for (int rep = 0; rep < SOLO_BW_REPEATS + 1; rep++) {
            gpuErrchk(hipMemsetAsync(d_bw_block_iters, 0,
                                     sizeof(uint64_t) * bw_count_elems, stream));
            const uint64_t duration_cycles = (uint64_t)SOLO_BW_WINDOW_US * clock_mhz;
            int path_arg = selected_path, bpx_arg = bpx;
            void* args[] = {&d_paths, &path_arg, &bpx_arg, (void*)&duration_cycles,
                            &d_bw_block_iters, &d_sink};
            gpuErrchk(hipLaunchCooperativeKernel((void*)solo_bw_kernel,
                                                 dim3(MAX_BW_CUS_PER_XCD * XCD_NUM),
                                                 dim3(BLOCKDIM_X), args, 0, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            gpuErrchk(hipMemcpy(h_bw_block_iters.data(), d_bw_block_iters,
                                sizeof(uint64_t) * bw_count_elems, hipMemcpyDeviceToHost));
            uint64_t block_iters = 0;
            for (int slot = 0; slot < bpx; slot++)
                block_iters += h_bw_block_iters[selected_path * MAX_BW_CUS_PER_XCD + slot];
            if (rep > 0)
                samples.push_back(block_iters * bytes_per_block_iter /
                                  ((double)SOLO_BW_WINDOW_US * 1e3));
        }
        return median(samples);
    };

    printf("[anomaly-moe] requests=%d anomaly_1=[%d,%d) anomaly_2=[%d,%d) hot_expert=%d "
           "balanced_tokens=%d hot_tokens=%d/%d "
           "latency_lines_per_path=%zu clock=%uMHz\n",
           n_requests, first_start, first_start + first_length,
           second_start, second_start + second_length, HOT_EXPERT,
           BALANCED_TOKENS_PER_EXPERT, HOT_EXPERT_TOKENS, TOTAL_TOKENS,
           usable_latency_len, clock_mhz);
    printf("[bw_schedule] anomaly_1=order(10,16,11,15,12,14,13)x2 after trigger; "
           "anomaly_2=bpx10 every 4th request after trigger\n");
    printf("[mapping] expert_i -> xcd_i -> hbm_i; home identification reuses moe/k1.h and moe/k2.h\n");
    fflush(stdout);

    vector<vector<double>> solo_probe(XCD_NUM, vector<double>(MAX_BW_BPX + 1, 0.0));
    vector<double> peak_probe(XCD_NUM);
    for (int x = 0; x < XCD_NUM; x++) {
        for (int i = 0; i < INFORMATIVE_BPX_COUNT; i++) {
            const int bpx = INFORMATIVE_BPX_ORDER[i];
            solo_probe[x][bpx] = solo_bw(x, bpx);
        }
        peak_probe[x] = solo_probe[x][MAX_BW_BPX];
        for (int i = 0; i < INFORMATIVE_BPX_COUNT; i++) {
            const int bpx = INFORMATIVE_BPX_ORDER[i];
            printf("calibration_bw,path=%d,bpx=%d,solo_probe_gbps=%.2f,peak_bpx=%d,"
                   "peak_gbps=%.2f,reach_pct=%.2f\n",
                   x, bpx, solo_probe[x][bpx], MAX_BW_BPX, peak_probe[x],
                   peak_probe[x] > 0.0 ? solo_probe[x][bpx] / peak_probe[x] * 100.0 : 0.0);
        }
        fflush(stdout);
    }

    const vector<int> balanced_tokens(XCD_NUM, BALANCED_TOKENS_PER_EXPERT);
    vector<int> hot_tokens(XCD_NUM, COLD_EXPERT_TOKENS);
    hot_tokens[HOT_EXPERT] = HOT_EXPERT_TOKENS;
    assert(accumulate(balanced_tokens.begin(), balanced_tokens.end(), 0) == TOTAL_TOKENS);
    assert(accumulate(hot_tokens.begin(), hot_tokens.end(), 0) == TOTAL_TOKENS);

    for (int i = 0; i < DEFAULT_WARMUP_REQUESTS; i++)
        run_pass(balanced_tokens, true, 0, -1, 0);
    vector<PassResult> calibration;
    for (int i = 0; i < calibration_requests; i++)
        calibration.push_back(run_pass(balanced_tokens, true, 0, -1, 0));

    Detector detector;
    detector.baseline_mean_ns.resize(XCD_NUM);
    for (int x = 0; x < XCD_NUM; x++) {
        vector<double> samples;
        for (const PassResult& pass : calibration) samples.push_back(pass.latency_mean_ns[x]);
        detector.baseline_mean_ns[x] = median(samples);
    }
    vector<double> calibration_scores;
    for (const PassResult& pass : calibration)
        calibration_scores.push_back(detector.score(pass).first);
    const double score_p99 = percentile(calibration_scores, 0.99);
    detector.threshold = max(0.08, score_p99 * 1.50);
    detector.reset_threshold = detector.threshold * 0.50;
    printf("calibration_detector,n=%d,score_median=%.5f,score_p99=%.5f,"
           "threshold=%.5f,reset_threshold=%.5f\n",
           calibration_requests, median(calibration_scores), score_p99,
           detector.threshold, detector.reset_threshold);
    for (int x = 0; x < XCD_NUM; x++)
        printf("calibration_latency,path=%d,baseline_mean_ns=%.3f\n",
               x, detector.baseline_mean_ns[x]);
    fflush(stdout);

    const vector<string> policies = {"baseline", "latency", "always_bw", "adaptive"};
    vector<vector<RequestRecord>> records;
    auto request_phase = [&](int request) -> pair<string, int> {
        if (request < first_start) return {"normal_before", 0};
        if (request < first_start + first_length) return {"anomaly_1", 1};
        if (request < second_start) return {"recovery", 0};
        if (request < second_start + second_length) return {"anomaly_2", 2};
        return {"normal_after", 0};
    };
    for (const string& policy : policies) {
        for (int i = 0; i < DEFAULT_WARMUP_REQUESTS; i++) {
            const bool lat = policy != "baseline";
            const int bw_mode = policy == "always_bw" ? 2 : 0;
            const int warmup_bpx = policy == "always_bw"
                ? INFORMATIVE_BPX_ORDER[i % INFORMATIVE_BPX_COUNT] : 0;
            run_pass(balanced_tokens, lat, bw_mode, -1, warmup_bpx);
        }

        Controller controller;
        vector<RequestRecord> policy_records;
        for (int request = 0; request < n_requests; request++) {
            const auto [phase, anomaly_id] = request_phase(request);
            const bool anomaly = anomaly_id != 0;
            const vector<int>& tokens = anomaly ? hot_tokens : balanced_tokens;
            const bool latency_enabled = policy != "baseline";
            int bw_mode = 0, bw_path = -1, active_bpx = 0;
            Controller::Decision decision;
            if (policy == "always_bw") {
                bw_mode = 2;
                active_bpx = INFORMATIVE_BPX_ORDER[request % INFORMATIVE_BPX_COUNT];
            } else if (policy == "adaptive") {
                decision = controller.decide();
                if (decision.active) {
                    bw_mode = 1;
                    bw_path = decision.path;
                    active_bpx = decision.bpx;
                }
            }

            RequestRecord record;
            record.policy = policy;
            record.request = request;
            record.phase = phase;
            record.anomaly_id = anomaly_id;
            record.anomaly = anomaly;
            record.bw_active = bw_mode != 0;
            record.encounter = decision.encounter;
            record.diagnosis_step = decision.diagnosis_step;
            record.bw_bpx = active_bpx;
            record.selected_path = policy == "adaptive" ? decision.path : bw_path;
            record.pass = run_pass(tokens, latency_enabled, bw_mode, bw_path, active_bpx);
            record.probe_loss_pct.assign(XCD_NUM, 0.0);
            record.informative.assign(XCD_NUM, 0);
            for (int x = 0; x < XCD_NUM; x++) {
                const bool path_active = bw_mode == 2 || (bw_mode == 1 && x == bw_path);
                if (!path_active) continue;
                const double loss = solo_probe[x][active_bpx] > 0.0
                    ? max(0.0, 1.0 - record.pass.bw_gbps[x] / solo_probe[x][active_bpx]) : 0.0;
                record.probe_loss_pct[x] = loss * 100.0;
                record.informative[x] =
                    loss >= MIN_PROBE_LOSS_FRAC &&
                    solo_probe[x][active_bpx] >= peak_probe[x] * MIN_PROBE_PEAK_FRAC;
            }

            if (policy == "latency" || policy == "adaptive") {
                const auto [score, selected] = detector.score(record.pass);
                record.score = score;
                record.detector_positive = score >= detector.threshold;
                if (policy == "adaptive") {
                    record.triggered = controller.update(score, selected, detector);
                    if (record.triggered) {
                        record.selected_path = controller.selected_path;
                        record.encounter = controller.encounter;
                    } else if (record.selected_path < 0) {
                        record.selected_path = selected;
                    }
                } else {
                    record.selected_path = selected;
                }
            }
            policy_records.push_back(record);
        }
        records.push_back(policy_records);
    }

    const vector<RequestRecord>& baseline = records[0];
    printf("request,policy,request_idx,phase,ground_truth_anomaly,anomaly_id,hot_expert,"
           "detector_score,detector_positive,triggered,encounter,diagnosis_step,"
           "selected_path,selected_cc,bw_active,bw_bpx,target_ns,matched_baseline_target_ns,"
           "target_slowdown_pct,wall_ns,selected_bw_gbps,selected_bw_loss_pct,"
           "selected_bw_informative,total_bw_gbps\n");
    for (const auto& policy_records : records) {
        for (const RequestRecord& record : policy_records) {
            const double total_bw =
                accumulate(record.pass.bw_gbps.begin(), record.pass.bw_gbps.end(), 0.0);
            const int selected = record.bw_active && record.selected_path >= 0
                ? record.selected_path : -1;
            const double selected_bw = selected >= 0 ? record.pass.bw_gbps[selected] : 0.0;
            const double selected_loss = selected >= 0 ? record.probe_loss_pct[selected] : 0.0;
            const int selected_informative = selected >= 0 ? record.informative[selected] : 0;
            const double matched_baseline = baseline[record.request].pass.target_ns;
            const double target_slowdown =
                (record.pass.target_ns / matched_baseline - 1.0) * 100.0;
            printf("request,%s,%d,%s,%d,%d,%d,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,"
                   "%.2f,%.2f,%.3f,%.2f,%.2f,%.2f,%d,%.2f\n",
                   record.policy.c_str(), record.request, record.phase.c_str(),
                   record.anomaly ? 1 : 0, record.anomaly_id, HOT_EXPERT, record.score,
                   record.detector_positive ? 1 : 0, record.triggered ? 1 : 0,
                   record.encounter, record.diagnosis_step, record.selected_path,
                   record.selected_path >= 0 ? (int)get_cc(record.selected_path) : -1,
                   record.bw_active ? 1 : 0, record.bw_bpx, record.pass.target_ns,
                   matched_baseline, target_slowdown, record.pass.wall_ns, selected_bw,
                   selected_loss, selected_informative, total_bw);
        }
    }
    printf("path,policy,request_idx,phase,ground_truth_anomaly,anomaly_id,bw_bpx,path,"
           "target_xcd_ns,latency_mean_ns,latency_cv,latency_max_ns,bw_gbps,"
           "probe_loss_pct,informative\n");
    for (const auto& policy_records : records) {
        for (const RequestRecord& record : policy_records) {
            for (int x = 0; x < XCD_NUM; x++) {
                printf("path,%s,%d,%s,%d,%d,%d,%d,%.2f,%.3f,%.6f,%.3f,%.2f,%.2f,%d\n",
                       record.policy.c_str(), record.request, record.phase.c_str(),
                       record.anomaly ? 1 : 0, record.anomaly_id, record.bw_bpx, x,
                       record.pass.target_xcd_ns[x], record.pass.latency_mean_ns[x],
                       record.pass.latency_cv[x], record.pass.latency_max_ns[x],
                       record.pass.bw_gbps[x], record.probe_loss_pct[x], record.informative[x]);
            }
        }
    }

    printf("summary,policy,mean_target_slowdown_pct,mean_wall_slowdown_pct,"
           "normal_target_slowdown_pct,anomaly_target_slowdown_pct,bw_duty_cycle_pct,"
           "first_trigger,first_trigger_delay,first_trigger_path,first_trigger_cc_correct,"
           "second_trigger,second_trigger_delay,second_trigger_path,second_trigger_cc_correct,"
           "hot_cc,detector_tp,detector_fp,diagnosis_n,diagnosis_informative_pct,"
           "diagnosis_mean_loss_pct,diagnosis_unique_bpx,first_diagnosis_n,"
           "first_diagnosis_informative_pct,first_unique_bpx,second_diagnosis_n,"
           "second_diagnosis_informative_pct,second_unique_bpx\n");
    for (const auto& policy_records : records) {
        double target_slowdown = 0.0, wall_slowdown = 0.0;
        double normal_slowdown = 0.0, anomaly_slowdown = 0.0;
        int normal_n = 0, anomaly_n = 0, bw_n = 0;
        int detector_tp = 0, detector_fp = 0;
        vector<int> trigger_request(3, -1), trigger_path(3, -1);
        vector<int> encounter_n(3, 0), encounter_informative(3, 0);
        vector<vector<int>> encounter_bpx(3, vector<int>(MAX_BW_BPX + 1, 0));
        int diagnosis_n = 0, diagnosis_informative = 0;
        double diagnosis_loss = 0.0;
        for (int i = 0; i < n_requests; i++) {
            const RequestRecord& record = policy_records[i];
            const double ts = (record.pass.target_ns / baseline[i].pass.target_ns - 1.0) * 100.0;
            const double ws = (record.pass.wall_ns / baseline[i].pass.wall_ns - 1.0) * 100.0;
            target_slowdown += ts;
            wall_slowdown += ws;
            if (record.anomaly) {
                anomaly_slowdown += ts;
                anomaly_n++;
            } else {
                normal_slowdown += ts;
                normal_n++;
            }
            if (record.bw_active) bw_n++;
            if (record.detector_positive) {
                if (record.anomaly) detector_tp++;
                else detector_fp++;
            }
            if (record.triggered && record.encounter >= 1 && record.encounter <= 2) {
                trigger_request[record.encounter] = record.request;
                trigger_path[record.encounter] = record.selected_path;
            }
            if (record.bw_active && record.policy == "adaptive" && record.selected_path >= 0) {
                diagnosis_n++;
                diagnosis_informative += record.informative[record.selected_path];
                diagnosis_loss += record.probe_loss_pct[record.selected_path];
                if (record.encounter >= 1 && record.encounter <= 2) {
                    encounter_n[record.encounter]++;
                    encounter_informative[record.encounter] +=
                        record.informative[record.selected_path];
                    encounter_bpx[record.encounter][record.bw_bpx] = 1;
                }
            }
        }
        int unique_bpx = 0;
        vector<int> encounter_unique(3, 0);
        for (int bpx = 1; bpx <= MAX_BW_BPX; bpx++) {
            const bool seen = encounter_bpx[1][bpx] || encounter_bpx[2][bpx];
            unique_bpx += seen ? 1 : 0;
            encounter_unique[1] += encounter_bpx[1][bpx];
            encounter_unique[2] += encounter_bpx[2][bpx];
        }
        const int hot_cc = (int)get_cc(HOT_EXPERT);
        printf("summary,%s,%.3f,%.3f,%.3f,%.3f,%.3f,"
               "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.2f,%.2f,%d,%d,%.2f,%d,%d,%.2f,%d\n",
               policy_records[0].policy.c_str(),
               target_slowdown / n_requests, wall_slowdown / n_requests,
               normal_n ? normal_slowdown / normal_n : 0.0,
               anomaly_n ? anomaly_slowdown / anomaly_n : 0.0,
               100.0 * bw_n / n_requests,
               trigger_request[1],
               trigger_request[1] >= 0 ? trigger_request[1] - first_start : -1,
               trigger_path[1],
               trigger_path[1] >= 0 && (int)get_cc(trigger_path[1]) == hot_cc ? 1 : 0,
               trigger_request[2],
               trigger_request[2] >= 0 ? trigger_request[2] - second_start : -1,
               trigger_path[2],
               trigger_path[2] >= 0 && (int)get_cc(trigger_path[2]) == hot_cc ? 1 : 0,
               hot_cc, detector_tp, detector_fp, diagnosis_n,
               diagnosis_n ? 100.0 * diagnosis_informative / diagnosis_n : 0.0,
               diagnosis_n ? diagnosis_loss / diagnosis_n : 0.0, unique_bpx,
               encounter_n[1],
               encounter_n[1] ? 100.0 * encounter_informative[1] / encounter_n[1] : 0.0,
               encounter_unique[1], encounter_n[2],
               encounter_n[2] ? 100.0 * encounter_informative[2] / encounter_n[2] : 0.0,
               encounter_unique[2]);
    }

    printf("phase_summary,policy,phase,n,mean_target_slowdown_pct,bw_n,bw_duty_cycle_pct,"
           "diagnosis_informative_pct,diagnosis_unique_bpx\n");
    const vector<string> phases =
        {"normal_before", "anomaly_1", "recovery", "anomaly_2", "normal_after"};
    for (const auto& policy_records : records) {
        for (const string& phase : phases) {
            int n = 0, bw_n = 0, informative_n = 0;
            double slowdown = 0.0;
            vector<int> seen_bpx(MAX_BW_BPX + 1, 0);
            for (const RequestRecord& record : policy_records) {
                if (record.phase != phase) continue;
                n++;
                slowdown +=
                    (record.pass.target_ns / baseline[record.request].pass.target_ns - 1.0) * 100.0;
                if (record.bw_active) {
                    bw_n++;
                    seen_bpx[record.bw_bpx] = 1;
                    if (record.policy == "adaptive" && record.selected_path >= 0)
                        informative_n += record.informative[record.selected_path];
                }
            }
            if (n == 0) continue;
            const int unique_bpx = accumulate(seen_bpx.begin(), seen_bpx.end(), 0);
            printf("phase_summary,%s,%s,%d,%.3f,%d,%.3f,%.2f,%d\n",
                   policy_records[0].policy.c_str(), phase.c_str(), n,
                   n ? slowdown / n : 0.0, bw_n, n ? 100.0 * bw_n / n : 0.0,
                   bw_n ? 100.0 * informative_n / bw_n : 0.0, unique_bpx);
        }
    }

    printf("diagnosis_summary,encounter,configured_rate,trigger_request,completion_request,"
           "diagnosis_span_requests,ping_n,anomaly_ping_n,anomaly_window_bw_duty_pct,"
           "unique_bpx,informative_pct,mean_pinged_target_slowdown_pct,"
           "anomaly_phase_mean_target_slowdown_pct\n");
    const vector<RequestRecord>& adaptive = records[3];
    for (int encounter = 1; encounter <= 2; encounter++) {
        int trigger = -1, completion = -1, ping_n = 0, anomaly_ping_n = 0, informative_n = 0;
        double pinged_slowdown = 0.0, phase_slowdown = 0.0;
        int phase_n = 0;
        vector<int> seen_bpx(MAX_BW_BPX + 1, 0);
        for (const RequestRecord& record : adaptive) {
            if (record.triggered && record.encounter == encounter) trigger = record.request;
            if (record.anomaly_id == encounter) {
                phase_n++;
                phase_slowdown +=
                    (record.pass.target_ns / baseline[record.request].pass.target_ns - 1.0) * 100.0;
            }
            if (!record.bw_active || record.encounter != encounter) continue;
            completion = record.request;
            ping_n++;
            if (record.anomaly_id == encounter) anomaly_ping_n++;
            seen_bpx[record.bw_bpx] = 1;
            informative_n += record.informative[record.selected_path];
            pinged_slowdown +=
                (record.pass.target_ns / baseline[record.request].pass.target_ns - 1.0) * 100.0;
        }
        const int span = trigger >= 0 && completion >= 0 ? completion - trigger : 0;
        const int unique_bpx = accumulate(seen_bpx.begin(), seen_bpx.end(), 0);
        const double configured_rate =
            encounter == 1 ? 1.0 : 1.0 / SECOND_ANOMALY_INTERVAL;
        printf("diagnosis_summary,%d,%.2f,%d,%d,%d,%d,%d,%.2f,%d,%.2f,%.3f,%.3f\n",
               encounter, configured_rate, trigger, completion, span, ping_n,
               anomaly_ping_n, phase_n ? 100.0 * anomaly_ping_n / phase_n : 0.0, unique_bpx,
               ping_n ? 100.0 * informative_n / ping_n : 0.0,
               ping_n ? pinged_slowdown / ping_n : 0.0,
               phase_n ? phase_slowdown / phase_n : 0.0);
    }
    return 0;
}
