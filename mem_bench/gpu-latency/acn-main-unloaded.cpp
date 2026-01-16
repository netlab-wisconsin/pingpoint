#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_complex.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sys/time.h>
#include <vector>

#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"

#include "acn.hpp"

using namespace std;

typedef int64_t dtype;

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#define PINNED_XCD  0
#ifndef PINNED_CC 
#define PINNED_CC   0
#endif

// NOTE: This is NVIDIA PTX; it is not valid on AMD. Keeping it only because your code had it.
// If you compile on AMD HIP, remove it or gate it properly.
__device__ unsigned int smid() {
    unsigned int r;
    asm("mov.u32 %0, %%smid;" : "=r"(r));
    return r;
}

template <typename T>
__global__ void pchase(T *buf, T *__restrict__ dummy_buf, int64_t N) {

    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    // printf("xcc_id: %u, se_id: %u, cu_id: %u\n", xcc_id, se_id, cu_id);
    
    // Constrain pchase thread to XCD 0
    if (xcc_id != PINNED_XCD) return;


    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    T *idx = buf;

    const int unroll_factor = 32;
#pragma unroll 1
    for (int64_t n = 0; n < N; n += unroll_factor) {
#pragma unroll
        for (int u = 0; u < unroll_factor; u++) {
            idx = (T *)*idx;
        }
    }

    if (tidx > 12313) {
        dummy_buf[0] = (T)idx;
    }
}

inline uint32_t get_cc(uint32_t xcc_id) {
    return (xcc_id / (XCD_NUM / CC_NUM)) % CC_NUM;
}

int main(int argc, char **argv) {
    printf("Pinned XCD: %d Pinned CC: %d\n", PINNED_XCD, PINNED_CC);

#ifdef __NVCC__
    GPU_ERROR(hipFuncSetAttribute(reinterpret_cast<const void *>(pchase<dtype>),
                                  hipFuncAttributePreferredSharedMemoryCarveout, 0));
#endif

    unsigned int clock = getGPUClock();

    const int cl_size = 128 / (int)sizeof(int64_t); // # of int64 in 128B
    const int skip_factor = 1;

    std::random_device rd;
    std::mt19937 g(rd());

    for (int64_t LEN = 16; LEN < (1 << 24); LEN = (int64_t)(LEN * 1.042) + 1 + rand() % 11) {

        if (LEN * skip_factor * cl_size * (int64_t)sizeof(dtype) > 120ll * 1024 * 1024) {
            LEN = (int64_t)(LEN * 1.1);
        }

        MeasurementSeries times;
        const int64_t iters = max(LEN, (int64_t)100000);

        for (int epoch = 0; epoch < 21; epoch++) {
            dtype *dbuf_base = nullptr;

            const size_t multiplicative_factor = CC_NUM * 1; // increase mult_factor if per-CC chunks size is too small
            const size_t n_dtype_dbuf =
                multiplicative_factor * (size_t)skip_factor * (size_t)cl_size * (size_t)LEN;

            GPU_ERROR(hipMalloc(&dbuf_base, n_dtype_dbuf * sizeof(dtype)));

            // Number of *cache lines* in dbuf, consistent with your (k % (skip_factor*cl_size)) grouping.
            const size_t n_cl_dbuf = n_dtype_dbuf / ((size_t)cl_size * (size_t)skip_factor);

            // Per-dtype CC identification.
            vector<uint32_t> dtype_home_cc(n_dtype_dbuf, (uint32_t)-1);
            vector<vector<uint32_t>> cc_dtypes(CC_NUM);

            // Per-cacheline CC identification.
            vector<uint32_t> cl_home_cc(n_cl_dbuf, (uint32_t)-1);
            vector<vector<uint32_t>> cc_cls(CC_NUM);

            {
                uint32_t *d_cycles = nullptr;
                GPU_ERROR(hipMalloc(&d_cycles, n_dtype_dbuf * (size_t)XCD_NUM * sizeof(uint32_t)));

                void *kernel_args[] = {
                    (void *)&dbuf_base,
                    (void *)&d_cycles,
                    (void *)&n_dtype_dbuf,
                };

                GPU_ERROR(hipLaunchCooperativeKernel(
                    identify_home<dtype>, dim3(1 * XCD_NUM), dim3(128 / sizeof(dtype)),
                    kernel_args, 0, 0));

                GPU_ERROR(hipDeviceSynchronize());

                vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(n_dtype_dbuf));
                for (int x = 0; x < XCD_NUM; x++) {
                    GPU_ERROR(hipMemcpy(h_cycles[x].data(),
                                        d_cycles + (size_t)x * n_dtype_dbuf,
                                        sizeof(uint32_t) * n_dtype_dbuf,
                                        hipMemcpyDeviceToHost));
                }

                for (size_t k = 0; k < n_dtype_dbuf; k++) {
                    uint32_t min_cycles = 0xFFFFFFFF;
                    int min_xcc = -1;
                    for (int x = 0; x < XCD_NUM; x++) {
                        uint32_t c = h_cycles[x][k];
                        if (c < min_cycles) {
                            min_cycles = c;
                            min_xcc = x;
                        }
#if DEBUG_LEVEL >= 3
                        cout << "dtype " << k << " xcc " << x << " cycles " << c << "\n";
#endif
                    }

                    uint32_t cc = get_cc((uint32_t)min_xcc);
                    dtype_home_cc[k] = cc;
                    cc_dtypes[cc].push_back((uint32_t)k);

                    if (k % ((size_t)skip_factor * (size_t)cl_size) == 0) {
                        size_t cl_idx = k / ((size_t)skip_factor * (size_t)cl_size);
                        if (cl_idx < n_cl_dbuf) {
                            cl_home_cc[cl_idx] = cc;
                            cc_cls[cc].push_back((uint32_t)cl_idx);
                        }
                    }
                }

                GPU_ERROR(hipFree(d_cycles));
            }

            #if DEBUG_LEVEL >= 1
            for (int cc = 0; cc < CC_NUM; cc++) {
                cout << "CC " << cc << " has " << cc_dtypes[cc].size() << " dtypes.\n";
                cout << "CC " << cc << " has " << cc_cls[cc].size() << " cache lines.\n";
            }
            #endif

            if (cc_cls[PINNED_CC].size() < (size_t)LEN) {
#if DEBUG_LEVEL >= 1
                cout << "Skipping epoch " << epoch << ": CC " << PINNED_CC
                     << " only has " << cc_cls[PINNED_CC].size()
                     << " cache lines, need LEN=" << LEN << "\n";
#endif
                GPU_ERROR(hipFree(dbuf_base));
                continue;
            }

            // Allocate buf for the FULL n_cl_dbuf domain (i.e., full dbuf domain).
            // In practice this means buf is sized to n_dtype_dbuf (the full dtype domain).
            dtype *buf = nullptr;
            GPU_ERROR(hipMallocManaged(&buf, n_dtype_dbuf * sizeof(dtype)));
            std::memset(buf, 0, n_dtype_dbuf * sizeof(dtype));

            dtype *dummy_buf = nullptr;
            GPU_ERROR(hipMallocManaged(&dummy_buf, sizeof(dtype)));
            dummy_buf[0] = 0;

            // Choose LEN cache lines from PINNED_CC, shuffle them
            vector<uint32_t> seq(LEN);
            for (int64_t i = 0; i < LEN; i++) {
                seq[i] = cc_cls[PINNED_CC][(size_t)i];
            }
            shuffle(seq.begin(), seq.end(), g);

#if DEBUG_LEVEL >= 2
            cout << "Access CL sequence (CL indices in full domain): ";
            for (int64_t i = 0; i < LEN; i++) cout << seq[(size_t)i] << " ";
            cout << "\n";
#endif

            // Build a cycle over those cache lines.
            // For each cache line, we write pointers for *all lanes* (cl_lane=0..cl_size-1),
            // so the CL is fully populated with valid pointer values.
            //
            // Element index for (cacheline cl_idx, lane cl_lane) is:
            //   elem = (cl_idx * cl_size + cl_lane) * skip_factor
            //
            // Stored value is an absolute device address:
            //   (uintptr_t)dbuf_base + next_elem * sizeof(dtype)
            //
            for (int cl_lane = 0; cl_lane < cl_size; cl_lane++) {
                for (int64_t i = 0; i < LEN; i++) {
                    uint32_t cur_cl  = seq[(size_t)i];
                    uint32_t next_cl = seq[(size_t)((i + 1) % LEN)];

                    size_t cur_elem  = ((size_t)cur_cl  * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;
                    size_t next_elem = ((size_t)next_cl * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;

                    // Bounds safety (should hold if n_cl_dbuf is consistent)
                    if (cur_elem >= n_dtype_dbuf || next_elem >= n_dtype_dbuf) {
                        cerr << "BUG: elem OOB: cur_elem=" << cur_elem
                             << " next_elem=" << next_elem
                             << " n_dtype_dbuf=" << n_dtype_dbuf << "\n";
                        GPU_ERROR(hipFree(buf));
                        GPU_ERROR(hipFree(dummy_buf));
                        GPU_ERROR(hipFree(dbuf_base));
                        return 1;
                    }

                    uintptr_t next_addr = (uintptr_t)dbuf_base + next_elem * sizeof(dtype);
                    buf[cur_elem] = (dtype)next_addr;

#if DEBUG_LEVEL >= 3
                    printf("cur_cl=%u lane=%d cur_elem=%zu -> next_cl=%u next_elem=%zu addr=%" PRIxPTR "\n",
                           cur_cl, cl_lane, cur_elem, next_cl, next_elem, next_addr);
#endif
                }
            }

            // Copy the full pointer table into device allocation.
            GPU_ERROR(hipMemcpy(dbuf_base, buf, n_dtype_dbuf * sizeof(dtype), hipMemcpyHostToDevice));
            GPU_ERROR(hipDeviceSynchronize());

            // Start pointer: lane 0 of the first cache line in seq[]
            size_t start_elem = ((size_t)seq[0] * (size_t)cl_size + 0) * (size_t)skip_factor;
            dtype *dbuf_start = dbuf_base + start_elem;

            // Warmup
            pchase<dtype><<<XCD_NUM, 1>>>(dbuf_start, dummy_buf, iters);
            pchase<dtype><<<XCD_NUM, 1>>>(dbuf_start, dummy_buf, iters);
            GPU_ERROR(hipDeviceSynchronize());

            hipEvent_t start, stop;
            GPU_ERROR(hipEventCreate(&start));
            GPU_ERROR(hipEventCreate(&stop));

            GPU_ERROR(hipEventRecord(start));
            pchase<dtype><<<XCD_NUM, 1>>>(dbuf_start, dummy_buf, iters); // 1 tb per xcd and prelude only xcd0 within the kernel
            GPU_ERROR(hipEventRecord(stop));

            GPU_ERROR(hipEventSynchronize(stop));
            float milliseconds = 0;
            GPU_ERROR(hipEventElapsedTime(&milliseconds, start, stop));

            times.add(milliseconds / 1000);

            GPU_ERROR(hipGetLastError());

            GPU_ERROR(hipEventDestroy(start));
            GPU_ERROR(hipEventDestroy(stop));

            GPU_ERROR(hipFree(buf));
            GPU_ERROR(hipFree(dummy_buf));
            GPU_ERROR(hipFree(dbuf_base));
        }

        double dt = times.value();
        double dtmed = times.median();
        double dtmin = times.getPercentile(0.05);
        double dtmax = times.getPercentile(0.95);

        cout << setw(9) << iters << " " << setw(5) << clock << " " //
             << setw(8) << skip_factor * LEN * cl_size * sizeof(dtype) / 1024.0 << " "
             << fixed << setprecision(1) << setw(8) << dt * 1000 << " " //
             << setw(7) << setprecision(1) << (double)dt / iters * clock * 1000 * 1000 << " "
             << (double)dtmed / iters * clock * 1000 * 1000 << " "
             << (double)dtmin / iters * clock * 1000 * 1000 << " "
             << (double)dtmax / iters * clock * 1000 * 1000 << "\n"
             << flush;
    }

    cout << "\n";
    return 0;
}
