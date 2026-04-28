#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_complex.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cinttypes>
#include <cstdlib>
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

#if (PINNED_CC < 0) || (PINNED_CC >= CC_NUM)
#error "PINNED_CC must be in [0, CC_NUM-1] for current GPU topology."
#endif

#ifndef BPX
#define BPX 304
#endif

// ----------------------------------------------------------------------------------
// BACKGROUND TRAFFIC KERNEL
// ----------------------------------------------------------------------------------
// Simply loops 'repeats' times over the buffer.
// We select 'repeats' to be large enough to outlast the pchase kernel.
__global__ void k_bg(dtype * __restrict__ buffer, size_t n_dtypes, int repeats) {
    float4 * ptr = reinterpret_cast<float4*>(buffer);
    size_t n_float4 = (n_dtypes * sizeof(dtype)) / sizeof(float4);

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    float4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};

    // Outer loop for duration control
    for (int r = 0; r < repeats; ++r) {
        // Grid-stride loop for Bandwidth Saturation
        for (size_t i = tid; i < n_float4; i += stride) {
            float4 val = ptr[i];
            accumulator.x += val.x;
            accumulator.y += val.y; 
            accumulator.z += val.z;
            accumulator.w += val.w;
        }
    }

    // Side effect to prevent dead code elimination
    if (accumulator.x == 12345678.9f) {
        ptr[0] = accumulator;
    }
}
// ----------------------------------------------------------------------------------

__device__ unsigned int smid() {
    unsigned int r;
    #ifdef __HIP_PLATFORM_AMD__
    r = 0; 
    #else
    asm("mov.u32 %0, %%smid;" : "=r"(r));
    #endif
    return r;
}

template <typename T>
__global__ void pchase(T *buf, T *__restrict__ dummy_buf, int64_t N) {
    uint32_t xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    
    // Only run on the pinned XCD
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
    const char* gpu_env = std::getenv("GPU_DEVICE");
    if (!gpu_env) gpu_env = std::getenv("HIP_DEVICE");
    if (gpu_env && gpu_env[0] != '\0') {
        int device_count = 0;
        GPU_ERROR(hipGetDeviceCount(&device_count));
        errno = 0;
        char* endptr = nullptr;
        long requested_device = std::strtol(gpu_env, &endptr, 10);
        if (errno != 0 || endptr == gpu_env || *endptr != '\0' ||
            requested_device < 0 || requested_device >= device_count) {
            cerr << "Invalid GPU device index from env (GPU_DEVICE/HIP_DEVICE): "
                 << gpu_env << ", available devices: " << device_count << "\n";
            return 1;
        }
        GPU_ERROR(hipSetDevice((int)requested_device));
        cout << "Using HIP device " << requested_device << " from env\n";
    }

    printf("Pinned XCD: %d Pinned CC: %d\n", PINNED_XCD, PINNED_CC);
    printf("Background Traffic Launches %d Blocks\n", BPX);

#ifdef __NVCC__
    GPU_ERROR(hipFuncSetAttribute(reinterpret_cast<const void *>(pchase<dtype>),
                                  hipFuncAttributePreferredSharedMemoryCarveout, 0));
#endif

    // unsigned int clock = getGPUClock();
    unsigned int clock = 2200; // set explicitly for mi350x
    
    const int cl_size = 128 / (int)sizeof(int64_t);
    const int skip_factor = 1;

    std::random_device rd;
    std::mt19937 g(rd());

    // -------------------------------------------------------------------------
    // STREAM SETUP
    // -------------------------------------------------------------------------
    hipStream_t stream_latency, stream_bw;
    
    // Use priorities to ensure the Latency kernel can start even if BG is heavy
    int priority_high, priority_low;
    GPU_ERROR(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    
    // stream_latency = High Priority
    // stream_bw      = Low Priority
    GPU_ERROR(hipStreamCreateWithPriority(&stream_latency, hipStreamNonBlocking, priority_high));
    GPU_ERROR(hipStreamCreateWithPriority(&stream_bw, hipStreamNonBlocking, priority_low));
    // -------------------------------------------------------------------------

    for (int64_t LEN = 16; LEN < (1 << 24); LEN = (int64_t)(LEN * 1.042) + 1 + rand() % 11) {

        // if (LEN * skip_factor * cl_size * (int64_t)sizeof(dtype) > 120ll * 1024 * 1024) {
        //     LEN = (int64_t)(LEN * 1.1);
        // }

        // @june (01/13) for faster exp
        if (LEN * skip_factor * cl_size * (int64_t)sizeof(dtype) > 4ll * 1024 * 1024) {
            LEN = (int64_t)(LEN * 1.2);
        }

        MeasurementSeries times;
        const int64_t iters = max(LEN, (int64_t)100000);

        for (int epoch = 0; epoch < 21; epoch++) {
            dtype *dbuf_base = nullptr;
            const size_t multiplicative_factor = CC_NUM * 1; 
            const size_t n_dtype_dbuf = multiplicative_factor * (size_t)skip_factor * (size_t)cl_size * (size_t)LEN;

            GPU_ERROR(hipMalloc(&dbuf_base, n_dtype_dbuf * sizeof(dtype)));
            const size_t n_cl_dbuf = n_dtype_dbuf / ((size_t)cl_size * (size_t)skip_factor);

            // --- Home Identification Logic (Simplified Copy) ---
             vector<uint32_t> dtype_home_cc(n_dtype_dbuf, (uint32_t)-1);
             vector<vector<uint32_t>> cc_dtypes(CC_NUM);
             vector<uint32_t> cl_home_cc(n_cl_dbuf, (uint32_t)-1);
             vector<vector<uint32_t>> cc_cls(CC_NUM);
             {
                 uint32_t *d_cycles = nullptr;
                 GPU_ERROR(hipMalloc(&d_cycles, n_dtype_dbuf * XCD_NUM * sizeof(uint32_t)));
                 void *args[] = {(void *)&dbuf_base, (void *)&d_cycles, (void *)&n_dtype_dbuf};
                 GPU_ERROR(hipLaunchCooperativeKernel(identify_home<dtype>, dim3(XCD_NUM), dim3(128/sizeof(dtype)), args, 0, 0));
                 GPU_ERROR(hipDeviceSynchronize());
                 vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(n_dtype_dbuf));
                 for(int x=0;x<XCD_NUM;x++) GPU_ERROR(hipMemcpy(h_cycles[x].data(), d_cycles+x*n_dtype_dbuf, sizeof(uint32_t)*n_dtype_dbuf, hipMemcpyDeviceToHost));
                 for(size_t k=0; k<n_dtype_dbuf; k++){
                     uint32_t m=UINT32_MAX; int mx=-1;
                     for(int x=0;x<XCD_NUM;x++) if(h_cycles[x][k]<m){m=h_cycles[x][k]; mx=x;}
                     uint32_t cc=get_cc(mx);
                     dtype_home_cc[k]=cc; cc_dtypes[cc].push_back(k);
                     if(k%(skip_factor*cl_size)==0) {
                         size_t cidx=k/(skip_factor*cl_size);
                         if(cidx<n_cl_dbuf) {cl_home_cc[cidx]=cc; cc_cls[cc].push_back(cidx);}
                     }
                 }
                 GPU_ERROR(hipFree(d_cycles));
             }
            // ------------------------------------------------

            if (cc_cls[PINNED_CC].size() < (size_t)LEN) {
                GPU_ERROR(hipFree(dbuf_base));
                continue;
            }

            dtype *buf = nullptr;
            GPU_ERROR(hipMallocManaged(&buf, n_dtype_dbuf * sizeof(dtype)));
            memset(buf, 0, n_dtype_dbuf * sizeof(dtype));
            dtype *dummy_buf = nullptr;
            GPU_ERROR(hipMallocManaged(&dummy_buf, sizeof(dtype)));
            dummy_buf[0] = 0;

            vector<uint32_t> seq(LEN);
            for(int64_t i=0; i<LEN; i++) seq[i] = cc_cls[PINNED_CC][i];
            shuffle(seq.begin(), seq.end(), g);

            // Link List
            for (int cl_lane = 0; cl_lane < cl_size; cl_lane++) {
                for (int64_t i = 0; i < LEN; i++) {
                    uint32_t cur = seq[i];
                    uint32_t nxt = seq[(i+1)%LEN];
                    size_t c_elem = (cur*cl_size + cl_lane)*skip_factor;
                    size_t n_elem = (nxt*cl_size + cl_lane)*skip_factor;
                    buf[c_elem] = (dtype)((uintptr_t)dbuf_base + n_elem*sizeof(dtype));
                }
            }
            GPU_ERROR(hipMemcpy(dbuf_base, buf, n_dtype_dbuf*sizeof(dtype), hipMemcpyHostToDevice));
            GPU_ERROR(hipDeviceSynchronize());

            size_t start_elem = (seq[0]*cl_size)*skip_factor;
            dtype *dbuf_start = dbuf_base + start_elem;

            // Warmup
            pchase<dtype><<<XCD_NUM, 1>>>(dbuf_start, dummy_buf, iters);
            GPU_ERROR(hipDeviceSynchronize());

            hipEvent_t start, stop;
            GPU_ERROR(hipEventCreate(&start));
            GPU_ERROR(hipEventCreate(&stop));

            // ----------------------------------------------------------------------------------
            // EXECUTION PHASE
            // ----------------------------------------------------------------------------------

            // 1. Launch Background Noise (Stream BW - Low Priority)
            // Launches huge work (iters * 10) to ensure it outlives pchase.
            dim3 bg_grid(BPX);
            dim3 bg_block(1024);
            k_bg<<<bg_grid, bg_block, 0, stream_bw>>>(dbuf_base, n_dtype_dbuf, iters * 10);

            // 2. Launch Latency Measurement (Stream Latency - High Priority)
            GPU_ERROR(hipEventRecord(start, stream_latency));
            pchase<dtype><<<XCD_NUM, 1, 0, stream_latency>>>(dbuf_start, dummy_buf, iters); 
            GPU_ERROR(hipEventRecord(stop, stream_latency));

            // 3. Wait for Latency Kernel to finish
            GPU_ERROR(hipStreamSynchronize(stream_latency));

            // 4. Cleanup Background
            // We just wait for it to finish its loops. 
            GPU_ERROR(hipStreamSynchronize(stream_bw));

            // ----------------------------------------------------------------------------------

            float milliseconds = 0;
            GPU_ERROR(hipEventElapsedTime(&milliseconds, start, stop));
            times.add(milliseconds / 1000);

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

    GPU_ERROR(hipStreamDestroy(stream_latency));
    GPU_ERROR(hipStreamDestroy(stream_bw));

    return 0;
}
