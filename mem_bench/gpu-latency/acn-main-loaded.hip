#include "hip/hip_runtime.h"
#include <cerrno>
#include <cstdlib>
#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
// #include <algorithm>
#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/time.h>

using namespace std;

typedef int64_t dtype;

#ifndef BPX
#define BPX 304
#endif

// ----------------------------------------------------------------------------------
// BACKGROUND TRAFFIC KERNEL
// ----------------------------------------------------------------------------------
// Loops 'repeats' times over the buffer to generate memory traffic.
// We cast to float4 to maximize bandwidth (128-bit loads).
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

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int64_t *idx = buf;

  const int unroll_factor = 32;
#pragma unroll 1
  for (int64_t n = 0; n < N; n += unroll_factor) {
#pragma unroll
    for (int u = 0; u < unroll_factor; u++) {
      idx = (int64_t *)*idx;
    }
  }

  if (tidx > 12313) {
    dummy_buf[0] = (int64_t)idx;
  }
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
      std::cerr << "Invalid GPU device index from env (GPU_DEVICE/HIP_DEVICE): "
                << gpu_env << ", available devices: " << device_count << "\n";
      return 1;
    }
    GPU_ERROR(hipSetDevice((int)requested_device));
    std::cout << "Using HIP device " << requested_device << " from env\n";
  }

  printf("Background Traffic Launches %d Blocks\n", BPX);

#ifdef __NVCC__
  GPU_ERROR(hipFuncSetAttribute(reinterpret_cast<const void*>(
      pchase<dtype>), hipFuncAttributePreferredSharedMemoryCarveout, 0));
#endif
  unsigned int clock = getGPUClock();

  const int cl_size = 128 / sizeof(int64_t);
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

  for (int64_t LEN = 16; LEN < (1 << 24); LEN = LEN * 1.042 + 1 + rand() % 11) {
    
    //if (LEN * skip_factor * cl_size * sizeof(dtype) > 120 * 1024 * 1024) {
    //  LEN *= 1.1;
    //}  

    // @june (01/13) for faster exp
    if (LEN * skip_factor * cl_size * (int64_t)sizeof(dtype) > 4ll * 1024 * 1024) {
      LEN = (int64_t)(LEN * 1.2);
    }
    
    MeasurementSeries times;
    const int64_t iters = max(LEN, (int64_t)100000);

    for (int i = 0; i < 21; i++) {

      vector<int64_t> order(LEN);
      int64_t *buf = NULL;
      int64_t *dbuf = NULL;
      dtype *dummy_buf = NULL;
      
      // Allocations
      GPU_ERROR(hipMallocManaged(&buf, skip_factor * cl_size * LEN * sizeof(dtype)));
      GPU_ERROR(hipMalloc(&dbuf, skip_factor * cl_size * LEN * sizeof(dtype)));
      GPU_ERROR(hipMallocManaged(&dummy_buf, sizeof(dtype)));
      
      // Data Init
      for (int64_t i = 0; i < LEN; i++) {
        order[i] = i + 1;
      }
      order[LEN - 1] = 0;

      shuffle(begin(order), end(order) - 1, g);
      for (int cl_lane = 0; cl_lane < cl_size; cl_lane++) {
        dtype idx = 0;
        for (int64_t i = 0; i < LEN; i++) {

          buf[(idx * cl_size + cl_lane) * skip_factor] =
              skip_factor *
              (order[i] * cl_size + cl_lane + (order[i] == 0 ? 1 : 0));
          idx = order[i];
        }
      }
      buf[skip_factor * (order[LEN - 2] * cl_size + cl_size - 1)] = 0;
      for (int64_t n = 0; n < LEN * cl_size * skip_factor; n++) {
        buf[n] = (int64_t)dbuf + buf[n] * sizeof(int64_t *);
      }

      GPU_ERROR(hipMemcpy(dbuf, buf,
                           skip_factor * cl_size * LEN * sizeof(dtype),
                           hipMemcpyHostToDevice));
      
      // Warmup
      pchase<dtype><<<1, 1>>>(dbuf, dummy_buf, iters);
      pchase<dtype><<<1, 1>>>(dbuf, dummy_buf, iters);
      GPU_ERROR(hipDeviceSynchronize());

      hipEvent_t start, stop;
      GPU_ERROR(hipEventCreate(&start));
      GPU_ERROR(hipEventCreate(&stop));

      // ----------------------------------------------------------------------------------
      // EXECUTION PHASE
      // ----------------------------------------------------------------------------------
      size_t n_dtypes = skip_factor * cl_size * LEN;
      
      // 1. Launch Background Noise (Stream BW - Low Priority)
      // Launches huge work (iters * 10) to ensure it outlives pchase.
      dim3 bg_grid(BPX);
      dim3 bg_block(1024);
      k_bg<<<bg_grid, bg_block, 0, stream_bw>>>(dbuf, n_dtypes, iters * 10);

      // 2. Launch Latency Measurement (Stream Latency - High Priority)
      GPU_ERROR(hipEventRecord(start, stream_latency));
      pchase<dtype><<<1, 1, 0, stream_latency>>>(dbuf, dummy_buf, iters);
      GPU_ERROR(hipEventRecord(stop, stream_latency));
      
      // 3. Wait for Latency Kernel to finish
      GPU_ERROR(hipStreamSynchronize(stream_latency));

      // 4. Cleanup Background
      // We wait for the background loops to finish
      GPU_ERROR(hipStreamSynchronize(stream_bw));
      // ----------------------------------------------------------------------------------

      float milliseconds = 0;
      GPU_ERROR(hipEventElapsedTime(&milliseconds, start, stop));

      times.add(milliseconds / 1000);

      GPU_ERROR(hipGetLastError());
      GPU_ERROR(hipFree(buf));
      GPU_ERROR(hipFree(dbuf));
      GPU_ERROR(hipFree(dummy_buf));
      
      GPU_ERROR(hipEventDestroy(start));
      GPU_ERROR(hipEventDestroy(stop));
    }
    double dt = times.value();
    double dtmed = times.median();
    double dtmin = times.getPercentile(0.05);
    double dtmax = times.getPercentile(0.95);
    cout << setw(9) << iters << " " << setw(5) << clock << " " //
         << setw(8) << skip_factor * LEN * cl_size * sizeof(dtype) / 1024.0
         << " "                                            //
         << fixed                                          //
         << setprecision(1) << setw(8) << dt * 1000 << " " //
         << setw(7) << setprecision(1)
         << (double)dt / iters * clock * 1000 * 1000 << " "
     
         << (double)dtmed / iters * clock * 1000 * 1000 << " "
         << (double)dtmin / iters * clock * 1000 * 1000 << " "
         << (double)dtmax / iters * clock * 1000 * 1000 << "\n"
         << flush;
  }
  
  GPU_ERROR(hipStreamDestroy(stream_latency));
  GPU_ERROR(hipStreamDestroy(stream_bw));
  
  cout << "\n";
  return 0;
}
