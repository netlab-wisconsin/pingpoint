#include "hip/hip_runtime.h"
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

__device__ unsigned int smid() {
  unsigned int r;

  asm("mov.u32 %0, %%smid;" : "=r"(r));

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

#ifdef __NVCC__
  GPU_ERROR(hipFuncSetAttribute(reinterpret_cast<const void*>(
      pchase<dtype>), hipFuncAttributePreferredSharedMemoryCarveout, 0));
#endif
  unsigned int clock = getGPUClock();

  const int cl_size = 128 / sizeof(int64_t);
  const int skip_factor = 1;

  std::random_device rd;
  std::mt19937 g(rd());

  for (int64_t LEN = 16; LEN < (1 << 24); LEN = LEN * 1.042 + 1 + rand() % 11) {
    if (LEN * skip_factor * cl_size * sizeof(dtype) > 120 * 1024 * 1024)
      LEN *= 1.1;

    MeasurementSeries times;
    const int64_t iters = max(LEN, (int64_t)100000);

    for (int i = 0; i < 21; i++) {

      vector<int64_t> order(LEN);
      int64_t *buf = NULL;
      int64_t *dbuf = NULL;
      dtype *dummy_buf = NULL;

      GPU_ERROR(
          hipMallocManaged(&buf, skip_factor * cl_size * LEN * sizeof(dtype)));
      GPU_ERROR(hipMalloc(&dbuf, skip_factor * cl_size * LEN * sizeof(dtype)));
      GPU_ERROR(hipMallocManaged(&dummy_buf, sizeof(dtype)));
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

      pchase<dtype><<<1, 1>>>(buf, dummy_buf, iters);
      pchase<dtype><<<1, 1>>>(buf, dummy_buf, iters);

      hipEvent_t start, stop;
      GPU_ERROR(hipEventCreate(&start));
      GPU_ERROR(hipEventCreate(&stop));

      GPU_ERROR(hipDeviceSynchronize());

      GPU_ERROR(hipEventRecord(start));
      pchase<dtype><<<1, 1>>>(buf, dummy_buf, iters);
      GPU_ERROR(hipEventRecord(stop));

      GPU_ERROR(hipEventSynchronize(stop));
      float milliseconds = 0;
      GPU_ERROR(hipEventElapsedTime(&milliseconds, start, stop));

      times.add(milliseconds / 1000);

      GPU_ERROR(hipGetLastError());
      GPU_ERROR(hipFree(buf));
      GPU_ERROR(hipFree(dbuf));
      GPU_ERROR(hipFree(dummy_buf));
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
  cout << "\n";
}
