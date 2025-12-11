#include "hip/hip_runtime.h"
#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
#include "../gpu-metrics/gpu-metrics.hpp"

#include <iomanip>
#include <iostream>

using namespace std;

#ifdef __NVCC__
using dtype = float;
#else
using dtype = float4;
#endif

dtype *dA, *dB;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = (dtype)1.1;
  }
}

template <int N, int iters, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int zero) {
  dtype localSum = (dtype)0;

  B += threadIdx.x;

#pragma unroll N / BLOCKSIZE> 32   ? 1 : 32 / (N / BLOCKSIZE)
  for (int iter = 0; iter < iters; iter++) {
    B += zero;
    auto B2 = B + N;
#pragma unroll N / BLOCKSIZE >= 64 ? 32 : N / BLOCKSIZE
    for (int i = 0; i < N; i += BLOCKSIZE) {
      localSum += B[i] * B2[i];
    }
    localSum *= (dtype)1.3;
  }
  if (localSum == (dtype)1233)
    A[threadIdx.x] += localSum;
}

template <int N, int iters, int blockSize> double callKernel(int blockCount) {
  sumKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB, 0);
  return 0.0;
}

template <int N, int blockSize> void measure() {
  const size_t iters = (size_t)1000000000 / N + 2;

  hipDeviceProp_t prop;
  int deviceId;
  GPU_ERROR(hipGetDevice(&deviceId));
  GPU_ERROR(hipGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sumKernel<N, iters, blockSize>, blockSize, 0));

  int blockCount = smCount * 1; // maxActiveBlocks;

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  GPU_ERROR(hipDeviceSynchronize());

  hipEvent_t start, stop;
  GPU_ERROR(hipEventCreate(&start));
  GPU_ERROR(hipEventCreate(&stop));

  for (int i = 0; i < 15; i++) {
    const size_t bufferCount = 2 * N + i * 1282;
    GPU_ERROR(hipMalloc(&dA, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(hipMalloc(&dB, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dB, bufferCount);

    dA += i;
    dB += i;

    GPU_ERROR(hipDeviceSynchronize());

    GPU_ERROR(hipEventRecord(start));
    callKernel<N, iters, blockSize>(blockCount);
    GPU_ERROR(hipEventRecord(stop));

    GPU_ERROR(hipEventSynchronize(stop));
    float milliseconds = 0;
    GPU_ERROR(hipEventElapsedTime(&milliseconds, start, stop));
    time.add(milliseconds / 1000);

    /*    measureDRAMBytesStart();
        callKernel<N, iters, blockSize>(blockCount);
        auto metrics = measureDRAMBytesStop();
        dram_read.add(metrics[0]);
        dram_write.add(metrics[1]);

        measureL2BytesStart();
        callKernel<N, iters, blockSize>(blockCount);
        metrics = measureL2BytesStop();
        L2_read.add(metrics[0]);
        L2_write.add(metrics[1]);
    */

    GPU_ERROR(hipFree(dA - i));
    GPU_ERROR(hipFree(dB - i));
  }
  double blockDV = 2 * N * sizeof(dtype);

  double bw = blockDV * blockCount * iters / time.minValue() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << setprecision(0) << setw(10) << time.value() * 1000.0 << "ms"    //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
       << setw(10) << bw << " GB/s"                                       //
       << setprecision(0) << setw(10)
       << dram_read.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << dram_write.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << L2_read.value() / time.minValue() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << L2_write.value() / time.minValue() / 1.0e9 << " GB/s " << endl; //
}

size_t constexpr expSeries(size_t N) {
  size_t val = 32 * 512;
  for (size_t i = 0; i < N; i++) {
    val *= 1.17;
  }
  return (val / 512) * 512;
}

int main(int argc, char **argv) {
  initMeasureMetric();
  // unsigned int clock = getGPUClock();
  cout << setw(13) << "data set"   //
       << setw(12) << "exec time"  //
       << setw(11) << "spread"     //
       << setw(15) << "Eff. bw"    //
       << setw(16) << "DRAM read"  //
       << setw(16) << "DRAM write" //
       << setw(16) << "L2 read"    //
       << setw(16) << "L2 store\n";

  initMeasureMetric();

  measure<128, 128>();
  measure<256, 256>();
  measure<512, 512>();
  measure<3 * 256, 256>();
  measure<2 * 512, 512>();
  measure<3 * 512, 512>();
  measure<4 * 512, 512>();
  measure<5 * 512, 512>();
  measure<6 * 512, 512>();
  measure<7 * 512, 512>();
  measure<8 * 512, 512>();
  measure<9 * 512, 512>();
  measure<10 * 512, 512>();
  measure<11 * 512, 512>();
  measure<12 * 512, 512>();
  measure<13 * 512, 512>();
  measure<14 * 512, 512>();
  measure<15 * 512, 512>();
  measure<16 * 512, 512>();
  measure<17 * 512, 512>();
  measure<18 * 512, 512>();
  measure<19 * 512, 512>();
  measure<20 * 512, 512>();
  measure<21 * 512, 512>();
  measure<22 * 512, 512>();
  measure<23 * 512, 512>();
  measure<24 * 512, 512>();
  measure<25 * 512, 512>();
  measure<26 * 512, 512>();
  measure<27 * 512, 512>();
  measure<28 * 512, 512>();
  measure<29 * 512, 512>();
  measure<30 * 512, 512>();
  measure<31 * 512, 512>();
  measure<32 * 512, 512>();

  measure<expSeries(1), 512>();
  measure<expSeries(2), 512>();
  measure<expSeries(3), 512>();
  measure<expSeries(4), 512>();
  measure<expSeries(5), 512>();
  measure<expSeries(6), 512>();
  measure<expSeries(7), 512>();
  measure<expSeries(8), 512>();
  measure<expSeries(9), 512>();
  measure<expSeries(10), 512>();
  measure<expSeries(11), 512>();
  measure<expSeries(12), 512>();
  measure<expSeries(13), 512>();
  measure<expSeries(14), 512>();
  measure<expSeries(16), 512>();
  measure<expSeries(17), 512>();
  measure<expSeries(18), 512>();
  measure<expSeries(19), 512>();
  measure<expSeries(20), 512>();
  measure<expSeries(21), 512>();
  measure<expSeries(22), 512>();
  measure<expSeries(23), 512>();
  measure<expSeries(24), 512>();
  measure<expSeries(25), 512>();
  measure<expSeries(26), 512>();
  measure<expSeries(27), 512>();
  measure<expSeries(28), 512>();
  measure<expSeries(29), 512>();
  measure<expSeries(30), 512>();
  measure<expSeries(31), 512>();
  measure<expSeries(32), 512>();
  measure<expSeries(33), 512>();
  measure<expSeries(34), 512>();
  measure<expSeries(35), 512>();
  measure<expSeries(36), 512>();
  measure<expSeries(37), 512>();
  measure<expSeries(38), 512>();
  measure<expSeries(39), 512>();
  measure<expSeries(40), 512>();
  measure<expSeries(41), 512>();
  measure<expSeries(42), 512>();
  measure<expSeries(43), 512>();
  measure<expSeries(44), 512>();
  measure<expSeries(45), 512>();
  measure<expSeries(46), 512>();
  measure<expSeries(47), 512>();
  measure<expSeries(48), 512>();
  measure<expSeries(49), 512>();
}
