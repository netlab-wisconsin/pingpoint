#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include <iomanip>
#include <iostream>

using namespace std;

const int64_t max_buffer_size = 256l * 1024 * 1024 + 2;
double *dA, *dB, *dC, *dD;

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 const double *__restrict__ C,
                                 const double *__restrict__ D, const size_t N,
                                 bool secretlyFalse);

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.23;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void read_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  double temp = B[tidx];

  if (secretlyFalse || temp == 123.0)
    A[tidx] = temp + spoiler[tidx];
}

template <typename T>
__global__ void scale_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = B[tidx] * 1.2;

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void triad_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N, bool secretlyFalse) {
  extern __shared__ double spoiler[];

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = B[tidx] * D[tidx] + C[tidx];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}

template <typename T>
__global__ void stencil1d3pt_kernel(T *A, const T *__restrict__ B,
                                    const T *__restrict__ C,
                                    const T *__restrict__ D, const size_t N,
                                    bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N - 1 || tidx == 0)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.5 * B[tidx - 1] - 1.0 * B[tidx] + 0.5 * B[tidx + 1];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}
template <typename T>
__global__ void stencil1d5pt_kernel(T *A, const T *__restrict__ B,
                                    const T *__restrict__ C,
                                    const T *__restrict__ D, const size_t N,
                                    bool secretlyFalse) {
  extern __shared__ double spoiler[];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N - 2 || tidx < 2)
    return;

  if (secretlyFalse)
    spoiler[threadIdx.x] = B[threadIdx.x];

  A[tidx] = 0.25 * B[tidx - 2] + 0.25 * B[tidx - 1] - 1.0 * B[tidx] +
            0.5 * B[tidx + 1] + 0.5 * B[tidx + 2];

  if (secretlyFalse)
    A[tidx] = spoiler[tidx];
}
void measureFunc(kernel_ptr_type func, int streamCount, int blockSize,
                 int blocksPerSM) {

#ifdef __NVCC__
  GPU_ERROR(cudaFuncSetAttribute(
      func, cudaFuncAttributePreferredSharedMemoryCarveout, 5));
#endif

  int maxActiveBlocks = 0;
  int spoilerSize = 1024;

  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, func, 32, spoilerSize));

  while (maxActiveBlocks > blocksPerSM) {
    spoilerSize += 256;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, func, 32, spoilerSize));
    // std::cout << maxActiveBlocks << " " << spoilerSize << "\n";
  }

  /*if (maxActiveBlocks != blocksPerSM)
    std::cout << "Configure " << maxActiveBlocks << " instead of "
              << blocksPerSM << "\n";
*/

  MeasurementSeries time;

  func<<<max_buffer_size / blockSize + 1, blockSize, spoilerSize>>>(
      dA, dB, dC, dD, max_buffer_size, false);

  GPU_ERROR(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  for (int iter = 0; iter < 31; iter++) {
    GPU_ERROR(cudaEventCreate(&start));
    GPU_ERROR(cudaEventCreate(&stop));
    GPU_ERROR(cudaEventRecord(start));
    func<<<max_buffer_size / blockSize + 1, blockSize, spoilerSize>>>(
        dA, dB, dC, dD, max_buffer_size, false);
    GPU_ERROR(cudaEventRecord(stop));
    GPU_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    GPU_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    time.add(milliseconds / 1000);
  }

  cout << fixed << setprecision(0) << setw(6) << " " << setw(5)
       << streamCount * max_buffer_size * sizeof(double) / time.median() * 1e-9;
  cout.flush();
}

void measureKernels(vector<pair<kernel_ptr_type, int>> kernels, int blockSize,
                    int blocksPerSM) {
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  if (deviceName.starts_with("AMD Radeon RX 6")) {
    prop.maxThreadsPerMultiProcessor = 1024;
    prop.multiProcessorCount *= 2;
  }

  // std::cout << prop.maxThreadsPerMultiProcessor << " "
  //           << prop.maxThreadsPerBlock << "\n";

  if (blockSize * blocksPerSM > prop.maxThreadsPerMultiProcessor ||
      blockSize > prop.maxThreadsPerBlock)
    return;

  int smCount = prop.multiProcessorCount;
  cout << setw(4) << blockSize << "   " << setw(7)
       << smCount * blockSize * blocksPerSM << "  " << setw(5) << setw(6)
       << blocksPerSM << "  " << setprecision(1) << setw(5)
       << (float)(blockSize * blocksPerSM) / prop.maxThreadsPerMultiProcessor *
              100.0
       << "%     |  GB/s: ";

  for (auto kernel : kernels) {
    measureFunc(kernel.first, kernel.second, blockSize, blocksPerSM);
  }

  cout << "\n";
}

int main(int argc, char **argv) {
  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, max_buffer_size * sizeof(double)));

  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dA, dA, dA, dA,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dB, dB, dB, dB,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dC, dC, dC, dC,
                                                    max_buffer_size, false);
  init_kernel<<<max_buffer_size / 1024 + 1, 1024>>>(dD, dD, dD, dD,
                                                    max_buffer_size, false);
  GPU_ERROR(cudaDeviceSynchronize());

  vector<pair<kernel_ptr_type, int>> kernels = {
      {init_kernel<double>, 1},         {read_kernel<double>, 1},
      {scale_kernel<double>, 2},        {triad_kernel<double>, 4},
      {stencil1d3pt_kernel<double>, 2}, {stencil1d5pt_kernel<double>, 2}};

  cout << "block smBlocks   threads    occ%   |                init"
       << "       read       scale     triad       3pt        5pt\n";

  // for (int blockSize = 32; blockSize <= 1024; blockSize += 32) {
  //   measureKernels(kernels, blockSize, 1);
  // }

  measureKernels(kernels, 16, 1);
  measureKernels(kernels, 32, 1);
  measureKernels(kernels, 48, 1);
  measureKernels(kernels, 64, 1);
  measureKernels(kernels, 80, 1);
  measureKernels(kernels, 96, 1);
  measureKernels(kernels, 112, 1);

  for (int warpCount = 4; warpCount <= 80; warpCount++) {

    int threadCount = warpCount * 32;
    if (threadCount / 32 % 2 == 0)
      // and (warpCount < 16 || warpCount % 8 == 0))
      measureKernels(kernels, threadCount / 2, 2);
    else if (warpCount < 6)
      measureKernels(kernels, threadCount, 1);
  }

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
  GPU_ERROR(cudaFree(dD));
}
