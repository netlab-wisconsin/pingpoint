#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include <iomanip>
#include <iostream>

#include "../../MeasurementSeries.hpp"
#include "../../dtime.hpp"
#include "../../gpu-clock.cuh"
#include "../../gpu-error.h"
#include "../../gpu-metrics/gpu-metrics.hpp"

#include "acn-mi300.hpp"

using namespace std;

typedef float4 dtype;

dtype *dA, *dbuf;
dtype **dB1, **dB2, **dB3, **dB4;

#define DEBUG_LEVEL 0

static inline __host__ __device__ uint32_t get_cc(uint32_t xcc_id)
{
    return (xcc_id / (XCD_NUM / CC_NUM)) % CC_NUM;
}

#ifndef InterCCHop
#define InterCCHop 0
#endif

__global__ void initKernel(dtype *A, size_t N)
{
    size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x)
    {
        A[idx] = (dtype)1.1;
    }
}

template <int N, int iters, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, dtype **__restrict__ B1, dtype **__restrict__ B2,
                          dtype **__restrict__ B3, dtype **__restrict__ B4, int zero)
{
    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    // printf("xcc_id: %u, se_id: %u, cu_id: %u\n", xcc_id, se_id, cu_id);
    uint32_t cc_id = get_cc(xcc_id);

    dtype localSum = (dtype)0;

    dtype **B;

    if (InterCCHop == 0) {
        if (cc_id == 0)         B = B1;
        else if (cc_id == 1)    B = B2;
        else if (cc_id == 2)    B = B3;
        else                    B = B4;
    } else if (InterCCHop == 1) {
        if (cc_id == 0)         B = B2;
        else if (cc_id == 1)    B = B3;
        else if (cc_id == 2)    B = B4;
        else                    B = B1;
    } else if (InterCCHop == 2) {
        if (cc_id == 0)         B = B3;
        else if (cc_id == 1)    B = B4;
        else if (cc_id == 2)    B = B1;
        else                    B = B2;
    } else if (InterCCHop == 3) {
        if (cc_id == 0)         B = B4;
        else if (cc_id == 1)    B = B1;
        else if (cc_id == 2)    B = B2;
        else                    B = B3;
    }

    B += threadIdx.x;

#pragma unroll N / BLOCKSIZE> 32   ? 1 : 32 / (N / BLOCKSIZE)
    for (int iter = 0; iter < iters; iter++)
    {
        B += zero;
        auto B2 = B + N;
#pragma unroll N / BLOCKSIZE >= 64 ? 32 : N / BLOCKSIZE
        for (int i = 0; i < N; i += BLOCKSIZE)
        {
            // localSum += B[i] * B2[i];
            localSum += (*(B[i])) * (*(B2[i]));
        }
        localSum *= (dtype)1.3;
    }
    if (localSum == (dtype)1233)
        A[threadIdx.x] += localSum;
}

template <int N, int iters, int blockSize>
double callKernel(int blockCount)
{
    // sumKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB, 0);
    sumKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB1, dB2, dB3, dB4, 0);
    return 0.0;
}

template <int N, int blockSize>
void measure()
{
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

    for (int i = 0; i < 15; i++)
    {
        const size_t bufferCount = 2 * N + i * 1282;
        GPU_ERROR(hipMalloc(&dA, bufferCount * sizeof(dtype)));
        initKernel<<<52, 256>>>(dA, bufferCount);

        const size_t multiplicative_factor = CC_NUM * 16;
        const size_t n_dtype_dbuf = bufferCount * multiplicative_factor;
        GPU_ERROR(hipMalloc(&dbuf, n_dtype_dbuf * sizeof(dtype)));
        initKernel<<<52, 256>>>(dbuf, n_dtype_dbuf);

        vector<uint32_t> dtype_home_cc(n_dtype_dbuf, (uint32_t)-1);
        vector<vector<dtype *>> cc_dtypes(CC_NUM);

        /* home identification */
        {
            uint32_t *d_cycles = nullptr;
            GPU_ERROR(hipMalloc(&d_cycles, n_dtype_dbuf * (size_t)XCD_NUM * sizeof(uint32_t)));

            void *kernel_args[] = {
                (void *)&dbuf,
                (void *)&d_cycles,
                (void *)&n_dtype_dbuf,
            };

            GPU_ERROR(hipLaunchCooperativeKernel(
                identify_home<dtype>, dim3(1 * XCD_NUM), dim3(128 / sizeof(dtype)), kernel_args, 0, 0));

            GPU_ERROR(hipDeviceSynchronize());

            vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(n_dtype_dbuf));
            for (int x = 0; x < XCD_NUM; x++)
            {
                GPU_ERROR(hipMemcpy(h_cycles[x].data(),
                                    d_cycles + (size_t)x * n_dtype_dbuf,
                                    sizeof(uint32_t) * n_dtype_dbuf,
                                    hipMemcpyDeviceToHost));
            }

            for (size_t k = 0; k < n_dtype_dbuf; k++)
            {
                uint32_t min_cycles = UINT32_MAX;
                int min_xcc = -1;
                for (int x = 0; x < XCD_NUM; x++)
                {
                    uint32_t c = h_cycles[x][k];
                    if (c < min_cycles)
                    {
                        min_cycles = c;
                        min_xcc = x;
                    }
#if DEBUG_LEVEL >= 3
                    cout << "dtype " << k << " xcc " << x << " cycles " << c << "\n";
#endif
                }

                uint32_t cc = get_cc((uint32_t)min_xcc);
                dtype_home_cc[k] = cc;
                cc_dtypes[cc].push_back(reinterpret_cast<dtype *>((size_t)dbuf + k * sizeof(dtype)));
            }

            GPU_ERROR(hipFree(d_cycles));

#if DEBUG_LEVEL >= 1
            for (int cc = 0; cc < CC_NUM; cc++)
            {
                cout << "CC " << cc << " has " << cc_dtypes[cc].size() << " dtypes.\n";
            }
#endif
        }

        GPU_ERROR(hipDeviceSynchronize());

        // this is where the trouble is.

        if (cc_dtypes[0].size() < 2*N || cc_dtypes[1].size() < 2*N ||
            cc_dtypes[2].size() < 2*N || cc_dtypes[3].size() < 2*N) {
            // printf("(N: %d, i: %d) Not enough dtypes in one of the CCs. Skipping this iteration.\n", N, i);
            GPU_ERROR(hipFree(dA));
            GPU_ERROR(hipFree(dbuf));
            continue;
        } else {
            // printf("(N: %d, i: %d) Enough dtypes in all CCs. Proceeding with this iteration.\n", N, i);
        }



        GPU_ERROR(hipMalloc(&dB1, cc_dtypes[0].size() * sizeof(dtype *)));
        GPU_ERROR(hipMalloc(&dB2, cc_dtypes[1].size() * sizeof(dtype *)));
        GPU_ERROR(hipMalloc(&dB3, cc_dtypes[2].size() * sizeof(dtype *)));
        GPU_ERROR(hipMalloc(&dB4, cc_dtypes[3].size() * sizeof(dtype *)));

        // printf("N: %d\n", N);
        // printf("dB1: %lu cc_dtypes[0]: %zu, 2N*(dtypes*): %lu, \n", bufferCount * sizeof(dtype *), cc_dtypes[0].size(), 2 * N * sizeof(dtype *));
        // printf("dB2: %lu cc_dtypes[1]: %zu, 2N*(dtypes*): %lu\n", bufferCount * sizeof(dtype *), cc_dtypes[1].size(), 2 * N * sizeof(dtype *));
        // printf("dB3: %lu cc_dtypes[2]: %zu, 2N*(dtypes*): %lu\n", bufferCount * sizeof(dtype *), cc_dtypes[2].size(), 2 * N * sizeof(dtype *));
        // printf("dB4: %lu cc_dtypes[3]: %zu, 2N*(dtypes*): %lu\n", bufferCount * sizeof(dtype *), cc_dtypes[3].size(), 2 * N * sizeof(dtype *));

        // GPU_ERROR(hipMemcpy(dB1, cc_dtypes[0].data(), sizeof(dtype *) * 2*N, hipMemcpyHostToDevice));
        // GPU_ERROR(hipMemcpy(dB2, cc_dtypes[1].data(), sizeof(dtype *) * 2*N, hipMemcpyHostToDevice));
        // GPU_ERROR(hipMemcpy(dB3, cc_dtypes[2].data(), sizeof(dtype *) * 2*N, hipMemcpyHostToDevice));
        // GPU_ERROR(hipMemcpy(dB4, cc_dtypes[3].data(), sizeof(dtype *) * 2*N, hipMemcpyHostToDevice));
        GPU_ERROR(hipMemcpy(dB1, cc_dtypes[0].data(), sizeof(dtype *) * cc_dtypes[0].size(), hipMemcpyHostToDevice));
        GPU_ERROR(hipMemcpy(dB2, cc_dtypes[1].data(), sizeof(dtype *) * cc_dtypes[1].size(), hipMemcpyHostToDevice));
        GPU_ERROR(hipMemcpy(dB3, cc_dtypes[2].data(), sizeof(dtype *) * cc_dtypes[2].size(), hipMemcpyHostToDevice));
        GPU_ERROR(hipMemcpy(dB4, cc_dtypes[3].data(), sizeof(dtype *) * cc_dtypes[3].size(), hipMemcpyHostToDevice));

        // shifting here @june: necessary?
        dA += i;
        dB1 += i;
        dB2 += i;
        dB3 += i;
        dB4 += i;

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
        GPU_ERROR(hipFree(dB1 - i));
        GPU_ERROR(hipFree(dB2 - i));
        GPU_ERROR(hipFree(dB3 - i));
        GPU_ERROR(hipFree(dB4 - i));
        GPU_ERROR(hipFree(dbuf));
    }
    double blockDV = 2 * N * sizeof(dtype);

#if 1 // for logging the latency of each load. Only used for L1 range. 06/02/2026.
    double bw = blockDV * blockCount * iters / time.minValue() / 1.0e9;
    double gpu_clock_hz = prop.clockRate * 1000.0; // clockRate is in kHz
    double loads_per_thread = (double)iters * (N / blockSize) * 2;
    double avg_latency_cycles = (time.minValue() * gpu_clock_hz) / loads_per_thread;
    cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
         << setprecision(0) << setw(10) << time.value() * 1000.0 << "ms"    //
         << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
         << setw(10) << bw << " GB/s"                                       //
         << setprecision(1) << setw(12) << avg_latency_cycles << " cy/ld"   //
         << setprecision(0) << setw(10)
         << dram_read.value() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(10)
         << dram_write.value() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(10)
         << L2_read.value() / time.minValue() / 1.0e9 << " GB/s " //
         << setprecision(0) << setw(10)
         << L2_write.value() / time.minValue() / 1.0e9 << " GB/s " << endl; //
#else
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
#endif
}

size_t constexpr expSeries(size_t N)
{
    size_t val = 32 * 512;
    for (size_t i = 0; i < N; i++)
    {
        val *= 1.17;
    }
    return (val / 512) * 512;
}

int main(int argc, char **argv)
{
    printf("Inter-CC hop: %d\n", InterCCHop);

#if 1 // for utilizing multi-GPU node
    int device_id;
    int device_count;
    hipError_t err = hipGetDevice(&device_id);
    hipError_t error = hipGetDeviceCount(&device_count);
    printf("GPU device ID: %d\n", device_id);
    printf("GPU device count: %d\n", device_count);
    // exit(0);
#endif

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

#if 1 // for logging. 06/02/2026.
    // L1 range
    // measure<1 * 256, 256>();
    // measure<3 * 256, 256>();

    // L2 range
    // measure<8 * 256, 256>(); // 64KB, skip
    // measure<8 * 512, 512>();
    // measure<16 * 512, 512>();
    // measure<17 * 512, 512>();
    // measure<18 * 512, 512>();
    // measure<19 * 512, 512>();
    // measure<20 * 512, 512>();
    // measure<21 * 512, 512>();
    // measure<22 * 512, 512>();
    // measure<23 * 512, 512>();
    // measure<24 * 512, 512>();
    // measure<25 * 512, 512>();
    // measure<26 * 512, 512>();
    // measure<27 * 512, 512>();
    // measure<28 * 512, 512>();
    // measure<29 * 512, 512>();
    // measure<30 * 512, 512>();
    // measure<31 * 512, 512>();
    // measure<32 * 512, 512>();
    // measure<128 * 256, 256>(); // 1024KB (1MB)

    // LLC range
    // measure<expSeries(4), 512>();
    // measure<expSeries(8), 512>();
    // measure<expSeries(12), 512>();
    // measure<expSeries(16), 512>();
    // measure<expSeries(20), 512>();
    // measure<expSeries(24), 512>();
    // measure<expSeries(28), 512>();
    // measure<expSeries(32), 512>();
    // measure<expSeries(36), 512>();
    // measure<expSeries(40), 512>();
    // measure<expSeries(44), 512>();

    // HBM range
    // measure<expSeries(45), 512>();
    // measure<expSeries(46), 512>();
    // measure<expSeries(47), 512>();
    // measure<expSeries(48), 512>();
    // measure<expSeries(49), 512>();
    // measure<expSeries(50), 512>();
    // measure<expSeries(51), 512>();
    // measure<expSeries(52), 512>();
    // measure<expSeries(53), 512>();
    // measure<expSeries(54), 512>();
    // measure<expSeries(55), 512>();
    // measure<expSeries(56), 512>();
    // measure<expSeries(57), 512>();
    // measure<expSeries(58), 512>();
    // measure<expSeries(59), 512>();
    // measure<expSeries(60), 512>();


#else
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
#endif
}
