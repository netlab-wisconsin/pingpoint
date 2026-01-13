#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h> // Include OpenMP for fast CPU verification

#define HIP_CHECK(command)                                                                               \
    {                                                                                                    \
        hipError_t status = command;                                                                     \
        if (status != hipSuccess)                                                                        \
        {                                                                                                \
            std::cerr << "Error: " << hipGetErrorString(status) << " at line " << __LINE__ << std::endl; \
            exit(1);                                                                                     \
        }                                                                                                \
    }

// -------------------------------------------------------------------------
// Kernel
// -------------------------------------------------------------------------
__global__ void calculateHistogramProfiled(
    float *imageData, int *histogram, uint32_t *latencySamples,
    size_t totalPixels, int channels, int numBins)
{

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < totalPixels; i += stride)
    {

        float brightness = 0.0f;
        size_t pixelOffset = i * channels;

        for (int c = 0; c < channels; ++c)
        {
            brightness += imageData[pixelOffset + c];
        }
        brightness /= channels;
        int bin = static_cast<int>(brightness * (numBins - 1));

        uint32_t start = __builtin_readcyclecounter();
        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");

        atomicAdd(&histogram[bin], 1);

        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");
        uint32_t end = __builtin_readcyclecounter();

        latencySamples[i] = (end - start);
    }
}

// -------------------------------------------------------------------------
// CPU Reference Implementation (Ground Truth)
// -------------------------------------------------------------------------
void calculateHistogramCPU(const std::vector<float> &imageData, std::vector<int> &histogram,
                           size_t totalPixels, int channels, int numBins)
{

    // Reset histogram
    std::fill(histogram.begin(), histogram.end(), 0);

// Use OpenMP to speed up the CPU reference check
// We use a reduction on the histogram array to avoid race conditions
// without using slow atomic captures for every pixel.

// Note: For 256 bins, we can afford thread-local copies.
#pragma omp parallel
    {
        std::vector<int> local_hist(numBins, 0);

#pragma omp for nowait
        for (size_t i = 0; i < totalPixels; ++i)
        {
            float brightness = 0.0f;
            size_t pixelOffset = i * channels;

            for (int c = 0; c < channels; ++c)
            {
                brightness += imageData[pixelOffset + c];
            }
            brightness /= channels;
            int bin = static_cast<int>(brightness * (numBins - 1));

            local_hist[bin]++;
        }

#pragma omp critical
        {
            for (int b = 0; b < numBins; ++b)
            {
                histogram[b] += local_hist[b];
            }
        }
    }
}

int main()
{
    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------
    const int channels = 3;
    const int numBins = 256;
    const int numBlocks = 8;
    const int threadsPerBlock = 256;
    const int totalThreads = numBlocks * threadsPerBlock;
    const size_t minLoopsPerThread = 100000;

    const size_t width = totalThreads * minLoopsPerThread;
    const size_t totalPixels = width;

    size_t dataSizeBytes = totalPixels * channels * sizeof(float);
    size_t histSizeBytes = numBins * sizeof(int);
    size_t latencySizeBytes = totalPixels * sizeof(uint32_t);

    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Total Pixels:        " << totalPixels << std::endl;
    std::cout << "Data Size:           " << (dataSizeBytes / 1024.0 / 1024.0) << " MB" << std::endl;

    // -------------------------------------------------------------------------
    // Host Alloc & Init
    // -------------------------------------------------------------------------
    std::vector<float> h_imageData(totalPixels * channels);
    std::vector<int> h_histogram_gpu(numBins, 0); // To store GPU results
    std::vector<int> h_histogram_cpu(numBins, 0); // To store CPU results
    std::vector<uint32_t> h_latencySamples(totalPixels);

    std::cout << "Initializing data with random values..." << std::endl;
// Fast parallel fill
#pragma omp parallel
    {
        unsigned int seed = 1234 + omp_get_thread_num();
#pragma omp for
        for (size_t i = 0; i < h_imageData.size(); ++i)
        {
            h_imageData[i] = (float)rand_r(&seed) / (float)RAND_MAX;
        }
    }

    // -------------------------------------------------------------------------
    // Device Alloc
    // -------------------------------------------------------------------------
    float *d_imageData = nullptr;
    int *d_histogram = nullptr;
    uint32_t *d_latencySamples = nullptr;

    HIP_CHECK(hipMalloc(&d_imageData, dataSizeBytes));
    HIP_CHECK(hipMalloc(&d_histogram, histSizeBytes));
    HIP_CHECK(hipMalloc(&d_latencySamples, latencySizeBytes));

    HIP_CHECK(hipMemcpy(d_imageData, h_imageData.data(), dataSizeBytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_histogram, 0, histSizeBytes));

    // -------------------------------------------------------------------------
    // GPU Execution
    // -------------------------------------------------------------------------
    std::cout << "Running GPU Kernel..." << std::endl;
    calculateHistogramProfiled<<<numBlocks, threadsPerBlock>>>(
        d_imageData, d_histogram, d_latencySamples, totalPixels, channels, numBins);
    HIP_CHECK(hipDeviceSynchronize());

    // Retrieve GPU Results
    HIP_CHECK(hipMemcpy(h_histogram_gpu.data(), d_histogram, histSizeBytes, hipMemcpyDeviceToHost));

    // Retrieve Latency (Optional for this step, but kept for completeness)
    std::cout << "Kernel finished. Copying " << totalPixels << " latency samples to host..." << std::endl;
    HIP_CHECK(hipMemcpy(h_latencySamples.data(), d_latencySamples, latencySizeBytes, hipMemcpyDeviceToHost));

    std::cout << "Sorting samples to find percentiles (this may take a moment)..." << std::endl;
    std::sort(h_latencySamples.begin(), h_latencySamples.end());

    // Calculate Indices
    size_t idx_p50 = (size_t)(totalPixels * 0.50);
    size_t idx_p99 = (size_t)(totalPixels * 0.99);
    size_t idx_p999 = (size_t)(totalPixels * 0.999);

    std::cout << "\n=== AtomicAdd Latency Results (Clock Cycles) ===" << std::endl;
    std::cout << "Min: " << h_latencySamples[0] << " cycles" << std::endl;
    std::cout << "P50: " << h_latencySamples[idx_p50] << " cycles" << std::endl;
    std::cout << "P99: " << h_latencySamples[idx_p99] << " cycles" << std::endl;
    std::cout << "Max: " << h_latencySamples[totalPixels - 1] << " cycles" << std::endl;

    // -------------------------------------------------------------------------
    // CPU Verification
    // -------------------------------------------------------------------------
    std::cout << "\nRunning CPU Verification (this may take a few seconds)..." << std::endl;
    double start_cpu = omp_get_wtime();

    calculateHistogramCPU(h_imageData, h_histogram_cpu, totalPixels, channels, numBins);

    double end_cpu = omp_get_wtime();
    std::cout << "CPU Time: " << (end_cpu - start_cpu) << "s" << std::endl;

    // -------------------------------------------------------------------------
    // Compare Results
    // -------------------------------------------------------------------------
    bool match = true;
    long total_gpu_pixels = 0;
    long total_cpu_pixels = 0;

    std::cout << "\n--- Correctness Check ---" << std::endl;
    for (int i = 0; i < numBins; ++i)
    {
        total_gpu_pixels += h_histogram_gpu[i];
        total_cpu_pixels += h_histogram_cpu[i];

        if (h_histogram_gpu[i] != h_histogram_cpu[i])
        {
            std::cout << "Mismatch at Bin " << i
                      << " | GPU: " << h_histogram_gpu[i]
                      << " | CPU: " << h_histogram_cpu[i] << std::endl;
            match = false;
            // Break early or show all? Let's show first few errors then stop
            if (i > 10)
            {
                std::cout << "...(truncating errors)";
                break;
            }
        }
    }

    std::cout << "Total Pixels GPU: " << total_gpu_pixels << std::endl;
    std::cout << "Total Pixels CPU: " << total_cpu_pixels << std::endl;

    if (match && (total_gpu_pixels == totalPixels))
    {
        std::cout << "RESULT: [ PASS ] - GPU matches CPU reference." << std::endl;
    }
    else
    {
        std::cout << "RESULT: [ FAIL ] - Data mismatch detected." << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_imageData));
    HIP_CHECK(hipFree(d_histogram));
    HIP_CHECK(hipFree(d_latencySamples));

    return 0;
}