// Made critical bugfix: Due to discrepancy in “iter” between k1 and k2
// k2 ended too early, leaving k1 run in isolated state. 
// Checked cache bw related rocprof metrics before and after the fix: e.g., 
// - Avg L2 Cache BW: 40 -> 5000

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <omp.h>
#include <cassert>
#include <iomanip>

#define XCD_NUM 8
#define CHUNK_SIZE (2 * 1024) // 2KB chunks

#define BPX_MAX 160
#ifndef BPX_STRIDE
#define BPX_STRIDE 10
#endif

#ifndef TPB
#define TPB 512
#endif

// Kernel K1 defaults
#ifndef K1_MAX_TPB
#define K1_MAX_TPB 512
#endif

// Kernel K2 defaults
#ifndef K2_MAX_TPB
#define K2_MAX_TPB 512
#endif

#define CHECK_CORRECTNESS 0
#define NB_EPOCH 21

#define DEBUG 0

#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        std::cerr << "Error: " << hipGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

// -------------------------------------------------------------------------
// Fused Kernel
// -------------------------------------------------------------------------
__global__ void k(
    /* k1 args */
    float* k1_imageData, int* k1_histogram, size_t k1_totalPixels, int k1_channels, int k1_numBins, uint32_t *k1_clk, const int k1_max_tpb,
    /* k2 args */
    float *k2_chunks1, float *k2_chunks2, float *k2_chunks3, float *k2_chunks4,
    float *k2_sink, const size_t k2_nchunks, const size_t k2_chunk_size, const int k2_max_tpb)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    
    int n_tbs_in_xcd = (gridDim.x / XCD_NUM); 
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd; 
    
    uint32_t xcc_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    // =========================================================================
    // K1: Histogram (Atomic Ops) - Runs on XCD 0, 2, 4, 6
    // =========================================================================
    if (xcc_id == 0 || xcc_id == 2 || xcc_id == 4 || xcc_id == 6) { 

        if (tbid_in_xcd > 0) return;
        if (tid >= k1_max_tpb) return;

        const int K1_ACTIVE_XCDS = 4;
        int k1_active_threads_per_block = (blockDim.x > k1_max_tpb) ? k1_max_tpb : blockDim.x;
        size_t k1_eff_stride = K1_ACTIVE_XCDS * k1_active_threads_per_block;
        int k1_logical_xcc_id = xcc_id / 2;
        size_t k1_eff_uid = (k1_logical_xcc_id * k1_active_threads_per_block) + tid;
        
        for (size_t i = k1_eff_uid; i < k1_totalPixels; i += k1_eff_stride) {
            
            float brightness = 0.0f;
            size_t pixelOffset = i * k1_channels;

            for (int c = 0; c < k1_channels; ++c) {
                brightness += k1_imageData[pixelOffset + c];
            }
            brightness /= k1_channels;
            int bin = static_cast<int>(brightness * (k1_numBins - 1));
            
            uint32_t start = __builtin_readcyclecounter();
            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); 

            atomicAdd(&k1_histogram[bin], 1);

            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); 
            uint32_t end = __builtin_readcyclecounter();

            k1_clk[i] = (end - start);
        }

    } else { 
    // =========================================================================
    // K2: Bandwidth Saturation - Runs on XCD 1, 3, 5, 7
    // =========================================================================

        if (tid >= k2_max_tpb) return;

        int k2_eff_uid = (xcc_id / 2) * k2_max_tpb * n_tbs_in_xcd + tbid_in_xcd * k2_max_tpb + tid;

        float4 reg_in1, reg_in2, reg_in3, reg_in4;
        float sink0 = 0, sink1 = 0, sink2 = 0, sink3 = 0;

        const size_t n_floats_per_chunk = k2_chunk_size / sizeof(float);
        const size_t n_chunks_per_iter_per_tb = (blockDim.x / (k2_chunk_size / 16));
        size_t n_iter = k2_nchunks / (n_tbs_in_xcd * n_chunks_per_iter_per_tb);
        #if DEBUG
        // if (xcc_id == 1 && tbid_in_xcd == 0 && tid == 0) {
        //     printf("K2: xcc_id=%u, n_tbs_in_xcd=%u, n_chunks_per_iter_per_tb=%zu, n_iter=%zu\n", xcc_id, n_tbs_in_xcd, n_chunks_per_iter_per_tb, n_iter);
        // }
        #endif

        for (size_t iter = 0; iter < n_iter; iter++) {
            for (size_t c = 0; c < n_chunks_per_iter_per_tb; c++) {
                size_t chunk_idx = c + tbid_in_xcd * n_chunks_per_iter_per_tb + iter * n_tbs_in_xcd;
                #if DEBUG
                // printf("K2: xcc_id=%u, tbid_in_xcd=%u, tid=%u, iter=%zu, c=%zu, chunk_idx=%zu\n", xcc_id, tbid_in_xcd, tid, iter, c, chunk_idx);
                #endif

                const size_t chunk_idx_to_float_idx = chunk_idx * n_floats_per_chunk;
                float4 *ptr_in1 = reinterpret_cast<float4*>(&k2_chunks1[chunk_idx_to_float_idx]);
                float4 *ptr_in2 = reinterpret_cast<float4*>(&k2_chunks2[chunk_idx_to_float_idx]);
                float4 *ptr_in3 = reinterpret_cast<float4*>(&k2_chunks3[chunk_idx_to_float_idx]);
                float4 *ptr_in4 = reinterpret_cast<float4*>(&k2_chunks4[chunk_idx_to_float_idx]);
                
                asm volatile(
                    "flat_load_dwordx4 %[OUT_D1],  %[IN_D1]\n\t"
                    "flat_load_dwordx4 %[OUT_C1],  %[IN_C1]\n\t"
                    "flat_load_dwordx4 %[OUT_B1],  %[IN_B1]\n\t"
                    "flat_load_dwordx4 %[OUT_A1],  %[IN_A1]\n\t" 
                    "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                    : [OUT_A1]"=v" (reg_in1), [OUT_B1]"=v" (reg_in2), [OUT_C1]"=v"(reg_in3), [OUT_D1]"=v" (reg_in4)
                    : [IN_A1]"v" (&ptr_in1[tid]), [IN_B1]"v" (&ptr_in2[tid]), [IN_C1]"v" (&ptr_in3[tid]), [IN_D1]"v" (&ptr_in4[tid])
                    : "memory"
                );
                
                sink0 += reg_in1.x + reg_in2.x + reg_in3.x + reg_in4.x;
                sink1 += reg_in1.y + reg_in2.y + reg_in3.y + reg_in4.y;
                sink2 += reg_in1.z + reg_in2.z + reg_in3.z + reg_in4.z;
                sink3 += reg_in1.w + reg_in2.w + reg_in3.w + reg_in4.w;
            }
        }

        k2_sink[k2_eff_uid] = sink0 + sink1 + sink2 + sink3; // Prevent optimization
    }
}


int main() {
    // --- 1. Configuration ---
    const int channels = 3;
    const int numBins = 256;
    
    // K1 Config: 4 XCDs * 512 Threads
    const int k1_active_xcds = 4;
    const int k1_active_threads = k1_active_xcds * min(TPB, K1_MAX_TPB);
    const size_t k1_minLoops = 1000;
    const size_t totalPixels = k1_active_threads * k1_minLoops;
    
    // K2 Config: Chunks
    const int k2_active_xcds = 4;
    const int k2_active_threads = k2_active_xcds * min(TPB, K2_MAX_TPB) * BPX_MAX; // @june has some flaw, as bpx scales in loop
    const size_t k2_minLoops = 1000;
    const int num_k2_chunks = k2_minLoops * (min(TPB, K2_MAX_TPB) / (CHUNK_SIZE/16) ) * BPX_MAX;

    size_t dataSizeBytes = totalPixels * channels * sizeof(float);
    size_t histSizeBytes = numBins * sizeof(int);
    size_t latencySizeBytes = totalPixels * sizeof(uint32_t);

    #if DEBUG
    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Fused Kernel Mode" << std::endl;
    std::cout << "K1 (Hist) Pixels:    " << totalPixels << std::endl;
    std::cout << "K2 (BW) Chunks:      " << num_k2_chunks << std::endl;
    #endif

    // --- 2. Host Alloc & Init ---
    std::vector<float> h_imageData(totalPixels * channels);
    std::vector<int> h_histogram_gpu(numBins, 0);
    // std::vector<int> h_histogram_cpu(numBins, 0);
    std::vector<uint32_t> h_latencySamples(totalPixels);

    #if DEBUG
    std::cout << "Initializing host data..." << std::endl;
    #endif

    #pragma omp parallel for
    for (size_t i = 0; i < h_imageData.size(); ++i) {
        h_imageData[i] = (float)rand() / (float)RAND_MAX;
    }

    // --- 3. Device Alloc (K1) ---
    float* d_k1_imageData = nullptr;
    int* d_k1_histogram = nullptr;
    uint32_t* d_k1_clk = nullptr;

    HIP_CHECK(hipMalloc(&d_k1_imageData, dataSizeBytes));
    HIP_CHECK(hipMalloc(&d_k1_histogram, histSizeBytes));
    HIP_CHECK(hipMalloc(&d_k1_clk, latencySizeBytes));

    HIP_CHECK(hipMemcpy(d_k1_imageData, h_imageData.data(), dataSizeBytes, hipMemcpyHostToDevice));
    
    // --- 4. Device Alloc & Prep (K2) ---
    float *d_k2_chunks1, *d_k2_chunks2, *d_k2_chunks3, *d_k2_chunks4;
    size_t k2_chunks_bufsize = num_k2_chunks * CHUNK_SIZE;
    HIP_CHECK(hipMalloc(&d_k2_chunks1, k2_chunks_bufsize));
    HIP_CHECK(hipMalloc(&d_k2_chunks2, k2_chunks_bufsize));
    HIP_CHECK(hipMalloc(&d_k2_chunks3, k2_chunks_bufsize));
    HIP_CHECK(hipMalloc(&d_k2_chunks4, k2_chunks_bufsize));

    size_t k2_sink_bufcnt = k2_active_xcds * BPX_MAX * K2_MAX_TPB;
    std::vector<float> h_k2_sink(k2_sink_bufcnt, 0.0f);
    float *d_k2_sink;
    HIP_CHECK(hipMalloc(&d_k2_sink, k2_sink_bufcnt * sizeof(float)));

    // Initialize K2 dummy data
    HIP_CHECK(hipMemset(d_k2_chunks1, 0, k2_chunks_bufsize));
    HIP_CHECK(hipMemset(d_k2_chunks2, 0, k2_chunks_bufsize));
    HIP_CHECK(hipMemset(d_k2_chunks3, 0, k2_chunks_bufsize));
    HIP_CHECK(hipMemset(d_k2_chunks4, 0, k2_chunks_bufsize));

    // --- 5. Loop ---
    for (int bpx = BPX_STRIDE; bpx <= BPX_MAX; bpx+=BPX_STRIDE) {
    // for (int bpx = 160; bpx <= BPX_MAX; bpx += BPX_STRIDE) {

        #if DEBUG
        std::cout << "Launching Fused Kernel over " << NB_EPOCH << " epochs..." << std::endl;
        std::cout << "bpx:          " << bpx << std::endl;
        std::cout << "gridDim.x:    " << bpx * XCD_NUM << std::endl;
        std::cout << "blockDim.x:   " << TPB << std::endl;
        #endif

        double avg_p05 = 0.0;
        double avg_p50 = 0.0;
        double avg_p90 = 0.0;
        double avg_p99 = 0.0;

        for (int epoch = 0; epoch < NB_EPOCH; ++epoch) {
            // Reset buffers
            HIP_CHECK(hipMemset(d_k1_histogram, 0, histSizeBytes));
            HIP_CHECK(hipMemset(d_k1_clk, 0, latencySizeBytes));

            hipLaunchKernelGGL(
                (k), 
                dim3(bpx * XCD_NUM), 
                dim3(TPB), 
                0, 
                0, 
                d_k1_imageData, d_k1_histogram, totalPixels, channels, numBins, d_k1_clk, K1_MAX_TPB,
                d_k2_chunks1, d_k2_chunks2, d_k2_chunks3, d_k2_chunks4, d_k2_sink,
                num_k2_chunks, CHUNK_SIZE, K2_MAX_TPB
            );
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());

            // --- 6. Results & Verification ---
            HIP_CHECK(hipMemcpy(h_latencySamples.data(), d_k1_clk, latencySizeBytes, hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(h_k2_sink.data(), d_k2_sink, k2_sink_bufcnt * sizeof(float), hipMemcpyDeviceToHost));

            // Sort to find percentiles
            std::sort(h_latencySamples.begin(), h_latencySamples.end());

            double current_p05 = (double)h_latencySamples[totalPixels * 0.05];
            double current_p50 = (double)h_latencySamples[totalPixels * 0.50];
            double current_p90 = (double)h_latencySamples[totalPixels * 0.90];
            double current_p99 = (double)h_latencySamples[totalPixels * 0.99];

            // Incremental Average Update
            avg_p05 += (current_p05 - avg_p05) / (epoch + 1);
            avg_p50 += (current_p50 - avg_p50) / (epoch + 1);
            avg_p90 += (current_p90 - avg_p90) / (epoch + 1);
            avg_p99 += (current_p99 - avg_p99) / (epoch + 1);
        }

        std::cout << std::setw(3) << bpx << " " 
                  << std::fixed << std::setprecision(2)
                  << avg_p05  << " "
                  << avg_p50 << " "
                  << avg_p90 << " "
                  << avg_p99 << std::endl;
    }

    // Cleanup
    HIP_CHECK(hipFree(d_k1_imageData));
    HIP_CHECK(hipFree(d_k1_histogram));
    HIP_CHECK(hipFree(d_k1_clk));
    HIP_CHECK(hipFree(d_k2_chunks1));
    HIP_CHECK(hipFree(d_k2_chunks2));
    HIP_CHECK(hipFree(d_k2_chunks3));
    HIP_CHECK(hipFree(d_k2_chunks4));
    HIP_CHECK(hipFree(d_k2_sink));

    return 0;
}