// bmk1_ntbx is deprecated since large #tbs per xcd cause scheduling/mlp overhead issues.
// this bmk use only 1 tb per xcd and cooperative group api for sync across tbs

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>

#include "bmk1_1tbx.h"

using namespace std;

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#ifndef XCDS_NUM
#define XCDS_NUM 8 // num xcds in mi300x
#endif

constexpr int PAGE_SIZE = (2 * 1024 * 1024); // 2MB huge page
constexpr int CHUNK_SIZE = (2 * 1024); // 2KB
// constexpr int CHUNK_SIZE = (4 * 1024); // 4KB chunk size
constexpr int N_PAGES = (128); // you can change

constexpr int THREADS_PER_WARP = (64);
constexpr int WARPS_PER_BLOCK = (CHUNK_SIZE / (16 * THREADS_PER_WARP)); // one block per chunk
constexpr int TPX = (1); // thread blocks per xcd

#ifndef DEBUG
#define DEBUG 1
#endif

int main() {

    /* configure thread blocks */

    const int n_blocks = (TPX * XCDS_NUM);
    const int n_threads_per_block = (WARPS_PER_BLOCK * THREADS_PER_WARP);
    const int total_threads = (n_blocks * n_threads_per_block);
    printf("n_xcds: %d, n_blocks: %d, n_blocks_per_xcd: %d, n_warps_per_block: %d, n_threads_per_warp: %d\n", XCDS_NUM, n_blocks, TPX, WARPS_PER_BLOCK, THREADS_PER_WARP);

    const long long data_size = ((long long)N_PAGES * PAGE_SIZE); 
    printf("data size: %lld MB\n", data_size / (1024 * 1024));

    /* allocate data */

    char *d_data;
    gpuErrchk(hipMalloc((void**)&d_data, data_size + 0x1000));
    d_data = (char*)(((uintptr_t)d_data & ~(0x0FFF)) + 0x1000); // page align

    /* allocate cycles array */

    uint32_t **d_cycles; // per-xcd, per-chunk cycles array
    gpuErrchk(hipMalloc((void**)&d_cycles, sizeof(uint32_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMalloc((void**)&d_cycles[x], sizeof(uint32_t) * (data_size / CHUNK_SIZE) ));
    }
    printf("allocated d_cycles array[%d][%d]\n", XCDS_NUM, (int)data_size / CHUNK_SIZE);

    int *d_home; // home 'xcc' per chunk 
    gpuErrchk(hipMalloc((void**)&d_home, sizeof(int) * (data_size / CHUNK_SIZE) ));

    /* create cu masked stream */

    hipDeviceProp_t props;
    gpuErrchk(hipGetDeviceProperties(&props, 0));
    
    hipStream_t stream;
    uint32_t cuMaskSize = (props.multiProcessorCount + 31) / 32;
    std::vector<uint32_t> cuMask(cuMaskSize, 0); // Initialize all CUs as disabled. Use (uint32_t)-1 to enable all.

    const size_t n_cus_enabled_per_xcd = n_blocks / 8;
    for (int i = 0; i < n_cus_enabled_per_xcd; ++i) { 
        cuMask[i/4] |= (0xffu << ((3-(i%4)) * 8));
    }
    printf("first %zu cus enabled per xcd\n", n_cus_enabled_per_xcd);
    #if DEBUG
    {
        printf("CU Mask: ");
        for (size_t i = 0; i < cuMaskSize; ++i) { printf("%08x ", cuMask[i]); }
        printf("\n" );
    }
    #endif
    
    gpuErrchk(hipExtStreamCreateWithCUMask(&stream, cuMaskSize, cuMask.data()));
    
    /* launch kernel */

    void *kernel_args[] = {
        (void*)&d_data,
        (void*)&data_size,
        (void*)&d_cycles,
        (void*)&d_home
    };

    gpuErrchk(hipLaunchCooperativeKernel(
        identify_home,
        dim3(n_blocks), dim3(n_threads_per_block), 
        kernel_args, 0, stream
    ));
    gpuErrchk(hipDeviceSynchronize());

    /* retrieve and process results */

    vector<vector<uint32_t>> h_cycles(XCDS_NUM, vector<uint32_t>(data_size / CHUNK_SIZE));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMemcpy(
            h_cycles[x].data(), 
            d_cycles[x], 
            sizeof(uint32_t) * (data_size / CHUNK_SIZE), 
            hipMemcpyDeviceToHost
        ));
    }
    
    vector<int> h_home(data_size / CHUNK_SIZE);
    for (size_t i = 0; i < (data_size / CHUNK_SIZE); i++) {
        uint32_t min_cycles = 0xFFFFFFFF;
        int min_xcc = -1;
        for (int xcc = 0; xcc < XCDS_NUM; xcc++) {
            uint32_t c = h_cycles[xcc][i];
            if (c < min_cycles) {
                min_cycles = c;
                min_xcc = xcc;
            }
            #if DEBUG
            printf("chunk[%zu] xcd %d: cycles %u\n", i, xcc, c);
            #endif
        }
        h_home[i] = min_xcc;
        printf("chunk[%zu]: home xcd %d\n", i, h_home[i]);
    }

    /* cleanup */

    gpuErrchk(hipFree(d_data));

    return 0;
}