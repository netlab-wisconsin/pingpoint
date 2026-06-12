// bmk1_ntbx is deprecated since large #tbs per xcd cause scheduling/mlp overhead issues.
// this bmk use only 1 tb per xcd and cooperative group api for sync across tbs

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>

#include "bmk1_1tbx.h"
#include "bmk3.h"

using namespace std;

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int main() {

    /* configure thread blocks */

    const int n_blocks = (TPX * XCDS_NUM);
    const int n_threads_per_block = (WARPS_PER_BLOCK * THREADS_PER_WARP);
    const int total_threads = (n_blocks * n_threads_per_block);
    printf("n_xcds: %d, n_blocks: %d, n_blocks_per_xcd: %d, n_warps_per_block: %d, n_threads_per_warp: %d\n", XCDS_NUM, n_blocks, TPX, WARPS_PER_BLOCK, THREADS_PER_WARP);

    const long long data_size = ((long long)N_PAGES * PAGE_SIZE); 
    const int n_chunks = (data_size / CHUNK_SIZE);
    printf("data size: %lld MB\n", data_size / (1024 * 1024));

    /* allocate data */

    char *d_data_alloc;
    gpuErrchk(hipMalloc((void**)&d_data_alloc, data_size + PAGE_SIZE));
    char *d_data = (char*)(((uintptr_t)d_data_alloc + PAGE_SIZE - 1) & ~(uintptr_t)(PAGE_SIZE - 1)); // 2MB page align

    /* allocate cycles array */

    uint32_t *d_cycles; // per-xcd, per-chunk cycles array
    gpuErrchk(hipMalloc((void**)&d_cycles, sizeof(uint32_t) * XCDS_NUM * n_chunks ));
    gpuErrchk(hipMemset(d_cycles, 0xff, sizeof(uint32_t) * XCDS_NUM * n_chunks));
    printf("allocated d_cycles array[%d][%d]\n", XCDS_NUM, n_chunks);

    /* create cu masked stream */

    hipStream_t stream;
    uint32_t cuMaskSize;
    std::vector<uint32_t> cuMask;
    const int n_cus_enabled_per_xcd = mask_cu(TPX, cuMaskSize, cuMask);
    if (n_cus_enabled_per_xcd < 0) {
        gpuErrchk(hipFree(d_data_alloc));
        gpuErrchk(hipFree(d_cycles));
        return 1;
    }
    printf("first %d cus enabled per xcd\n", n_cus_enabled_per_xcd);
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
        (void*)&n_chunks
    };

    gpuErrchk(hipLaunchCooperativeKernel(
        identify_home,
        dim3(n_blocks), dim3(n_threads_per_block), 
        kernel_args, 0, stream
    ));
    gpuErrchk(hipDeviceSynchronize());

    /* retrieve and process results */

    uint32_t *h_cycles;
    h_cycles = (uint32_t*)malloc(sizeof(uint32_t) * XCDS_NUM * n_chunks);
    gpuErrchk(hipMemcpy(
        h_cycles, 
        d_cycles, 
        sizeof(uint32_t) * XCDS_NUM * n_chunks, 
        hipMemcpyDeviceToHost
    ));
    
    vector<int> h_home(n_chunks);
    size_t missing_cycles = 0;
    size_t chunks_without_valid_xcd = 0;
    int exit_code = 0;
    for (size_t i = 0; i < n_chunks; i++) {
        uint32_t min_cycles = UINT32_MAX;
        int min_xcc = -1;
        for (int xcc = 0; xcc < XCDS_NUM; xcc++) {
            // uint32_t c = h_cycles[xcc][i];
            uint32_t c = h_cycles[xcc * n_chunks + i];
            if (c == UINT32_MAX) {
                missing_cycles++;
                #if DEBUG
                printf("chunk[%zu] xcd %d: missing\n", i, xcc);
                #endif
                continue;
            }
            if (c < min_cycles) {
                min_cycles = c;
                min_xcc = xcc;
            }
            #if DEBUG
                printf("chunk[%zu] xcd %d: cycles %u\n", i, xcc, c);
            #endif
        }
        if (min_xcc < 0) {
            chunks_without_valid_xcd++;
            continue;
        }
        h_home[i] = min_xcc;
        printf("chunk[%zu]: home xcd %d\n", i, h_home[i]);
    }
    if (missing_cycles != 0) {
        fprintf(stderr,
                "warning: %zu/%zu d_cycles slots were not written; scheduler/XCD coverage assumption may be wrong\n",
                missing_cycles,
                (size_t)XCDS_NUM * n_chunks);
    }
    if (chunks_without_valid_xcd != 0) {
        fprintf(stderr, "error: %zu chunks had no valid XCD measurements\n", chunks_without_valid_xcd);
        exit_code = 1;
    }

    /* cleanup */

    free(h_cycles);
    gpuErrchk(hipFree(d_data_alloc));
    gpuErrchk(hipFree(d_cycles));

    return exit_code;
}
