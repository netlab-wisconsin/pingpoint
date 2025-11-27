// extension of mi300x_mapping/bmk1 to multiple thread blocks per xcd

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>

#include "bmk1.h"

#ifndef TPX
#define TPX 1
#endif

using namespace std;

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr int PAGE_SIZE = (2 * 1024 * 1024); // 2MB huge page
constexpr int CHUNK_SIZE = (4 * 1024); // 4KB chunk size

#define XCDS_NUM 8
#define DEBUG 0

__global__ void k(uint64_t **xcd_chunks, size_t *xcd_chunks_size, uint32_t **cycles) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    // print tbid_in_xcd
    int n_tbs_in_xcd = (gridDim.x / XCDS_NUM); // number of thread blocks in each xcd
    int tbid_in_xcd = (bid / XCDS_NUM) % n_tbs_in_xcd; // thread block id within xcd
    if (tid == 0) {
        #if DEBUG
        printf("bid %d: tbid_in_xcd %d\n", bid, tbid_in_xcd);
        #endif
    }

    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    if (tid == 0) {
        #if DEBUG
        printf("bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", bid, xcc_id, se_id, cu_id);
        #endif
    }

    uint32_t total_cycles = 0, start = 0, end = 0;
    size_t n_iter = xcd_chunks_size[xcc_id] / n_tbs_in_xcd;
    if (tid == 0) {
        #if DEBUG
        printf("bid %d (tbid_in_xcd %d) on xcc %d: n_iter %zu\n", bid, tbid_in_xcd, xcc_id, n_iter);
        #endif
    }

    // warmup
    for (size_t iter = 0; iter < n_iter; iter++) {
        size_t chunk_idx = tbid_in_xcd + iter * n_tbs_in_xcd;
        uint64_t ptr = xcd_chunks[xcc_id][chunk_idx];
        uint4 *data_ptr = (uint4*)ptr;
    
        asm volatile (
            "flat_load_dwordx4 v[0:3], %0\n\t"
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            :
            : "v"(&data_ptr[tid])
            : "memory", "v0", "v1", "v2", "v3"
        );
    }

    // measurement
    for (size_t iter = 0; iter < n_iter; iter++) {
        size_t chunk_idx = tbid_in_xcd + iter * n_tbs_in_xcd;
        uint64_t ptr = xcd_chunks[xcc_id][chunk_idx];
        uint4 *data_ptr = (uint4*)ptr;
        
        start = __builtin_readcyclecounter();
        asm volatile (
            "flat_load_dwordx4 v[0:3], %0\n\t"
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            :
            : "v"(&data_ptr[tid])
            : "memory", "v0", "v1", "v2", "v3"
        );
        end = __builtin_readcyclecounter();
        total_cycles += (end - start);
    }
    
    if (tid == 0) {
        cycles[xcc_id][tbid_in_xcd] = total_cycles;
    }
}


int main() {

    const int n_threads_per_warp = 64;
    const int n_warps_per_block = 4; // while max=16, set 4 s.t. each tb loads 64*4*16B=4KB chunk per iter

    /* configure thread blocks */

    const int n_blocks = (TPX * XCDS_NUM);
    const int n_threads_per_block = (n_warps_per_block * n_threads_per_warp);
    const int total_threads = (n_blocks * n_threads_per_block);
    printf("n_blocks: %d, n_blocks_per_xcd: %d, n_warps_per_block: %d, n_threads_per_warp: %d\n", n_blocks, TPX, n_warps_per_block, n_threads_per_warp);

    /* allocate data */

    const int n_pages = 128; // := 256 MB to fill up LLC
    const size_t data_size = (n_pages * PAGE_SIZE);
    const int n_4KB_chunks = data_size / CHUNK_SIZE;
    printf("data_size: %d MB\n", (int)data_size / (1024 * 1024));

    char *d_data;
    gpuErrchk(hipMalloc((void**)&d_data, data_size + 0x1000));
    d_data = (char*)(((uintptr_t)d_data & ~(0x0FFF)) + 0x1000);

    /* allocate cycles array */

    uint32_t **d_cycles; // per-xcd, per-4KB chunk. record cycles for home identification 
    gpuErrchk(hipMalloc((void**)&d_cycles, sizeof(uint32_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMalloc((void**)&d_cycles[x], sizeof(uint32_t) * n_4KB_chunks ));
    }
    printf("allocated d_cycles array[%d][%d]\n", XCDS_NUM, n_4KB_chunks);

    int *d_home; // per-4KB chunk. record home xcd
    gpuErrchk(hipMalloc((void**)&d_home, sizeof(int) * n_4KB_chunks ));

    /* create cu masked stream */

    hipDeviceProp_t props;
    gpuErrchk(hipGetDeviceProperties(&props, 0));
    
    hipStream_t stream;
    uint32_t cuMaskSize = (props.multiProcessorCount + 31) / 32;
    vector<uint32_t> cuMask(cuMaskSize, 0); // Initialize all CUs as disabled. Use (uint32_t)-1 to enable all.

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
    
    /* launch home identification kernel */

    hipLaunchKernelGGL(
        identify_home,
        dim3(n_blocks), dim3(n_threads_per_block), 
        0, stream,
        (void*)d_data, data_size, d_cycles, d_home
    );
    gpuErrchk(hipDeviceSynchronize());

    /* retrieve and process results */
    
    uint32_t h_cycles[XCDS_NUM][n_4KB_chunks];;
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMemcpy(&h_cycles[x][0], d_cycles[x], sizeof(uint32_t) * n_4KB_chunks, hipMemcpyDeviceToHost));
    }
    
    int h_home[n_4KB_chunks];
    for (size_t i = 0; i < n_4KB_chunks; i++) {
        uint32_t min_cycles = 0xFFFFFFFF;
        int min_xcc = -1;
        for (int xcc = 0; xcc < XCDS_NUM; xcc++) {
            uint32_t c = h_cycles[xcc][i];
            if (c < min_cycles) {
                min_cycles = c;
                min_xcc = xcc;
            }
        }
        h_home[i] = min_xcc;
        #if DEBUG
        {
            printf("4KB chunk[%zu]: home xcd %d\n", i, h_home[i]);
        }
        #endif
    }

    /* group per-xcd 4KB chunk pointers */
    
    vector<vector<uint64_t>> xcd_chunks(XCDS_NUM);
    for (size_t i = 0; i < n_4KB_chunks; i++) {
        xcd_chunks[h_home[i]].push_back(reinterpret_cast<uint64_t>(d_data) + i * (CHUNK_SIZE));
    }

    uint64_t **d_xcd_chunks;
    gpuErrchk(hipMalloc((void**)&d_xcd_chunks, sizeof(uint64_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        size_t n_chunks = xcd_chunks[x].size();
        gpuErrchk(hipMalloc((void**)&d_xcd_chunks[x], sizeof(uint64_t) * n_chunks ));
        gpuErrchk(hipMemcpy(d_xcd_chunks[x], xcd_chunks[x].data(), sizeof(uint64_t) * n_chunks, hipMemcpyHostToDevice));
    }   

    /* count # per-xcd 4KB chunk */

    // ensure minimal 8MB = 2 * l2_size per-xcd chunks in order to thrash l2
    // conservative. since 2 xcds on same iod actually share the home data, another approach can be
    // ensure minimal 8MB per-iod chunks 
    vector<size_t> xcd_chunks_size(XCDS_NUM);
    const int min_n_chunks_per_xcd = 2 * 1024; // 4KB * 2k = 8MB
    int curr_min_n_chunks_per_xcd = INT_MAX;
    for (int xcd = 0; xcd < XCDS_NUM; xcd++) {
        xcd_chunks_size[xcd] = xcd_chunks[xcd].size();
        if (xcd_chunks[xcd].size() < curr_min_n_chunks_per_xcd) curr_min_n_chunks_per_xcd = xcd_chunks[xcd].size();
        printf("xcd %d has %zu 4KB chunks (%.2f MB)\n", xcd, xcd_chunks[xcd].size(), (xcd_chunks[xcd].size() * 4.0) / 1024);
    }
    assert(curr_min_n_chunks_per_xcd >= min_n_chunks_per_xcd);
    
    size_t *d_xcd_chunks_size;
    gpuErrchk(hipMalloc((void**)&d_xcd_chunks_size, sizeof(size_t) * XCDS_NUM));
    gpuErrchk(hipMemcpy(d_xcd_chunks_size, xcd_chunks_size.data(), sizeof(size_t) * XCDS_NUM, hipMemcpyHostToDevice));

    /* allocate cycles_k array */

    uint32_t **d_cycles_k; // per-xcd, per-tb in xcd. record cycles for bw measurement
    gpuErrchk(hipMalloc((void**)&d_cycles_k, sizeof(uint32_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMalloc((void**)&d_cycles_k[x], sizeof(uint32_t) * TPX ));
    }

    /* launch bw saturating kernel */

    hipLaunchKernelGGL(
        k,
        dim3(n_blocks), dim3(n_threads_per_block), 
        0, stream,
        d_xcd_chunks, 
        d_xcd_chunks_size,
        d_cycles_k
    );
    gpuErrchk(hipDeviceSynchronize());

    /* retrieve and process results */

    uint32_t h_cycles_k[XCDS_NUM][TPX];
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMemcpy(&h_cycles_k[x][0], d_cycles_k[x], sizeof(uint32_t) * TPX, hipMemcpyDeviceToHost));
    }

    // print in GB/s per tb
    for (int xcd = 0; xcd < XCDS_NUM; xcd++) {
        uint32_t max_xcd_cycles = 0;

        for (int tbid_in_xcd = 0; tbid_in_xcd < TPX; tbid_in_xcd++) {
            // find max cycles among tbs in this xcd
            if (h_cycles_k[xcd][tbid_in_xcd] > max_xcd_cycles) {
                max_xcd_cycles = h_cycles_k[xcd][tbid_in_xcd];
            }

            #if DEBUG
            {
                uint32_t cycles = h_cycles_k[xcd][tbid_in_xcd];
                double time_sec = (double)cycles / 2.1e9; // 2.1GHz
                double bytes = (double)((xcd_chunks_size[xcd] / TPX) * CHUNK_SIZE); // total bytes accessed
                double bw_GBps = (bytes / time_sec) / 1e9;
                printf("xcd %d, tb %d: cycles %u, time %.6f sec, bytes %.2f MB\n", xcd, tbid_in_xcd, cycles, time_sec, bytes / (1024*1024));
            }
            #endif
        }

        // print bw based on max cycles among tbs in this xcd
        double time_sec = (double)max_xcd_cycles / 2.1e9; // 2.1GHz
        double bytes = (double)(xcd_chunks_size[xcd] * CHUNK_SIZE); // total bytes accessed
        double bw_GBps = (bytes / time_sec) / 1e9;
        printf("xcd %d: max cycles %u, time %.6f sec, bytes %.2f MB, bw %.2f GB/s\n", xcd, max_xcd_cycles, time_sec, bytes / (1024*1024), bw_GBps);
    }   
    
    /* cleanup */
    gpuErrchk(hipFree(d_data));

    return 0;
}