#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef XCDS_NUM
#define XCDS_NUM 8 // num xcds in mi300x
#endif

__device__ int d_barrier_count = 0;
__device__ int d_barrier_sense = 0;

__device__ __forceinline__
void global_barrier()
{
    __shared__ int local_sense; // One local copy per block
    if (threadIdx.x == 0) {
        int old_sense = d_barrier_sense;
        local_sense   = !old_sense;

        int arrived = atomicAdd(&d_barrier_count, 1);
        if (arrived == gridDim.x - 1) {
            d_barrier_count = 0;
            __threadfence(); // Make all global writes visible before releasing others
            d_barrier_sense = local_sense;
        }
    }
    __syncthreads(); // Make sure all threads in the block see local_sense
    while (d_barrier_sense != local_sense) {} // Spin until last block flips global sense
    __syncthreads(); // Ensure all threads in this block observe the sense change before proceeding
}


__global__ void identify_home(void *data, size_t size, uint32_t **d_cycles, int *d_home) {
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

    uint4 *data_u4 = (uint4*)data;
    size_t n_16Bs = size / sizeof(uint4);
    size_t n_iter = n_16Bs / (blockDim.x * n_tbs_in_xcd); // thread blocks on same xcd streams over the data chunks
    if (bid == 0 && tid == 0) {
        printf("n_16Bs: %zu, n_iter: %zu\n", n_16Bs, n_iter);
        printf("working set per xcd: %.2f MB\n", (n_16Bs * sizeof(uint4) / XCDS_NUM) / (1024.0 * 1024.0));
    }

    // uint4 tmp;
    
    // warm up
    for (size_t i = 0; i < n_iter; i++) {
        size_t index = (i * n_tbs_in_xcd + tbid_in_xcd) * blockDim.x + tid;
        if (tid == 0) {
            #if DEBUG
            printf("[warmup] (iter:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, bid, tbid_in_xcd, index, index + blockDim.x - 1);
            #endif
        }
        asm volatile(
            // "flat_load_dwordx4 %0, %1\n\t" // might want l1 bypass
            "flat_load_dwordx4 v[0:3], %0\n\t" // might want l1 bypass
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            // : "=v"(tmp)
            :
            : "v"(&data_u4[index])
            : "memory", "v0", "v1", "v2", "v3"
        );
    }
    global_barrier();

    // latency measurement
    uint32_t start = 0, end = 0;
    uint32_t cycle = 0; // troubleshooting: cycles[n_iter] led to memory issues
    for (size_t i = 0; i < n_iter; i++) {
        size_t index = (i * n_tbs_in_xcd + tbid_in_xcd) * blockDim.x + tid;
        if (tid == 0) {
            #if DEBUG
            printf("[warmup] (iter:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, bid, tbid_in_xcd, index, index + blockDim.x - 1);
            #endif
        }
        start = __builtin_readcyclecounter();
        asm volatile(
            // "flat_load_dwordx4 %0, %1\n\t" // might want l1 bypass
            "flat_load_dwordx4 v[0:3], %0\n\t" // might want l1 bypass
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            // : "=v"(tmp)
            :
            : "v"(&data_u4[index])
            : "memory", "v0", "v1", "v2", "v3"
        );
        end = __builtin_readcyclecounter();
        cycle = end - start;
        if (tid == 0) {
            // every thread block reads chunk boundary at each iter
            // only tid 0 writes, since we've set tb data per iter == CHUNK_SIZE
            d_cycles[xcc_id][i * n_tbs_in_xcd + tbid_in_xcd] = cycle;
            #if DEBUG
            printf("iter %zu tbid_in_xcd %d (bid %d, xcd %d): %u cycles\n", i, tbid_in_xcd, bid, xcc_id, cycle);
            #endif
        }
        global_barrier(); // sync globally between iters
    }
}