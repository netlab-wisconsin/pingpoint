#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef XCD_NUM
#define XCD_NUM 8
#endif

// MI350X topology: 8 XCDs organized as 2 CCs.
#ifndef CC_NUM
#define CC_NUM 2
#endif

#define DEBUG 0

#define USE_GLOBAL_BARRIER 0

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

typedef int64_t dtype;

template <typename T>
__global__ void identify_home(T *data, uint32_t *cycles, const long long n_dtypes) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    assert(XCD_NUM == gridDim.x);
    assert(blockDim.x == 128 / sizeof(T));

    // printf("blockDim.x: %d, n_dtypes: %lld\n", blockDim.x, n_dtypes);

    // for this bmk, both global_barrier and cooperative_groups are supported
    #if not USE_GLOBAL_BARRIER
    cg::grid_group grid = cg::this_grid();
    #endif

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

    const size_t n_iter = n_dtypes / blockDim.x;
    const size_t inner_size = min(n_dtypes * sizeof(T), 64 * 1024 * 1024); // 64MB per inner loop (given TLB lat jump at 64MB)
    // const size_t n_outer = (sizeof(int64_t) * n_dtypes) / inner_size;
    const size_t n_outer = ((sizeof(int64_t) * n_dtypes) + inner_size - 1) / inner_size;
    const size_t n_inner = inner_size / (sizeof(int64_t) * blockDim.x); // num of accesses per tb in inner loop
    // printf("n_outer: %zu, n_inner: %zu, n_iter: %zu\n", n_outer, n_inner, n_iter);
    assert (n_iter <= n_outer * n_inner);
    
    for (size_t i = 0; i < n_outer; i++) {
        // warmup 
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (index >= n_dtypes) continue; // boundary check
            asm volatile(
                "flat_load_dwordx2 v[0:1], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data[index])
                : "memory", "v0", "v1"
            );
        }
        #if USE_GLOBAL_BARRIER
        global_barrier();
        #else
        grid.sync();
        #endif

        // measurement
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (index >= n_dtypes) continue; // boundary check
            uint32_t start = __builtin_readcyclecounter();
            asm volatile(
                "flat_load_dwordx2 v[0:1], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data[index])
                : "memory", "v0", "v1"
            );
            uint32_t end = __builtin_readcyclecounter();
            uint32_t cycle = end - start;
            const size_t cycles_index = (size_t)xcc_id * n_dtypes + index; // (01/25/25) fix: from int to size_t, to avoid overflow
            cycles[cycles_index] = cycle;
            #if USE_GLOBAL_BARRIER
            global_barrier();
            #else
            grid.sync();
            #endif
        }
    }
}
