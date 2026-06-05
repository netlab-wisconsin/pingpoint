#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

#ifndef XCDS_NUM
#define XCDS_NUM 8 // num xcds in mi300x
#endif

constexpr int PAGE_SIZE = (2 * 1024 * 1024); // 2MB huge page
// constexpr int CHUNK_SIZE = (2 * 1024); // 2KB chunk size
// constexpr int CHUNK_SIZE = (4 * 1024); // 4KB chunk size
constexpr int CHUNK_SIZE = (8 * 1024); // 8KB chunk size
constexpr int N_PAGES = (128); // you can change

constexpr int THREADS_PER_WARP = (64);
constexpr int WARPS_PER_BLOCK = (CHUNK_SIZE / (16 * THREADS_PER_WARP)); // one block per chunk
constexpr int TPX = (1); // thread blocks per xcd

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef LOG_CYCLE
#define LOG_CYCLE 0
#endif

#ifndef USE_GLOBAL_BARRIER
#define USE_GLOBAL_BARRIER 0
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

// note: 12/13 convert to 1d c_cycles and calculate index with n_chunks.
__global__ void identify_home(void *data, size_t size, uint32_t *d_cycles, int n_chunks) {
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

    // for this bmk, assert 1 tb per xcd
    assert(n_tbs_in_xcd * XCDS_NUM == gridDim.x); assert(n_tbs_in_xcd == 1);

    assert(blockDim.x == CHUNK_SIZE / sizeof(uint4));

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

    uint4 *data_u4 = (uint4*)data;
    const size_t n_16Bs = size / sizeof(uint4);
    const size_t n_iter = n_16Bs / (blockDim.x * n_tbs_in_xcd); // thread blocks on same xcd streams over the data chunks
    if (bid == 0 && tid == 0) {
        #if DEBUG
        printf("n_16Bs: %zu, n_iter: %zu\n", n_16Bs, n_iter);
        printf("working set per xcd: %.2f MB\n", (n_16Bs * sizeof(uint4) / XCDS_NUM) / (1024.0 * 1024.0));
        #endif
    }

    const size_t inner_size = 64 * 1024 * 1024; // 64MB per inner loop (given TLB lat jump at 64MB)
    const size_t n_outer = size / inner_size;
    const size_t n_inner = inner_size / (16 * blockDim.x); // num of accesses per tb in inner loop
    assert (n_iter == n_outer * n_inner);
    
    for (size_t i = 0; i < n_outer; i++) {
        // warmup 
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (tid == 0) {
                #if DEBUG
                printf("[warmup] (outer:%zu, inner:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, j, bid, tbid_in_xcd, index, index + blockDim.x - 1);
                #endif
            }
            asm volatile(
                "flat_load_dwordx4 v[0:3], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data_u4[index])
                : "memory", "v0", "v1", "v2", "v3"
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
            if (tid == 0) {
                #if DEBUG
                printf("[actual] (outer:%zu, inner:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, j, bid, tbid_in_xcd, index, index + blockDim.x - 1);
                #endif
            }
            uint32_t start = __builtin_readcyclecounter();
            asm volatile(
                "flat_load_dwordx4 v[0:3], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data_u4[index])
                : "memory", "v0", "v1", "v2", "v3"
            );
            uint32_t end = __builtin_readcyclecounter();
            uint32_t cycle = end - start;
            if (tid == 0) {
                if (xcc_id < XCDS_NUM) {
                    const int d_cycles_index = xcc_id * n_chunks + (i * n_inner + j) * n_tbs_in_xcd + tbid_in_xcd;
                    d_cycles[d_cycles_index] = cycle;
                }
                #if LOG_CYCLE
                printf("outer %zu inner %zu tbid_in_xcd %d (bid %d, xcd %d): %u cycles\n", i, j, tbid_in_xcd, bid, xcc_id, cycle);
                #endif
            }
            #if USE_GLOBAL_BARRIER
            global_barrier();
            #else
            grid.sync();
            #endif
        }
    }
}
