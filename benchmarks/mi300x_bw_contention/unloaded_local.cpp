#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#ifndef TPX
#define TPX 1
#endif

using namespace std;
namespace cg = cooperative_groups;

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr int XCDS_NUM = 8;

constexpr int THREADS_PER_WARP = 64;
// constexpr int WARPS_PER_BLOCK = 16;  
constexpr int WARPS_PER_BLOCK = 4;

constexpr int PAGE_SIZE = (2 * 1024 * 1024); // 2MB huge page
constexpr int CHUNK_SIZE = (4 * 1024); // 4KB chunk size

struct chunk4KB {
    char data[CHUNK_SIZE];
};

#define DEBUG 0


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


__global__ void k(void *data, size_t size, uint32_t **d_cycles, int *d_home) {
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
    size_t n_chunks = size / sizeof(uint4);
    size_t n_iter = n_chunks / (blockDim.x * n_tbs_in_xcd); // thread blocks on same xcd streams over the data chunks
    if (bid == 0 && tid == 0) {
        printf("n_chunks: %zu, n_iter: %zu\n", n_chunks, n_iter);
        printf("working set per xcd: %.2f MB\n", (n_chunks * sizeof(uint4) / XCDS_NUM) / (1024.0 * 1024.0));
    }

    uint4 tmp;
    
    // warm up
    for (size_t i = 0; i < n_iter; i++) {
        size_t index = (i * n_tbs_in_xcd + tbid_in_xcd) * blockDim.x + tid;
        if (tid == 0) {
            #if DEBUG
            printf("[warmup] (iter:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, bid, tbid_in_xcd, index, index + blockDim.x - 1);
            #endif
        }
        asm volatile(
            "flat_load_dwordx4 %0, %1\n\t" // might want l1 bypass
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            : "=v"(tmp)
            : "v"(&data_u4[index])
            : "memory"
        );
    }
    global_barrier();

    // latency measurement
    uint32_t start = 0, end = 0;
    uint32_t cycles[n_iter]; // temporary... move to host inititated 
    for (size_t i = 0; i < n_iter; i++) {
        size_t index = (i * n_tbs_in_xcd + tbid_in_xcd) * blockDim.x + tid;
        if (tid == 0) {
            #if DEBUG
            printf("[warmup] (iter:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, bid, tbid_in_xcd, index, index + blockDim.x - 1);
            #endif
        }
        start = __builtin_readcyclecounter();
        asm volatile(
            "flat_load_dwordx4 %0, %1\n\t" // might want l1 bypass
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            : "=v"(tmp)
            : "v"(&data_u4[index])
            : "memory"
        );
        end = __builtin_readcyclecounter();
        cycles[i] = end - start;

        global_barrier(); // sync globally between iters
    }
    global_barrier();

    // every thread block reads 4KB boundary at each iter
    if (tid == 0) {
        for (size_t i = 0; i < n_iter; i++) {
            d_cycles[xcc_id][i * n_tbs_in_xcd + tbid_in_xcd] = cycles[i];
            // #if DEBUG
            printf("iter %zu tbid_in_xcd %d (bid %d, xcd %d): %u cycles\n", i, tbid_in_xcd, bid, xcc_id, cycles[i]);
            // #endif
        }
    }
    global_barrier(); // sync global memory writes

    // get winner!
    if (xcc_id == 0 && tid == 0) {
        for (size_t i = 0; i < n_iter; i++) {
            uint32_t min_cycles = 0xFFFFFFFF;
            int min_xcc = -1;
            for (int xcc = 0; xcc < XCDS_NUM; xcc++) {
                uint32_t c = d_cycles[xcc][i * n_tbs_in_xcd + tbid_in_xcd];
                if (c < min_cycles) {
                    min_cycles = c;
                    min_xcc = xcc;
                }
            }
            d_home[i * n_tbs_in_xcd + tbid_in_xcd] = min_xcc;
        }
    }
    global_barrier(); // sync global memory writes
    
    if (bid == 0 && tid == 0) {
        for (size_t i = 0; i < size / (4 * 1024); i++) {
            printf("4KB chunk[%zu]: home xcd %d\n", i, d_home[i]);
        }
    }

}

int main() {

    /* configure thread blocks */

    const int n_blocks = (TPX * XCDS_NUM);
    const int n_threads_per_block = (WARPS_PER_BLOCK * THREADS_PER_WARP);
    const int total_threads = (n_blocks * n_threads_per_block);
    printf("n_blocks: %d, n_blocks_per_xcd: %d, n_warps_per_block: %d, n_threads_per_warp: %d\n", n_blocks, TPX, WARPS_PER_BLOCK, THREADS_PER_WARP);

    const int n_pages = 128; // := 256 MB to fill up LLC
    const size_t data_size = (n_pages * PAGE_SIZE); 
    printf("data_size: %d MB\n", (int)data_size / (1024 * 1024));

    /* allocate data */

    char *d_data;
    gpuErrchk(hipMalloc((void**)&d_data, data_size + 0x1000));
    d_data = (char*)(((uintptr_t)d_data & ~(0x0FFF)) + 0x1000); // page align

    /* allocate cycles array */

    uint32_t **d_cycles; // per-xcd, per-4KB cycles array
    gpuErrchk(hipMalloc((void**)&d_cycles, sizeof(uint32_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMalloc((void**)&d_cycles[x], sizeof(uint32_t) * (data_size / (4 * 1024)) ));
    }
    printf("allocated d_cycles array[%d][%d]\n", XCDS_NUM, (int)data_size / (4 * 1024));

    int *d_home; // home 'xcc' per 4KB 
    gpuErrchk(hipMalloc((void**)&d_home, sizeof(int) * (data_size / (4 * 1024)) ));

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

    hipLaunchKernelGGL(
        k,
        dim3(n_blocks), dim3(n_threads_per_block), 
        0, stream,
        (void*)d_data, data_size, d_cycles, d_home
    );
    gpuErrchk(hipDeviceSynchronize());

    /* cleanup */

    gpuErrchk(hipFree(d_data));

    return 0;
}