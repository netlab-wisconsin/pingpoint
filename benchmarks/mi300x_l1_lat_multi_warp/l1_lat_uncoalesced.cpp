// A *multi* warp (n warps * 64 threads) pchase
// each thread accesses strided elements to disable coalescing

// Each array element points to +1 index array element
// Each thread starts pchase from array elements strided by $stride = 2, 4, 8, 16, 32, 64

#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
#include <cassert>

/* you can change */
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 16 // number of warps per block
#endif
#ifndef REPEAT_TIMES
#define REPEAT_TIMES 4096
#endif
#ifndef STRIDE
#define STRIDE 64
#endif

/* don't change */
#define THREADS_PER_WARP 64
#define THREADS_PER_BLOCK (THREADS_PER_WARP*WARPS_PER_BLOCK)
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define L1_SIZE 4096 // L1 size in 64-bit. MI300X L1 size is 32KB, i.e., 4K of 64-bit 
#define ARRAY_SIZE 4096
#define MAX_CLOCK 2100000 // MI300X max clock is 2.1GHz, i.e., 2100000 khz clk

#define LOG_LEVEL 0

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void l1_lat(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint32_t posArraySize, uint32_t stride)
{
    const uint32_t wid = (threadIdx.x / THREADS_PER_WARP);
    const uint32_t wtid = (threadIdx.x % THREADS_PER_WARP);
    const uint32_t tid = threadIdx.x;
    const uint32_t uid = blockIdx.x * blockDim.x + tid;

    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    #if LOG_LEVEL >= 2
    printf("wid %d wtid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", wid, wtid, xcc_id, se_id, cu_id);
    #endif

    /* Warmup L1 cache */

    // thread 0 initializes the pointer-chasing array
    // pointer chasing stride is warp_size. e.g., p[0]->p[64*stride], p[1]->p[1+64*stride], ...
    if (tid == 0) {
        const uint32_t stride = THREADS_PER_WARP * STRIDE;
        for (uint32_t i = 0; i<posArraySize - stride; i++) {
            posArray[i] = (uint64_t)(posArray + i + stride);
        }
        for (uint32_t i = posArraySize - stride; i<posArraySize; i++) {
            posArray[i] = (uint64_t)(posArray + (i - (posArraySize - stride)));
        }
    }
    asm volatile ("s_barrier"); // synchronize all threads

    /* >8 End Warmup L1 cache. */

    /* Repeat L1 cache access */

    // each adjacent thread starts from strided elements
    // this disables coalescing at TA
    uint64_t *ptr = posArray + ((wid * THREADS_PER_WARP * stride) + (wtid * stride)) % posArraySize;
    #if LOG_LEVEL >= 2
    printf("wid %d wtid %d: start ptr_idx %d\n", wid, wtid, 0 + ((wid * THREADS_PER_WARP * stride) + (wtid * stride)) % posArraySize);
    #endif
    uint64_t ptr1, ptr0;

    asm volatile (
        "flat_load_dwordx2 %0, %1\n\t"
        "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
        : "=v"(ptr1) : "v"(ptr): "memory");

    // start timing
    uint32_t start = 0;
    start = __builtin_readcyclecounter();
    asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME

    // a thread loads 1 64bit element per iteration, strided by "1" across iterations
    for(uint32_t i=0; i<REPEAT_TIMES; ++i) {
        asm volatile (
            "flat_load_dwordx2 %0, %1\n\t"
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            : "=v"(ptr0) : "v"((uint64_t*)ptr1) : "memory");
        ptr1 = ptr0;    //swap the register for the next load
    }

    // stop timing
    uint32_t stop = 0;
    stop = __builtin_readcyclecounter();
    asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME */
    
    // write time and data back to memory
    startClk[uid] = start;
    stopClk[uid] = stop;
}

int main(){
    printf("multi warp uncoalesced pchase: n_warps=%d, array_size=%d, repeat=%d, stride=%d\n", WARPS_PER_BLOCK, ARRAY_SIZE, REPEAT_TIMES, STRIDE);
    assert (ARRAY_SIZE <= L1_SIZE); // ensure all arrays fit in L1

    uint32_t *startClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));
    uint32_t *stopClk = (uint32_t*) malloc(TOTAL_THREADS*sizeof(uint32_t));

    uint32_t *startClk_g;
    uint32_t *stopClk_g;
    uint64_t *posArray_g;

    gpuErrchk( hipMalloc(&startClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
    gpuErrchk( hipMalloc(&stopClk_g, TOTAL_THREADS*sizeof(uint32_t)) );
    gpuErrchk( hipMalloc(&posArray_g, ARRAY_SIZE*sizeof(uint64_t) + 0x1000) );

    // page aligning
    posArray_g = (uint64_t*)(((uint64_t)posArray_g & ~(0x0FFF)) + 0x1000);
    
    hipLaunchKernelGGL(l1_lat, dim3(BLOCKS_NUM), dim3(THREADS_PER_BLOCK), 0, 0, startClk_g, stopClk_g, posArray_g, ARRAY_SIZE, STRIDE);

    gpuErrchk( hipPeekAtLastError() );

    gpuErrchk( hipMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), hipMemcpyDeviceToHost) );

    /* Set clock rate */

    int clock_instruction_rate;
    bool use_max_clock_rate = true;
    if (use_max_clock_rate) {
        clock_instruction_rate = MAX_CLOCK;
    } else {
        // use prop clock rate
        hipDeviceProp_t prop;
        gpuErrchk( hipGetDeviceProperties(&prop, 0) );//1000000khz clk
        clock_instruction_rate = prop.clockInstructionRate;
    }

    /* >8 End Set clock rate. */

    /* Calculate average latency */

    // avg latency in cycles
    double avg_latency_per_load = 0, avg_accum_latency_over_repeats = 0;
    long long count = 0;
    for (uint32_t i=0; i<TOTAL_THREADS; ++i) {
        count++;
        avg_accum_latency_over_repeats += ((stopClk[i]-startClk[i]) - avg_accum_latency_over_repeats) / count;
        #if LOG_LEVEL >= 1
        printf("wid: %d wtid: %d latency: %d clk\n", i / THREADS_PER_WARP, i % THREADS_PER_WARP, (int)((stopClk[i]-startClk[i]) / REPEAT_TIMES));
        #endif
    }
    avg_latency_per_load = avg_accum_latency_over_repeats / REPEAT_TIMES;

    /* >8 End Calculate average latency. */

    printf("L1 latency (per %luB load) = %d (clk) %d (ns @ %.1f GHz)\n", sizeof(uint64_t), (int)avg_latency_per_load, (int)(avg_latency_per_load * 1e6 / clock_instruction_rate), (double)clock_instruction_rate / 1e6);

    return 0;
}