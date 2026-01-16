// single threaded pchase

// A single thread pchase sequentially across the array
// Each array element poinits to +1 index array element
// $repeat = 4096 (1x full traversal) and 8192 (2x full traversal)

// Troubleshooting notes

// Using ARRAY_SIZE directly inside the kernel, instead of passing it as a parameter,
// somehow avoids L1 cache access while initializing the pointer-chasing array and 
// results in reduced TCP_TOTAL_CACHE_ACCESSES_sum in rocprof output. While this does
// not affect the latency measurement, try to pass by parameter when verifying the 
// memory access pattern using rocprof.

#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime.h"
#include <cassert>

/* don't change */
#define THREADS_PER_BLOCK 1
#define BLOCKS_NUM 1
#define TOTAL_THREADS (THREADS_PER_BLOCK*BLOCKS_NUM)
#define L1_SIZE 4096 // L1 size in 64-bit. MI300X L1 size is 32KB, i.e., 4K of 64-bit 
#define MAX_CLOCK 2100000 // MI300X max clock is 2.1GHz, i.e., 2100000 khz clk

/* you can change */
#ifndef ARRAY_SIZE
#define ARRAY_SIZE 4096 // array size e.g., 512 * 8B = 4KB < L1 size (32KB), to access only VL1D
#endif
#ifndef REPEAT_TIMES
#define REPEAT_TIMES 4096
#endif

#define DEBUG 0

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void l1_lat(uint32_t *startClk, uint32_t *stopClk, uint64_t *posArray, uint32_t posArraySize){
    uint32_t tid = threadIdx.x;
    uint32_t uid = blockIdx.x * blockDim.x + tid;
    assert (uid == 0 && tid == 0); // only one thread
    
    /* Warmup L1 cache */

    // thread 0 initializes the pointer-chasing array
    // circular linked list
    if (tid == 0) {
        uint32_t i = 0;
        for (; i<posArraySize-1; i++) {
            posArray[i] = (uint64_t)(posArray + i + 1);
        }
        posArray[i] = (uint64_t)posArray;
    }
    asm volatile ("s_barrier"); // synchronize all threads

    /* >8 End Warmup L1 cache. */

    /* Repeat L1 cache access */

    uint64_t *ptr = posArray;
    uint64_t ptr1, ptr0;

    asm volatile (
        "flat_load_dwordx2 %0, %1\n\t"
        "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
        : "=v"(ptr1) : "v"(ptr): "memory");
    asm volatile ("s_barrier"); // synchronize all threads

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
    printf("pchase: array_size=%d, repeat=%d\n", ARRAY_SIZE, REPEAT_TIMES);
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

    hipLaunchKernelGGL(l1_lat, dim3(BLOCKS_NUM), dim3(THREADS_PER_BLOCK), 0, 0, startClk_g, stopClk_g, posArray_g, ARRAY_SIZE);

    gpuErrchk( hipPeekAtLastError() );

    gpuErrchk( hipMemcpy(startClk, startClk_g, TOTAL_THREADS*sizeof(uint32_t), hipMemcpyDeviceToHost) );
    gpuErrchk( hipMemcpy(stopClk, stopClk_g, TOTAL_THREADS*sizeof(uint32_t), hipMemcpyDeviceToHost) );

    // avg latency in cycles
    float accum_latency_over_repeats = (float)(stopClk[0]-startClk[0]);
    float avg_latency_per_repeat = accum_latency_over_repeats / REPEAT_TIMES;

    // set clock rate
    int clock_instruction_rate;
    bool use_max_clock_rate = true;
    if (use_max_clock_rate) {
        clock_instruction_rate = MAX_CLOCK; // 2100000 khz clk
    } else {
        // use prop clock rate
        hipDeviceProp_t prop;
        gpuErrchk( hipGetDeviceProperties(&prop, 0) );//1000000khz clk
        clock_instruction_rate = prop.clockInstructionRate;
    }

    printf("L1 latency (per %luB load) = %d (clk) %d (ns) @ %.1f GHz\n", sizeof(uint64_t), (int)avg_latency_per_repeat, (int)(avg_latency_per_repeat * 1e6 / clock_instruction_rate), (double)clock_instruction_rate / 1e6);

    // printf("Accumulated L1 latency over repeats = %d (clk) %d (ns) @ %.1f GHz\n", (int)accum_latency_over_repeats, (int)(accum_latency_over_repeats * 1e6 / clock_instruction_rate), (double)clock_instruction_rate / 1e6);

    return 0;
}