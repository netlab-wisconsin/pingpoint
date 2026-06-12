// cu masking

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

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

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 8
#endif

#ifndef BLOCKS_NUM
#define BLOCKS_NUM 32
#endif

constexpr int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

#ifndef ENABLED_CUS_PER_XCD_NUM
#define ENABLED_CUS_PER_XCD_NUM 4
#endif

#ifndef MASK_MODE
#define MASK_MODE 0
#endif

#ifndef MASK_WORD
#define MASK_WORD 0
#endif

#ifndef MASK_VALUE
#define MASK_VALUE 0u
#endif

constexpr bool DEBUG = false;

__global__ void k() {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    // https://github.com/ROCm/ROCm/issues/2059 Obtain CU physical ID from HIP kernel?
    // https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf 3.12. Hardware ID Registers
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    if (tid == 0) {
        printf("bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", bid, xcc_id, se_id, cu_id);
    }
}

// Used below code as a reference: 
// https://github.com/hibagus/SC2025_TransferBench/blob/90927c69e79e9952b75cb050d446a076faaefce6/src/client/EnvVars.hpp
// Below discussion talks about it too:
// https://github.com/ROCm/ROCm/issues/1862 valid bits in CU mask API hipExtStreamCreateWithCUMask
int main() {
    hipError_t err;
    hipStream_t stream;

    /* create cu mask */

    uint32_t cuMaskSize;
    vector<uint32_t> cuMask;
    #if MASK_MODE
    if (mask_one(MASK_WORD, MASK_VALUE, cuMaskSize, cuMask) < 0) {
        return 1;
    }
    cout << "mode=single"
         << " mask_word=" << MASK_WORD
         << " mask_value=0x" << hex << MASK_VALUE << dec
         << " blocks=" << BLOCKS_NUM
         << " threads_per_block=" << THREADS_PER_BLOCK << endl;
    #else
    if (mask_cu(ENABLED_CUS_PER_XCD_NUM, cuMaskSize, cuMask) < 0) {
        return 1;
    }
    cout << "mode=cumulative"
         << " enabled_cus_per_xcd=" << ENABLED_CUS_PER_XCD_NUM
         << " blocks=" << BLOCKS_NUM
         << " threads_per_block=" << THREADS_PER_BLOCK << endl;
    #endif

    /* create cu masked stream & launch kernel */
    
    err = hipExtStreamCreateWithCUMask(&stream, cuMaskSize, cuMask.data());
    if (err != hipSuccess) {
        cerr << "Error creating stream with CU mask: " << hipGetErrorString(err) << endl;
        return 1;
    }

    hipLaunchKernelGGL(
        k,
        dim3(BLOCKS_NUM), dim3(THREADS_PER_BLOCK), 
        0, stream
    );

    gpuErrchk(hipStreamSynchronize(stream));
    gpuErrchk(hipStreamDestroy(stream));
    
    return 0;
}
