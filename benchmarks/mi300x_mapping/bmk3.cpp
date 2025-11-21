// cu masking

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr int THREADS_PER_BLOCK = 8; // you can change
constexpr int BLOCKS_NUM = 16; // you can change
constexpr int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

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
    printf("bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", bid, xcc_id, se_id, cu_id);
}

// Used below code as a reference: 
// https://github.com/hibagus/SC2025_TransferBench/blob/90927c69e79e9952b75cb050d446a076faaefce6/src/client/EnvVars.hpp
// Below discussion talks about it too:
// https://github.com/ROCm/ROCm/issues/1862 valid bits in CU mask API hipExtStreamCreateWithCUMask
int main() {
    hipError_t err;
    hipStream_t stream;

    // Get device properties to determine the number of CUs
    hipDeviceProp_t props;
    int deviceId = 0; // Assuming device 0
    err = hipGetDeviceProperties(&props, deviceId);
    if (err != hipSuccess) {
        std::cerr << "Error getting device properties: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    /* create cu mask */
    uint32_t cuMaskSize = (props.multiProcessorCount + 31) / 32;
    std::vector<uint32_t> cuMask(cuMaskSize, 0); // Initialize all CUs as disabled. Use (uint32_t)-1 to enable all.

    // mi300x cu mask structure
    // mi300x has 8 xcds, each xcd has 4 ses, and each se has 10 cus (but only 9 cus are usable, or at least controllable via cu mask)
    // the 4 ses are indexed 3,2,1,0. xcds and cus are indexed normally.
    // cu mask bit layout is as follows:
    // cuMask[0]: x0s3c0, x1s3c0, x2s3c0, x3s3c0, x4s3c0, x5s3c0, x6s3c0, x7s3c0 => 0xff000000
    //            x0s2c0, x1s2c0, x2s2c0, x3s2c0, x4s2c0, x5s2c0, x6s2c0, x7s2c0 => 0x00ff0000
    //            x0s1c0, x1s1c0, x2s1c0, x3s1c0, x4s1c0, x5s1c0, x6s1c0, x7s1c0 => 0x0000ff00
    //            x0s0c0, x1s0c0, x2s0c0, x3s0c0, x4s0c0, x5s0c0, x6s0c0, x7s0c0 => 0x000000ff
    // cuMask[1]: x0s3c1, x1s3c1, x2s3c1, x3s3c1, x4s3c1, x5s3c1, x6s3c1, x7s3c1 => 0xff000000
    //            x0s2c1, x1s2c1, x2s2c1, x3s2c1, x4s2c1, x5s2c1, x6s2c1, x7s2c1 => 0x00ff0000
    //            x0s1c1, x1s1c1, x2s1c1, x3s1c1, x4s1c1, x5s1c1, x6s1c1, x7s1c1 => 0x0000ff00
    //            x0s0c1, x1s0c1, x2s0c1, x3s0c1, x4s0c1, x5s0c1, x6s0c1, x7s0c1 => 0x000000ff
    // ...
    // cuMask[9]: x0s3c9, x1s3c9, x2s3c9, x3s3c9, x4s3c9, x5s3c9, x6s3c9, x7s3c9 => 0xff000000
    //            x0s2c9, x1s2c9, x2s2c9, x3s2c9, x4s2c9, x5s2c9, x6s2c9, x7s2c9 => 0x00ff0000
    //            x0s1c9, x1s1c9, x2s1c9, x3s1c9, x4s1c9, x5s1c9, x6s1c9, x7s1c9 => 0x0000ff00
    //            x0s0c9, x1s0c9, x2s0c9, x3s0c9, x4s0c9, x5s0c9, x6s0c9, x7s0c9 => 0x000000ff


    // cuMask[0] = 0xff000000; printf("cuMask=%u enable cu0 across 8 xcds\n", cuMask[0]); // cuid=0 seid=3
    // cuMask[0] = 0x00ff0000; printf("cuMask=%u enable cu1 across 8 xcds\n", cuMask[0]); // cuid=0 seid=2
    // cuMask[0] = 0x0000ff00; printf("cuMask=%u enable cu2 across 8 xcds\n", cuMask[0]); // cuid=0 seid=1
    // cuMask[0] = 0x000000ff; printf("cuMask=%u enable cu3 across 8 xcds\n", cuMask[0]); // cuid=0 seid=0

    // cuMask[1] = 0xff000000; printf("cuMask[1]=%u enable cu4 across 8 xcds\n", cuMask[1]); // cuid=1  seid=3
    // cuMask[1] = 0x00ff0000; printf("cuMask[1]=%u enable cu5 across 8 xcds\n", cuMask[1]); // cuid=1  seid=2
    // cuMask[1] = 0x0000ff00; printf("cuMask[1]=%u enable cu6 across 8 xcds\n", cuMask[1]); // cuid=1  seid=1
    // cuMask[1] = 0x000000ff; printf("cuMask[1]=%u enable cu7 across 8 xcds\n", cuMask[1]); // cuid=1  seid=0

    // cuMask[2] = 0xff000000; printf("cuMask[2]=%u enable cu8 across 8 xcds\n", cuMask[2]); // cuid=2  seid=3
    // cuMask[2] = 0x00ff0000; printf("cuMask[2]=%u enable cu9 across 8 xcds\n", cuMask[2]); // cuid=2  seid=2
    // cuMask[2] = 0x0000ff00; printf("cuMask[2]=%u enable cu10 across 8 xcds\n", cuMask[2]); // cuid=2  seid=1
    // cuMask[2] = 0x000000ff; printf("cuMask[2]=%u enable cu11 across 8 xcds\n", cuMask[2]); // cuid=2  seid=0

    // cuMask[3] = 0xff000000; printf("cuMask[3]=%u enable cu12 across 8 xcds\n", cuMask[3]); // cuid=3  seid=3
    // cuMask[3] = 0x00ff0000; printf("cuMask[3]=%u enable cu13 across 8 xcds\n", cuMask[3]); // cuid=3  seid=2
    // cuMask[3] = 0x0000ff00; printf("cuMask[3]=%u enable cu14 across 8 xcds\n", cuMask[3]); // cuid=3  seid=1
    // cuMask[3] = 0x000000ff; printf("cuMask[3]=%u enable cu15 across 8 xcds\n", cuMask[3]); // cuid=3  seid=0

    // cuMask[4] = 0xff000000; printf("cuMask[4]=%u enable cu16 across 8 xcds\n", cuMask[4]); // cuid=4  seid=3
    // cuMask[4] = 0x00ff0000; printf("cuMask[4]=%u enable cu17 across 8 xcds\n", cuMask[4]); // cuid=4  seid=2
    // cuMask[4] = 0x0000ff00; printf("cuMask[4]=%u enable cu18 across 8 xcds\n", cuMask[4]); // cuid=4  seid=1
    // cuMask[4] = 0x000000ff; printf("cuMask[4]=%u enable cu19 across 8 xcds\n", cuMask[4]); // cuid=4  seid=0

    // cuMask[5] = 0xff000000; printf("cuMask[5]=%u enable cu20 across 8 xcds\n", cuMask[5]); // cuid=5  seid=3
    // cuMask[5] = 0x00ff0000; printf("cuMask[5]=%u enable cu21 across 8 xcds\n", cuMask[5]); // cuid=5  seid=2
    // cuMask[5] = 0x0000ff00; printf("cuMask[5]=%u enable cu22 across 8 xcds\n", cuMask[5]); // cuid=5  seid=1
    // cuMask[5] = 0x000000ff; printf("cuMask[5]=%u enable cu23 across 8 xcds\n", cuMask[5]); // cuid=5  seid=0

    // cuMask[6] = 0xff000000; printf("cuMask[6]=%u enable cu24 across 8 xcds\n", cuMask[6]); // cuid=6  seid=3
    // cuMask[6] = 0x00ff0000; printf("cuMask[6]=%u enable cu25 across 8 xcds\n", cuMask[6]); // cuid=6  seid=2
    // cuMask[6] = 0x0000ff00; printf("cuMask[6]=%u enable cu26 across 8 xcds\n", cuMask[6]); // cuid=6  seid=1
    // cuMask[6] = 0x000000ff; printf("cuMask[6]=%u enable cu27 across 8 xcds\n", cuMask[6]); // cuid=6  seid=0

    // cuMask[7] = 0xff000000; printf("cuMask[7]=%u enable cu28 across 8 xcds\n", cuMask[7]); // cuid=7  seid=3
    // cuMask[7] = 0x00ff0000; printf("cuMask[7]=%u enable cu29 across 8 xcds\n", cuMask[7]); // cuid=7  seid=2
    // cuMask[7] = 0x0000ff00; printf("cuMask[7]=%u enable cu30 across 8 xcds\n", cuMask[7]); // cuid=7  seid=1
    // cuMask[7] = 0x000000ff; printf("cuMask[7]=%u enable cu31 across 8 xcds\n", cuMask[7]); // cuid=7  seid=0

    // cuMask[8] = 0xff000000; printf("cuMask[8]=%u enable cu32 across 8 xcds\n", cuMask[8]); // cuid=8  seid=3
    // cuMask[8] = 0x00ff0000; printf("cuMask[8]=%u enable cu33 across 8 xcds\n", cuMask[8]); // cuid=8  seid=2
    // cuMask[8] = 0x0000ff00; printf("cuMask[8]=%u enable cu34 across 8 xcds\n", cuMask[8]); // cuid=8  seid=1
    // cuMask[8] = 0x000000ff; printf("cuMask[8]=%u enable cu35 across 8 xcds\n", cuMask[8]); // cuid=8  seid=0
    
    // cuMask[9] = 0xff000000; printf("cuMask[9]=%u enable cu36 across 8 xcds\n", cuMask[9]); // cuid=9  seid=3
    // cuMask[9] = 0x00ff0000; printf("cuMask[9]=%u enable cu37 across 8 xcds\n", cuMask[9]); // cuid=9  seid=2
    // cuMask[9] = 0x0000ff00; printf("cuMask[9]=%u enable cu38 across 8 xcds\n", cuMask[9]); // cuid=9  seid=1
    // cuMask[9] = 0x000000ff; printf("cuMask[9]=%u enable cu39 across 8 xcds\n", cuMask[9]); // cuid=9  seid=0

    /* 8> end of create cu mask */

    /* create cu masked stream & launch kernel */
    
    err = hipExtStreamCreateWithCUMask(&stream, cuMaskSize, cuMask.data());
    if (err != hipSuccess) {
        std::cerr << "Error creating stream with CU mask: " << hipGetErrorString(err) << std::endl;
        return 1;
    }

    hipLaunchKernelGGL(
        k,
        dim3(BLOCKS_NUM), dim3(THREADS_PER_BLOCK), 
        0, stream
    );

    gpuErrchk(hipStreamSynchronize(stream));
    gpuErrchk(hipStreamDestroy(stream));

    /* >8 end of create cu masked stream & launch kernel */
    
    return 0;
}