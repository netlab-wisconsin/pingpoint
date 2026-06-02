#pragma once

#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

#define XCD_NUM 8
#define CC_NUM  2
#define CU_NUM  32

#define L2_SIZE (4 * 1024 * 1024) // 4 MB
#define LLC_SIZE (256 * 1024 * 1024) // 256 MB

// GPU error check
#ifndef gpuErrchk
#define gpuErrchk(ans) { hipAssert((ans), __FILE__, __LINE__); }
inline void hipAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"HIPassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

inline uint32_t get_cc(uint32_t xcc_id) {
    return (xcc_id / (XCD_NUM / CC_NUM)) % CC_NUM;
}

// cu_active_mask: A 40-bit mask where the Nth bit enables the Nth CU.
//                 (e.g., bit 0 = CU0, bit 1 = CU1 ... bit 39 = CU39)
//                 This applies the same mask to all XCDs (0xff).
int mask_cu(const uint64_t cu_active_mask, uint32_t &cuMaskSize, vector<uint32_t> &cuMask) 
{
    hipError_t err;

    // Get device properties to determine the number of CUs
    hipDeviceProp_t props;
    int deviceId = 0; // Assuming device 0
    err = hipGetDeviceProperties(&props, deviceId);
    if (err != hipSuccess) {
        std::cerr << "Error getting device properties: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    cuMaskSize = (props.multiProcessorCount + 31) / 32;
    cuMask.assign(cuMaskSize, 0); 

    // The mask supports up to 40 CUs (indices 0 to 9 in the vector, 4 CUs per index)    
    int enabled_count = 0;
    for (int i = 0; i < CU_NUM; ++i) {
        if ((cu_active_mask >> i) & 1) {
            int idx = i / 4;
            if (idx >= cuMaskSize) continue;
            cuMask[idx] |= (0xffu << ((3-(i%4)) * 8));
            enabled_count++;
        }
    }
    
    return enabled_count;
}