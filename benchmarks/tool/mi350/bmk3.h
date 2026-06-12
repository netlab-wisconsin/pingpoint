#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

// ncus: #cus enabled per xcd
// returns total #cus enabled, -1 on error
int mask_cu(const size_t ncus, uint32_t &cuMaskSize, vector<uint32_t> &cuMask) 
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
    cuMask.resize(cuMaskSize, 0); // Initialize all CUs as disabled. Use (uint32_t)-1 to enable all.

    /* cu masking rule */

    // MI350 CU mask structure validated by bmk3_wordXX_YYYYYYYY sweeps.
    // Each byte lane enables the same (se, cu) on all 8 XCDs.
    //
    // For cuMask[w], w in [0, 7]:
    //   0xff000000 -> x0..x7, se3, cu(w + 1)
    //   0x00ff0000 -> x0..x7, se2, cu(w)
    //   0x0000ff00 -> x0..x7, se1, cu(w + 1)
    //   0x000000ff -> x0..x7, se0, cu(w)
    //
    // Therefore the cumulative loop below enables CUs in this order:
    //   se3/cu1, se2/cu0, se1/cu1, se0/cu0,
    //   se3/cu2, se2/cu1, se1/cu2, se0/cu1, ...

    /* 8> end of cu masking rule */

    const size_t n_cus_enabled_per_xcd = min(ncus, 32); // max 32 cus can be enabled per xcd
    for (int i = 0; i < n_cus_enabled_per_xcd; ++i) { 
        cuMask[i/4] |= (0xffu << ((3-(i%4)) * 8));
    }
    
    return n_cus_enabled_per_xcd;
}

int mask_one(const size_t mask_word, const uint32_t mask_value,
             uint32_t &cuMaskSize, vector<uint32_t> &cuMask)
{
    hipDeviceProp_t props;
    int deviceId = 0;
    hipError_t err = hipGetDeviceProperties(&props, deviceId);
    if (err != hipSuccess) {
        cerr << "Error getting device properties: " << hipGetErrorString(err) << endl;
        return -1;
    }

    cuMaskSize = (props.multiProcessorCount + 31) / 32;
    cuMask.assign(cuMaskSize, 0);
    if (mask_word >= cuMask.size()) {
        cerr << "MASK_WORD " << mask_word << " is out of range for cuMaskSize "
             << cuMaskSize << endl;
        return -1;
    }

    cuMask[mask_word] = mask_value;
    return 0;
}
