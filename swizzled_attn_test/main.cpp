#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cinttypes> 
#include <unistd.h>

// Local headers
#include "main.h"
#include "k1.h"
#include "k2.h"
#include "ppnt.h"         
#include "swizzled_attn.h" 

#define TARGET_BLOCKDIM_X (1024)

using namespace std;

// Helper to replace external GPU clock
unsigned int getGPUClock() {
    hipDeviceProp_t prop;
    gpuErrchk(hipGetDeviceProperties(&prop, 0));
    return prop.clockRate / 1000; // MHz
}

// -----------------------------------------------------------------------------------------
// Helper: NUMA-Aware Memory Allocation (Replicating moe.cpp logic)
// -----------------------------------------------------------------------------------------
// In moe.cpp, "Home Identification" is usually done by:
// 1. Determining which XCD/Memory Node corresponds to ID 'h'
// 2. Allocating memory that physically resides there.
// Since specific APIs (like k1::home_identification) might be complex, 
// we will assume standard HIP behavior: Memory allocated by a thread running on XCD 'N' 
// often prefers HBM 'N', OR we use specific hipMemPool APIs if available.
//
// However, looking at the provided moe.cpp snippet, it calls `ppnt::prepare_ping_data`.
// I will update ppnt.h to handle this correctly.

int main(int argc, char **argv) {
    // 1. Setup Device
    int deviceId = 0;
    hipSetDevice(deviceId);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, deviceId);
    
    unsigned int clock = prop.clockRate / 1000;
    cout << "[INFO] Device: " << prop.name << " | Clock: " << clock << " MHz" << endl;

    hipStream_t stream;
    gpuErrchk(hipStreamCreate(&stream));

    // =========================================================================
    // Occupancy Calculation (Robust)
    // =========================================================================
    int numBlocksPerSm = 0;
    gpuErrchk(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, 
        ppnt::fused_kernel<AttnSwizzleTargetFn, AttnSwizzleArgs>, 
        TARGET_BLOCKDIM_X, 
        0));

    int max_active_blocks = numBlocksPerSm * prop.multiProcessorCount;
    int requested_grid = XCD_NUM * 320; 
    int physical_grid_size = std::min(requested_grid, max_active_blocks);
    // Align to XCD_NUM to ensure even distribution
    physical_grid_size = (physical_grid_size / XCD_NUM) * XCD_NUM;

    cout << "[INFO] Grid Size: " << physical_grid_size 
         << " (" << (physical_grid_size/XCD_NUM) << " blocks/XCD)" << endl;

    // 2. PPNT Setup (Replicating moe.cpp full setup)
    int n_plan = 0;
    ppnt::PingSpec *h_plan = nullptr;
    ppnt::PingSpec *d_plan = nullptr;
    ppnt::PingOut  *d_out  = nullptr;

    // Generate Plans (Now handled inside ppnt.h similar to moe.cpp)
    ppnt::create_plans(h_plan, n_plan, clock);
    cout << "[INFO] Generated " << n_plan << " profiling plans (Full Grid)." << endl;
    
    // Allocate Plan Spec on Device
    gpuErrchk(hipMalloc(&d_plan, n_plan * sizeof(ppnt::PingSpec)));
    
    // Allocate Outputs
    gpuErrchk(hipMalloc(&d_out, n_plan * sizeof(ppnt::PingOut)));
    for (int i=0; i<n_plan; ++i) {
        uint64_t* d_iterClk;
        gpuErrchk(hipMalloc(&d_iterClk, h_plan[i].iters * sizeof(uint64_t)));
        gpuErrchk(hipMemcpy(&(d_out[i].iterClk), &d_iterClk, sizeof(uint64_t*), hipMemcpyHostToDevice));
    }
    
    // Prepare Data (Critical: NUMA/Home Identification)
    // This function inside ppnt.h will now contain the logic to allocate on specific HBMs
    ppnt::prepare_ping_data(h_plan, d_plan, n_plan);

    // 3. Attention Setup
    int B=16, H=16, S=1024, D=64, BLOCK_M=64; 
    int mode = SWIZZLED_HEAD_FIRST; 
    if (argc > 1) mode = atoi(argv[1]);
    
    size_t attn_n = (size_t)B * H * S * D;
    cout << "[INFO] Attn Params: B=" << B << " H=" << H << " S=" << S << " D=" << D 
         << " Mode=" << mode << endl;

    float *dQ, *dK, *dV, *dO;
    gpuErrchk(hipMalloc(&dQ, attn_n * sizeof(float)));
    gpuErrchk(hipMalloc(&dK, attn_n * sizeof(float)));
    gpuErrchk(hipMalloc(&dV, attn_n * sizeof(float)));
    gpuErrchk(hipMalloc(&dO, attn_n * sizeof(float)));
    
    gpuErrchk(hipMemset(dQ, 0, attn_n * sizeof(float)));
    gpuErrchk(hipMemset(dK, 0, attn_n * sizeof(float)));
    gpuErrchk(hipMemset(dV, 0, attn_n * sizeof(float)));

    AttnSwizzleArgs h_args = {dQ, dK, dV, dO, B, H, S, D, BLOCK_M, XCD_NUM, mode};
    AttnSwizzleArgs* d_args = nullptr;
    gpuErrchk(hipMalloc(&d_args, sizeof(AttnSwizzleArgs)));
    gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(AttnSwizzleArgs), hipMemcpyHostToDevice, stream));

    // 4. Launch Profiling
    cout << "[INFO] Launching Fused Kernel..." << endl;
    
    AttnSwizzleTargetFn fn{};
    size_t _n_plan = n_plan; 
    
    // Arguments: [TargetFn, Args*, Plan*, n_plans, Output*]
    void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

    gpuErrchk(hipLaunchCooperativeKernel(
        (void*)ppnt::fused_kernel<AttnSwizzleTargetFn, AttnSwizzleArgs>,
        dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        
    gpuErrchk(hipStreamSynchronize(stream));
    
    // 5. Report Results
    ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);

    // Cleanup
    gpuErrchk(hipFree(d_args));
    gpuErrchk(hipFree(dQ)); gpuErrchk(hipFree(dK)); 
    gpuErrchk(hipFree(dV)); gpuErrchk(hipFree(dO));
    gpuErrchk(hipFree(d_plan));
    gpuErrchk(hipFree(d_out)); 
    
    cout << "[INFO] Done." << endl;
    return 0;
}