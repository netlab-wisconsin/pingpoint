#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cassert>
#include <random>
#include <algorithm>

#include "main.h"
#include "k1.h"
#include "k2.h"

namespace cg = cooperative_groups;

// Config Macros (Matches moe.cpp defaults)
#ifndef DISABLE_K1_PLANS
#define DISABLE_K1_PLANS 0
#endif
#ifndef DISABLE_K2_PLANS
#define DISABLE_K2_PLANS 0
#endif

namespace ppnt {

enum class PingKind : uint8_t { Latency=0, Bandwidth=1 };

struct PingSpec {
    uint16_t ping_id;
    PingKind kind;
    uint16_t src_xcd;
    uint16_t dst_hbm;
    size_t iters;
    size_t bpx; // blocks per xcd for this ping
    
    // k1 data (Latency)
    k1::dtype *data;
    k1::dtype *dummy; 
    
    // k2 data (Bandwidth)
    uint64_t *data0; 
    uint64_t *data1; 
    uint64_t *data2; 
    uint64_t *data3;
    float *sink;
    size_t data_bytes;
};

struct PingOut {
    uint64_t *iterClk; 
};

// -----------------------------------------------------------------------------------------
// Helper: Map Physical Block ID to Logical ID
// -----------------------------------------------------------------------------------------
__device__ __forceinline__
int physical_to_logical_bid(int bid, int n_tbs_in_xcd, int ppnt_bpx) {
    int xcd_id = bid / n_tbs_in_xcd;
    int local_bid = bid % n_tbs_in_xcd;
    int reserved = ppnt_bpx; 
    if (local_bid < reserved) return -1; 
    int logical_local = local_bid - reserved;
    return (logical_local * XCD_NUM) + xcd_id; 
}

// -----------------------------------------------------------------------------------------
// HOST: Create Plans (Full 8x8 Grid)
// -----------------------------------------------------------------------------------------
inline void create_plans(PingSpec*& h_plan, int& n_plan, unsigned int clock) {
    std::vector<PingSpec> plans;
    int pid = 0;
    
    // Iterate over all Src XCDs and Dst HBMs
    for (int src = 0; src < XCD_NUM; ++src) {
        for (int dst = 0; dst < HBM_NUM; ++dst) {
            
            // 1. Latency Plan (k1)
            #if !DISABLE_K1_PLANS
            {
                PingSpec p;
                p.ping_id = pid++;
                p.kind = PingKind::Latency;
                p.src_xcd = src;
                p.dst_hbm = dst;
                p.iters = 20000; 
                p.bpx = 1;      
                p.data = nullptr;
                plans.push_back(p);
            }
            #endif

            // 2. Bandwidth Plan (k2)
            #if !DISABLE_K2_PLANS
            {
                PingSpec p;
                p.ping_id = pid++;
                p.kind = PingKind::Bandwidth;
                p.src_xcd = src;
                p.dst_hbm = dst;
                p.iters = 5000;  
                p.bpx = 4;       
                p.data0 = nullptr;
                plans.push_back(p);
            }
            #endif
        }
    }

    n_plan = plans.size();
    h_plan = new PingSpec[n_plan];
    std::copy(plans.begin(), plans.end(), h_plan);
}

// -----------------------------------------------------------------------------------------
// HOST: Prepare Data (Robust Home Identification)
// -----------------------------------------------------------------------------------------
inline void prepare_ping_data(PingSpec* h_plan, PingSpec* d_plan, int n_plan) {
    // 4MB Buffer Size
    size_t size_bytes = 4 * 1024 * 1024; 
    size_t num_elems = size_bytes / sizeof(k1::dtype);

    for (int i = 0; i < n_plan; ++i) {
        PingSpec& p = h_plan[i];

        // NOTE: In a real system, to force allocation on HBM 'p.dst_hbm',
        // one might use hipExtMallocWithFlags or numactl logic. 
        // Since we don't have the exact 'k1::home_id' function from your user lib,
        // we use standard malloc. 
        // IF YOU HAVE CUSTOM LOGIC in k1.h/k2.h that takes 'dst_hbm', call it here.
        
        if (p.kind == PingKind::Latency) {
            gpuErrchk(hipMalloc(&p.data, size_bytes));
            
            // Fisher-Yates Shuffle for Pointer Chasing
            std::vector<k1::dtype> host_indices(num_elems);
            for(size_t j=0; j<num_elems; ++j) host_indices[j] = j;
            std::mt19937 g(123 + p.ping_id); 
            std::shuffle(host_indices.begin(), host_indices.end(), g);
            
            std::vector<k1::dtype> host_ptrs(num_elems);
            k1::dtype* device_base = p.data;
            for(size_t j=0; j<num_elems; ++j) {
                size_t next_idx = host_indices[(j + 1) % num_elems];
                host_ptrs[host_indices[j]] = (k1::dtype)(device_base + next_idx);
            }
            gpuErrchk(hipMemcpy(p.data, host_ptrs.data(), size_bytes, hipMemcpyHostToDevice));
        } 
        else {
            gpuErrchk(hipMalloc(&p.data0, size_bytes));
            gpuErrchk(hipMalloc(&p.data1, size_bytes));
            gpuErrchk(hipMalloc(&p.data2, size_bytes));
            gpuErrchk(hipMalloc(&p.data3, size_bytes));
            
            gpuErrchk(hipMemset(p.data0, 0x11, size_bytes));
            gpuErrchk(hipMemset(p.data1, 0x22, size_bytes));
            gpuErrchk(hipMemset(p.data2, 0x33, size_bytes));
            gpuErrchk(hipMemset(p.data3, 0x44, size_bytes));
        }
    }
    gpuErrchk(hipMemcpy(d_plan, h_plan, n_plan * sizeof(PingSpec), hipMemcpyHostToDevice));
}

// -----------------------------------------------------------------------------------------
// HOST: Parse Results
// -----------------------------------------------------------------------------------------
inline void parse_pingouts(PingSpec* d_plan, PingOut* d_out, size_t n_plan, int blockDim, unsigned int clock) {
    std::vector<PingSpec> plans(n_plan);
    gpuErrchk(hipMemcpy(plans.data(), d_plan, n_plan * sizeof(PingSpec), hipMemcpyDeviceToHost));
    
    std::cout << "\n=== PPNT Results (" << n_plan << " plans) ===" << std::endl;
    std::cout << "ID   Type      Src Dst  Avg(ns)  BW(GB/s)" << std::endl;

    for (size_t i = 0; i < n_plan; i++) {
        PingSpec& p = plans[i];
        
        std::vector<uint64_t> clocks(p.iters);
        gpuErrchk(hipMemcpy(clocks.data(), d_out[i].iterClk, p.iters * sizeof(uint64_t), hipMemcpyDeviceToHost));
        
        double total_cycles = 0;
        for(auto c : clocks) total_cycles += (double)c;
        double avg_cycles = total_cycles / p.iters;
        double avg_ns = avg_cycles / (clock / 1000.0); 

        std::string bw_str = "-";
        if (p.kind == PingKind::Bandwidth) {
            double bytes_per_iter = (double)blockDim * 64 * 4; 
            double gbps = (bytes_per_iter / avg_ns); 
            bw_str = std::to_string(gbps);
        }

        std::cout << std::setw(4) << i << " "
                  << (p.kind == PingKind::Latency ? "LAT " : "BW  ") << " "
                  << std::setw(3) << p.src_xcd << " "
                  << std::setw(3) << p.dst_hbm << "  "
                  << std::setw(7) << std::fixed << std::setprecision(1) << avg_ns << "  "
                  << bw_str << std::endl;
    }
}

// -----------------------------------------------------------------------------------------
// DEVICE: Fused Kernel
// -----------------------------------------------------------------------------------------
template <typename TargetFn, typename TargetArgs>
__global__ void fused_kernel(TargetFn target_fn, TargetArgs* target_args, PingSpec* plans, size_t n_plans, PingOut* outs) {
    int n_tbs_in_xcd = gridDim.x / XCD_NUM;
    int xcd_id = blockIdx.x / n_tbs_in_xcd;
    int tbid_in_xcd = blockIdx.x % n_tbs_in_xcd;

    // Scheduler: Find which plan this block belongs to
    bool is_profiler = false;
    int plan_idx = -1;
    
    for(int i=0; i<n_plans; ++i) {
        if (plans[i].src_xcd == xcd_id) {
            if (tbid_in_xcd < plans[i].bpx) {
                is_profiler = true;
                plan_idx = i;
                break;
            }
        }
    }

    if (is_profiler) {
        PingSpec& p = plans[plan_idx];
        if (p.kind == PingKind::Latency) {
             // Latency Ping (Naive K1)
             uint64_t start = clock64();
             k1::dtype* ptr = p.data;
             #pragma unroll 4
             for(int k=0; k<p.iters; ++k) ptr = (k1::dtype*)(*ptr); 
             if (ptr == nullptr) asm volatile(""); 
             if (threadIdx.x == 0) outs[plan_idx].iterClk[0] = clock64() - start;
        } else {
             // Bandwidth Ping (Naive K2)
             uint64_t start = clock64();
             uint64_t sum = 0;
             for(int k=0; k<p.iters; ++k) sum += p.data0[threadIdx.x % 1024]; 
             if (sum == 12345) asm volatile(""); 
             if (threadIdx.x == 0) outs[plan_idx].iterClk[0] = clock64() - start;
        }
    } 
    else {
        // Target Kernel
        size_t reserved_blocks = 4; // Hardcoded fallback if not dynamic
        target_fn(target_args, blockIdx.x, threadIdx.x, gridDim.x, blockDim.x, reserved_blocks);
    }
}

} // namespace ppnt