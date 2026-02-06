#pragma once

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "ppnt.h" 
#include "main.h"

// =============================================================================================
// SWIZZLED ATTENTION (Adapted for PPNT)
// =============================================================================================

// Mapping modes
enum MapMode : int {
  NAIVE_BLOCK_FIRST = 0,   // iterate blocks then heads
  NAIVE_HEAD_FIRST  = 1,   // iterate heads then blocks
  SWIZZLED_HEAD_FIRST = 2  // Figure 11 logic: confine blocks of a head to one XCD region
};

struct AttnSwizzleArgs {
    float *Q, *K, *V, *O;
    int B, H, S, D;
    int BLOCK_M;
    int num_xcd;
    int mode; // MapMode
};

// Helper for indexing: [B, H, S, D]
__device__ __forceinline__ int idx_bhsd(int b, int h, int s, int d,
                                        int B, int H, int S, int D) {
    return ((b * H + h) * S + s) * D + d;
}

// Swizzle logic adapted to device function
// Maps linear logical ID 'wid' -> (b, h, m_block)
__device__ __forceinline__ 
void swizzle_map_wid(int wid, int &b, int &h, int &m_block,
                     int B, int H, int S, int BLOCK_M,
                     int num_xcd, int mode) 
{
    int num_m_blocks = S / BLOCK_M; 
    
    if (mode == NAIVE_BLOCK_FIRST) {
        // wid = ((b * H + h) * num_m_blocks) + m_block
        m_block = wid % num_m_blocks;
        int tmp = wid / num_m_blocks;
        h = tmp % H;
        b = tmp / H;
    } 
    else if (mode == NAIVE_HEAD_FIRST) {
        // wid = ((b * num_m_blocks + m_block) * H) + h
        h = wid % H;
        int tmp = wid / H;
        m_block = tmp % num_m_blocks;
        b = tmp / num_m_blocks;
    }
    else { // SWIZZLED_HEAD_FIRST
        // 1. Identify which XCD this wid physically lands on.
        int global_wid_div_xcd = wid / num_xcd;
        int xcd_id = wid % num_xcd;

        // How many heads per XCD?
        int heads_per_xcd = H / num_xcd; // Assumes H divides num_xcd
        
        // The tasks for this specific XCD are ordered: (Batch, LocalHead, M_Block)
        // task_idx_local = global_wid_div_xcd
        
        // task_idx_local = ((b * heads_per_xcd) + h_local) * num_m_blocks + m_block
        m_block = global_wid_div_xcd % num_m_blocks;
        int tmp = global_wid_div_xcd / num_m_blocks;
        int h_local = tmp % heads_per_xcd;
        b = tmp / heads_per_xcd;
        
        // Map local head back to global head using interleaved logic (Head0->XCD0, Head1->XCD1...)
        h = h_local * num_xcd + xcd_id;
    }
}

struct AttnSwizzleTargetFn {
    __device__ __forceinline__
    void operator()(const AttnSwizzleArgs* __restrict__ args,
                    int bid, int tid, int gridDimX, int blockDimX, size_t n_ppnt_tbs_in_xcd) const 
    {
        // 1. Calculate Logical Block ID (Workgroup ID)
        int n_tbs_in_xcd = gridDimX / XCD_NUM;
        int wid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);

        // 2. Determine Workload Bounds
        int num_m_blocks = args->S / args->BLOCK_M;
        size_t total_logical_blocks = (size_t)args->B * args->H * num_m_blocks;

        // If grid is larger than workload, or this specific block is out of bounds
        if (wid >= total_logical_blocks) return;

        // 3. Map WID to (b, h, m_block)
        int b, h, m_block;
        swizzle_map_wid(wid, b, h, m_block, 
                        args->B, args->H, args->S, args->BLOCK_M, 
                        args->num_xcd, args->mode);

        // 4. Per-thread work (Naive Attention / Memory Stress)
        int s_out_start = m_block * args->BLOCK_M;
        int s_limit = args->S;
        int D = args->D;

        // Iterate over the rows assigned to this thread
        for (int i = tid; i < args->BLOCK_M; i += blockDimX) {
            int s_out = s_out_start + i;
            if (s_out >= s_limit) continue;

            // Compute 1 output row: O[b,h,s,:]
            for (int d_out = 0; d_out < D; d_out++) {
                float sum_val = 0.0f;
                
                // Read Q
                float q_val = args->Q[idx_bhsd(b, h, s_out, d_out, args->B, args->H, args->S, D)];

                // Scan K (inner dim S) - generating Cross-XCD traffic if K is remote
                for (int s_in = 0; s_in < args->S; s_in++) {
                     float k_val = args->K[idx_bhsd(b, h, s_in, d_out, args->B, args->H, args->S, D)];
                     sum_val += q_val * k_val;
                }
                
                // Read V 
                float v_val = args->V[idx_bhsd(b, h, s_out, d_out, args->B, args->H, args->S, D)];
                
                // Write O
                args->O[idx_bhsd(b, h, s_out, d_out, args->B, args->H, args->S, D)] = sum_val + v_val;
            }
        }
    }
};