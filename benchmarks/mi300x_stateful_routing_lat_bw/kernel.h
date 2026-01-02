#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_complex.h>

#ifndef XCD_NUM
#define XCD_NUM 8
#endif

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif


// per-xcd barrier
__device__ int d_xcd_barrier_count[XCDS_NUM] = {0};
// __device__ int d_xcd_barrier_sense[XCDS_NUM] = {0};
__device__ volatile int d_xcd_barrier_sense[XCDS_NUM] = {0}; // avoid spin

__device__ __forceinline__
void xcd_barrier(int xcd_id, int blocks_per_xcd)
{
    __shared__ int local_sense;  // one per block
    if (threadIdx.x == 0) {
        int old_sense = d_xcd_barrier_sense[xcd_id];
        local_sense   = !old_sense;

        int arrived = atomicAdd(&d_xcd_barrier_count[xcd_id], 1);
        if (arrived == blocks_per_xcd - 1) {
            d_xcd_barrier_count[xcd_id] = 0;
            __threadfence(); // Make global writes visible on this XCD before release
            d_xcd_barrier_sense[xcd_id] = local_sense;
        }
    }
    __syncthreads(); // Make sure all threads in the block see local_sense
    while (d_xcd_barrier_sense[xcd_id] != local_sense) {} // Spin until last block flips global sense
    __syncthreads(); // all threads in this block observe sense change before proceeding
}


// fused kernel
template <typename T>
__global__ void k(
    /* k1 args */
    T *k1_buf, T *__restrict__ k1_dummy_buf, const int64_t k1_N, const uint32_t k1_xcd, uint32_t *k1_startClk, uint32_t *k1_stopClk,
    /* k2 args */
    uint64_t *k2_chunks1, uint64_t *k2_chunks2, uint64_t *k2_chunks3, uint64_t *k2_chunks4,
    size_t *k2_offset1, size_t *k2_offset2, size_t *k2_offset3, size_t *k2_offset4,
    size_t *k2_chunks_size, const size_t k2_chunk_size, const uint32_t k2_xcd, const uint32_t k2_hbm, const uint32_t k2_tpb, uint32_t *k2_startClk, uint32_t *k2_stopClk)
{

    cg::grid_group grid = cg::this_grid(); // @june: required?

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    int n_tbs_in_xcd = (gridDim.x / XCD_NUM); // number of thread blocks in each xcd
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd; // thread block id within xcd
    int tid_in_xcd = tbid_in_xcd * blockDim.x + tid; // thread id within xcd
    
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

    if (tid == 0) {
        #if DEBUG_LEVEL >= 1
        printf("bid %d xcd %d tbid_in_xcd %d\n", bid, xcc_id, tbid_in_xcd);
        #endif
    }

    if (xcc_id == k1_xcd) { /* k1 */

        // prelude: for k1, only the first thread in xcd 0 runs
        int tidx = threadIdx.x + blockIdx.x * blockDim.x;
        if (tbid_in_xcd != 0 || threadIdx.x != 0) return;

        T *idx = k1_buf;

        // 일단 전체 elapsed cycle을 잡아보자. 더 finer-grained한 측정은 나중에
        *k1_startClk = __builtin_readcyclecounter();
        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");

        const int unroll_factor = 32;
        #pragma unroll 1
        for (int64_t n = 0; n < k1_N; n += unroll_factor) {
            #pragma unroll
            for (int u = 0; u < unroll_factor; u++) {
                idx = (T *)*idx;
            }
        }
        asm volatile("s_waitcnt vmcnt(0)");
        
        // 일단 전체 elapsed cycle을 잡아보자. 더 finer-grained한 측정은 나중에
        *k1_stopClk = __builtin_readcyclecounter();
        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");
        
        if (tidx > 12313) {
            k1_dummy_buf[0] = (T)idx;
        }

    } else { /* k2 */

        // prelude: for k2, only the threads in xcd 1 run
        if (xcc_id != k2_xcd) return;

        // 여기
        if (tid >= k2_tpb) return;

        float4 reg_in1, reg_in2, reg_in3, reg_in4;
        float sink0 = 0, sink1 = 0, sink2 = 0, sink3 = 0;

        size_t n_iter = k2_chunks_size[k2_hbm] / n_tbs_in_xcd;
        if (tid == 0) {
            #if DEBUG_LEVEL >= 1
            printf("bid %d (tbid_in_xcd %d) on xcc %d: n_iter %zu\n", bid, tbid_in_xcd, xcc_id, n_iter);
            #endif
        }

        // start timing
        uint32_t start = 0;
        start = __builtin_readcyclecounter();
        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME


        for (size_t iter = 0; iter < n_iter; iter++) {
            size_t chunk_idx = tbid_in_xcd + iter * n_tbs_in_xcd;

            float4 *ptr_in1 = reinterpret_cast<float4*>(k2_chunks1[k2_offset1[k2_hbm] + chunk_idx]);
            float4 *ptr_in2 = reinterpret_cast<float4*>(k2_chunks2[k2_offset2[k2_hbm] + chunk_idx]);
            float4 *ptr_in3 = reinterpret_cast<float4*>(k2_chunks3[k2_offset3[k2_hbm] + chunk_idx]);
            float4 *ptr_in4 = reinterpret_cast<float4*>(k2_chunks4[k2_offset4[k2_hbm] + chunk_idx]);
            
            asm volatile(
                "flat_load_dwordx4 %[OUT_D1],  %[IN_D1]\n\t"
                "flat_load_dwordx4 %[OUT_C1],  %[IN_C1]\n\t"
                "flat_load_dwordx4 %[OUT_B1],  %[IN_B1]\n\t"
                "flat_load_dwordx4 %[OUT_A1],  %[IN_A1]\n\t" 
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                : [OUT_A1]"=v" (reg_in1), [OUT_B1]"=v" (reg_in2), [OUT_C1]"=v"(reg_in3), [OUT_D1]"=v" (reg_in4)
                : [IN_A1]"v" (&ptr_in1[tid]), [IN_B1]"v" (&ptr_in2[tid]), [IN_C1]"v" (&ptr_in3[tid]), [IN_D1]"v" (&ptr_in4[tid])
                : "memory"
            );

            sink0 += reg_in1.x + reg_in2.x + reg_in3.x + reg_in4.x;
            sink1 += reg_in1.y + reg_in2.y + reg_in3.y + reg_in4.y;
            sink2 += reg_in1.z + reg_in2.z + reg_in3.z + reg_in4.z;
            sink3 += reg_in1.w + reg_in2.w + reg_in3.w + reg_in4.w;
        }

        // stop timing
        uint32_t stop = 0;
        stop = __builtin_readcyclecounter();
        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME

        if (tid == 0) {
            k2_startClk[tbid_in_xcd] = start;
            k2_stopClk[tbid_in_xcd] = stop;
        } 
    }
}


template <typename T>
__global__ void k1(T *buf, T *__restrict__ dummy_buf, const int64_t N, const uint32_t pinned_xcd) {

    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    // printf("xcc_id: %u, se_id: %u, cu_id: %u\n", xcc_id, se_id, cu_id);
    
    // Constrain pchase thread to pinned_xcd
    if (xcc_id != pinned_xcd) return;

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    T *idx = buf;

    const int unroll_factor = 32;
#pragma unroll 1
    for (int64_t n = 0; n < N; n += unroll_factor) {
#pragma unroll
        for (int u = 0; u < unroll_factor; u++) {
            idx = (T *)*idx;
        }
    }

    if (tidx > 12313) {
        dummy_buf[0] = (T)idx;
    }
}