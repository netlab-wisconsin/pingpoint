#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_complex.h>


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


__global__ void k2(uint64_t *xcd_chunks1, uint64_t *xcd_chunks2, 
                    uint64_t *xcd_chunks3, uint64_t *xcd_chunks4,
                    size_t *xcd_chunks_offset1, size_t *xcd_chunks_offset2,
                    size_t *xcd_chunks_offset3, size_t *xcd_chunks_offset4,
                    size_t *xcd_chunks_size, size_t chunk_size, 
                    uint32_t *cycles_start, uint32_t *cycles_stop) 
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;
    cg::grid_group grid = cg::this_grid();

    // print tbid_in_xcd
    int n_tbs_in_xcd = (gridDim.x / XCDS_NUM); // number of thread blocks in each xcd
    int tbid_in_xcd = (bid / XCDS_NUM) % n_tbs_in_xcd; // thread block id within xcd
    if (tid == 0) {
        #if DEBUG
        printf("bid %d: tbid_in_xcd %d\n", bid, tbid_in_xcd);
        #endif
    }

    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    if (tid == 0) {
        #if DEBUG
        printf("bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", bid, xcc_id, se_id, cu_id);
        #endif
    }

    float4 reg_in1, reg_in2, reg_in3, reg_in4;
    float sink0 = 0, sink1 = 0, sink2 = 0, sink3 = 0;

    // warmup
    // only 1 thread per xcd warms up entirely
    if (tbid_in_xcd == 0 && tid == 0) {
        for (size_t i = 0; i < xcd_chunks_size[xcc_id]; i++) {

            float4 *ptr_in1 = reinterpret_cast<float4*>(xcd_chunks1[xcd_chunks_offset1[xcc_id] + i]);
            float4 *ptr_in2 = reinterpret_cast<float4*>(xcd_chunks2[xcd_chunks_offset2[xcc_id] + i]);
            float4 *ptr_in3 = reinterpret_cast<float4*>(xcd_chunks3[xcd_chunks_offset3[xcc_id] + i]);
            float4 *ptr_in4 = reinterpret_cast<float4*>(xcd_chunks4[xcd_chunks_offset4[xcc_id] + i]);

            for (size_t offset = 0; offset < (chunk_size / sizeof(float4)); offset++) {
                asm volatile(
                    "flat_load_dwordx4 %[OUT_D1],  %[IN_D1]\n\t"
                    "flat_load_dwordx4 %[OUT_C1],  %[IN_C1]\n\t"
                    "flat_load_dwordx4 %[OUT_B1],  %[IN_B1]\n\t"
                    "flat_load_dwordx4 %[OUT_A1],  %[IN_A1]\n\t" 
                    "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                    : [OUT_A1]"=v" (reg_in1), [OUT_B1]"=v" (reg_in2), [OUT_C1]"=v"(reg_in3), [OUT_D1]"=v" (reg_in4)
                    : [IN_A1]"v" (ptr_in1), [IN_B1]"v" (ptr_in2), [IN_C1]"v" (ptr_in3), [IN_D1]"v" (ptr_in4)
                    : "memory"
                );

                sink0 += reg_in1.x + reg_in2.x + reg_in3.x + reg_in4.x;
                sink1 += reg_in1.y + reg_in2.y + reg_in3.y + reg_in4.y;
                sink2 += reg_in1.z + reg_in2.z + reg_in3.z + reg_in4.z;
                sink3 += reg_in1.w + reg_in2.w + reg_in3.w + reg_in4.w;
            }
        }
    }
    xcd_barrier(xcc_id, n_tbs_in_xcd); // sync xcd

    grid.sync();

    // measurement
    size_t n_iter = xcd_chunks_size[xcc_id] / n_tbs_in_xcd;
    if (tid == 0) {
        #if DEBUG
        printf("bid %d (tbid_in_xcd %d) on xcc %d: n_iter %zu\n", bid, tbid_in_xcd, xcc_id, n_iter);
        #endif
    }

    // start timing
    uint32_t start = 0;
    start = __builtin_readcyclecounter();
    asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME

    for (size_t iter = 0; iter < n_iter; iter++) {
        size_t chunk_idx = tbid_in_xcd + iter * n_tbs_in_xcd;

        float4 *ptr_in1 = reinterpret_cast<float4*>(xcd_chunks1[xcd_chunks_offset1[xcc_id] + chunk_idx]);
        float4 *ptr_in2 = reinterpret_cast<float4*>(xcd_chunks2[xcd_chunks_offset2[xcc_id] + chunk_idx]);
        float4 *ptr_in3 = reinterpret_cast<float4*>(xcd_chunks3[xcd_chunks_offset3[xcc_id] + chunk_idx]);
        float4 *ptr_in4 = reinterpret_cast<float4*>(xcd_chunks4[xcd_chunks_offset4[xcc_id] + chunk_idx]);
        
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
        cycles_start[xcc_id * n_tbs_in_xcd + tbid_in_xcd] = start;
        cycles_stop[xcc_id * n_tbs_in_xcd + tbid_in_xcd] = stop;
    }

    grid.sync(); // sync all xcds post measurement
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