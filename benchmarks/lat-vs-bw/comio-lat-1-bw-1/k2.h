#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "main.h"

using namespace std;
namespace cg = cooperative_groups;

#define DEBUG_K2_HOME 0
#define DEBUG_K2_KERNEL 0

#define USE_GLOBAL_BARRIER 0

namespace k2 {

__device__ int d_barrier_count = 0;
__device__ int d_barrier_sense = 0;
    
__device__ __forceinline__
void global_barrier()
{
    __shared__ int local_sense; // One local copy per block
    if (threadIdx.x == 0) {
        int old_sense = d_barrier_sense;
        local_sense   = !old_sense;

        int arrived = atomicAdd(&d_barrier_count, 1);
        if (arrived == gridDim.x - 1) {
            d_barrier_count = 0;
            __threadfence(); // Make all global writes visible before releasing others
            d_barrier_sense = local_sense;
        }
    }
    __syncthreads(); // Make sure all threads in the block see local_sense
    while (d_barrier_sense != local_sense) {} // Spin until last block flips global sense
    __syncthreads(); // Ensure all threads in this block observe the sense change before proceeding
}

// note: 12/13 convert to 1d c_cycles and calculate index with n_chunks.
__global__ void identify_home(void *data, size_t size, uint32_t *d_cycles, int n_chunks) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    // print tbid_in_xcd
    int n_tbs_in_xcd = (gridDim.x / XCD_NUM); // number of thread blocks in each xcd
    int tbid_in_xcd = (bid / XCD_NUM) % n_tbs_in_xcd; // thread block id within xcd
    if (tid == 0) {
        #if DEBUG_K2_HOME
        printf("bid %d: tbid_in_xcd %d\n", bid, tbid_in_xcd);
        #endif
    }

    // for this bmk, assert 1 tb per xcd
    assert(n_tbs_in_xcd * XCD_NUM == gridDim.x); assert(n_tbs_in_xcd == 1);

    // for this bmk, assert 256 threads per block if chunk size is 4KB (128 for 2KB)
    assert(blockDim.x == 128);
    // assert(blockDim.x == 256);

    // for this bmk, both global_barrier and cooperative_groups are supported
    #if not USE_GLOBAL_BARRIER
    cg::grid_group grid = cg::this_grid();
    #endif

    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    if (tid == 0) {
        #if DEBUG_K2_HOME
        printf("bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", bid, xcc_id, se_id, cu_id);
        #endif
    }

    uint4 *data_u4 = (uint4*)data;
    const size_t n_16Bs = size / sizeof(uint4);
    const size_t n_iter = n_16Bs / (blockDim.x * n_tbs_in_xcd); // thread blocks on same xcd streams over the data chunks
    if (bid == 0 && tid == 0) {
        #if DEBUG_K2_HOME
        printf("n_16Bs: %zu, n_iter: %zu\n", n_16Bs, n_iter);
        printf("working set per xcd: %.2f MB\n", (n_16Bs * sizeof(uint4) / XCD_NUM) / (1024.0 * 1024.0));
        #endif
    }

    const size_t inner_size = 64 * 1024 * 1024; // 64MB per inner loop (given TLB lat jump at 64MB)
    const size_t n_outer = size / inner_size;
    const size_t n_inner = inner_size / (16 * blockDim.x); // num of accesses per tb in inner loop
    assert (n_iter == n_outer * n_inner);
    
    for (size_t i = 0; i < n_outer; i++) {
        // warmup 
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (tid == 0) {
                #if DEBUG_K2_HOME
                printf("[warmup] (outer:%zu, inner:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, j, bid, tbid_in_xcd, index, index + blockDim.x - 1);
                #endif
            }
            asm volatile(
                "flat_load_dwordx4 v[0:3], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data_u4[index])
                : "memory", "v0", "v1", "v2", "v3"
            );
        }
        #if USE_GLOBAL_BARRIER
        global_barrier();
        #else
        grid.sync();
        #endif

        // measurement
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (tid == 0) {
                #if DEBUG_K2_HOME
                printf("[actual] (outer:%zu, inner:%zu, bid:%d, tbid_xcd:%d) accessing data_u4[%zu..%zu]\n", i, j, bid, tbid_in_xcd, index, index + blockDim.x - 1);
                #endif
            }
            uint32_t start = __builtin_readcyclecounter();
            asm volatile(
                "flat_load_dwordx4 v[0:3], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data_u4[index])
                : "memory", "v0", "v1", "v2", "v3"
            );
            uint32_t end = __builtin_readcyclecounter();
            uint32_t cycle = end - start;
            if (tid == 0) {
                const int d_cycles_index = xcc_id * n_chunks + (i * n_inner + j) * n_tbs_in_xcd + tbid_in_xcd;
                d_cycles[d_cycles_index] = cycle;
                #if DEBUG_K2_HOME
                printf("outer %zu inner %zu tbid_in_xcd %d (bid %d, xcd %d): %u cycles\n", i, j, tbid_in_xcd, bid, xcc_id, cycle);
                #endif
            }
            #if USE_GLOBAL_BARRIER
            global_barrier();
            #else
            grid.sync();
            #endif
        }
    }
}

int home_identification(
    const vector<char*> &k2_d_data,
    const size_t k2_data_size,
    const size_t k2_n_chunks,
    const int k2_n_datas,
    vector<vector<int>> &k2_h_home,
    vector<vector<size_t>> &k2_h_xcd_chunks_size
) {

    // per-data, per-xcd, per-chunk. record cycles for home identification
    // last two dimensions are flattened
    vector<uint32_t*> d_cycles(k2_n_datas);
    for (int i = 0; i < k2_n_datas; i++) {
        gpuErrchk(hipMalloc((void**)&d_cycles[i], sizeof(uint32_t) * XCD_NUM * k2_n_chunks));
        #if DEBUG_K2_HOME
        printf("allocated d_cycles[%d] array[%d][%zu]\n", i, XCD_NUM, k2_n_chunks);
        #endif
    }

    for (int i = 0; i < k2_n_datas; i++) {
        #if DEBUG_K2_HOME
        printf("Identifying home xcd for data[%d]...\n", i);
        #endif

        void *kernel_args[] = {
            (void*)&k2_d_data[i],
            (void*)&k2_data_size,
            (void*)&d_cycles[i], // XCD_NUM * n_chunks
            (void*)&k2_n_chunks
        };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)identify_home,
            dim3(XCD_NUM), // one block per xcd
            dim3(128), // 128 threads per block
            kernel_args,
            0,
            0
        ));
        gpuErrchk(hipDeviceSynchronize());

        /* retrieve and process results */

        vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(k2_n_chunks));
        for (int x = 0; x < XCD_NUM; x++) {
            gpuErrchk(hipMemcpy(
                h_cycles[x].data(),
                &d_cycles[i][x * k2_n_chunks],
                sizeof(uint32_t) * k2_n_chunks,
                hipMemcpyDeviceToHost
            ));
        }

        for (size_t k = 0; k < k2_n_chunks; k++) {
            uint32_t min_cycles = 0xFFFFFFFF;
            int min_xcc = -1;
            for (int x = 0; x < XCD_NUM; x++) {
                uint32_t c = h_cycles[x][k];
                if (c < min_cycles) {
                    min_cycles = c;
                    min_xcc = x;
                }
            }
            k2_h_home[i][k] = min_xcc;
            k2_h_xcd_chunks_size[i][min_xcc]++;
            #if DEBUG_K2_HOME
            printf("data[%d] chunk[%zu]: home xcd %d\n", i, k, k2_h_home[i][k]);
            #endif
        }
        #if DEBUG_K2_HOME
        for (int x = 0; x < XCD_NUM; x++) {
            printf("data[%d] xcd %d: n_chunks %zu\n", i, x, k2_h_xcd_chunks_size[i][x]);
        }
        printf("\n");
        #endif
    }

    return 0;
}

__global__ void k(
    uint64_t *k2_chunks1, uint64_t *k2_chunks2, uint64_t *k2_chunks3, uint64_t *k2_chunks4,
    size_t *k2_offset1, size_t *k2_offset2, size_t *k2_offset3, size_t *k2_offset4,
    float *k2_sink, size_t *k2_chunks_size, const size_t k2_chunk_size, 
    const int64_t k2_N, const uint32_t k2_xcd, const uint32_t k2_hbm)
{

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

    // prelude: for k2, only the threads in xcd 1 run
    if (xcc_id != k2_xcd) return;
    #if DEBUG_K2_KERNEL
    if (tid == 0) {
        printf("k2 (bid:%d) launched on xcd:%u se:%u cu:%u\n",
            bid, xcc_id, se_id, cu_id);

    }
    #endif

    float4 reg_in1, reg_in2, reg_in3, reg_in4;
    float sink0 = 0, sink1 = 0, sink2 = 0, sink3 = 0;

    const size_t n_chunks_per_iter_per_tb = (blockDim.x / (k2_chunk_size / 16));
    size_t n_iter = k2_chunks_size[k2_hbm] / (n_tbs_in_xcd * n_chunks_per_iter_per_tb);

    // for (size_t iter = 0; iter < n_iter; iter++) { /* initial logic w/o k2_N */
    for (int64_t i = 0; i < k2_N; i++) {
        size_t iter = i % n_iter; // Wrap around using modulo if i exceeds available n_iter
        for (size_t c = 0; c < n_chunks_per_iter_per_tb; c++) {
            size_t chunk_idx = c+ tbid_in_xcd * n_chunks_per_iter_per_tb + iter * n_tbs_in_xcd;

            float4 *ptr_in1 = reinterpret_cast<float4*>(k2_chunks1[k2_offset1[k2_hbm] + chunk_idx]);
            float4 *ptr_in2 = reinterpret_cast<float4*>(k2_chunks2[k2_offset2[k2_hbm] + chunk_idx]);
            float4 *ptr_in3 = reinterpret_cast<float4*>(k2_chunks3[k2_offset3[k2_hbm] + chunk_idx]);
            float4 *ptr_in4 = reinterpret_cast<float4*>(k2_chunks4[k2_offset4[k2_hbm] + chunk_idx]);
            
            asm volatile(
                "flat_load_dwordx4 %[OUT_D1],  %[IN_D1]\n\t"
                "flat_load_dwordx4 %[OUT_C1],  %[IN_C1]\n\t"
                "flat_load_dwordx4 %[OUT_B1],  %[IN_B1]\n\t"
                "flat_load_dwordx4 %[OUT_A1],  %[IN_A1]\n\t" 
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t" // necessary?
                : [OUT_A1]"=v" (reg_in1), [OUT_B1]"=v" (reg_in2), [OUT_C1]"=v"(reg_in3), [OUT_D1]"=v" (reg_in4)
                : [IN_A1]"v" (&ptr_in1[tid]), [IN_B1]"v" (&ptr_in2[tid]), [IN_C1]"v" (&ptr_in3[tid]), [IN_D1]"v" (&ptr_in4[tid])
                : "memory"
            );
        }

        sink0 += reg_in1.x + reg_in2.x + reg_in3.x + reg_in4.x;
        sink1 += reg_in1.y + reg_in2.y + reg_in3.y + reg_in4.y;
        sink2 += reg_in1.z + reg_in2.z + reg_in3.z + reg_in4.z;
        sink3 += reg_in1.w + reg_in2.w + reg_in3.w + reg_in4.w;
    }

    k2_sink[uid] = sink0 + sink1 + sink2 + sink3; // prevent optimization
}

} // namespace k2