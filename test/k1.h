#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

#include "main.h"

using namespace std;
namespace cg = cooperative_groups;

#define DEBUG_K1_HOME 0
#define DEBUG_K1_KERNEL 0

#ifndef USE_GLOBAL_BARRIER
#define USE_GLOBAL_BARRIER 0
#endif

namespace k1 {

typedef int64_t dtype;

__device__ int _d_barrier_count = 0;
__device__ int _d_barrier_sense = 0;

__device__ __forceinline__
void _global_barrier()
{
    __shared__ int local_sense; // One local copy per block
    if (threadIdx.x == 0) {
        int old_sense = _d_barrier_sense;
        local_sense   = !old_sense;

        int arrived = atomicAdd(&_d_barrier_count, 1);
        if (arrived == gridDim.x - 1) {
            _d_barrier_count = 0;
            __threadfence(); // Make all global writes visible before releasing others
            _d_barrier_sense = local_sense;
        }
    }
    __syncthreads(); // Make sure all threads in the block see local_sense
    while (_d_barrier_sense != local_sense) {} // Spin until last block flips global sense
    __syncthreads(); // Ensure all threads in this block observe the sense change before proceeding
}

typedef int64_t dtype;

template <typename T>
__global__ void identify_home(T *data, uint32_t *cycles, const long long n_dtypes) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    assert(XCD_NUM == gridDim.x);
    assert(blockDim.x == 128 / sizeof(T));

    // printf("blockDim.x: %d, n_dtypes: %lld\n", blockDim.x, n_dtypes);

    // for this bmk, both _global_barrier and cooperative_groups are supported
    #if not USE_GLOBAL_BARRIER
    cg::grid_group grid = cg::this_grid();
    #endif

    // print (xcc_id,cu_id) of each block
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

#if DEBUG_K1_HOME
    if (tid == 0) {
        printf("bid %d: (xcc_id: %u, se_id: %u, cu_id: %u)\n", bid, xcc_id, se_id, cu_id);

    }
#endif

    const size_t n_iter = n_dtypes / blockDim.x;
    const size_t inner_size = min(n_dtypes * sizeof(T), 64 * 1024 * 1024); // 64MB per inner loop (given TLB lat jump at 64MB)
    // const size_t n_outer = (sizeof(int64_t) * n_dtypes) / inner_size;
    const size_t n_outer = ((sizeof(int64_t) * n_dtypes) + inner_size - 1) / inner_size;
    const size_t n_inner = inner_size / (sizeof(int64_t) * blockDim.x); // num of accesses per tb in inner loop
    // printf("n_outer: %zu, n_inner: %zu, n_iter: %zu\n", n_outer, n_inner, n_iter);
    assert (n_iter <= n_outer * n_inner);

#if DEBUG_K1_HOME
    if (tid == 0) {
        printf("bid %d: n_outer: %zu, n_inner: %zu, n_iter: %zu\n", bid, n_outer, n_inner, n_iter);
    }
#endif
    
    for (size_t i = 0; i < n_outer; i++) {
        // warmup 
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (index >= n_dtypes) continue; // boundary check
            asm volatile(
                "flat_load_dwordx2 v[0:1], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data[index])
                : "memory", "v0", "v1"
            );
        }
        #if USE_GLOBAL_BARRIER
        _global_barrier();
        #else
        grid.sync();
        #endif

        // measurement
        for (size_t j = 0; j < n_inner; j++) {
            size_t index = (i * n_inner + j) * blockDim.x + tid;
            if (index >= n_dtypes) continue; // boundary check
            uint32_t start = __builtin_readcyclecounter();
            asm volatile(
                "flat_load_dwordx2 v[0:1], %0\n\t"
                "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                :
                : "v"(&data[index])
                : "memory", "v0", "v1"
            );
            uint32_t end = __builtin_readcyclecounter();
            uint32_t cycle = end - start;
            const size_t cycles_index = (size_t)xcc_id * n_dtypes + index; // (01/25/25) fix: from int to size_t, to avoid overflow
            cycles[cycles_index] = cycle;
            #if USE_GLOBAL_BARRIER
            _global_barrier();
            #else
            grid.sync();
            #endif
        }
    }
}


int home_identification(
    dtype *dbuf_base,
    const size_t n_dtype_dbuf,
    const size_t n_cl_dbuf,
    const size_t cl_size,
    const size_t cl_bytes,
    const size_t skip_factor,
    vector<uint32_t> &dtype_home_xcd,
    vector<uint32_t> &cl_home_xcd,
    vector<vector<uint32_t>> &xcd_dtypes,
    vector<vector<uint32_t>> &xcd_cls
) {
    uint32_t *d_cycles = nullptr;
    gpuErrchk(hipMalloc(&d_cycles, n_dtype_dbuf * (size_t)XCD_NUM * sizeof(uint32_t)));

    void *kernel_args[] = {
        (void *)&dbuf_base,
        (void *)&d_cycles,
        (void *)&n_dtype_dbuf,
    };

    #if DEBUG_K1_HOME
    cout << "K1 home identification kernel launch parameters:\n"
         << "  n_dtype_dbuf: " << n_dtype_dbuf
         << "  n_cl_dbuf: " << n_cl_dbuf 
         << "\n" << flush;
    #endif

    #if not USE_GLOBAL_BARRIER
    gpuErrchk(hipLaunchCooperativeKernel(
        identify_home<dtype>, dim3(1 * XCD_NUM), dim3(128 / sizeof(dtype)),
        kernel_args, 0, 0));
    #else
    gpuErrchk(hipLaunchKernel(
        (void*)identify_home<dtype>, dim3(XCD_NUM), dim3(128 / sizeof(dtype)),
        kernel_args, 0, 0));
    #endif
    gpuErrchk(hipDeviceSynchronize());

    vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(n_dtype_dbuf));
    for (int x = 0; x < XCD_NUM; x++) {
        gpuErrchk(hipMemcpy(h_cycles[x].data(),
                            d_cycles + (size_t)x * n_dtype_dbuf,
                            sizeof(uint32_t) * n_dtype_dbuf,
                            hipMemcpyDeviceToHost));
    }

    for (size_t k = 0; k < n_dtype_dbuf; k++) {
        uint32_t min_cycles = UINT32_MAX;
        int min_xcc = -1;
        for (int x = 0; x < XCD_NUM; x++) {
            uint32_t c = h_cycles[x][k];
            if (c < min_cycles) {
                min_cycles = c;
                min_xcc = x;
            }
            #if DEBUG_K1_HOME
            cout << "dtype " << k << " xcc " << x << " cycles " << c << "\n";
            #endif
        }

        dtype_home_xcd[k] = min_xcc;
        xcd_dtypes[min_xcc].push_back((uint32_t)k);

        if (k % (skip_factor * cl_size) == 0) {
            size_t cl_idx = k / (skip_factor * cl_size);
            if (cl_idx < n_cl_dbuf) {
                cl_home_xcd[cl_idx] = min_xcc;
                xcd_cls[min_xcc].push_back((uint32_t)cl_idx);
            }
        }
    }

    gpuErrchk(hipFree(d_cycles));

    #if DEBUG_K1_HOME
    for (int x = 0; x < XCD_NUM; x++) {
        cout << "XCD " << x << " has " << xcd_dtypes[x].size() << " dtypes.\n";
        cout << "XCD " << x << " has " << xcd_cls[x].size() << " cache lines.\n";
        cout << "XCD " << x << " has " << xcd_cls[x].size() * cl_bytes / (1024 * 1024) << " MB of pinned HBM chunks.\n";
    }
    #endif

    // boundary check
    for (int x = 0; x < XCD_NUM; x++) {
#if 0
        if (xcd_cls[x].size() * cl_bytes < (64 * 1024 * 1024)) {
            cout << "K1 pinned HBM" << "[" << x << "]" << " chunks size " << xcd_cls[x].size() * cl_bytes / (1024 * 1024) << " MB" //
                << " is smaller than 64 MB" << "\n" << flush;
            return -1;
        }
#endif
    }

    return 0;
}

// Note (01/28/25) Converted to __device__ to call in fused ppnt kernel
template <typename T>
__device__ void k(
    T *buf, T *__restrict__ dummy_buf, const int64_t N,
    /* Pingout */ 
    uint64_t *po_iterClk
) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int uid = bid * blockDim.x + tid;
    assert(tid == 0); // single thread per block

#if 0
    uint32_t cu_id, xcc_id, se_id;
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 13, 3)" : "=r"(se_id));
    asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
    assert (se_id == se && cu_id == cu); // must be set by host code
    if (xcc_id != xcd) return; // prelude xcd
#if DEBUG_K1_KERNEL
    printf("(bid:%d,tid:%d,uid:%d) k1 launched on xcd:%u se:%u cu:%u\n",
           bid, tid, uid, xcc_id, se_id, cu_id);
#endif
#endif

    T *idx = buf;

    const int unroll_factor = 32;
#pragma unroll 1
    for (int64_t n = 0; n < N; n += unroll_factor) {
#pragma unroll
        for (int u = 0; u < unroll_factor; u++) {
            // Note (01/28/25) inserted for pingout
            uint64_t start = 0;
            start = __builtin_readcyclecounter();
            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); 

            idx = (T *)*idx;

            // Note (01/28/25) inserted for pingout
            uint64_t end = 0;
            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");
            end = __builtin_readcyclecounter();
            po_iterClk[n + u] = end - start;
        }
    }

    if (uid > 12313) {
        dummy_buf[0] = (T)idx;
    }
}

} // namespace k1