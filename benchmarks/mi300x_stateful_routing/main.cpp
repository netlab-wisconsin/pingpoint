#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_complex.h>

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sys/time.h>
#include <vector>

// #include "../../mem_bench/MeasurementSeries.hpp"
// #include "../../mem_bench/dtime.hpp"
#include "../../mem_bench/gpu-clock.cuh"
#include "../../mem_bench/gpu-error.h"

#include "../mi300x_mapping/bmk1_1tbx.h"
#include "main.h"

using namespace std;
namespace cg = cooperative_groups;

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#define K1_PINNED_XCD           0    // Kernel 1 pinned xcd
#ifndef K1_PINNED_XCD_CHUNKS 
#define K1_PINNED_XCD_CHUNKS    2
#endif

#define K2_PINNED_XCD           1    // Kernel 2 pinned xcd
#ifndef K2_PINNED_XCD_CHUNKS
#define K2_PINNED_XCD_CHUNKS    3
#endif

typedef int64_t dtype;

inline uint32_t get_cc(uint32_t xcc_id) {
    return (xcc_id / (XCD_NUM / CC_NUM)) % CC_NUM;
}

int main(int argc, char **argv) {
    printf("K1 pinned XCD: %d K1 pinned CC: %d\n", K1_PINNED_XCD, get_cc(K1_PINNED_XCD_CHUNKS));
    printf("K2 pinned XCD: %d K2 pinned CC: %d\n", K2_PINNED_XCD, get_cc(K2_PINNED_XCD_CHUNKS));

    unsigned int clock = getGPUClock();
    const int cl_size = 128 / (int)sizeof(dtype);

    std::random_device rd;
    std::mt19937 g(rd());

    /* K1 data allocation */ 
    dtype *dbuf_base = nullptr;

    const int64_t LEN = 8192;
    const int skip_factor = 1;
    const size_t multiplicative_factor = XCD_NUM * 1;
    const size_t n_dtype_dbuf = multiplicative_factor * (size_t)skip_factor * (size_t)cl_size * (size_t)LEN;
    const size_t n_cl_dbuf = n_dtype_dbuf / ((size_t)cl_size * (size_t)skip_factor);
    GPU_ERROR(hipMalloc(&dbuf_base, n_dtype_dbuf * sizeof(dtype)));

    // Per-dtype xcd identification.
    vector<uint32_t> dtype_home_xcd(n_dtype_dbuf, (uint32_t)-1);
    vector<vector<uint32_t>> xcd_dtypes(XCD_NUM);

    // Per-cacheline xcd identification.
    vector<uint32_t> cl_home_xcd(n_cl_dbuf, (uint32_t)-1);
    vector<vector<uint32_t>> xcd_cls(XCD_NUM);

    /* K1 data home identification */
    {
        uint32_t *d_cycles = nullptr;
        GPU_ERROR(hipMalloc(&d_cycles, n_dtype_dbuf * (size_t)XCD_NUM * sizeof(uint32_t)));

        void *kernel_args[] = {
            (void *)&dbuf_base,
            (void *)&d_cycles,
            (void *)&n_dtype_dbuf,
        };

        GPU_ERROR(hipLaunchCooperativeKernel(
            _identify_home<dtype>, dim3(1 * XCD_NUM), dim3(128 / sizeof(dtype)),
            kernel_args, 0, 0));

        GPU_ERROR(hipDeviceSynchronize());

        vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(n_dtype_dbuf));
        for (int x = 0; x < XCD_NUM; x++) {
            GPU_ERROR(hipMemcpy(h_cycles[x].data(),
                                d_cycles + (size_t)x * n_dtype_dbuf,
                                sizeof(uint32_t) * n_dtype_dbuf,
                                hipMemcpyDeviceToHost));
        }

        for (size_t k = 0; k < n_dtype_dbuf; k++) {
            uint32_t min_cycles = 0xFFFFFFFF;
            int min_xcc = -1;
            for (int x = 0; x < XCD_NUM; x++) {
                uint32_t c = h_cycles[x][k];
                if (c < min_cycles) {
                    min_cycles = c;
                    min_xcc = x;
                }
                #if DEBUG_LEVEL >= 3
                cout << "dtype " << k << " xcc " << x << " cycles " << c << "\n";
                #endif
            }

            dtype_home_xcd[k] = min_xcc;
            xcd_dtypes[min_xcc].push_back((uint32_t)k);

            if (k % ((size_t)skip_factor * (size_t)cl_size) == 0) {
                size_t cl_idx = k / ((size_t)skip_factor * (size_t)cl_size);
                if (cl_idx < n_cl_dbuf) {
                    cl_home_xcd[cl_idx] = min_xcc;
                    xcd_cls[min_xcc].push_back((uint32_t)cl_idx);
                }
            }
        }

        GPU_ERROR(hipFree(d_cycles));
    
        #if DEBUG_LEVEL >= 1
        for (int x = 0; x < XCD_NUM; x++) {
            cout << "XCD " << x << " has " << xcd_dtypes[x].size() << " dtypes.\n";
            cout << "XCD " << x << " has " << xcd_cls[x].size() << " cache lines.\n";
        }
        #endif
        
        if (xcd_cls[K1_PINNED_XCD_CHUNKS].size() < (size_t)LEN) {
            // TODO: Handle insufficient cache lines
            printf("TODO: handle corner case\n");

            // #if DEBUG_LEVEL >= 1
            //     cout << "Skipping epoch " << epoch << ": XCD " << K1_PINNED_XCD_CHUNKS
            //          << " only has " << xcd_cls[K1_PINNED_XCD_CHUNKS].size()
            //          << " cache lines, need LEN=" << LEN << "\n";
            // #endif
            // GPU_ERROR(hipFree(dbuf_base));
            // continue;
        }
    }

    /* K2 thread block configuration*/
    const int k2_bpx = 1; // TODO: scale
    const int k2_n_blocks = (k2_bpx * XCD_NUM);

    const int k2_page_size = (2 * 1024 * 1024); // 2MB huge page
    const int k2_chunk_size = (2 * 1024); // 2KB
    const int k2_n_threads_per_warp = 64;
    const int k2_n_warps_per_block = (k2_chunk_size / (k2_n_threads_per_warp * 16)); // set to make each tb load 1 chunk per iter. each thread loads 16B per iter.
    assert( (k2_n_threads_per_warp * k2_n_warps_per_block) <= 1024 );   

    const int k2_n_threads_per_block = (k2_n_warps_per_block * k2_n_threads_per_warp);
    const int k2_total_threads = (k2_n_blocks * k2_n_threads_per_block);
    printf("k2_n_blocks: %d, k2_n_blocks_per_xcd: %d, k2_n_warps_per_block: %d, k2_n_threads_per_warp: %d\n", k2_n_blocks, k2_bpx, k2_n_warps_per_block, k2_n_threads_per_warp);

    /* K2 data allocation */

    const int k2_n_datas = 4;

    // const long long k2_n_pages = (128 << 6); // 16GB per input data
    const long long k2_n_pages = 128; // 256MB per input data
    const long long k2_data_size = (k2_n_pages * k2_page_size);
    const size_t k2_n_chunks = k2_data_size / k2_chunk_size;
    printf("k2_n_datas: %d, k2_data_size: %lld MB, k2_chunk_size: %d KB\n", k2_n_datas, k2_data_size / (1024 * 1024), k2_chunk_size / 1024);

    vector<char*> k2_d_data(k2_n_datas);
    for (int i = 0; i < k2_n_datas; i++) {
        GPU_ERROR(hipMalloc((void**)&k2_d_data[i], k2_data_size + 0x1000));
        k2_d_data[i] = (char*)(((uintptr_t)k2_d_data[i] & ~(0x0FFF)) + 0x1000);
    }

    

    // home identification for k2 data
    vector<vector<int>> k2_h_home(k2_n_datas, vector<int>(k2_n_chunks));
    vector<vector<size_t>> k2_h_xcd_chunks_size(k2_n_datas, vector<size_t>(XCD_NUM, 0)); // count #chunks per-data per-xcd. init to 0

    {
        // per-data, per-xcd, per-chunk. record cycles for home identification
        // last two dimensions are flattened
        vector<uint32_t*> d_cycles(k2_n_datas);
        for (int i = 0; i < k2_n_datas; i++) {
            GPU_ERROR(hipMalloc((void**)&d_cycles[i], sizeof(uint32_t) * XCD_NUM * k2_n_chunks));
            printf("allocated d_cycles[%d] array[%d][%zu]\n", i, XCD_NUM, k2_n_chunks);
        }

        for (int i = 0; i < k2_n_datas; i++) {
            printf("Identifying home xcd for data[%d]...\n", i);

            void *kernel_args[] = {
                (void*)&k2_d_data[i],
                (void*)&k2_data_size,
                (void*)&d_cycles[i], // XCD_NUM * n_chunks
                (void*)&k2_n_chunks
            };

            GPU_ERROR(hipLaunchCooperativeKernel(
                identify_home,
                dim3(1 * XCD_NUM), dim3(128), 
                kernel_args, 0, 0
            ));
            GPU_ERROR(hipDeviceSynchronize());

            /* retrieve and process results */

            vector<vector<uint32_t>> h_cycles(XCD_NUM, vector<uint32_t>(k2_n_chunks));
            for (int x = 0; x < XCD_NUM; x++) {
                GPU_ERROR(hipMemcpy(
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
                #if DEBUG
                printf("data[%d] chunk[%zu]: home xcd %d\n", i, k, k2_h_home[i][k]);
                #endif
            }
            #if DEBUG
            for (int x = 0; x < XCD_NUM; x++) {
                printf("data[%d] xcd %d: n_chunks %zu\n", i, x, k2_h_xcd_chunks_size[i][x]);
            }
            printf("\n");
            #endif
        }

    }

    // gpu-lat L195부터. for loop 내로 넣어야 할 수 도.

    return 0;
}