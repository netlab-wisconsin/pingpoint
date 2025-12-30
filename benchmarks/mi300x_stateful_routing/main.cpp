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

#include "../../mem_bench/MeasurementSeries.hpp"
#include "../../mem_bench/dtime.hpp"
#include "../../mem_bench/gpu-clock.cuh"
#include "../../mem_bench/gpu-error.h"

#include "../mi300x_mapping/bmk1_1tbx.h"
#include "main.h"
#include "kernel.h"

using namespace std;
namespace cg = cooperative_groups;

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 0
#endif

#ifndef K1_PINNED_XCD
#define K1_PINNED_XCD   0    // Kernel 1 pinned xcd
#endif

#ifndef K1_PINNED_HBM
#define K1_PINNED_HBM   2   // Kernel 1 pinned hbm
#endif

#ifndef K2_PINNED_XCD
#define K2_PINNED_XCD   1    // Kernel 2 pinned xcd
#endif

#ifndef K2_PINNED_HBM
#define K2_PINNED_HBM   3   // Kernel 2 pinned hbm
#endif

#ifndef K2_BPX_MAX
#define K2_BPX_MAX 40
#endif

typedef int64_t dtype;


inline uint32_t get_cc(uint32_t xcc_id) {
    return (xcc_id / (XCD_NUM / CC_NUM)) % CC_NUM;
}

int main(int argc, char **argv) {

    printf("K1 %d->%d K2 %d->%d\n", K1_PINNED_XCD, K1_PINNED_HBM, K2_PINNED_XCD, K2_PINNED_HBM);

    std::random_device rd;
    std::mt19937 g(rd());
    
    const size_t cl_bytes = 128;

    /* K1 data allocation */ 
    dtype *dbuf_base = nullptr;

    const size_t LEN = (1 << 16);
    const size_t cl_size = cl_bytes / sizeof(dtype);
    const size_t skip_factor = 1;
    const size_t multiplicative_factor = XCD_NUM * 1;
    const size_t n_dtype_dbuf = multiplicative_factor * skip_factor * cl_size * LEN;
    const size_t n_cl_dbuf = n_dtype_dbuf / (cl_size * skip_factor);
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

            if (k % (skip_factor * cl_size) == 0) {
                size_t cl_idx = k / (skip_factor * cl_size);
                if (cl_idx < n_cl_dbuf) {
                    cl_home_xcd[cl_idx] = min_xcc;
                    xcd_cls[min_xcc].push_back((uint32_t)cl_idx);
                }
            }
        }

        GPU_ERROR(hipFree(d_cycles));
    
        #if DEBUG_LEVEL >= 2
        for (int x = 0; x < XCD_NUM; x++) {
            cout << "XCD " << x << " has " << xcd_dtypes[x].size() << " dtypes.\n";
            cout << "XCD " << x << " has " << xcd_cls[x].size() << " cache lines.\n";
        }
        #endif
        
        {
            // boundary check
            // single-CC LLC range is 1MB - 16MB
            
            if (xcd_cls[K1_PINNED_HBM].size() * cl_bytes < (1 * 1024 * 1024)) {
                cout << "K1 pinned HBM chunks size " << xcd_cls[K1_PINNED_HBM].size() * cl_bytes << " MB" //
                    << " is smaller than 1 MB" << "\n" << flush;
                return -1;
            }

            if (xcd_cls[K1_PINNED_HBM].size() * cl_bytes > (16 * 1024 * 1024)) {
                cout << "K1 pinned HBM chunks size " << xcd_cls[K1_PINNED_HBM].size() * cl_bytes << " MB" //
                    << " is larger than 16 MB" << "\n" << flush;
                return -1;
            }
        }
    }

    #if DEBUG_LEVEL >= 1
    cout << "K1" << " " << K1_PINNED_XCD << "->" << K1_PINNED_HBM << "(" << get_cc(K1_PINNED_HBM) << ")" << " " //
         << n_dtype_dbuf * sizeof(dtype) / (1024 * 1024) << " MB " //
         << xcd_cls[K1_PINNED_HBM].size() * cl_bytes / (1024 * 1024) << " MB " //
         << "\n" << flush;
    #endif


    /* K2 data allocation */

    const int k2_n_datas = 4;

    const long long k2_n_pages = (128 << 6); // 16GB per input data
    // const long long k2_n_pages = 128; // 256MB per input data
    const int k2_page_size = (2 * 1024 * 1024); // 2MB huge page
    const long long k2_data_size = (k2_n_pages * k2_page_size);

    const int k2_chunk_size = (2 * 1024); // 2KB
    const size_t k2_n_chunks = k2_data_size / k2_chunk_size;

    vector<char*> k2_d_data(k2_n_datas);
    for (int i = 0; i < k2_n_datas; i++) {
        GPU_ERROR(hipMalloc((void**)&k2_d_data[i], k2_data_size + 0x1000));
        k2_d_data[i] = (char*)(((uintptr_t)k2_d_data[i] & ~(0x0FFF)) + 0x1000);
    }

    // Per-chunk xcd identification.
    vector<vector<int>> k2_h_home(k2_n_datas, vector<int>(k2_n_chunks));
    vector<vector<size_t>> k2_h_xcd_chunks_size(k2_n_datas, vector<size_t>(XCD_NUM, 0)); // count #chunks per-data per-xcd. init to 0

    /* K2 data home identification */
    {
        // per-data, per-xcd, per-chunk. record cycles for home identification
        // last two dimensions are flattened
        vector<uint32_t*> d_cycles(k2_n_datas);
        for (int i = 0; i < k2_n_datas; i++) {
            GPU_ERROR(hipMalloc((void**)&d_cycles[i], sizeof(uint32_t) * XCD_NUM * k2_n_chunks));
            #if DEBUG_LEVEL >= 2
            printf("allocated d_cycles[%d] array[%d][%zu]\n", i, XCD_NUM, k2_n_chunks);
            #endif
        }

        for (int i = 0; i < k2_n_datas; i++) {
            #if DEBUG_LEVEL >= 2
            printf("Identifying home xcd for data[%d]...\n", i);
            #endif

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
                #if DEBUG_LEVEL >= 3
                printf("data[%d] chunk[%zu]: home xcd %d\n", i, k, k2_h_home[i][k]);
                #endif
            }
            #if DEBUG_LEVEL >= 2
            for (int x = 0; x < XCD_NUM; x++) {
                printf("data[%d] xcd %d: n_chunks %zu\n", i, x, k2_h_xcd_chunks_size[i][x]);
            }
            printf("\n");
            #endif
        }
    }

    /* group per-xcd chunk pointers */
    
    vector<vector<vector<uint64_t>>> k2_xcd_chunks(k2_n_datas, vector<vector<uint64_t>>(XCD_NUM));
    for (int i = 0; i < k2_n_datas; i++) {
        for (size_t k = 0; k < k2_n_chunks; k++) {
            k2_xcd_chunks[i][k2_h_home[i][k]].push_back(reinterpret_cast<uint64_t>(k2_d_data[i]) + k * (k2_chunk_size));
        }
    }
    // safety check
    for (int i = 0; i < k2_n_datas; i++) {
        for (int x = 0; x < XCD_NUM; x++) {
            assert( k2_xcd_chunks[i][x].size() == k2_h_xcd_chunks_size[i][x] );
            #if DEBUG_LEVEL >= 2
            printf("data[%d] xcd %d: n_chunks %zu\n", i, x, k2_xcd_chunks[i][x].size());
            #endif
        }
    }

    vector<uint64_t*> k2_d_xcd_chunks(k2_n_datas);
    vector<size_t*> k2_d_xcd_chunks_offset(k2_n_datas); // set starting point for each xcd in 1d k2_d_xcd_chunks array
    for (int i = 0; i < k2_n_datas; i++) {
        GPU_ERROR(hipMalloc((void**)&k2_d_xcd_chunks[i], sizeof(uint64_t) * k2_n_chunks ));
        GPU_ERROR(hipMalloc((void**)&k2_d_xcd_chunks_offset[i], sizeof(size_t) * XCD_NUM));
        size_t _offset = 0;
        for (int x = 0; x < XCD_NUM; x++) {
            size_t _n_chunks = k2_h_xcd_chunks_size[i][x];
            GPU_ERROR(hipMemcpy(&k2_d_xcd_chunks[i][_offset], k2_xcd_chunks[i][x].data(), sizeof(uint64_t) * _n_chunks, hipMemcpyHostToDevice));
            k2_d_xcd_chunks_offset[i][x] = _offset;
            _offset += _n_chunks;
        }
    }

    /* count # per-xcd chunk */
    
    vector<size_t> k2_xcd_chunks_size(XCD_NUM, SIZE_MAX);
    for (int i = 0; i < k2_n_datas; i++) {
        for (int x = 0; x < XCD_NUM; x++) {
            // Determine the minimum chunk count for this XCD across all datasets.
            // This establishes a safe common upper bound, preventing out-of-bounds access on datasets with fewer chunks.
            k2_xcd_chunks_size[x] = min(k2_xcd_chunks_size[x], k2_h_xcd_chunks_size[i][x]);
        }   
    }
    // ensure minimal 8MB = 2 * l2_size per-xcd chunks in order to thrash l2
    // conservative. since 2 xcds on same iod actually share the home data, another approach can be
    // ensure minimal 8MB per-iod chunks 
    const int min_n_chunks_per_xcd = ((4*2 * 1024 * 1024) / (k2_chunk_size)); // minimal #chunks >= 8MB
    for (int x = 0; x < XCD_NUM; x++) {
        #if DEBUG_LEVEL >= 2
        printf("xcd %d: min n_chunks %zu\n", x, k2_xcd_chunks_size[x]);
        #endif
        assert (k2_xcd_chunks_size[x] * k2_n_datas >= min_n_chunks_per_xcd); // k2_xcd_chunks_size[x] is minimal #chunks per xcd among all datas. 
    }
    
    size_t *k2_d_xcd_chunks_size;
    GPU_ERROR(hipMalloc((void**)&k2_d_xcd_chunks_size, sizeof(size_t) * XCD_NUM));
    GPU_ERROR(hipMemcpy(k2_d_xcd_chunks_size, k2_xcd_chunks_size.data(), sizeof(size_t) * XCD_NUM, hipMemcpyHostToDevice));

    #if DEBUG_LEVEL >= 1
    cout << "K2" << " " << K2_PINNED_XCD << "->" << K2_PINNED_HBM << "(" << get_cc(K2_PINNED_HBM) << ")" << " " //
         << k2_n_datas * k2_data_size / (1024 * 1024) << " MB " //
         << (k2_h_xcd_chunks_size[0][K2_PINNED_HBM] + k2_h_xcd_chunks_size[1][K2_PINNED_HBM] + k2_h_xcd_chunks_size[2][K2_PINNED_HBM] + k2_h_xcd_chunks_size[3][K2_PINNED_HBM]) * k2_chunk_size / (1024 * 1024) << " MB " //
         << "\n" << flush;
    #endif

    unsigned int clock = getGPUClock();

    for (int k2_bpx = 1; k2_bpx <= K2_BPX_MAX; k2_bpx++) {

        MeasurementSeries times;
        const int64_t iters = max(LEN, (int64_t)100000);

        for (int epoch = 0; epoch < 21; epoch++) {

            // Allocate buf for the FULL n_cl_dbuf domain (i.e., full dbuf domain).
            // In practice this means buf is sized to n_dtype_dbuf (the full dtype domain).
            dtype *buf = nullptr;
            GPU_ERROR(hipMallocManaged(&buf, n_dtype_dbuf * sizeof(dtype)));
            std::memset(buf, 0, n_dtype_dbuf * sizeof(dtype));

            dtype *dummy_buf = nullptr;
            GPU_ERROR(hipMallocManaged(&dummy_buf, sizeof(dtype)));
            dummy_buf[0] = 0;

            // Choose LEN cache lines from PINNED_CC, shuffle them
            vector<uint32_t> seq(LEN);
            for (int64_t i = 0; i < LEN; i++) {
                seq[i] = xcd_cls[K1_PINNED_HBM][(size_t)i];
            }
            shuffle(seq.begin(), seq.end(), g);

            #if DEBUG_LEVEL >= 2
            cout << "Access CL sequence (CL indices in full domain): ";
            for (int64_t i = 0; i < LEN; i++) cout << seq[(size_t)i] << " ";
            cout << "\n";
            #endif

            // Build a cycle over those cache lines.
            // For each cache line, we write pointers for *all lanes* (cl_lane=0..cl_size-1),
            // so the CL is fully populated with valid pointer values.
            //
            // Element index for (cacheline cl_idx, lane cl_lane) is:
            //   elem = (cl_idx * cl_size + cl_lane) * skip_factor
            //
            // Stored value is an absolute device address:
            //   (uintptr_t)dbuf_base + next_elem * sizeof(dtype)
            //
            for (int cl_lane = 0; cl_lane < cl_size; cl_lane++) {
                for (int64_t i = 0; i < LEN; i++) {
                    uint32_t cur_cl  = seq[(size_t)i];
                    uint32_t next_cl = seq[(size_t)((i + 1) % LEN)];

                    size_t cur_elem  = ((size_t)cur_cl  * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;
                    size_t next_elem = ((size_t)next_cl * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;

                    // Bounds safety (should hold if n_cl_dbuf is consistent)
                    if (cur_elem >= n_dtype_dbuf || next_elem >= n_dtype_dbuf) {
                        cerr << "BUG: elem OOB: cur_elem=" << cur_elem
                                << " next_elem=" << next_elem
                                << " n_dtype_dbuf=" << n_dtype_dbuf << "\n";
                        GPU_ERROR(hipFree(buf));
                        GPU_ERROR(hipFree(dummy_buf));
                        GPU_ERROR(hipFree(dbuf_base));
                        return 1;
                    }

                    uintptr_t next_addr = (uintptr_t)dbuf_base + next_elem * sizeof(dtype);
                    buf[cur_elem] = (dtype)next_addr;

                    #if DEBUG_LEVEL >= 3
                    printf("cur_cl=%u lane=%d cur_elem=%zu -> next_cl=%u next_elem=%zu addr=%" PRIxPTR "\n",
                            cur_cl, cl_lane, cur_elem, next_cl, next_elem, next_addr);
                    #endif
                }
            }

            // Copy the full pointer table into device allocation.
            GPU_ERROR(hipMemcpy(dbuf_base, buf, n_dtype_dbuf * sizeof(dtype), hipMemcpyHostToDevice));
            GPU_ERROR(hipDeviceSynchronize());

            // Start pointer: lane 0 of the first cache line in seq[]
            size_t start_elem = ((size_t)seq[0] * (size_t)cl_size + 0) * (size_t)skip_factor;
            dtype *dbuf_start = dbuf_base + start_elem;

            
            // K1 warmup
            k1<dtype><<<XCD_NUM, 1>>>(dbuf_start, dummy_buf, iters, K1_PINNED_XCD);
            k1<dtype><<<XCD_NUM, 1>>>(dbuf_start, dummy_buf, iters, K1_PINNED_XCD);
            GPU_ERROR(hipDeviceSynchronize());


            /* K2 thread block configuration */

            const int k2_n_blocks = (k2_bpx * XCD_NUM);
            const int k2_n_threads_per_warp = 64;

            // Since each thread loads 16B per iter, `k2_n_warps_per_block` set s.t. each tb loads 1 chunk per iter
            const int k2_n_warps_per_block = (k2_chunk_size / (k2_n_threads_per_warp * 16)); 
            assert((k2_n_threads_per_warp * k2_n_warps_per_block) <= 1024); // assert max 1024 threads per block

            const int k2_n_threads_per_block = (k2_n_warps_per_block * k2_n_threads_per_warp);
            const int k2_total_threads = (k2_n_blocks * k2_n_threads_per_block);

            /* fused kernel launch */

            uint32_t *d_k1_startClk, *d_k1_stopClk;
            GPU_ERROR(hipMalloc(&d_k1_startClk, sizeof(uint32_t) * iters));
            GPU_ERROR(hipMalloc(&d_k1_stopClk, sizeof(uint32_t) * iters));
            
            k<dtype><<<k2_n_blocks, k2_n_threads_per_block>>>(
                dbuf_start, dummy_buf, iters, K1_PINNED_XCD, d_k1_startClk, d_k1_stopClk,
                k2_d_xcd_chunks[0], k2_d_xcd_chunks[1], k2_d_xcd_chunks[2], k2_d_xcd_chunks[3],
                k2_d_xcd_chunks_offset[0], k2_d_xcd_chunks_offset[1], k2_d_xcd_chunks_offset[2], k2_d_xcd_chunks_offset[3],
                k2_d_xcd_chunks_size, k2_chunk_size, K2_PINNED_XCD, K2_PINNED_HBM,
                nullptr, nullptr
            );
            GPU_ERROR(hipDeviceSynchronize());
            GPU_ERROR(hipGetLastError());

            vector<uint32_t> h_start(iters);
            vector<uint32_t> h_stop(iters);

            GPU_ERROR(hipMemcpy(h_start.data(), d_k1_startClk, sizeof(uint32_t) * iters, hipMemcpyDeviceToHost));
            GPU_ERROR(hipMemcpy(h_stop.data(), d_k1_stopClk, sizeof(uint32_t) * iters, hipMemcpyDeviceToHost));

            // for (int64_t i = 0; i < iters; i++) {
            //     uint32_t start_cycles = h_start[(size_t)i];
            //     uint32_t stop_cycles  = h_stop[(size_t)i];
            //     uint32_t elapsed = stop_cycles - start_cycles;
            //     printf("K1 iter %lld: start %u stop %u elapsed %u\n", i, start_cycles, stop_cycles, elapsed);
            //     // times.add((double)elapsed / (double)clock);
            // }

            // 일단 전체 elapsed cycle을 잡아보자. 더 finer-grained한 측정은 나중에
            times.add((double)(h_stop[0] - h_start[0]) / ((double)clock * 1e6));

            // Cleanup cycle buffers for this epoch
            GPU_ERROR(hipFree(d_k1_startClk));
            GPU_ERROR(hipFree(d_k1_stopClk));
            
            GPU_ERROR(hipFree(buf));
            GPU_ERROR(hipFree(dummy_buf));
            // GPU_ERROR(hipFree(dbuf_base));

        } /* >8 end epoch */

        double dt = times.value();
        double dtmed = times.median();
        double dtmin = times.getPercentile(0.05);
        double dtmax = times.getPercentile(0.95);

        cout << setw(2) << k2_bpx << " "
             << setw(9) << iters << " " << setw(5) << clock << " " //
             << setw(8) << skip_factor * LEN * cl_size * sizeof(dtype) / 1024.0 << " "
             << fixed << setprecision(1) << setw(8) << dt * 1000 << " " //
             << setw(7) << setprecision(1) << (double)dt / iters * clock * 1000 * 1000 << " "
             << (double)dtmed / iters * clock * 1000 * 1000 << " "
             << (double)dtmin / iters * clock * 1000 * 1000 << " "
             << (double)dtmax / iters * clock * 1000 * 1000 << "\n"
             << flush;
        
    } /* >8 end bpx */

    return 0;
}