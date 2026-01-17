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

#include "../../../mem_bench/MeasurementSeries.hpp"
#include "../../../mem_bench/dtime.hpp"
#include "../../../mem_bench/gpu-clock.cuh"

#include "main.h"
#include "fused_kernel.h"
#include "k1.h"
#include "k2.h"

using namespace std;
namespace cg = cooperative_groups;

#ifndef DEBUG_LEVEL
#define DEBUG_LEVEL 1
#endif

#ifndef K2_PINNED_XCD
#define K2_PINNED_XCD   1    // Kernel 2 pinned xcd
#endif

#ifndef K2_PINNED_HBM
#define K2_PINNED_HBM   3   // Kernel 2 pinned hbm
#endif

#ifndef K2_BPX_MIN
#define K2_BPX_MIN 2
#endif

#ifndef K2_BPX_MAX
#define K2_BPX_MAX 40
#endif

#ifndef K2_TPB
#define K2_TPB 128
#endif

typedef int64_t dtype;


inline uint32_t get_cc(uint32_t xcc_id) {
    return (xcc_id / (XCD_NUM / CC_NUM)) % CC_NUM;
}

int main(int argc, char **argv) {

    // K1 on even XCDs and HBMs while K2 on specific odd XCD and HBM to avoid LLC channel conflict 
    cout << "K1 on all 8 links: "
         << "CC0<->CC2" << " " << "CC2<->CC4" << " " << "CC4<->CC6" << " " << "CC6<->CC0" << " "
         << "using even XCDs and HBMs only."
         << "\n" << flush;
    cout << "K2 on: " << "CC" << get_cc(K2_PINNED_XCD) << "->" << "CC" << get_cc(K2_PINNED_HBM) << " "
         << "using XCD" << K2_PINNED_XCD << "->" << "HBM" << K2_PINNED_HBM
         << "\n" << flush; 

    // Assertions
    assert(K2_BPX_MIN > 1); // In K1 in fused kernel k, two bpx are utilized for two outgoing links of xcd

    std::random_device rd;
    std::mt19937 g(rd());
    
    const size_t cl_bytes = 128;


    // =============================================================================================
    // K1 SETUP
    // =============================================================================================


    // Data allocation 
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

    // Data home identification
    if (k1::home_identification(
            dbuf_base,
            n_dtype_dbuf,
            n_cl_dbuf,
            cl_size,
            cl_bytes,
            skip_factor,
            dtype_home_xcd,
            cl_home_xcd,
            xcd_dtypes,
            xcd_cls) == -1)
        return -1;


    // =============================================================================================
    // K2 SETUP
    // =============================================================================================


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

    // Data home identification
    if (k2::home_identification(
            k2_d_data,
            k2_data_size,
            k2_n_chunks,
            k2_n_datas,
            k2_h_home,
            k2_h_xcd_chunks_size) == -1)
        return -1;

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
    cout << "K2" << " "
         << k2_n_datas * k2_data_size / (1024 * 1024) << " MB " //
         << (k2_h_xcd_chunks_size[0][K2_PINNED_HBM] + k2_h_xcd_chunks_size[1][K2_PINNED_HBM] + k2_h_xcd_chunks_size[2][K2_PINNED_HBM] + k2_h_xcd_chunks_size[3][K2_PINNED_HBM]) * k2_chunk_size / (1024 * 1024) << " MB " //
         << "\n" << flush;
    #endif


    // =============================================================================================
    // KERNEL LAUNCH
    // =============================================================================================


    unsigned int clock = getGPUClock();

    const int EIGHT = 8; // k1 for 8 links
    
    for (int k2_bpx = K2_BPX_MIN; k2_bpx <= K2_BPX_MAX; k2_bpx++) {

        const int64_t iters = max(LEN, (int64_t)100000);
        MeasurementSeries times0, times1, times2, times3, times4, times5, times6, times7; 
        MeasurementSeries k2_bws;

        for (int epoch = 0; epoch < 21; epoch++) {

            // Allocate buf for the FULL n_cl_dbuf domain (i.e., full dbuf domain).
            // In practice this means buf is sized to n_dtype_dbuf (the full dtype domain).
            dtype *buf = nullptr;
            GPU_ERROR(hipMallocManaged(&buf, n_dtype_dbuf * sizeof(dtype)));
            std::memset(buf, 0, n_dtype_dbuf * sizeof(dtype));

            dtype *dummy_buf = nullptr;
            GPU_ERROR(hipMallocManaged(&dummy_buf, sizeof(dtype)));
            dummy_buf[0] = 0;

            // Important! Since only HBM 0 2 4 6 is used, buffer map [0-7] logic: CC 0 1 2 3 0 1 2 3
            vector<dtype*> dbuf_start_vector(EIGHT, nullptr);

            for (int v = 0; v < EIGHT; v++) {

                // Choose LEN cache lines from PINNED_CC, shuffle them
                vector<uint32_t> seq(LEN);
                for (int64_t i = 0; i < LEN; i++) {
                    // Important! v%4*2 => HBM 0 2 4 6 0 2 4 6
                    seq[i] = xcd_cls[(v%4)*2][(size_t)i];
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

                        #if DEBUG_LEVEL >= 2
                        printf("cur_cl=%u lane=%d cur_elem=%zu -> next_cl=%u next_elem=%zu addr=%" PRIxPTR "\n",
                                cur_cl, cl_lane, cur_elem, next_cl, next_elem, next_addr);
                        #endif
                    }
                }

                // Start pointer: lane 0 of the first cache line in seq[]
                size_t start_elem = ((size_t)seq[0] * (size_t)cl_size + 0) * (size_t)skip_factor;
                dtype *dbuf_start = dbuf_base + start_elem;

                dbuf_start_vector[v] = dbuf_start;
            }

            // Copy the full pointer table into device allocation.
            GPU_ERROR(hipMemcpy(dbuf_base, buf, n_dtype_dbuf * sizeof(dtype), hipMemcpyHostToDevice));
            GPU_ERROR(hipDeviceSynchronize());

            // K1 warmup
            // two tb per xcd for two outgoing links utilization
            const int TWO = 2;
            k1::k<dtype><<<XCD_NUM * TWO, 1>>>(
                dbuf_start_vector[0], dbuf_start_vector[1], dbuf_start_vector[2], dbuf_start_vector[3],
                dbuf_start_vector[4], dbuf_start_vector[5], dbuf_start_vector[6], dbuf_start_vector[7],
                dummy_buf, iters
            );
            k1::k<dtype><<<XCD_NUM * TWO, 1>>>(
                dbuf_start_vector[0], dbuf_start_vector[1], dbuf_start_vector[2], dbuf_start_vector[3],
                dbuf_start_vector[4], dbuf_start_vector[5], dbuf_start_vector[6], dbuf_start_vector[7],
                dummy_buf, iters
            );

            GPU_ERROR(hipDeviceSynchronize());


            /***************************************************************************************
             * K2 THREAD BLOCK CONFIGURATION
             * This part is important to saturate link bandwidth 
             * Different from *original* code, now each thread block loads (tb_size/128) chunks per iter
             * To do so, k2_n_warps_per_block is set accordingly
             **************************************************************************************/


            const int k2_n_blocks = (k2_bpx * XCD_NUM);
            const int k2_n_threads_per_warp = 64;

            // *original* code
            // Since each thread loads 16B per iter, `k2_n_warps_per_block` set s.t. each tb loads 1 chunk per iter
            // const int k2_n_warps_per_block = (k2_chunk_size / (k2_n_threads_per_warp * 16)); 

            // modified code
            const int k2_n_warps_per_block = K2_TPB / k2_n_threads_per_warp;

            const int k2_n_threads_per_block = (k2_n_warps_per_block * k2_n_threads_per_warp);
            assert(k2_n_threads_per_block % 128 == 0); // assert multiple 128 => each block loads multiples of 1 chunk per iter
            assert(k2_n_threads_per_block <= 1024); // assert max 1024 threads per block
            const int k2_total_threads = (k2_n_blocks * k2_n_threads_per_block);

            uint32_t *d_k1_startClk, *d_k1_stopClk; 
            GPU_ERROR(hipMalloc((void**)&d_k1_startClk, sizeof(uint32_t) * EIGHT ));
            GPU_ERROR(hipMalloc((void**)&d_k1_stopClk, sizeof(uint32_t) * EIGHT ));

            // per-xcd, per-tb in xcd. record start/stop cycles for bw measurement
            uint32_t *d_k2_startClk, *d_k2_stopClk; 
            GPU_ERROR(hipMalloc((void**)&d_k2_startClk, sizeof(uint32_t) * k2_bpx ));
            GPU_ERROR(hipMalloc((void**)&d_k2_stopClk, sizeof(uint32_t) * k2_bpx ));


            /***************************************************************************************
             * END K2 THREAD BLOCK CONFIGURATION
             ***************************************************************************************/


            #if DEBUG_LEVEL >= 1
            cout << "Launching fused kernel k with"
                 << " n_chunks_per_block_per_iter=" << ((k2_n_threads_per_block * 16) / k2_chunk_size)
                 << " k2_n_threads_per_block=" << k2_n_threads_per_block 
                 << " k2_n_blocks=" << k2_n_blocks 
                 << " n_chunks_per_iter_per_tb=" << ((k2_n_threads_per_block * 16) / k2_chunk_size)
                 << "\n" << flush;
            #endif

            // Launch fused kernel k
            k<dtype><<<k2_n_blocks, k2_n_threads_per_block>>>(
                dbuf_start_vector[0], dbuf_start_vector[1], dbuf_start_vector[2], dbuf_start_vector[3],
                dbuf_start_vector[4], dbuf_start_vector[5], dbuf_start_vector[6], dbuf_start_vector[7],
                dummy_buf, iters, d_k1_startClk, d_k1_stopClk,
                k2_d_xcd_chunks[0], k2_d_xcd_chunks[1], k2_d_xcd_chunks[2], k2_d_xcd_chunks[3],
                k2_d_xcd_chunks_offset[0], k2_d_xcd_chunks_offset[1], k2_d_xcd_chunks_offset[2], k2_d_xcd_chunks_offset[3],
                k2_d_xcd_chunks_size, k2_chunk_size, K2_PINNED_XCD, K2_PINNED_HBM, K2_TPB,
                d_k2_startClk, d_k2_stopClk
            );
            GPU_ERROR(hipDeviceSynchronize());
            GPU_ERROR(hipGetLastError());

            // parse k2 results 

            vector<uint32_t> h_k2_cycles_start(k2_bpx);
            vector<uint32_t> h_k2_cycles_stop(k2_bpx);
            GPU_ERROR(hipMemcpy(h_k2_cycles_start.data(), d_k2_startClk, sizeof(uint32_t) * k2_bpx, hipMemcpyDeviceToHost));
            GPU_ERROR(hipMemcpy(h_k2_cycles_stop.data(), d_k2_stopClk, sizeof(uint32_t) * k2_bpx, hipMemcpyDeviceToHost));

            // for avg lat calc
            double avg_xcd_cycles = 0.0;
            // for global bw calc
            uint32_t min_xcd_cycles_start = 0xFFFFFFFF;
            uint32_t max_xcd_cycles_stop = 0;
            for (int tbid_in_xcd = 0; tbid_in_xcd < k2_bpx; tbid_in_xcd++) {
                // incremental avg
                avg_xcd_cycles += ((h_k2_cycles_stop[tbid_in_xcd] - h_k2_cycles_start[tbid_in_xcd]) - avg_xcd_cycles) / (tbid_in_xcd + 1);

                // set earliest start cycle for this xcd
                if (h_k2_cycles_start[tbid_in_xcd] < min_xcd_cycles_start) {
                    min_xcd_cycles_start = h_k2_cycles_start[tbid_in_xcd];
                }
                // set latest stop cycle for this xcd
                if (h_k2_cycles_stop[tbid_in_xcd] > max_xcd_cycles_stop) {
                    max_xcd_cycles_stop = h_k2_cycles_stop[tbid_in_xcd];
                }
            }


            /***************************************************************************************
             * K2 GLOBAL BANDWIDTH CALCULATION
             * This part is confusing but don't manipulate *bytes*
             * For example, bytes *= (k2_n_threads_per_block / 128) is not needed
             ***************************************************************************************/

            
            // bw based on max cycles among tbs in this xcd
            double bytes = (double)(k2_d_xcd_chunks_size[K2_PINNED_HBM] * k2_chunk_size * k2_n_datas); // total bytes accessed by this xcd
            const double global_time_sec = (double)(max_xcd_cycles_stop - min_xcd_cycles_start) / ((double)clock * 1e6);
            const double global_bw_GBps = (bytes / global_time_sec) / 1e9;
            k2_bws.add(global_bw_GBps);

            #if DEBUG_LEVEL >= 1
            printf("epoch %d xcd %d: global bw %.2f GB/s\n", epoch, K2_PINNED_XCD, global_bw_GBps);
            #endif

            // end parse k2 results

            vector<uint32_t> h_start(EIGHT);
            vector<uint32_t> h_stop(EIGHT);

            GPU_ERROR(hipMemcpy(h_start.data(), d_k1_startClk, sizeof(uint32_t) * EIGHT, hipMemcpyDeviceToHost));
            GPU_ERROR(hipMemcpy(h_stop.data(), d_k1_stopClk, sizeof(uint32_t) * EIGHT, hipMemcpyDeviceToHost));

            // 일단 전체 elapsed cycle을 잡아보자. 더 finer-grained한 측정은 나중에
            times0.add((double)(h_stop[0] - h_start[0]) / ((double)clock * 1e6));
            times1.add((double)(h_stop[1] - h_start[1]) / ((double)clock * 1e6));
            times2.add((double)(h_stop[2] - h_start[2]) / ((double)clock * 1e6));
            times3.add((double)(h_stop[3] - h_start[3]) / ((double)clock * 1e6));
            times4.add((double)(h_stop[4] - h_start[4]) / ((double)clock * 1e6));
            times5.add((double)(h_stop[5] - h_start[5]) / ((double)clock * 1e6));
            times6.add((double)(h_stop[6] - h_start[6]) / ((double)clock * 1e6));
            times7.add((double)(h_stop[7] - h_start[7]) / ((double)clock * 1e6));

            // Cleanup cycle buffers for this epoch
            GPU_ERROR(hipFree(d_k1_startClk));
            GPU_ERROR(hipFree(d_k1_stopClk));
            
            GPU_ERROR(hipFree(buf));

            GPU_ERROR(hipFree(dummy_buf));

        } /* >8 end epoch */

        double dt0 = times0.value();
        double dtmed0 = times0.median();
        double dtmin0 = times0.getPercentile(0.05);
        double dtmax0 = times0.getPercentile(0.95);

        double dt1 = times1.value();
        double dtmed1 = times1.median();
        double dtmin1 = times1.getPercentile(0.05);
        double dtmax1 = times1.getPercentile(0.95);

        double dt2 = times2.value();
        double dtmed2 = times2.median();
        double dtmin2 = times2.getPercentile(0.05);
        double dtmax2 = times2.getPercentile(0.95);

        double dt3 = times3.value();
        double dtmed3 = times3.median();
        double dtmin3 = times3.getPercentile(0.05);
        double dtmax3 = times3.getPercentile(0.95);

        double dt4 = times4.value();
        double dtmed4 = times4.median();
        double dtmin4 = times4.getPercentile(0.05);
        double dtmax4 = times4.getPercentile(0.95);

        double dt5 = times5.value();
        double dtmed5 = times5.median();
        double dtmin5 = times5.getPercentile(0.05);
        double dtmax5 = times5.getPercentile(0.95);

        double dt6 = times6.value();
        double dtmed6 = times6.median();
        double dtmin6 = times6.getPercentile(0.05);
        double dtmax6 = times6.getPercentile(0.95);

        double dt7 = times7.value();
        double dtmed7 = times7.median();
        double dtmin7 = times7.getPercentile(0.05);
        double dtmax7 = times7.getPercentile(0.95);

        cout << setw(2) << k2_bpx << " " << setw(2) << K2_TPB << " "
             << setw(9) << iters << " " << setw(5) << clock << " " 
             << setw(8) << skip_factor * LEN * cl_size * sizeof(dtype) / 1024.0 << " "
             << fixed << setprecision(1) << setw(8) << dt0 * 1000 << " " 
                << setw(7) << setprecision(1) << (double)dt0 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed0 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin0 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax0 / iters * clock * 1000 * 1000 << " "
             << fixed << setprecision(1) << setw(8) << dt1 * 1000 << " " 
                << setw(7) << setprecision(1) << (double)dt1 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed1 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin1 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax1 / iters * clock * 1000 * 1000 << " "
            << fixed << setprecision(1) << setw(8) << dt2 * 1000 << " "
                << setw(7) << setprecision(1) << (double)dt2 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed2 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin2 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax2 / iters * clock * 1000 * 1000 << " "
            << fixed << setprecision(1) << setw(8) << dt3 * 1000 << " "
                << setw(7) << setprecision(1) << (double)dt3 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed3 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin3 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax3 / iters * clock * 1000 * 1000 << " "
            << fixed << setprecision(1) << setw(8) << dt4 * 1000 << " " 
                << setw(7) << setprecision(1) << (double)dt4 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed4 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin4 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax4 / iters * clock * 1000 * 1000 << " "
            << fixed << setprecision(1) << setw(8) << dt5 * 1000 << " " 
                << setw(7) << setprecision(1) << (double)dt5 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed5 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin5 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax5 / iters * clock * 1000 * 1000 << " "
            << fixed << setprecision(1) << setw(8) << dt6 * 1000 << " " 
                << setw(7) << setprecision(1) << (double)dt6 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed6 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin6 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax6 / iters * clock * 1000 * 1000 << " "
            << fixed << setprecision(1) << setw(8) << dt7 * 1000 << " " 
                << setw(7) << setprecision(1) << (double)dt7 / iters * clock * 1000 * 1000 << " "
                << (double)dtmed7 / iters * clock * 1000 * 1000 << " "
                << (double)dtmin7 / iters * clock * 1000 * 1000 << " "
                << (double)dtmax7 / iters * clock * 1000 * 1000 << " "
             << k2_bws.value() << "GB/s" << "\n"
             << flush;
        
    } /* >8 end bpx */

    return 0;
}