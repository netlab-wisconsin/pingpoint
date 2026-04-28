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

#include "../mem_bench/gpu-clock.cuh"
#include "../mem_bench/MeasurementSeries.hpp"

#include "main.h"
#include "ppnt.h"
#include "ppnt_only.h"
#include "moe_only.h"
#include "k1.h"
#include "k2.h"

#define DEBUG_LEVEL 1 // 0: MoE, 1: PPNT, 2+: Misc
#define FAST 1 // set for faster debugging, set 0 for actual measurement
#define PPNT_PLAN_SELECTED_ONLY 1 // set to only generate selected plans

using namespace std;

#define TARGET_BLOCKDIM_X (1024)

#define DISABLE_K1_PLANS 1
#define DISABLE_K2_PLANS 0

#define PPNT_PROFILE_ATTENTION1 1
#define PPNT_PROFILE_ATTENTION2 0
#define PPNT_PROFILE_SCATTER    1
#define PPNT_PROFILE_FFN1       1
#define PPNT_PROFILE_RELU       0        
#define PPNT_PROFILE_FFN2       0
#define PPNT_PROFILE_GATHER     1


int main(int argc, char **argv) {

    unsigned int clock = getGPUClock();
    
    // =============================================================================================
    // K1 SETUP
    // =============================================================================================
    
    // Starting pointers to pchase in each HBM
    // Original varname was dbuf_start_vector
    vector<k1::dtype*> k1_dbuf_start_ptrs_per_hbm(HBM_NUM, nullptr);
    // (Note 01/29/26) dummy_buf is very important to avoid compiler optimization
    // inside the k1 kernel.
    k1::dtype *k1_dummy_buf;
    size_t k1_profile_iters = -1;
    {
#if !(DISABLE_K1_PLANS) 

        k1::dtype *dbuf_base = nullptr;
        gpuErrchk(hipMallocManaged(&k1_dummy_buf, sizeof(k1::dtype)));
        k1_dummy_buf[0] = 0;
        /*
         * LEN: The number of cache lines targeted for the pointer-chasing chain per HBM stack.
         * n_dtype_dbuf: Size of primary data buffer (dbuf_base) in unit dtypes
         * n_cl_dbuf: Size of primary data buffer (dbuf_base) in unit cache lines
         */

#if (FAST)
        // When profiling/fast, use reduced LEN with *2 mult factor to avoid below OOB error
        // But don't further reduce LEN. 1<<15 is almost minimal for bypassing L2.
        const size_t LEN = (1 << 16); // 8MB (per HBM)
        const size_t multiplicative_factor = XCD_NUM * 2;
        k1_profile_iters = 10000;
#elif 1
        const size_t LEN = (1 << 22); // 512MB (per HBM)
        const size_t multiplicative_factor = XCD_NUM * 1;
        k1_profile_iters = max(LEN, 10000);
#else
        // TODO: initiate two different buffers for LLC and HBM tests
        const size_t K1_LEN_LLC = (1 << 16);
        const size_t K1_LEN_HBM = (1 << 22);
        const size_t multiplicative_factor = XCD_NUM * 1;
        // TODO: set k1_profile_iters
#endif 

        const size_t cl_bytes = 128;
        const size_t cl_size = cl_bytes / sizeof(k1::dtype);
        const size_t skip_factor = 1;
        const size_t n_dtype_dbuf = multiplicative_factor * skip_factor * cl_size * LEN;
        const size_t n_cl_dbuf = n_dtype_dbuf / (cl_size * skip_factor);
        gpuErrchk(hipMalloc(&dbuf_base, n_dtype_dbuf * sizeof(k1::dtype)));

        // Per-dtype hbm identification.
        vector<uint32_t> dtype_home_xcd(n_dtype_dbuf, (uint32_t)-1);
        vector<vector<uint32_t>> hbm_dtypes(HBM_NUM);

        // Per-cacheline hbm identification.
        vector<uint32_t> cl_home_xcd(n_cl_dbuf, (uint32_t)-1);
        vector<vector<uint32_t>> hbm_cls(HBM_NUM);

        if (k1::home_identification(
                dbuf_base, n_dtype_dbuf, n_cl_dbuf, cl_size, cl_bytes, skip_factor,
                dtype_home_xcd, cl_home_xcd, hbm_dtypes, hbm_cls) == -1)
            return -1;

#if DEBUG_LEVEL >= 1
        for (int v = 0; v < HBM_NUM; v++) {
            string level = hbm_dtypes[v].size() * sizeof(k1::dtype) > L2_SIZE ? 
                        (hbm_dtypes[v].size() * sizeof(k1::dtype) > LLC_SIZE ? "hbm" : "llc") : "l2";
            cout << "K1 pinned data: hbm" << v << " "
                 << hbm_dtypes[v].size() * sizeof(k1::dtype) / (1024 * 1024) << "MB at " << level << "\n" << flush;
        }
#endif

        // buf: Temporary host-side pointer-chase table, with size equaling dbuf_base
        // each element stores the dbuf_base addr to chase next when arrived at that element
        // (Note 01/29/26) buf is fine to be host-side buffer since it's copied back to dbuf_base.
        // So, I changed it from hipMallocManaged to normal host vector allocation.
        vector<k1::dtype> buf(n_dtype_dbuf, 0);
        random_device rd;
        mt19937 g(rd());
        
        for (int v = 0; v < HBM_NUM; v++) {

            // Choose LEN cache lines and shuffle them
            vector<uint32_t> seq(LEN);
            for (size_t i = 0; i < LEN; i++) seq[i] = hbm_cls[v][i];
            shuffle(seq.begin(), seq.end(), g);

#if DEBUG_LEVEL >= 3
            cout << "Access CL sequence (CL indices in full domain): ";
            for (int64_t i = 0; i < LEN; i++) cout << seq[i] << " ";
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
                for (size_t i = 0; i < LEN; i++) {
                    uint32_t cur_cl  = seq[i];
                    uint32_t next_cl = seq[(i + 1) % LEN];

                    size_t cur_elem  = ((size_t)cur_cl  * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;
                    size_t next_elem = ((size_t)next_cl * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;

                    // Bounds safety (should hold if n_cl_dbuf is consistent)
                    if (cur_elem >= n_dtype_dbuf || next_elem >= n_dtype_dbuf) {
                        cerr << "BUG: elem OOB: cur_elem=" << cur_elem
                                << " next_elem=" << next_elem
                                << " n_dtype_dbuf=" << n_dtype_dbuf << "\n";
                        gpuErrchk(hipFree(k1_dummy_buf));
                        gpuErrchk(hipFree(dbuf_base));
                        return 1;
                    }

                    uintptr_t next_addr = (uintptr_t)dbuf_base + next_elem * sizeof(k1::dtype);
                    buf[cur_elem] = (k1::dtype)next_addr;

#if DEBUG_LEVEL >= 3
                    uintptr_t cur_addr = (uintptr_t)dbuf_base + cur_elem * sizeof(k1::dtype);
                    printf("cur_cl=%u lane=%d cur_elem=%zu cur_addr=%" PRIxPTR " -> next_cl=%u next_elem=%zu next_addr=%" PRIxPTR "\n",
                        cur_cl, cl_lane, cur_elem, cur_addr, next_cl, next_elem, next_addr);
#endif
                }
            }
            size_t start_elem = ((size_t)seq[0] * cl_size + 0) * skip_factor;
            k1_dbuf_start_ptrs_per_hbm[v] = dbuf_base + start_elem;
        }

        // Copy the full pointer table into device allocation.
        gpuErrchk(hipMemcpy(dbuf_base, buf.data(), n_dtype_dbuf * sizeof(k1::dtype), hipMemcpyHostToDevice));
        gpuErrchk(hipDeviceSynchronize());
#endif // DISABLE_K1_PLANS
    }


    // =============================================================================================
    // K2 SETUP
    // =============================================================================================

    // Number of k2 datas
    const int k2_n_datas = 4;
    // Chunk pointers, sorted by allocated HBM (per each data)
    vector<uint64_t*> k2_d_chunks_per_hbm(k2_n_datas);
    // Starting offset for each HBM's chunks in the k2_d_chunks_per_hbm (per each data)
    vector<vector<size_t>> k2_h_offsets(k2_n_datas, vector<size_t>(HBM_NUM));
    // Minimal allocated #chunks across k2 datas (per HBM)
    // e.g, if 100/90/80/70 chunks allocated to hbm0 in datas 0/1/2/3, k2_d_chunks_per_hbm_count[0] = 70
    vector<size_t> k2_min_num_chunks_over_n_datas(HBM_NUM);
    size_t k2_profile_iters = -1;
    {
#if !(DISABLE_K2_PLANS)

#if (FAST)
        const long long k2_n_pages = 512; // set 1024MB per input data, for faster debugging
#else
        const long long k2_n_pages = (128 << 6); // set 16GB per input data
#endif
        const int k2_page_size = PAGE_SIZE;
        const long long k2_data_size = (k2_n_pages * k2_page_size);
        const int k2_chunk_size = CHUNK_SIZE;
        const size_t k2_n_chunks = k2_data_size / k2_chunk_size;

        vector<char*> k2_d_data(k2_n_datas);
        for (int i = 0; i < k2_n_datas; i++) {
            gpuErrchk(hipMalloc((void**)&k2_d_data[i], k2_data_size + 0x1000));
            k2_d_data[i] = (char*)(((uintptr_t)k2_d_data[i] & ~(0x0FFF)) + 0x1000);
        }

        vector<vector<int>> k2_h_home(k2_n_datas, vector<int>(k2_n_chunks));
        vector<vector<size_t>> k2_h_xcd_chunks_size(k2_n_datas, vector<size_t>(XCD_NUM, 0));

        if (k2::home_identification(k2_d_data, k2_data_size, k2_n_chunks, k2_n_datas, k2_h_home, k2_h_xcd_chunks_size) == -1)
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

        for (int i = 0; i < k2_n_datas; i++) {
            gpuErrchk(hipMalloc((void**)&k2_d_chunks_per_hbm[i], sizeof(uint64_t) * k2_n_chunks ));
            size_t _offset = 0;
            for (int x = 0; x < XCD_NUM; x++) {
                size_t _n_chunks = k2_h_xcd_chunks_size[i][x];
                gpuErrchk(hipMemcpy(&k2_d_chunks_per_hbm[i][_offset], k2_xcd_chunks[i][x].data(), sizeof(uint64_t) * _n_chunks, hipMemcpyHostToDevice));
                k2_h_offsets[i][x] = _offset; 
                _offset += _n_chunks;
            }
        }

        /* count # per-xcd chunk */
        fill(k2_min_num_chunks_over_n_datas.begin(), k2_min_num_chunks_over_n_datas.end(), SIZE_MAX); // init to max
        for (int i = 0; i < k2_n_datas; i++) {
            for (int x = 0; x < XCD_NUM; x++) {
                // Determine the minimum chunk count for this XCD across all datasets.
                // This establishes a safe common upper bound, preventing out-of-bounds access on datasets with fewer chunks.
                k2_min_num_chunks_over_n_datas[x] = min(k2_min_num_chunks_over_n_datas[x], k2_h_xcd_chunks_size[i][x]);
            }   
        }

#if DEBUG_LEVEL >= 1
        for (int v = 0; v < HBM_NUM; v++) {
            size_t k2_h_xcd_chunks_size_sum = 0;
            for (int i = 0; i < k2_n_datas; i++) {
                k2_h_xcd_chunks_size_sum += k2_h_xcd_chunks_size[i][v];
            }
            size_t k2_h_xcd_chunks_size_sum_bytes = k2_h_xcd_chunks_size_sum * k2_chunk_size;
            string level = k2_h_xcd_chunks_size_sum_bytes > L2_SIZE ? 
                        (k2_h_xcd_chunks_size_sum_bytes > LLC_SIZE ? "hbm" : "llc") : "l2";
            cout << "K2 pinned data: hbm" << v << " "
                 << k2_h_xcd_chunks_size_sum_bytes / (1024 * 1024) << "MB at " << level 
                 << "\n" << flush;
        }
#endif

#if (FAST)
        k2_profile_iters = 10000;
#else
        k2_profile_iters = max(*min_element(k2_min_num_chunks_over_n_datas.begin(), k2_min_num_chunks_over_n_datas.end()), (size_t)10000);
#endif

#endif // DISABLE_K2_PLANS
    }

    // =============================================================================================
    // PPNT SETUP (Plan Generation)
    // =============================================================================================
    
    vector<ppnt::PingSpec> h_plan;
    vector<ppnt::PingOut>  h_out;
    ppnt::PingSpec *d_plan;
    ppnt::PingOut  *d_out;

#if !(DISABLE_K1_PLANS)
    // --- Add Latency Plans ---

#if !(PPNT_PLAN_SELECTED_ONLY)
    for (int x = 0; x < XCD_NUM; x++) {
        for (int v = 0; v < HBM_NUM; v++) {
            ppnt::PingSpec p;
            p.ping_id = (int)h_plan.size();
            p.kind    = ppnt::PingKind::Latency;
            p.src_xcd = x;
            p.dst_hbm = v;
            p.iters   = k1_profile_iters; 
            p.bpx     = 1;
            p.data    = k1_dbuf_start_ptrs_per_hbm[p.dst_hbm];
            p.dummy   = k1_dummy_buf; 
            h_plan.push_back(p);

            ppnt::PingOut o;
            gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
            h_out.push_back(o);
        }
    }
#else
    // Just a few latency plans for fast debugging
    // Specifically, src_xcd = 0 to dst_hbm = 0,2,4,6

    {
        ppnt::PingSpec p;
        p.ping_id = (int)h_plan.size();
        p.kind    = ppnt::PingKind::Latency;
        p.src_xcd = 0;
        p.dst_hbm = 0;
        p.iters   = k1_profile_iters; 
        p.bpx     = 1;
        p.data    = k1_dbuf_start_ptrs_per_hbm[p.dst_hbm];
        p.dummy   = k1_dummy_buf; 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);
    }

    {
        ppnt::PingSpec p;
        p.ping_id = (int)h_plan.size();
        p.kind    = ppnt::PingKind::Latency;
        p.src_xcd = 0;
        p.dst_hbm = 2;
        p.iters   = k1_profile_iters; 
        p.bpx     = 1;
        p.data    = k1_dbuf_start_ptrs_per_hbm[p.dst_hbm];
        p.dummy   = k1_dummy_buf; 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);
    }

    {
        ppnt::PingSpec p;
        p.ping_id = (int)h_plan.size();
        p.kind    = ppnt::PingKind::Latency;
        p.src_xcd = 0;
        p.dst_hbm = 4;
        p.iters   = k1_profile_iters; 
        p.bpx     = 1;
        p.data    = k1_dbuf_start_ptrs_per_hbm[p.dst_hbm];
        p.dummy   = k1_dummy_buf; 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);
    }

    {
        ppnt::PingSpec p;
        p.ping_id = (int)h_plan.size();
        p.kind    = ppnt::PingKind::Latency;
        p.src_xcd = 0;
        p.dst_hbm = 6;
        p.iters   = k1_profile_iters; 
        p.bpx     = 1;
        p.data    = k1_dbuf_start_ptrs_per_hbm[p.dst_hbm];
        p.dummy   = k1_dummy_buf; 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);
    }

#endif // !(PPNT_PLAN_SELECTED_ONLY)

#endif // !DISABLE_K1_PLANS 

#if !(DISABLE_K2_PLANS)
    // --- Add Bandwidth Plans ---

#if !(PPNT_PLAN_SELECTED_ONLY)
    for (int x = 0; x < XCD_NUM; x++) {
        for (int v = 0; v < HBM_NUM; v++) {
            for (int bpx : {1,2,4,8,16}) {
                ppnt::PingSpec p;
                p.ping_id         = (int)h_plan.size(); // auto increments
                p.kind            = ppnt::PingKind::Bandwidth;
                p.src_xcd         = x;
                p.dst_hbm         = v;
                p.iters           = k2_profile_iters; 
                p.bpx             = bpx;
                p.data_bytes      = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[p.dst_hbm]; // per data
                p.data0           = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][p.dst_hbm];
                p.data1           = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][p.dst_hbm];
                p.data2           = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][p.dst_hbm];
                p.data3           = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][p.dst_hbm];
                // Note (01/28/25) This will definitely lead to OOB if k2_bpx > 1. Must modify the current implementation of 
                // having `XCD_NUM` as a substitute for real gridDim.x of the k2 profiler kernel
                // TODO: fix!!
                gpuErrchk(hipMalloc(&p.sink, sizeof(float) * (TARGET_BLOCKDIM_X * XCD_NUM))); 
                h_plan.push_back(p);

                ppnt::PingOut o;
                gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
                h_out.push_back(o);
            }
        }
    }
#else
    // Just a few bandwidth plans for fast debugging
    // Specifically, src_xcd = 0 to dst_hbm = 0,2 and bpx = 2,4,8,16

    {
        ppnt::PingSpec p;
        p.ping_id         = (int)h_plan.size(); // auto increments
        p.kind            = ppnt::PingKind::Bandwidth;
        p.src_xcd         = 0;
        p.dst_hbm         = 0;
        p.iters           = k2_profile_iters; 
        p.bpx             = 2;
        p.data_bytes      = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[p.dst_hbm]; // per data
        p.data0           = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][p.dst_hbm];
        p.data1           = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][p.dst_hbm];
        p.data2           = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][p.dst_hbm];
        p.data3           = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][p.dst_hbm];
        // Note (01/28/25) This will definitely lead to OOB if k2_bpx > 1. Must modify the current implementation of 
        // having `XCD_NUM` as a substitute for real gridDim.x of the k2 profiler kernel
        // TODO: fix!!
        gpuErrchk(hipMalloc(&p.sink, sizeof(float) * (TARGET_BLOCKDIM_X * XCD_NUM))); 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);

    }

    {
        ppnt::PingSpec p;
        p.ping_id         = (int)h_plan.size(); // auto increments
        p.kind            = ppnt::PingKind::Bandwidth;
        p.src_xcd         = 0;
        p.dst_hbm         = 0;
        p.iters           = k2_profile_iters; 
        p.bpx             = 4;
        p.data_bytes      = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[p.dst_hbm]; // per data
        p.data0           = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][p.dst_hbm];
        p.data1           = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][p.dst_hbm];
        p.data2           = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][p.dst_hbm];
        p.data3           = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][p.dst_hbm];
        // Note (01/28/25) This will definitely lead to OOB if k2_bpx > 1. Must modify the current implementation of 
        // having `XCD_NUM` as a substitute for real gridDim.x of the k2 profiler kernel
        // TODO: fix!!
        gpuErrchk(hipMalloc(&p.sink, sizeof(float) * (TARGET_BLOCKDIM_X * XCD_NUM))); 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);

    }

    {
        ppnt::PingSpec p;
        p.ping_id         = (int)h_plan.size(); // auto increments
        p.kind            = ppnt::PingKind::Bandwidth;
        p.src_xcd         = 0;
        p.dst_hbm         = 0;
        p.iters           = k2_profile_iters; 
        p.bpx             = 8;
        p.data_bytes      = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[p.dst_hbm]; // per data
        p.data0           = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][p.dst_hbm];
        p.data1           = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][p.dst_hbm];
        p.data2           = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][p.dst_hbm];
        p.data3           = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][p.dst_hbm];
        // Note (01/28/25) This will definitely lead to OOB if k2_bpx > 1. Must modify the current implementation of 
        // having `XCD_NUM` as a substitute for real gridDim.x of the k2 profiler kernel
        // TODO: fix!!
        gpuErrchk(hipMalloc(&p.sink, sizeof(float) * (TARGET_BLOCKDIM_X * XCD_NUM))); 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);

    }

    {
        ppnt::PingSpec p;
        p.ping_id         = (int)h_plan.size(); // auto increments
        p.kind            = ppnt::PingKind::Bandwidth;
        p.src_xcd         = 0;
        p.dst_hbm         = 0;
        p.iters           = k2_profile_iters;
        p.bpx             = 16; 
        p.data_bytes      = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[p.dst_hbm]; // per data
        p.data0           = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][p.dst_hbm];
        p.data1           = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][p.dst_hbm];
        p.data2           = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][p.dst_hbm];
        p.data3           = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][p.dst_hbm];
        // Note (01/28/25) This will definitely lead to OOB if k2_bpx > 1. Must modify the current implementation of 
        // having `XCD_NUM` as a substitute for real gridDim.x of the k2 profiler kernel
        // TODO: fix!!
        gpuErrchk(hipMalloc(&p.sink, sizeof(float) * (TARGET_BLOCKDIM_X * XCD_NUM))); 
        h_plan.push_back(p);

        ppnt::PingOut o;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * p.iters * p.bpx));
        h_out.push_back(o);
    }

#endif // !(PPNT_PLAN_SELECTED_ONLY)

#endif // !DISABLE_K2_PLANS

    // Copy plan/out to device
    size_t n_plan = h_plan.size();
    if (n_plan > 0) {
        gpuErrchk(hipMalloc(&d_plan, sizeof(ppnt::PingSpec) * n_plan));
        gpuErrchk(hipMalloc(&d_out,  sizeof(ppnt::PingOut)  * n_plan));
        gpuErrchk(hipMemcpy(d_plan, h_plan.data(), sizeof(ppnt::PingSpec) * n_plan, hipMemcpyHostToDevice));
        gpuErrchk(hipMemcpy(d_out,  h_out.data(),  sizeof(ppnt::PingOut)  * n_plan, hipMemcpyHostToDevice));
    }

    // =============================================================================================
    // K2-ONLY PLAN SETUP FOR FEATURE 2 (co-run measurements using PingOut2 + fused_kernel2)
    // =============================================================================================

    vector<ppnt::PingSpec> h_k2plan;
    vector<ppnt::PingOut2> h_k2out2;
    for (size_t i = 0; i < n_plan; i++) {
        if (h_plan[i].kind == ppnt::PingKind::Bandwidth) {
            h_k2plan.push_back(h_plan[i]);
            ppnt::PingOut2 o2;
            gpuErrchk(hipMalloc(&o2.iterClk,       sizeof(uint64_t) * h_plan[i].iters * h_plan[i].bpx));
            gpuErrchk(hipMalloc(&o2.moe_start_clk, sizeof(uint64_t)));
            gpuErrchk(hipMalloc(&o2.moe_end_clk,   sizeof(uint64_t)));
            h_k2out2.push_back(o2);
        }
    }
    size_t n_k2plan = h_k2plan.size();
    ppnt::PingSpec* d_k2plan = nullptr;
    ppnt::PingOut2* d_k2out2 = nullptr;
    if (n_k2plan > 0) {
        gpuErrchk(hipMalloc(&d_k2plan, sizeof(ppnt::PingSpec) * n_k2plan));
        gpuErrchk(hipMalloc(&d_k2out2, sizeof(ppnt::PingOut2) * n_k2plan));
        gpuErrchk(hipMemcpy(d_k2plan, h_k2plan.data(), sizeof(ppnt::PingSpec) * n_k2plan, hipMemcpyHostToDevice));
        gpuErrchk(hipMemcpy(d_k2out2, h_k2out2.data(), sizeof(ppnt::PingOut2) * n_k2plan, hipMemcpyHostToDevice));
    }


    // =============================================================================================
    // MoE SETUP
    // =============================================================================================

    // 1. Dimensions
    // const int T = (argc > 1) ? atoi(argv[1]) : TARGET_BLOCKDIM_X * (CU_NUM-1) * XCD_NUM;
    const int T = (argc > 1) ? atoi(argv[1]) : 32768; // 32K tokens default
    const int d = (argc > 2) ? atoi(argv[2]) : 8192; 
    const int E = (argc > 3) ? atoi(argv[3]) : 8;    
    const int hidden = 4 * d;
#if IMBALANCED_DISTRIBUTION
    const int cap = ((T + 64 - 1) / 64) * 64;
#else
    const int cap = (T + E - 1) / E + 64; 
#endif

#if DEBUG_LEVEL >= 0
    printf("[MoE] Setup: T=%d d=%d hidden=%d E=%d cap=%d\n", T, d, hidden, E, cap);
#endif

    hipStream_t stream;
    gpuErrchk(hipStreamCreate(&stream));

    // 2. Cooperative Grid Configuration
    // Calculate once, use for all persistent kernels (Attn, FFN, Gather)
    int physical_grid_size;
    {
        const int num_sms = CU_NUM * XCD_NUM;
#if 0
        const int max_blocks_per_sm = 2;
#else
        // TODO: Each CU can host up to 2K threads, which is 2 blocks. However, this often leads to
        // profiling TB and target TBs being scheduled on the same CU, causing interference. Thus, 
        // we limit to 1 block per SM for more consistent profiling results. 
        // This should be revisited, potentially leverage CU mask to isolate profiling "CUs" for target.
        const int max_blocks_per_sm = 1;
#endif
        const int target_physical_blocks = max_blocks_per_sm * num_sms;
        
        // Round down to nearest multiple of XCD_NUM
        physical_grid_size = (target_physical_blocks / XCD_NUM) * XCD_NUM;
        if (physical_grid_size < XCD_NUM) physical_grid_size = XCD_NUM;
        
        int logical_blocks = physical_grid_size - XCD_NUM;

#if DEBUG_LEVEL >= 1
        cout << "[PPNT] Cooperative Grid: " << physical_grid_size 
             << " physical blocks (" << logical_blocks << " logical workers)\n" << flush;
#endif
    }

    // 3. Model Weights (Allocation & Initialization)
    float *d_Wqkv = nullptr, *d_Wo = nullptr, *d_W1 = nullptr, *d_W2 = nullptr;
    {
        // Allocation
        gpuErrchk(hipMalloc(&d_Wqkv, sizeof(float) * d * (3*d)));
        gpuErrchk(hipMalloc(&d_Wo,   sizeof(float) * (3*d) * d));
        gpuErrchk(hipMalloc(&d_W1,   sizeof(float) * (size_t)E * d * hidden));
        gpuErrchk(hipMalloc(&d_W2,   sizeof(float) * (size_t)E * hidden * d));

        // Initialization (Scoped to free host memory immediately)
        // Attention weights zeroed for simplicity
        gpuErrchk(hipMemsetAsync(d_Wqkv, 0, sizeof(float) * d * (3*d), stream));
        gpuErrchk(hipMemsetAsync(d_Wo,   0, sizeof(float) * (3*d) * d, stream));

        // FFN weights random
        float *h_W1 = (float*)malloc(sizeof(float) * (size_t)E * d * hidden);
        float *h_W2 = (float*)malloc(sizeof(float) * (size_t)E * hidden * d);
        fill_random(h_W1, (size_t)E * d * hidden);
        fill_random(h_W2, (size_t)E * hidden * d);
        gpuErrchk(hipMemcpyAsync(d_W1, h_W1, sizeof(float) * (size_t)E * d * hidden, hipMemcpyHostToDevice, stream));
        gpuErrchk(hipMemcpyAsync(d_W2, h_W2, sizeof(float) * (size_t)E * hidden * d, hipMemcpyHostToDevice, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        free(h_W1);
        free(h_W2);
    }

    // 4. Model Inputs (Allocation & Initialization)
    float *d_X = nullptr;
    int   *d_eid = nullptr;
    {
        gpuErrchk(hipMalloc(&d_X,   sizeof(float) * (size_t)T * d));
        gpuErrchk(hipMalloc(&d_eid, sizeof(int)   * (size_t)T));

        // vector<float> h_X(T * d);
        // vector<int> h_eid(T);
        float *h_X = (float*)malloc(sizeof(float) * (size_t)T * d);
        int   *h_eid = (int*)malloc(sizeof(int)   * (size_t)T);
        fill_random(h_X, (size_t)T * d);

#if IMBALANCED_DISTRIBUTION
        // Deterministic, exact counts per expert (90% to E0), for experimental reproducibility
        fill_expert_ids_fixed(h_eid, T, E);
#else
        fill_expert_ids(h_eid, T, E);
#endif

#if DEBUG_LEVEL >= 0
        vector<int> counts(E, 0);
        for (size_t t = 0; t < (size_t)T; t++) counts[h_eid[t]]++;
        cout << "[MoE] Token Dist: ";
        for (int e = 0; e < E; e++) cout << counts[e] << " ";
        cout << "\n" << flush;
#endif

        gpuErrchk(hipMemcpyAsync(d_X,   h_X,   sizeof(float) * (size_t)T * d,   hipMemcpyHostToDevice, stream));
        gpuErrchk(hipMemcpyAsync(d_eid, h_eid, sizeof(int)   * (size_t)T, hipMemcpyHostToDevice, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        free(h_X);
        free(h_eid);
    }

    // 5. Intermediate Buffers & Output
    // Attention buffers
    float *d_QKV = nullptr, *d_AttnOut = nullptr;
    gpuErrchk(hipMalloc(&d_QKV,     sizeof(float) * (size_t)T * (3*d)));
    gpuErrchk(hipMalloc(&d_AttnOut, sizeof(float) * (size_t)T * d));

    // MoE buffers
    float *d_Xexp = nullptr, *d_Tmp = nullptr, *d_Yexp = nullptr, *d_Y = nullptr;
    int   *d_pos = nullptr, *d_cnt = nullptr;
    
    gpuErrchk(hipMalloc(&d_pos,  sizeof(int)   * (size_t)T));
    gpuErrchk(hipMalloc(&d_cnt,  sizeof(int)   * (size_t)E));
    gpuErrchk(hipMalloc(&d_Xexp, sizeof(float) * (size_t)E * cap * d));
    gpuErrchk(hipMalloc(&d_Tmp,  sizeof(float) * (size_t)E * cap * hidden));
    gpuErrchk(hipMalloc(&d_Yexp, sizeof(float) * (size_t)E * cap * d));
    gpuErrchk(hipMalloc(&d_Y,    sizeof(float) * (size_t)T * d));

    // Reset counters
    gpuErrchk(hipMemsetAsync(d_cnt, 0, sizeof(int) * (size_t)E, stream));

    // Theoretical bytes accessed per kernel (read + write, floats unless noted)
    const size_t B_attn_qkv = sizeof(float) * ((size_t)T*d + (size_t)d*3*d + (size_t)T*3*d);
    const size_t B_attn_out = sizeof(float) * ((size_t)T*3*d + (size_t)3*d*d + (size_t)T*d);
    const size_t B_scatter  = sizeof(float)*(size_t)T*d*2 + sizeof(int)*(size_t)T*2;
    const size_t B_ffn1     = sizeof(float)*((size_t)T*d + (size_t)E*d*hidden + (size_t)T*hidden);
    const size_t B_relu     = sizeof(float)*(size_t)T*hidden*2;
    const size_t B_ffn2     = sizeof(float)*((size_t)T*hidden + (size_t)E*hidden*d + (size_t)T*d);
    const size_t B_gather   = sizeof(float)*(size_t)T*d*2 + sizeof(int)*(size_t)T;

    // =============================================================================================
    // PPNT EXECUTION WITH MoE KERNELS (integrated solo + co-run throughput)
    // =============================================================================================

    // 0. PPNT ONLY: For baseline profiling
    {
#if DEBUG_LEVEL >= 1
        cout << "\n\nPPNT ONLY BASELINE" << "\n" << flush;
#endif
        ppnt::TargetArgsT h_args{};
        ppnt::TargetArgsT* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(ppnt::TargetArgsT)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(ppnt::TargetArgsT), hipMemcpyHostToDevice, stream));
        
        ppnt::TargetFnT fn{};
        size_t _n_plan = n_plan;
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<ppnt::TargetFnT, ppnt::TargetArgsT>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // 1. ATTENTION LAYER: QKV Projection (X -> QKV)
    { 
#if DEBUG_LEVEL >= 1
        cout << "\n\nATTENTION1 (QKV)" << "\n" << flush;
#endif
        AttnQKVArgs h_args = {d_X, d_Wqkv, d_QKV, T, d};
        AttnQKVArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(AttnQKVArgs)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(AttnQKVArgs), hipMemcpyHostToDevice, stream));

        AttnQKVTargetFn fn{};
        // Solo throughput (target only)
        {
            hipEvent_t ev_s, ev_e;
            gpuErrchk(hipEventCreate(&ev_s));
            gpuErrchk(hipEventCreate(&ev_e));
            size_t n_plan_solo = 0;
            void* kargs_solo[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan_solo, (void*)&d_out };
            gpuErrchk(hipEventRecord(ev_s, stream));
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)ppnt::fused_kernel<AttnQKVTargetFn, AttnQKVArgs>,
                dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_solo, 0, stream));
            gpuErrchk(hipEventRecord(ev_e, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            float ms = 0;
            gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
            ppnt::log_solo_throughput("AttnQKV", ms, B_attn_qkv);
            gpuErrchk(hipEventDestroy(ev_s));
            gpuErrchk(hipEventDestroy(ev_e));
        }

        // Co-run throughput (target + k2 ping)
        if (n_k2plan > 0) {
            for (size_t pi = 0; pi < n_k2plan; ++pi) {
                const ppnt::PingSpec& p = h_k2plan[pi];
                hipEvent_t ev_s, ev_e;
                gpuErrchk(hipEventCreate(&ev_s));
                gpuErrchk(hipEventCreate(&ev_e));
                ppnt::init_moe_clks(h_k2out2, stream);
                ppnt::PingSpec* d_k2plan_one = d_k2plan + pi;
                ppnt::PingOut2* d_k2out2_one = d_k2out2 + pi;
                size_t n_k2plan_one = 1;
                void* kargs_corun[] = { (void*)&fn, (void*)&d_args, (void*)&d_k2plan_one, (void*)&n_k2plan_one, (void*)&d_k2out2_one };
                gpuErrchk(hipEventRecord(ev_s, stream));
                gpuErrchk(hipLaunchCooperativeKernel(
                    (void*)ppnt::fused_kernel2<AttnQKVTargetFn, AttnQKVArgs>,
                    dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_corun, 0, stream));
                gpuErrchk(hipEventRecord(ev_e, stream));
                gpuErrchk(hipStreamSynchronize(stream));
                float ms = 0;
                gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
                ppnt::log_corun_throughput("AttnQKV", ms, B_attn_qkv, &p);
                gpuErrchk(hipEventDestroy(ev_s));
                gpuErrchk(hipEventDestroy(ev_e));
            }
        }

        size_t _n_plan = (PPNT_PROFILE_ATTENTION1) ? n_plan : 0; // set to 0 to bypass ppnt execution
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<AttnQKVTargetFn, AttnQKVArgs>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // 2. ATTENTION LAYER: Output Projection (QKV -> AttnOut)
    { 
#if DEBUG_LEVEL >= 1
        cout << "\n\nATTENTION2 (Out)" << "\n" << flush;
#endif
        AttnOutArgs h_args = {d_QKV, d_Wo, d_AttnOut, T, d};
        AttnOutArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(AttnOutArgs)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(AttnOutArgs), hipMemcpyHostToDevice, stream));

        AttnOutTargetFn fn{};
        // Solo throughput (target only)
        {
            hipEvent_t ev_s, ev_e;
            gpuErrchk(hipEventCreate(&ev_s));
            gpuErrchk(hipEventCreate(&ev_e));
            size_t n_plan_solo = 0;
            void* kargs_solo[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan_solo, (void*)&d_out };
            gpuErrchk(hipEventRecord(ev_s, stream));
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)ppnt::fused_kernel<AttnOutTargetFn, AttnOutArgs>,
                dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_solo, 0, stream));
            gpuErrchk(hipEventRecord(ev_e, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            float ms = 0;
            gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
            ppnt::log_solo_throughput("AttnOut", ms, B_attn_out);
            gpuErrchk(hipEventDestroy(ev_s));
            gpuErrchk(hipEventDestroy(ev_e));
        }

        // Co-run throughput (target + k2 ping)
        if (n_k2plan > 0) {
            for (size_t pi = 0; pi < n_k2plan; ++pi) {
                const ppnt::PingSpec& p = h_k2plan[pi];
                hipEvent_t ev_s, ev_e;
                gpuErrchk(hipEventCreate(&ev_s));
                gpuErrchk(hipEventCreate(&ev_e));
                ppnt::init_moe_clks(h_k2out2, stream);
                ppnt::PingSpec* d_k2plan_one = d_k2plan + pi;
                ppnt::PingOut2* d_k2out2_one = d_k2out2 + pi;
                size_t n_k2plan_one = 1;
                void* kargs_corun[] = { (void*)&fn, (void*)&d_args, (void*)&d_k2plan_one, (void*)&n_k2plan_one, (void*)&d_k2out2_one };
                gpuErrchk(hipEventRecord(ev_s, stream));
                gpuErrchk(hipLaunchCooperativeKernel(
                    (void*)ppnt::fused_kernel2<AttnOutTargetFn, AttnOutArgs>,
                    dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_corun, 0, stream));
                gpuErrchk(hipEventRecord(ev_e, stream));
                gpuErrchk(hipStreamSynchronize(stream));
                float ms = 0;
                gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
                ppnt::log_corun_throughput("AttnOut", ms, B_attn_out, &p);
                gpuErrchk(hipEventDestroy(ev_s));
                gpuErrchk(hipEventDestroy(ev_e));
            }
        }

        size_t _n_plan = (PPNT_PROFILE_ATTENTION2) ? n_plan : 0; // set to 0 to bypass ppnt execution
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<AttnOutTargetFn, AttnOutArgs>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // 3. MOE: Scatter (AttnOut -> Xexp)
    {
#if DEBUG_LEVEL >= 1
        cout << "\n\nSCATTER" << "\n" << flush;
#endif
        // Explicitly defining input for clarity
        float* d_moe_input = d_AttnOut;

        ScatterArgs h_args = {d_moe_input, d_eid, d_Xexp, d_pos, d_cnt, T, d, E, cap};
        ScatterArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(ScatterArgs)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(ScatterArgs), hipMemcpyHostToDevice, stream));

        ScatterTargetFn fn{};
        size_t _n_plan = (PPNT_PROFILE_SCATTER) ? n_plan : 0; // set to 0 to bypass ppnt execution
        void* kernelArgs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        // Note: Scatter uses a calculated grid based on T, plus 1 profile block per XCD
        dim3 target_blockD(TARGET_BLOCKDIM_X);
        dim3 profile_blockD(1024);
        dim3 blockD(max(target_blockD.x, profile_blockD.x));
        dim3 target_gridD((T + target_blockD.x - 1) / target_blockD.x); 
        dim3 profile_gridD(XCD_NUM); 
        dim3 gridD(target_gridD.x + profile_gridD.x);

        // Solo throughput (target only)
        {
            hipEvent_t ev_s, ev_e;
            gpuErrchk(hipEventCreate(&ev_s));
            gpuErrchk(hipEventCreate(&ev_e));
            gpuErrchk(hipMemsetAsync(d_cnt, 0, sizeof(int) * (size_t)E, stream));
            size_t n_plan_solo = 0;
            void* kargs_solo[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan_solo, (void*)&d_out };
            gpuErrchk(hipEventRecord(ev_s, stream));
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)ppnt::fused_kernel<ScatterTargetFn, ScatterArgs>,
                dim3(gridD), dim3(blockD), kargs_solo, 0, stream));
            gpuErrchk(hipEventRecord(ev_e, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            float ms = 0;
            gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
            ppnt::log_solo_throughput("Scatter", ms, B_scatter);
            gpuErrchk(hipEventDestroy(ev_s));
            gpuErrchk(hipEventDestroy(ev_e));
        }

        // Co-run throughput (target + k2 ping)
        if (n_k2plan > 0) {
            for (size_t pi = 0; pi < n_k2plan; ++pi) {
                const ppnt::PingSpec& p = h_k2plan[pi];
                hipEvent_t ev_s, ev_e;
                gpuErrchk(hipEventCreate(&ev_s));
                gpuErrchk(hipEventCreate(&ev_e));
                gpuErrchk(hipMemsetAsync(d_cnt, 0, sizeof(int) * (size_t)E, stream));
                ppnt::init_moe_clks(h_k2out2, stream);
                ppnt::PingSpec* d_k2plan_one = d_k2plan + pi;
                ppnt::PingOut2* d_k2out2_one = d_k2out2 + pi;
                size_t n_k2plan_one = 1;
                void* kargs_corun[] = { (void*)&fn, (void*)&d_args, (void*)&d_k2plan_one, (void*)&n_k2plan_one, (void*)&d_k2out2_one };
                gpuErrchk(hipEventRecord(ev_s, stream));
                gpuErrchk(hipLaunchCooperativeKernel(
                    (void*)ppnt::fused_kernel2<ScatterTargetFn, ScatterArgs>,
                    dim3(gridD), dim3(blockD), kargs_corun, 0, stream));
                gpuErrchk(hipEventRecord(ev_e, stream));
                gpuErrchk(hipStreamSynchronize(stream));
                float ms = 0;
                gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
                ppnt::log_corun_throughput("Scatter", ms, B_scatter, &p);
                gpuErrchk(hipEventDestroy(ev_s));
                gpuErrchk(hipEventDestroy(ev_e));
            }
        }

        gpuErrchk(hipMemsetAsync(d_cnt, 0, sizeof(int) * (size_t)E, stream));
        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<ScatterTargetFn, ScatterArgs>,
            dim3(gridD), dim3(blockD), kernelArgs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // 4. MOE: FFN1 (Xexp -> Tmp)
    { 
#if DEBUG_LEVEL >= 1
        cout << "\n\nFFN1" << "\n" << flush;
#endif
        Gemm1Args h_args = {d_Xexp, d_W1, d_Tmp, d_cnt, E, cap, d, hidden};
        Gemm1Args* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(Gemm1Args)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(Gemm1Args), hipMemcpyHostToDevice, stream));

        Gemm1TargetFn fn{};
        // Solo throughput (target only)
        {
            hipEvent_t ev_s, ev_e;
            gpuErrchk(hipEventCreate(&ev_s));
            gpuErrchk(hipEventCreate(&ev_e));
            size_t n_plan_solo = 0;
            void* kargs_solo[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan_solo, (void*)&d_out };
            gpuErrchk(hipEventRecord(ev_s, stream));
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)ppnt::fused_kernel<Gemm1TargetFn, Gemm1Args>,
                dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_solo, 0, stream));
            gpuErrchk(hipEventRecord(ev_e, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            float ms = 0;
            gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
            ppnt::log_solo_throughput("FFN1", ms, B_ffn1);
            gpuErrchk(hipEventDestroy(ev_s));
            gpuErrchk(hipEventDestroy(ev_e));
        }

        // Co-run throughput (target + k2 ping)
        if (n_k2plan > 0) {
            for (size_t pi = 0; pi < n_k2plan; ++pi) {
                const ppnt::PingSpec& p = h_k2plan[pi];
                hipEvent_t ev_s, ev_e;
                gpuErrchk(hipEventCreate(&ev_s));
                gpuErrchk(hipEventCreate(&ev_e));
                ppnt::init_moe_clks(h_k2out2, stream);
                ppnt::PingSpec* d_k2plan_one = d_k2plan + pi;
                ppnt::PingOut2* d_k2out2_one = d_k2out2 + pi;
                size_t n_k2plan_one = 1;
                void* kargs_corun[] = { (void*)&fn, (void*)&d_args, (void*)&d_k2plan_one, (void*)&n_k2plan_one, (void*)&d_k2out2_one };
                gpuErrchk(hipEventRecord(ev_s, stream));
                gpuErrchk(hipLaunchCooperativeKernel(
                    (void*)ppnt::fused_kernel2<Gemm1TargetFn, Gemm1Args>,
                    dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_corun, 0, stream));
                gpuErrchk(hipEventRecord(ev_e, stream));
                gpuErrchk(hipStreamSynchronize(stream));
                float ms = 0;
                gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
                ppnt::log_corun_throughput("FFN1", ms, B_ffn1, &p);
                gpuErrchk(hipEventDestroy(ev_s));
                gpuErrchk(hipEventDestroy(ev_e));
            }
        }

        size_t _n_plan = (PPNT_PROFILE_FFN1) ? n_plan : 0; // set to 0 to bypass ppnt execution
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<Gemm1TargetFn, Gemm1Args>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // 5. MOE: ReLU (Tmp -> Tmp)
    { 
#if DEBUG_LEVEL >= 1
        cout << "\n\nRELU" << "\n" << flush;
#endif
        ReluArgs h_args = {d_Tmp, d_cnt, E, cap, hidden};
        ReluArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(ReluArgs)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(ReluArgs), hipMemcpyHostToDevice, stream));

        ReluTargetFn fn{};
        // Solo throughput (target only)
        {
            hipEvent_t ev_s, ev_e;
            gpuErrchk(hipEventCreate(&ev_s));
            gpuErrchk(hipEventCreate(&ev_e));
            size_t n_plan_solo = 0;
            void* kargs_solo[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan_solo, (void*)&d_out };
            gpuErrchk(hipEventRecord(ev_s, stream));
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)ppnt::fused_kernel<ReluTargetFn, ReluArgs>,
                dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_solo, 0, stream));
            gpuErrchk(hipEventRecord(ev_e, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            float ms = 0;
            gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
            ppnt::log_solo_throughput("ReLU", ms, B_relu);
            gpuErrchk(hipEventDestroy(ev_s));
            gpuErrchk(hipEventDestroy(ev_e));
        }

        // Co-run throughput (target + k2 ping)
        if (n_k2plan > 0) {
            for (size_t pi = 0; pi < n_k2plan; ++pi) {
                const ppnt::PingSpec& p = h_k2plan[pi];
                hipEvent_t ev_s, ev_e;
                gpuErrchk(hipEventCreate(&ev_s));
                gpuErrchk(hipEventCreate(&ev_e));
                ppnt::init_moe_clks(h_k2out2, stream);
                ppnt::PingSpec* d_k2plan_one = d_k2plan + pi;
                ppnt::PingOut2* d_k2out2_one = d_k2out2 + pi;
                size_t n_k2plan_one = 1;
                void* kargs_corun[] = { (void*)&fn, (void*)&d_args, (void*)&d_k2plan_one, (void*)&n_k2plan_one, (void*)&d_k2out2_one };
                gpuErrchk(hipEventRecord(ev_s, stream));
                gpuErrchk(hipLaunchCooperativeKernel(
                    (void*)ppnt::fused_kernel2<ReluTargetFn, ReluArgs>,
                    dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_corun, 0, stream));
                gpuErrchk(hipEventRecord(ev_e, stream));
                gpuErrchk(hipStreamSynchronize(stream));
                float ms = 0;
                gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
                ppnt::log_corun_throughput("ReLU", ms, B_relu, &p);
                gpuErrchk(hipEventDestroy(ev_s));
                gpuErrchk(hipEventDestroy(ev_e));
            }
        }

        size_t _n_plan = (PPNT_PROFILE_RELU) ? n_plan : 0; // set to 0 to bypass ppnt execution
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<ReluTargetFn, ReluArgs>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // 6. MOE: FFN2 (Tmp -> Yexp)
    { 
#if DEBUG_LEVEL >= 1
        cout << "\n\nFFN2" << "\n" << flush;
#endif
        Gemm2Args h_args = {d_Tmp, d_W2, d_Yexp, d_cnt, E, cap, d, hidden};
        Gemm2Args* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(Gemm2Args)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(Gemm2Args), hipMemcpyHostToDevice, stream));

        Gemm2TargetFn fn{};
        // Solo throughput (target only)
        {
            hipEvent_t ev_s, ev_e;
            gpuErrchk(hipEventCreate(&ev_s));
            gpuErrchk(hipEventCreate(&ev_e));
            size_t n_plan_solo = 0;
            void* kargs_solo[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan_solo, (void*)&d_out };
            gpuErrchk(hipEventRecord(ev_s, stream));
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)ppnt::fused_kernel<Gemm2TargetFn, Gemm2Args>,
                dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_solo, 0, stream));
            gpuErrchk(hipEventRecord(ev_e, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            float ms = 0;
            gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
            ppnt::log_solo_throughput("FFN2", ms, B_ffn2);
            gpuErrchk(hipEventDestroy(ev_s));
            gpuErrchk(hipEventDestroy(ev_e));
        }

        // Co-run throughput (target + k2 ping)
        if (n_k2plan > 0) {
            for (size_t pi = 0; pi < n_k2plan; ++pi) {
                const ppnt::PingSpec& p = h_k2plan[pi];
                hipEvent_t ev_s, ev_e;
                gpuErrchk(hipEventCreate(&ev_s));
                gpuErrchk(hipEventCreate(&ev_e));
                ppnt::init_moe_clks(h_k2out2, stream);
                ppnt::PingSpec* d_k2plan_one = d_k2plan + pi;
                ppnt::PingOut2* d_k2out2_one = d_k2out2 + pi;
                size_t n_k2plan_one = 1;
                void* kargs_corun[] = { (void*)&fn, (void*)&d_args, (void*)&d_k2plan_one, (void*)&n_k2plan_one, (void*)&d_k2out2_one };
                gpuErrchk(hipEventRecord(ev_s, stream));
                gpuErrchk(hipLaunchCooperativeKernel(
                    (void*)ppnt::fused_kernel2<Gemm2TargetFn, Gemm2Args>,
                    dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_corun, 0, stream));
                gpuErrchk(hipEventRecord(ev_e, stream));
                gpuErrchk(hipStreamSynchronize(stream));
                float ms = 0;
                gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
                ppnt::log_corun_throughput("FFN2", ms, B_ffn2, &p);
                gpuErrchk(hipEventDestroy(ev_s));
                gpuErrchk(hipEventDestroy(ev_e));
            }
        }

        size_t _n_plan = (PPNT_PROFILE_FFN2) ? n_plan : 0; // set to 0 to bypass ppnt execution
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<Gemm2TargetFn, Gemm2Args>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // 7. MOE: Gather (Yexp -> Y)
    { 
#if DEBUG_LEVEL >= 1
        cout << "\n\nGATHER" << "\n" << flush;
#endif
        GatherArgs h_args = {d_Yexp, d_pos, d_Y, T, d, cap};
        GatherArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(GatherArgs)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(GatherArgs), hipMemcpyHostToDevice, stream));

        GatherTargetFn fn{};
        // Solo throughput (target only)
        {
            hipEvent_t ev_s, ev_e;
            gpuErrchk(hipEventCreate(&ev_s));
            gpuErrchk(hipEventCreate(&ev_e));
            size_t n_plan_solo = 0;
            void* kargs_solo[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan_solo, (void*)&d_out };
            gpuErrchk(hipEventRecord(ev_s, stream));
            gpuErrchk(hipLaunchCooperativeKernel(
                (void*)ppnt::fused_kernel<GatherTargetFn, GatherArgs>,
                dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_solo, 0, stream));
            gpuErrchk(hipEventRecord(ev_e, stream));
            gpuErrchk(hipStreamSynchronize(stream));
            float ms = 0;
            gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
            ppnt::log_solo_throughput("Gather", ms, B_gather);
            gpuErrchk(hipEventDestroy(ev_s));
            gpuErrchk(hipEventDestroy(ev_e));
        }

        // Co-run throughput (target + k2 ping)
        if (n_k2plan > 0) {
            for (size_t pi = 0; pi < n_k2plan; ++pi) {
                const ppnt::PingSpec& p = h_k2plan[pi];
                hipEvent_t ev_s, ev_e;
                gpuErrchk(hipEventCreate(&ev_s));
                gpuErrchk(hipEventCreate(&ev_e));
                ppnt::init_moe_clks(h_k2out2, stream);
                ppnt::PingSpec* d_k2plan_one = d_k2plan + pi;
                ppnt::PingOut2* d_k2out2_one = d_k2out2 + pi;
                size_t n_k2plan_one = 1;
                void* kargs_corun[] = { (void*)&fn, (void*)&d_args, (void*)&d_k2plan_one, (void*)&n_k2plan_one, (void*)&d_k2out2_one };
                gpuErrchk(hipEventRecord(ev_s, stream));
                gpuErrchk(hipLaunchCooperativeKernel(
                    (void*)ppnt::fused_kernel2<GatherTargetFn, GatherArgs>,
                    dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs_corun, 0, stream));
                gpuErrchk(hipEventRecord(ev_e, stream));
                gpuErrchk(hipStreamSynchronize(stream));
                float ms = 0;
                gpuErrchk(hipEventElapsedTime(&ms, ev_s, ev_e));
                ppnt::log_corun_throughput("Gather", ms, B_gather, &p);
                gpuErrchk(hipEventDestroy(ev_s));
                gpuErrchk(hipEventDestroy(ev_e));
            }
        }

        size_t _n_plan = (PPNT_PROFILE_GATHER) ? n_plan : 0; // set to 0 to bypass ppnt execution
        void* kargs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&_n_plan, (void*)&d_out };

        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<GatherTargetFn, GatherArgs>,
            dim3(physical_grid_size), dim3(TARGET_BLOCKDIM_X), kargs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));
        ppnt::parse_pingouts(d_plan, d_out, _n_plan, TARGET_BLOCKDIM_X, clock);
    }

    // Cleanup resources here if needed (omitted for brevity)
    return 0;
}
