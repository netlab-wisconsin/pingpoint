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

#include "main.h"
#include "ppnt.h"
#include "moe.h"
#include "k1.h"
#include "k2.h"

#define DEBUG_LEVEL 2 // 0: PPNT, 1: Memalloc, 2+: Misc
#define FAST 1 // set for faster debugging, set 0 for actual measurement

using namespace std;

#define TARGET_BLOCKDIM_X (1024)

int main(int argc, char **argv) {

    unsigned int clock = getGPUClock();
    
    // =============================================================================================
    // K1 SETUP
    // =============================================================================================
    
    // Starting pointers to pchase in each HBM
    // Original varname was dbuf_start_vector
    vector<k1::dtype*> k1_dbuf_start_ptrs_per_hbm(HBM_NUM, nullptr);
    size_t k1_profile_iters = -1;
    { /* k1 setup scope */
        k1::dtype *dbuf_base = nullptr;

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
        k1_profile_iters = 100000;
#elif 1
        const size_t LEN = (1 << 22); // 512MB (per HBM)
        const size_t multiplicative_factor = XCD_NUM * 1;
        k1_profile_iters = max(LEN, 100000);
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
                hbm_dtypes,
                hbm_cls) == -1)
            return -1;

#if DEBUG_LEVEL >= 1
        for (int v = 0; v < HBM_NUM; v++) {
            string level = hbm_dtypes[v].size() * sizeof(k1::dtype) > L2_SIZE ? 
                        (hbm_dtypes[v].size() * sizeof(k1::dtype) > LLC_SIZE ? "hbm" : "llc") 
                        : "l2";
            assert (level != "l2"); // K1 data should be at least in LLC
            cout << "K1 pinned data: " << "hbm" << v << " "
                << hbm_dtypes[v].size() * sizeof(k1::dtype) / (1024 * 1024) << "MB" << " "
                << "at " << level << " "
                << "\n" << flush;
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
            for (size_t i = 0; i < LEN; i++) {
                seq[i] = hbm_cls[v][i];
            }
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

            // Start pointer: lane 0 of the first cache line in seq[]
            size_t start_elem = ((size_t)seq[0] * (size_t)cl_size + 0) * (size_t)skip_factor;
            k1::dtype *dbuf_start = dbuf_base + start_elem;
            k1_dbuf_start_ptrs_per_hbm[v] = dbuf_start;
        }

        // Copy the full pointer table into device allocation.
        gpuErrchk(hipMemcpy(dbuf_base, buf.data(), n_dtype_dbuf * sizeof(k1::dtype), hipMemcpyHostToDevice));
        gpuErrchk(hipDeviceSynchronize());
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
    { /* k2 setup scope */

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

        for (int i = 0; i < k2_n_datas; i++) {
            gpuErrchk(hipMalloc((void**)&k2_d_chunks_per_hbm[i], sizeof(uint64_t) * k2_n_chunks ));
            size_t _offset = 0;
            for (int x = 0; x < XCD_NUM; x++) {
                size_t _n_chunks = k2_h_xcd_chunks_size[i][x];
                gpuErrchk(hipMemcpy(&k2_d_chunks_per_hbm[i][_offset], k2_xcd_chunks[i][x].data(), sizeof(uint64_t) * _n_chunks, hipMemcpyHostToDevice));
                k2_h_offsets[i][x] = _offset; // Store in host vector
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

        // ensure minimal 8MB = 2 * l2_size per-xcd chunks in order to thrash l2
        // conservative. since 2 xcds on same iod actually share the home data, another approach can be
        // ensure minimal 8MB per-iod chunks 
        const int min_n_chunks_per_xcd = ((4*2 * 1024 * 1024) / (k2_chunk_size)); // minimal #chunks >= 8MB
        for (int x = 0; x < XCD_NUM; x++) {
#if DEBUG_LEVEL >= 2
            printf("xcd %d: min n_chunks %zu\n", x, k2_min_num_chunks_over_n_datas[x]);
#endif
            assert (k2_min_num_chunks_over_n_datas[x] * k2_n_datas >= min_n_chunks_per_xcd); // k2_min_num_chunks_over_n_datas[x] is minimal #chunks per xcd among all datas. 
        }

#if DEBUG_LEVEL >= 1
        for (int x = 0; x < XCD_NUM; x++) {
            size_t _size = (k2_h_xcd_chunks_size[0][x] + k2_h_xcd_chunks_size[1][x] + \
                            k2_h_xcd_chunks_size[2][x] + k2_h_xcd_chunks_size[3][x]) * k2_chunk_size;

            string level = _size > L2_SIZE ? 
                        (_size > LLC_SIZE ? "hbm" : "llc") 
                        : "l2";
            assert (level != "l2"); // K2 data should be at least in LLC
            cout << "K2 pinned data: " << "hbm" << x << " "
                << _size / (1024 * 1024) << "MB" << " "
                << "at " << level << " "
                << "\n" << flush;
        }
#endif
#if (FAST)
        k2_profile_iters = 100000;
#else
        k2_profile_iters = max(
            *min_element(k2_min_num_chunks_over_n_datas.begin(), k2_min_num_chunks_over_n_datas.end()),
            (size_t)100000);
#endif
    }

    // =============================================================================================
    // PPNT SETUP
    // =============================================================================================
    
    vector<ppnt::PingSpec> h_plan;
    vector<ppnt::PingOut>  h_out;
    ppnt::PingSpec *d_plan;
    ppnt::PingOut  *d_out;

    {
        // --- Add Latency Plan ---
        ppnt::PingSpec p;
        p.ping_id         = (int)h_plan.size(); // auto increments
        p.kind            = ppnt::PingKind::Latency;
        p.src_xcd         = 0;
        p.dst_hbm         = 0;
        p.iters           = k1_profile_iters; 
        // p.data_bytes      = --- IGNORE ---
        p.data0           = (uint64_t*)k1_dbuf_start_ptrs_per_hbm[p.dst_hbm]; // casting k1::dtype* to uint64_t* to match ppnt spec
        h_plan.push_back(p);

        // -- Add Latency Out ---
        ppnt::PingOut o;
        o.ping_id = p.ping_id;
        o.kind    = p.kind;
        o.src_xcd = p.src_xcd;
        o.dst_hbm = p.dst_hbm;
        o.iters   = p.iters;
        gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * o.iters ));
        h_out.push_back(o);
    }

//     {
//         // --- Add Bandwidth Plan ---
//         ppnt::PingSpec p;
//         p.ping_id         = (int)h_plan.size(); // auto increments
//         p.kind            = ppnt::PingKind::Bandwidth;
//         p.src_xcd         = 0;
//         p.dst_hbm         = 0;
//         p.iters           = k2_profile_iters; 
//         p.data_bytes      = CHUNK_SIZE * k2_min_num_chunks_over_n_datas[p.dst_hbm]; // per data
//         p.data0           = k2_d_chunks_per_hbm[0] + k2_h_offsets[0][p.dst_hbm];
//         p.data1           = k2_d_chunks_per_hbm[1] + k2_h_offsets[1][p.dst_hbm];
//         p.data2           = k2_d_chunks_per_hbm[2] + k2_h_offsets[2][p.dst_hbm];
//         p.data3           = k2_d_chunks_per_hbm[3] + k2_h_offsets[3][p.dst_hbm];
//         // Note (01/28/25) This will definitely lead to OOB if k2_bpx > 1. Must modify the current implementation of 
//         // having `XCD_NUM` as a substitute for real gridDim.x of the k2 profiler kernel
//         // TODO: fix!!
//         gpuErrchk(hipMalloc(&p.sink, sizeof(float) * (TARGET_BLOCKDIM_X * XCD_NUM))); 
//         h_plan.push_back(p);

//         // -- Add Bandwidth Out ---
//         ppnt::PingOut o;
//         o.ping_id = p.ping_id;
//         o.kind    = p.kind;
//         o.src_xcd = p.src_xcd;
//         o.dst_hbm = p.dst_hbm;
//         o.iters   = p.iters;
// #if 1
//         gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * o.iters ));
// #else
//         const int k2_bpx = 1; // TODO: set to a proper value that saturates bw
//         gpuErrchk(hipMalloc(&o.iterClk, sizeof(uint64_t) * o.iters * k2_bpx)); // each k2_bpx writes one clk value per-iteration
// #endif
//         h_out.push_back(o);
//     }

    // Copy plan/out to device
    size_t n_plan = h_plan.size();
    if (n_plan > 0) {
        gpuErrchk(hipMalloc(&d_plan, sizeof(ppnt::PingSpec) * n_plan));
        gpuErrchk(hipMalloc(&d_out,  sizeof(ppnt::PingOut)  * n_plan));
        gpuErrchk(hipMemcpy(d_plan, h_plan.data(), sizeof(ppnt::PingSpec) * n_plan, hipMemcpyHostToDevice));
        gpuErrchk(hipMemcpy(d_out,  h_out.data(),  sizeof(ppnt::PingOut)  * n_plan, hipMemcpyHostToDevice));
    }

    // =============================================================================================
    // MoE SETUP
    // =============================================================================================

    // num tokens set intentionally for full occupancy (excluding the profiling TBs)
    const int T = (argc > 1) ? atoi(argv[1]) : TARGET_BLOCKDIM_X * (CU_NUM-1) * XCD_NUM;
    const int d = (argc > 2) ? atoi(argv[2]) : 2048; // model dim
    const int E = (argc > 3) ? atoi(argv[3]) : 8;    // experts
    const int hidden = 4 * d;

#if IMBALANCED_DISTRIBUTION
    const int cap = (T + 64 - 1) / 64; // an expert can take all tokens in the worst case
#else
    const int cap = (T + E - 1) / E + 64; // add headroom to avoid overflow
#endif

#if DEBUG_LEVEL >= 1
    printf("T=%d d=%d hidden=%d E=%d cap=%d\n", T, d, hidden, E, cap);
#endif

    hipStream_t stream;
    gpuErrchk(hipStreamCreate(&stream));

    // Host buffers
    vector<float> h_X(T * d);
    vector<int> h_eid(T);
    fill_random(h_X);
    fill_expert_ids(h_eid, E);
#if DEBUG_LEVEL >= 2
    // print distribution
    vector<int> counts(E, 0);
    for (auto x : h_eid)
        counts[x]++;
    cout << "Expert token distribution: ";
    for (int e = 0; e < E; e++) cout << counts[e] << " ";
    cout << "\n" << flush;
#endif

    // Device buffers
    float *d_X = nullptr, *d_Xexp = nullptr, *d_Y = nullptr;
    int *d_eid = nullptr, *d_pos = nullptr, *d_cnt = nullptr;
    gpuErrchk(hipMalloc(&d_X, sizeof(float) * h_X.size()));
    gpuErrchk(hipMalloc(&d_eid, sizeof(int) * h_eid.size()));
    gpuErrchk(hipMalloc(&d_Xexp, sizeof(float) * (size_t)E * cap * d));
    gpuErrchk(hipMalloc(&d_pos, sizeof(int) * (size_t)T));
    gpuErrchk(hipMalloc(&d_cnt, sizeof(int) * (size_t)E));
    gpuErrchk(hipMalloc(&d_Y, sizeof(float) * (size_t)T * d));

    gpuErrchk(hipMemcpyAsync(d_X, h_X.data(), sizeof(float) * h_X.size(), hipMemcpyHostToDevice, stream));
    gpuErrchk(hipMemcpyAsync(d_eid, h_eid.data(), sizeof(int) * h_eid.size(), hipMemcpyHostToDevice, stream));
    gpuErrchk(hipMemsetAsync(d_cnt, 0, sizeof(int) * (size_t)E, stream));

    { 
        // =========================================================================================
        // PPNT
        // =========================================================================================

        // Dispatch
        // X[T,d] -> Xexp[E,cap,d]

        // args setup
        DispatchArgs h_args = {d_X, d_eid, d_Xexp, d_pos, d_cnt, T, d, E, cap};
        DispatchArgs* d_args = nullptr;
        gpuErrchk(hipMalloc(&d_args, sizeof(DispatchArgs)));
        gpuErrchk(hipMemcpyAsync(d_args, &h_args, sizeof(DispatchArgs), hipMemcpyHostToDevice, stream));

        DispatchTargetFn fn{};
        void* kernelArgs[] = { (void*)&fn, (void*)&d_args, (void*)&d_plan, (void*)&n_plan, (void*)&d_out };

        // Take max of blockDims of target and profile
        dim3 target_blockD(TARGET_BLOCKDIM_X);
        dim3 profile_blockD(1024);
        dim3 blockD(max(target_blockD.x, profile_blockD.x));

        // Add gridDims of target and profile
        dim3 target_gridD((T + target_blockD.x - 1) / target_blockD.x); // #tokens/tpb
        dim3 profile_gridD(XCD_NUM); // 1 profile TB per XCD
        dim3 gridD(target_gridD.x + profile_gridD.x);

#if DEBUG_LEVEL >= 0
        cout << "[PPNT] " 
             << "Launching DISPATCH with "
             << n_plan << " pings using "
             << "gridDim(" << gridD.x << "," << gridD.y << "," << gridD.z << ")" << " "
             << "blockDim(" << blockD.x << "," << blockD.y << "," << blockD.z << ")" << " "
             << "\n" << flush;
#endif
        gpuErrchk(hipLaunchCooperativeKernel(
            (void*)ppnt::fused_kernel<DispatchTargetFn, DispatchArgs>,
            dim3(gridD), dim3(blockD),
            kernelArgs, 0, stream));
        gpuErrchk(hipStreamSynchronize(stream));

        // Check PingOut results
        // TODO: temporarily done here, move to end of main later on.
        for (size_t i = 0; i < n_plan; i++) {
            ppnt::PingOut o;
            gpuErrchk(hipMemcpy(&o, &d_out[i], sizeof(ppnt::PingOut), hipMemcpyDeviceToHost));

            // Copy per-iteration clock data to host
            // TODO: this may also need to change when k2_bpx > 1
            vector<uint64_t> h_iterClk(o.iters);
            gpuErrchk(hipMemcpy(h_iterClk.data(), o.iterClk, sizeof(uint64_t) * o.iters, hipMemcpyDeviceToHost));

            // Compute average cycles per iteration
            uint64_t total_cycles = 0;
            for (size_t it = 0; it < o.iters; it++) {
                total_cycles += h_iterClk[it];
            }
            const double avg_cycles = (double)total_cycles / (double)o.iters;
            const double avg_ns = (avg_cycles / (double)clock) * 1e3; // 1e3 as clock in MHz 

            // Compute bandwidth for Bandwidth pings
            string bw_str = "N/A"; // in GB/s
            if (o.kind == ppnt::PingKind::Bandwidth) {
                size_t iter_bytes = blockD.x * k2_n_datas * 16;
                double bw_gbps = ((double)iter_bytes / (avg_ns)) ; // GB/s
                bw_str = to_string(bw_gbps);
            }

            // Report
            cout << "[PPNT] Ping id=" << i << " "
                 << "kind=" << ((o.kind == ppnt::PingKind::Latency) ? "Latency" : "Bandwidth") << " "
                 << "src_xcd=" << o.src_xcd << " "
                 << "dst_hbm=" << o.dst_hbm << " "
                 << "iters=" << o.iters << " "
                 << fixed << setprecision(1)
                 << "avg_ns=" << avg_ns << " "
                 << "avg_GBps=" << bw_str << " "
                 << "\n" << flush;
        }

    } /* ppnt */

    // // --- Allocate per-expert weights ---
    // vector<float> h_W1((size_t)E * d * hidden);
    // vector<float> h_W2((size_t)E * hidden * d);
    // fill_random(h_W1);
    // fill_random(h_W2);

    // float *d_W1=nullptr, *d_W2=nullptr, *d_Tmp=nullptr, *d_Yexp=nullptr;
    // gpuErrchk(hipMalloc(&d_W1, sizeof(float) * h_W1.size()));
    // gpuErrchk(hipMalloc(&d_W2, sizeof(float) * h_W2.size()));
    // gpuErrchk(hipMemcpyAsync(d_W1, h_W1.data(), sizeof(float) * h_W1.size(), hipMemcpyHostToDevice, stream));
    // gpuErrchk(hipMemcpyAsync(d_W2, h_W2.data(), sizeof(float) * h_W2.size(), hipMemcpyHostToDevice, stream));

    // // Tmp: [E*cap, hidden], Yexp: [E*cap, d]
    // gpuErrchk(hipMalloc(&d_Tmp,  sizeof(float) * (size_t)E * cap * hidden));
    // gpuErrchk(hipMalloc(&d_Yexp, sizeof(float) * (size_t)E * cap * d));

    // // --- Launch per-expert GEMM1 ---
    // dim3 block1(16, 8, 1);
    // dim3 grid1((hidden + block1.x - 1) / block1.x,
    //         (cap    + block1.y - 1) / block1.y,
    //         E);
    // hipLaunchKernelGGL(expert_gemm1_naive, grid1, block1, 0, stream,
    //                 d_Xexp, d_W1, d_Tmp, d_cnt, E, cap, d, hidden);

    // // --- ReLU Tmp ---
    // int total_tmp = E * cap * hidden;
    // dim3 blockA(256);
    // dim3 gridA((total_tmp + blockA.x - 1) / blockA.x);
    // hipLaunchKernelGGL(relu_tmp_expert, gridA, blockA, 0, stream,
    //                 d_Tmp, d_cnt, E, cap, hidden);

    // // --- Launch per-expert GEMM2 ---
    // dim3 block2(16, 8, 1);
    // dim3 grid2((d   + block2.x - 1) / block2.x,
    //         (cap + block2.y - 1) / block2.y,
    //         E);
    // hipLaunchKernelGGL(expert_gemm2_naive, grid2, block2, 0, stream,
    //                 d_Tmp, d_W2, d_Yexp, d_cnt, E, cap, d, hidden);

    // // --- Combine back to token order ---
    // // For top-1 routing, your prior moe_gather is fine:
    // hipLaunchKernelGGL(moe_gather, gridD, blockD, 0, stream,
    //                 d_Yexp, d_pos, d_Y, T, d, cap);

    // // Cleanup later: d_W1, d_W2, d_Tmp, d_Yexp

}

