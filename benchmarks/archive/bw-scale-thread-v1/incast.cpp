// all xcds generate incast iod traffic

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>

#include "../tool/bmk1_ntbx.h"
#include "../tool/bmk3.h"

#ifndef BPX
#define BPX 1
#endif

#ifndef XCDS_NUM
#define XCDS_NUM 8
#endif

#ifndef TGT_XCD
#define TGT_XCD 0 // target xcd for incast
#endif

using namespace std;

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr int PAGE_SIZE = (2 * 1024 * 1024); // 2MB huge page
constexpr int CHUNK_SIZE = (2 * 1024); // chunk size (2KB/4KB). default 2KB to match interleaving granularity
constexpr int CLOCK = 2.1e9; // 2.1GHz

#define D_LOG_TID_0 0
#define DEBUG 0
#define LOG_RESULT_VERBOSE 0

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


__global__ void k(uint64_t **xcd_chunks, size_t *xcd_chunks_size, uint32_t **cycles_start, uint32_t **cycles_stop) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

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

    // set dst as incast target
    const uint32_t xcd_dst = TGT_XCD;

    // warmup
    // only 1 thread per xcd warms up entirely
    if (tbid_in_xcd == 0 && tid == 0) {
        for (size_t i = 0; i < xcd_chunks_size[xcd_dst]; i++) {
            uint64_t ptr = xcd_chunks[xcd_dst][i];
            uint4 *data_ptr = (uint4*)ptr;
            for (size_t offset = 0; offset < (CHUNK_SIZE / sizeof(uint4)); offset++) {
                asm volatile (
                    "flat_load_dwordx4 v[0:3], %0\n\t"
                    "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                    :
                    : "v"(&data_ptr[offset])
                    : "memory", "v0", "v1", "v2", "v3"
                );
            }
        }
    }
    xcd_barrier(xcc_id, n_tbs_in_xcd); // sync xcd

    // measurement
    size_t n_iter = xcd_chunks_size[xcd_dst] / n_tbs_in_xcd;
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
        uint64_t ptr = xcd_chunks[xcd_dst][chunk_idx];
        uint4 *data_ptr = (uint4*)ptr;
        
        // see tid 0 cycle counts for debug
        #if D_LOG_TID_0
        uint32_t log_start = 0, log_end = 0;
        if (tid == 0) {
            log_start = __builtin_readcyclecounter();
            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME
        }
        #endif
        
        asm volatile (
            "flat_load_dwordx4 v[0:3], %0\n\t"
            // "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            :
            : "v"(&data_ptr[tid])
            : "memory", "v0", "v1", "v2", "v3"
        );

        // see tid 0 cycle counts for debug
        #if D_LOG_TID_0
        if (tid == 0) {
            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // ensure load done before reading cycle counter
            log_end = __builtin_readcyclecounter();
            asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME
            printf("(debug) iter %zu tbid_in_xcd %d (bid %d, xcd %d): %u cycles\n", iter, tbid_in_xcd, bid, xcc_id, (log_end - log_start));
        }
        #endif
    }
    asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");

    // stop timing
    uint32_t stop = 0;
    stop = __builtin_readcyclecounter();
    asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"); // per ISA manual, need waitcnt after S_MEMTIME

    if (tid == 0) {
        cycles_start[xcc_id][tbid_in_xcd] = start;
        cycles_stop[xcc_id][tbid_in_xcd] = stop;
    }
}


int main() {
    printf("incast to xcd %d\n", TGT_XCD);

    const int n_threads_per_warp = 64;
    const int n_warps_per_block = (CHUNK_SIZE / (n_threads_per_warp * 16)); // set to make each tb load 1 chunk per iter. each thread loads 16B per iter.
    assert( (n_threads_per_warp * n_warps_per_block) <= 1024 );

    /* configure thread blocks */

    const int n_blocks = (BPX * XCDS_NUM);
    const int n_threads_per_block = (n_warps_per_block * n_threads_per_warp);
    const int total_threads = (n_blocks * n_threads_per_block);
    printf("n_xcds: %d, n_blocks: %d, n_blocks_per_xcd: %d, n_warps_per_block: %d, n_threads_per_warp: %d\n", XCDS_NUM, n_blocks, BPX, n_warps_per_block, n_threads_per_warp);
    
    /* allocate data */

    const long long n_pages = (128 << 8); // set to 128 (=256 MB) to fit in LLC
    const long long data_size = (n_pages * PAGE_SIZE);
    const int n_chunks = data_size / CHUNK_SIZE;
    printf("data_size: %d MB, chunk_size: %d KB\n", (int)data_size / (1024 * 1024), CHUNK_SIZE / 1024);

    char *d_data;
    gpuErrchk(hipMalloc((void**)&d_data, data_size + 0x1000));
    d_data = (char*)(((uintptr_t)d_data & ~(0x0FFF)) + 0x1000);

    /* allocate cycles array */

    uint32_t **d_cycles; // per-xcd, per-chunk. record cycles for home identification 
    gpuErrchk(hipMalloc((void**)&d_cycles, sizeof(uint32_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMalloc((void**)&d_cycles[x], sizeof(uint32_t) * n_chunks ));
    }
    printf("allocated d_cycles array[%d][%d]\n", XCDS_NUM, n_chunks);

    int *d_home; // per-chunk. record home xcd
    gpuErrchk(hipMalloc((void**)&d_home, sizeof(int) * n_chunks ));

    /* create cu masked stream */

    uint32_t cuMaskSize;
    vector<uint32_t> cuMask;
    const size_t n_cus_to_enable_per_xcd = (n_blocks / XCDS_NUM);
    const size_t n_cus_enabled_per_xcd = mask_cu(n_cus_to_enable_per_xcd, cuMaskSize, cuMask);
    printf("first %zu cus enabled per xcd\n", n_cus_enabled_per_xcd);
    #if DEBUG
    {
        printf("CU Mask: ");
        for (size_t i = 0; i < cuMaskSize; ++i) { printf("%08x ", cuMask[i]); }
        printf("\n" );
    }
    #endif
    
    hipStream_t stream;
    gpuErrchk(hipExtStreamCreateWithCUMask(&stream, cuMaskSize, cuMask.data()));

    /* launch home identification kernel */

    hipLaunchKernelGGL(
        identify_home,
        dim3(n_blocks), dim3(n_threads_per_block), 
        0, stream,
        (void*)d_data, data_size, d_cycles, d_home
    );
    gpuErrchk(hipDeviceSynchronize());

    /* retrieve and process results */
    
    vector<vector<uint32_t>> h_cycles(XCDS_NUM, vector<uint32_t>(n_chunks));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMemcpy(h_cycles[x].data(), d_cycles[x], sizeof(uint32_t) * n_chunks, hipMemcpyDeviceToHost));
    }
    
    vector<int> h_home(n_chunks);
    for (size_t i = 0; i < n_chunks; i++) {
        uint32_t min_cycles = UINT32_MAX;
        int min_xcc = -1;
        for (int xcc = 0; xcc < XCDS_NUM; xcc++) {
            uint32_t c = h_cycles[xcc][i];
            if (c < min_cycles) {
                min_cycles = c;
                min_xcc = xcc;
            }
        }
        h_home[i] = min_xcc;
        #if DEBUG
        {
            printf("4KB chunk[%zu]: home xcd %d\n", i, h_home[i]);
        }
        #endif
    }

    /* group per-xcd 4KB chunk pointers */
    
    vector<vector<uint64_t>> xcd_chunks(XCDS_NUM);
    for (size_t i = 0; i < n_chunks; i++) {
        xcd_chunks[h_home[i]].push_back(reinterpret_cast<uint64_t>(d_data) + i * (CHUNK_SIZE));
    }

    uint64_t **d_xcd_chunks;
    gpuErrchk(hipMalloc((void**)&d_xcd_chunks, sizeof(uint64_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        size_t n_chunks = xcd_chunks[x].size();
        gpuErrchk(hipMalloc((void**)&d_xcd_chunks[x], sizeof(uint64_t) * n_chunks ));
        gpuErrchk(hipMemcpy(d_xcd_chunks[x], xcd_chunks[x].data(), sizeof(uint64_t) * n_chunks, hipMemcpyHostToDevice));
    }   

    /* count # per-xcd chunk */

    // ensure minimal 8MB = 2 * l2_size per-xcd chunks in order to thrash l2
    // conservative. since 2 xcds on same iod actually share the home data, another approach can be
    // ensure minimal 8MB per-iod chunks 
    vector<size_t> xcd_chunks_size(XCDS_NUM);
    const int min_n_chunks_per_xcd = ((4*2 * 1024 * 1024) / (CHUNK_SIZE)); // minimal #chunks >= 8MB 
    int curr_min_n_chunks_per_xcd = INT_MAX;
    for (int xcd = 0; xcd < XCDS_NUM; xcd++) {
        xcd_chunks_size[xcd] = xcd_chunks[xcd].size();
        if (xcd_chunks[xcd].size() < curr_min_n_chunks_per_xcd) curr_min_n_chunks_per_xcd = xcd_chunks[xcd].size();
        printf("xcd %d has %zu chunks (%.2f MB)\n", xcd, xcd_chunks[xcd].size(), (xcd_chunks[xcd].size() * CHUNK_SIZE) / (1024.0 * 1024.0));
    }
    assert(curr_min_n_chunks_per_xcd >= min_n_chunks_per_xcd);
    
    size_t *d_xcd_chunks_size;
    gpuErrchk(hipMalloc((void**)&d_xcd_chunks_size, sizeof(size_t) * XCDS_NUM));
    gpuErrchk(hipMemcpy(d_xcd_chunks_size, xcd_chunks_size.data(), sizeof(size_t) * XCDS_NUM, hipMemcpyHostToDevice));

    /* allocate cycles_* array */

    // per-xcd, per-tb in xcd. record start/stop cycles for bw measurement
    uint32_t **d_cycles_start, **d_cycles_stop; 
    gpuErrchk(hipMalloc((void**)&d_cycles_start, sizeof(uint32_t*) * XCDS_NUM));
    gpuErrchk(hipMalloc((void**)&d_cycles_stop, sizeof(uint32_t*) * XCDS_NUM));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMalloc((void**)&d_cycles_start[x], sizeof(uint32_t) * BPX ));
        gpuErrchk(hipMalloc((void**)&d_cycles_stop[x], sizeof(uint32_t) * BPX ));
    }

    /* launch bw saturating kernel */

    hipLaunchKernelGGL(
        k,
        dim3(n_blocks), dim3(n_threads_per_block), 
        0, stream,
        d_xcd_chunks, 
        d_xcd_chunks_size,
        d_cycles_start,
        d_cycles_stop
    );
    gpuErrchk(hipDeviceSynchronize());

    /* retrieve and process results */

    vector<vector<uint32_t>> h_cycles_start(XCDS_NUM, vector<uint32_t>(BPX));
    vector<vector<uint32_t>> h_cycles_stop(XCDS_NUM, vector<uint32_t>(BPX));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMemcpy(h_cycles_start[x].data(), d_cycles_start[x], sizeof(uint32_t) * BPX, hipMemcpyDeviceToHost));
        gpuErrchk(hipMemcpy(h_cycles_stop[x].data(), d_cycles_stop[x], sizeof(uint32_t) * BPX, hipMemcpyDeviceToHost));
    }

    for (int xcd = 0; xcd < XCDS_NUM; xcd++) {
        // for avg lat calc
        double avg_xcd_cycles = 0.0;
        // for global bw calc
        uint32_t min_xcd_cycles_start = UINT32_MAX;
        uint32_t max_xcd_cycles_stop = 0;

        // set dst xcd as incast target
        const uint32_t xcd_dst = TGT_XCD;

        for (int tbid_in_xcd = 0; tbid_in_xcd < BPX; tbid_in_xcd++) {
            // incremental avg
            avg_xcd_cycles += ((h_cycles_stop[xcd][tbid_in_xcd] - h_cycles_start[xcd][tbid_in_xcd]) - avg_xcd_cycles) / (tbid_in_xcd + 1);

            // set earliest start cycle for this xcd
            if (h_cycles_start[xcd][tbid_in_xcd] < min_xcd_cycles_start) {
                min_xcd_cycles_start = h_cycles_start[xcd][tbid_in_xcd];
            }
            // set latest stop cycle for this xcd
            if (h_cycles_stop[xcd][tbid_in_xcd] > max_xcd_cycles_stop) {
                max_xcd_cycles_stop = h_cycles_stop[xcd][tbid_in_xcd];
            }

            #if LOG_RESULT_VERBOSE
            {
                // log per-tb results
                uint32_t cycles = h_cycles_stop[xcd][tbid_in_xcd] - h_cycles_start[xcd][tbid_in_xcd];
                double bytes = (double)((xcd_chunks_size[xcd_dst] / BPX) * CHUNK_SIZE); // total bytes accessed by this tb
                double cycles_per_load = (double)cycles / (bytes / n_threads_per_block / 16.0); // per 16B load
                printf("xcd %d, tb %d: start %u, stop %u, lat(16B) %.2f cycles\n", 
                        xcd, tbid_in_xcd, h_cycles_start[xcd][tbid_in_xcd], h_cycles_stop[xcd][tbid_in_xcd], cycles_per_load);
            }
            #endif
        }

        // avg. latency (RTT)
        const double bytes = (double)(xcd_chunks_size[xcd_dst] * CHUNK_SIZE); // total bytes accessed by this xcd
        const double lat_cycles_16B = avg_xcd_cycles / (bytes / (n_threads_per_block * BPX) / 16.0);  // avg cycles per 16B load
        printf("xcd %d: avg lat(16B) %.2f cycles\n", xcd, lat_cycles_16B);

        // global bw
        // bw based on max cycles among tbs in this xcd
        const double global_time_sec = (double)(max_xcd_cycles_stop - min_xcd_cycles_start) / CLOCK;
        const double global_bw_GBps = (bytes / global_time_sec) / 1e9;
        printf("xcd %d: global bw %.2f GB/s\n", xcd, global_bw_GBps);

        // rolling window bw
        // bw based on rolling window of 2 * avg RTT cycles
        const double window = 2 * lat_cycles_16B; // 2 * RTT
        const double window_time_sec = (double)window / CLOCK; // 2.1GHz
        double peak_window_bw_GBps = 0.0;
        for (uint32_t t = min_xcd_cycles_start; t < max_xcd_cycles_stop; t+=window) {
            double window_bytes = 0.0;
            for (int tbid_in_xcd = 0; tbid_in_xcd < BPX; tbid_in_xcd++) {
                if ( (t >= h_cycles_start[xcd][tbid_in_xcd]) && (t < h_cycles_stop[xcd][tbid_in_xcd]) ) {
                    // increment bytes by the approx. proportion loaded in this window
                    double bytes_contrib = (double)((xcd_chunks_size[xcd_dst] / BPX) * CHUNK_SIZE) * \
                                        (window / (h_cycles_stop[xcd][tbid_in_xcd] - h_cycles_start[xcd][tbid_in_xcd]));
                    window_bytes += bytes_contrib;
                }
            }
            const double window_bw_GBps = (window_bytes / window_time_sec) / 1e9;
            if (window_bw_GBps > peak_window_bw_GBps) {
                peak_window_bw_GBps = window_bw_GBps;
            }
            
            #if LOG_RESULT_VERBOSE
            {
                printf("xcd %d: window bw %.2f GB/s [%u-%u]\n",
                        xcd, window_bw_GBps, (unsigned int)(t), (unsigned int)(t + window));
            }
            #endif
        }
        printf("xcd %d: peak window bw %.2f GB/s\n", xcd, peak_window_bw_GBps);

        // print some misc
        printf("xcd %d: time %.6f sec, bytes %.2f MB\n", xcd, global_time_sec, bytes / (1024 * 1024));
    }

    /* cleanup */
    gpuErrchk(hipFree(d_data));

    return 0;
}