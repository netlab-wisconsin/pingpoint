// all xcds generate tornado iod traffic

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <vector>
#include <hip/hip_runtime.h>

#include "../mi300x_mapping/bmk1_1tbx.h"
#include "../mi300x_mapping/bmk3.h"

#ifndef BPX
#define BPX 1
#endif

#ifndef XCDS_NUM
#define XCDS_NUM 8
#endif

#ifndef XTO
#define XTO 2 // xcd tornado offset. 2 = 1 iod hop, 4 = 2 iod hops
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


__global__ void k(uint64_t *xcd_chunks1, uint64_t *xcd_chunks2, 
                    uint64_t *xcd_chunks3, uint64_t *xcd_chunks4,
                    size_t *xcd_chunks_offset1, size_t *xcd_chunks_offset2,
                    size_t *xcd_chunks_offset3, size_t *xcd_chunks_offset4,
                    size_t *xcd_chunks_size, uint32_t *cycles_start, 
                    uint32_t *cycles_stop) 
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

    // set dst as tornado
    const uint32_t xcd_dst = (xcc_id + XTO) % XCDS_NUM;

    // warmup
    // only 1 thread per xcd warms up entirely
    if (tbid_in_xcd == 0 && tid == 0) {
        for (size_t i = 0; i < xcd_chunks_size[xcd_dst]; i++) {

            float4 *ptr_in1 = reinterpret_cast<float4*>(xcd_chunks1[xcd_chunks_offset1[xcd_dst] + i]);
            float4 *ptr_in2 = reinterpret_cast<float4*>(xcd_chunks2[xcd_chunks_offset2[xcd_dst] + i]);
            float4 *ptr_in3 = reinterpret_cast<float4*>(xcd_chunks3[xcd_chunks_offset3[xcd_dst] + i]);
            float4 *ptr_in4 = reinterpret_cast<float4*>(xcd_chunks4[xcd_chunks_offset4[xcd_dst] + i]);

            for (size_t offset = 0; offset < (CHUNK_SIZE / sizeof(float4)); offset++) {
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

        float4 *ptr_in1 = reinterpret_cast<float4*>(xcd_chunks1[xcd_chunks_offset1[xcd_dst] + chunk_idx]);
        float4 *ptr_in2 = reinterpret_cast<float4*>(xcd_chunks2[xcd_chunks_offset2[xcd_dst] + chunk_idx]);
        float4 *ptr_in3 = reinterpret_cast<float4*>(xcd_chunks3[xcd_chunks_offset3[xcd_dst] + chunk_idx]);
        float4 *ptr_in4 = reinterpret_cast<float4*>(xcd_chunks4[xcd_chunks_offset4[xcd_dst] + chunk_idx]);
        
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
        cycles_start[xcc_id * BPX + tbid_in_xcd] = start;
        cycles_stop[xcc_id * BPX + tbid_in_xcd] = stop;
    }

    grid.sync(); // sync all xcds post measurement
}


int main() {
    switch (XTO) {
        case 2:
            printf("tornado traffic (1 hop)\n");
            break;
        case 4:
            printf("tornado traffic (2 hops)\n");
            break;
        case 6:
            printf("tornado traffic (3 hops)\n");
            break;
        default:
            printf("invalid XTO %d\n", XTO);
            exit(1);
    }

    const int n_threads_per_warp = 64;
    const int n_warps_per_block = (CHUNK_SIZE / (n_threads_per_warp * 16)); // set to make each tb load 1 chunk per iter. each thread loads 16B per iter.
    assert( (n_threads_per_warp * n_warps_per_block) <= 1024 );

    /* configure thread blocks */

    const int n_blocks = (BPX * XCDS_NUM);
    const int n_threads_per_block = (n_warps_per_block * n_threads_per_warp);
    const int total_threads = (n_blocks * n_threads_per_block);
    printf("n_xcds: %d, n_blocks: %d, n_blocks_per_xcd: %d, n_warps_per_block: %d, n_threads_per_warp: %d\n", XCDS_NUM, n_blocks, BPX, n_warps_per_block, n_threads_per_warp);
    
    /* allocate data */

    const int n_datas = 4;

    const long long n_pages = (128 << 6); // 16GB per input data
    // const long long n_pages = 128; // 256MB per input data
    const long long data_size = (n_pages * PAGE_SIZE);
    const size_t n_chunks = data_size / CHUNK_SIZE;
    printf("n_datas: %d, data_size: %lld MB, chunk_size: %d KB\n", n_datas, data_size / (1024 * 1024), CHUNK_SIZE / 1024);

    vector<char*> d_data(n_datas);
    for (int i = 0; i < n_datas; i++) {
        gpuErrchk(hipMalloc((void**)&d_data[i], data_size + 0x1000));
        d_data[i] = (char*)(((uintptr_t)d_data[i] & ~(0x0FFF)) + 0x1000);
    }

    /* allocate cycles array */

    // per-data, per-xcd, per-chunk. record cycles for home identification
    // last two dimensions are flattened
    vector<uint32_t*> d_cycles(n_datas); 
    for (int i = 0; i < n_datas; i++) {
        gpuErrchk(hipMalloc((void**)&d_cycles[i], sizeof(uint32_t) * XCDS_NUM * n_chunks ));
        printf("allocated d_cycles[%d] array[%d][%zu]\n", i, XCDS_NUM, n_chunks);
    }

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

    vector<vector<int>> h_home(n_datas, vector<int>(n_chunks));
    vector<vector<size_t>> h_xcd_chunks_size(n_datas, vector<size_t>(XCDS_NUM, 0)); // count #chunks per-data per-xcd. init to 0

    for (int i = 0; i < n_datas; i++) {
        printf("Identifying home xcd for data[%d]...\n", i);

        void *kernel_args[] = {
            (void*)&d_data[i],
            (void*)&data_size,
            (void*)&d_cycles[i], // XCDS_NUM * n_chunks
            (void*)&n_chunks
        };

        gpuErrchk(hipLaunchCooperativeKernel(
            identify_home,
            dim3(1 * 8), dim3(128), 
            kernel_args, 0, stream
        ));
        gpuErrchk(hipDeviceSynchronize());

        /* retrieve and process results */
        
        vector<vector<uint32_t>> h_cycles(XCDS_NUM, vector<uint32_t>(n_chunks));
        for (int x = 0; x < XCDS_NUM; x++) {
            gpuErrchk(hipMemcpy(
                h_cycles[x].data(), 
                &d_cycles[i][x * n_chunks], 
                sizeof(uint32_t) * n_chunks, 
                hipMemcpyDeviceToHost
            ));
        }

        for (size_t k = 0; k < n_chunks; k++) {
            uint32_t min_cycles = 0xFFFFFFFF;
            int min_xcc = -1;
            for (int x = 0; x < XCDS_NUM; x++) {
                uint32_t c = h_cycles[x][k];
                if (c < min_cycles) {
                    min_cycles = c;
                    min_xcc = x;
                }
            }
            h_home[i][k] = min_xcc;
            h_xcd_chunks_size[i][min_xcc]++;
            #if DEBUG
            printf("data[%d] chunk[%zu]: home xcd %d\n", i, k, h_home[i][k]);
            #endif
        }
        #if DEBUG
        for (int x = 0; x < XCDS_NUM; x++) {
            printf("data[%d] xcd %d: n_chunks %zu\n", i, x, h_xcd_chunks_size[i][x]);
        }
        printf("\n");
        #endif
    }

    /* group per-xcd chunk pointers */
    
    vector<vector<vector<uint64_t>>> xcd_chunks(n_datas, vector<vector<uint64_t>>(XCDS_NUM));
    for (int i = 0; i < n_datas; i++) {
        for (size_t k = 0; k < n_chunks; k++) {
            xcd_chunks[i][h_home[i][k]].push_back(reinterpret_cast<uint64_t>(d_data[i]) + k * (CHUNK_SIZE));
        }
    }
    // safety check
    for (int i = 0; i < n_datas; i++) {
        for (int x = 0; x < XCDS_NUM; x++) {
            assert( xcd_chunks[i][x].size() == h_xcd_chunks_size[i][x] );
            printf("data[%d] xcd %d: n_chunks %zu\n", i, x, xcd_chunks[i][x].size());
        }
    }

    vector<uint64_t*> d_xcd_chunks(n_datas);
    vector<size_t*> d_xcd_chunks_offset(n_datas); // set starting point for each xcd in 1d d_xcd_chunks array
    for (int i = 0; i < n_datas; i++) {
        gpuErrchk(hipMalloc((void**)&d_xcd_chunks[i], sizeof(uint64_t) * n_chunks ));
        gpuErrchk(hipMalloc((void**)&d_xcd_chunks_offset[i], sizeof(size_t) * XCDS_NUM));
        size_t _offset = 0;
        for (int x = 0; x < XCDS_NUM; x++) {
            size_t _n_chunks = h_xcd_chunks_size[i][x];
            gpuErrchk(hipMemcpy(&d_xcd_chunks[i][_offset], xcd_chunks[i][x].data(), sizeof(uint64_t) * _n_chunks, hipMemcpyHostToDevice));
            d_xcd_chunks_offset[i][x] = _offset;
            _offset += _n_chunks;
        }
    }

    /* count # per-xcd chunk */
    
    vector<size_t> xcd_chunks_size(XCDS_NUM, SIZE_MAX);
    for (int i = 0; i < n_datas; i++) {
        for (int x = 0; x < XCDS_NUM; x++) {
            // Determine the minimum chunk count for this XCD across all datasets.
            // This establishes a safe common upper bound, preventing out-of-bounds access on datasets with fewer chunks.
            xcd_chunks_size[x] = min(xcd_chunks_size[x], h_xcd_chunks_size[i][x]);
        }   
    }
    // ensure minimal 8MB = 2 * l2_size per-xcd chunks in order to thrash l2
    // conservative. since 2 xcds on same iod actually share the home data, another approach can be
    // ensure minimal 8MB per-iod chunks 
    const int min_n_chunks_per_xcd = ((4*2 * 1024 * 1024) / (CHUNK_SIZE)); // minimal #chunks >= 8MB
    for (int x = 0; x < XCDS_NUM; x++) {
        printf("xcd %d: min n_chunks %zu\n", x, xcd_chunks_size[x]);
        assert (xcd_chunks_size[x] * n_datas >= min_n_chunks_per_xcd); // xcd_chunks_size[x] is minimal #chunks per xcd among all datas. 
    }
    
    size_t *d_xcd_chunks_size;
    gpuErrchk(hipMalloc((void**)&d_xcd_chunks_size, sizeof(size_t) * XCDS_NUM));
    gpuErrchk(hipMemcpy(d_xcd_chunks_size, xcd_chunks_size.data(), sizeof(size_t) * XCDS_NUM, hipMemcpyHostToDevice));

    /* allocate cycles_* array */

    // per-xcd, per-tb in xcd. record start/stop cycles for bw measurement
    uint32_t *d_cycles_start, *d_cycles_stop; 
    gpuErrchk(hipMalloc((void**)&d_cycles_start, sizeof(uint32_t) * XCDS_NUM * BPX ));
    gpuErrchk(hipMalloc((void**)&d_cycles_stop, sizeof(uint32_t) * XCDS_NUM * BPX ));

    /* launch bw saturating kernel */

    void *kernel_args[] = {
        (void*)&d_xcd_chunks[0],
        (void*)&d_xcd_chunks[1],
        (void*)&d_xcd_chunks[2],
        (void*)&d_xcd_chunks[3],
        (void*)&d_xcd_chunks_offset[0],
        (void*)&d_xcd_chunks_offset[1],
        (void*)&d_xcd_chunks_offset[2],
        (void*)&d_xcd_chunks_offset[3],
        (void*)&d_xcd_chunks_size,
        (void*)&d_cycles_start,
        (void*)&d_cycles_stop
    };

    gpuErrchk(hipLaunchCooperativeKernel(
        k,
        dim3(n_blocks), dim3(n_threads_per_block), 
        kernel_args, 0, stream
    ));
    gpuErrchk(hipDeviceSynchronize());

    /* retrieve and process results */

    vector<vector<uint32_t>> h_cycles_start(XCDS_NUM, vector<uint32_t>(BPX));
    vector<vector<uint32_t>> h_cycles_stop(XCDS_NUM, vector<uint32_t>(BPX));
    for (int x = 0; x < XCDS_NUM; x++) {
        gpuErrchk(hipMemcpy(
            h_cycles_start[x].data(), 
            &d_cycles_start[x * BPX], 
            sizeof(uint32_t) * BPX, 
            hipMemcpyDeviceToHost
        ));
        gpuErrchk(hipMemcpy(
            h_cycles_stop[x].data(), 
            &d_cycles_stop[x * BPX], 
            sizeof(uint32_t) * BPX, 
            hipMemcpyDeviceToHost
        ));
    }

    for (int xcd = 0; xcd < XCDS_NUM; xcd++) {
        // for avg lat calc
        double avg_xcd_cycles = 0.0;
        // for global bw calc
        uint32_t min_xcd_cycles_start = 0xFFFFFFFF;
        uint32_t max_xcd_cycles_stop = 0;

        // set dst xcd as tornado
        const uint32_t xcd_dst = (xcd + XTO) % XCDS_NUM;

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
                double bytes = (double)((xcd_chunks_size[xcd_dst] / BPX) * CHUNK_SIZE * n_datas); // total bytes accessed by this tb
                double cycles_per_load = (double)cycles / (bytes / n_threads_per_block / 16.0); // per 16B load
                printf("xcd %d, tb %d: start %u, stop %u, lat(16B) %.2f cycles, bw %.2f GB/s\n", 
                        xcd, tbid_in_xcd, h_cycles_start[xcd][tbid_in_xcd], h_cycles_stop[xcd][tbid_in_xcd], cycles_per_load, (bytes / ((double)cycles / CLOCK)) / 1e9);
            }
            #endif
        }

        // avg. latency (RTT)
        const double bytes = (double)(xcd_chunks_size[xcd_dst] * CHUNK_SIZE * n_datas); // total bytes accessed by this xcd
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
                    double bytes_contrib = (double)((xcd_chunks_size[xcd_dst] / BPX) * CHUNK_SIZE * n_datas) * \
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

    return 0;
}