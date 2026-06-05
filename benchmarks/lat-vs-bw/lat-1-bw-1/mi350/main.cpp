#include <hip/hip_runtime.h>
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

#include "../../../../mem_bench/MeasurementSeries.hpp"
#include "../../../../mem_bench/dtime.hpp"
#include "../../../../mem_bench/gpu-clock.cuh"

#include "main.h"
#include "k1.h"
#include "k2.h"

using namespace std;
typedef int64_t dtype;

#ifndef K1_PINNED_XCD
#define K1_PINNED_XCD 0
#endif

#ifndef K1_PINNED_CC
#define K1_PINNED_CC 0
#endif

#ifndef K2_PINNED_XCD
#define K2_PINNED_XCD 1
#endif

#ifndef K2_PINNED_CC
#define K2_PINNED_CC 1
#endif

/*
 * (Note 01/25/25) While removed, two K2 flows were instantiated to test YX routing's path overlap
 * flow 1: K2_PINNED_XCD_1, K2_PINNED_CC_1
 * flow 2: K2_PINNED_XCD_2, K2_PINNED_CC_2
 * K1 was disabled, and K2 kernel preluded the two XCDs. k2_cc was recognized using below code:
 * const uint32_t k2_cc = (xcc_id == k2_xcd1) ? k2_cc1 : k2_cc2;
 */

#ifndef SKIP_HOME_IDENTIFICATION
#define SKIP_HOME_IDENTIFICATION 0
#endif

#ifndef K2_BPX_MIN
#define K2_BPX_MIN 1
#endif

#ifndef K2_BPX_MAX
#define K2_BPX_MAX 40
#endif

#ifndef K2_TPB
#define K2_TPB 512
#endif

#ifndef EPOCHS
#define EPOCHS 21 // set to 1 when profiling
#endif

#define DEBUG_LEVEL 2
#define FAST 0 // set for faster debugging, set 0 for actual measurement

int main(int argc, char **argv)
{

#ifdef PROFILE
    printf("Running in PROFILE mode\n");
#else
    printf("Running in RUN mode\n");
#endif

#if DEBUG_LEVEL >= 0
    cout << "K1_PINNED_XCD: " << K1_PINNED_XCD << " " << "K1_PINNED_CC: " << K1_PINNED_CC << " "
         << "K2_PINNED_XCD: " << K2_PINNED_XCD << " " << "K2_PINNED_CC: " << K2_PINNED_CC << " "
         << "\n" << flush;

    cout << "K2_TPB: " << K2_TPB << " "
         << "\n" << flush;
#endif

    std::random_device rd;
    std::mt19937 g(rd());

    const size_t cl_bytes = 128;

    ////////////////////////// K1 SETUP //////////////////////////

    // Data allocation
    dtype *dbuf_base = nullptr;

#ifdef PROFILE
    // When profiling, use reduced LEN with *2 mult factor to avoid below OOB error
    // But don't further reduce LEN. 1<<15 is almost minimal for bypassing L2.
    const size_t LEN = (1 << 15);
    const size_t multiplicative_factor = XCD_NUM * 2;
#elif 0
    // Utilize HBM
    const size_t LEN = (1 << 22); // set to use HBM for K1
    const size_t multiplicative_factor = XCD_NUM * 1;
#else
    // Utilize LLC
    const size_t LEN = (1 << 16); // set to use LLC for K1
    const size_t multiplicative_factor = XCD_NUM * 1;
#endif

    const size_t cl_size = cl_bytes / sizeof(dtype);
    const size_t skip_factor = 1;
    const size_t n_dtype_dbuf = multiplicative_factor * skip_factor * cl_size * LEN;
    const size_t n_cl_dbuf = n_dtype_dbuf / (cl_size * skip_factor);
    gpuErrchk(hipMalloc(&dbuf_base, n_dtype_dbuf * sizeof(dtype)));

    // Per-dtype xcd identification.
    vector<uint32_t> dtype_home_xcd(n_dtype_dbuf, (uint32_t)-1);
    vector<vector<uint32_t>> xcd_dtypes(XCD_NUM);

    // Per-cacheline xcd identification.
    vector<uint32_t> cl_home_xcd(n_cl_dbuf, (uint32_t)-1);
    vector<vector<uint32_t>> xcd_cls(XCD_NUM);

#if !SKIP_HOME_IDENTIFICATION
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

#if DEBUG_LEVEL >= 1
    string level = xcd_dtypes[K1_PINNED_CC].size() * sizeof(dtype) > L2_SIZE ?
                    (xcd_dtypes[K1_PINNED_CC].size() * sizeof(dtype) > LLC_SIZE ? "hbm" : "llc")
                    : "l2";
    assert (level != "l2"); // K1 data should be at least in LLC
    cout << "K1 pinned data: " << "cc" << K1_PINNED_CC << " "
            << xcd_dtypes[K1_PINNED_CC].size() * sizeof(dtype) / (1024 * 1024) << "MB" << " "
            << "at " << level << " "
            << "\n" << flush;
#endif
#else
    // Skip home identification: populate xcd_cls[K1_PINNED_CC] with all cache line indices
    for (size_t k = 0; k < n_cl_dbuf; k++)
        xcd_cls[K1_PINNED_CC].push_back((uint32_t)k);
#if DEBUG_LEVEL >= 1
    cout << "K1 home identification skipped: using all " << n_cl_dbuf << " cls\n" << flush;
#endif
#endif

    ////////////////////////// K2 SETUP //////////////////////////

    const int k2_n_datas = 4;

#if !(FAST)
    const long long k2_n_pages = (128 << 6); // set 16GB per input data
#else
    const long long k2_n_pages = 512; // set 1024MB per input data, for faster debugging
#endif

    const int k2_page_size = (2 * 1024 * 1024); // 2MB huge page
    const long long k2_data_size = (k2_n_pages * k2_page_size);

    const int k2_chunk_size = K2_CHUNK_SIZE;
    const size_t k2_n_chunks = k2_data_size / k2_chunk_size;

    vector<char *> k2_d_data(k2_n_datas);
    for (int i = 0; i < k2_n_datas; i++)
    {
        gpuErrchk(hipMalloc((void **)&k2_d_data[i], k2_data_size + 0x1000));
        k2_d_data[i] = (char *)(((uintptr_t)k2_d_data[i] & ~(0x0FFF)) + 0x1000);
    }

    // Per-chunk xcd identification.
    vector<vector<int>> k2_h_home(k2_n_datas, vector<int>(k2_n_chunks));
    vector<vector<size_t>> k2_h_xcd_chunks_size(k2_n_datas, vector<size_t>(XCD_NUM, 0)); // count #chunks per-data per-xcd. init to 0

#if !SKIP_HOME_IDENTIFICATION
    // Data home identification
    if (k2::home_identification(
            k2_d_data,
            k2_data_size,
            k2_n_chunks,
            k2_n_datas,
            k2_h_home,
            k2_h_xcd_chunks_size) == -1)
        return -1;
#else
    // Skip home identification: assign all chunks to K2_PINNED_CC
    for (int i = 0; i < k2_n_datas; i++) {
        for (size_t k = 0; k < k2_n_chunks; k++) {
            k2_h_home[i][k] = K2_PINNED_CC;
            k2_h_xcd_chunks_size[i][K2_PINNED_CC]++;
        }
    }
#if DEBUG_LEVEL >= 1
    cout << "K2 home identification skipped: using all " << k2_n_chunks << " chunks per data\n" << flush;
#endif
#endif

    /* group per-xcd chunk pointers */

    vector<vector<vector<uint64_t>>> k2_xcd_chunks(k2_n_datas, vector<vector<uint64_t>>(XCD_NUM));
    for (int i = 0; i < k2_n_datas; i++)
    {
        for (size_t k = 0; k < k2_n_chunks; k++)
        {
            k2_xcd_chunks[i][k2_h_home[i][k]].push_back(reinterpret_cast<uint64_t>(k2_d_data[i]) + k * (k2_chunk_size));
        }
    }
    // safety check
    for (int i = 0; i < k2_n_datas; i++)
    {
        for (int x = 0; x < XCD_NUM; x++)
        {
            assert(k2_xcd_chunks[i][x].size() == k2_h_xcd_chunks_size[i][x]);
#if DEBUG_LEVEL >= 2
            printf("data[%d] xcd %d: n_chunks %zu\n", i, x, k2_xcd_chunks[i][x].size());
#endif
        }
    }

    vector<uint64_t *> k2_d_xcd_chunks(k2_n_datas);
    vector<size_t *> k2_d_xcd_chunks_offset(k2_n_datas); // set starting point for each xcd in 1d k2_d_xcd_chunks array
    for (int i = 0; i < k2_n_datas; i++)
    {
        gpuErrchk(hipMalloc((void **)&k2_d_xcd_chunks[i], sizeof(uint64_t) * k2_n_chunks));
        gpuErrchk(hipMalloc((void **)&k2_d_xcd_chunks_offset[i], sizeof(size_t) * XCD_NUM));
        size_t _offset = 0;
        for (int x = 0; x < XCD_NUM; x++)
        {
            size_t _n_chunks = k2_h_xcd_chunks_size[i][x];
            gpuErrchk(hipMemcpy(&k2_d_xcd_chunks[i][_offset], k2_xcd_chunks[i][x].data(), sizeof(uint64_t) * _n_chunks, hipMemcpyHostToDevice));
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

    // ensure minimal 8MB = 2 * l2_size per-CC chunks in order to thrash l2
    const int min_n_chunks_per_cc = ((4*2 * 1024 * 1024) / (k2_chunk_size)); // minimal #chunks >= 8MB

    for (int x = 0; x < CC_NUM; x++)
    {
#if DEBUG_LEVEL >= 2
        printf("cc %d: min n_chunks %zu\n", x, k2_xcd_chunks_size[x]);
#endif
#if !SKIP_HOME_IDENTIFICATION
        assert(k2_xcd_chunks_size[x] * k2_n_datas >= min_n_chunks_per_cc); // k2_xcd_chunks_size[x] is minimal #chunks per cc among all datas.
#else
        if (x == K2_PINNED_CC)
            assert(k2_xcd_chunks_size[x] * k2_n_datas >= min_n_chunks_per_cc);
#endif
    }

    size_t *k2_d_xcd_chunks_size;
    gpuErrchk(hipMalloc((void **)&k2_d_xcd_chunks_size, sizeof(size_t) * XCD_NUM));
    gpuErrchk(hipMemcpy(k2_d_xcd_chunks_size, k2_xcd_chunks_size.data(), sizeof(size_t) * XCD_NUM, hipMemcpyHostToDevice));

    // safety check
    size_t k2_total_pinned_chunks = accumulate(k2_h_xcd_chunks_size.begin(), k2_h_xcd_chunks_size.end(), 0UL,
                                               [](size_t sum, const auto &row)
                                               { return sum + row[K2_PINNED_CC]; });

#if 1
    // Utilize HBM
    assert(k2_total_pinned_chunks * k2_chunk_size >= LLC_SIZE);
#else
    // Utilize LLC
    assert(k2_total_pinned_chunks * k2_chunk_size >= L2_SIZE);
#endif

#if DEBUG_LEVEL >= 1
    cout << "K2 pinned data: "<< "cc" << K2_PINNED_CC << " "
        << (k2_h_xcd_chunks_size[0][K2_PINNED_CC] + k2_h_xcd_chunks_size[1][K2_PINNED_CC] + \
            k2_h_xcd_chunks_size[2][K2_PINNED_CC] + k2_h_xcd_chunks_size[3][K2_PINNED_CC]) * \
            k2_chunk_size / (1024 * 1024) << "MB "
        << "\n" << flush;
#endif

    ////////////////////////// MAIN TEST LOOP //////////////////////////

    // unsigned int clock = getGPUClock();
    unsigned int clock = 2200; // set to 2200MHz as mi3508x node has some issue.

    cout << setw(3) << "BPX" << " "
         << setw(5) << "Clk" << " "
         << setw(9) << "K1_It" << " "
         << setw(9) << "K2_It" << " "
         << setw(7) << "Mean" << " "
         << setw(7) << "P90" << " "
         << setw(7) << "P95" << " "
         << setw(7) << "P99" << " "
         << setw(7) << "GB/s"
         << "\n"
         << string(70, '-') // Optional: Divider line
         << "\n"
         << flush;

    for (int k2_bpx = K2_BPX_MIN; k2_bpx <= K2_BPX_MAX; k2_bpx++)
    {
        MeasurementSeries k1_times, k2_times;
#if !(FAST)
        const int64_t k1_iters = max(LEN, (int64_t)100000);
#else
        const int64_t k1_iters = (int64_t)100000; // for faster debugging
#endif

        // Multiplying by > 1 may make k2 to more outlive k1
        // but benefit is not clear for now, whereas it increases total run time too much
        const int64_t k2_iters = k1_iters * 1;

        ////////////////////////// STREAM SETUP //////////////////////////

        // streams
        // stream with CU mask and Priority is set for each K1 and K2
        hipStream_t k1_stream, k2_stream;

#if 1
        // cu masking
        uint64_t k1_cuMask_bits, k2_cuMask_bits;
        uint32_t k1_cuMaskSize, k2_cuMaskSize;
        vector<uint32_t> k1_cuMask, k2_cuMask;
        k1_cuMask_bits = (1ULL << 0); // use SE3 CU0 only
        assert(mask_cu(k1_cuMask_bits, k1_cuMaskSize, k1_cuMask) == 1);
        const int valid_k2_cu_count = min(k2_bpx, CU_NUM - 1);   // can use up to CU_NUM-1 CUs (exclude CU0)
        k2_cuMask_bits = ((1ULL << valid_k2_cu_count) - 1) << 1; // Create mask of 1s and shift left by 1 to skip CU 0
                                                                 // Example (count=2): 0011 -> 0110 (Enables 1 & 2)
        assert(mask_cu(k2_cuMask_bits, k2_cuMaskSize, k2_cuMask) == valid_k2_cu_count);
        gpuErrchk(hipExtStreamCreateWithCUMask(&k1_stream, k1_cuMaskSize, k1_cuMask.data()));
        gpuErrchk(hipExtStreamCreateWithCUMask(&k2_stream, k2_cuMaskSize, k2_cuMask.data()));
#else
        // simple streams
        gpuErrchk(hipStreamCreate(&k1_stream));
        gpuErrchk(hipStreamCreate(&k2_stream));
#endif

#if 0
        // prio
        // Set k1_stream to high priority, k2_stream to low priority
        // (01/16) disabled due to conflicting APIs (Mask vs. Priority)
        hipStreamAttrValue k1_stream_attr, k2_stream_attr;
        int prio_hi, prio_lo;
        gpuErrchk(hipDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
        k1_stream_attr.priority = prio_hi; k2_stream_attr.priority = prio_lo;
        gpuErrchk(hipStreamSetAttribute(k1_stream, hipStreamAttributePriority, &k1_stream_attr));
        gpuErrchk(hipStreamSetAttribute(k2_stream, hipStreamAttributePriority, &k2_stream_attr));
#endif

        for (int epoch = 0; epoch < EPOCHS; epoch++)
        {

            ////////////////////////// K1 RANDOMIZATION //////////////////////////

            // Allocate buf for the FULL n_cl_dbuf domain (i.e., full dbuf domain).
            // In practice this means buf is sized to n_dtype_dbuf (the full dtype domain).
            dtype *buf = nullptr;
            gpuErrchk(hipMallocManaged(&buf, n_dtype_dbuf * sizeof(dtype)));
            std::memset(buf, 0, n_dtype_dbuf * sizeof(dtype));

            dtype *dummy_buf = nullptr;
            gpuErrchk(hipMallocManaged(&dummy_buf, sizeof(dtype)));
            dummy_buf[0] = 0;

            // Choose LEN cache lines from PINNED_CC, shuffle them
            vector<uint32_t> seq(LEN);
            for (int64_t i = 0; i < LEN; i++)
            {
                seq[i] = xcd_cls[K1_PINNED_CC][(size_t)i];
            }
            shuffle(seq.begin(), seq.end(), g);

#if DEBUG_LEVEL >= 3
            cout << "Access CL sequence (CL indices in full domain): ";
            for (int64_t i = 0; i < LEN; i++)
                cout << seq[(size_t)i] << " ";
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
            for (int cl_lane = 0; cl_lane < cl_size; cl_lane++)
            {
                for (int64_t i = 0; i < LEN; i++)
                {
                    uint32_t cur_cl = seq[(size_t)i];
                    uint32_t next_cl = seq[(size_t)((i + 1) % LEN)];

                    size_t cur_elem = ((size_t)cur_cl * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;
                    size_t next_elem = ((size_t)next_cl * (size_t)cl_size + (size_t)cl_lane) * (size_t)skip_factor;

                    // Bounds safety (should hold if n_cl_dbuf is consistent)
                    if (cur_elem >= n_dtype_dbuf || next_elem >= n_dtype_dbuf)
                    {
                        cerr << "BUG: elem OOB: cur_elem=" << cur_elem
                             << " next_elem=" << next_elem
                             << " n_dtype_dbuf=" << n_dtype_dbuf << "\n";
                        gpuErrchk(hipFree(buf));
                        gpuErrchk(hipFree(dummy_buf));
                        gpuErrchk(hipFree(dbuf_base));
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
            gpuErrchk(hipMemcpy(dbuf_base, buf, n_dtype_dbuf * sizeof(dtype), hipMemcpyHostToDevice));
            gpuErrchk(hipDeviceSynchronize());

            // Start pointer: lane 0 of the first cache line in seq[]
            size_t start_elem = ((size_t)seq[0] * (size_t)cl_size + 0) * (size_t)skip_factor;
            dtype *dbuf_start = dbuf_base + start_elem;

#if 0
            // start/stop cycles for lat measurement
            // (01/16) unused for now
            uint32_t *d_k1_startClk, *d_k1_stopClk;
            GPU_ERROR(hipMalloc(&d_k1_startClk, sizeof(uint32_t) * iters));
            GPU_ERROR(hipMalloc(&d_k1_stopClk, sizeof(uint32_t) * iters));
#endif

            ////////////////////////// K2 CONFIGURATION //////////////////////////

            const int k2_n_blocks = (k2_bpx * XCD_NUM);
            const int k2_n_threads_per_warp = 64;
            const int k2_n_warps_per_block = K2_TPB / k2_n_threads_per_warp;
            const int k2_n_threads_per_block = (k2_n_warps_per_block * k2_n_threads_per_warp);
            assert(k2_n_threads_per_block % 128 == 0); // assert multiple 128 => each block loads multiples of 1 chunk per iter
            assert(k2_n_threads_per_block <= 1024);    // assert max 1024 threads per block
            const int k2_total_threads = (k2_n_blocks * k2_n_threads_per_block);

#if 0
            // per-xcd, per-tb in xcd. record start/stop cycles for bw measurement
            // (01/16) unused for now
            uint32_t *d_k2_startClk, *d_k2_stopClk;
            GPU_ERROR(hipMalloc((void**)&d_k2_startClk, sizeof(uint32_t) * k2_bpx ));
            GPU_ERROR(hipMalloc((void**)&d_k2_stopClk, sizeof(uint32_t) * k2_bpx ));
#endif

            float *k2_dummy_sink;
            gpuErrchk(hipMallocManaged(&k2_dummy_sink, sizeof(float) * k2_total_threads));


            ////////////////////////// KERNEL LAUNCH //////////////////////////

#if 0 // (01/25/25) disable warmup due to LLC caching effect
            // K1 warmup
            k1::k<dtype><<<dim3(XCD_NUM), dim3(1), 0, k1_stream>>>(dbuf_start, dummy_buf, k1_iters, K1_PINNED_XCD, 3, 0); // SE3 CU0
            k1::k<dtype><<<dim3(XCD_NUM), dim3(1), 0, k1_stream>>>(dbuf_start, dummy_buf, k1_iters, K1_PINNED_XCD, 3, 0); // SE3 CU0
            gpuErrchk(hipDeviceSynchronize());
#endif 

            // Events
            hipEvent_t k1_start, k1_stop;
            hipEvent_t k2_start, k2_stop;
            gpuErrchk(hipEventCreate(&k1_start));
            gpuErrchk(hipEventCreate(&k1_stop));
            gpuErrchk(hipEventCreate(&k2_start));
            gpuErrchk(hipEventCreate(&k2_stop));

            // K2 launch
            gpuErrchk(hipEventRecord(k2_start, k2_stream));
            k2::k<<<dim3(k2_n_blocks), dim3(k2_n_threads_per_block), 0, k2_stream>>>(
                k2_d_xcd_chunks[0], k2_d_xcd_chunks[1], k2_d_xcd_chunks[2], k2_d_xcd_chunks[3],
                k2_d_xcd_chunks_offset[0], k2_d_xcd_chunks_offset[1], k2_d_xcd_chunks_offset[2], k2_d_xcd_chunks_offset[3],
                k2_dummy_sink, k2_d_xcd_chunks_size, k2_chunk_size, k2_iters, K2_PINNED_XCD, K2_PINNED_CC);
            gpuErrchk(hipEventRecord(k2_stop, k2_stream));

            // K1 launch
            gpuErrchk(hipEventRecord(k1_start, k1_stream));
            k1::k<dtype><<<dim3(XCD_NUM), dim3(1), 0, k1_stream>>>(dbuf_start, dummy_buf, k1_iters, K1_PINNED_XCD, 3, 0); // SE3 CU0
            gpuErrchk(hipEventRecord(k1_stop, k1_stream));

            // Wait stream
            gpuErrchk(hipStreamSynchronize(k1_stream));
            gpuErrchk(hipStreamSynchronize(k2_stream));


            ////////////////////////// METRICS COLLECTION //////////////////////////


            float k1_msec = 0.0f, k2_msec = 0.0f;
            gpuErrchk(hipEventElapsedTime(&k1_msec, k1_start, k1_stop));
            k1_times.add(k1_msec / 1000.0); // seconds
            gpuErrchk(hipEventElapsedTime(&k2_msec, k2_start, k2_stop));
            k2_times.add(k2_msec / 1000.0); // seconds


            ////////////////////////// CLEANUP //////////////////////////


            gpuErrchk(hipEventDestroy(k1_start));
            gpuErrchk(hipEventDestroy(k1_stop));
            gpuErrchk(hipEventDestroy(k2_start));
            gpuErrchk(hipEventDestroy(k2_stop));
            gpuErrchk(hipFree(buf));
            gpuErrchk(hipFree(dummy_buf));
            gpuErrchk(hipFree(k2_dummy_sink));

        } /* >8 end epoch */

        // streams cleanup
        gpuErrchk(hipStreamDestroy(k1_stream));
        gpuErrchk(hipStreamDestroy(k2_stream));

        ////////////////////////// RESULTS PRINTING //////////////////////////

        double k1_dt = k1_times.value();
        double k1_dt90 = k1_times.getPercentile(0.90);
        double k1_dt95 = k1_times.getPercentile(0.95);
        double k1_dt99 = k1_times.getPercentile(0.99);

        // Calculate how many chunks are processed per TB per Iter (Logic from K2)
        // Note: 16 is sizeof(float4)
        double chunks_per_iter_per_tb = (double)K2_TPB / (k2_chunk_size / 16.0);
        // Total bytes = Iters * Datas * (Bytes per chunk) * (Chunks per iter per TB) * (Num TBs)
        // Note: don't multiply by XCD_NUM! we're preluding only single XCD
        double bytes = k2_iters * k2_n_datas * k2_chunk_size * chunks_per_iter_per_tb * k2_bpx;
        double bw_GBps = bytes / (k2_times.value() * 1e9);

        cout << setw(3) << k2_bpx << " "
             << setw(5) << clock << " "
             << setw(9) << k1_iters << " "
             << setw(9) << k2_iters << " "
             << fixed << setprecision(1)
             << setw(7) << (double)k1_dt / k1_iters * clock * 1000 * 1000 << " "
             << setw(7) << (double)k1_dt90 / k1_iters * clock * 1000 * 1000 << " "
             << setw(7) << (double)k1_dt95 / k1_iters * clock * 1000 * 1000 << " "
             << setw(7) << (double)k1_dt99 / k1_iters * clock * 1000 * 1000 << " "
             << setw(7) << bw_GBps << " "
             << "\n"
             << flush;

    } /* >8 end bpx */

    return 0;
}
