// determination of llc slice

#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

using namespace std;
namespace cg = cooperative_groups;

// GPU error check
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true){
    if (code != hipSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

constexpr int THREADS_PER_BLOCK = 8;
constexpr int BLOCKS_NUM = 8;
constexpr int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

__global__ void k(char *data, uint32_t **d_cycles, size_t size) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int uid = bid * blockDim.x + tid;

    cg::grid_group grid = cg::this_grid();

    // print (xcc_id,cu_id) of each block
    if (tid == 0) {
        uint32_t cu_id, xcc_id;
        asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=r"(cu_id));
        asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));
        printf("bid %d: (xcc_id: %u, cu_id: %u)\n", bid, xcc_id, cu_id);
    }

    // access data array in cacheline chunk (= uint4 * threads_per_block)
    // 16B * 8 = 128B = 1 cache line per thread block
    uint4 *data_u4 = (uint4*)data;
    int num_ccls = size / 128; // cache line idx in data array
    assert(num_ccls == size / sizeof(uint4) / THREADS_PER_BLOCK);
    int num_pages = size / 4096;

    if (bid == 0 && tid == 0) printf("total num_ccls = %d\n", num_ccls);
    uint4 tmp;

    // latency measurement
    uint32_t start = 0, end = 0;

    for (int cid = 0; cid < num_ccls; cid++) {
        for (size_t b = 0; b < BLOCKS_NUM; b++) {
            // each thread block access same data chunk in turn
            if (bid == b) {

                if (cid % 32 == 0) {
                    printf("skip latency measurement for page boundary\n");
                } else {
                    start = __builtin_readcyclecounter();
                }

                asm volatile(
                    "flat_load_dwordx4 %0, %1\n\t"
                    "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                    : "=v"(tmp)
                    : "v"(&data_u4[cid * THREADS_PER_BLOCK + tid])
                    : "memory"
                );

                if (cid % 32 == 0) {
                    // printf("skip latency measurement for page boundary\n");
                } else {
                    end = __builtin_readcyclecounter();
                    printf("cid %d: (bid: %d, tid: %d) data_u4[%d] %u cycles\n", cid, bid, tid, cid * THREADS_PER_BLOCK + tid, end - start);
                    if (tid == 0) {
                        // only tid 0 writes
                        d_cycles[bid][cid] = end - start;
                    }
                }

            }
            grid.sync();
        }
    }

    uint32_t min_bid_per_cid[num_ccls];
    for (int cid = 0; cid < num_ccls; cid++) {
        // get bid with min d_cycles[bid][cid]
        uint32_t min_cycles = 0xFFFFFFFF;;
        int min_bid = -1;
        for (size_t b = 0; b < BLOCKS_NUM; b++) {
            uint32_t cycles = d_cycles[b][cid];
            if (cycles < min_cycles) {
                min_cycles = cycles;
                min_bid = b;
            }
        }
        min_bid_per_cid[cid] = min_bid;

        int pg_idx = cid / 32; // page index
        if (bid == 0 && tid == 0) {
            printf("pid %d, cid %d: data_u4[%d,..,%d] home @bid %d\n", pg_idx, cid, cid * THREADS_PER_BLOCK, (cid+1) * THREADS_PER_BLOCK-1, min_bid);
        }
    }
    grid.sync();

    // invalidate all the cached lines before measurement
    // use buffer_inv sc1 sc0
    for (int cid = 0; cid < num_ccls; cid++) {
        asm volatile(
            // invalidate cid
            "buffer_inv sc1 sc0\n\t"
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            :
            : "v"(&data_u4[cid * THREADS_PER_BLOCK + tid])
            : "memory"
        );
    }
    grid.sync();

    // start measurement
    for (int cid = 0; cid < num_ccls; cid++) {
        int pg_idx = cid / 32;
        int min_bid = min_bid_per_cid[cid];

        for (int p = 0; p < num_pages; p++) {
            for (int pcid = 0; pcid < 32; pcid++) {
                if (min_bid == bid) {

                    // wait for others to invalidate cid
                    grid.sync();

                    // home bid always reads from cid
                    start = __builtin_readcyclecounter();
                    asm volatile(
                        "flat_load_dwordx4 %0, %1\n\t"
                        "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                        : "=v"(tmp)
                        : "v"(&data_u4[cid * THREADS_PER_BLOCK + tid])
                        : "memory"
                    );
                    end = __builtin_readcyclecounter();

                    grid.sync(); // unnecessary, just for separating printing region
                    printf("final read: (bid: %d, tid: %d) (pid: %d, cid: %d) (p: %d, pcid: %d) data_u4[%d] home @bid %d %u cycles\n", bid, tid, pg_idx, cid % 32, p, pcid, cid * THREADS_PER_BLOCK + tid, min_bid, end - start);

                    // grid.sync(); // for others to write something into cid
                } else {

                    // invalidate cid
                    tmp.x = tmp.y = tmp.z = tmp.w = 0xDEADBEEF;
                    data_u4[cid * THREADS_PER_BLOCK + tid] = tmp;
                    grid.sync();

                    // other bid writes to the same cache line within the page of this cache line
                    size_t ptr = ((p * 32 + pcid) * THREADS_PER_BLOCK + tid) % (size / sizeof(uint4));
                    tmp.x = tmp.y = tmp.z = tmp.w = 0xDEADBEEF;
                    data_u4[ptr] = tmp;

                    grid.sync(); // unnecessary, just for separating printing region
                    printf("hammering write: (bid: %d, tid: %d) (p: %d, pcid: %d) data_u4[%zu]\n", bid, tid, p, pcid, ptr);

                    // grid.sync(); // write something into cid to make bid reach llc again
                    // tmp.x = tmp.y = tmp.z = tmp.w = 0xDEADBEEF;
                    // data_u4[cid * THREADS_PER_BLOCK + tid] = tmp;
                    // asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");
                }
                grid.sync();
            }
        }
    }
}

int main() {
    size_t n_pages = 4; // you can change
    printf("bmk2: allocate %zu pages. on cache line granularity, compare load latency from home iod's xcc with hammering writes from different xccs on other cache lines from same home iod across pages to get llc slice sharing sets\n", n_pages);

    size_t data_size = (4 * 1024 * n_pages); // 4KB page size, n_pages pages
    vector<char> h_data(data_size, 0);

    for (size_t i = 0; i < data_size; i++) {
        h_data[i] = static_cast<char>(i & 0xFF);
    }

    char *d_data;

    gpuErrchk(hipMalloc((void**)&d_data, data_size + 0x1000));
    d_data = (char*)(((uintptr_t)d_data & ~(0x0FFF)) + 0x1000); // page align

    gpuErrchk(hipMemcpy(d_data, h_data.data(), data_size, hipMemcpyHostToDevice));

    uint32_t **d_cycles; // per-d_data, per-block cycles array (only tid=0 writes)
    gpuErrchk(hipMalloc((void**)&d_cycles, sizeof(uint32_t*) * BLOCKS_NUM));
    for (int b = 0; b < BLOCKS_NUM; b++) {
        gpuErrchk(hipMalloc((void**)&d_cycles[b], sizeof(uint32_t) * (data_size / sizeof(uint4) / THREADS_PER_BLOCK)));
    }
    printf("allocated d_cycles array of size %zu\n", data_size / sizeof(uint4) / THREADS_PER_BLOCK);

    void *kernel_args[] = {
        (void*)&d_data,
        (void*)&d_cycles,
        (void*)&data_size
    };

    gpuErrchk(hipLaunchCooperativeKernel(
        (void*)k,
        BLOCKS_NUM, THREADS_PER_BLOCK,
        kernel_args, 0, 0
    ));
    gpuErrchk(hipDeviceSynchronize());

    gpuErrchk(hipFree(d_data));

    return 0;
}