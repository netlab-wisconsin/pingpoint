// determination of home node
// cache line access in shuffled order

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

constexpr int THREADS_PER_BLOCK = 8; // each thread loads 16B, so 8*16B=128B=1 cache line
constexpr int BLOCKS_NUM = 8;
constexpr int TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

__global__ void k(char *data, size_t size, int *access_order) {
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

    // access data array in uint4 * threads_per_block chunks
    // 16B * 8 = 128B = 1 cache line per thread block
    uint4 *data_u4 = (uint4*)data;
    int iter = size / sizeof(uint4) / THREADS_PER_BLOCK;
    uint4 tmp;

    // latency measurement
    uint32_t start = 0, end = 0;

    // pre-load boundaries to warm up cache / tlb
    for (int i = 0; i < iter; i+=32) {
        asm volatile(
            "flat_load_dwordx4 %0, %1\n\t"
            "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
            : "=v"(tmp)
            : "v"(&data_u4[i * THREADS_PER_BLOCK + tid])
            : "memory"
        );
    }

    // access in shuffled order
    for (int _i = 0; _i < iter; _i++) {
        int i = access_order[_i];

        for (size_t b = 0; b < BLOCKS_NUM; b++) {
            // each thread block access same data chunk in turn
            if (bid == b) {

                if (i % 32 == 0) {
                    printf("skip latency measurement for page boundary\n");
                } else {
                    start = __builtin_readcyclecounter();
                }

                asm volatile(
                    "flat_load_dwordx4 %0, %1\n\t"
                    "s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t"
                    : "=v"(tmp)
                    : "v"(&data_u4[i * THREADS_PER_BLOCK + tid])
                    : "memory"
                );

                if (i % 32 == 0) {
                    // printf("skip latency measurement for page boundary\n");
                } else {
                    end = __builtin_readcyclecounter();
                    printf("iter %d: (bid: %d, tid: %d) data_u4[%d] %u cycles\n", i, bid, tid, i * THREADS_PER_BLOCK + tid, end - start);
                }

            }
            grid.sync();
        }
    }
}

// observation: consecutive data are interleaved across iods in page granularity
// observation: within a page, each half-page is mapped to a hbm stack
int main() {
    size_t n_pages = 4; // you can change
    printf("bmk1_1: allocate %zu pages. on cache line granularity, compare load latency from different xcc\n", n_pages);

    size_t data_size = (4 * 1024 * n_pages); // 4KB page size, n_pages pages
    vector<char> h_data(data_size, 0);

    for (size_t i = 0; i < data_size; i++) {
        h_data[i] = static_cast<char>(i & 0xFF);
    }

    char *d_data;

    gpuErrchk(hipMalloc((void**)&d_data, data_size + 0x1000));
    d_data = (char*)(((uintptr_t)d_data & ~(0x0FFF)) + 0x1000); // page align

    gpuErrchk(hipMemcpy(d_data, h_data.data(), data_size, hipMemcpyHostToDevice));

    // create shuffled access order
    int iter = data_size / sizeof(uint4) / THREADS_PER_BLOCK;
    int access_order[iter];
    for (int i = 0; i < iter; i++) access_order[i] = i;
    for (int i = 0; i < iter; i++) std::swap(access_order[i], access_order[rand() % iter]);
    int *d_access_order;
    gpuErrchk(hipMalloc((void**)&d_access_order, sizeof(int) * iter));
    gpuErrchk(hipMemcpy(d_access_order, access_order, sizeof(int) * iter, hipMemcpyHostToDevice));

    void *kernel_args[] = {
        (void*)&d_data,
        (void*)&data_size,
        (void*)&d_access_order
    };

    gpuErrchk(hipLaunchCooperativeKernel(
        (void*)k,
        BLOCKS_NUM, THREADS_PER_BLOCK,
        kernel_args, 0, 0
    ));
    gpuErrchk(hipDeviceSynchronize());

    gpuErrchk(hipFree(d_data));
    gpuErrchk(hipFree(d_access_order));

    return 0;
}