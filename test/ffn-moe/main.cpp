#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

#include "dispatch_gather.h"
#include "gemm.h"
#include "../main.h"

#define DEBUG_LEVEL 2

/*
  Utility: initialize host vectors with random floats / random expert ids.
*/
static void fill_random(std::vector<float> &v, float scale = 0.01f)
{
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto &x : v)
        x = dist(rng);
}

static void fill_expert_ids(std::vector<int> &ids, int E)
{
    std::mt19937 rng(456);

#if 1
    // imbalanced token distribution
    double S = 0.9; // set among {0.25,0.50,0.75,0.99} for the token fraction of hot expert
    int e_hot = 0; // index of hot expert
    std::vector<double> p(E, (1.0 - S) / (double)(E - 1));
    p[e_hot] = S;
    std::discrete_distribution<int> dist(p.begin(), p.end());
#else
    // uniform token distribution
    std::uniform_int_distribution<int> dist(0, E - 1);
#endif
    for (auto &x : ids)
        x = dist(rng) % E;

#if DEBUG_LEVEL >= 1
    // print distribution
    std::vector<int> counts(E, 0);
    for (auto x : ids)
        counts[x]++;
    cout << "Expert token distribution: ";
    for (int e = 0; e < E; e++) cout << counts[e] << " ";
    cout << "\n" << flush;
#endif
}

int main(int argc, char **argv)
{
    // Minimal sizes for a functional run; bump these as needed.
    const int T = (argc > 1) ? std::atoi(argv[1]) : 2048; // tokens
    const int d = (argc > 2) ? std::atoi(argv[2]) : 2048; // model dim
    const int E = (argc > 3) ? std::atoi(argv[3]) : 8;    // experts
    const int hidden = 4 * d;

#if 1
    const int cap = (T + 64 - 1) / 64; // an expert can take all tokens in the worst case
#else
    const int cap = (T + E - 1) / E + 64; // add headroom to avoid overflow
#endif
    printf("T=%d d=%d hidden=%d E=%d cap=%d\n", T, d, hidden, E, cap);

    hipStream_t stream;
    gpuErrchk(hipStreamCreate(&stream));

    // Host buffers
    std::vector<float> h_X(T * d);
    std::vector<int> h_eid(T);
    fill_random(h_X);
    fill_expert_ids(h_eid, E);

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

    // ---- Dispatch: X[T,d] -> Xexp[E,cap,d] ----
    dim3 blockD(256);
    dim3 gridD((T + blockD.x - 1) / blockD.x); // #tokens/tpb
#if DEBUG_LEVEL >= 1
    printf("dispatch: gridD(%d,%d,%d) blockD(%d,%d,%d)\n",
           gridD.x, gridD.y, gridD.z,
           blockD.x, blockD.y, blockD.z);
#endif
    hipLaunchKernelGGL(moe_dispatch, gridD, blockD, 0, stream,
                       d_X, d_eid, d_Xexp, d_pos, d_cnt, T, d, E, cap);

    // --- Allocate per-expert weights ---
    std::vector<float> h_W1((size_t)E * d * hidden);
    std::vector<float> h_W2((size_t)E * hidden * d);
    fill_random(h_W1);
    fill_random(h_W2);

    float *d_W1=nullptr, *d_W2=nullptr, *d_Tmp=nullptr, *d_Yexp=nullptr;
    gpuErrchk(hipMalloc(&d_W1, sizeof(float) * h_W1.size()));
    gpuErrchk(hipMalloc(&d_W2, sizeof(float) * h_W2.size()));
    gpuErrchk(hipMemcpyAsync(d_W1, h_W1.data(), sizeof(float) * h_W1.size(), hipMemcpyHostToDevice, stream));
    gpuErrchk(hipMemcpyAsync(d_W2, h_W2.data(), sizeof(float) * h_W2.size(), hipMemcpyHostToDevice, stream));

    // Tmp: [E*cap, hidden], Yexp: [E*cap, d]
    gpuErrchk(hipMalloc(&d_Tmp,  sizeof(float) * (size_t)E * cap * hidden));
    gpuErrchk(hipMalloc(&d_Yexp, sizeof(float) * (size_t)E * cap * d));

    // --- Launch per-expert GEMM1 ---
    dim3 block1(16, 8, 1);
    dim3 grid1((hidden + block1.x - 1) / block1.x,
            (cap    + block1.y - 1) / block1.y,
            E);
    hipLaunchKernelGGL(expert_gemm1_naive, grid1, block1, 0, stream,
                    d_Xexp, d_W1, d_Tmp, d_cnt, E, cap, d, hidden);

    // --- ReLU Tmp ---
    int total_tmp = E * cap * hidden;
    dim3 blockA(256);
    dim3 gridA((total_tmp + blockA.x - 1) / blockA.x);
    hipLaunchKernelGGL(relu_tmp_expert, gridA, blockA, 0, stream,
                    d_Tmp, d_cnt, E, cap, hidden);

    // --- Launch per-expert GEMM2 ---
    dim3 block2(16, 8, 1);
    dim3 grid2((d   + block2.x - 1) / block2.x,
            (cap + block2.y - 1) / block2.y,
            E);
    hipLaunchKernelGGL(expert_gemm2_naive, grid2, block2, 0, stream,
                    d_Tmp, d_W2, d_Yexp, d_cnt, E, cap, d, hidden);

    // --- Combine back to token order ---
    // For top-1 routing, your prior moe_gather is fine:
    hipLaunchKernelGGL(moe_gather, gridD, blockD, 0, stream,
                    d_Yexp, d_pos, d_Y, T, d, cap);

    // Cleanup later: d_W1, d_W2, d_Tmp, d_Yexp

}
