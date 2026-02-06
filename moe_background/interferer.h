#pragma once

#include <hip/hip_runtime.h>

// -------------------- Background "attention-like" interferer --------------------
// Not correctness-focused: just streams through QKV/Out buffers with some FMAs to keep pipes busy.

__global__ void bg_attn_interferer(const float* __restrict__ X,
                                  const float* __restrict__ Wqkv,
                                  float* __restrict__ QKV,
                                  int T, int d,
                                  const int* __restrict__ stop_flag,
                                  int iters_per_check)
{
    const uint64_t tid    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;

    const uint64_t Td   = (uint64_t)T * (uint64_t)d;
    const uint64_t Wsz  = (uint64_t)d * (uint64_t)(3 * (uint64_t)d);     // d*(3d)
    const uint64_t Qsz  = (uint64_t)T * (uint64_t)(3 * (uint64_t)d);     // T*(3d)

    float acc = 0.0f;

    while (true) {
        for (int it = 0; it < iters_per_check; ++it) {
            for (uint64_t i = tid; i < Td; i += stride) {
                float x = X[i];
                float w = Wqkv[(i * 13ull) % Wsz];
                acc = fmaf(x, w, acc);

                uint64_t o = (i * 7ull) % Qsz;
                QKV[o] = acc;
            }
        }
        if (__ldg(stop_flag) != 0) break;
    }
    if (tid == 0) QKV[0] = acc;
}

