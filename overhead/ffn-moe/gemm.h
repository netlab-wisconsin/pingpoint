#pragma once

#include <hip/hip_runtime.h>

// --- Add these kernels (per-expert FFN) ---

// tmp[e, cap, hidden] = Xexp[e, cap, d] * W1[e, d, hidden]
__global__ void expert_gemm1_naive(const float* __restrict__ Xexp,   // [E*cap, d]
                                   const float* __restrict__ W1,     // [E, d, hidden]
                                   float* __restrict__ Tmp,          // [E*cap, hidden]
                                   const int* __restrict__ cnt,      // [E]
                                   int E, int cap, int d, int hidden)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x; // hidden col
  int t = blockIdx.y * blockDim.y + threadIdx.y; // local token within cap
  int e = blockIdx.z;                            // expert
  if (e >= E || t >= cap || j >= hidden) return;

  int ne = cnt[e];
  if (t >= ne) {
    // optional: zero padding
    Tmp[(e * cap + t) * hidden + j] = 0.f;
    return;
  }

  const float* x = Xexp + (e * cap + t) * d;
  const float* w = W1 + ((size_t)e * d * hidden); // base of expert e
  float acc = 0.f;
  for (int k = 0; k < d; k++) {
    acc += x[k] * w[(size_t)k * hidden + j];
  }
  Tmp[(e * cap + t) * hidden + j] = acc;
}

__device__ __forceinline__ float relu(float x) { return x > 0.f ? x : 0.f; }

// ReLU in-place on Tmp for only valid tokens (t < cnt[e])
__global__ void relu_tmp_expert(float* __restrict__ Tmp,     // [E*cap, hidden]
                                const int* __restrict__ cnt,// [E]
                                int E, int cap, int hidden)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x; // over E*cap*hidden
  int total = E * cap * hidden;
  if (idx >= total) return;

  int h = idx % hidden;
  int tc = idx / hidden;       // in [0, E*cap)
  int t  = tc % cap;
  int e  = tc / cap;

  int ne = cnt[e];
  if (t >= ne) return; // skip padding slots

  Tmp[idx] = relu(Tmp[idx]);
}

// Yexp[e, cap, d] = Tmp[e, cap, hidden] * W2[e, hidden, d]
__global__ void expert_gemm2_naive(const float* __restrict__ Tmp,   // [E*cap, hidden]
                                   const float* __restrict__ W2,    // [E, hidden, d]
                                   float* __restrict__ Yexp,        // [E*cap, d]
                                   const int* __restrict__ cnt,     // [E]
                                   int E, int cap, int d, int hidden)
{
  int j = blockIdx.x * blockDim.x + threadIdx.x; // output dim
  int t = blockIdx.y * blockDim.y + threadIdx.y; // local token within cap
  int e = blockIdx.z;                            // expert
  if (e >= E || t >= cap || j >= d) return;

  int ne = cnt[e];
  if (t >= ne) {
    Yexp[(e * cap + t) * d + j] = 0.f;
    return;
  }

  const float* tmp = Tmp + (e * cap + t) * hidden;
  const float* w = W2 + ((size_t)e * hidden * d);
  float acc = 0.f;
  for (int k = 0; k < hidden; k++) {
    acc += tmp[k] * w[(size_t)k * d + j];
  }
  Yexp[(e * cap + t) * d + j] = acc;
}
