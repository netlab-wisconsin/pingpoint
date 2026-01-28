#pragma once

#include <hip/hip_runtime.h>

/*
  Token dispatch / permutation

  Inputs:
    X[T, d] row-major
    expert_id[T] in [0, E)
    E experts
    cap = capacity per expert in output layout
      output expert-major buffer: Xexp[E, cap, d], flattened
  Outputs:
    Xexp (expert-major)
    pos[T] = global slot in Xexp where token t was placed: pos[t] = e*cap + local_idx

  This is the classic dispatch step used in MoE to group tokens by expert.
*/
__global__ void moe_dispatch(const float* __restrict__ X,
                             const int* __restrict__ expert_id,
                             float* __restrict__ Xexp,
                             int* __restrict__ pos,
                             int* __restrict__ counters, // [E], init to 0
                             int T, int d, int E, int cap)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;

  int e = expert_id[t];
  if (e < 0 || e >= E) return;

  int local = atomicAdd(&counters[e], 1);
  if (local >= cap) {
    // overflow: drop token (or clamp). For functional code, just mark invalid.
    pos[t] = -1;
    return;
  }

  int slot = e * cap + local;
  pos[t] = slot;

  // copy d floats
  const float* src = X + t * d;
  // The destination of each write is Xexp[slot,*]. If Xexp is placed remote relative to the producer XCDs, the traffic becomes IO–IO intensive.
  // If Xexp is placed such that many writes land on a small set of IO–MEM partitions (address-directed), it can trigger IO–MEM skew/backpressure.
  float* dst = Xexp + slot * d;
  for (int j = 0; j < d; j++) dst[j] = src[j]; // This writes the d-dim activation for token t into expert e’s contiguous region
}


/*
  Inverse gather:
    Given expert-major buffer Xexp and pos[T], reconstruct Y[T,d] in original token order.
*/
__global__ void moe_gather(const float* __restrict__ Xexp,
                           const int* __restrict__ pos,
                           float* __restrict__ Y,
                           int T, int d, int cap /*unused but kept for symmetry*/)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;

  int slot = pos[t];
  if (slot < 0) {
    // if dropped, zero out
    float* dst = Y + t * d;
    for (int j = 0; j < d; j++) dst[j] = 0.f;
    return;
  }

  const float* src = Xexp + slot * d;
  float* dst = Y + t * d;
  for (int j = 0; j < d; j++) dst[j] = src[j];
}