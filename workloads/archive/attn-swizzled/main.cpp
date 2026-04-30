// attention_swizzle_mapping.hip
// hipcc -O2 attention_swizzle_mapping.hip -o attn_swizzle
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define HIP_CHECK(cmd) do {                                  \
  hipError_t e = (cmd);                                      \
  if (e != hipSuccess) {                                     \
    fprintf(stderr, "HIP error %s:%d: %s\n",                 \
            __FILE__, __LINE__, hipGetErrorString(e));       \
    std::exit(1);                                            \
  }                                                          \
} while(0)

// Layout: Q,K,V,O are [B, H, S, D] contiguous row-major (D fastest).
__device__ __forceinline__ int idx_bhsd(int b, int h, int s, int d,
                                        int B, int H, int S, int D) {
  return ((b * H + h) * S + s) * D + d;
}

// Mapping modes
enum MapMode : int {
  NAIVE_BLOCK_FIRST = 0,   // iterate blocks then heads
  NAIVE_HEAD_FIRST  = 1,   // iterate heads then blocks, but round-robin blocks across XCDs (no confinement)
  SWIZZLED_HEAD_FIRST = 2  // Figure 11 logic: confine blocks of a head to one XCD region (logical)
};

// This function maps linear workgroup id "wid" into (batch, head, q_block).
// It matches the spirit of Figure 11: remap head/block indices to improve locality.
// NOTE: In real MI300X, "XCD assignment" is emergent from scheduler + CU mask. Here we only
// permute the execution order / block id -> (head, block) mapping.
__device__ __forceinline__
void map_wid_to_bhb(int wid,
                    int B, int Hq, int blocks_per_head,
                    int num_xcd,
                    MapMode mode,
                    int &batch, int &head, int &qblock)
{
  // Total workgroups = B * Hq * blocks_per_head
  // Base coordinates in "naive head-first" order: [batch][head][block]
  int wid_per_batch = wid % (Hq * blocks_per_head);
  batch = wid / (Hq * blocks_per_head);

  if (mode == NAIVE_HEAD_FIRST) {
    head   = wid_per_batch / blocks_per_head;
    qblock = wid_per_batch % blocks_per_head;
    return;
  }

  if (mode == NAIVE_BLOCK_FIRST) {
    // [batch][block][head]
    int blk = wid_per_batch / Hq;
    int hd  = wid_per_batch % Hq;
    head = hd;
    qblock = blk;
    return;
  }

  // SWIZZLED_HEAD_FIRST:
  // Adopt Figure 11-style swizzle:
  // - Partition heads across XCDs: heads_per_xcd = Hq / num_xcd (assume divisible for simplicity)
  // - chunk_size = num_xcd * blocks_per_head
  // - head_offset selects which head group
  // - block_offset selects qblock within a head, aligned by XCD lane
  int heads_per_xcd = Hq / num_xcd;
  int chunk_size = num_xcd * blocks_per_head;

  // Figure 11 uses wid_per_batch derived from program_id and batch.
  // Here wid_per_batch is already within [0, Hq*blocks_per_head).
  int xcd_lane = (wid_per_batch % num_xcd);                // "which XCD lane"
  int head_group = wid_per_batch / (num_xcd * blocks_per_head); // which set of heads across lanes

  head = xcd_lane * heads_per_xcd + head_group;
  int within_chunk = (wid_per_batch % chunk_size);
  qblock = within_chunk / num_xcd;

  // Clamp if Hq not divisible or tail exists (keep functional)
  if (head >= Hq) head = Hq - 1;
  if (qblock >= blocks_per_head) qblock = blocks_per_head - 1;
}

// A very naive attention forward for correctness + mapping experiments.
// Each workgroup computes one query BLOCK_M tile for one (batch, head).
// WARNING: O(S^2) per query row; keep S small for functional runs.
__global__ void attn_forward_naive(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ O,
                                  int B, int H, int S, int D,
                                  int BLOCK_M,
                                  int num_xcd,
                                  int mode_int)
{
  int wid = (int)blockIdx.x;
  MapMode mode = (MapMode)mode_int;

  int blocks_per_head = (S + BLOCK_M - 1) / BLOCK_M;
  int batch, head, qblock;
  map_wid_to_bhb(wid, B, H, blocks_per_head, num_xcd, mode, batch, head, qblock);

  int q_start = qblock * BLOCK_M;
  int q_end = q_start + BLOCK_M;
  if (q_end > S) q_end = S;

  // Single-thread reference implementation per workgroup (functional, not optimized).
  // You can later replace with FA2 tiling; the mapping logic remains.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    for (int qs = q_start; qs < q_end; qs++) {
      // Compute scores s_j = dot(Q[qs], K[j]) / sqrt(D)
      float inv_sqrt_d = 1.0f / sqrtf((float)D);

      // Softmax stable: find max
      float max_s = -1e30f;
      for (int ks = 0; ks < S; ks++) {
        float dot = 0.f;
        for (int d = 0; d < D; d++) {
          float qv = Q[idx_bhsd(batch, head, qs, d, B, H, S, D)];
          float kv = K[idx_bhsd(batch, head, ks, d, B, H, S, D)];
          dot += qv * kv;
        }
        float s = dot * inv_sqrt_d;
        if (s > max_s) max_s = s;
      }

      // Sum exp
      float sum = 0.f;
      for (int ks = 0; ks < S; ks++) {
        float dot = 0.f;
        for (int d = 0; d < D; d++) {
          float qv = Q[idx_bhsd(batch, head, qs, d, B, H, S, D)];
          float kv = K[idx_bhsd(batch, head, ks, d, B, H, S, D)];
          dot += qv * kv;
        }
        float s = dot * inv_sqrt_d;
        sum += __expf(s - max_s);
      }
      float inv_sum = 1.0f / sum;

      // Output O[qs, d] = sum_j softmax(s_j)*V[j,d]
      for (int d = 0; d < D; d++) {
        float acc = 0.f;
        for (int ks = 0; ks < S; ks++) {
          float dot = 0.f;
          for (int dd = 0; dd < D; dd++) {
            float qv = Q[idx_bhsd(batch, head, qs, dd, B, H, S, D)];
            float kv = K[idx_bhsd(batch, head, ks, dd, B, H, S, D)];
            dot += qv * kv;
          }
          float w = __expf(dot * inv_sqrt_d - max_s) * inv_sum;
          float vv = V[idx_bhsd(batch, head, ks, d, B, H, S, D)];
          acc += w * vv;
        }
        O[idx_bhsd(batch, head, qs, d, B, H, S, D)] = acc;
      }
    }
  }
}

static void fill_rand(std::vector<float>& v, float scale=0.01f) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-scale, scale);
  for (auto &x: v) x = dist(rng);
}

int main(int argc, char** argv)
{
  // Keep defaults small: this kernel is O(S^2).
  int B = (argc > 1) ? std::atoi(argv[1]) : 1;
  int H = (argc > 2) ? std::atoi(argv[2]) : 8;
  int S = (argc > 3) ? std::atoi(argv[3]) : 256;
  int D = (argc > 4) ? std::atoi(argv[4]) : 64;
  int BLOCK_M = (argc > 5) ? std::atoi(argv[5]) : 32;
  int num_xcd = (argc > 6) ? std::atoi(argv[6]) : 8;
  int mode = (argc > 7) ? std::atoi(argv[7]) : (int)SWIZZLED_HEAD_FIRST;

  if (H % num_xcd != 0) {
    printf("Warning: H %% num_xcd != 0; swizzle clamps head.\n");
  }

  size_t n = (size_t)B * H * S * D;
  printf("B=%d H=%d S=%d D=%d BLOCK_M=%d num_xcd=%d mode=%d  (total floats=%zu)\n",
         B,H,S,D,BLOCK_M,num_xcd,mode,n);

  std::vector<float> hQ(n), hK(n), hV(n), hO(n, 0.f);
  fill_rand(hQ); fill_rand(hK); fill_rand(hV);

  float *dQ=nullptr, *dK=nullptr, *dV=nullptr, *dO=nullptr;
  HIP_CHECK(hipMalloc(&dQ, sizeof(float)*n));
  HIP_CHECK(hipMalloc(&dK, sizeof(float)*n));
  HIP_CHECK(hipMalloc(&dV, sizeof(float)*n));
  HIP_CHECK(hipMalloc(&dO, sizeof(float)*n));
  HIP_CHECK(hipMemcpy(dQ, hQ.data(), sizeof(float)*n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, hK.data(), sizeof(float)*n, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, hV.data(), sizeof(float)*n, hipMemcpyHostToDevice));

  int blocks_per_head = (S + BLOCK_M - 1) / BLOCK_M;
  int grid = B * H * blocks_per_head;

  dim3 block(1,1,1); // single-thread workgroup for functional correctness
  dim3 gridDim(grid);

  hipLaunchKernelGGL(attn_forward_naive, gridDim, block, 0, 0,
                     dQ, dK, dV, dO, B, H, S, D, BLOCK_M, num_xcd, mode);
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy(hO.data(), dO, sizeof(float)*n, hipMemcpyDeviceToHost));
  printf("Done. O[0]=%f\n", hO[0]);

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
  return 0;
}
