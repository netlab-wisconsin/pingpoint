#pragma once

#include <hip/hip_runtime.h>

#include "../test/main.h"
#include "../test/k1.h"
#include "../test/ppnt.h"

namespace dbq {

struct HotReadUpdateArgs {
    k1::dtype* table;
    int table_len;
    int hot_read_idx;
    int hot_update_idx;
    int reader_xcd;
    int writer_xcd;
    int iters;
    k1::dtype update_delta;
    k1::dtype* sink;
};

struct HotReadUpdateTargetFn {
    __device__ __forceinline__
    void operator()(const HotReadUpdateArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const
    {
        const int n_tbs_in_xcd = gridDimX / XCD_NUM;
        const int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        const int logical_grid_size = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;
        const int logical_uid = logical_bid * blockDimX + tid;
        const int logical_total_threads = logical_grid_size * blockDimX;

        uint32_t xcc_id;
        asm volatile ("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=r"(xcc_id));

        if ((int)xcc_id != a->reader_xcd && (int)xcc_id != a->writer_xcd) return;
        if (logical_total_threads <= 0 || logical_uid >= logical_total_threads) return;

        k1::dtype acc = 0;
        for (int it = logical_uid; it < a->iters; it += logical_total_threads) {
            if ((int)xcc_id == a->reader_xcd) {
                acc += a->table[a->hot_read_idx];
            }
            if ((int)xcc_id == a->writer_xcd) {
                atomicAdd((unsigned long long*)&a->table[a->hot_update_idx],
                          (unsigned long long)a->update_delta);
            }
        }
        if ((int)xcc_id == a->reader_xcd) {
            a->sink[logical_uid] = acc;
        }
    }
};

struct PointLookupArgs {
    const k1::dtype* table;
    const int* query_indices;
    int n_queries;
    k1::dtype* out;
};

struct PointLookupTargetFn {
    __device__ __forceinline__
    void operator()(const PointLookupArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const
    {
        const int n_tbs_in_xcd = gridDimX / XCD_NUM;
        const int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        const int logical_grid_size = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;
        const int logical_uid = logical_bid * blockDimX + tid;
        const int logical_total_threads = logical_grid_size * blockDimX;

        for (int q = logical_uid; q < a->n_queries; q += logical_total_threads) {
            a->out[q] = a->table[a->query_indices[q]];
        }
    }
};

struct VectorDistanceArgs {
    const float* vectors;
    const float* query;
    float* dists;
    int n_rows;
    int dim;
};

struct VectorDistanceTargetFn {
    __device__ __forceinline__
    void operator()(const VectorDistanceArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const
    {
        const int n_tbs_in_xcd = gridDimX / XCD_NUM;
        const int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        const int logical_grid_size = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;
        const int logical_uid = logical_bid * blockDimX + tid;
        const int logical_total_threads = logical_grid_size * blockDimX;

        for (int row = logical_uid; row < a->n_rows; row += logical_total_threads) {
            const float* vec = a->vectors + (size_t)row * a->dim;
            float acc = 0.f;
            for (int j = 0; j < a->dim; j++) {
                const float d = vec[j] - a->query[j];
                acc += d * d;
            }
            a->dists[row] = acc;
        }
    }
};

struct RangeFilterArgs {
    const k1::dtype* table;
    int table_len;
    k1::dtype low;
    k1::dtype high;
    unsigned int* count_out;
};

struct RangeFilterTargetFn {
    __device__ __forceinline__
    void operator()(const RangeFilterArgs* __restrict__ a,
                    int bid, int tid, int gridDimX, int blockDimX, int n_ppnt_tbs_in_xcd) const
    {
        const int n_tbs_in_xcd = gridDimX / XCD_NUM;
        const int logical_bid = ppnt::physical_to_logical_bid(bid, n_tbs_in_xcd, n_ppnt_tbs_in_xcd);
        const int logical_grid_size = (n_tbs_in_xcd - n_ppnt_tbs_in_xcd) * XCD_NUM;
        const int logical_uid = logical_bid * blockDimX + tid;
        const int logical_total_threads = logical_grid_size * blockDimX;

        unsigned int local = 0;
        for (int i = logical_uid; i < a->table_len; i += logical_total_threads) {
            const k1::dtype v = a->table[i];
            if (v >= a->low && v <= a->high) local++;
        }
        if (local) atomicAdd(a->count_out, local);
    }
};

} // namespace dbq
