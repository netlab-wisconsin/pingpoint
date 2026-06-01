#!/bin/bash

SUFFIX="053126"

# to the different hbm
for i in {1..160}; do
    make -f Makefile.profile K1_PINNED_XCD=0 K2_PINNED_XCD=0 K1_PINNED_HBM=0 K2_PINNED_HBM=1 \
        K2_BPX_MIN=${i} K2_BPX_MAX=${i} K2_TPB=1024 \
        BIN_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/lat-1-bw-1/bin-${SUFFIX}
done

# to the same hbm
# for i in {1..160}; do
#     make -f Makefile.profile K1_PINNED_XCD=0 K2_PINNED_XCD=0 K1_PINNED_HBM=0 K2_PINNED_HBM=0 \
#         K2_BPX_MIN=${i} K2_BPX_MAX=${i} K2_TPB=1024 \
#         BIN_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/lat-1-bw-1/bin-${SUFFIX}
# done