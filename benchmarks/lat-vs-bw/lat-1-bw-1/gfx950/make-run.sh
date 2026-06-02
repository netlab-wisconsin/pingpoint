#!/bin/bash

K2_BPX_MIN=1
K2_BPX_MAX=128

make -B -f Makefile.run K1_PINNED_XCD=0 K1_PINNED_HBM=0 K2_PINNED_XCD=0 K2_PINNED_HBM=0 \
    K2_BPX_MIN=${K2_BPX_MIN} K2_BPX_MAX=${K2_BPX_MAX} K2_TPB=1024 \
    BIN_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/lat-1-bw-1/gfx950/bin