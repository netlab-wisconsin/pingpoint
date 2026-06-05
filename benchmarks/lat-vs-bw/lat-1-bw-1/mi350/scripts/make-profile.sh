#!/bin/bash

# for i in {1..64}; do
for i in {1..128}; do
    K2_BPX_MIN=${i}
    K2_BPX_MAX=${i}

    make -B -f Makefile.profile K1_PINNED_XCD=0 K1_PINNED_CC=0 K2_PINNED_XCD=0 K2_PINNED_CC=0 \
        K2_BPX_MIN=${K2_BPX_MIN} K2_BPX_MAX=${K2_BPX_MAX} K2_TPB=512 \
        BIN_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/lat-1-bw-1/gfx950/bin
done
