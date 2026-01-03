#!/bin/bash

# Run makefile with different make variables

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"

K2_PINNED_XCD_LIST=(1 3 5 7)
K2_PINNED_HBM_LIST=(1 3 5 7)

K2_BPX_MIN=2    # Must be > 1.
K2_BPX_MAX=160

K2_TPB_LIST=(128 256 512)

for xcd in "${K2_PINNED_XCD_LIST[@]}"; do
    for hbm in "${K2_PINNED_HBM_LIST[@]}"; do
        for tpb in "${K2_TPB_LIST[@]}"; do
            make K2_PINNED_XCD=$xcd K2_PINNED_HBM=$hbm K2_BPX_MIN=$K2_BPX_MIN K2_BPX_MAX=$K2_BPX_MAX K2_TPB=$tpb
        done
    done
done