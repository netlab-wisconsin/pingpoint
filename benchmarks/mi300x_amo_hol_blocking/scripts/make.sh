#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"

TPB_LIST=(128 256 512)
K1_MAX_TPB_LIST=(128 256 512)
K2_MAX_TPB_LIST=(128 256 512)

for t1 in "${TPB_LIST[@]}"; do
    for t2 in "${K1_MAX_TPB_LIST[@]}"; do
        for t3 in "${K2_MAX_TPB_LIST[@]}"; do
            # Ensure K1_MAX_TPB and K2_MAX_TPB are not greater than TPB
            if [ $t2 -gt $t1 ] || [ $t3 -gt $t1 ]; then
                continue
            fi
            make TPB=$t1 K1_MAX_TPB=$t2 K2_MAX_TPB=$t3
        done
        # additionally, make k1 only by setting k2 max tpb to 0
        make TPB=$t1 K1_MAX_TPB=$t2 K2_MAX_TPB=0
    done
done