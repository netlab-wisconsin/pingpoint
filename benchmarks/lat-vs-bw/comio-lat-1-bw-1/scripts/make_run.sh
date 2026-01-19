#!/bin/bash

K2_BPX_MIN=1
K2_BPX_MAX=160

# 4 combinates of lat: (local,remote) x bw: (local,remote)
COMBINATIONS=(
    "0 1"  # Local Lat, Local BW
    "0 3"  # Local Lat, Remote BW
    "2 1"  # Remote Lat, Local BW
    "2 7"  # Remote Lat, Remote BW
)

for combo in "${COMBINATIONS[@]}"; do
    # Split the combination into K1_PINNED_HBM and K2_PINNED_HBM
    K1_PINNED_HBM=$(echo $combo | cut -d' ' -f1)
    K2_PINNED_HBM=$(echo $combo | cut -d' ' -f2)

    make -f Makefile.run K1_PINNED_HBM=${K1_PINNED_HBM} K2_PINNED_HBM=${K2_PINNED_HBM} \
        K2_BPX_MIN=${K2_BPX_MIN} K2_BPX_MAX=${K2_BPX_MAX}
done