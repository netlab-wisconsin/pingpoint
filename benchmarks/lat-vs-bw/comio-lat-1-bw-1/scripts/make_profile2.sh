#!/bin/bash

K2_BPX_PROF_START=1
K2_BPX_PROF_END=160

# 4 combinates of lat: (local,remote) x bw: (local,remote)
COMBINATIONS=(
    # "0 1"  # Local Lat, Local BW
    "0 3"  # Local Lat, Remote BW
    # "2 1"  # Remote Lat, Local BW
    # "2 7"  # Remote Lat, Remote BW
)

for combo in "${COMBINATIONS[@]}"; do
    # Split the combination into K1_PINNED_HBM and K2_PINNED_HBM
    K1_PINNED_HBM=$(echo $combo | cut -d' ' -f1)
    K2_PINNED_HBM=$(echo $combo | cut -d' ' -f2)

    for i in $(seq "$K2_BPX_PROF_START" "$K2_BPX_PROF_END"); do
        make -f Makefile.profile K1_PINNED_HBM=${K1_PINNED_HBM} K2_PINNED_HBM=${K2_PINNED_HBM} \
            K2_BPX_MIN=${i} K2_BPX_MAX=${i} # Compile with same K2_BPX_MIN and K2_BPX_MAX
    done
done