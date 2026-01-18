#!/bin/bash

PROF=1

# Explicitly define the combinations
combinations=(
  "0 1"
  "0 3"
  "2 1"
  "2 7"
)

for combo in "${combinations[@]}"; do
  # Split the combination into K1_PINNED_HBM and K2_PINNED_HBM
  K1_PINNED_HBM=$(echo $combo | cut -d' ' -f1)
  K2_PINNED_HBM=$(echo $combo | cut -d' ' -f2)

  for i in $(seq 1 160); do
    make -f Makefile.profile K1_PINNED_HBM=${K1_PINNED_HBM} K2_PINNED_HBM=${K2_PINNED_HBM} K2_BPX_MIN=${i} K2_BPX_MAX=${i} PROFILE=${PROF}
  done
done