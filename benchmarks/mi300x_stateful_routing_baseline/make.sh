#!/bin/bash
set -euo pipefail

mkdir -p bin

for k1_xcd in $(seq 0 2 6); do
  for k1_hbm in $(seq 0 2 6); do
    for k2_xcd in $(seq 1 2 7); do
      for k2_hbm in $(seq 1 2 7); do
        echo "Building K1_XCD=$k1_xcd K1_HBM=$k1_hbm K2_XCD=$k2_xcd K2_HBM=$k2_hbm"
        make -B \
          K1_PINNED_XCD="$k1_xcd" \
          K1_PINNED_HBM="$k1_hbm" \
          K2_PINNED_XCD="$k2_xcd" \
          K2_PINNED_HBM="$k2_hbm"
      done
    done
  done
done
