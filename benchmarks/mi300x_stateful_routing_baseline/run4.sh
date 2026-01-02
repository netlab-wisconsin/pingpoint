#!/bin/bash
set -euo pipefail

export HIP_VISIBLE_DEVICES="0"

for k1_xcd in 2; do
  for k1_hbm in $(seq 4 2 6); do
    for k2_xcd in $(seq 1 2 7); do
      for k2_hbm in $(seq 1 2 7); do
        echo "Running K1_XCD=$k1_xcd K1_HBM=$k1_hbm K2_XCD=$k2_xcd K2_HBM=$k2_hbm"
        ./bin/stateful_routing_${k1_xcd}${k1_hbm}${k2_xcd}${k2_hbm} |& tee results/1230/again/${k1_xcd}${k1_hbm}${k2_xcd}${k2_hbm}.out || true
      done
    done
  done
done