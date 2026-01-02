#!/bin/bash
set -euo pipefail

mkdir -p bin

for k1_xcd in 6; do
  for k1_hbm in $(seq 0 2 6); do
    for k2_xcd in $(seq 1 2 7); do
      for k2_hbm in $(seq 1 2 7); do

        if [[ $k1_xcd -eq 6 && $k1_hbm -eq 0 && $k2_xcd -eq 1 ]]; then
          continue
        fi

        if [[ $k1_xcd -eq 6 && $k1_hbm -eq 0 && $k2_xcd -eq 3 && $k2_hbm -eq 1 ]]; then
          continue
        fi

        if [[ $k1_xcd -eq 6 && $k1_hbm -eq 0 && $k2_xcd -eq 3 && $k2_hbm -eq 3 ]]; then
          continue
        fi

        if [[ $k1_xcd -eq 6 && $k1_hbm -eq 0 && $k2_xcd -eq 3 && $k2_hbm -eq 5 ]]; then
          continue
        fi



        echo "Running K1_XCD=$k1_xcd K1_HBM=$k1_hbm K2_XCD=$k2_xcd K2_HBM=$k2_hbm"
        ./bin/stateful_routing_${k1_xcd}${k1_hbm}${k2_xcd}${k2_hbm} |& tee results/1230/all/${k1_xcd}${k1_hbm}${k2_xcd}${k2_hbm}.out || true
      done
    done
  done
done
