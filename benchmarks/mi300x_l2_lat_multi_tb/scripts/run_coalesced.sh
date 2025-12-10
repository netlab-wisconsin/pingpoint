#!/bin/bash

BIN_DIR="./bin"
OUT_DIR="./results/1209_coalesced"

mkdir -p "$OUT_DIR"

BLOCKS_LIST="8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128"
WARPS_LIST=$(seq 1 16)
REPEAT_LIST="4096 8192"

for b in $BLOCKS_LIST; do
  for w in $WARPS_LIST; do
    for r in $REPEAT_LIST; do

      exe="${BIN_DIR}/l2_lat_coal_b_${b}_w_${w}_r_${r}"
      out="${OUT_DIR}/l2_lat_coal_b_${b}_w_${w}_r_${r}.out"

      if [[ -x "$exe" ]]; then
        echo "Running b=${b}, w=${w}, r=${r}"
        "$exe" > "$out" 2>&1
        ((total++))
      else
        echo "SKIP missing binary: $exe"
      fi

    done
  done
done
