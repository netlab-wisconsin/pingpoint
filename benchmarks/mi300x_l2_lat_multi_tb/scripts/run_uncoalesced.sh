#!/bin/bash

BIN_DIR="./bin"
OUT_DIR="./results/1210_uncoalesced"

mkdir -p "$OUT_DIR"

BLOCKS=$(seq 8 8 128)
WARPS=$(seq 1 16)
STRIDES="2 4 8 16"
REPEATS="4096 8192"

for b in $BLOCKS; do
  for w in $WARPS; do
    for s in $STRIDES; do
      for r in $REPEATS; do

        exe="${BIN_DIR}/l2_lat_uncoal_b_${b}_w_${w}_s_${s}_r_${r}"
        out="${OUT_DIR}/l2_lat_uncoal_b_${b}_w_${w}_s_${s}_r_${r}.out"

        if [[ -x "$exe" ]]; then
          echo "Running BLOCKS=$b WARPS=$w STRIDE=$s REPEAT=$r"
          "$exe" > "$out" 2>&1
        else
          echo "SKIP: binary not found: $exe"
        fi

      done
    done
  done
done
