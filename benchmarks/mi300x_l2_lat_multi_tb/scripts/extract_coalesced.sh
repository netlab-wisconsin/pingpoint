#!/bin/bash

OUT_DIR="./results/1210_coalesced"

BLOCKS_LIST="8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128"
WARPS_LIST=$(seq 1 16)
# REPEAT_LIST="4096 8192"
REPEAT_LIST="4096"

echo "b,w,clk"

for b in $BLOCKS_LIST; do
  for w in $WARPS_LIST; do
    for r in $REPEAT_LIST; do
        echo "Processing BLOCKS=$b WARPS=$w REPEAT=$r"
        file="${OUT_DIR}/l2_lat_coal_b_${b}_w_${w}_r_${r}.out"

        if [[ -f "$file" ]]; then
            clk=$(grep -oP '=\s*\K[0-9]+(?=\s*\(clk\))' "$file")
            echo "$b,$w,$clk" >> ${OUT_DIR}/summary_coalesced.csv
        else
            echo "Missing: $file"
        fi

    done
  done
done
