#!/bin/bash

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_bw_contention

out="${BASE_DIR}/results/112825_tornado_save/tornado_hop1_bw.csv"
rm -f "$out"

for bpx in $(seq 1 275); do
    file="${BASE_DIR}/results/112825_tornado_save/tornado_bpx_${bpx}_hop_1.out"
    if [[ ! -f "$file" ]]; then
        echo "Warning: $file not found" >&2
        continue
    fi

    pws=()
    for x in $(seq 0 7); do
        # Example line:
        # xcd 0: peak window bw 13.81 GB/s
        val=$(grep "xcd $x: peak window bw" "$file" | awk '{print $6}')
        pws+=("$val")
    done

    # Output: bpx,pw0,...,pw7
    echo "$bpx,${pws[0]},${pws[1]},${pws[2]},${pws[3]},${pws[4]},${pws[5]},${pws[6]},${pws[7]}" >> "$out"
done
