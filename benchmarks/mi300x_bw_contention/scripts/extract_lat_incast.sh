#!/bin/bash

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_bw_contention

for tgt in $(seq 0 7); do
    out="${BASE_DIR}/results/112825_incast_save/incast_tgt_${tgt}_lat.csv"
    rm -f "$out"

    for bpx in $(seq 1 275); do
        file="${BASE_DIR}/results/112825_incast_save/incast_bpx_${bpx}_tgt_${tgt}.out"
        if [[ ! -f "$file" ]]; then
            echo "Warning: $file not found" >&2
            continue
        fi

        # Extract integer latencies for xcd0..xcd7
        lats=()
        for x in $(seq 0 7); do
            # Example line:
            # xcd 0: avg lat(16B) 622.84 cycles
            val=$(grep "xcd $x: avg lat(16B)" "$file" | awk '{print int($5)}')
            lats+=("$val")
        done

        # Output line: bpx,lat0,...,lat7
        echo "$bpx,${lats[0]},${lats[1]},${lats[2]},${lats[3]},${lats[4]},${lats[5]},${lats[6]},${lats[7]}" >> "$out"
    done
done

