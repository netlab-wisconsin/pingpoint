#!/bin/bash

INPUT_DIR="$1"
OUTPUT_FILE="${INPUT_DIR}/tornado3_bpx_*_bw.csv3"

for bpx in $(seq 1 275); do
    FILE_PATH="${INPUT_DIR}/tornado_bpx_${bpx}_hop_3.out"
    awk '
    /n_blocks_per_xcd:/ {
        # extract n_blocks_per_xcd
        for (i = 1; i <= NF; i++) {
            if ($i ~ /n_blocks_per_xcd:/) {
                n_blocks_per_xcd = $(i+1)
                sub(/,/, "", n_blocks_per_xcd)
            }
        }
    }

    /^xcd [0-7]: global bw/ {
        xcd = $2
        sub(/:/, "", xcd)
        bw[xcd] = $(NF-1)
    }

    END {
        # print header (optional – remove if not wanted)
        printf "%s", n_blocks_per_xcd
        for (i = 0; i < 8; i++) {
            printf ",%s", bw[i]
        }
        printf "\n"
    }
    ' "$FILE_PATH" >> "$OUTPUT_FILE"
done
