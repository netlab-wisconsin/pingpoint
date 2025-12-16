#!/bin/bash

INPUT_DIR="$1"
OUTPUT_FILE="${INPUT_DIR}/tornado3_bpx_*_lat.csv3"

for bpx in $(seq 1 275); do
    FILE_PATH="${INPUT_DIR}/tornado_bpx_${bpx}_hop_3.out"
  awk '
    /n_blocks_per_xcd:/ {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /n_blocks_per_xcd:/) {
          n_blocks_per_xcd = $(i+1)
          sub(/,/, "", n_blocks_per_xcd)
        }
      }
    }

    # xcd 3: avg lat(16B) 619.85 cycles
    /^xcd [0-7]: avg lat\(16B\)/ {
      xcd = $2; sub(/:/, "", xcd)
      lat[xcd] = $(NF-1)
    }

    END {
      # n_blocks_per_xcd first
      printf "%s", n_blocks_per_xcd

      # then lat[0..7]
      for (i = 0; i < 8; i++) printf ",%s", (lat[i] == "" ? "NA" : lat[i])

      printf "\n"
    }
  ' "$FILE_PATH" >> "$OUTPUT_FILE"
done
