#!/bin/bash

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/comio-lat-1-bw-1
BIN_DIR=${BASE_DIR}/bin
RESULTS_DIR=/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/comio-lat-1-vs-bw-1/raw

TARGETS=(
  "lat_0_bw_1" # local lat, local bw
  "lat_0_bw_3" # local lat, remote bw
  "lat_2_bw_1" # remote lat, local bw
  "lat_2_bw_3" # remote lat, remote bw
)

for TARGET in "${TARGETS[@]}"; do
    echo "Running benchmark: ${TARGET}"
    ${BIN_DIR}/${TARGET} |& tee ${RESULTS_DIR}/${TARGET}.out
done