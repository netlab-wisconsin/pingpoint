#!/bin/bash

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR="${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/comio-lat-1-bw-1"
BIN_DIR="${BASE_DIR}/bin"
RESULTS_DIR="/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/comio-lat-1-vs-bw-1/raw"

K2_BPX_MIN=1
K2_BPX_MAX=100 # XCD full occupancy at ~80 BPX
SUFFIX="run"

# 4 combinates of lat: (local,remote) x bw: (local,remote)
COMBINATIONS=(
    "0 1"  # Local Lat, Local BW
    "0 3"  # Local Lat, Remote BW
    "2 1"  # Remote Lat, Local BW
    "2 7"  # Remote Lat, Remote BW
)

for combo in "${COMBINATIONS[@]}"; do
    # Split the combination into K1_PINNED_HBM and K2_PINNED_HBM
    K1_PINNED_HBM=$(echo $combo | cut -d' ' -f1)
    K2_PINNED_HBM=$(echo $combo | cut -d' ' -f2)

    TARGET="lat_${lat_hbm}_bw_${bw_hbm}_bpx_${K2_BPX_MIN}_${K2_BPX_MAX}_${SUFFIX}"
    ${BIN_DIR}/${TARGET} |& tee ${RESULTS_DIR}/${TARGET}.out
done