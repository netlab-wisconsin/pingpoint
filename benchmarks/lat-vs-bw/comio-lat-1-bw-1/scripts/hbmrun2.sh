#!/bin/bash

K1_PINNED_XCD=0
K2_PINNED_XCD=1

K1_PINNED_HBM=0
K2_PINNED_HBM=3

K2_TPB=1024
SUFFIX=hbmrange

./bin/lat_xcd_${K1_PINNED_XCD}_hbm_${K1_PINNED_HBM}_bw_xcd_${K2_PINNED_XCD}_hbm_${K2_PINNED_HBM}_bpx_1_80_tpb_${K2_TPB}_${SUFFIX} |& tee /work1/sinclair/junyeol/ici-workspace/sigcomm-exp/3.6-lat-vs-bw/raw/lat_xcd_${K1_PINNED_XCD}_hbm_${K1_PINNED_HBM}_bw_xcd_${K2_PINNED_XCD}_hbm_${K2_PINNED_HBM}_bpx_1_80_tpb_${K2_TPB}_${SUFFIX}.out