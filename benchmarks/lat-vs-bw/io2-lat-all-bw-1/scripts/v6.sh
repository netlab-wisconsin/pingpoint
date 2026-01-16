#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
RESULTS_DIR="$BASE_DIR/results/0102/saturation4"

for tpb in 128 256 512; do
    ${BIN_DIR}/lat_all_all_bw_5_1_${tpb} |& tee ${RESULTS_DIR}/51_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_5_3_${tpb} |& tee ${RESULTS_DIR}/53_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_5_7_${tpb} |& tee ${RESULTS_DIR}/57_${tpb}.v6

    ${BIN_DIR}/lat_all_all_bw_7_1_${tpb} |& tee ${RESULTS_DIR}/71_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_7_3_${tpb} |& tee ${RESULTS_DIR}/73_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_7_5_${tpb} |& tee ${RESULTS_DIR}/75_${tpb}.v6

    ${BIN_DIR}/lat_all_all_bw_1_3_${tpb} |& tee ${RESULTS_DIR}/13_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_1_5_${tpb} |& tee ${RESULTS_DIR}/15_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_1_7_${tpb} |& tee ${RESULTS_DIR}/17_${tpb}.v6

    ${BIN_DIR}/lat_all_all_bw_3_1_${tpb} |& tee ${RESULTS_DIR}/31_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_3_5_${tpb} |& tee ${RESULTS_DIR}/35_${tpb}.v6
    ${BIN_DIR}/lat_all_all_bw_3_7_${tpb} |& tee ${RESULTS_DIR}/37_${tpb}.v6
done