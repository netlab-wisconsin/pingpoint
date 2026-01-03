#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
RESULTS_DIR="$BASE_DIR/results/0102/saturation4"

${BIN_DIR}/lat_all_all_bw_5_1 |& tee ${RESULTS_DIR}/51.v6
${BIN_DIR}/lat_all_all_bw_5_3 |& tee ${RESULTS_DIR}/53.v6
${BIN_DIR}/lat_all_all_bw_5_7 |& tee ${RESULTS_DIR}/57.v6

${BIN_DIR}/lat_all_all_bw_7_1 |& tee ${RESULTS_DIR}/71.v6
${BIN_DIR}/lat_all_all_bw_7_3 |& tee ${RESULTS_DIR}/73.v6
${BIN_DIR}/lat_all_all_bw_7_5 |& tee ${RESULTS_DIR}/75.v6

${BIN_DIR}/lat_all_all_bw_1_3 |& tee ${RESULTS_DIR}/13.v6
${BIN_DIR}/lat_all_all_bw_1_5 |& tee ${RESULTS_DIR}/15.v6
${BIN_DIR}/lat_all_all_bw_1_7 |& tee ${RESULTS_DIR}/17.v6

${BIN_DIR}/lat_all_all_bw_3_1 |& tee ${RESULTS_DIR}/31.v6
${BIN_DIR}/lat_all_all_bw_3_5 |& tee ${RESULTS_DIR}/35.v6
${BIN_DIR}/lat_all_all_bw_3_7 |& tee ${RESULTS_DIR}/37.v6