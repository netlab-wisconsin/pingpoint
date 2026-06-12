#!/bin/bash

RESULT_DIR=${WORK}/ici-workspace/sigcomm-exp/revision/com-io-mi350

HIP_VISIBLE_DEVICES=0 ./bin/lat_xcd_0_cc_0_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_0_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950.out &

HIP_VISIBLE_DEVICES=1 ./bin/lat_xcd_0_cc_0_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_0_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950.out &

HIP_VISIBLE_DEVICES=2 ./bin/lat_xcd_0_cc_1_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_1_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950.out &

HIP_VISIBLE_DEVICES=3 ./bin/lat_xcd_0_cc_1_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_1_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950.out &