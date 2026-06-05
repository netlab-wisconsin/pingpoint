#!/bin/bash

RESULT_DIR=${WORK}/ici-workspace/sigcomm-exp/revision/com-io-mi350

HIP_VISIBLE_DEVICES=4 ./bin/lat_xcd_0_cc_0_bw_xcd_1_cc_0_bpx_1_128_tpb_512_iomem 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_0_bw_xcd_1_cc_0_bpx_1_128_tpb_512_iomem.out &

HIP_VISIBLE_DEVICES=5 ./bin/lat_xcd_0_cc_0_bw_xcd_1_cc_1_bpx_1_128_tpb_512_iomem 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_0_bw_xcd_1_cc_1_bpx_1_128_tpb_512_iomem.out &

HIP_VISIBLE_DEVICES=6 ./bin/lat_xcd_0_cc_1_bw_xcd_1_cc_0_bpx_1_128_tpb_512_iomem 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_1_bw_xcd_1_cc_0_bpx_1_128_tpb_512_iomem.out &

HIP_VISIBLE_DEVICES=7 ./bin/lat_xcd_0_cc_1_bw_xcd_1_cc_1_bpx_1_128_tpb_512_iomem 2>&1 | tee ${RESULT_DIR}/lat_xcd_0_cc_1_bw_xcd_1_cc_1_bpx_1_128_tpb_512_iomem.out &

wait