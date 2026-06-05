#!/bin/bash

HIP_VISIBLE_DEVICES=0 ./bin/lat_xcd_0_cc_0_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee results/lat_xcd_0_cc_0_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950.out &

HIP_VISIBLE_DEVICES=1 ./bin/lat_xcd_0_cc_0_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee results/lat_xcd_0_cc_0_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950.out &

HIP_VISIBLE_DEVICES=4 ./bin/lat_xcd_0_cc_1_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee results/lat_xcd_0_cc_1_bw_xcd_1_cc_0_bpx_1_128_tpb_512_run_gfx950.out &

HIP_VISIBLE_DEVICES=5 ./bin/lat_xcd_0_cc_1_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950 2>&1 | tee results/lat_xcd_0_cc_1_bw_xcd_1_cc_1_bpx_1_128_tpb_512_run_gfx950.out &