#!/bin/bash

./bin/l1_lat_uncoal_w_2_s_2_r_4096 |& tee results/120825/l1_lat_uncoal_w_2_s_2_r_4096.out
./bin/l1_lat_uncoal_w_4_s_2_r_4096 |& tee results/120825/l1_lat_uncoal_w_4_s_2_r_4096.out
./bin/l1_lat_uncoal_w_8_s_2_r_4096 |& tee results/120825/l1_lat_uncoal_w_8_s_2_r_4096.out
./bin/l1_lat_uncoal_w_16_s_2_r_4096 |& tee results/120825/l1_lat_uncoal_w_16_s_2_r_4096.out

./bin/l1_lat_uncoal_w_2_s_4_r_4096 |& tee results/120825/l1_lat_uncoal_w_2_s_4_r_4096.out
./bin/l1_lat_uncoal_w_4_s_4_r_4096 |& tee results/120825/l1_lat_uncoal_w_4_s_4_r_4096.out
./bin/l1_lat_uncoal_w_8_s_4_r_4096 |& tee results/120825/l1_lat_uncoal_w_8_s_4_r_4096.out
./bin/l1_lat_uncoal_w_16_s_4_r_4096 |& tee results/120825/l1_lat_uncoal_w_16_s_4_r_4096.out

./bin/l1_lat_uncoal_w_2_s_8_r_4096 |& tee results/120825/l1_lat_uncoal_w_2_s_8_r_4096.out
./bin/l1_lat_uncoal_w_4_s_8_r_4096 |& tee results/120825/l1_lat_uncoal_w_4_s_8_r_4096.out
./bin/l1_lat_uncoal_w_8_s_8_r_4096 |& tee results/120825/l1_lat_uncoal_w_8_s_8_r_4096.out
# ./bin/l1_lat_uncoal_w_16_s_8_r_4096 |& tee results/120825/l1_lat_uncoal_w_16_s_8_r_4096.out

./bin/l1_lat_uncoal_w_2_s_16_r_4096 |& tee results/120825/l1_lat_uncoal_w_2_s_16_r_4096.out
./bin/l1_lat_uncoal_w_4_s_16_r_4096 |& tee results/120825/l1_lat_uncoal_w_4_s_16_r_4096.out
# ./bin/l1_lat_uncoal_w_8_s_16_r_4096 |& tee results/120825/l1_lat_uncoal_w_8_s_16_r_4096.out
# ./bin/l1_lat_uncoal_w_16_s_16_r_4096 |& tee results/120825/l1_lat_uncoal_w_16_s_16_r_4096.out