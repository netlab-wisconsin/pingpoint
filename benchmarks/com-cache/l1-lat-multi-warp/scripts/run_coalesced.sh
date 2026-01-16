#!/bin/bash

# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_2_r_4096 |& tee results/120825/l1_lat_coal_w_2_r_4096.out
# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_2_r_8192 |& tee results/120825/l1_lat_coal_w_2_r_8192.out
# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_4_r_4096 |& tee results/120825/l1_lat_coal_w_4_r_4096.out
# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_4_r_8192 |& tee results/120825/l1_lat_coal_w_4_r_8192.out
# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_8_r_4096 |& tee results/120825/l1_lat_coal_w_8_r_4096.out
# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_8_r_8192 |& tee results/120825/l1_lat_coal_w_8_r_8192.out
# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_16_r_4096 |& tee results/120825/l1_lat_coal_w_16_r_4096.out
# srun -N 1 -p mi3001x -w k002-004-v3 -t 0:10:00 ./bin/l1_lat_coal_w_16_r_8192 |& tee results/120825/l1_lat_coal_w_16_r_8192.out

# ./bin/l1_lat_coal_w_2_r_4096 |& tee results/120825/l1_lat_coal_w_2_r_4096.verbose
# ./bin/l1_lat_coal_w_2_r_8192 |& tee results/120825/l1_lat_coal_w_2_r_8192.verbose
# ./bin/l1_lat_coal_w_4_r_4096 |& tee results/120825/l1_lat_coal_w_4_r_4096.verbose
# ./bin/l1_lat_coal_w_4_r_8192 |& tee results/120825/l1_lat_coal_w_4_r_8192.verbose
# ./bin/l1_lat_coal_w_8_r_4096 |& tee results/120825/l1_lat_coal_w_8_r_4096.verbose
# ./bin/l1_lat_coal_w_8_r_8192 |& tee results/120825/l1_lat_coal_w_8_r_8192.verbose
# ./bin/l1_lat_coal_w_16_r_4096 |& tee results/120825/l1_lat_coal_w_16_r_4096.verbose
# ./bin/l1_lat_coal_w_16_r_8192 |& tee results/120825/l1_lat_coal_w_16_r_8192.verbose

./bin/l1_lat_coal_w_1_r_4096 |& tee results/120825/l1_lat_coal_w_1_r_4096.out
./bin/l1_lat_coal_w_2_r_4096 |& tee results/120825/l1_lat_coal_w_2_r_4096.out
./bin/l1_lat_coal_w_3_r_4096 |& tee results/120825/l1_lat_coal_w_3_r_4096.out
./bin/l1_lat_coal_w_4_r_4096 |& tee results/120825/l1_lat_coal_w_4_r_4096.out
./bin/l1_lat_coal_w_5_r_4096 |& tee results/120825/l1_lat_coal_w_5_r_4096.out
./bin/l1_lat_coal_w_6_r_4096 |& tee results/120825/l1_lat_coal_w_6_r_4096.out
./bin/l1_lat_coal_w_7_r_4096 |& tee results/120825/l1_lat_coal_w_7_r_4096.out
# ./bin/l1_lat_coal_w_8_r_4096 |& tee results/120825/l1_lat_coal_w_8_r_4096.out
# ./bin/l1_lat_coal_w_9_r_4096 |& tee results/120825/l1_lat_coal_w_9_r_4096.out
# ./bin/l1_lat_coal_w_10_r_4096 |& tee results/120825/l1_lat_coal_w_10_r_4096.out
# ./bin/l1_lat_coal_w_11_r_4096 |& tee results/120825/l1_lat_coal_w_11_r_4096.out
# ./bin/l1_lat_coal_w_12_r_4096 |& tee results/120825/l1_lat_coal_w_12_r_4096.out
# ./bin/l1_lat_coal_w_13_r_4096 |& tee results/120825/l1_lat_coal_w_13_r_4096.out
# ./bin/l1_lat_coal_w_14_r_4096 |& tee results/120825/l1_lat_coal_w_14_r_4096.out
# ./bin/l1_lat_coal_w_15_r_4096 |& tee results/120825/l1_lat_coal_w_15_r_4096.out
# ./bin/l1_lat_coal_w_16_r_4096 |& tee results/120825/l1_lat_coal_w_16_r_4096.out