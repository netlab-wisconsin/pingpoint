#!/bin/bash

for WARPS in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
do
    ./bin/l1_lat_uncoal_w_${WARPS}_s_16_r_4096 |& tee results/120825/l1_lat_uncoal_w_${WARPS}_s_16_r_4096.out
done