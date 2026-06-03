#!/bin/bash

ARCH="mi350"

SUFFIX="HBM"

for i in $(seq 48 60); do
    srun -N 1 -p mi3501x -t 04:00:00 ../bin/acn-main-hop0-mi350-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop0-mi350-HBM-${i}.out
done
