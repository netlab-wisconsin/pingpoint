#!/bin/bash

for bpx in $(seq 1 160); do
    ${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_bw_contention/bin/local_bpx_$bpx |& tee ${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_bw_contention/results/local_bpx_$bpx.out
done

echo "All done."