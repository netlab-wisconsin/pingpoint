#!/bin/bash

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_bw_contention

for bpx in $(seq 1 275); do
    for hop in 1 2; do
        ${BASE_DIR}/bin/tornado_bpx_${bpx}_hop_${hop} |& tee ${BASE_DIR}/results/tornado_bpx_${bpx}_hop_${hop}.out
    done
done

echo "All done."