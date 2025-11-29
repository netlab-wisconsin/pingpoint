#!/bin/bash

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_bw_contention

for bpx in $(seq 1 275); do
    ${BASE_DIR}/bin/local_bpx_$bpx |& tee ${BASE_DIR}/results/local_bpx_$bpx.out
done

echo "All done."