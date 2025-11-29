#!/bin/bash

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_bw_contention

for bpx in $(seq 1 275); do
    for tgt in $(seq 0 7); do
        ${BASE_DIR}/bin/incast_bpx_${bpx}_tgt_${tgt} |& tee ${BASE_DIR}/results/incast_bpx_${bpx}_tgt_${tgt}.out
    done
done

echo "All done."