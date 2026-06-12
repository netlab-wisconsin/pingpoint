#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate pacn-python3.13 

ROCPROFILER_COMPUTE_INSTALL_DIR=/work1/sinclair/junyeol/pacn-workspace/rocprofiler-compute
export PATH=${ROCPROFILER_COMPUTE_INSTALL_DIR}/bin:$PATH
export PYTHONPATH=${ROCPROFILER_COMPUTE_INSTALL_DIR}/python-libs

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/lat-1-bw-1/gfx950
BIN_DIR=${BASE_DIR}/bin

# set workload dir to current directory's workloads
WORKLOAD_DIR=${WORK}/ici-workspace/sigcomm-exp/comio-lat-1-vs-bw-1/workloads/gfx950

K1_PINNED_XCD=0
K1_PINNED_CC=0

K2_PINNED_XCD=0
K2_PINNED_CC=0

# K2_BPX_MIN=1
# K2_BPX_MAX=1
K2_TPB=512

# for i in {1..64}; do
#     TARGET="bpx_${i}_${i}_tpb_${K2_TPB}_prof_gfx950"
#     OUTPUT_DIR=${WORKLOAD_DIR}/${TARGET}
#     mkdir -p ${OUTPUT_DIR}

#     rocprof-compute profile -n ${TARGET} --path ${OUTPUT_DIR} --no-roof \
#     --block tatd vl1d l2 l2_per_channel \
#     -- ${BIN_DIR}/${TARGET}
# done

for i in {1..128}; do
    TARGET="bpx_${i}_${i}_tpb_${K2_TPB}_prof_gfx950"
    OUTPUT_DIR=${WORKLOAD_DIR}/${TARGET}
    mkdir -p ${OUTPUT_DIR}

    rocprof-compute profile -n ${TARGET} --path ${OUTPUT_DIR} --no-roof \
    --block tatd vl1d l2 l2_per_channel \
    -- ${BIN_DIR}/${TARGET}
done
