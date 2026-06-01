#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate pacn-python3.13 

ROCPROFILER_COMPUTE_INSTALL_DIR=/work1/sinclair/junyeol/pacn-workspace/rocprofiler-compute
export PATH=${ROCPROFILER_COMPUTE_INSTALL_DIR}/bin:$PATH
export PYTHONPATH=${ROCPROFILER_COMPUTE_INSTALL_DIR}/python-libs

PATH_SUFFIX="053126"

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/lat-1-bw-1
BIN_DIR=${BASE_DIR}/bin-${PATH_SUFFIX}

# set workload dir to current directory's workloads
WORKLOAD_DIR=/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/comio-lat-1-vs-bw-1/workloads-${PATH_SUFFIX}

K1_PINNED_XCD=0
K1_PINNED_HBM=0

K2_PINNED_XCD=0
K2_PINNED_HBM=0

# K2_BPX_MIN=1
# K2_BPX_MAX=1
K2_TPB=1024

SUFFIX="prof"

TARGET="lat_xcd_${K1_PINNED_XCD}_hbm_${K1_PINNED_HBM}_bw_xcd_${K2_PINNED_XCD}_hbm_${K2_PINNED_HBM}_bpx_${K2_BPX_MIN}_${K2_BPX_MAX}_tpb_${K2_TPB}_${SUFFIX}"
OUTPUT_DIR=${WORKLOAD_DIR}/${TARGET}/MI300X_A1
mkdir -p ${OUTPUT_DIR}

for i in {1..160}; do
    K2_BPX_MIN=${i}
    K2_BPX_MAX=${i}

    TARGET="lat_xcd_${K1_PINNED_XCD}_hbm_${K1_PINNED_HBM}_bw_xcd_${K2_PINNED_XCD}_hbm_${K2_PINNED_HBM}_bpx_${K2_BPX_MIN}_${K2_BPX_MAX}_tpb_${K2_TPB}_${SUFFIX}"
    OUTPUT_DIR=${WORKLOAD_DIR}/${TARGET}/MI300X_A1
    mkdir -p ${OUTPUT_DIR}

    rocprof-compute profile -n ${TARGET} --path ${OUTPUT_DIR} --no-roof \
    --block tatd vl1d l2 l2_per_channel \
    -- ${BIN_DIR}/${TARGET}
done

# rocprof-compute profile -n ${TARGET} --path ${OUTPUT_DIR} --no-roof \
#     --block tatd vl1d l2 l2_per_channel \
#     -- ${BIN_DIR}/${TARGET}