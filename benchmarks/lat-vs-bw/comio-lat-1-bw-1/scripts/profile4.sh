#!/bin/bash

source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate rocprofiler-compute

ROCM_PATH=/opt/rocm-6.4.1
ROCPROF_COMPUTE=${ROCM_PATH}/bin/rocprof-compute

# Encountered errors when LD_LIBRARY_PATH is not set:
# - symbol lookup error: libamdhip64.so
# export LD_LIBRARY_PATH=${ROCM_PATH}/lib:${LD_LIBRARY_PATH}

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/lat-vs-bw/comio-lat-1-bw-1
BIN_DIR=${BASE_DIR}/bin

# set workload dir to current directory's workloads
WORKLOAD_DIR=/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/comio-lat-1-vs-bw-1/workloads

K1_PINNED_HBM=2
K2_PINNED_HBM=7

set +e # continue on error

for i in $(seq 1 160); do
  TARGET="lat_${K1_PINNED_HBM}_bw_${K2_PINNED_HBM}_bpxmin_${i}_bpxmax_${i}_prof"
  OUTPUT_DIR=${WORKLOAD_DIR}/${TARGET}/MI300X_A1
  mkdir -p ${OUTPUT_DIR}
  echo "Profiling and analyzing ${TARGET}..."
  ${ROCPROF_COMPUTE} profile -n ${TARGET} --path ${OUTPUT_DIR} --no-roof --quiet -- ${BIN_DIR}/${TARGET}
  ${ROCPROF_COMPUTE} analyze --path ${OUTPUT_DIR} --quiet > ${OUTPUT_DIR}/analyze.out # Use `--kernel $idx` to profile specific kernel listed at --list-stats 
done