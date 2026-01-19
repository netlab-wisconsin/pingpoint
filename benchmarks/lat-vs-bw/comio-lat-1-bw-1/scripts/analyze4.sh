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

K2_BPX_PROF_START=1
K2_BPX_PROF_END=160

SUFFIX="prof"

K1_KERNEL_IDX=
K2_KERNEL_IDX=

if [ -z "$K1_KERNEL_IDX" ] || [ -z "$K2_KERNEL_IDX" ]; then
  echo "Error: Both K1_KERNEL_IDX and K2_KERNEL_IDX must be set. Lookup inside thPlease set them before running the script."
  exit 1
fi

for i in $(seq "$K2_BPX_PROF_START" "$K2_BPX_PROF_END"); do
  TARGET="lat_${K1_PINNED_HBM}_bw_${K2_PINNED_HBM}_bpx_${i}_${i}_${SUFFIX}"
  OUTPUT_DIR=${WORKLOAD_DIR}/${TARGET}/MI300X_A1

  # analyze k1
  ${ROCPROF_COMPUTE} analyze --kernel ${K1_KERNEL_IDX} --path ${OUTPUT_DIR} --quiet > ${OUTPUT_DIR}/analyze_k1.out # Use `--kernel $idx` to profile specific kernel listed at --list-stats 

  # analyze k2
  ${ROCPROF_COMPUTE} analyze --kernel ${K2_KERNEL_IDX} --path ${OUTPUT_DIR} --quiet > ${OUTPUT_DIR}/analyze_k2.out # Use `--kernel $idx` to profile specific kernel listed at --list-stats 
done