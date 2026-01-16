#!/bin/bash

# Directory Setup
BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
mkdir -p "$BIN_DIR"

# Compiler and Path Settings
HIP_HOME="/opt/rocm-7.1.0"
CC="$HIP_HOME/bin/hipcc"
OPTS="--amdgpu-target=gfx942"
INCLUDES="-I$HIP_HOME/include/rocprofiler/ -I$HIP_HOME/hsa/include/hsa"
LDFLAGS="-L$HIP_HOME/rocprofiler/lib -lrocprofiler64 -lhsa-runtime64 -lrocm_smi64 -ldl"

# Constants
K1_PINNED_XCD=0
K1_PINNED_HBM=2
K2_PINNED_XCD=5
K2_PINNED_HBM=7
K2_BPX_MAX=160

# Values to scale
TPB_VALUES=(2 4 8 16 32 64 128)

echo "Starting compilation for K2_TPB scaling..."

for K2_TPB in "${TPB_VALUES[@]}"; do
    # Define output filename including the TPB value
    OUTPUT="$BIN_DIR/stateful_routing_${K1_PINNED_XCD}${K1_PINNED_HBM}${K2_PINNED_XCD}${K2_PINNED_HBM}_TPB${K2_TPB}"
    
    echo "Compiling for K2_TPB=$K2_TPB -> $OUTPUT"
    
    $CC $OPTS $INCLUDES $LDFLAGS \
        -DK1_PINNED_XCD=$K1_PINNED_XCD \
        -DK1_PINNED_HBM=$K1_PINNED_HBM \
        -DK2_PINNED_XCD=$K2_PINNED_XCD \
        -DK2_PINNED_HBM=$K2_PINNED_HBM \
        -DK2_BPX_MAX=$K2_BPX_MAX \
        -DK2_TPB=$K2_TPB \
        -o "$OUTPUT" main.cpp

    if [ $? -eq 0 ]; then
        echo "Successfully compiled: $OUTPUT"
    else
        echo "Error: Compilation failed for K2_TPB=$K2_TPB"
        exit 1
    fi
done

echo "All compilations complete."