#!/bin/bash

BASE_DIR="$(pwd)"
BIN_DIR="${BASE_DIR}/bin"
OUT_DIR="${BASE_DIR}/results/1213_local"
BPX_MAX=275

for t in $(seq 1 ${BPX_MAX}); do
    exe="${BIN_DIR}/local_bpx_${t}"
    out="${OUT_DIR}/local_bpx_${t}.out"

    if [[ -x "${exe}" ]]; then
        echo "=== Running ${exe} ==="
        "${exe}" > "${out}" 2>&1 || {
            echo "Error running ${exe}"
            exit 1
        }
    fi
done
