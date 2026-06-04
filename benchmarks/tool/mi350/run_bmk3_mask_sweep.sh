#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
RESULTS_DIR="${SCRIPT_DIR}/results"
HIPCC="${HIPCC:-/opt/rocm-7.1.0/bin/hipcc}"
HIPOPTS="${HIPOPTS:---amdgpu-target=gfx950}"

mkdir -p "${BIN_DIR}" "${RESULTS_DIR}"

compile() {
    local out="$1"
    shift
    "${HIPCC}" ${HIPOPTS} -o "${BIN_DIR}/${out}" "${SCRIPT_DIR}/bmk3.cpp" "$@"
}

run_case() {
    local bin="$1"
    local out="$2"
    "${BIN_DIR}/${bin}" > "${RESULTS_DIR}/${out}"
}

values=(0xff000000 0x00ff0000 0x0000ff00 0x000000ff)

for word in {0..7}; do
    for value in "${values[@]}"; do
        tag="$(printf 'bmk3_word%02d_%08x' "${word}" "${value}")"
        compile "${tag}" \
            -DMASK_MODE=1 \
            -DMASK_WORD="${word}" \
            -DMASK_VALUE="${value}" \
            -DBLOCKS_NUM=8 \
            -DTHREADS_PER_BLOCK=8
        run_case "${tag}" "${tag}.out"
    done
done

for enabled in 1 2 3 4 8 16 32; do
    blocks=$((enabled * 8))
    tag="$(printf 'bmk3_cumulative_%02d' "${enabled}")"
    compile "${tag}" \
        -DENABLED_CUS_PER_XCD_NUM="${enabled}" \
        -DBLOCKS_NUM="${blocks}" \
        -DTHREADS_PER_BLOCK=8
    run_case "${tag}" "${tag}.out"
done

echo "Wrote bmk3 mask sweep results to ${RESULTS_DIR}"
