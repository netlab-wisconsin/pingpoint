#!/bin/bash

BASE_DIR=${HOME}/workspace/ici-workspace/ici/benchmarks/mi300x_amo_hol_blocking
BIN_DIR=${BASE_DIR}/bin

TARGETS=(
  "amo_hol_blocking_128_128_128"
  "amo_hol_blocking_256_128_128"
  "amo_hol_blocking_256_128_256"
  "amo_hol_blocking_256_256_128"
  "amo_hol_blocking_256_256_256"
  "amo_hol_blocking_512_128_128"
  "amo_hol_blocking_512_128_256"
  "amo_hol_blocking_512_128_512"
  "amo_hol_blocking_512_256_128"
  "amo_hol_blocking_512_256_256"
  "amo_hol_blocking_512_256_512"
  "amo_hol_blocking_512_512_128"
  "amo_hol_blocking_512_512_256"
  "amo_hol_blocking_512_512_512"
)

for TARGET in "${TARGETS[@]}"; do
    echo "Running benchmark: ${TARGET}"
    ${BIN_DIR}/${TARGET} |& tee ${BASE_DIR}/results/${TARGET}.out
done