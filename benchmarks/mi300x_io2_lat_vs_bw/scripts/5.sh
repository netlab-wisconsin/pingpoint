#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
RESULTS_DIR="$BASE_DIR/results/0101"

${BIN_DIR}/stateful_routing_51 |& tee ${RESULTS_DIR}/51.out
${BIN_DIR}/stateful_routing_53 |& tee ${RESULTS_DIR}/53.out
${BIN_DIR}/stateful_routing_57 |& tee ${RESULTS_DIR}/57.out
