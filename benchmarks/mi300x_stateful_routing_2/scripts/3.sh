#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
RESULTS_DIR="$BASE_DIR/results/0101"

${BIN_DIR}/stateful_routing_31 |& tee ${RESULTS_DIR}/31.out
${BIN_DIR}/stateful_routing_35 |& tee ${RESULTS_DIR}/35.out
${BIN_DIR}/stateful_routing_37 |& tee ${RESULTS_DIR}/37.out
