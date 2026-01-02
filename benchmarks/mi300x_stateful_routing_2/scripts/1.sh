#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
RESULTS_DIR="$BASE_DIR/results/0101"

${BIN_DIR}/stateful_routing_13 |& tee ${RESULTS_DIR}/13.out
${BIN_DIR}/stateful_routing_15 |& tee ${RESULTS_DIR}/15.out
${BIN_DIR}/stateful_routing_17 |& tee ${RESULTS_DIR}/17.out
