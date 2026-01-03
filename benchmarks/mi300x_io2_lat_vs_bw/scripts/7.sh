#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
RESULTS_DIR="$BASE_DIR/results/0101"

${BIN_DIR}/stateful_routing_71 |& tee ${RESULTS_DIR}/71.out
${BIN_DIR}/stateful_routing_73 |& tee ${RESULTS_DIR}/73.out
${BIN_DIR}/stateful_routing_75 |& tee ${RESULTS_DIR}/75.out

