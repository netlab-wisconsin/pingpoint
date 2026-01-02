#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"
RESULTS_DIR="$BASE_DIR/results/0101"

${BIN_DIR}/stateful_routing_13 |& tee ${RESULTS_DIR}/13.v8
${BIN_DIR}/stateful_routing_15 |& tee ${RESULTS_DIR}/15.v8
${BIN_DIR}/stateful_routing_17 |& tee ${RESULTS_DIR}/17.v8

${BIN_DIR}/stateful_routing_31 |& tee ${RESULTS_DIR}/31.v8
${BIN_DIR}/stateful_routing_35 |& tee ${RESULTS_DIR}/35.v8
${BIN_DIR}/stateful_routing_37 |& tee ${RESULTS_DIR}/37.v8

${BIN_DIR}/stateful_routing_51 |& tee ${RESULTS_DIR}/51.v8
${BIN_DIR}/stateful_routing_53 |& tee ${RESULTS_DIR}/53.v8
${BIN_DIR}/stateful_routing_57 |& tee ${RESULTS_DIR}/57.v8

${BIN_DIR}/stateful_routing_71 |& tee ${RESULTS_DIR}/71.v8
${BIN_DIR}/stateful_routing_73 |& tee ${RESULTS_DIR}/73.v8
${BIN_DIR}/stateful_routing_75 |& tee ${RESULTS_DIR}/75.v8
