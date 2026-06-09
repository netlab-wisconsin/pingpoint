#!/bin/bash

P_ACTIVE_XCD=2
P_TARGET_CC=0
BE_WORKERS_PER_XCD=8

# Define pairs as "WORKERS:QPS": Two configs that moderately saturate QPS
for pair in "4:700000" "8:1050000"; do
    P_WORKERS="${pair%%:*}"
    P_ARRIVAL_QPS="${pair#*:}"
    make -B policy-b \
      P_ACTIVE_XCD="${P_ACTIVE_XCD}" \
      P_TARGET_CC="${P_TARGET_CC}" \
      P_WORKERS="${P_WORKERS}" \
      P_ARRIVAL_QPS="${P_ARRIVAL_QPS}" \
      BE_WORKERS_PER_XCD="${BE_WORKERS_PER_XCD}"
done