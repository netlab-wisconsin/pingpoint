#!/bin/bash

P_ACTIVE_XCD=2
P_TARGET_CC=0

# Define pairs as "WORKERS:QPS": Five configs that fully saturate QPS
for pair in "1:100000" "2:500000" "4:1000000" "8:1500000" "16:1800000"; do
    P_WORKERS="${pair%%:*}"
    P_ARRIVAL_QPS="${pair#*:}"
    make -B policy-a \
      P_ACTIVE_XCD="${P_ACTIVE_XCD}" \
      P_TARGET_CC="${P_TARGET_CC}" \
      P_WORKERS="${P_WORKERS}" \
      P_ARRIVAL_QPS="${P_ARRIVAL_QPS}"
done


# Define pairs as "WORKERS:QPS": Two configs that moderately saturate QPS
for pair in "4:700000" "8:1050000"; do
    P_WORKERS="${pair%%:*}"
    P_ARRIVAL_QPS="${pair#*:}"
    make -B policy-a \
      P_ACTIVE_XCD="${P_ACTIVE_XCD}" \
      P_TARGET_CC="${P_TARGET_CC}" \
      P_WORKERS="${P_WORKERS}" \
      P_ARRIVAL_QPS="${P_ARRIVAL_QPS}"
done