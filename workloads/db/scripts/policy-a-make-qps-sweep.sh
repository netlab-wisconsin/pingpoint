#!/bin/bash

P_ACTIVE_XCD=2
P_TARGET_CC=0
P_WORKERS=4

BE_WORKERS_PER_XCD=8 # sweep 8 or 16
BE_BATCH_CHUNKS=64

for P_ARRIVAL_QPS in 800000 1000000 1200000 1400000 1600000 1800000 2000000; do
    make -B policy-a \
      P_ACTIVE_XCD="${P_ACTIVE_XCD}" \
      P_TARGET_CC="${P_TARGET_CC}" \
      P_WORKERS="${P_WORKERS}" \
      P_ARRIVAL_QPS="${P_ARRIVAL_QPS}"
done 

# More fine-grained within 1.4M--1.6M, which is around the knee of the curve.
for P_ARRIVAL_QPS in 1425000 1450000 1475000 1500000 1525000 1550000 1575000; do
    make -B policy-a \
      P_ACTIVE_XCD="${P_ACTIVE_XCD}" \
      P_TARGET_CC="${P_TARGET_CC}" \
      P_WORKERS="${P_WORKERS}" \
      P_ARRIVAL_QPS="${P_ARRIVAL_QPS}"
done 