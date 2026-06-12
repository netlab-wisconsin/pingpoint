#!/bin/bash

P_ACTIVE_XCD=2
P_TARGET_CC=0
P_WORKERS=4

BE_WORKERS_PER_XCD=32 # 8/16 don't affect P with hashed batches; 32 does.
BE_BATCH_CHUNKS=64

# 0.8M -> 1.5M in 25K steps. Policy e leaves P's CC untouched, so the knee
# should recover to policy a's (~1.43M) while BE keeps full bandwidth.
for P_ARRIVAL_QPS in $(seq 800000 25000 1500000); do
    make -B policy-e \
      P_ACTIVE_XCD="${P_ACTIVE_XCD}" \
      P_TARGET_CC="${P_TARGET_CC}" \
      P_WORKERS="${P_WORKERS}" \
      P_ARRIVAL_QPS="${P_ARRIVAL_QPS}" \
      BE_WORKERS_PER_XCD="${BE_WORKERS_PER_XCD}" \
      BE_BATCH_CHUNKS="${BE_BATCH_CHUNKS}"
done
