#!/bin/bash

P_ACTIVE_XCD=2
P_TARGET_CC=0
P_WORKERS=4

BE_WORKERS_PER_XCD=32 # 8/16 don't affect P with hashed batches; 32 does.
BE_BATCH_CHUNKS=64

RESULT_DIR="${WORK}/ici-workspace/sigcomm-exp/revision/db/raw/policy-e" # change
mkdir -p "${RESULT_DIR}"

# 0.8M -> 1.5M in 25K steps. Policy e leaves P's CC untouched, so the knee
# should recover to policy a's (~1.43M) while BE keeps full bandwidth.
for P_ARRIVAL_QPS in $(seq 800000 25000 1500000); do
    ./bin/db-e_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS} 2>&1 | tee "${RESULT_DIR}/db-e_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS}.out"
done
