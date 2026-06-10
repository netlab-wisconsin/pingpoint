#!/bin/bash

P_ACTIVE_XCD=2
P_TARGET_CC=0
P_WORKERS=4

RESULT_DIR="${WORK}/ici-workspace/sigcomm-exp/revision/db/raw/policy-a" # change
mkdir -p "${RESULT_DIR}"

# 0.8M -> 1.6M in 25K steps (33 points), covering the solo knee (~1.43M)
# and the corun knee (~1.0M-1.1M with BE_WORKERS_PER_XCD=32).
for P_ARRIVAL_QPS in $(seq 800000 25000 1500000); do
    ./bin/db-a_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS} 2>&1 | tee "${RESULT_DIR}/db-a_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}.out"
done
