#!/bin/bash

P_ACTIVE_XCD=2
P_TARGET_CC=0
P_WORKERS=4

BE_WORKERS_PER_XCD=32 # 8/16 don't affect P with hashed batches; 32 does.
BE_BATCH_CHUNKS=64
BE_THROTTLE_PCT=50 # 75% leaves CC0 ~25% of BE demand, matching policy c's residual

RESULT_DIR="${WORK}/ici-workspace/sigcomm-exp/revision/db/raw/policy-d-${BE_THROTTLE_PCT}" # change
mkdir -p "${RESULT_DIR}"

# 0.8M -> 1.5M in 25K steps, covering the solo knee (~1.43M) and the
# policy-b knee (~1.0M-1.1M); policy d should track policy c for P,
# with BE throughput cut to ~(100-BE_THROTTLE_PCT)%.
for P_ARRIVAL_QPS in $(seq 800000 25000 1500000); do
    ./bin/db-d_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS}_T_${BE_THROTTLE_PCT} 2>&1 | tee "${RESULT_DIR}/db-d_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS}_T_${BE_THROTTLE_PCT}.out"
done
