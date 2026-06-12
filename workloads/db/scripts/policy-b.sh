#!/bin/bash

RESULT_DIR=${HOME}/workspace/ici-workspace/ici/workloads/db/results

P_ACTIVE_XCD=2
P_TARGET_CC=0

BE_WORKERS_PER_XCD=8 # sweep 8 or 16

# Use policy-a-moderate's two configs that moderately saturate path bw (30-50%) and 70% of solo QPS saturation

HIP_VISIBLE_DEVICES=-1
for BE_BATCH_CHUNKS in 64 128 256 512; do
    P_WORKERS=4 && P_ARRIVAL_QPS=700000
    export HIP_VISIBLE_DEVICES=$((HIP_VISIBLE_DEVICES + 1))
    ./bin/db-b_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS} 2>&1 | tee ${RESULT_DIR}/db-b_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS}.out &

    P_WORKERS=8 && P_ARRIVAL_QPS=1050000
    export HIP_VISIBLE_DEVICES=$((HIP_VISIBLE_DEVICES + 1))
    ./bin/db-b_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS} 2>&1 | tee ${RESULT_DIR}/db-b_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS}.out &
done