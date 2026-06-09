#!/bin/bash

RESULT_DIR=${HOME}/workspace/ici-workspace/ici/workloads/db/results

P_ACTIVE_XCD=2
P_TARGET_CC=0
BE_WORKERS_PER_XCD=8

# Use policy-a-moderate's two configs that moderately saturate path bw (30-50%) and 70% of solo QPS saturation

P_WORKERS=4 && P_ARRIVAL_QPS=700000
HIP_VISIBLE_DEVICES=0 ./bin/db-a_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_${BE_WORKERS_PER_XCD} 2>&1 | tee ${RESULT_DIR}/db-a_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_${BE_WORKERS_PER_XCD}.out &

P_WORKERS=8 && P_ARRIVAL_QPS=1050000
HIP_VISIBLE_DEVICES=1 ./bin/db-a_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_${BE_WORKERS_PER_XCD} 2>&1 | tee ${RESULT_DIR}/db-a_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_${BE_WORKERS_PER_XCD}.out &
