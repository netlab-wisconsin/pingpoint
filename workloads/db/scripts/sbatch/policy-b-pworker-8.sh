#!/bin/bash
#SBATCH --job-name=policy-b-pworker-8
#SBATCH --output=/home1/junyeol/workspace/ici-workspace/ici/workloads/db/results/policy-b-pworker-8.out
#SBATCH --error=/home1/junyeol/workspace/ici-workspace/ici/workloads/db/results/policy-b-pworker-8.err

P_ACTIVE_XCD=2
P_TARGET_CC=0
# Use policy-a-moderate's two configs that moderately saturate path bw (30-50%) and 70% of solo QPS saturation
P_WORKERS=8
P_ARRIVAL_QPS=1050000
# Policy B
BE_WORKERS_PER_XCD=16
BE_BATCH_CHUNKS=64

./bin/db-b_P_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_BE_${BE_WORKERS_PER_XCD}_${BE_BATCH_CHUNKS}
