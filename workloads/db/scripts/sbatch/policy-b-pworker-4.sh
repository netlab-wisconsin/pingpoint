#!/bin/bash
#SBATCH --job-name=policy-b-pworker-4
#SBATCH --output=/home1/junyeol/workspace/ici-workspace/ici/workloads/db/results/policy-b-pworker-4.out
#SBATCH --error=/home1/junyeol/workspace/ici-workspace/ici/workloads/db/results/policy-b-pworker-4.err

P_ACTIVE_XCD=2
P_TARGET_CC=0
# Use policy-a-moderate's two configs that moderately saturate path bw (30-50%) and 70% of solo QPS saturation
P_WORKERS=4
P_ARRIVAL_QPS=700000
# Policy B
BE_WORKERS_PER_XCD=8

./bin/db-b_${P_ACTIVE_XCD}_${P_TARGET_CC}_${P_WORKERS}_${P_ARRIVAL_QPS}_be${BE_WORKERS_PER_XCD}