#!/usr/bin/env bash
#SBATCH --job-name=bmk1_1tbx
#SBATCH --output=results/bmk1_1tbx.%j.out
#SBATCH --error=results/bmk1_1tbx.%j.err

if [ -n "${SLURM_JOB_ID:-}" ] ; then
    SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}')
else
    SCRIPT_PATH=$(realpath "$0")
fi

make bmk1_1tbx
BIN="./bin/bmk1_1tbx"
"${BIN}"
