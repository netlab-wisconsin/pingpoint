#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe"
N_REQUESTS="${N_REQUESTS:-120}"
FIRST_START="${FIRST_ANOMALY_START:-40}"
FIRST_LENGTH="${FIRST_ANOMALY_LENGTH:-20}"
SECOND_START="${SECOND_ANOMALY_START:-90}"
SECOND_LENGTH="${SECOND_ANOMALY_LENGTH:-10}"

mkdir -p "$OUT/raw"
mkdir -p "$OUT/parsed"
STEM="anomaly_moe_n${N_REQUESTS}_a1_${FIRST_START}_${FIRST_LENGTH}_a2_${SECOND_START}_${SECOND_LENGTH}_skew40_first7_second10_11_12"
RAW="$OUT/raw/${STEM}.out"
SUMMARY="$OUT/parsed/${STEM}_summary.csv"
TIMESERIES="$OUT/parsed/${STEM}_timeseries.csv"
"$PROJ/bin/anomaly_moe" "$N_REQUESTS" "$FIRST_START" "$FIRST_LENGTH" \
    "$SECOND_START" "$SECOND_LENGTH" \
  2>&1 | tee "$RAW"
"$PROJ/scripts/summarize.py" "$RAW" "$TIMESERIES" > "$SUMMARY"
printf 'summary: %s\n' "$SUMMARY"
printf 'time series: %s\n' "$TIMESERIES"
