#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe"
N_REQUESTS="${N_REQUESTS:-120}"
ANOMALY_START="${ANOMALY_START:-40}"
ANOMALY_LENGTH="${ANOMALY_LENGTH:-24}"
BW_BPX="${ANOMALY_BW_BPX:-10}"

mkdir -p "$OUT/raw"
mkdir -p "$OUT/parsed"
RAW="$OUT/raw/anomaly_moe_n${N_REQUESTS}_a${ANOMALY_START}_l${ANOMALY_LENGTH}_bpx${BW_BPX}.out"
SUMMARY="$OUT/parsed/anomaly_moe_n${N_REQUESTS}_a${ANOMALY_START}_l${ANOMALY_LENGTH}_bpx${BW_BPX}_summary.csv"
ANOMALY_BW_BPX="$BW_BPX" \
  "$PROJ/bin/anomaly_moe" "$N_REQUESTS" "$ANOMALY_START" "$ANOMALY_LENGTH" \
  2>&1 | tee "$RAW"
"$PROJ/scripts/summarize.py" "$RAW" > "$SUMMARY"
printf 'summary: %s\n' "$SUMMARY"
