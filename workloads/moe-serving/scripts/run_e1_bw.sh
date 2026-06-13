#!/usr/bin/env bash
# E1 (moe-serving): target FFN1 throughput vs. ping rate.
#   Conditions: baseline + BANDWIDTH (K2) pings.   Latency (K1) plans compiled out.
#   Customized plan set (PPNT_PLAN_SELECTED_ONLY=1): src_xcd 0 -> dst_hbm 0..7 x #active-CU 1..16.
# Optional: set N_PASSES env to override the per-condition serving-loop length.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="${WORK:?set WORK}/ici-workspace/sigcomm-exp/revision/overhead/raw"
BIN="$PROJ/bin/moe_serving"
mkdir -p "$OUT"

cd "$PROJ"
make clean
make CCFLAGS="-DDISABLE_K1_PLANS=1 -DDISABLE_K2_PLANS=0"

for regime in prefill decode; do
    args=("$regime"); [ -n "${N_PASSES:-}" ] && args+=("$N_PASSES")
    "$BIN" "${args[@]}" 2>&1 | tee "$OUT/moe_serving_e1_bw_${regime}.out"
done
