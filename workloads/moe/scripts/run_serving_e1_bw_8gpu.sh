#!/usr/bin/env bash
# E1 (moe-serving): sweep the 128 BANDWIDTH (K2) plans across 8 GPUs, one process per GPU,
# each running 1/8 of the plans (16 each) via MOE_BW_NSHARD/MOE_BW_SHARD.
#   - Latency (K1) plans compiled out; baseline runs in every shard (per-GPU reference).
#   - Output: one CSV per (regime, gpu) shard under $OUT; concatenate shards afterward.
# Env: NGPU (default 8), N_PASSES (per-condition serving-loop length, optional).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="${WORK:?set WORK}/ici-workspace/sigcomm-exp/revision/overhead/raw"
BIN="$PROJ/bin/moe_serving_bw"
NGPU="${NGPU:-8}"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "build first: scripts/make_serving_e1_bw.sh"; exit 1; }

for regime in prefill decode; do
    echo "=== regime=$regime : launching $NGPU GPU shards ==="
    pids=()
    for g in $(seq 0 $((NGPU - 1))); do
        args=("$regime"); [ -n "${N_PASSES:-}" ] && args+=("$N_PASSES")
        HIP_VISIBLE_DEVICES="$g" MOE_BW_NSHARD="$NGPU" MOE_BW_SHARD="$g" \
            "$BIN" "${args[@]}" > "$OUT/moe_serving_e1_bw_${regime}_shard${g}.out" 2>&1 &
        pids+=($!)
    done
    rc=0
    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then echo "  shard $i FAILED"; rc=1; fi
    done
    [ "$rc" -eq 0 ] && echo "=== regime=$regime done ===" || { echo "=== regime=$regime had failures ==="; exit 1; }
done
