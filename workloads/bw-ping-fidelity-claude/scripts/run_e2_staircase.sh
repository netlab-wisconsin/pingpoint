#!/usr/bin/env bash
# E2 (bw-ping-fidelity): run the STAIRCASE target-load fidelity sweep.
# Output: per-window (target_gbps, probe_gbps) across bpx=1..16 -> $OUT/bw_fidelity_staircase.out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="${WORK:?set WORK}/ici-workspace/sigcomm-exp/revision/overhead/raw"
BIN="$PROJ/bin/bw_fidelity_staircase"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "build first: scripts/make_e2_staircase.sh"; exit 1; }

"$BIN" 2>&1 | tee "$OUT/bw_fidelity_staircase.out"
