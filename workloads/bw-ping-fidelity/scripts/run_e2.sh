#!/usr/bin/env bash
# E2 (bw-ping-fidelity): build + run the calibrated-source vs bw-probe fidelity sweep.
# Output: per-window (target_gbps, probe_gbps) across bpx=1..16 -> $OUT/bw_fidelity.out
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="${WORK:?set WORK}/ici-workspace/sigcomm-exp/revision/overhead/raw"
BIN="$PROJ/bin/bw_fidelity"
mkdir -p "$OUT"

cd "$PROJ"
make

"$BIN" 2>&1 | tee "$OUT/bw_fidelity.out"
