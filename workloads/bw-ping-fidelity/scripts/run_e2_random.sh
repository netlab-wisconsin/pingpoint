#!/usr/bin/env bash
# E2 (bw-ping-fidelity): run the RANDOM target-load fidelity sweep.
# Output: calibration, per-window estimates, and error summaries.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="${BW_FIDELITY_OUT:-${WORK:?set WORK}/ici-workspace/sigcomm-exp/revision/bw-ping-fidelity}"
BIN="$PROJ/bin/bw_fidelity_random"
mkdir -p "$OUT"
[ -x "$BIN" ] || { echo "build first: scripts/make_e2_random.sh"; exit 1; }

"$BIN" 2>&1 | tee "$OUT/bw_fidelity_random.out"
