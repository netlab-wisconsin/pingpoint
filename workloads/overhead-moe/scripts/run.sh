#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
OUT="/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/overhead-moe"
N_MEASURED="${N_MEASURED:-20}"
REGIME="${REGIME:-both}"

mkdir -p "$OUT"
"$PROJ/bin/overhead_moe" "$REGIME" "$N_MEASURED" 2>&1 | tee "$OUT/overhead_moe_${REGIME}_n${N_MEASURED}.out"

