#!/usr/bin/env bash
# Build bw_fidelity for the STAIRCASE target-load pattern. Run before run_e2_staircase.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

cd "$PROJ"
make clean TARGET="$PROJ/bin/bw_fidelity_staircase"
make TARGET="$PROJ/bin/bw_fidelity_staircase" CCFLAGS="-DTARGET_LOAD_PATTERN=LOAD_STAIRCASE"
