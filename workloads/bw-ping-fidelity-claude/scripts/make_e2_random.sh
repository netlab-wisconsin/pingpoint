#!/usr/bin/env bash
# Build bw_fidelity for the RANDOM target-load pattern. Run before run_e2_random.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

cd "$PROJ"
make clean TARGET="$PROJ/bin/bw_fidelity_random"
make TARGET="$PROJ/bin/bw_fidelity_random" CCFLAGS="-DTARGET_LOAD_PATTERN=LOAD_RANDOM"
