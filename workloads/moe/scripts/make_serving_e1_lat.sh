#!/usr/bin/env bash
# Build moe_serving for E1 LATENCY runs: baseline + K1 (latency) pings; bandwidth (K2) compiled out.
# Run before scripts/run_e1_lat.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

cd "$PROJ"
make -f Makefile.serving clean TARGET="$PROJ/bin/moe_serving_lat"
make -f Makefile.serving TARGET="$PROJ/bin/moe_serving_lat" CCFLAGS="-DDISABLE_K1_PLANS=0 -DDISABLE_K2_PLANS=1"
