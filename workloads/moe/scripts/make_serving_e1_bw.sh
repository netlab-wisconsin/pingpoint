#!/usr/bin/env bash
# Build moe_serving for E1 BANDWIDTH runs: baseline + K2 (bandwidth) pings; latency (K1) compiled out.
# Run before scripts/run_e1_bw.sh or scripts/run_e1_bw_8gpu.sh.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"

cd "$PROJ"
make -f Makefile.serving clean TARGET="$PROJ/bin/moe_serving_bw"
make -f Makefile.serving TARGET="$PROJ/bin/moe_serving_bw" CCFLAGS="-DDISABLE_K1_PLANS=1 -DDISABLE_K2_PLANS=0"
