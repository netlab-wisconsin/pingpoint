#!/bin/bash
set -euo pipefail

export HIP_VISIBLE_DEVICES="0"

for x in 2611 6211 6255; do
  echo "Running fixed case $x"
  ./bin/stateful_routing_$x |& tee results/1230/all/$x.out4 || true
done