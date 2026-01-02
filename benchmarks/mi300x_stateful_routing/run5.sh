#!/bin/bash
set -euo pipefail

mkdir -p bin

# ./bin/stateful_routing_0233 |& tee results/1230/all/0233.out || true

# ./bin/stateful_routing_0253 |& tee results/1230/all/0253.out || true

# ./bin/stateful_routing_0271 |& tee results/1230/all/0271.out || true

./bin/stateful_routing_0273 |& tee results/1230/all/0273.out || true
