#!/bin/bash

SUFFIX="L2"

HIP_VISIBLE_DEVICES=0 ./bin/acn-main-hop0-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop0-${SUFFIX}.out &
HIP_VISIBLE_DEVICES=1 ./bin/acn-main-hop1-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop1-${SUFFIX}.out &
HIP_VISIBLE_DEVICES=2 ./bin/acn-main-hop2-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop2-${SUFFIX}.out &
HIP_VISIBLE_DEVICES=3 ./bin/acn-main-hop3-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop3-${SUFFIX}.out &

SUFFIX="LLC"

HIP_VISIBLE_DEVICES=4 ./bin/acn-main-hop0-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop0-${SUFFIX}.out &
HIP_VISIBLE_DEVICES=5 ./bin/acn-main-hop1-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop1-${SUFFIX}.out &
HIP_VISIBLE_DEVICES=6 ./bin/acn-main-hop2-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop2-${SUFFIX}.out &
HIP_VISIBLE_DEVICES=7 ./bin/acn-main-hop3-${SUFFIX} 2>&1 | tee results-loaded-lat/acn-main-hop3-${SUFFIX}.out &

wait
