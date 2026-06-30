#!/bin/bash

i=42
HIP_VISIBLE_DEVICES=0 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

i=43
HIP_VISIBLE_DEVICES=1 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

i=44
HIP_VISIBLE_DEVICES=2 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

i=45
HIP_VISIBLE_DEVICES=3 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

i=46
HIP_VISIBLE_DEVICES=4 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

i=47
HIP_VISIBLE_DEVICES=5 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

i=48
HIP_VISIBLE_DEVICES=6 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

i=49
HIP_VISIBLE_DEVICES=7 ./bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee results-063026/acn-main-hop1-mi350-HBM-${i}.out &

wait
