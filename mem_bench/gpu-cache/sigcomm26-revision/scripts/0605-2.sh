#!/bin/bash

HIP_VISIBLE_DEVICES=0 ./bin/acn-main-hop0-mi350-21-30 2>&1 | tee ./results/mi350-hop0-21-30.out &
HIP_VISIBLE_DEVICES=1 ./bin/acn-main-hop1-mi350-21-30 2>&1 | tee ./results/mi350-hop1-21-30.out &

HIP_VISIBLE_DEVICES=2 ./bin/acn-main-hop0-mi350-31-40 2>&1 | tee ./results/mi350-hop0-31-40.out &
HIP_VISIBLE_DEVICES=3 ./bin/acn-main-hop1-mi350-31-40 2>&1 | tee ./results/mi350-hop1-31-40.out &

HIP_VISIBLE_DEVICES=4 ./bin/acn-main-hop0-mi350-41-46 2>&1 | tee ./results/mi350-hop0-41-46.out &
HIP_VISIBLE_DEVICES=5 ./bin/acn-main-hop1-mi350-41-46 2>&1 | tee ./results/mi350-hop1-41-46.out &

HIP_VISIBLE_DEVICES=6 ./bin/acn-main-hop0-mi350-47-52 2>&1 | tee ./results/mi350-hop0-47-52.out &
HIP_VISIBLE_DEVICES=7 ./bin/acn-main-hop1-mi350-47-52 2>&1 | tee ./results/mi350-hop1-47-52.out &