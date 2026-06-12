#!/bin/bash

HIP_VISIBLE_DEVICES=0 ./bin/acn-main-hop0-mi350-1-10 2>&1 | tee ./results/mi350-hop0-1-10.out &
HIP_VISIBLE_DEVICES=1 ./bin/acn-main-hop1-mi350-1-10 2>&1 | tee ./results/mi350-hop1-1-10.out &

HIP_VISIBLE_DEVICES=2 ./bin/acn-main-hop0-mi350-11-20 2>&1 | tee ./results/mi350-hop0-11-20.out &
HIP_VISIBLE_DEVICES=3 ./bin/acn-main-hop1-mi350-11-20 2>&1 | tee ./results/mi350-hop1-11-20.out &