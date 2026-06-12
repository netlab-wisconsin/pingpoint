#!/bin/bash

i=45
HIP_VISIBLE_DEVICES=0 ../bin/acn-main-hop0-mi350-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop0-mi350-HBM-${i}.out &

i=46
HIP_VISIBLE_DEVICES=1 ../bin/acn-main-hop0-mi350-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop0-mi350-HBM-${i}.out &

i=47
HIP_VISIBLE_DEVICES=2 ../bin/acn-main-hop0-mi350-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop0-mi350-HBM-${i}.out &

wait