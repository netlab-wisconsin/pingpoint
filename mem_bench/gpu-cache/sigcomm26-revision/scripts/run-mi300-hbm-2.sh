#!/bin/bash

i=48
HIP_VISIBLE_DEVICES=0 ../bin/acn-main-hop2-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop2-mi300-HBM-${i}.out &

i=49
HIP_VISIBLE_DEVICES=1 ../bin/acn-main-hop2-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop2-mi300-HBM-${i}.out &

i=50
HIP_VISIBLE_DEVICES=2 ../bin/acn-main-hop2-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop2-mi300-HBM-${i}.out &

i=51
HIP_VISIBLE_DEVICES=3 ../bin/acn-main-hop2-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop2-mi300-HBM-${i}.out &

i=48
HIP_VISIBLE_DEVICES=4 ../bin/acn-main-hop3-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop3-mi300-HBM-${i}.out &

i=49
HIP_VISIBLE_DEVICES=5 ../bin/acn-main-hop3-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop3-mi300-HBM-${i}.out &

i=50
HIP_VISIBLE_DEVICES=6 ../bin/acn-main-hop3-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop3-mi300-HBM-${i}.out &

i=51
HIP_VISIBLE_DEVICES=7 ../bin/acn-main-hop3-mi300-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop3-mi300-HBM-${i}.out &

wait