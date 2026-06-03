#!/bin/bash

i=46
HIP_VISIBLE_DEVICES=0 ../bin/acn-main-hop1-mi350-HBM-${i} 2>&1 | tee ../results-060226/acn-main-hop1-mi350-HBM-${i}.out