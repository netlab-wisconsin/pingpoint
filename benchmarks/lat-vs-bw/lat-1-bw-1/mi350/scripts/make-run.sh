#!/bin/bash

## Note that K1 going to LLC or HBM is controlled in the main by K1_UTILIZE_HBM
## Set accordingly to test COM-IO or IO-MEM.
## Just keep K2 to go to HBM (also in the main)

SUFFIX=iomem # set accordingly
K1_XCD=0
K2_XCD=1

for K1_CC in 0 1; do
    for K2_CC in 0 1; do
        make -B -f Makefile.run \
            K1_PINNED_XCD=${K1_XCD} K1_PINNED_CC=${K1_CC} \
            K2_PINNED_XCD=${K2_XCD} K2_PINNED_CC=${K2_CC} \
            SUFFIX=${SUFFIX}
    done
done
