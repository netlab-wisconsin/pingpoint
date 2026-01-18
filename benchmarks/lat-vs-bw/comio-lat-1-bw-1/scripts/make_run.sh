#!/bin/bash

# 4 combinates of lat: (local,remote) x bw: (local,remote)
make -f Makefile.run K1_PINNED_HBM=0 K2_PINNED_HBM=1
make -f Makefile.run  K1_PINNED_HBM=0 K2_PINNED_HBM=3
make -f Makefile.run  K1_PINNED_HBM=2 K2_PINNED_HBM=1
make -f Makefile.run  K1_PINNED_HBM=2 K2_PINNED_HBM=7