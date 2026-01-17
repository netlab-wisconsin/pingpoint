#!/bin/bash

BASE_DIR=$(pwd)
BIN_DIR="$BASE_DIR/bin"

# 4 combinates of lat: (local,remote) x bw: (local,remote)
make K1_PINNED_HBM=0 K2_PINNED_HBM=1
make K1_PINNED_HBM=0 K2_PINNED_HBM=3
make K1_PINNED_HBM=2 K2_PINNED_HBM=1
make K1_PINNED_HBM=2 K2_PINNED_HBM=7