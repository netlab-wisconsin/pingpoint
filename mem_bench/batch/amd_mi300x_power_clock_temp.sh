#!/bin/bash

if [ $# -lt 3 ]
  then
    echo "No arguments supplied."
    echo "Usage: $0 [GPU_Index] [Sampling Interval (second)] [File Name]"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="./outputs_smi/"
mkdir -p $OUTPUT_PATH
FILE_PATH="${OUTPUT_PATH}/${TIMESTAMP}_${3}.csv"

# Printing Header
TEMP_OUT=($(amd-smi metric --csv -g $1 --power --clock --temperature))
echo "Timestamp,${TEMP_OUT[0]}" | tee -a $FILE_PATH

# Printing Contents
while true
do
    TIME="$(date +%s)"
    TEMP_OUT=($(amd-smi metric --csv -g $1 --power --clock --temperature))
    echo "${TIME},${TEMP_OUT[1]}" | tee -a $FILE_PATH
    sleep $2
done
