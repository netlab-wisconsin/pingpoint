#!/bin/bash

if [ $# -lt 2 ]
  then
    echo "No arguments supplied."
    echo "Usage: $0 [Sampling Interval (second)] [File Name]"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="./outputs_smi/"
mkdir -p $OUTPUT_PATH
FILE_PATH="${OUTPUT_PATH}/${TIMESTAMP}_${2}.csv"

# Printing Header
TEMP_OUT=($(amd-smi metric --csv --power --clock --temperature))
header=$(echo ${TEMP_OUT[0]} | sed -e 's/\r//g')
echo "Timestamp,$header,$header,$header,$header,$header,$header,$header,$header," | tee -a $FILE_PATH

# Printing Contents
while true
do
    TIME="$(date +%s)"
    TEMP_OUT=($(amd-smi metric --csv --power --clock --temperature))
    gpu0=$(echo ${TEMP_OUT[1]} | sed -e 's/\r//g')
    gpu1=$(echo ${TEMP_OUT[2]} | sed -e 's/\r//g')
    gpu2=$(echo ${TEMP_OUT[3]} | sed -e 's/\r//g')
    gpu3=$(echo ${TEMP_OUT[4]} | sed -e 's/\r//g')
    gpu4=$(echo ${TEMP_OUT[5]} | sed -e 's/\r//g')
    gpu5=$(echo ${TEMP_OUT[6]} | sed -e 's/\r//g')
    gpu6=$(echo ${TEMP_OUT[7]} | sed -e 's/\r//g')
    gpu7=$(echo ${TEMP_OUT[8]} | sed -e 's/\r//g')
    
    echo "${TIME},$gpu0,$gpu1,$gpu2,$gpu3,$gpu4,$gpu5,$gpu6,$gpu7" | tee -a $FILE_PATH
    sleep $1
done
