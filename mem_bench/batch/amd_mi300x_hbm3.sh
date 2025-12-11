#!/bin/bash

if [ $# -lt 3 ]
  then
    echo "No arguments supplied."
    echo "Usage: $0 [GPU_Index] [Sampling Interval (second)] [File Name]"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="./outputs_hbm3/"
mkdir -p $OUTPUT_PATH
FILE_PATH="${OUTPUT_PATH}/${TIMESTAMP}_${3}.csv"

#saved_IFS="$IFS"
#IFS=$'\n'
# Printing Header
echo "Timestamp,GPU" | tee -a $FILE_PATH

# Printing Contents
while true
do
    TIME="$(date +%s)"
    TEMP_OUT=($(rocm-smi -d $1 --showmemuse | grep "Memory Activity:"))
    gpu=$(echo ${TEMP_OUT[4]} | sed -e 's/\r//g')
    
    echo "${TIME},$gpu" | tee -a $FILE_PATH
    sleep $2
done
#IFS="$saved_IFS"