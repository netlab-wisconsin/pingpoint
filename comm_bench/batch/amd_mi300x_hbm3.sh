#!/bin/bash

if [ $# -lt 2 ]
  then
    echo "No arguments supplied."
    echo "Usage: $0 [Sampling Interval (second)] [File Name]"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="./outputs_hbm3/"
mkdir -p $OUTPUT_PATH
FILE_PATH="${OUTPUT_PATH}/${TIMESTAMP}_${2}.csv"

#saved_IFS="$IFS"
#IFS=$'\n'
# Printing Header
echo "Timestamp,GPU_0,GPU_1,GPU_2,GPU_3,GPU_4,GPU_5,GPU_6,GPU_7" | tee -a $FILE_PATH

# Printing Contents
while true
do
    TIME="$(date +%s)"
    TEMP_OUT=($(rocm-smi --showmemuse | grep "Memory Activity:"))
    gpu0=$(echo ${TEMP_OUT[4]} | sed -e 's/\r//g')
    gpu1=$(echo ${TEMP_OUT[9]} | sed -e 's/\r//g')
    gpu2=$(echo ${TEMP_OUT[14]} | sed -e 's/\r//g')
    gpu3=$(echo ${TEMP_OUT[19]} | sed -e 's/\r//g')
    gpu4=$(echo ${TEMP_OUT[24]} | sed -e 's/\r//g')
    gpu5=$(echo ${TEMP_OUT[29]} | sed -e 's/\r//g')
    gpu6=$(echo ${TEMP_OUT[34]} | sed -e 's/\r//g')
    gpu7=$(echo ${TEMP_OUT[39]} | sed -e 's/\r//g')
    
    echo "${TIME},$gpu0,$gpu1,$gpu2,$gpu3,$gpu4,$gpu5,$gpu6,$gpu7" | tee -a $FILE_PATH
    sleep $1
done
#IFS="$saved_IFS"