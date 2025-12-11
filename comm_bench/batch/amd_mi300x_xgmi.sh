#!/bin/bash

remove_read="'read': "
remove_write="'write': "

if [ $# -lt 2 ]
  then
    echo "No arguments supplied."
    echo "Usage: $0 [Sampling Interval (second)] [File Name]"
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="./outputs_xgmi/"
mkdir -p $OUTPUT_PATH
FILE_PATH="${OUTPUT_PATH}/${TIMESTAMP}_${2}.csv"

saved_IFS="$IFS"
IFS=$'\n'
# Printing Header
TEMP_OUT=($(amd-smi xgmi -m --csv))
header=$(echo ${TEMP_OUT[0]} | sed -e 's/\r//g')
echo "Timestamp,gpu0r0,gpu0w0,gpu0r1,gpu0w1,gpu0r2,gpu0w2,gpu0r3,gpu0w3,gpu0r4,gpu0w4,gpu0r5,gpu0w5,gpu0r6,gpu0w6,gpu0r7,gpu0w7,gpu1r0,gpu1w0,gpu1r1,gpu1w1,gpu1r2,gpu1w2,gpu1r3,gpu1w3,gpu1r4,gpu1w4,gpu1r5,gpu1w5,gpu1r6,gpu1w6,gpu1r7,gpu1w7,gpu2r0,gpu2w0,gpu2r1,gpu2w1,gpu2r2,gpu2w2,gpu2r3,gpu2w3,gpu2r4,gpu2w4,gpu2r5,gpu2w5,gpu2r6,gpu2w6,gpu2r7,gpu2w7,gpu3r0,gpu3w0,gpu3r1,gpu3w1,gpu3r2,gpu3w2,gpu3r3,gpu3w3,gpu3r4,gpu3w4,gpu3r5,gpu3w5,gpu3r6,gpu3w6,gpu3r7,gpu3w7,gpu4r0,gpu4w0,gpu4r1,gpu4w1,gpu4r2,gpu4w2,gpu4r3,gpu4w3,gpu4r4,gpu4w4,gpu4r5,gpu4w5,gpu4r6,gpu4w6,gpu4r7,gpu4w7,gpu5r0,gpu5w0,gpu5r1,gpu5w1,gpu5r2,gpu5w2,gpu5r3,gpu5w3,gpu5r4,gpu5w4,gpu5r5,gpu5w5,gpu5r6,gpu5w6,gpu5r7,gpu5w7,gpu6r0,gpu6w0,gpu6r1,gpu6w1,gpu6r2,gpu6w2,gpu6r3,gpu6w3,gpu6r4,gpu6w4,gpu6r5,gpu6w5,gpu6r6,gpu6w6,gpu6r7,gpu6w7,gpu7r0,gpu7w0,gpu7r1,gpu7w1,gpu7r2,gpu7w2,gpu7r3,gpu7w3,gpu7r4,gpu7w4,gpu7r5,gpu7w5,gpu7r6,gpu7w6,gpu7r7,gpu7w7" | tee -a $FILE_PATH

# Printing Contents
while true
do
    IFS=$'\n'
    TIME="$(date +%s)"
    TEMP_OUT=($(amd-smi xgmi -m --csv))
    gpu0=$(echo ${TEMP_OUT[1]} | sed -e 's/\r//g')
    gpu1=$(echo ${TEMP_OUT[2]} | sed -e 's/\r//g')
    gpu2=$(echo ${TEMP_OUT[3]} | sed -e 's/\r//g')
    gpu3=$(echo ${TEMP_OUT[4]} | sed -e 's/\r//g')
    gpu4=$(echo ${TEMP_OUT[5]} | sed -e 's/\r//g')
    gpu5=$(echo ${TEMP_OUT[6]} | sed -e 's/\r//g')
    gpu6=$(echo ${TEMP_OUT[7]} | sed -e 's/\r//g')
    gpu7=$(echo ${TEMP_OUT[8]} | sed -e 's/\r//g')

    IFS=","
    read -ra temp_gpu <<< "$gpu0"
    gpu0r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu0r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu0r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu0r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu0r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu0r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu0r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu0r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu0w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    read -ra temp_gpu <<< "$gpu1"
    gpu1r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu1r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu1r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu1r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu1r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu1r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu1r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu1r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu1w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    read -ra temp_gpu <<< "$gpu2"
    gpu2r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu2r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu2r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu2r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu2r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu2r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu2r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu2r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu2w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    read -ra temp_gpu <<< "$gpu3"
    gpu3r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu3r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu3r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu3r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu3r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu3r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu3r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu3r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu3w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    read -ra temp_gpu <<< "$gpu4"
    gpu4r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu4r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu4r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu4r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu4r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu4r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu4r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu4r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu4w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    read -ra temp_gpu <<< "$gpu5"
    gpu5r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu5r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu5r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu5r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu5r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu5r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu5r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu5r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu5w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    read -ra temp_gpu <<< "$gpu6"
    gpu6r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu6r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu6r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu6r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu6r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu6r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu6r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu6r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu6w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    read -ra temp_gpu <<< "$gpu7"
    gpu7r0=$(echo ${temp_gpu[7]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w0=$(echo ${temp_gpu[8]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu7r1=$(echo ${temp_gpu[11]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w1=$(echo ${temp_gpu[12]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu7r2=$(echo ${temp_gpu[15]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w2=$(echo ${temp_gpu[16]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu7r3=$(echo ${temp_gpu[19]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w3=$(echo ${temp_gpu[20]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu7r4=$(echo ${temp_gpu[23]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w4=$(echo ${temp_gpu[24]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu7r5=$(echo ${temp_gpu[27]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w5=$(echo ${temp_gpu[28]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu7r6=$(echo ${temp_gpu[31]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w6=$(echo ${temp_gpu[32]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g")
    gpu7r7=$(echo ${temp_gpu[35]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/read//g" | sed -e "s/ //g")
    gpu7w7=$(echo ${temp_gpu[36]} | sed -e "s/'//g" | sed -e "s/://g" | sed -e "s/write//g" | sed -e "s/ //g" | sed -e "s/}//g" | sed -e "s/]//g")

    echo "${TIME},$gpu0r0,$gpu0w0,$gpu0r1,$gpu0w1,$gpu0r2,$gpu0w2,$gpu0r3,$gpu0w3,$gpu0r4,$gpu0w4,$gpu0r5,$gpu0w5,$gpu0r6,$gpu0w6,$gpu0r7,$gpu0w7,$gpu1r0,$gpu1w0,$gpu1r1,$gpu1w1,$gpu1r2,$gpu1w2,$gpu1r3,$gpu1w3,$gpu1r4,$gpu1w4,$gpu1r5,$gpu1w5,$gpu1r6,$gpu1w6,$gpu1r7,$gpu1w7,$gpu2r0,$gpu2w0,$gpu2r1,$gpu2w1,$gpu2r2,$gpu2w2,$gpu2r3,$gpu2w3,$gpu2r4,$gpu2w4,$gpu2r5,$gpu2w5,$gpu2r6,$gpu2w6,$gpu2r7,$gpu2w7,$gpu3r0,$gpu3w0,$gpu3r1,$gpu3w1,$gpu3r2,$gpu3w2,$gpu3r3,$gpu3w3,$gpu3r4,$gpu3w4,$gpu3r5,$gpu3w5,$gpu3r6,$gpu3w6,$gpu3r7,$gpu3w7,$gpu4r0,$gpu4w0,$gpu4r1,$gpu4w1,$gpu4r2,$gpu4w2,$gpu4r3,$gpu4w3,$gpu4r4,$gpu4w4,$gpu4r5,$gpu4w5,$gpu4r6,$gpu4w6,$gpu4r7,$gpu4w7,$gpu5r0,$gpu5w0,$gpu5r1,$gpu5w1,$gpu5r2,$gpu5w2,$gpu5r3,$gpu5w3,$gpu5r4,$gpu5w4,$gpu5r5,$gpu5w5,$gpu5r6,$gpu5w6,$gpu5r7,$gpu5w7,$gpu6r0,$gpu6w0,$gpu6r1,$gpu6w1,$gpu6r2,$gpu6w2,$gpu6r3,$gpu6w3,$gpu6r4,$gpu6w4,$gpu6r5,$gpu6w5,$gpu6r6,$gpu6w6,$gpu6r7,$gpu6w7,$gpu7r0,$gpu7w0,$gpu7r1,$gpu7w1,$gpu7r2,$gpu7w2,$gpu7r3,$gpu7w3,$gpu7r4,$gpu7w4,$gpu7r5,$gpu7w5,$gpu7r6,$gpu7w6,$gpu7r7,$gpu7w7" | tee -a $FILE_PATH

    sleep $1
done
IFS="$saved_IFS"