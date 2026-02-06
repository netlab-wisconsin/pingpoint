#!/bin/bash

# Check if a filename was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <logfile>"
    exit 1
fi

LOG_FILE=$1

# Parse the file
# 1. Search for lines containing "VRAM Total Used Memory"
# 2. Print the last field ($NF) on that line, which is the number
awk '/VRAM Total Used Memory/ {print $NF}' "$LOG_FILE"