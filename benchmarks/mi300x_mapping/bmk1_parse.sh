#!/bin/bash

# Parse bmk1 output into csv
# Format: cache_line_id, cycle_xcd0, cycle_xcd1, ..., cycle_xcd7
# Usage: ./bmk1_parse.sh <bmk1_output.log> > <bmk1_output.csv>

INPUT_FILE="$1"

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

grep "tid: 0)" "$INPUT_FILE" | \
awk '
{
    # extract iteration
    iter_raw = $2
    gsub(":", "", iter_raw)

    # extract bid
    bid_raw = $4
    gsub("[^0-9]", "", bid_raw)

    # extract cycle number: second-to-last field
    cycle = $(NF-1)

    # if new iteration, flush previous row
    if (iter_raw != current_iter) {
        if (current_iter != "") {
            print line
        }
        current_iter = iter_raw
        line = current_iter
    }

    # append cycle for this bid
    line = line "," cycle
}
END {
    # print final stored line
    if (line != "") print line
}'
