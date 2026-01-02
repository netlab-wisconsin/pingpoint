#!/bin/bash

# Check if input file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE="$1.csv"

# 1. sed '1,3d': Skips the first 3 lines.
# 2. sed 's/\*\]//g': Removes the source tags.
# 3. tr -d '\n': Removes all newlines to create a single stream of text.
# 4. sed 's/\([0-9]\{1,2\} 128\)/\n\1/g': Re-inserts newlines before each new record 
#    (identifying a record by a 1-2 digit number followed by '128').
# 5. awk: Picks the specific fields and formats them.

sed '1,3d' "$INPUT_FILE" | \
sed 's/\*\]//g' | \
tr -d '\n' | \
sed 's/\([0-9]\{1,2\} 128\)/\n\1/g' | \
awk '
NF > 40 {
    # Extract GB value from the last field (e.g., "5.9GB/s" -> "5.9")
    gb = $(NF); 
    sub(/GB\/s/, "", gb);
    
    # Print requested columns
    # bpx($1), gb, clk($4), followed by the cycles
    # printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", \
    # $1, gb, $4, $7, $12, $17, $22, $27, $32, $37, $42
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", \
    $4, gb, $7, $12, $17, $22, $27, $32, $37, $42
}' > "$OUTPUT_FILE"

echo "Parsing complete. Data saved to $OUTPUT_FILE"