import re
from collections import defaultdict

# Input/output
input_file = 'result/bmk2_pg4.log' # Change this to your actual file name
output_file = 'result/bmk2_pg4.csv' # Change this to your actual file name

# Dictionary to store cycles: key=(pid,cid,p,pcid) → cycles
data = {}

# -----------------------
# Step 1: Parse the log
# -----------------------
with open(input_file, 'r') as f:
    for line in f:
        if 'final read:' in line:
            match = re.search(
                r'\(pid: (\d+), cid: (\d+)\) \(p: (\d+), pcid: (\d+)\).*?(\d+) cycles',
                line
            )
            if match:
                pid  = int(match.group(1))
                cid  = int(match.group(2))
                p    = int(match.group(3))
                pcid = int(match.group(4))
                cycles = int(match.group(5))
                data[(pid, cid, p, pcid)] = cycles

# -----------------------
# Step 2: Organize cycles
# -----------------------
# Group by (pid,cid,p)
grouped = defaultdict(dict)

for (pid, cid, p, pcid), cycles in data.items():
    grouped[(pid, cid, p)][pcid] = cycles

# -----------------------
# Step 3: Write cycles-only CSV
# -----------------------
with open(output_file, 'w') as f:
    # Sort keys for stable ordering
    for key in sorted(grouped.keys()):
        pid, cid, p = key
        # Collect cycles for pcid 0..31
        line_values = []
        for pcid in range(32):
            if pcid in grouped[key]:
                line_values.append(str(grouped[key][pcid]))
            else:
                line_values.append('N/A')
        # Write as CSV row (no metadata, cycles only)
        f.write(','.join(line_values) + '\n')

print(f"Wrote cycles-only CSV to {output_file}")
