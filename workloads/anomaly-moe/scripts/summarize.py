#!/usr/bin/env python3
"""Extract the E3 summary and adaptive timeline from an anomaly-moe raw output."""

import csv
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} RAW_OUTPUT", file=sys.stderr)
        return 1

    path = pathlib.Path(sys.argv[1])
    lines = path.read_text().splitlines()
    summary_header = next((line for line in lines if line.startswith("summary,policy,")), None)
    request_header = next((line for line in lines if line.startswith("request,policy,")), None)
    if summary_header is None or request_header is None:
        print("missing request or summary CSV section", file=sys.stderr)
        return 1

    summary_rows = [line for line in lines if line.startswith("summary,") and line != summary_header]
    adaptive_rows = [
        line for line in lines
        if line.startswith("request,adaptive,")
    ]

    writer = csv.writer(sys.stdout)
    writer.writerow(["E3 policy summary"])
    writer.writerow(next(csv.reader([summary_header]))[1:])
    for row in summary_rows:
        writer.writerow(next(csv.reader([row]))[1:])
    writer.writerow([])
    writer.writerow(["Adaptive anomaly/trigger timeline"])
    request_columns = next(csv.reader([request_header]))
    keep = [
        "request_idx", "ground_truth_anomaly", "detector_score", "detector_positive",
        "triggered", "selected_path", "bw_active", "target_ns", "bw0_loss_pct",
        "bw0_informative",
    ]
    indices = [request_columns.index(column) for column in keep]
    writer.writerow(keep)
    for row_text in adaptive_rows:
        row = next(csv.reader([row_text]))
        if row[request_columns.index("ground_truth_anomaly")] == "1" or \
                row[request_columns.index("triggered")] == "1" or \
                row[request_columns.index("bw_active")] == "1":
            writer.writerow([row[index] for index in indices])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

