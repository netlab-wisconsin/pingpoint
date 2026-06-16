#!/usr/bin/env python3
"""Extract E3 summaries and an optional plot-ready adaptive time series."""

import csv
import pathlib
import sys


def row_values(text: str) -> list[str]:
    return next(csv.reader([text]))


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(f"usage: {sys.argv[0]} RAW_OUTPUT [TIMESERIES_CSV]", file=sys.stderr)
        return 1

    path = pathlib.Path(sys.argv[1])
    lines = path.read_text().splitlines()
    summary_header = next((line for line in lines if line.startswith("summary,policy,")), None)
    phase_header = next((line for line in lines if line.startswith("phase_summary,policy,")), None)
    diagnosis_header = next(
        (line for line in lines if line.startswith("diagnosis_summary,encounter,")), None
    )
    request_header = next((line for line in lines if line.startswith("request,policy,")), None)
    path_header = next((line for line in lines if line.startswith("path,policy,")), None)
    if any(header is None for header in (
            summary_header, phase_header, diagnosis_header, request_header, path_header)):
        print("missing request or summary CSV section", file=sys.stderr)
        return 1

    summary_rows = [line for line in lines if line.startswith("summary,") and line != summary_header]
    phase_rows = [
        line for line in lines if line.startswith("phase_summary,") and line != phase_header
    ]
    diagnosis_rows = [
        line for line in lines
        if line.startswith("diagnosis_summary,") and line != diagnosis_header
    ]
    adaptive_rows = [line for line in lines if line.startswith("request,adaptive,")]
    path_columns = row_values(path_header)
    path_request_idx = path_columns.index("request_idx")
    path_path = path_columns.index("path", 1)
    path_latency = path_columns.index("latency_mean_ns")
    adaptive_latency = {}
    for row_text in lines:
        if not row_text.startswith("path,adaptive,"):
            continue
        row = row_values(row_text)
        adaptive_latency[(row[path_request_idx], row[path_path])] = row[path_latency]

    writer = csv.writer(sys.stdout)
    writer.writerow(["E3 policy summary"])
    writer.writerow(row_values(summary_header)[1:])
    for row in summary_rows:
        writer.writerow(row_values(row)[1:])
    writer.writerow([])
    writer.writerow(["E3 phase summary"])
    writer.writerow(row_values(phase_header)[1:])
    for row in phase_rows:
        writer.writerow(row_values(row)[1:])
    writer.writerow([])
    writer.writerow(["Adaptive diagnosis comparison"])
    writer.writerow(row_values(diagnosis_header)[1:])
    for row in diagnosis_rows:
        writer.writerow(row_values(row)[1:])
    writer.writerow([])
    writer.writerow(["Adaptive diagnostic timeline"])
    request_columns = row_values(request_header)
    keep = [
        "request_idx", "phase", "anomaly_id", "target_slowdown_pct",
        "detector_target_slowdown_pct", "detector_score", "detector_positive", "triggered",
        "encounter", "diagnosis_step", "selected_path", "selected_cc", "bw_active", "bw_bpx",
        "selected_bw_loss_pct", "selected_bw_informative",
    ]
    indices = [request_columns.index(column) for column in keep]
    output_columns = keep[:6] + [
        "hot_path_latency_mean_ns",
        "selected_path_latency_mean_ns",
    ] + keep[6:]

    def timeline_row(row: list[str]) -> list[str]:
        values = [row[index] for index in indices]
        request_idx = row[request_columns.index("request_idx")]
        selected_path = row[request_columns.index("selected_path")]
        hot_latency = adaptive_latency.get((request_idx, "0"), "")
        selected_latency = adaptive_latency.get((request_idx, selected_path), "")
        return values[:6] + [hot_latency, selected_latency] + values[6:]

    writer.writerow(output_columns)
    for row_text in adaptive_rows:
        row = row_values(row_text)
        if row[request_columns.index("ground_truth_anomaly")] == "1" or \
                row[request_columns.index("triggered")] == "1" or \
                row[request_columns.index("bw_active")] == "1":
            writer.writerow(timeline_row(row))

    if len(sys.argv) == 3:
        timeseries_path = pathlib.Path(sys.argv[2])
        timeseries_path.parent.mkdir(parents=True, exist_ok=True)
        with timeseries_path.open("w", newline="") as output:
            timeseries_writer = csv.writer(output)
            timeseries_writer.writerow(output_columns)
            for row_text in adaptive_rows:
                row = row_values(row_text)
                timeseries_writer.writerow(timeline_row(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
