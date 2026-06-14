#!/usr/bin/env python3

import argparse
import collections
import csv


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description="Summarize overhead-moe CSV output")
    parser.add_argument("result")
    args = parser.parse_args()

    samples = []
    summaries = []
    with open(args.result, newline="") as result:
        for row in csv.reader(result):
            if row and row[0] == "sample" and row[1] != "regime":
                samples.append(row)
            elif row and row[0] == "summary" and row[1] != "regime":
                summaries.append(row)

    sample_counts = collections.Counter(row[1] for row in samples)
    summary_counts = collections.Counter(row[1] for row in summaries)
    schedule_mismatches = 0
    for row in samples:
        rate_tenths = round(float(row[3]) * 10)
        request_idx = int(row[4])
        expected = ((request_idx + 1) * rate_tenths // 10) > (
            request_idx * rate_tenths // 10
        )
        schedule_mismatches += int(row[5]) != expected
    rate_mismatches = sum(
        abs(float(row[3]) - float(row[4])) > 1e-9 for row in summaries
    )

    print(f"samples={dict(sample_counts)} summaries={dict(summary_counts)}")
    print(
        f"sample_schedule_mismatches={schedule_mismatches} "
        f"summary_rate_mismatches={rate_mismatches}"
    )
    for regime in ("prefill", "decode"):
        rows = [row for row in summaries if row[1] == regime]
        scaling_mae = mean(
            abs(float(row[13]) - float(row[4]) * float(row[10])) for row in rows
        )
        wall_gap = mean(float(row[18]) - float(row[13]) for row in rows)
        print(
            f"{regime}: scaling_mae_pp={scaling_mae:.4f} "
            f"mean_wall_minus_target_pp={wall_gap:.4f}"
        )

    print("rate_1_target_slowdown_pct_and_ping_gbps")
    print("bpx,prefill_slowdown,prefill_gbps,decode_slowdown,decode_gbps")
    for bpx in (1, 4, 8, 10, 11, 12, 13, 14, 15, 16):
        values = {}
        for regime in ("prefill", "decode"):
            row = next(
                row
                for row in summaries
                if row[1] == regime and int(row[2]) == bpx and float(row[4]) == 1.0
            )
            values[regime] = (float(row[10]), float(row[20]))
        print(
            f"{bpx},{values['prefill'][0]:.2f},{values['prefill'][1]:.2f},"
            f"{values['decode'][0]:.2f},{values['decode'][1]:.2f}"
        )


if __name__ == "__main__":
    main()
