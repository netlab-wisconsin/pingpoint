#!/usr/bin/env python3
"""Summarize E2 accuracy/coverage/cost for multi-ping diagnostic episodes.

The raw E2 output gives per-bpx/window target bandwidth with and without the
bandwidth ping.  For a bpx set, this script treats each selected bpx as one
pinged target iteration in a diagnostic episode.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_BPX = tuple(range(10, 17))
FLOAT_FIELDS = (
    "capacity_gbps",
    "solo_probe_gbps",
    "probe_gbps",
    "probe_loss_pct",
    "solo_target_gbps",
    "target_gbps",
    "target_est_gbps",
    "abs_err_gbps",
    "rel_err_pct",
    "sum_gbps",
)
INT_FIELDS = ("bpx", "window", "thr_cyc", "informative")


def parse_bpx_list(text: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"bad bpx range: {part}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("empty bpx list")
    return tuple(dict.fromkeys(out))


def parse_raw(path: Path) -> tuple[str, list[dict[str, float | int]]]:
    pattern = path.stem
    rows: list[dict[str, float | int]] = []
    header: list[str] | None = None
    pattern_re = re.compile(r"\bpattern=([^ ]+)")

    with path.open(newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[bw-fidelity]"):
                match = pattern_re.search(line)
                if match:
                    pattern = match.group(1)
                continue
            if line.startswith("bpx,window,"):
                header = next(csv.reader([line]))
                continue
            if header is None or not line[0].isdigit():
                continue

            parts = next(csv.reader([line]))
            if len(parts) != len(header):
                continue
            row: dict[str, float | int] = dict(zip(header, parts))  # type: ignore[arg-type]
            for key in INT_FIELDS:
                row[key] = int(row[key])  # type: ignore[arg-type]
            for key in FLOAT_FIELDS:
                row[key] = float(row[key])  # type: ignore[arg-type]
            rows.append(row)

    if not rows:
        raise ValueError(f"no per-window rows found in {path}")
    return pattern, rows


def rel_time_from_gbps(gbps: float | int) -> float:
    value = float(gbps)
    if value <= 0.0:
        return math.nan
    return 1.0 / value


def evaluate(
    pattern: str,
    rows: list[dict[str, float | int]],
    bpx_set: tuple[int, ...],
    selection: str,
) -> dict[str, str | int | float]:
    by_window: dict[int, dict[int, dict[str, float | int]]] = defaultdict(dict)
    for row in rows:
        by_window[int(row["window"])][int(row["bpx"])] = row

    abs_errors: list[float] = []
    rel_errors: list[float] = []
    episode_slowdowns: list[float] = []
    episode_extra_vs_one_iter: list[float] = []
    single_ping_slowdowns: list[float] = []
    covered_windows = 0
    informative_pings = 0
    total_pings = 0

    for window in sorted(by_window):
        selected = [by_window[window][bpx] for bpx in bpx_set if bpx in by_window[window]]
        if len(selected) != len(bpx_set):
            continue
        total_pings += len(selected)

        baseline_times = [rel_time_from_gbps(row["solo_target_gbps"]) for row in selected]
        pinged_times = [rel_time_from_gbps(row["target_gbps"]) for row in selected]
        if all(math.isfinite(v) for v in baseline_times + pinged_times):
            extra_times = [pinged - base for pinged, base in zip(pinged_times, baseline_times)]
            baseline_sum = sum(baseline_times)
            if baseline_sum > 0.0:
                episode_slowdowns.append(sum(extra_times) / baseline_sum * 100.0)
            baseline_one_iter = baseline_sum / len(baseline_times)
            if baseline_one_iter > 0.0:
                episode_extra_vs_one_iter.append(sum(extra_times) / baseline_one_iter * 100.0)
            for pinged, base in zip(pinged_times, baseline_times):
                if base > 0.0:
                    single_ping_slowdowns.append((pinged - base) / base * 100.0)

        informative = [row for row in selected if int(row["informative"]) == 1]
        informative_pings += len(informative)
        if not informative:
            continue
        covered_windows += 1
        estimate = sum(float(row["target_est_gbps"]) for row in informative) / len(informative)
        truth = sum(float(row["target_gbps"]) for row in informative) / len(informative)
        error = abs(estimate - truth)
        abs_errors.append(error)
        if truth > 1e-9:
            rel_errors.append(error / truth * 100.0)

    windows = len(by_window)
    return {
        "pattern": pattern,
        "selection": selection,
        "num_involved_pings": len(bpx_set),
        "bpx_set": " ".join(str(bpx) for bpx in bpx_set),
        "mae_gbps": mean(abs_errors),
        "mape_pct": mean(rel_errors),
        "informative_window_coverage_pct": 100.0 * covered_windows / windows if windows else 0.0,
        "informative_ping_coverage_pct": 100.0 * informative_pings / total_pings if total_pings else 0.0,
        "episode_slowdown_pct": mean(episode_slowdowns),
        "episode_extra_vs_one_target_iter_pct": mean(episode_extra_vs_one_iter),
        "mean_single_ping_slowdown_pct": mean(single_ping_slowdowns),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def best_by_mape(
    pattern: str,
    rows: list[dict[str, float | int]],
    candidates: tuple[int, ...],
) -> list[dict[str, str | int | float]]:
    out: list[dict[str, str | int | float]] = []
    for k in range(1, len(candidates) + 1):
        evaluated = [
            evaluate(pattern, rows, combo, "best_mape")
            for combo in itertools.combinations(candidates, k)
        ]
        best = min(
            evaluated,
            key=lambda row: (
                float(row["mape_pct"]),
                -float(row["informative_window_coverage_pct"]),
                float(row["episode_extra_vs_one_target_iter_pct"]),
                str(row["bpx_set"]),
            ),
        )
        out.append(best)
    return out


def fmt_row(row: dict[str, str | int | float]) -> dict[str, str | int]:
    formatted: dict[str, str | int] = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.4f}"
        else:
            formatted[key] = value
    return formatted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare E2 bpx-set accuracy, informative coverage, and target overhead."
    )
    parser.add_argument("raw", nargs="+", type=Path, help="E2 raw .out file(s)")
    parser.add_argument(
        "--candidate-bpx",
        default="10-16",
        help="candidate bpx values/ranges for best-set search, default: 10-16",
    )
    parser.add_argument(
        "--explicit-set",
        action="append",
        default=[],
        help="extra bpx set to report, e.g. 10 or 10-16; may be repeated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write CSV here instead of stdout",
    )
    args = parser.parse_args()

    candidates = parse_bpx_list(args.candidate_bpx)
    explicit_sets = [parse_bpx_list(text) for text in args.explicit_set]

    output_rows: list[dict[str, str | int | float]] = []
    for raw_path in args.raw:
        pattern, rows = parse_raw(raw_path)
        output_rows.extend(best_by_mape(pattern, rows, candidates))
        for bpx_set in explicit_sets:
            output_rows.append(evaluate(pattern, rows, bpx_set, "explicit"))

    fieldnames = [
        "pattern",
        "selection",
        "num_involved_pings",
        "bpx_set",
        "mae_gbps",
        "mape_pct",
        "informative_window_coverage_pct",
        "informative_ping_coverage_pct",
        "episode_slowdown_pct",
        "episode_extra_vs_one_target_iter_pct",
        "mean_single_ping_slowdown_pct",
    ]

    out_f = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(fmt_row(row))
    finally:
        if args.output:
            out_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
