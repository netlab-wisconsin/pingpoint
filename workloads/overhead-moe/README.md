# Overhead MoE (E1)

This benchmark measures the impact of bandwidth pings on synthetic FFN1 prefill and decode
targets. It intentionally does not modify or depend on the old `workloads/moe/moe_serving.*`
experiment.

Key properties:

- Target blocks always use local slots `0..15`; ping blocks are placed after them.
- Ping lifetime is bounded by target completion, so no long ping tail serializes later requests.
- Each measured request is paired with a no-ping pass using the identical `bpx` grid shape.
- Sampling rate `k/10` is exact and evenly spaced:
  `floor((i + 1) * k / 10) > floor(i * k / 10)`.
- Condition order deterministically alternates low/high rates and low/high `bpx` values so
  ping strength is not correlated with elapsed run time.
- Output separates conditional pinged-request slowdown from workload-average slowdown.
- Achieved ping bandwidth and bytes per pinged request are measured.
- The prefill target reuses each loaded weight across four token rows; decode streams each
  expert's weights once.
- The ping uses a fixed 512 MB streaming working set, larger than LLC, selected with the original
  MoE bandwidth-ping home classifier.

Build and run:

```bash
scripts/build.sh
scripts/run.sh
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/overhead-moe/scripts/summarize.py \
  /path/to/overhead_moe_both_n50.out [rate]
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/overhead-moe/scripts/parse_all_rates.sh
```

`N_MEASURED` must be a positive multiple of 10. `REGIME` may be `prefill`, `decode`, or `both`.
`OVERHEAD_BPX` and `OVERHEAD_RATE_TENTHS` optionally select one configuration for validation.
Raw outputs are written under the result directory's `raw/` subdirectory. The relocated
summarizer's optional `rate` is one of `0.1, 0.2, ..., 1.0` and defaults to `1.0`.
`parse_all_rates.sh` writes one valid CSV table per rate under the result directory's `parsed/`
subdirectory.
