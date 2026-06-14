# E1 Final Results

Final run:

```text
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/overhead-moe/
  overhead_moe_final_both_n50.out
```

The run contains 50 paired baseline/workload requests for every combination of two target
regimes, ten bandwidth-ping strengths, and ten exact sampling rates.

## Validation

- 5,000 raw samples and 100 condition summaries are present for each regime.
- All 200 conditions report the requested sampling rate exactly.
- All 10,000 raw samples match the deterministic accumulator schedule.
- Workload-average target slowdown closely follows
  `sampling_rate * conditional_pinged_request_slowdown`:
  - prefill MAE: 0.0153 percentage points
  - decode MAE: 0.0278 percentage points
- Mean wall-time slowdown exceeds target-only slowdown by 0.2429 percentage points for prefill
  and 0.5063 percentage points for decode. The ping no longer creates the old long tail.

## Rate 1.0

Target slowdown and achieved bandwidth for requests that always run the ping:

| bpx | Prefill slowdown | Prefill GB/s | Decode slowdown | Decode GB/s |
|---:|---:|---:|---:|---:|
| 1 | 0.09% | 55.17 | 0.21% | 54.93 |
| 4 | 0.33% | 193.77 | 1.17% | 212.91 |
| 8 | 0.72% | 380.62 | 3.44% | 403.69 |
| 10 | 0.89% | 471.85 | 5.19% | 486.86 |
| 11 | 0.99% | 511.22 | 6.17% | 519.01 |
| 12 | 1.10% | 558.73 | 7.13% | 543.86 |
| 13 | 1.31% | 602.99 | 9.59% | 568.48 |
| 14 | 1.23% | 638.80 | 15.55% | 581.08 |
| 15 | 1.62% | 685.29 | 21.61% | 591.62 |
| 16 | 4.24% | 702.92 | 26.99% | 598.83 |

At rate 1.0, averaged across the informative `bpx=10..16` range:

| Regime | Target slowdown | Wall slowdown | Ping GB/s |
|---|---:|---:|---:|
| Prefill | 1.63% | 2.25% | 595.97 |
| Decode | 13.18% | 14.21% | 555.68 |

Decode target slowdown is about 8.1x prefill slowdown over this informative range.

## Interpretation

1. Bandwidth pings are costly but rate-tunable. Conditional slowdown remains nearly constant as
   the requested rate changes, while workload-average slowdown scales linearly with that rate.
2. Decode is substantially more sensitive because it streams FFN weights with little reuse.
   Prefill reuses each loaded weight across four token rows and is much less affected.
3. Keep `bpx={1,4,8}` as low-cost controls, not as informational ping configurations. The primary
   E1 comparison should use `bpx=10..16`, matching E2's useful bandwidth-ping range.
4. The run intentionally retains the original MoE home classifier. Classifier repeatability is
   outside this E1 revision's scope.

Reproduce the integrity checks and rate-1.0 table with:

```bash
scripts/summarize.py /work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/overhead-moe/overhead_moe_final_both_n50.out
```
