# E3 Results

## Current Equal-Phase Experiment

Final run:

```text
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/raw/
  anomaly_moe_n64_a1_16_16_a2_48_16_first14_second_bpx10_i4.out
```

The trace contains four equal 16-request phases: normal, first anomaly, recovery/normal, and second
anomaly. Both anomalies triggered after one request.

| Encounter | BW schedule | Ping requests / anomaly | Unique bpx | Informative | Mean pinged slowdown | Anomaly-window slowdown |
|---|---|---:|---:|---:|---:|---:|
| First anomaly | `{10,16,11,15,12,14,13}` twice | 14/16 (87.5%) | 7 | 100% | 56.785% | 49.791% |
| Second anomaly | `bpx=10` every fourth request | 4/16 (25.0%) | 1 | 100% | 40.401% | 10.434% |

The first anomaly spends nearly its entire post-detection window performing broad diagnosis. The
second uses the learned `bpx=10` probe periodically, reducing anomaly-window target overhead by
79.0% relative to the first anomaly.

Overall adaptive target slowdown was `15.098%`, compared with `127.636%` for always-on BW pings.
Always-on latency target slowdown was effectively zero (`-0.039%` measured).

An independent repeat preserved 100% informative coverage and the exact 4/16 second-anomaly
schedule. Its first anomaly needed three requests to confirm detection, so 13 broad-sweep pings ran
inside the anomaly and the final ping ran on the first recovery request. This is a consequence of
between-request reactive control: strict 14/16 placement requires confirmation by the second
anomaly request.

Plot-ready output:

```text
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/parsed/
  anomaly_moe_n64_a1_16_16_a2_48_16_first14_second_bpx10_i4_timeseries.csv
```

## Previous Two-Anomaly Experiment

Final run:

```text
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/raw/
  anomaly_moe_n200_a1_40_28_a2_100_60_r2_2.out
```

The trace contains identical hot-expert anomalies at requests `[40,68)` and `[100,160)`.
Both encounters diagnose the selected path with the same mixed-order informative set
`{10,16,11,15,12,14,13}`. The first diagnosis uses rate 1.0; the second uses exact deterministic
rate 0.2. Each diagnosis stops after covering the full set once.

### Adaptive Diagnosis Comparison

| Encounter | Rate | Trigger delay | Completion span | BW pings | Unique bpx | Informative | Mean pinged slowdown | Anomaly-window slowdown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| First | 1.0 | 1 request | 7 requests | 7 | 7 | 100% | 48.389% | 12.181% |
| Second | 0.2 | 1 request | 35 requests | 7 | 7 | 100% | 48.810% | 5.829% |

Lowering the rate does not make an individual BW ping cheaper: mean slowdown on pinged requests is
nearly identical. It spreads the same diagnostic coverage across more requests, reducing the
time-averaged anomaly-window overhead by 52.1% while increasing diagnosis completion time by 5x.

### Policy Comparison

| Policy | Mean target slowdown | Mean wall slowdown | BW duty cycle |
|---|---:|---:|---:|
| Target only | 0.000% | 0.000% | 0.0% |
| Always-on latency | 0.003% | 0.230% | 0.0% |
| Always-on BW, all paths | 122.419% | 106.676% | 100.0% |
| Adaptive two-tier | 3.443% | 3.222% | 7.0% |

Both anomalies triggered after one request and localized to cache complex 0. The controller re-armed
during recovery and did not issue BW pings outside the two diagnosis episodes. All 14 adaptive BW
measurements were informative.

Plot-ready output:

```text
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/parsed/
  anomaly_moe_n200_a1_40_28_a2_100_60_r2_2_timeseries.csv
```

## Initial Single-Anomaly Results

Final initial run:

```text
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/raw/
  anomaly_moe_n120_a40_l24_bpx10.out
```

Independent repeat:

```text
/work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/raw/
  anomaly_moe_n120_a40_l24_bpx10_repeat.out
```

The trace contains 120 measured requests. Requests `40..63` route 57/64 tokens to expert 0;
all other requests route eight tokens to every expert. The anomaly therefore occupies 20% of the
trace. Reactive diagnosis uses `bpx=10`.

## Calibration

- Detector threshold: `0.08`
- Solo `bpx=10` ping reach: 66.1% to 74.5% of each path's `bpx=16` peak
- Informative diagnosis criterion: at least 20% probe loss and at least 55% peak reach

## Policy Comparison

| Policy | Target slowdown, run 1 / repeat | Wall slowdown, run 1 / repeat | BW duty cycle |
|---|---:|---:|---:|
| Target only | 0.000% / 0.000% | 0.000% / 0.000% | 0.0% |
| Always-on latency | 0.052% / 0.062% | 0.304% / 0.507% | 0.0% |
| Always-on BW, all paths | 92.194% / 85.185% | 77.088% / 70.911% | 100.0% |
| Adaptive two-tier | 1.524% / 0.875% | 1.519% / 1.271% | 2.5% |

## Adaptive Behavior

- All 24 anomaly requests exceeded the detector threshold.
- One normal request exceeded the threshold: the first request immediately after recovery.
- The controller triggered after one to two requests.
- Run 1 selected path 0; the repeat selected path 1. Both are in cache complex 0, which contains
  hot expert/XCD 0.
- Bandwidth diagnosis ran for only three requests in each run.
- All six diagnoses across both runs were informative.
- Mean BW-ping loss during diagnosis was 46.59% in run 1 and 29.57% in the repeat.

The adaptive policy reduced BW-ping duty cycle from 100% to 2.5% and target slowdown from 92.194%
to 1.524% in run 1, and from 85.185% to 0.875% in the repeat, relative to always monitoring every
path with BW pings.

## Interpretation

The latency tier cheaply detects and localizes sustained imbalance to the affected cache complex.
A short BW burst then confirms that the selected path has little bandwidth headroom, after which
the controller remains latched and avoids repeatedly paying diagnosis cost during the same anomaly.

Exact path selection within cache complex 0 varied between path 0 and path 1. This is consistent
with the two XCDs sharing a cache complex and means the result should not be presented as reliable
single-XCD localization.

The current controller executes on the host between request launches. GPU target and wall timing
exclude host-side controller processing; a persistent GPU-resident controller remains future work.
