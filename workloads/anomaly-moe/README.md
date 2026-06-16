# Anomaly MoE (E3)

This benchmark prototypes adaptive two-tier monitoring for a path-localized decode-FFN anomaly:

- Eight experts are explicitly mapped one-to-one to the eight XCDs and corresponding HBM paths.
- Every request contains 64 routed tokens.
- Normal requests use the balanced distribution `{8,8,8,8,8,8,8,8}`.
- Anomaly requests use `{57,1,1,1,1,1,1,1}`, making expert/XCD/HBM path 0 hot.
- Cheap pointer-chase latency pings continuously measure every path.
- A detector uses cross-path skew in normalized latency means:
  `max(path_mean / calibrated_path_mean) - median(path_mean / calibrated_path_mean)`.
- The timeline contains two identical hot-expert anomalies separated by recovery.
- Two consecutive strong detections trigger bandwidth diagnosis on the most affected path from the
  second detection.
- During the first anomaly, diagnosis covers `{10,16,11,15,12,14,13}` twice in deterministic
  mixed order, filling the 14 requests remaining after detection.
- During the second anomaly, diagnosis uses only `bpx=10` at full rate after detection, for a
  14-request BW-ping burst.
- The controller latches after detection and does not repeatedly trigger until recovery.

The existing `workloads/moe/k1.h` and `k2.h` home-identification logic is reused unchanged.
Target, latency, and bandwidth pings have bounded lifetimes: all stop when the decode-FFN request
completes. The grid shape and target CU budget remain fixed across all policies.

The current prototype makes controller decisions on the host between request launches. Reported
target and wall overhead measure the bounded GPU request kernels and exclude host-side controller
processing. Moving the controller into a persistent GPU serving loop is future work.

When a logical request needs both latency detection and BW diagnosis, the benchmark runs two
non-overlapping GPU passes: target+latency first for `detector_score`, then target+BW for
`target_slowdown_pct` and BW-ping loss. The parsed time series also reports
`detector_target_slowdown_pct`, the target overhead from the latency-only detector pass.

MI300X groups two XCDs per cache complex. ACN contention can make either path in the affected
complex show the largest normalized latency skew, so E3 evaluates localization at cache-complex
granularity and records the selected path used for bandwidth diagnosis.

Four policies run over the same timeline:

1. `baseline`: target only.
2. `latency`: always-on latency pings.
3. `always_bw`: latency and bandwidth pings always active on all paths.
4. `adaptive`: always-on latency pings and reactive bandwidth diagnosis, using full-rate diagnosis
   on the first encounter and lower-rate diagnosis on the second.

The default 70-request timeline contains five equal 14-request phases:

- normal: requests `[0,14)`
- first anomaly: `[14,28)`
- recovery/normal: `[28,42)`
- second anomaly: `[42,56)`
- final recovery/normal: `[56,70)`

The first anomaly represents intensive diagnosis across the full informative `bpx` set. The second
represents full-rate learned monitoring using only `bpx=10`.
Because decisions occur between requests, the 14-ping diagnosis bursts intentionally spill into
recovery after a two-request detection delay. This lets the plot show BW-diagnosis overhead
continuing as the latency-ping anomaly signal falls back.

Build and run:

```bash
scripts/build.sh
scripts/run.sh
```

Useful overrides:

```bash
N_REQUESTS=70 FIRST_ANOMALY_START=14 FIRST_ANOMALY_LENGTH=14 \
  SECOND_ANOMALY_START=42 SECOND_ANOMALY_LENGTH=14 scripts/run.sh
ANOMALY_CALIBRATION_REQUESTS=30 scripts/run.sh
```

Summarize a raw output:

```bash
scripts/summarize.py /work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/raw/FILE.out \
  /work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/parsed/FILE_timeseries.csv
```

`scripts/run.sh` saves both a summary and a plot-ready adaptive time series under `parsed/`. The
time series directly reports matched-baseline target slowdown, detector score, triggers, BW-active
requests, selected path, `bpx`, probe loss, and informativeness.
