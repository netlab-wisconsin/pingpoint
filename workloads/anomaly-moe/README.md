# Anomaly MoE (E3)

This benchmark prototypes adaptive two-tier monitoring for a path-localized decode-FFN anomaly:

- Eight experts are explicitly mapped one-to-one to the eight XCDs and corresponding HBM paths.
- Every request contains 64 routed tokens.
- Normal requests use the balanced distribution `{8,8,8,8,8,8,8,8}`.
- Anomaly requests use `{57,1,1,1,1,1,1,1}`, making expert/XCD/HBM path 0 hot.
- Cheap pointer-chase latency pings continuously measure every path.
- A detector uses cross-path skew in normalized latency means:
  `max(path_mean / calibrated_path_mean) - median(path_mean / calibrated_path_mean)`.
- Two consecutive detections agreeing on the same cache complex trigger a three-request
  bandwidth-ping burst on the most affected path in that complex.
- The controller latches the anomaly after diagnosis and does not repeatedly trigger until recovery.

The existing `workloads/moe/k1.h` and `k2.h` home-identification logic is reused unchanged.
Target, latency, and bandwidth pings have bounded lifetimes: all stop when the decode-FFN request
completes. The grid shape and target CU budget remain fixed across all policies.

The current prototype makes controller decisions on the host between request launches. Reported
target and wall overhead measure the bounded GPU request kernels and exclude host-side controller
processing. Moving the controller into a persistent GPU serving loop is future work.

MI300X groups two XCDs per cache complex. ACN contention can make either path in the affected
complex show the largest normalized latency skew, so E3 evaluates localization at cache-complex
granularity and records the selected path used for bandwidth diagnosis.

Four policies run over the same timeline:

1. `baseline`: target only.
2. `latency`: always-on latency pings.
3. `always_bw`: latency and bandwidth pings always active on all paths.
4. `adaptive`: always-on latency pings and reactive bandwidth diagnosis.

Build and run:

```bash
scripts/build.sh
scripts/run.sh
```

Useful overrides:

```bash
N_REQUESTS=120 ANOMALY_START=40 ANOMALY_LENGTH=24 ANOMALY_BW_BPX=10 scripts/run.sh
ANOMALY_CALIBRATION_REQUESTS=30 scripts/run.sh
```

Summarize a raw output:

```bash
scripts/summarize.py /work1/sinclair/junyeol/ici-workspace/sigcomm-exp/revision/anomaly-moe/raw/FILE.out
```

`scripts/run.sh` also saves this summary under the result directory's `parsed/` subdirectory.
