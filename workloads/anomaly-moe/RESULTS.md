# E3 Initial Results

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
