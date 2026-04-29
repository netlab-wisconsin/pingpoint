# test-db-copilot

Prototype for congestion localization on GPU DB/vector workloads using PPNT pings.

## What it does

- Builds a ping plan with latency + bandwidth pings from `src_xcd={0,1}` to `dst_hbm={4,0}`.
- Runs **ping-only** and **co-run** modes (one ping at a time) for DB-like kernels:
  - Hot entry read/update (`reader_xcd=0`, `writer_xcd=1`)
  - Point lookup
  - Vector distance pass
  - Range filter
- Prints per-ping slowdown ratio:
  - `ratio = ping_mean_ns(corun) / ping_mean_ns(ping-only)`

This lets you compare the “hot entry at HBM4” case vs “migrated to HBM0” case.

## Build

```bash
cd test-db-copilot
make
# or override ROCm path:
# HIP_HOME=/opt/rocm-7.2.0 make
```

## Run

```bash
./bin/db_query_corun [db_len] [query_n] [vec_rows] [vec_dim]
```

Defaults are tuned for prototype runs if arguments are omitted.
