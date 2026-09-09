# MAPD-B — Codebase Guide: k-means|| on Dask

Function-level reference of this repository: **what each part of the code
does**, and how the pieces fit together. Written against the current
source tree; for the historical development log see `docs/CHANGES.md`, for
the critical assessment see `docs/CODE_REVIEW.md`.

Reference paper: Bahmani et al., *Scalable K-Means++*, VLDB 2012
(`docs/1203.6402v1.pdf`).

---

## Contents

1. [What this repository does](#1-what-this-repository-does)
2. [Module-by-module reference](#2-module-by-module-reference)
   - 2.1 [Cluster bootstrap — `launch_cluster.py`](#21-launch_clusterpy)
   - 2.2 [Data ingestion — `data_loader.py`](#22-data_loaderpy)
   - 2.3 [The parallel engine — `kmeans_parallel.py`](#23-kmeans_parallelpy)
   - 2.4 [Serial baselines — `kmeans_serial.py`](#24-kmeans_serialpy)
   - 2.5 [Benchmark driver — `benchmark.py`](#25-benchmarkpy)
   - 2.6 [3-way comparison — `kmeans_comparison.py`](#26-kmeans_comparisonpy)
   - 2.7 [Paper reproduction — `paper_experiments.py`](#27-paper_experimentspy)
   - 2.8 [Analysis modules — `benchmark_analysis.py`, `comparison_analysis.py`](#28-analysis-modules)
3. [Notebooks and scripts](#3-notebooks-and-scripts)
4. [Data contracts and invariants](#4-data-contracts-and-invariants)
5. [Mapping to the paper](#5-mapping-to-the-paper)
6. [Design rationale](#6-design-rationale)
7. [Running the project](#7-running-the-project)

---

## 1. What this repository does

The project implements the **k-means|| initialization** (parallel
k-means++, Algorithm 2 of Bahmani et al. 2012) on a **Dask cluster**, and
compares it against two serial scikit-learn baselines — **k-means++** and
**random seeding** — all followed by standard Lloyd's iterations. Dataset:
**KDD Cup 1999** (10% and full), plus a synthetic GaussMixture used to
reproduce one figure of the paper.

The end-to-end pipeline:

```
kddcup.data.gz ──stream──> N Parquet shards (master disk, cached)
        │
        │ client.scatter (one shard per worker, round-robin)
        ▼
workers: pass 1 (per-shard stats) ──> global mean/std/min/max, constant cols
workers: pass 2 (standardize)     ──> dask.array X (n, d), one chunk per shard
        │
        ▼
kmeans_parallel.compute_starting_centroids(X)   # k-means||: r sampling
        │                                        # rounds + weighted
        ▼                                        # reclustering to k
kmeans_parallel.fit(X)                          # distributed Lloyd's
        │
        ▼
cost (inertia) via the shared inertia_of_bag()  # same formula everywhere
```

In parallel, the same dataset (materialized client-side) feeds the serial
baselines in `kmeans_serial.py`, so all three methods are driven through
the same two-phase API and evaluated with the identical cost formula.

**The controlled metric is cost (inertia), not wall-clock time**: the
serial baselines run on the client machine while k-means|| runs on the
cluster, so times compare hardware, not algorithms.

---

## 2. Module-by-module reference

### 2.1 `launch_cluster.py`

Dask cluster management over SSH.

**Constants** (top of file): `HEAD_IP`, `WORKER_IPS` (the 8 course VMs),
`SCHEDULER_PORT`/`DASHBOARD_PORT`, `SSH_CONNECT_OPTIONS`,
`remote_python=/home/ubuntu/pyvenv/bin/python3`. These match the real
infrastructure — do not "fix" them (`AGENTS.md` rule 2).

- **`launch_cluster(n_workers, block=False, startup_timeout=30.0)` →
  `(cluster, client)`**: starts an `SSHCluster` with the head node as
  scheduler and the first `n_workers` entries of `WORKER_IPS` as workers;
  connects a `Client`; then `client.wait_for_workers(n_workers)` blocks
  until all requested workers are connected (on timeout it prints an
  explicit warning and continues with what is available). With
  `block=True` (only from the command line,
  `python -m src.launch_cluster -n N`) it keeps the cluster alive until
  CTRL+C — useful to run the cluster standalone, but never from a
  notebook (it would block the cell forever).
- **`shutdown_cluster(cluster, client)`**: orderly close of client and
  cluster. Call it between notebook iterations that change `n_workers`,
  and at the end of a session.
- **`_enable_pickle_by_value()`** (internal, called by `launch_cluster`):
  scheduler and workers are started over SSH with `cwd=home` and do not
  have this repo on `sys.path`; task graphs referencing `src.*` functions
  would fail to deserialize with `ModuleNotFoundError`. Registering the
  `src` modules with `cloudpickle.register_pickle_by_value` makes the code
  travel **inside the graph** (`docs/CHANGES.md` 2026-08-24). Idempotent;
  fails soft with a warning.

### 2.2 `data_loader.py`

End-to-end data pipeline, bounded-memory even on the full KDD dataset.

**Constants**: `CATEGORICAL_COLS = ["protocol_type", "service", "flag"]`,
`LABEL_COL = "label"` — the non-numeric and non-feature columns of KDD.

- **`load_dataset(dataset_url, raw_gz_path, parquet_path, col_names,
  n_partitions=4, client=None, force_download=False)` →
  `(X, (mean, std))`**. The main entry point. Steps:
  1. **Download** the `.gz` once to `raw_gz_path` (skipped if cached,
     unless `force_download`).
  2. **Convert to Parquet shards** (`_write_shards`): the CSV is read in
     fixed-size chunks by pandas and written as `n_partitions` snappy
     Parquet files into the directory `parquet_path` — the master never
     holds the whole CSV in RAM. Shards are **cached**: re-running with
     the same directory reuses them.
  3. **Scatter shards to workers**: each shard file is read once and
     pushed with `client.scatter`, round-robin over the workers — one
     shard lives on exactly one worker, no worker holds the whole dataset,
     and nothing is copied to worker disks. With the project convention
     `n_partitions = 8 × workers`, every worker holds exactly 8 shards.
     (If `n_partitions < n_workers`, some workers receive no shard and
     stay idle.)
  4. **Preprocess on the workers, in two passes**:
     - pass 1 (`_shard_stats`): per-shard reduced statistics
       (count/sum/sumsq/min/max) → global mean, sample std (ddof=1),
       min/max, and the list of **constant columns** (min == max — for
       KDD that is `num_outbound_cmds`);
     - pass 2 (`_shard_matrix`): re-read each shard, drop categorical and
       label columns, coerce to numeric, drop NaN rows, drop constant
       columns, standardize with the global mean/std → dense `(m, d)`
       matrix.
  5. **Return** `(X, (mean, std))`: `X` is a `dask.array` `(n, d)` with
     **one chunk per shard and known shapes** (built by
     `_delayed_matrices_to_array`), so `X.shape`, `X.nbytes`,
     `X.npartitions` and slicing work without computing; `mean`/`std` are
     pandas Series indexed by feature name, to go back to the original
     coordinates.

  `client` is required (shards are scattered through it).
  `parquet_path` is a **directory** of shards on the master.

- **`_count_lines_gz(filepath)`** (internal): counts lines of the `.gz` to
  size the chunks. Opened with `newline="\n"`: in default text mode a file
  with CRLF endings (Windows) would be counted twice per record.
- **`_clean_shard_df(df, constant_cols)`** (internal): the per-shard
  preprocessing shared by both passes (drop columns, `to_numeric`, dropna,
  drop constants).
- **`_delayed_matrices_to_array(matrix_tasks, row_counts, n_features)`**
  (internal, also used by `benchmark._build_bag`): wraps each delayed
  matrix with `da.from_delayed` (which does not accept lists) and
  concatenates along axis 0. Chunk shapes **must** be known: with
  `(np.nan, d)` chunks dask treats unknown sizes as 1 in
  concatenate/slicing and silently produces wrong results — the per-shard
  counts from pass 1 (post-dropna, same preprocessing as pass 2) are
  exact.
- **`make_gauss_mixture(n, k, d=15, R=1.0, seed=None)` →
  `(X, y, centers)`**: the synthetic GaussMixture of the paper's Fig 5.2
  (k centers ~ N(0, R·I_d), uniform assignment, points ~ N(center, I_d)).
  Pure in-memory, no disk/network.
- **`array_to_dask(X, n_partitions=4)`**: numpy `(n, d)` → `dask.array`
  in `n_partitions` chunks — the **same format** produced by
  `load_dataset`, for local tests and the GaussMixture without Parquet.

### 2.3 `kmeans_parallel.py`

The core: k-means|| seeding + distributed Lloyd's. The engine's design
principle, stated in the module docstring:

> Every algorithm step is **one vectorized NumPy task per partition**
> operating on a dense `(m, d)` matrix, returning **only the needed
> reductions**. Only `k × d` matrices and scalars cross partition
> boundaries.

**Per-partition helpers** (module-level, so the scheduler serializes them
without closures):

- **`_bag_to_matrices(X)`**: input → flat list of Delayed, one dense
  `(m, d)` matrix per partition. Accepts a `dask.array` (its 2-D chunks
  used as-is; note `Array.to_delayed()` returns a *nested* ndarray on the
  chunk grid and must be flattened with `.ravel()`) or a legacy `dask.bag`
  of rows (stacked with `_stack_rows` — backward compatibility, used by
  the local smoke test).
- **`_persist_matrices(X)`**: `_bag_to_matrices` + `dask.persist` — the
  partition matrices are **materialized exactly once** as cluster futures;
  seeding and fit both reuse them, so no pass ever re-reads the source.
- **`_pairwise_d2(M, C)`**: `d2[i,j] = ‖M[i] − C[j]‖²` via the quadratic
  expansion `‖x‖² + ‖c‖² − 2·x·c` — all distances as **one BLAS matmul**,
  clipped at 0 against floating-point noise. Used where the full `(m, k)`
  matrix is affordable (e.g. the single initial centroid).
- **`_pairwise_d2_argmin_chunked(M, C, chunk_k=100)`**: same nearest-center
  computation but visiting the centroids in chunks of `chunk_k`, keeping a
  running (best distance, best index) per point. **Never materializes the
  full `(m, k)` matrix** — with `k = 500..1000` and large partitions that
  matrix can weigh several GB per worker and caused past out-of-memory
  crashes. This is the workhorse of the engine (seeding, fit, classify,
  inertia).
- **`_lloyd_pass(M, prev_labels, C)`**: one Lloyd's assignment on one
  partition, in a single vectorized sweep: chunked distances → `argmin`
  labels → segmented sums via `np.add.at` → `bincount` counts → partition
  cost → **count of labels changed vs. the previous iteration** (fused
  into the pass, so the strict convergence check costs no extra sweep).
  Returns `(sums (k,d), counts (k,), cost, changed, labels (m,) int32)`.
- **`_init_state(M, c0)`** / **`_update_state(M, state, new_centroids,
  start_idx)`**: the k-means|| per-partition state `(m, 2)` = (current
  min `d²`, nearest-candidate index), initialized against the first center
  and updated after each sampling round. `_update_state` **fuses the round
  cost** into the update: it returns `(new_state, partial_cost)`, so only
  a scalar per partition crosses to the client — the state itself stays
  worker-side.
- **`_sample_round(M, state, l, cost, round_seed_seq)`**: Bernoulli
  sampling (Algorithm 2 of the paper): each point independently with
  `p = min(1, l·d²/φ)`, using a local RNG derived from the
  (partition, round) child SeedSequence — deterministic and
  parallel-safe.
- **`_sample_round_exact(M, state, l, round_seed_seq)`**: EXACT `l` points
  per round without replacement, `p ∝ d²` — the protocol of **Fig 5.1
  only**. Efraimidis–Spirakis keys `u^(1/d²)`; each partition returns only
  its **local top-l** `(key, index)` pairs; the client merges them into
  the global top-l (the top-l of a union is the union of the top-l's) and
  fetches the chosen rows afterwards. Only `l` pairs per partition cross
  the network, never the points. Zero-distance points get key `−inf` and
  are never sampleable (consistent with Bernoulli `p = 0`).
- **`_rows_at` / `_row_at` / `_matrix_shape` / `_state_cost` /
  `_partition_bincount` / `_labels_partition` / `_inertia_partial`**:
  small per-partition reductions/fetches used by the drivers and the
  class.
- **`resolve_rounds(l, k, r=None, alpha=1.0, psi=None, policy="auto")`**:
  the **single** place where the number of k-means|| rounds is decided:
  - `policy="auto"` (default, the paper's protocol): `l/k ≤ 0.1` → **15
    rounds** (small oversampling needs more rounds to accumulate ≥ k
    candidates); otherwise an explicit `r` wins; otherwise the estimate
    `round(alpha·log ψ)`;
  - `policy="fixed"`: always and only the explicit `r` (required) — the
    escape hatch used by the paper-reproduction drivers; `r = 0` selects
    the random baseline.
- **`inertia_of_bag(X, centroids)`**: the **single shared implementation**
  of the cost (inertia) — one `_inertia_partial` task per partition,
  summed. Used by `kmeans_parallel.inertia()` and
  `benchmark.calculate_inertia()`, so "cost" means the same number in
  every CSV.

**The class — `kmeans_parallel(k, l, r=None)`**, driven in two phases:

- **`compute_starting_centroids(X, alpha=1, l=None, max_iter=None,
  seed=None, track_centroids=False, policy="auto",
  sampling="bernoulli")`** — the k-means|| initialization:
  1. Persist partition matrices once; compute `n_points` and global row
     offsets from the partition shapes.
  2. Split `SeedSequence(seed)` into **three independent children**:
     `ss_init` (initial center / random baseline), `ss_body` (sampling
     rounds), `ss_reclust` (final weighted reclustering). Two RNGs built
     from the same sequence would replay the identical stream (correlated
     draws) — hence the spawn tree. Same seed ⇒ bit-identical run, on any
     number of workers.
  3. **`r = 0` with `policy="fixed"`** short-circuits before touching the
     data: `k` uniform global indices → one `_rows_at` fetch per partition
     → done (`n_rounds_ = 0`, no reclustering). This is the paper's
     Random baseline (the `r = 0` point of Fig 5.2 and Table 3).
  4. **Initial center**: one uniform global index from `ss_init`; only the
     owning partition fetches the row (`_row_at`).
  5. **Initial cost ψ** = sum of per-partition `_state_cost` scalars. If
     `ψ = 0` every point coincides with the initial center: return `k`
     copies of it, honoring the contract that `starting_centroids` always
     has `k` rows.
  6. **Rounds** (count from `resolve_rounds`; per-(partition, round) RNGs
     pre-derived in bulk): sample with the chosen scheme (`"bernoulli"` or
     `"exact"`), append candidates, update the worker-side state with
     `_update_state` (cost fused). The updated state is persisted and kept
     as a **symbolic reference** for the next round; the client receives
     one scalar per partition per round. Stops early if the cost hits 0;
     the **effective** round count is stored in `n_rounds_` and recorded
     in the CSVs as `r_effective`.
  7. **Candidate weights**: one final `_partition_bincount` reduction —
     how many points are assigned to each candidate.
  8. **Weighted reclustering to k** (client-side, sklearn):
     `KMeans(n_clusters=k, n_init=1, random_state=...)` fit on the
     candidate pool weighted by the assignment counts. `n_init=1` matches
     the paper (single k-means++ initialization, not sklearn's 10
     restarts); `random_state` comes from the `ss_reclust` branch so the
     whole seeding stays reproducible.
  9. If the candidate pool is smaller than `k` (too small `l`/low `r`),
     sklearn raises `ValueError` mentioning `n_clusters` — the drivers
     treat **only that** error as a known benign failure (row with null
     costs); any other error propagates as a real bug.
  10. With `track_centroids=True`, `n_centroids_history_` records the
      cumulative pool size after each round.

- **`fit(X, max_iter=100, tol=1e-4, track_convergence=False)`** —
  distributed Lloyd's from `starting_centroids`. Each iteration: one
  `_lloyd_pass` task per partition; the results are **persisted** so the
  task graph stays flat; only `t[:4]` (sums, counts, cost, changed) are
  computed to the client, while `t[4]` — the `(m,)` label vector —
  **never leaves the worker** and becomes the next iteration's
  `prev_labels`. The client merges the reductions, updates centers
  (empty clusters stay frozen at their previous value, with a one-time
  warning), and stops when **no point changed cluster** (`changed == 0`,
  the strict sklearn-style check) or, as a fallback, when the centroid
  shift drops below `tol · max(1, ‖C‖)` (relative tolerance; rarely
  reached at large `k`). With `track_convergence=True`,
  `cost_history_`/`iter_times_` record per-iteration cost and wall time;
  `n_iter_` counts the completed Lloyd's updates.

- **`classify(X)`**: nearest-center label per point, as a `dask.bag` (one
  `_labels_partition` task per partition) — per-row semantics for
  downstream analysis.
- **`inertia(X)`**: delegates to `inertia_of_bag`.

**Attribute contract**: `centroids` = the raw candidate pool (NOT k rows);
`starting_centroids` = the `k` centers after reclustering (input of
`fit`); `final_centroids` = output of `fit` (used by `classify`/`inertia`).
Calling `fit`/`classify`/`inertia` out of order raises a clear
`RuntimeError`.

### 2.4 `kmeans_serial.py`

Serial baselines with the **same two-phase API** as `kmeans_parallel`, so
all three methods are driven by identical driver code:

- **`kmeans_serial(k, init="k-means++" | "random")`**.
- **`compute_starting_centroids(X, seed=None, n_local_trials=None)`**:
  `init="k-means++"` → `sklearn.cluster.kmeans_plusplus`
  (`n_local_trials=None` = sklearn's greedy default, `1` = the paper's
  plain Algorithm 1); `init="random"` → `k` uniform distinct points.
- **`fit(X, max_iter=100, tol=1e-4)`**: sklearn `KMeans(n_init=1)`
  starting from the seeded centers — matching the parallel engine's
  "single initialization" protocol; `n_iter_` mirrors sklearn's count.

Everything runs single-machine on the client — that is the point of a
baseline.

### 2.5 `benchmark.py`

Benchmark driver (sweeps over k, l/k, workers, partitions).

- **`RESULTS_DIR = "results"`** — relative: run from the repo root.
- **`_build_bag(client, X, num_partitions)`** (internal): scatter
  `np.array_split(X, num_partitions)` chunks to the workers and wrap the
  futures as a `dask.array` via `_delayed_matrices_to_array` — data moves
  once per configuration and never transits through the task graph as
  serialized arguments.
- **`calculate_inertia(X, centroids)`**: thin alias of the shared
  `inertia_of_bag`.
- **`run_single_test(client, k, l, r, num_partitions, ...)` →
  `(result, X_bag)`**: one configuration. Times seeding and Lloyd's
  separately, records `initial_cost`/`final_cost` via the shared inertia,
  `r_effective`, `n_lloyd_iters`, the seed, and (optionally) the
  convergence/centroid histories. The only benign failure is the known
  "candidate pool < k" case (null costs); anything else propagates. The
  lazy dataset is returned too, so the caller can reuse it.
- **`run_benchmark(client, X_bag=None, combinations=None, k_values=None,
  ...)`**: a grid over `combinations` of `(n_workers, num_partitions,
  l_over_k, r)` and `k_values`. Two correctness details:
  - **real averaging** — repetition `i` uses `seed + i`, so mean/std over
    `averaging_iterations` has statistical meaning;
  - **one client-side materialization** — `X_arr` is gathered once and
    re-scattered per partition count, `client.cancel`-ing the previous
    distribution to free worker memory.
  Every result row is **appended to the CSV immediately**, so an
  interruption loses nothing already computed.
- **`combinations_fn(n_workers, l_over_k, r)`**: the standard
  configuration factory — `num_partitions = 8 × workers`.
- **`run_worker_sweep(X_bag_or_arr, workers_list, ...)`**: for each worker
  count, restart the SSH cluster (`launch_cluster(n)` → grid →
  `shutdown_cluster`), concatenating into one DataFrame with a
  `workers_cfg` column.

### 2.6 `kmeans_comparison.py`

The 3-way comparison driver (k-means|| vs k-means++ vs random), following
the paper's evaluation setup.

- **`_materialize_bag(client, X_bag)`** (internal): gather the distributed
  dataset into one client-side numpy array for the serial baselines —
  partition-wise, so only a handful of compact blocks travel the network.
  Pass-through if given a plain ndarray.
- **`pilot_timing_check(X_arr, k, ...)`**: time one serial run at the
  smallest planned `k` before committing to the full loop (greedy
  k-means++ seeding can be slow at this scale).
- **`run_comparison(client, X_bag, k_values, parallel_combinations,
  seed=42, averaging_iterations=11, ...)`**: for each `k`, run k-means||
  (configurable partitions/`l`/`r`) and both serial baselines,
  `averaging_iterations` times each with distinct seeds (`seed + i`).
  Records `cost_seed` (right after seeding), `cost_final` (after Lloyd's),
  `n_lloyd_iters`, per-phase times, and `r_effective`; appends everything
  to a timestamped CSV in `results/`. Analysis aggregates with the
  **median** (the paper's statistic).
  One documented subtlety: sklearn's `tol` scales with data variance while
  `kmeans_parallel.fit` uses a relative Frobenius-norm shift; in practice
  both stop on strict label stability at large `k`, so `cost_final`
  remains comparable across engines.

### 2.7 `paper_experiments.py`

Purpose-built drivers + plots to reproduce the paper's artifacts
(Partition baseline excluded everywhere per `docs/ANALYSIS_PLAN.md`):

- **`run_fig51(client, X_bag, k_values=(17,33,65,129),
  l_over_k_values=(1,2,4), r_values=1..10, ...)`**: Fig 5.1 — final cost
  vs rounds on KDD 10%, with **exact-`l`** sampling.
- **`run_fig52(client=None, R_values=(1,10,100),
  l_over_k_values=(0.1,...,10), r_values=0..15, k=50, n=10k, d=15, ...)`**:
  Fig 5.2 — final cost vs rounds on the synthetic GaussMixture (regenerated
  per `R` with deterministic seed), with the k-means++ horizontal
  reference; `r = 0` is the random-baseline path. Runs locally (no cluster
  needed at this scale).
- **`run_table34(client, X_bag_full, k_values=(500,1000),
  l_over_k_values=(0.1,...,10), r_fixed=5, ...)`**: Tables 3/4 — KDD full,
  **`policy="fixed", r=5`** (the paper uses `r = 5` even at `l/k = 0.1`,
  so the automatic rule must not apply), plus the Random baseline as the
  `r = 0` path. Records seed/final costs (raw; the ×10⁻¹⁰ scale is applied
  at analysis time) and separate seeding/Lloyd's times for Table 4.
- All drivers share `_run_one_parallel`, which handles the known
  "candidate pool < k" failure benignly (row with `failed=True`, NaN
  costs) so overnight sweeps don't stop.
- Output side: **`plot_fig51`**, **`plot_fig52`** (log-y, medians — the
  paper's conventions), **`table34_cost_table`** (pivot layout, ×10⁻¹⁰
  scale, reusing `comparison_analysis.format_paper_table`),
  **`table34_time_table`**, and `_save_results` (timestamped CSV in
  `results/`).

### 2.8 Analysis modules

- **`benchmark_analysis.BenchmarkAnalyzer`** (dataclass): generic
  CSV-to-plots analyzer for the parameter sweeps — group by arbitrary
  columns (`compute_grouped_stats`: mean/std/n_runs per metric), print
  summaries (`print_summary`), and produce errorbar plots faceted by
  parameter value (`plot_all`: one figure per facet combination, one
  subplot per metric). Used by `analysis.ipynb`.
- **`comparison_analysis`**: tables and plots for the **categorical**
  3-way comparison, kept separate because "method vs method" is a
  different shape of problem than "metric vs parameter":
  `summarize_comparison` (long-format mean/std/median per variant × k),
  `format_paper_table` (the paper's pivot layout: rows = variant, columns
  = per-k seed/final blocks), `plot_cost_by_method` (grouped bars),
  `plot_cost_vs_rounds` (cost vs r per l/k, with horizontal baselines).

---

## 3. Notebooks and scripts

- **`notebooks/analysis.ipynb`**: cluster boot → data load → single run
  with convergence/centroid tracking → benchmark grids (sweeps over k,
  l/k, workers, partitions) → `BenchmarkAnalyzer` plots into `figures/`.
- **`notebooks/comparison.ipynb`**: the 3-way comparison via
  `run_comparison` + tables/plots.
- **`notebooks/paper_reproduction.ipynb`**: flag-gated sections for the
  paper experiments (Fig 5.1, Fig 5.2, Tables 3/4) with obtained-vs-paper
  write-up.
- **`notebooks/run.ipynb`**: scratch/driver notebook for cluster runs.

All notebooks use the **cluster-absolute** dataset paths
(`/home/ubuntu/...`, `/tmp/...`) — per `AGENTS.md` rule 2, these must not
be "fixed". Run them from the repo root with the `mapd-b` kernel; strip
outputs before committing (`python agents/clean_notebooks.py`).

- **`scripts/sync_workers.py`**: deploy `src/` + `requirements.txt` + a
  `.pth` file to every worker VM (the workers are separate machines
  without the repo); re-run after every `git pull`.
- **`scripts/check_cluster_env.py`**: verify in one shot that all workers
  run the exact package versions from the freeze (born from the 2026-08-24
  incident where re-provisioned workers without sklearn crashed with
  `KilledWorker` — `docs/CHANGES.md`).
- **`agents/`** (gitignored local harness): `check_env.py` (local deps),
  `smoke_test.py` (no-cluster functional test: determinism, round policy,
  usage guards, r=0 baseline, exact sampling, golden regression),
  `loader_test.py` (shard pipeline end-to-end on a local client),
  `clean_notebooks.py`, golden fixtures.

---

## 4. Data contracts and invariants

- **Engine input format**: a `dask.array` `(n, d)` of 2-D chunks (one
  chunk per partition — what `load_dataset`/`array_to_dask`/`_build_bag`
  produce). A legacy `dask.bag` of 1-D rows is still accepted
  (`_bag_to_matrices` stacks it) and is used by the local smoke test.
- **kmeans_parallel attributes** (§2.3): `centroids` (raw pool) →
  `starting_centroids` (k rows, post-reclustering) → `final_centroids`
  (post-fit). `n_rounds_`/`n_iter_` count what was actually executed.
- **Result CSV schemas**:
  - `benchmark.py`: `k, l, r, r_effective, partitions, initial_cost,
    final_cost, time, lloyd_time, n_lloyd_iters, seed, workers, l_over_k`
    (+ `cost_history`, `iter_times`, `n_centroids_history` when tracked);
  - `kmeans_comparison.py`: `method, k, l, r, r_effective, partitions,
    l_over_k, seed, cost_seed, cost_final, n_lloyd_iters, time_seed,
    time_fit`;
  - `paper_experiments.py`: `artifact, method, k, l, r, l_over_k,
    partitions, sampling, run, seed, r_effective, cost_seed, cost_final,
    n_lloyd_iters, time_seed, time_fit, failed` (+ `cost_*_e10` for
    table34).
- **Determinism**: all randomness in `kmeans_parallel` derives from one
  `SeedSequence(seed)` with a spawn tree (initial center / rounds /
  reclustering, then per-(partition, round)). Same seed ⇒ bit-identical
  runs on any worker count. No global `np.random` anywhere in `src/`
  (`AGENTS.md` rule 4).
- **Cost comparability**: every "cost" in every CSV comes from the single
  `inertia_of_bag` implementation (serial baselines evaluated on the same
  standardized array). Wall-clock times are recorded for reference but are
  **not comparable** across serial and distributed methods (different
  hardware).
- **Cluster constants** (IPs, `/home/ubuntu/...`, `/tmp/...` paths,
  remote python) match the real course infrastructure and are intentional.

---

## 5. Mapping to the paper

| Paper artifact | Where in the code |
|---|---|
| Algorithm 2 (k-means\|\| seeding) | `kmeans_parallel.compute_starting_centroids` (Bernoulli `sampling="bernoulli"`: `p = min(1, l·d²/φ)`) |
| Algorithm 1 (k-means++ baseline) | `kmeans_serial` with `init="k-means++"` (`n_local_trials=1` for the plain variant) |
| Random baseline | `compute_starting_centroids(policy="fixed", r=0)` or `kmeans_serial(init="random")` |
| Weighted reclustering (step 8) | final `KMeans(..., n_init=1).fit(candidates, sample_weight=...)` in `compute_starting_centroids` |
| Round-count rule (`l/k ≤ 0.1 ⇒ 15`, else `O(log ψ)`) | `resolve_rounds` (`policy="auto"`) |
| Fig 5.1 (KDD 10%, exact-l) | `paper_experiments.run_fig51` / `plot_fig51` (`sampling="exact"`) |
| Fig 5.2 (GaussMixture) | `paper_experiments.run_fig52` / `plot_fig52` + `data_loader.make_gauss_mixture` |
| Tables 3/4 (KDD full, r=5) | `paper_experiments.run_table34` / `table34_cost_table` / `table34_time_table` (`policy="fixed"`) |
| Table 6 (Spam) | **out of scope** per `docs/ANALYSIS_PLAN.md` |
| Partition baseline | **excluded** per `docs/ANALYSIS_PLAN.md` |

---

## 6. Design rationale

Why the parallelization is built this way (condensed; full discussion in
`docs/CODE_REVIEW.md`):

- **Per-partition dense matrices, not per-row functions.** The naive
  "bag.map a lambda per row" style costs millions of Python calls plus a
  row shuffle on every regrouping. Here every step is one NumPy task per
  partition — the whole distance pass is a single BLAS matmul. Measured
  locally (20k points): ≈30× faster than the row-granular engine, with
  bit-identical final cost (`docs/CHANGES.md` 2026-08-23).
- **State stays on the workers.** Distributed k-means has natural
  `O(m)` state (per-point nearest-center distance during seeding,
  per-point labels during Lloyd's); shipping it to the client every round
  is the classic way to strangle a Dask implementation. Both states live
  as symbolic nodes in the task graph; the client sees one scalar per
  partition per round (cost fused into the update) and `k×d` reductions
  per iteration — communication is `O(P·k·d)` per round, not `O(m)`.
- **One `SeedSequence`, spawned.** Distributed randomness fails two ways:
  the global `np.random` (non-reproducible under the threaded scheduler —
  the old bug) and sharing one stream across consumers (correlated draws).
  The spawn tree gives every consumer an independent stream and bit-identical
  reruns, which is also what makes the sweeps' averaging meaningful and the
  golden regression possible.
- **The algorithm matches the engine.** k-means++ needs `k` sequential
  passes; k-means|| needs `O(log ψ)` embarrassingly parallel rounds
  (each point's sampling decision depends only on broadcast scalars), plus
  one small client-side reclustering — precisely the map-reduce shape Dask
  (and the paper's own Hadoop) executes well.
- **Same stack as the baselines.** The serial comparisons are
  scikit-learn; the distributed engine uses the identical NumPy cost
  formula (`inertia_of_bag`), removing cross-implementation differences
  from the comparison.
- **Cost, not wall-clock** (§1): hardware-independent and exactly the
  quantity the paper's theory bounds.

**Honest limitations** (kept visible by design): serial vs distributed
times are not comparable; `tol` semantics differ slightly between sklearn
and the distributed Lloyd's (strict label stability dominates at large
`k`, documented in `src/kmeans_comparison.py`); same-seed numbers are
bit-comparable only within the same data representation (the shard
pipeline partitions rows differently from the legacy bag — medians over
seeds are the right comparison); if `n_partitions < n_workers`, some
workers receive no shard and stay idle (the project convention is
`8 × workers`).

---

## 7. Running the project

```bash
conda activate mapd-b                                    # canonical env
conda run -n mapd-b python agents/check_env.py           # deps + versions
conda run -n mapd-b python agents/smoke_test.py          # no-cluster engine test
conda run -n mapd-b python agents/loader_test.py         # no-cluster loader test
python -m src.launch_cluster -n N                        # start SSH cluster
```

Run notebooks from the repo root with the `mapd-b` kernel; write outputs
only under `results/` (CSVs) and `figures/` (plots); strip notebook
outputs before committing (`python agents/clean_notebooks.py`). Before
finishing any change to `src/`, run `agents/smoke_test.py` (determinism +
golden regression included).
