# MAPD-B — Code Guide: k-means|| on Dask

Human-readable walkthrough of this repository: **what** each piece of code
does, **step by step**, with the emphasis on the parallel engine
(`src/kmeans_parallel.py`) and **why** the chosen parallelization is the
right one. Companion document to `docs/CODE_REVIEW.md` (critical review +
roadmap) and `docs/CHANGES.md` (development log); this file is the linear,
from-scratch explanation.

Reference paper: Bahmani et al., *Scalable K-Means++*, VLDB 2012
(`docs/1203.6402v1.pdf`).

---

## Contents

1. [What this repository does](#1-what-this-repository-does)
2. [The pipeline, step by step](#2-the-pipeline-step-by-step)
   - 2.1 [Cluster bootstrap](#21-cluster-bootstrap-launch_clusterpy)
   - 2.2 [Data ingestion](#22-data-ingestion-data_loaderpy)
   - 2.3 [Scattering data to workers](#23-scattering-data-to-workers-benchmarkpy)
   - 2.4 [Parallel k-means|| seeding — the core](#24-parallel-k-means-seeding--the-core)
   - 2.5 [Distributed Lloyd's iterations](#25-distributed-lloyds-iterations-fit)
   - 2.6 [classify and inertia](#26-classify-and-inertia)
   - 2.7 [Serial baselines](#27-serial-baselines-kmeans_serialpy)
   - 2.8 [Experiment drivers](#28-experiment-drivers)
   - 2.9 [Analysis, notebooks, scripts](#29-analysis-notebooks-scripts)
3. [Why this parallelization is the right one](#3-why-this-parallelization-is-the-right-one)
4. [Running the project and conventions](#4-running-the-project-and-conventions)

---

## 1. What this repository does

The project implements the **k-means|| initialization** (parallel
k-means++, Algorithm 2 of Bahmani et al. 2012) on a **Dask cluster**, and
compares it against two serial scikit-learn baselines, **k-means++** and
**random seeding**, all followed by standard Lloyd's iterations. Dataset:
**KDD Cup 1999** (10% and full), plus a synthetic GaussMixture used to
reproduce one figure of the paper.

Two measures are recorded for every run:

- **Cost** (inertia): sum over all points of the squared distance to the
  nearest center — computed right after seeding and again after Lloyd's
  converges. This is the *controlled metric* of the whole project.
- **Time**: recorded for reference only. Wall-clock times between the
  serial client-side baselines and the distributed method are **not**
  comparable (different hardware: one laptop process vs. an 8-node
  cluster), so conclusions are always drawn on cost, never on time
  (`AGENTS.md` rule 5).

### Repository map

```
src/
  kmeans_parallel.py        the parallel engine: k-means|| seeding + distributed Lloyd's (Dask Bag)
  kmeans_serial.py          serial baselines: k-means++ / random seeding + sklearn fit
  data_loader.py            download -> Parquet -> preprocess -> Dask Bag of rows
  launch_cluster.py         SSHCluster bootstrap (hardcoded head + worker IPs)
  benchmark.py              run_single_test / run_benchmark / run_worker_sweep grid runners
  kmeans_comparison.py      driver: 3-way seeding comparison (paper's seed/final protocol)
  benchmark_analysis.py     BenchmarkAnalyzer: errorbar sweep plots from result CSVs
  comparison_analysis.py    tables/plots dedicated to the 3-way comparison
  paper_experiments.py      drivers + plots to reproduce Tables 3/4 and Figs 5.1/5.2
notebooks/
  analysis.ipynb            parameter sweeps (k, l/k, workers, partitions)
  comparison.ipynb          k-means|| vs k-means++ vs random
  paper_reproduction.ipynb  flag-gated reproduction of the paper's experiments
scripts/
  sync_workers.py           deploy src/ + requirements to all worker VMs
  check_cluster_env.py      one-shot probe that all workers have the right package versions
docs/                       CHANGES.md (dev log), CODE_REVIEW.md, ANALYSIS_PLAN.md, TODO.md
data/ results/              gitignored, regenerable (parquet / benchmark CSVs)
figures/                    committed plots used in the report
```

### The algorithm in one paragraph

Standard k-means++ picks one new center per pass with probability
proportional to the squared distance to the nearest center already chosen —
*k sequential passes* over the whole dataset, which does not scale to
`k` in the hundreds on millions of points. k-means|| fixes this: at every
**round**, every point `x` is sampled *independently* with probability
`min(1, l·d²(x,C)/φ)`, where `φ = φ_X(C)` is the sum of squared distances
of all points to their nearest center. Each round adds ≈`l` candidate
centers to the pool (with `l` the oversampling factor, e.g. `l = k`), and
only `O(log φ)` rounds are needed (in practice 5–15). The pool of ≈`l·r`
candidates is finally **reclustered down to `k` centers** with a single
weighted k-means++ pass on the client, where each candidate is weighted by
the number of points assigned to it. The paper proves this keeps the
theoretical guarantees of k-means++ while doing only `O(log φ)` passes
instead of `k`.

Everything else in this repo exists to run that algorithm on a cluster and
to show, empirically, that it beats the serial baselines in cost.

---

## 2. The pipeline, step by step

### 2.1 Cluster bootstrap (`launch_cluster.py`)

`launch_cluster(n_workers)` (`src/launch_cluster.py:107`) opens a
`dask.distributed.SSHCluster` over hardcoded node IPs (head `10.67.22.194`
+ a list of 8 worker IPs, `src/launch_cluster.py:27-37`), using the remote
Python interpreter of the VMs (`/home/ubuntu/pyvenv/bin/python3`).

```python
cluster = SSHCluster(hosts=[HEAD_IP] + active_worker_ips, ...)
client = Client(cluster)
client.wait_for_workers(n_workers, timeout=startup_timeout)
```

Three details matter:

- **Wait for workers.** `client.wait_for_workers` was previously accepted
  but never called, so sweeps silently proceeded with missing nodes
  (`docs/CHANGES.md` 2026-08-23). On timeout the code prints a loud warning
  and continues with what is available.
- **Pickle by value.** Scheduler and workers are started over SSH with
  `cwd=home` and do **not** have this repo on `sys.path`; task graphs that
  reference `src.*` functions would fail at deserialization with
  `ModuleNotFoundError`. `_enable_pickle_by_value()`
  (`src/launch_cluster.py:73`) registers the `src` modules for
  `cloudpickle.register_pickle_by_value`, so the code travels **inside the
  graph** and no remote process needs to import the project
  (`docs/CHANGES.md` 2026-08-24).
- **`scripts/sync_workers.py` / `scripts/check_cluster_env.py`** are the
  belt-and-braces: deploy `src/` + `requirements.txt` to every worker VM
  (re-run after each `git pull`), and verify in one shot that all workers
  run the exact package versions from the freeze. They exist because in
  August 2026 workers were re-provisioned without sklearn, which crashed
  the worker *process* during task execution (`KilledWorker`) — see
  `docs/CHANGES.md` 2026-08-24 for the full causal chain.

### 2.2 Data ingestion (`data_loader.py`)

`load_dataset()` (`src/data_loader.py:21`) turns the raw KDD Cup `.gz`
into a distributed **Dask Bag of 1-D NumPy rows**:

1. **Download** the `.gz` once (skipped if cached).
2. **Convert to Parquet** chunk-wise (`src/data_loader.py:47-60`): pandas
   reads fixed-size chunks and never holds the whole CSV in memory; a
   `ParquetWriter` appends them with snappy compression.
3. **Replicate to workers** (`src/data_loader.py:62-71`): the Parquet bytes
   are pushed with `client.run(...)` so *every* worker has the full file on
   its own disk **at the same absolute path** — required because Dask
   reads partitions locally and the metadata must agree across nodes
   (`AGENTS.md` rule 3).
4. **Read as a Dask DataFrame** with `dd.read_parquet(..., split_row_groups=True)`
   (`src/data_loader.py:73`) — from here on, reading is automatically
   parallel across workers.
5. **Preprocess** (`src/data_loader.py:76-104`): drop the 3 categorical
   columns and the label, coerce the rest to numeric, drop NaN rows, drop
   constant columns (e.g. `num_outbound_cmds`), then standardize with the
   **global** mean/std (two lazy passes: `dask.compute(ddf.mean(), ddf.std())`).
6. **Convert to a Dask Bag** (`src/data_loader.py:106-110`):

```python
X_bag = ddf.to_bag(index=False).map(
    lambda row: np.array(row, dtype=np.float64)
)
```

Each bag element is a single data row as a 1-D `float64` array; partitions
are preserved. This per-row representation is the *input format* of the
parallel engine (Section 2.4) — the engine immediately stacks rows back
into dense partition matrices.

The module also provides `make_gauss_mixture()` (`src/data_loader.py:123`,
the synthetic dataset of the paper's Fig 5.2) and `array_to_bag()`
(`src/data_loader.py:137`, same Bag format from a local NumPy array, for
tests and local runs).

### 2.3 Scattering data to workers (`benchmark.py`)

Benchmarks do not go through `load_dataset`'s Bag directly. Instead the
client materializes the dataset **once** as a NumPy array and re-distributes
it for each partitioning configuration:

```python
def _build_bag(client, X, num_partitions):            # src/benchmark.py:30
    chunks = np.array_split(X, num_partitions)
    futures = client.scatter(chunks)                  # chunks live on workers
    return db.from_delayed([dask.delayed(f) for f in futures])
```

`client.scatter` places each chunk on a worker as a future; the Bag is then
just references to those futures, so the data **never transits through the
task graph** client→scheduler→worker as serialized task arguments.

Two related lessons from `docs/CHANGES.md` 2026-07-22 are baked into the
drivers:

- **Never gather the bag directly.** Each row is its own tiny object
  (~500k of them at 10% scale); a direct gather saturates the
  client–scheduler connection (`CommClosedError`). The fix used everywhere
  is to `vstack` each partition **on the worker, inside the graph**
  (`dask.delayed(np.vstack)` per partition, e.g.
  `_materialize_bag` in `src/kmeans_comparison.py:31`), so only
  `len(partitions)` compact blocks travel over the network.
- **Never `repartition()` a Bag per configuration.** It re-reads and
  re-preprocesses the Parquet pipeline from scratch (or, persisted,
  becomes a poorly parallelized bottleneck) and risked OOM on the
  finalize step. Scatter + fresh Bag per configuration is both faster and
  memory-safe; the previous bag is explicitly cancelled with
  `client.cancel(old_bag)` when the partition count changes
  (`src/benchmark.py:199-204`).

### 2.4 Parallel k-means|| seeding — the core

Everything below lives in `src/kmeans_parallel.py` in
`kmeans_parallel.compute_starting_centroids()`
(`src/kmeans_parallel.py:324`). The engine's design principle, stated in
the module docstring (`src/kmeans_parallel.py:8-22`):

> The bag is split into delayed partitions, each stacked into a dense
> `(m, d)` matrix, and every algorithm step becomes **one task per
> partition** that returns **only the needed reductions** (per-cluster
> sums, counts, cost, changed labels). Only `k × d` matrices cross
> partition boundaries.

#### Stage 0 — persist partition matrices once

```python
parts = _persist_matrices(X)          # src/kmeans_parallel.py:359
```

`_persist_matrices` (`src/kmeans_parallel.py:61`) = `_bag_to_matrices` +
`dask.persist`: each bag partition is stacked into one dense matrix
(`_stack_rows`, `src/kmeans_parallel.py:48`) and the matrices are
**materialized exactly once** as cluster futures. Seeding and fit both
reuse these futures; without persistence every round/iteration would
re-vstack thousands of tiny rows from the bag — the most expensive part of
the old engine (≈`2r + 2` redundant passes avoided, `docs/CHANGES.md`
2026-08-23).

`shapes = dask.compute(*[dask.delayed(_matrix_shape)(p) ...])`
(`src/kmeans_parallel.py:360`) then yields each partition's `(m, d)` so the
client can build **global row offsets** (`offsets =
np.cumsum(...)`, line 385) and the total point count `n_points`.

#### Stage 1 — deterministic randomness: one seed → three independent streams

```python
ss_init, ss_body, ss_reclust = np.random.SeedSequence(seed).spawn(3)   # line 372
```

All randomness in the engine derives from **one** `SeedSequence`:

- `ss_init` → the uniform initial center (and the `r=0` random baseline);
- `ss_body` → the per-round sampling;
- `ss_reclust` → the `random_state` of the final weighted reclustering.

Three *independent children* are required because two `default_rng`s built
from the *same* SeedSequence would replay the identical stream (correlated
draws) — a real bug found in review (`docs/CHANGES.md` 2026-08-23).
Giving the same `seed` therefore produces **bit-identical** runs, regardless
of scheduler thread/execution order (`AGENTS.md` rule 4). No `np.random`
global state is touched anywhere in `src/`.

#### Stage 2 — the `r=0` random baseline (fixed policy)

With `policy="fixed", r=0` the method short-circuits *before touching the
data* (`src/kmeans_parallel.py:377-394`): `k` uniform global indices are
drawn from `ss_init`, mapped to (partition, local index) via the offsets,
and fetched with one `_rows_at` task per partition. This is the paper's
**Random baseline** (the `r=0` point of Fig 5.2 and Table 3) — no rounds,
no reclustering, `n_rounds_ = 0`.

#### Stage 3 — uniform initial center

One global index `i ~ Uniform(0, n_points)` is drawn from `ss_init`
(`src/kmeans_parallel.py:398-399`); only the owning partition fetches the
single row with a `_row_at` delayed task. Every point gets equal weight
here — exactly like k-means++'s first center.

#### Stage 4 — per-partition state and the initial cost ψ

For each partition, a **state matrix** `(m, 2)` is computed lazily:

```python
state_delays = [dask.delayed(_init_state, pure=False)(p, initial_centroid[0])
                for p in parts]                              # lines 419-422
```

`_init_state` (`src/kmeans_parallel.py:130`) stores, per point:
`(d²(x, c0), nearest_center_index)`. All pairwise distances use the
quadratic expansion in `_pairwise_d2` (`src/kmeans_parallel.py:88`):

```python
d2 = m_sq[:, None] + c_sq[None, :] - 2.0 * (M @ C.T)   # one BLAS matmul
np.maximum(d2, 0.0, out=d2)                            # clip fp noise
```

`‖x − c‖² = ‖x‖² + ‖c‖² − 2·x·c` is a **single matrix multiply** for all
rows against all centers — instead of `m × k` Python-level computations —
with the clip guarding against tiny negative values from floating-point
arithmetic. This one function is the workhorse of the entire engine
(seeding, fit, classify, inertia).

The initial cost is then the sum of per-partition **scalars**:

```python
psi = float(sum(dask.compute(*[dask.delayed(_state_cost)(s) for s in state_delays])))
```

`_state_cost` returns `state[:, 0].sum()` — `φ_X({c0})`. The `(m, 2)`
matrices themselves never leave the workers. If `psi == 0` every point
coincides with the initial center; the method returns `k` copies of it
(`src/kmeans_parallel.py:428-435`), honoring the contract that
`starting_centroids` always has `k` rows.

#### Stage 5 — how many rounds? `resolve_rounds`

`resolve_rounds(l, k, r, alpha, psi, policy)` (`src/kmeans_parallel.py:225`)
is the **single** place where the round count is decided (previously the
paper's rule was duplicated in engine and drivers and silently clobbered an
explicit `r`):

- `policy="auto"` (default, the paper's protocol):
  - if `l/k ≤ 0.1` → **15 rounds** (small oversampling needs more rounds to
    accumulate ≥ k candidates);
  - else an explicit `r` wins;
  - else estimate `round(alpha · log(psi))`.
- `policy="fixed"`: always and only the explicit `r` (required) — the
  escape hatch used by the paper-reproduction drivers, where the tables use
  `r=5` even at `l/k=0.1` (and `r=0` gives the random baseline of Stage 2).

Whatever branch runs, the **effective** round count is stored in
`self.n_rounds_` and recorded in the result CSVs as column `r_effective`
(`docs/CHANGES.md` 2026-08-23), so sweeps always compare configurations on
what was *actually executed*.

#### Stage 6 — the sampling rounds

Per-round RNGs are **pre-derived in bulk**, one independent child per
partition, then one grandchild per round:

```python
child_seeds = ss_body.spawn(len(parts))
round_seeds = [child.spawn(n_rounds) for child in child_seeds]   # lines 449-450
```

`(partition, round)` gets its own `SeedSequence` — deterministic and
parallel-safe, with no shared streams between draws. The loop over rounds
(`src/kmeans_parallel.py:454-521`) then has two interchangeable sampling
schemes:

**Bernoulli (default, Algorithm 2 of the paper).** One vectorized task per
partition (`_sample_round`, `src/kmeans_parallel.py:139`):

```python
probs = np.minimum(1.0, state[:, 0] * l / cost)
mask = rng.random(M.shape[0]) < probs
return M[mask]
```

Every point is sampled independently with `p = min(1, l·d²(x,C)/φ)`;
only the sampled rows (≈`l` total per round) travel to the client.

**Exact-ℓ (Fig 5.1 protocol only).** Exactly `l` points per round, without
replacement, with probability ∝ `d²` — the Efraimidis–Spirakis scheme
(`_sample_round_exact`, `src/kmeans_parallel.py:156`): each point gets key
`u^(1/d²)`, `u ~ U(0,1)`; each partition returns only its **local top-l**
keys and indices; the client merges the keys and takes the **global** top-l
(the top-l of a union = union of the top-l's). Across the network travel
only `l` (key, index) pairs per partition — never the points — and the
chosen rows are fetched afterwards with one `_rows_at` task per partition.
Zero-distance points get key `−inf` and are never sampleable, consistent
with the Bernoulli case (`p = 0`).

After sampling, the new candidates are appended to `self.centroids`, and
the **state update is fused with the round cost** (`_update_state`,
`src/kmeans_parallel.py:198`):

```python
update_tasks = [dask.delayed(_update_state, pure=False)(p, s, new_centroids_arr, start_idx)
                for p, s in zip(parts, state_delays)]           # lines 508-513
state_delays = [u[0] for u in update_tasks]                     # state stays worker-side
cost = float(sum(dask.compute(*[u[1] for u in update_tasks])))  # scalars only
```

`_update_state` compares each point's current `d²` against the new
candidates, lowers it where closer, records the new nearest index — and
returns `(new_state, partial_cost)`. The updated state is kept as a
**symbolic reference** feeding the next round's tasks; the client receives
one scalar per partition. In the pre-refactor engine the full `(m, 2)`
state matrices round-tripped client↔cluster at every round (tens of MB on
KDD full — `docs/CHANGES.md` 2026-08-23).

#### Stage 7 — candidate weights

One final reduction over the states (`src/kmeans_parallel.py:530-534`):

```python
weights = sum(dask.compute(*[dask.delayed(_partition_bincount, pure=True)(
    s, len(self.centroids)) for s in state_delays]))
```

Each partition returns a `k`-vector counting how many of its points are
assigned to each **candidate** center; the sum is the global weight of
every candidate. Only `n_candidates` integers reach the client.

#### Stage 8 — weighted reclustering down to k

```python
kmeans = KMeans(n_clusters=self.k, n_init=1, random_state=reclustering_random_state)
kmeans.fit(np.vstack(self.centroids), sample_weight=centroids_weights)
self.starting_centroids = kmeans.cluster_centers_        # lines 543-545
```

The ≈`l·r`-candidate pool is shrunk to the `k` final starting centers with
a **weighted k-means++** (each candidate weighted by its point count). Two
deliberate choices: `n_init=1` because the paper uses a *single*
k-means++ initialization, not sklearn's default 10 restarts (also keeps
this client-side step fast at large `k` — it used to risk the
client–scheduler heartbeat timeout, `docs/CHANGES.md` 2026-07-22); and the
`random_state` is drawn from the `ss_reclust` branch of the SeedSequence so
the whole seeding stays reproducible.

**Edge cases handled explicitly:** empty partitions (every helper guards
`M.shape[0] == 0`), `ψ = 0`, and a candidate pool smaller than `k` (too
small `l`): the drivers treat the resulting sklearn `ValueError` with
`"n_clusters"` in the message as a known, benign failure and record a row
with `cost=None` (`src/benchmark.py:73-97`) — any other `ValueError`
propagates as a real bug.

### 2.5 Distributed Lloyd's iterations (`fit`)

`fit(X, max_iter, tol)` (`src/kmeans_parallel.py:549`) runs plain Lloyd's
k-means starting from the seeded centers, fully distributed. One task per
partition per iteration:

```python
tasks = [dask.delayed(_lloyd_pass, pure=False)(p, prev_labels[j], centroids_arr)
         for j, p in enumerate(parts)]                     # lines 593-596
reductions = [t[:4] for t in tasks]                        # line 600
results = dask.compute(*reductions)
prev_labels = [t[4] for t in tasks]                        # line 614
```

`_lloyd_pass` (`src/kmeans_parallel.py:105`) does, in a single vectorized
sweep: pairwise `d²` against all `k` centers → `argmin` labels → segmented
sums via `np.add.at` → `bincount` counts → partition cost → **count of
labels changed vs. the previous iteration**. Its return value is the tuple
`(sums (k,d), counts (k,), cost, changed, labels (m,))`, and `fit` exploits
the split:

- **to the client** go only `t[:4]` — the small reductions (`k×d` sums,
  `k` counts, one scalar of cost, one scalar of changed labels);
- **`t[4]`, the `(m,)` label vector, never leaves the worker**: it remains
  in the graph as the `prev_labels` input of the *next* iteration's task.

Before this refactor the `(m,)` int32 label vectors travelled up and down
from the client at every iteration (`docs/CHANGES.md` 2026-08-23).

The client then merges the reductions, updates centers
(`new_centroids[populated] = sums/counts`), keeps empty clusters frozen at
their previous value (with a one-time warning, `src/kmeans_parallel.py:617-624`),
and stops when **no point changed cluster** (`changed == 0`,
`src/kmeans_parallel.py:639`) — the strict, sklearn-style convergence
check, fused into the assignment pass so no extra data sweep is needed —
or as a fallback when the raw centroid-shift norm drops below `tol`
(rarely reached at large `k`).

Optional `track_convergence=True` records the per-iteration cost and wall
time in `cost_history_` / `iter_times_` (used by the notebooks' convergence
plots and by the comparison's `n_lloyd_iters` column).

### 2.6 `classify` and `inertia`

- `classify(X)` (`src/kmeans_parallel.py:649`) returns a **Bag** of
  nearest-center labels, one vectorized task per partition
  (`_labels_partition`). It's a Bag again so downstream analysis keeps the
  familiar per-row semantics.
- `inertia(X)` (`src/kmeans_parallel.py:665`) delegates to the shared
  `inertia_of_bag(X_bag, centroids)` (`src/kmeans_parallel.py:281`) — the
  *single* implementation of the cost formula, also used by
  `benchmark.calculate_inertia()` (`src/benchmark.py:38`). Guarantees the
  "cost" number in every CSV is computed identically everywhere: sum, over
  partitions, of per-partition `_inertia_partial` reductions.

### 2.7 Serial baselines (`kmeans_serial.py`)

`kmeans_serial` (`src/kmeans_serial.py:16`) mirrors the same two-phase API
(`compute_starting_centroids` → `fit`) so all three methods are driven by
identical code:

- `init="k-means++"` → `sklearn.cluster.kmeans_plusplus` seeding
  (with `n_local_trials=None` = sklearn's greedy default, or `1` for the
  paper's plain Algorithm 1);
- `init="random"` → `k` uniform distinct points;
- `fit` → sklearn `KMeans(n_init=1)` starting from those centers, to match
  the parallel engine's "single initialization" protocol.

Everything runs single-threaded on the client — that is the point of a
baseline.

### 2.8 Experiment drivers

**`benchmark.run_single_test`** (`src/benchmark.py:46`): one
`(k, l, r, partitions)` configuration on a cluster. Builds/scatters the bag
(§2.3), times seeding and fit separately, computes the final cost via the
shared inertia, and returns a dict including `r_effective` (the round count
actually executed — see §2.4 Stage 5) and the seed used.

**`benchmark.run_benchmark`** (`src/benchmark.py:128`): a grid of
combinations over `k_values`, writing a timestamped CSV into `results/`.
Two correctness details that matter:

- **Real averaging.** The i-th repetition uses `seed + i`, not the same
  seed, so mean/std over `averaging_iterations` runs has statistical
  meaning (with a fixed seed the 10 runs would be bit-identical and the
  error bars zero — a fixed review defect, `docs/CHANGES.md` 2026-08-23).
- **One client-side materialization.** `X_arr` is gathered once (§2.3's
  worker-side vstack trick) and re-scattered per partitioning
  configuration, cancelling the previous bag.

**`benchmark.run_worker_sweep`** (`src/benchmark.py:242`): for each worker
count in `workers_list`, restart the SSH cluster
(`launch_cluster(n)` → grid from `combinations_fn(n)` → `shutdown_cluster`),
concatenating everything into one DataFrame with a `workers_cfg` column.
This executes the pending "vary workers *and* partitions simultaneously"
TODO (`docs/TODO.md`).

**`kmeans_comparison.run_comparison`** (`src/kmeans_comparison.py:59`):
the 3-way comparison following the paper's protocol — for each `k`, run
k-means|| (configurable partitions/`l/k`/`r`), serial k-means++ and random,
`averaging_iterations` times each with distinct seeds, recording `cost_seed`
(cost right after seeding), `cost_final` (after Lloyd's), Lloyd's iteration
counts, and per-phase times. Analysis aggregates with the **median**
(the paper's statistic). One documented subtlety: sklearn's `tol` scales
with the data variance while `kmeans_parallel.fit` uses the raw Frobenius
norm of the centroid shift; in practice both stop on strict label stability
at large `k`, so `cost_final` remains comparable across engines
(`src/kmeans_comparison.py:73-79`).

**`paper_experiments.py`** (`src/paper_experiments.py`): purpose-built
drivers + plots to reproduce the paper's artifacts — `run_fig51` (KDD 10%,
**exact-ℓ** sampling, cost vs rounds), `run_fig52` (GaussMixture, `r = 0..15`
with the k-means++ reference line, runnable locally), `run_table34` (KDD
full, `policy="fixed", r=5` per the paper's tables), with `plot_fig51/52`
and `table34_cost_table/table34_time_table` for the output side.

### 2.9 Analysis, notebooks, scripts

- **`benchmark_analysis.BenchmarkAnalyzer`** (`src/benchmark_analysis.py`):
  generic CSV analyzer — group by arbitrary columns, compute mean/std of
  any metric, produce errorbar plots faceted by parameter value (used by
  `analysis.ipynb` for the sweeps).
- **`comparison_analysis`** (`src/comparison_analysis.py`): tables and
  plots for the categorical 3-way comparison (cost tables scaled ×10⁻¹⁰
  as in the paper, `plot_cost_vs_rounds`, etc.), kept separate from
  `BenchmarkAnalyzer` because "method vs method" is a different shape of
  problem than "metric vs parameter".
- **Notebooks**: `analysis.ipynb` (cluster boot → data load → single run
  with convergence/centroid tracking → benchmark grid → plots),
  `comparison.ipynb` (3-way comparison + tables/plots), and
  `paper_reproduction.ipynb` (flag-gated sections for the paper
  experiments, with obtained-vs-paper placeholders). All use the cluster
  absolute dataset paths (`/home/ubuntu/...`, `/tmp/...`) — per
  `AGENTS.md` rule 2, these must not be "fixed".
- **Scripts**: `sync_workers.py`, `check_cluster_env.py` (§2.1), plus the
  gitignored local harness in `agents/` (`check_env.py`, `smoke_test.py`,
  `clean_notebooks.py`, golden-regression fixtures).

---

## 3. Why this parallelization is the right one

### 3.1 The right granularity: per-partition dense matrices, not per-row functions

The original engine applied Python lambdas row-by-row inside `bag.map` and
collected results with a `foldby` that **shuffled rows across partitions**.
That is the natural "naive Dask" style, and it is wrong for this workload:
millions of tiny Python calls, plus a full row shuffle every time the data
is regrouped.

The refactor (2026-08-23, `docs/CHANGES.md`) turns every algorithm step
into **one NumPy task per partition operating on a dense `(m, d)` matrix**.
Why that is right:

- **Vectorization**: all point-level arithmetic becomes BLAS — the entire
  pairwise-distance pass is one `M @ C.T` matmul via the quadratic
  expansion (`_pairwise_d2`). No Python loop touches individual rows.
- **Locality**: a partition's data is touched by exactly one task per step,
  on one worker. Nothing reshuffles rows between partitions — across
  partition boundaries travel only `k × d` matrices and scalars.
- **Measured, not theoretical**: locally (20k points, d=15, k=20,
  8 partitions) seeding went 12.80s → 0.42s, fit 1.24s → 0.04s, inertia
  0.62s → 0.02s — ≈30×, with bit-identical final cost and golden
  regression fixtures preserved (`docs/CHANGES.md` 2026-08-23).

### 3.2 The right communication pattern: state stays on the workers

Distributed k-means has a natural "state": per-point nearest-center
distance/index during seeding, per-point labels during Lloyd's. Shipping
that state to the client every round/iteration is the single most common
way to strangle a Dask implementation — the state is `O(m)` and the useful
result of each step is `O(k·d)`.

This engine keeps both states **as symbolic nodes in the task graph**:

- k-means|| `(m, 2)` state: chained from round to round
  (`state_delays = [u[0] for u in update_tasks]`); the client sees one cost
  scalar per partition per round (`_update_state` fuses it) and one
  `k`-vector of weights at the end.
- Lloyd's `(m,)` labels: `t[4]` stays in the graph as next iteration's
  `prev_labels`; the client sees `k×d` sums, `k` counts, cost, changed.

Combined with the per-partition matrices being persisted **once**
(`_persist_matrices`), a round or iteration costs one vectorized sweep per
partition plus a tiny reduction — communication is `O(P·k·d)` per round
instead of `O(m)`.

### 3.3 The right RNG discipline: determinism without shared streams

Distributed randomness has two failure modes: the global `np.random`
(non-reproducible — draws happen from scheduler threads in
non-deterministic order, the old bug of `docs/CHANGES.md` 2026-07-19) and
sharing one stream across consumers (correlated draws). The design here —
one `SeedSequence(seed).spawn(3)` into initial-center / rounds /
reclustering branches, then per-(partition, round) spawns — gives
**independent** streams for every consumer and **bit-identical** reruns for
a given seed, on any number of workers. Reproducibility of a parallel
stochastic algorithm is itself an argument for this design: it makes the
sweeps' averaging meaningful and lets `agents/smoke_test.py` regression-test
the engine against golden fixtures.

### 3.4 The right algorithm: why k-means|| (and not just k-means++ on Dask)

The parallelization choice mirrors the *algorithm's* structure, which is
why the two reinforce each other:

- k-means++ needs **k sequential passes** over the data (each new center
  depends on all previous ones). Parallelizing it means either paying
  `O(k)` global synchronizations or abandoning its exact sampling.
- k-means|| needs only **`O(log φ)` rounds** — typically 5–15 even for
  `k = 1000` — and each round is an **embarrassingly parallel** pass: every
  point's sampling decision depends only on its own distance to the current
  center set and on the current cost φ, both broadcast scalars/centers.
  This is precisely the shape that map-reduce-style engines (Dask,
  Hadoop — the paper's own setting) execute well: independent per-partition
  computation + a tiny reduction per round.
- The final weighted reclustering to `k` is the only inherently sequential
  piece, and it is `O(l·r·k)` on a small candidate pool — small enough to
  run on the client with sklearn (n_init=1, per the paper), which also
  keeps the serial-vs-distributed cost comparison honest: all three methods
  feed the *same* Lloyd's procedure.

### 3.5 The right engine: why Dask Bag + delayed (not Spark, not raw MPI)

- **Same Python stack as the baselines.** The serial comparisons are
  scikit-learn; running the distributed method in the same language with
  the identical NumPy cost formula (`inertia_of_bag`, shared by engine and
  drivers) removes cross-implementation differences from the comparison.
- **Iterative graph semantics.** The round-to-round and iteration-to-
  iteration state chains (3.2) are *dataflow edges* — trivial in Dask's
  delayed graph, where the next step simply references the previous
  step's output. No checkpointing, no shuffle, no separate state service.
- **Explicit data placement.** `client.scatter` futures + persisted
  partition matrices give precise control over where data lives and when
  it is materialized — control that matters at this scale (the
  `repartition` and direct-gather failures in `docs/CHANGES.md`
  2026-07-22 were both data-placement mistakes).
- **Operational fit.** `SSHCluster` over the 8 fixed VMs matches the course
  infrastructure; the pickle-by-value shim (§2.1) plus `sync_workers.py`
  make code deployment a non-issue.

### 3.6 The right metric: cost, not wall-clock

Because the baselines run on one laptop core and k-means|| runs on 8 VMs,
time comparisons would compare hardware, not algorithms — a deliberate
non-goal (`AGENTS.md` rule 5). Cost (inertia) is hardware-independent and
is exactly the quantity the paper's theory bounds, which is why every CSV
records `cost_seed`/`cost_final` computed by one shared code path.

### 3.7 Honest limitations (kept visible by design)

- Wall-clock times are recorded but never compared across engines.
- `tol` semantics differ slightly between sklearn and the distributed
  Lloyd's; strict label-stability dominates at large `k`, documented in
  `src/kmeans_comparison.py:73-79`.
- Still open (`docs/TODO.md`): the worker×partition sweep on the real
  cluster (implemented in `run_worker_sweep`, pending cluster time),
  unmanaged-memory investigation, and the out-of-core phase (sharded
  Parquet per worker instead of full replicas).

---

## 4. Running the project and conventions

```bash
conda activate mapd-b                       # canonical env (see AGENTS.md)
conda run -n mapd-b python agents/check_env.py      # deps + versions sanity
conda run -n mapd-b python agents/smoke_test.py     # no-cluster functional test
python -m src.launch_cluster -n N           # start SSH cluster with N workers
```

Run notebooks from the repo root with the `mapd-b` kernel; strip outputs
before committing with `python agents/clean_notebooks.py`. Write results
only under `results/` (CSVs) and `figures/` (plots).

Rules worth repeating (from `AGENTS.md`):

- Never commit to master; work on feature branches.
- Do not "fix" the cluster-absolute dataset paths or the hardcoded node
  IPs — they match the real infrastructure.
- Seeding is deterministic by construction; do not reintroduce global
  `np.random` in `src/`.
- Serial vs distributed wall-clock times are not comparable.
- Before finishing any change to `src/`: run `agents/smoke_test.py`
  (determinism + golden regression checks included).

Operational notes from the incident log: never restart the notebook kernel
with a live cluster (SSH sessions die and take the remote processes down);
clean orphaned schedulers on port 8786 before `launch_cluster`; after
every boot run `check_cluster_env` and after every `git pull` re-run
`sync_workers.py` (`docs/CHANGES.md` 2026-08-24).
