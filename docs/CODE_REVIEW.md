# Code description and critical review

Companion document to `problems.md`. Two parts: (1) what the code does,
(2) critical assessment of the Dask parallelization with an improvement
roadmap. Part 3 lists the gaps to reproduce Bahmani et al. (2012),
Tables 3/4/6 and Figures 5.1/5.2.

---

## Part 1 — What the code does

The project implements the **k-means|| initialization** (Bahmani et al.,
VLDB 2012, `docs/1203.6402v1.pdf`) on a Dask cluster and compares it
against **k-means++** and **random** seeding. The standard workflow starts
from `notebooks/analysis.ipynb`:

1. **Cluster bootstrap** — `src.launch_cluster.launch_cluster(n_workers)`
   opens an `SSHCluster` over hardcoded node IPs (head `10.67.22.194` +
   worker list, remote python `/home/ubuntu/pyvenv/bin/python3`) and
   returns `(cluster, client)`.
2. **Data ingestion** — `src.data_loader.load_dataset()`:
   - downloads KDD Cup 1999 (`.gz`, full or 10%),
   - streams it chunk-wise into a single snappy Parquet file (never loads
     the whole CSV in memory),
   - `client.run()`s the Parquet bytes onto every worker's local disk at
     the same absolute path,
   - opens it with `dd.read_parquet(..., split_row_groups=True)`,
     drops the 3 categorical columns and `label`, drops constant columns,
     standardizes with global mean/std (computed in two lazy passes),
   - returns a **Dask Bag of 1-D NumPy arrays** (one row per element) plus
     the `(mean, std)` used for scaling.
3. **k-means|| + Lloyd's** — `src.kmeans_parallel.kmeans_parallel`:
    - `compute_starting_centroids(X_bag, seed, ...)`: Algorithm 2 of the
      paper. Uniform initial center; per round, each point is sampled with
      Bernoulli probability `min(1, l*d2(x,C)/phi_X(C))`; a per-partition
      "state" matrix `(m, 2)` of `(min_d2, nearest_center_idx)` is chained
      through delayed tasks and stays worker-side (only round-cost scalars
      reach the client); after `r` rounds (resolved by `resolve_rounds`,
      recorded in `n_rounds_` / CSV column `r_effective`) the candidate
      pool is reclustered to `k` centers with a weighted scikit-learn KMeans
      (n_init=1, k-means++ init) on the client.
    - `fit(X_bag, ...)`: distributed Lloyd's on persisted partition
      matrices. Each iteration runs one vectorized task per partition that
      returns only small reductions (`k×d` sums, counts, cost,
      label-changed count); the per-point labels stay in the task graph as
      input of the next iteration. Stops when no label changes or the
      centroid shift < `tol`.
    - `classify` / `inertia`: nearest-center label and phi_X(C) for the
      final centroids (inertia shares `inertia_of_bag` with benchmark.py).
4. **Baselines** — `src.kmeans_serial.kmeans_serial` (client-side,
   scikit-learn): `kmeans_plusplus` seeding or uniform random, then
   sklearn `KMeans` Lloyd's with `n_init=1` from those centers.
5. **Experiment drivers**:
   - `src.benchmark.run_single_test / run_benchmark`: one or a grid of
     `(k, l, r, workers, partitions)` runs; scatters the dataset to workers
     via `client.scatter` (`_build_bag`), times seeding and Lloyd's
     separately, appends rows to `results/*.csv`.
   - `src.kmeans_comparison.run_comparison`: three-way seeding comparison
     (cost after seeding and after Lloyd's, averaged over seeds), following
     the paper's seed/final protocol.
   - `src.benchmark_analysis.BenchmarkAnalyzer` and
     `src.comparison_analysis`: CSV → grouped stats → errorbar plots /
     tables into `figures/`.

`notebooks/comparison.ipynb` drives step 5's comparison;
`analysis.ipynb` drives the parameter sweeps.

## Part 2 — Critical assessment of the Dask implementation

### What is right

- **Scatter-based bag construction** (`_build_bag`): data moves to workers
  once per partition count, not through the task graph. Correct choice.
- **Persisted partition matrices + distributed state** (2026-08-23 round 2):
  partition matrices are materialized once per call (`_persist_matrices`);
  the k-means|| state and Lloyd's labels never travel to the client — only
  scalars and `k×d` blocks cross the network.
- **Reclustering on the client** (weighted KMeans on O(r*l) candidates):
  matches the paper's Step 8 and keeps the heavy work distributed.
- **Strict convergence check** in `fit()` (no label change) — matches
  Lloyd's definition and sklearn behavior; the earlier centroid-shift-only
  criterion stalled for large k.
- **Cost computed inside the assignment pass** when tracking convergence:
  no extra pass over the data.
- **Real seed averaging in both drivers**: repetition i uses `seed + i`,
  effective rounds recorded (`r_effective`), so cost statistics across
  repetitions are meaningful.

### Bottleneck: row-granular Bag + per-point Python lambdas

The dominant issue. Every point-level operation
(`map(lambda x: norm(x - C))`, Bernoulli `filter`, `foldby` binops,
`classify`, `inertia`) launches **one Python function call per point per
pass**. For KDD full (4.8M points):

- ~5M+ Python invocations *per Lloyd iteration*; the interpreter, not
  NumPy, is the bottleneck. NumPy vectorization never sees a matrix — only
  single rows — so BLAS is idle.
- The task graph holds millions of fine-grained tasks; scheduler overhead
  (serialization of closures over `centroids_arr`!) grows with `k`: every
  per-point lambda closes over the full centroid matrix.
- `foldby` on a Bag is a shuffle-based groupby: per-point tuples
  `(idx, (row, d2))` are serialized and shuffled. Moving *rows* through
  the shuffle is far more expensive than moving per-cluster partial sums.

**Consequence**: wall-clock is dominated by per-task overhead, so scaling
workers shows diminishing returns early, and `l`/`k` sweeps measure
scheduler throughput more than numerical throughput. This is consistent
with the unmanaged-memory and slow-sweep issues noted in `docs/CHANGES.md`.

**Status: RESOLVED (2026-08-23, commit `5f3371e`).** The engine was
rewritten on delayed partition matrices: one vectorized NumPy task per
partition per pass (stacked `(m, d)` matrices, quadratic-expansion
distances via a single BLAS matmul, `np.add.at` segment sums), no row
shuffle, convergence check fused into the assignment pass. Measured
locally (n=20k, d=15, k=20, 8 partitions): seed 12.80s → 0.42s,
fit 1.24s → 0.04s, inertia 0.62s → 0.02s (~30x); final cost unchanged
(672313.9799) and golden regression bit-identical on fixed starting
centroids (`agents/fixtures/`).

**Still open (structural)**: represent the dataset as a **Dask Array**
`(n, d)` blocked by rows (or keep the parquet as a Dask DataFrame) and do
Lloyd's with array ops. After the partition refactor this is now a
representation change only, not a complexity change — deferred, together
with the out-of-core loader, to the dedicated next phase.

### Correctness / robustness caveats

1. **Seed non-reproducibility** (known, `docs/CHANGES.md` 2026-07-19):
   Bernoulli sampling used the global `np.random` from multiple scheduler
   threads in nondeterministic order.
   **Status: RESOLVED (`bc66c19`).** All draws now come from
   `SeedSequence(seed)` with per-(partition, round) RNGs; same seed ⇒
   bit-identical runs (verified in `smoke_test.py`). This includes the
   weighted-reclustering `random_state`, which was a second, previously
   unnoticed reproducibility hole (sklearn k-means++ used the global RNG).
2. **`fit()` convergence check extra full pass** — **RESOLVED**: the
   label-change count is computed inside the assignment pass itself
   (`_lloyd_pass` compares with the previous iteration's per-partition
   labels). No extra pass remains.
3. **Empty-cluster edge case**: silent before — **RESOLVED**: `fit()` now
   emits a warning (once per fit) when a center receives no points.
4. **psi == 0 early-return broke the (k, d) contract** — **RESOLVED**:
   `starting_centroids` is now `np.repeat(c0, k, axis=0)` (cost 0 either
   way; degenerate data only).
5. **Global `np.random.seed(seed)` side effect** — **RESOLVED**: removed;
   the instance uses its own `Generator`/`SeedSequence`.
6. **`run_benchmark` materializes the full dataset client-side** once to
   re-scatter per partition count. Fine for <= full KDD (~1.3 GB float64);
   STILL OPEN — moves to the out-of-core phase (together with conditional
   `persist` and per-worker sharding).

### Review round 2 (2026-08-23, commits `835dfb9` + `cf17263`) — all RESOLVED

7. **Fake averaging in `run_benchmark`**: every repetition reused the same
   seed ⇒ bit-identical runs, meaningless mean/std. **RESOLVED**: repetition
   i uses `seed + i`; `seed` column recorded in CSVs.
8. **Silent `r` override, duplicated in two layers**: `l/k<=0.1 ⇒ 15` was
   applied both in the engine and in the driver, discarding an explicit
   `r`. **RESOLVED**: single `resolve_rounds()` (`policy="auto"` keeps the
   paper rule as historical default; `"fixed"` bypasses it); drivers record
   the effective rounds (`n_rounds_` / CSV column `r_effective`).
9. **Duplicate RNG stream**: two Generators on the same `SeedSequence`
   duplicated the stream (reclustering `random_state` shared bits with the
   initial-centroid draw); `entropy % 2**32` collided for seeds 2^32 apart.
   **RESOLVED**: `spawn(3)` tree + pre-derived per-(partition, round)
   sequences. NB: sampling changes at equal seed vs round 1 (intentional;
   fit golden fixtures unaffected).
10. **Client-side state ping-pong**: full `(m,2)` state matrices (seeding)
    and `(m,)` label vectors (Lloyd's) crossed client↔cluster every
    round/iteration. **RESOLVED**: chained Delayed state, fused cost scalar,
    labels kept in-graph (`t[4]`); partition matrices persisted once.
11. **Broad `except ValueError`** masking real bugs as "FAILED" rows.
    **RESOLVED**: only the known too-few-candidates reclustering error is
    caught (message-checked); everything else propagates.
12. Minor batch: unused `startup_timeout`/dead `POLL_INTERVAL`
    (now enforced via `wait_for_workers`), unconditional dataset download
    (cached), `meta` dtype-as-type (now `'float64'`), mean-instead-of-median
    in Fig-5.x plots (now `stat="median"` default), missing usage guards
    (added `RuntimeError`s), inertia duplication (shared `inertia_of_bag`).

### Optimality verdict

The *algorithm* implements the paper faithfully (Bernoulli sampling,
weighted reclustering, strict Lloyd's). After the 2026-08-23 refactor the
*parallelization strategy* is also appropriate for the task: every pass is
a partition-level dense linear-algebra reduction with only k×d partials
crossing task boundaries, and seeding is deterministic. Confirmed by the
~30x local speedup with bit-identical results. Remaining opportunities
(Dask Array representation, out-of-core operation) are deliberately
deferred to a dedicated phase — see `docs/ANALYSIS_PLAN.md` and TODO.


## Part 3 — Gaps to reproduce the paper (Tables 3/4/6, Fig 5.1/5.2)

Reference: `docs/1203.6402v1.pdf`. Partition baseline excluded per plan.

| Artifact | Paper protocol | Status in repo |
|---|---|---|
| Table 3 (KDD full cost) | k in {500,1000}, r=5, l in {0.1k,0.5k,k,2k,10k} + Random; cost x1e-10; median protocol | **Driver ready**: `paper_experiments.run_table34` (policy="fixed" r=5 — the auto rule must NOT apply here; Random via r=0 path); needs cluster session |
| Table 4 (times) | init + Lloyd time per method (Hadoop) | **Driver ready**: same run records `time_seed`/`time_fit`; `table34_time_table` pivots them. Adapted: same table structure measured on our Dask cluster (not comparable to paper's absolute numbers) |
| Table 6 (Lloyd iterations, Spam) | k in {20,50,100}; Random / k-means++ / k-means||(0.5k,5) / (2k,5); avg over 10 runs | **OUT OF SCOPE** per user decision (ANALYSIS_PLAN): no Spam loader |
| Fig 5.1 (cost vs rounds, KDD 10%) | k in {17,33,65,129}, l/k in {1,2,4}, r = 1..10, **exactly l samples per round** (joint distribution), median of 11 runs | **Driver ready**: `run_fig51` with `sampling="exact"` (distributed Efraimidis–Spirakis); needs cluster session |
| Fig 5.2 (cost vs rounds, GaussMixture) | n=10k, 15-dim, R in {1,10,100}, l/k in {0.1,..,10}, r = 0..15, k-means++ horizontal reference | **Driver ready, local-runnable**: `run_fig52` + `make_gauss_mixture`; r=0 degrades to uniform random k centers as required |

New work status (from the original gap list):

1. ~~`make_gauss_mission` / Spam loader~~ — GaussMixture DONE (`data_loader.make_gauss_mixture`, `array_to_bag`); Spam dropped by scope decision.
2. ~~sampling modes / r=0 / iteration count~~ — ALL DONE (`sampling="bernoulli"|"exact"`, r=0 random baseline under policy="fixed", `n_iter_` + `n_rounds_`).
3. ~~sweep drivers + plotting~~ — DONE (`src/paper_experiments.py`: run_fig51/fig52/table34, plot_fig51/fig52, table34_cost/time tables; known pool<k failures handled benignly for overnight robustness).
4. Notebook skeleton DONE (`notebooks/paper_reproduction.ipynb`, flag-gated sections). Remaining: EXECUTION on the SSH cluster (sessions A/B of ANALYSIS_PLAN) after B0/B1 validation passes.
