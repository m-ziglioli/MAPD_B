# Analysis plan — reproduction of Bahmani et al. (2012)

**Status: PREREQUISITES IMPLEMENTED, EXECUTION PENDING** (2026-08-23).
All "Code prerequisites" below are done and locally validated
(`src/paper_experiments.py`, `sampling="exact"`, r=0 random baseline,
`make_gauss_mixture`/`array_to_bag`, `notebooks/paper_reproduction.ipynb`;
see CHANGES.md). What remains are the cluster sessions in "Execution
phases" (after B0 checklist + B1 validation) and the out-of-core item in
"Deferred separately". Original decisions preserved below. Reference paper:
`docs/1203.6402v1.pdf` (arXiv 1203.6402, "Scalable K-MEANS++").

## Scope

Datasets (user decision): **KDD Cup 1999 full**, **KDD Cup 1999 10%**,
**GaussMixture (synthetic)** only. The Spam dataset and therefore **Table 6
are out of scope**. The **Partition** baseline is excluded everywhere.

| Artifact | Paper protocol | Adaptation |
|---|---|---|
| Table 3 (cost, KDD full) | k ∈ {500,1000}, r=5, ℓ ∈ {0.1k, 0.5k, k, 2k, 10k} + Random; cost scaled ×10⁻¹⁰ | median protocol like the rest of the paper (11 runs); report seed + final cost, raw and scaled |
| Table 4 (time, KDD full) | init + Lloyd time per method, Hadoop | same table structure timed on our Dask cluster; absolute numbers NOT comparable to the paper (AGENTS.md rule 5); Random bounded to 20 Lloyd iterations (paper's parallel protocol) |
| Fig 5.1 (cost vs rounds, KDD 10%) | k ∈ {17,33,65,129}, ℓ/k ∈ {1,2,4}, r = 1..10 (log x-axis; k=129 panel extends further), **exactly ℓ points sampled per round** from the joint d²/φ distribution, median of 11 runs, final cost after Lloyd's, log y | exact-ℓ sampling mode required |
| Fig 5.2 (cost vs rounds, GaussMixture) | n=10⁴, d=15, R ∈ {1,10,100}, ℓ/k ∈ {0.1,0.5,1,2,10}, r = 0..15, k-means++ horizontal reference, final cost, log y | k = 50 (user decision; matches Table 1 setting); r=0 ≡ uniform random k centers (Random baseline) |

Preprocessing: GaussMixture needs none by construction; no scaling decisions
pending (Spam dropped).

## Code prerequisites (to implement when resuming)

1. `src/data_loader.py`:
   - `make_gauss_mixture(n, k, d=15, R, seed)` — k centers ~ N(0, R·I_d),
     points ~ N(center, I_d) with equal weights;
   - `array_to_bag(X, n_partitions)` — local numpy → Bag of 1-D rows.
2. `src/kmeans_parallel.py`:
   - `sampling="bernoulli"|"exact"` in `compute_starting_centroids`
     (exact = per round, sample exactly ℓ points without replacement with
     probability ∝ d²(x,C)/φ_X(C) — the Fig 5.1 protocol; note the paper
     uses this variant *only* for Fig 5.1);
   - `r=0` → uniform random k centers (no reclustering), so Fig 5.2's
     x-axis starts at the Random baseline;
   - `n_iter_` exposure (may already exist after the perf refactor).
3. New `src/paper_experiments.py`:
   - `run_fig51` / `run_fig52` / `run_table3` / `run_table4` sweep drivers
     → CSVs in `results/`;
   - plotting: Fig 5.1 (2×2 panels, log-y median curves per ℓ/k), Fig 5.2
     (3 panels R∈{1,10,100}, curves per ℓ/k + KM++ horizontal line)
     → `figures/`.
4. `notebooks/paper_reproduction.ipynb` — one section per artifact with
   obtained-vs-paper side-by-side tables; GaussMixture sections runnable
   locally, KDD sections require the SSH cluster.

## Execution phases (when resumed)

1. Implement prerequisites; validate every driver end-to-end locally on
   tiny grids (GaussMixture, reduced parameters) under the `mapd-b` env.
2. Cluster session A: Fig 5.1 (KDD 10%) — moderate cost.
3. Cluster session B: Tables 3/4 (KDD full) — the expensive one; plan
   ℓ/k sweep × k ∈ {500,1000} × 11 runs; consider running overnight.
4. Generate figures, compare with paper values, write up in
   `docs/RIASSUNTO` style.

## Known risks

- Seed determinism: resolved by the perf refactor (per-partition seeded
  RNGs) — medians over runs remain the reporting protocol regardless.
- Table 3's Random baseline cost is ~10 orders of magnitude above
  k-means|| in the paper; verify our pipeline reproduces the outlier-driven
  blow-up before the full sweep (cheap single-run sanity check first).
- Full-KDD runs at k=1000, ℓ=10k produce large candidate pools
  (~r·ℓ = 50k) — reclustering step cost is client-side and grows with the
  pool; acceptable but worth timing once early.

## Deferred separately (not part of this plan)

- Larger-than-memory (out-of-core) capability: partitioned Parquet dataset
  + per-worker shards instead of replicas, conditional persist, no
  client-side materialization in `run_benchmark`. Decided 2026-08-23 to do
  as its own phase with its own benchmarks.
