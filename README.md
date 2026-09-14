# MAPD-B — Parallel k-means|| on Dask

Parallel implementation of **k-means||** (Bahmani et al., *Scalable K-Means++*, VLDB 2012) on a Dask cluster, evaluated against serial **k-means++** / **random** seeding, followed by distributed Lloyd's iterations. Dataset: **KDD Cup 1999** (10% and full) and a synthetic **GaussMixture** for the paper's Fig. 5.2.

Course project — MAPD-B. Clustering cost (inertia) is the controlled metric; wall-clock times are reported only for reference because serial and distributed code run on different hardware.

---

## How to read this submission

**Start with `notebooks/final_analysis.ipynb`.**

That single notebook is the complete, self-contained final work — introduction, method, and all experiments with inline figures and tables. No other notebook needs to be run to evaluate the project.

> **Note on the filename:** in this development repository the file is stored as `notebooks/final_presentation.ipynb`. In the cleaned submission it is delivered as `notebooks/final_analysis.ipynb` (same content, renamed). Open either — they are identical.

All other notebooks (`analysis.ipynb`, `run.ipynb`) and the `docs/` folder are development artifacts and are **not** part of the submission. The `docs/` folder has been omitted intentionally.

---

## Repository structure

```
├── notebooks/
│   └── final_analysis.ipynb        # evaluated notebook (this repo: final_presentation.ipynb)
├── src/                            # core code, imported as src.<module>
│   ├── kmeans_parallel.py          # k-means|| seeding + distributed Lloyd's (dask.array)
│   ├── kmeans_serial.py            # serial baselines: k-means++ / random (scikit-learn)
│   ├── data_loader.py              # download → Parquet shards → scatter → standardize → dask.array
│   ├── benchmark.py                # run_single_test / run_benchmark grid runner
│   ├── paper_experiments.py        # paper reproduction drivers (Figs. 5.1/5.2, Tables 3/4)
│   └── launch_cluster.py           # Dask SSHCluster bootstrap
├── data/                           # raw .gz (gitignored, regenerable)
├── results/                        # benchmark CSVs 
├── figures/                        # committed plots rendered in the final notebook
├── scripts/                        # cluster helpers (environment checks, worker sync)
├── environment.yml
└── requirements.txt                # exact freeze of the cluster VMs
```

* `data/` and `results/` are never committed, they are recreated locally by `data_loader.load_dataset()` and by the benchmark drivers. `figures/` contains the plots already embedded in the final notebook.

---

## What the code does

```
kddcup.data.gz --stream--> N Parquet shards (cached on disk)
        │
        │  client.scatter  (one shard per worker, round-robin)
        ▼
workers: pass 1  per-shard stats  ──> global mean / std / constant columns
workers: pass 2  standardize      ──> dask.array X (n, d), one chunk per shard
        │
        ▼
kmeans_parallel.compute_starting_centroids(X)   # r rounds of D²-sampling + weighted reclustering to k
        │
        ▼
kmeans_parallel.fit(X)                          # distributed Lloyd's iterations
        │
        ▼
inertia_of_bag(X, centroids)                    # shared cost function (same for serial baselines)
```

Key invariants:

* Every algorithm step is one vectorized NumPy task per partition — only `k × d` matrices and scalars cross partition boundaries.
* Seeding is deterministic: all randomness derives from `SeedSequence(seed)` with per-(partition, round) RNGs. Same seed gives bit-identical results.
* Preprocessing drops the non-numeric columns (`protocol_type`, `service`, `flag`), the `label` column, and constant columns, coerces to numeric, and z-scores with the global mean/std before any distance computation.

---

## Dataset

* **KDD Cup 1999** — network intrusion records. 10% subset (~494k points) for development and Fig. 5.1; full set (~4.9M points) for Tables 3/4 and the worker/partition sweep. 41 columns → 38 numeric features after cleaning.
* **GaussMixture** — synthetic `make_gauss_mixture(n=10k, d=15, k=50, R∈{1,10,100})` used for Fig. 5.2. Pure in-memory, no disk I/O.

---

## View the final results

1. Open `notebooks/final_analysis.ipynb`.
2. The notebook is fully rendered, figures and Tables are visible without re-execution.

---

## Reference

Bahmani, Moseley, Vattani, Kumar, Vassilvitskii — *Scalable K-Means++*, VLDB 2012.

## AI use

Generative AI was used throughout this project mainly as a debugging aid: resolving errors in the code, diagnosing failures in the distributed (Dask) pipeline, and working through implementation bottlenecks that came up while developing and running the algorithm on the cluster. All code and results were reviewed, tested, and validated by the team before being included in this report.
 
