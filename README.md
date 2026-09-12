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
├── data/                           # raw .gz + Parquet shards (gitignored, regenerable)
├── results/                        # benchmark CSVs (gitignored, regenerable)
├── figures/                        # committed plots rendered in the final notebook
├── scripts/                        # cluster helpers (environment checks, worker sync)
├── environment.yml
└── requirements.txt                # exact freeze of the cluster VMs
```

* `data/` and `results/` are never committed — they are recreated locally by `data_loader.load_dataset()` and by the benchmark drivers. `figures/` contains the plots already embedded in the final notebook.

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

## Environment and setup

Canonical environment is `mapd-b` (Python 3.13), frozen to the cluster VMs (`requirements.txt` header: dask/distributed 2026.6.0, numpy 2.2.6, pandas 2.3.3, pyarrow 24.0.0, scikit-learn 1.7.2, matplotlib 3.10.9, scipy 1.15.3).

```bash
# create the environment (once)
conda env create -f environment.yml
# or equivalently
conda create -n mapd-b python=3.13 -y
conda run -n mapd-b python -m pip install -r requirements.txt

conda activate mapd-b
python -m ipykernel install --user --name mapd-b --display-name "Python (mapd-b)"
```

Run all notebooks and scripts **from the repository root** so that `import src.*` and the relative `results/` / `figures/` paths resolve correctly:

```bash
jupyter notebook notebooks/final_analysis.ipynb
# select kernel: Python (mapd-b)
```

On the course VMs the notebooks expect `conda run -n mapd-b` and the `mapd-b` Jupyter kernel at `C:\Users\lcdit\anaconda3\Scripts\jupyter.exe` (bare `jupyter` is not on PATH).

---

## Running the final notebook

1. Open `notebooks/final_analysis.ipynb` and select the `mapd-b` kernel.
2. The notebook is fully rendered — figures and Tables 3/4 are visible without re-execution.
3. To re-execute, a running Dask cluster is required for the KDD sections (the GaussMixture section runs locally). The first code cells handle `launch_cluster` / `load_dataset`; dataset URLs and shard paths (`/home/ubuntu/...`, `/tmp/...`) match the course infrastructure and should be adapted for a different machine.

Outputs are written only to `results/` (CSVs) and `figures/` (PNGs). `data/` is written only by `data_loader`.

---

## Reference

Bahmani, Moseley, Vattani, Kumar, Vassilvitskii — *Scalable K-Means++*, VLDB 2012.
