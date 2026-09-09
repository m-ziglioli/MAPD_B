# MAPD-B — k-means|| on Dask

Parallel implementation of the **k-means|| initialization** (Bahmani et al.,
VLDB 2012) on a Dask cluster, benchmarked against **k-means++** and **random
seeding** (serial scikit-learn baselines), followed by distributed Lloyd's
iterations. Dataset: KDD Cup 1999 (10% and full), plus a synthetic
GaussMixture used to reproduce one figure of the paper.

## Repository layout

```
├── src/                    core modules (import as src.<module>)
│   ├── kmeans_parallel.py      k-means|| init + distributed Lloyd's fit (dask.array)
│   ├── kmeans_serial.py        serial baselines: k-means++ / random seeding
│   ├── kmeans_comparison.py    driver: k-means|| vs k-means++ vs random
│   ├── benchmark.py            run_single_test / run_benchmark grid runner
│   ├── benchmark_analysis.py   BenchmarkAnalyzer: sweep plots from result CSVs
│   ├── comparison_analysis.py  tables/plots for the 3-way comparison
│   ├── data_loader.py          download → Parquet shards → preprocess → dask.array
│   ├── paper_experiments.py    paper reproduction drivers (Fig 5.1/5.2, Tables 3/4)
│   └── launch_cluster.py       SSHCluster bootstrap (head + worker IPs)
├── notebooks/
│   ├── analysis.ipynb          k-means|| benchmarks: sweeps over k, l/k, workers, partitions
│   ├── comparison.ipynb        k-means|| vs k-means++ vs random comparison
│   ├── paper_reproduction.ipynb     paper artifact reproduction (flag-gated sections)
│   └── run.ipynb               scratch/driver notebook for cluster runs
├── data/                   raw + processed datasets (gitignored, regenerable)
├── results/                benchmark CSVs (gitignored, regenerable)
├── figures/                committed plots used in the report
├── docs/
│   ├── CHANGES.md              development changelog
│   ├── GUIDE.md                function-level codebase reference
│   ├── CODE_REVIEW.md          critical assessment of the parallelization
│   ├── ANALYSIS_PLAN.md        paper reproduction plan (frozen)
│   └── TODO.md                 open items
├── environment.yml / requirements.txt
```

## Setup

```bash
conda env create -f environment.yml     # or: pip install -r requirements.txt
conda activate mapd-b
python -m ipykernel install --user --name mapd-b --display-name "Python (mapd-b)"
```

`requirements.txt` replicates the exact package freeze of the cluster nodes
(see header), so local runs use the same dask/numpy/pyarrow stack as the VMs.

Run notebooks and scripts **from the repository root**, so that `src.*`
imports and the relative `results/`, `figures/` paths resolve correctly.

## Cluster

`src/launch_cluster.py` starts a `dask.distributed.SSHCluster` using the
hardcoded head/worker IPs and remote Python `/home/ubuntu/pyvenv/bin/python3`.
Update those constants for your own nodes.

```bash
python -m src.launch_cluster -n 4        # standalone, keeps cluster alive
```

or from a notebook:

```python
from src.launch_cluster import launch_cluster, shutdown_cluster
cluster, client = launch_cluster(n_workers=2)
```

Notes:
- `data_loader.load_dataset()` scatters one Parquet shard per worker via
  `client.scatter`, then preprocesses it in two passes on the workers and
  returns a `dask.array` `(n, d)` with known chunk shapes. Dataset path
  constants in the notebooks point to the cluster filesystem
  (`/home/ubuntu/...`, `/tmp/...`) — adjust them to your setup.
- The same code must be present on every node at the same path.

## Reproducing

1. Start the cluster and load the dataset (first cells of either notebook).
2. `analysis.ipynb`: parameter sweeps via `run_single_test` /
   `run_benchmark` → CSVs in `results/`, plots via `BenchmarkAnalyzer`
   into `figures/`.
3. `comparison.ipynb`: three-way seeding comparison via
   `kmeans_comparison.run_comparison` (cost after seeding and after Lloyd's,
   averaged over seeds, following Bahmani et al.'s setup).
4. `paper_reproduction.ipynb`: reproduction of the paper's synthetic and
   full-scale artifacts (Fig 5.1/5.2, Tables 3/4) via `src/paper_experiments.py`.

## Known issues

- Wall-clock times between serial and parallel methods are not directly
  comparable (different hardware: client vs cluster); cost/inertia is the
  controlled metric.
- `run_benchmark` still gathers the full dataset client-side once per run to
  re-scatter per partition count — fine up to full KDD (~1.3 GB float64);
  fully out-of-core re-partitioning remains deferred (see `docs/TODO.md`).

See `docs/CHANGES.md` for the full development log and `docs/GUIDE.md` for
a function-level codebase reference.
