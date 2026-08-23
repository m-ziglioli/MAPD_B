# MAPD-B — k-means|| on Dask

Parallel implementation of the **k-means|| initialization** (Bahmani et al.,
VLDB 2012) on a Dask cluster, benchmarked against **k-means++** and **random
seeding** (serial scikit-learn baselines), followed by distributed Lloyd's
iterations. Dataset: KDD Cup 1999.

## Repository layout

```
├── src/                    core modules (import as src.<module>)
│   ├── kmeans_parallel.py      k-means|| init + distributed Lloyd's fit (Dask Bag)
│   ├── kmeans_serial.py        serial baselines: k-means++ / random seeding
│   ├── kmeans_comparison.py    driver: k-means|| vs k-means++ vs random
│   ├── benchmark.py            run_single_test / run_benchmark grid runner
│   ├── benchmark_analysis.py   BenchmarkAnalyzer: sweep plots from result CSVs
│   ├── comparison_analysis.py  tables/plots for the 3-way comparison
│   ├── data_loader.py          download → Parquet → preprocess → Dask Bag
│   └── launch_cluster.py       SSHCluster bootstrap (head + worker IPs)
├── notebooks/
│   ├── analysis.ipynb          k-means|| benchmarks: sweeps over k, l/k, workers, partitions
│   └── comparison.ipynb        k-means|| vs k-means++ vs random comparison
├── data/                   raw + processed datasets (gitignored, regenerable)
├── results/                benchmark CSVs (gitignored, regenerable)
├── figures/                committed plots used in the report
├── docs/
│   ├── CHANGES.md              development changelog
│   ├── RIASSUNTO_confronto_kmeans.md   summary of the comparison results
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
- `data_loader.load_dataset()` copies the Parquet file to every worker at the
  **same absolute path** (`parquet_path_workers`); dataset path constants in
  the notebooks point to the cluster filesystem (`/home/ubuntu/...`,
  `/tmp/`) — adjust them to your setup.
- The same code must be present on every node at the same path.

## Reproducing

1. Start the cluster and load the dataset (first cells of either notebook).
2. `analysis.ipynb`: parameter sweeps via `run_single_test` /
   `run_benchmark` → CSVs in `results/`, plots via `BenchmarkAnalyzer`
   into `figures/`.
3. `comparison.ipynb`: three-way seeding comparison via
   `kmeans_comparison.run_comparison` (cost after seeding and after Lloyd's,
   averaged over seeds, following Bahmani et al.'s setup).

## Known issues

- Seed reproducibility is broken under Dask's threaded scheduler: Bernoulli
  sampling in k-means|| uses `np.random.uniform()` from multiple threads in
  nondeterministic order (see `docs/CHANGES.md`).
- Wall-clock times between serial and parallel methods are not directly
  comparable (different hardware: client vs cluster).
