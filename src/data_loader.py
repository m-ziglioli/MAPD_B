"""
data_loader.py
==============
End-to-end data pipeline for the KDD Cup 1999 dataset, bounded-memory even
on the full dataset:

1. Master downloads the .gz (skipped if cached, unless force_download).
2. Master converts the .gz into ``n_partitions`` Parquet shard files inside
   the directory ``parquet_path`` (streaming: the whole CSV is never held
   in RAM). Shards are cached: re-running with the same parameters reuses
   them.
3. Each shard is scattered to ONE worker (round-robin via
   ``client.scatter``): no worker holds the entire dataset, and the master
   holds at most one shard in RAM at a time (no full-file ``f.read()``
   broadcast).
4. Preprocessing runs on the workers in two passes:
   - pass 1: per-shard reduced statistics -> global mean/std/min/max and
     constant columns;
   - pass 2: per-shard standardization -> dense (m, d) matrix.
5. Returns a ``dask.array`` (n, d) whose chunks are the per-shard matrices
   (one chunk per partition, known shapes), plus (mean, std) as pandas
   Series to go back to the original coordinates.

Synthetic generators and in-memory helpers (used for the paper
reproduction, see docs/ANALYSIS_PLAN.md) live at the bottom: no network or
disk access.
"""

import os
import io
import gzip
import urllib.request
import warnings

import numpy as np
import pandas as pd
import dask
import dask.array as da
import pyarrow as pa
import pyarrow.parquet as pq

# Non-numeric columns (dropped) and the label column (not a feature).
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COL = "label"


def _count_lines_gz(filepath):
    """Count the number of lines in a .gz file (used for partitioning)."""
    count = 0
    # newline="\n" disables universal-newline translation: in default text
    # mode a file with CRLF line endings (Windows) is counted twice (once
    # per \r and once per \n). An explicit newline counts real records.
    with gzip.open(filepath, "rt", newline="\n") as f:
        for _ in f:
            count += 1
    return count


def _write_shards(raw_gz_path, shard_files, col_names):
    """Convert the .gz into one Parquet shard per chunk, streaming: the
    master never loads the entire dataset in memory."""
    n_total_rows = _count_lines_gz(raw_gz_path)
    chunk_size = int(np.ceil(n_total_rows / len(shard_files)))
    print("Converting .gz -> Parquet shards...")
    schema = pa.schema({col: pa.string() for col in col_names})
    with gzip.open(raw_gz_path, "rt") as f_in:
        reader = pd.read_csv(
            f_in, header=None, names=col_names, dtype=str, chunksize=chunk_size
        )
        for i, chunk_df in enumerate(reader):
            table = pa.Table.from_pandas(chunk_df, schema=schema)
            pq.write_table(table, shard_files[i], compression="snappy")
    print(f"Parquet shards created ({len(shard_files)} files, snappy).")


def _clean_shard_df(df, constant_cols=()):
    """Per-shard preprocessing, identical to the old dask.dataframe
    pipeline: drop categorical/label columns, coerce to numeric, dropna,
    drop constant columns."""
    df = df.drop(columns=CATEGORICAL_COLS + [LABEL_COL], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df


def _shard_stats(data):
    """Pass 1: reduced statistics of one shard (count, sum, sumsq, min, max)
    plus the column list (deterministic order = col_names order)."""
    df = pd.read_parquet(io.BytesIO(data))
    df = _clean_shard_df(df)
    cols = list(df.columns)
    n = len(df)
    d = len(cols)
    if n == 0:
        # Empty shard after dropna: must not influence the global min/max.
        return n, np.zeros(d), np.zeros(d), np.full(d, np.inf), np.full(d, -np.inf), cols
    return (
        n,
        df.sum().to_numpy(dtype=np.float64),
        (df ** 2).sum().to_numpy(dtype=np.float64),
        df.min().to_numpy(dtype=np.float64),
        df.max().to_numpy(dtype=np.float64),
        cols,
    )


def _shard_matrix(data, constant_cols, final_cols, mean, std):
    """Pass 2: re-read one shard, re-apply preprocessing and return the
    standardized (m, d) matrix (same column order as ``final_cols``)."""
    df = pd.read_parquet(io.BytesIO(data))
    df = _clean_shard_df(df, constant_cols)
    df = df[final_cols]
    return (df.to_numpy(dtype=np.float64) - mean) / std


def _delayed_matrices_to_array(matrix_tasks, row_counts, n_features):
    """List of Delayed (each a 2-D (m_i, d) matrix) -> dask.array (n, d)
    with one chunk per Delayed.

    ``da.from_delayed`` does not accept lists, so each shard is wrapped
    individually and then concatenated along axis 0. Chunk shapes MUST be
    known: with ``(np.nan, d)`` dask treats unknown chunks as size 1 in
    concatenate/slicing and silently produces wrong results. The per-shard
    counts come from pass 1 (post-dropna, same preprocessing as pass 2, so
    they are exact)."""
    parts = [
        da.from_delayed(t, shape=(int(m), n_features), dtype=np.float64)
        for t, m in zip(matrix_tasks, row_counts)
    ]
    if len(parts) == 1:
        return parts[0]
    return da.concatenate(parts, axis=0)


def load_dataset(dataset_url, raw_gz_path, parquet_path, col_names,
                 n_partitions=4, client=None, force_download=False,
                 parquet_path_workers=None, **kwargs):
    """
    End-to-end data pipeline, bounded-memory even on the full dataset.

    Parameters
    ----------
    dataset_url : str
        URL of the compressed dataset (used only if raw_gz_path is missing
        or force_download=True).
    raw_gz_path : str
        Local path of the cached .gz file on the master.
    parquet_path : str
        Directory (on the master) where the Parquet shards are written and
        cached. Shards are scattered directly to the workers from here;
        nothing is copied to the workers' disks.
    col_names : list of str
        Column names of the CSV records (KDD Cup has no header).
    n_partitions : int
        Number of Parquet shards == number of chunks of the returned array
        == number of partitions seen by the k-means engine. Scattered
        round-robin over the workers: with the project convention
        n_partitions = 8 * workers each worker holds exactly 8 shards.
    client : dask.distributed.Client
        Required: shards are scattered through it.
    force_download : bool
        Re-download the .gz and re-write the shards even if cached.
    parquet_path_workers : str, optional
        DEPRECATED, ignored: kept only for backward compatibility with
        notebooks that still pass it. Workers now receive shards via
        ``client.scatter`` and never write Parquet to their own disks
        (see ``docs/CHANGES.md`` 2026-09-09). If supplied, a
        ``DeprecationWarning`` is emitted and the value is ignored.

    Returns
    -------
    X : dask.array (n, d), float64
        Standardized features; one chunk per shard (known shapes).
    (mean, std) : pandas Series
        Global mean/std used for standardization, indexed by feature name.
    """
    # Backward-compat shim: parquet_path_workers was used by the old
    # single-file Parquet pipeline (client.run broadcast to worker disks).
    # The shard pipeline scatters bytes and never uses it; keep the kwarg
    # so old notebook cells that still pass it do not raise TypeError.
    # It is intentionally ignored (warn once) rather than restored.
    if parquet_path_workers is not None or "parquet_path_workers" in kwargs:
        warnings.warn(
            "parquet_path_workers is deprecated and ignored: shards are now "
            "scattered via client.scatter, workers never write Parquet to disk",
            DeprecationWarning,
            stacklevel=2,
        )
    # Positional compat: old call was (..., parquet_path, parquet_path_workers, col_names)
    # If col_names looks like a parquet path string and n_partitions holds the real col_names,
    # shift them (covers stale positional calls like load_dataset(url, raw, pq, pq_w, cols)
    # which now maps pq_w->col_names and cols->n_partitions). Keyword calls are unaffected.
    if isinstance(col_names, (str, bytes)) and isinstance(n_partitions, (list, tuple)):
        # old: col_names slot holds pq_workers string, n_partitions slot holds real col_names list
        col_names, n_partitions, parquet_path_workers = n_partitions, 4, None
        # also shift client/force_download if they were passed positionally after
        # (heuristic: if client looks like int, it was really n_partitions)
        if isinstance(client, int) and not isinstance(client, bool):
            n_partitions = client
            client = None
    elif isinstance(col_names, (str, bytes)) and isinstance(parquet_path_workers, (list, tuple)):
        col_names, parquet_path_workers = parquet_path_workers, None

    if client is None:
        raise ValueError("load_dataset requires a Dask client (pass client=client)")

    # Ensure src.* tasks are serializable by value even when the cluster was
    # started standalone and the notebook only did Client(SCHEDULER_ADDRESS).
    # _enable_pickle_by_value is client-process local and idempotent; calling
    # it here makes load_dataset self-contained and avoids ModuleNotFoundError
    # on workers (see src/launch_cluster.py and docs/CHANGES.md 2026-08-24).
    try:
        from src.launch_cluster import _enable_pickle_by_value
        _enable_pickle_by_value()
    except Exception:
        pass  # soft-fail: if cloudpickle missing, let the task error surface normally

    os.makedirs(parquet_path, exist_ok=True)

    # --- 1. Download the .gz (cached) ---
    if force_download or not os.path.exists(raw_gz_path):
        print("Downloading compressed dataset...")
        urllib.request.urlretrieve(dataset_url, raw_gz_path)
    else:
        print(f"Using cached dataset: {raw_gz_path}")

    # --- 2. GZ -> Parquet shards on the master (streaming, cached) ---
    shard_files = [
        os.path.join(parquet_path, f"shard_{i:05d}.parquet")
        for i in range(n_partitions)
    ]
    if force_download or not all(os.path.exists(f) for f in shard_files):
        _write_shards(raw_gz_path, shard_files, col_names)
    else:
        print(f"Using cached parquet shards: {parquet_path}")

    # --- 3. Scatter one shard per worker (round-robin) ---
    # The master holds at most one shard in RAM at a time; scattered futures
    # are sticky, so tasks consuming them run where the shard lives. Note:
    # if n_partitions < n_workers some workers receive no shard and stay idle.
    # Use hash=False to avoid hashing large bytes; do not use direct=True
    # with explicit workers (stale scheduler_info causes KeyError in
    # scheduler.update_data, see VM traceback tcp://10.67.22.121:35575).
    # Let scheduler choose placement - it already does round-robin-ish and
    # avoids the Stream is closed bottleneck via hash=False.
    futures = []
    for f in shard_files:
        with open(f, "rb") as fh:
            data = fh.read()
        # hash=False avoids CPU hashing of large shards, broadcast=False is default (one copy)
        futures.append(client.scatter(data, hash=False))
    # Delayed references to the shards already materialized on the workers.
    delayed_shards = [dask.delayed(fu) for fu in futures]

    # --- 4a. Pass 1: global statistics ---
    # Pin compute to the passed client to avoid the default-client trap:
    # dask.compute without scheduler uses get_client() (most recently created
    # Client), which may differ from the `client` that owns the scattered
    # futures when notebooks have created two Client objects. Pinning avoids
    # "already forgotten" cancellations. See analysis in prior assistant turn.
    stats = dask.compute(*[dask.delayed(_shard_stats)(s) for s in delayed_shards], scheduler=client)
    counts = [s[0] for s in stats]
    sums = np.vstack([s[1] for s in stats])
    sq_sums = np.vstack([s[2] for s in stats])
    mins = np.vstack([s[3] for s in stats])
    maxs = np.vstack([s[4] for s in stats])
    cols = stats[0][5]

    total_count = float(sum(counts))
    global_sum = sums.sum(axis=0)
    global_sq = sq_sums.sum(axis=0)
    global_min = mins.min(axis=0)
    global_max = maxs.max(axis=0)

    mean_all = global_sum / total_count
    # Sample variance (ddof=1), matching the old dask.dataframe pipeline.
    var_all = (global_sq - total_count * mean_all ** 2) / (total_count - 1)
    std_all = np.sqrt(np.maximum(var_all, 0.0))

    # Constant columns carry no information (expected: 'num_outbound_cmds').
    constant_cols = [c for c, lo, hi in zip(cols, global_min, global_max) if lo == hi]
    final_cols = [c for c in cols if c not in constant_cols]
    final_idx = [cols.index(c) for c in final_cols]
    final_mean = mean_all[final_idx]
    final_std = std_all[final_idx]
    print("Constant columns:", constant_cols)

    # --- 4b. Pass 2: standardized per-shard matrices -> dask.array ---
    matrix_tasks = [
        dask.delayed(_shard_matrix)(s, constant_cols, final_cols, final_mean, final_std)
        for s in delayed_shards
    ]
    X = _delayed_matrices_to_array(matrix_tasks, counts, len(final_cols))

    mean_series = pd.Series(final_mean, index=final_cols)
    std_series = pd.Series(final_std, index=final_cols)

    print(f"Distributed dask.array created with {X.npartitions} partitions.")
    print("Number of samples:", int(total_count))
    return X, (mean_series, std_series)


# ---------------------------------------------------------------------------
# Synthetic generators and in-memory helpers (used for the paper
# reproduction, see docs/ANALYSIS_PLAN.md): no network/disk access.
# ---------------------------------------------------------------------------

def make_gauss_mixture(n, k, d=15, R=1.0, seed=None):
    """GaussMixture of Fig 5.2 of Bahmani et al. (2012): k centers ~
    N(0, R*I_d), each point assigned to a center uniformly at random and
    drawn as N(center, I_d), equal weights.

    Returns (X (n,d) float64, y (n,) labels, centers (k,d)).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, float(R), size=(k, d))
    y = rng.integers(0, k, size=n)
    X = centers[y] + rng.normal(0.0, 1.0, size=(n, d))
    return X.astype(np.float64), y.astype(np.int64), centers


def array_to_dask(X, n_partitions=4):
    """Numpy array (n,d) -> dask.array (n,d) in ``n_partitions`` chunks: the
    SAME format produced by load_dataset (one chunk = one 2-D partition
    matrix), useful for local tests and for the GaussMixture without going
    through Parquet."""
    n_partitions = max(1, min(int(n_partitions), X.shape[0]))
    chunk = int(np.ceil(X.shape[0] / n_partitions))
    return da.from_array(X.astype(np.float64), chunks=(chunk,))
