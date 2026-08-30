import os
import io
import gzip
import urllib.request
import pandas as pd
import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.parquet as pq

# Colonne non numeriche (droppate) e colonna label (non e' una feature).
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COL = "label"


# helper to count number lines in a gz file, used for partitioning later
def _count_lines_gz(filepath):
    count = 0
    # newline="\n" disabilita la conversione universal newline: con file
    # Windows (\r\n) gzip testo in modalita' default conta le righe il doppio
    # (ogni \r e \n come fine riga). Con newline esplicito conta i record reali.
    with gzip.open(filepath, "rt", newline="\n") as f:
        for _ in f:
            count += 1
    return count


def _write_shards(raw_gz_path, shard_files, col_names, n_partitions):
    """Converte il .gz in ``n_partitions`` shard Parquet (uno per chunk),
    in streaming: il master non carica mai l'intero dataset in memoria."""
    n_total_rows = _count_lines_gz(raw_gz_path)
    chunk_size = int(np.ceil(n_total_rows / n_partitions))
    print("Converting .gz -> Parquet shards...")
    schema = pa.schema({col: pa.string() for col in col_names})
    with gzip.open(raw_gz_path, "rt") as f_in:
        reader = pd.read_csv(
            f_in, header=None, names=col_names, dtype=str, chunksize=chunk_size
        )
        for i, chunk_df in enumerate(reader):
            table = pa.Table.from_pandas(chunk_df, schema=schema)
            pq.write_table(table, shard_files[i], compression='snappy')
    print(f"Parquet shards created ({len(shard_files)} files, snappy).")


def _clean_shard_df(df, constant_cols=()):
    """Preprocessing per-shard, identico alla vecchia pipeline ddf:
    drop colonne categoriche/label, to_numeric, dropna, drop costanti."""
    df = df.drop(columns=CATEGORICAL_COLS + [LABEL_COL], errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()
    if constant_cols:
        df = df.drop(columns=constant_cols)
    return df


def _shard_stats(data):
    """Pass 1: statistiche ridotte di uno shard (count, sum, sumsq, min, max)
    + elenco colonne (ordine deterministico = ordine di col_names)."""
    df = pd.read_parquet(io.BytesIO(data))
    df = _clean_shard_df(df)
    cols = list(df.columns)
    n = len(df)
    d = len(cols)
    if n == 0:
        # shard vuoto dopo dropna: non deve influenzare min/max globali
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
    """Pass 2: rilegge lo shard, riapplica il preprocessing e ritorna la
    matrice (m, d) standardizzata (stesso ordine colonne ``final_cols``)."""
    df = pd.read_parquet(io.BytesIO(data))
    df = _clean_shard_df(df, constant_cols)
    df = df[final_cols]
    return (df.to_numpy(dtype=np.float64) - mean) / std


def load_dataset(dataset_url, raw_gz_path, parquet_path, parquet_path_workers, col_names, n_partitions=4, client=None, force_download=False):
    """
    Pipeline dati end-to-end, in memoria limitata anche sul 100%:

    1. Master scarica il .gz (skip se gia' in cache, a meno di force_download).
    2. Master converte il .gz in ``n_partitions`` shard Parquet dentro la
       directory ``parquet_path`` (streaming, mai l'intero dataset in RAM).
    3. Ogni shard viene scatterato a UN worker (round-robin via
       ``client.scatter``): nessun worker tiene l'intero dataset, e il master
       tiene in RAM un solo shard alla volta (niente ``f.read()`` full-file).
    4. Preprocessing in due passate sui worker:
       - pass 1: statistiche ridotte per shard -> global mean/std/min/max +
         colonne costanti;
       - pass 2: standardizzazione per shard -> matrice densa (m, d).
    5. Ritorna una ``dask.array`` (n, d) i cui chunk sono le matrici per
       shard, piu' (mean, std) come Series pandas per tornare alle coordinate
       originali.

    ``parquet_path_workers`` e' mantenuto per retrocompatibilita' di firma
    (i notebook lo passano ancora) ma non e' piu' usato: i worker ricevono gli
    shard direttamente via scatter, senza copiare file sul loro disco.
    """
    os.makedirs(parquet_path, exist_ok=True)

    # --- 1. Download GZ ---
    if force_download or not os.path.exists(raw_gz_path):
        print("Downloading compressed dataset...")
        urllib.request.urlretrieve(dataset_url, raw_gz_path)
    else:
        print(f"Using cached dataset: {raw_gz_path}")

    # --- 2. GZ -> shard Parquet (master, streaming) ---
    shard_files = [os.path.join(parquet_path, f"shard_{i:05d}.parquet") for i in range(n_partitions)]
    if force_download or not all(os.path.exists(f) for f in shard_files):
        _write_shards(raw_gz_path, shard_files, col_names, n_partitions)
    else:
        print(f"Using cached parquet shards: {parquet_path}")

    # --- 3. Scatter degli shard ai worker (round-robin, RAM master limitata) ---
    if client is None:
        raise ValueError("load_dataset richiede un client Dask (passare client=client)")
    futures = []
    for f in shard_files:
        with open(f, "rb") as fh:
            futures.append(client.scatter(fh.read()))

    # riferimenti ritardati agli shard gia' materializzati sui worker
    delayed_shards = [dask.delayed(fu) for fu in futures]

    # --- 4a. Pass 1: statistiche globali ---
    stats = dask.compute(*[dask.delayed(_shard_stats)(s) for s in delayed_shards])
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
    var_all = (global_sq - total_count * mean_all ** 2) / (total_count - 1)
    std_all = np.sqrt(np.maximum(var_all, 0.0))

    constant_cols = [c for c, lo, hi in zip(cols, global_min, global_max) if lo == hi]
    final_cols = [c for c in cols if c not in constant_cols]
    final_idx = [cols.index(c) for c in final_cols]
    final_mean = mean_all[final_idx]
    final_std = std_all[final_idx]

    print("Constant columns:", constant_cols)

    # --- 4b. Pass 2: matrici standardizzate per shard -> dask.array ---
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
# Generatori sintetici e helper (per la riproduzione dell'articolo, vedi
# docs/ANALYSIS_PLAN.md): nessun accesso a rete/disco, tutto in memoria.
# ---------------------------------------------------------------------------

def make_gauss_mixture(n, k, d=15, R=1.0, seed=None):
    """GaussMixture della Fig 5.2 di Bahmani et al. (2012): k centri ~
    N(0, R*I_d), ogni punto assegnato a un centro uniformemente e poi
    estratto come N(centro, I_d), pesi uguali.

    Ritorna (X (n,d) float64, y (n,) label, centers (k,d)).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, float(R), size=(k, d))
    y = rng.integers(0, k, size=n)
    X = centers[y] + rng.normal(0.0, 1.0, size=(n, d))
    return X.astype(np.float64), y.astype(np.int64), centers


def _delayed_matrices_to_array(matrix_tasks, row_counts, n_features):
    """Lista di Delayed (ognuno una matrice 2D (m_i, d) gia' standardizzata)
    -> dask.array (n, d) con un chunk per Delayed.

    ``da.from_delayed`` non accetta liste e le shape devono essere NOTE:
    con chunk (np.nan, d) il concatenate/slicing di dask tratta i chunk
    sconosciuti come grandezza 1 e producono risultati sbagliati. Le conte
    per shard arrivano dalla passata 1 (post-dropna, stesso preprocessing
    della passata 2, quindi esatte)."""
    parts = [
        da.from_delayed(t, shape=(int(m), n_features), dtype=np.float64)
        for t, m in zip(matrix_tasks, row_counts)
    ]
    if len(parts) == 1:
        return parts[0]
    return da.concatenate(parts, axis=0)


def array_to_dask(X, n_partitions=4):
    """Array numpy (n,d) -> dask.array (n,d) in ``n_partitions`` chunk: lo
    STESSO formato prodotto da load_dataset (un chunk = matrice 2D per
    partizione), utile per i test locali e per il GaussMixture senza passare
    da Parquet."""
    n_partitions = max(1, min(int(n_partitions), X.shape[0]))
    chunk = int(np.ceil(X.shape[0] / n_partitions))
    return da.from_array(X.astype(np.float64), chunks=(chunk,))
