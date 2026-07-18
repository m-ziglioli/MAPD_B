import os
import gzip
import urllib.request
import pandas as pd
import numpy as np
import dask
import dask.dataframe as dd
import dask.bag as db
import pyarrow as pa
import pyarrow.parquet as pq


# helper to count number lines in a gz file, used for partitioning later
def _count_lines_gz(filepath):
    count = 0
    with gzip.open(filepath, 'rt') as f:
        for _ in f:
            count += 1
    return count

def load_dataset(dataset_url, raw_gz_path, parquet_path, parquet_path_workers, col_names, n_partitions=4, client=None): # Local path on each worker's disk 
                                  # has to be the same as master's, otherwise metadata issues 
    """
    1. Master downloads the .gz file, converts it to a Parquet file,
    separated into chunks of the specified size (total size / number of partitions).

    Then the Parquet dataset is converted to a distributed Dask bag of NumPy arrays. Steps:
      2. Read Parquet with Dask – workers read row groups in parallel.
      3. Preprocess: drop categorical columns, scale numeric features.
      4. Return a Dask bag where each element is a 1-D array of features.
    """
    os.makedirs("./data", exist_ok=True)

    # --- Download GZ file---
    print("Downloading compressed dataset...")
    urllib.request.urlretrieve(dataset_url, raw_gz_path)

    # --- Convert to Parquet---
    n_total_rows = _count_lines_gz(raw_gz_path)
    chunk_size = int(np.ceil(n_total_rows / n_partitions))
    print("Converting .gz -> Parquet chunk sizes...")
    schema = pa.schema({col: pa.string() for col in col_names})
    with gzip.open(raw_gz_path, "rt") as f_in, \
         pq.ParquetWriter(parquet_path, schema, compression='snappy') as writer:
        # pandas reads and discards chunks – never loads the whole dataset
        # (assumption: if a single worker can handle a single uncompressed partition, so can the master)
        for chunk_df in pd.read_csv(
            f_in,
            header=None,
            names=col_names,
            dtype=str,
            chunksize=chunk_size # equal to number of rows read at a time
        ):
            table = pa.Table.from_pandas(chunk_df, schema=schema)
            writer.write_table(table)
    print("Parquet file created (compressed with snappy).")

    #############################################################
    # Read the compressed Parquet file (master only)
    with open(parquet_path, 'rb') as f:
        parquet_bytes = f.read()
    
    # Copy it to every worker's disk (each worker gets its own full .parquet file on its disk, same path for all)
    def save_parquet_to_workers(data):
        with open(parquet_path_workers, 'wb') as f:
            f.write(data)
    client.run(save_parquet_to_workers, parquet_bytes) # assuming already existing client
        
    ddf = dd.read_parquet(parquet_path_workers, split_row_groups=True) # dask distributed dataframe, so from here on will be automatically parallelized (optimized for column operations)
    print("Number of partitions before preprocessing:", ddf.npartitions)

    # --- 3. Preprocessing ---
    # Drop the known categorical columns immediately (they are not numeric)
    categorical_cols = ["protocol_type", "service", "flag"]
    ddf = ddf.drop(columns=categorical_cols) 
    # Now convert the remaining columns to numeric
    ddf = ddf.map_partitions(
    lambda df: df.apply(pd.to_numeric, errors="coerce"),
    meta={c: 'f8' for c in ddf.columns}
    )
    # d) Keep only feature columns (remove 'label')
    # because we are using clustering cost only as a metric, this is not required
    ddf = ddf.drop(columns=['label'])
    # Drop rows that still contain NaN (if any numerical field was invalid)
    ddf = ddf.dropna()
    # check for constant columns by comparing their min and max (uninformative -> drop them) 
    col_min, col_max = dask.compute(ddf.min(), ddf.max())
    constant_cols = col_min[col_min == col_max].index.tolist()
    print("Constant columns:", constant_cols) # should be only the one named 'num_outbound_cmds'
    ddf = ddf.drop(columns=constant_cols)
    # e) Standardize using global mean & std (computed lazily, then applied)
    # result is a small (around 40 elements) Series, so not an issue when materialized anywhere
    print("Computing global mean and std (first pass over data)...")
    mean, std = dask.compute(ddf.mean(), ddf.std())
    # Apply scaling partition‑wise (workers do the work)
    feature_cols = list(ddf.columns)   # these are the names of the remaining (numeric) columns
    ddf = ddf.map_partitions(lambda df: (df - mean) / std,
                             meta={c: float for c in feature_cols})
                             # keep column names as metadata
    
    # --- 4. Convert to Dask bag of NumPy arrays ---
    # because this is what is used in Analysis notebook 
    X_bag = ddf.to_bag(index=False).map(
    lambda row: np.array(row, dtype=np.float64)
    )
    # final number of partitions might be smaller because dropna() might fuse small partitions
    print(f"Distributed bag created with {X_bag.npartitions} partitions.")
    print("Number of samples:", ddf.shape[0].compute())
    # return mean and std as well if want to go back to original coordinates later
    return X_bag, (mean,std)