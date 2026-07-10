"""
download_rcv1.py

Setup script (to be run ONCE on the VM) that downloads the RCV1 dataset via
scikit-learn and saves it in an optimized .npz format, ready to be loaded
later in an analysis notebook with Dask.

Usage:
    python download_rcv1.py
    python download_rcv1.py --force            # re-download/re-save even if already present
    python download_rcv1.py --processed-dir /data/rcv1

Output:
    <processed-dir>/rcv1_X.npz   sparse feature matrix (804414 x 47236)
    <processed-dir>/rcv1_y.npz   sparse target matrix  (804414 x 103)
    <processed-dir>/meta.json    metadata (shape, nnz, download time, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import scipy.sparse as sp
from sklearn.datasets import fetch_rcv1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_rcv1")

MIN_FREE_GB = 5  # safety threshold for free disk space


def check_disk_space(path: Path, min_free_gb: float = MIN_FREE_GB) -> None:
    """Check that there is enough free disk space before downloading."""
    total, used, free = shutil.disk_usage(path)
    free_gb = free / (1024**3)
    log.info("Free disk space (%s): %.2f GB", path, free_gb)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough free space on {path}: {free_gb:.2f} GB "
            f"(minimum required: {min_free_gb} GB). Free up space before proceeding."
        )


def download_and_save(
    data_home: Path,
    processed_dir: Path,
    force: bool = False,
) -> dict:
    """
    Download RCV1 (if not already cached by scikit-learn) and save X, y in
    .npz format under processed_dir. Returns a metadata dictionary.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    x_path = processed_dir / "rcv1_X.npz"
    y_path = processed_dir / "rcv1_y.npz"
    meta_path = processed_dir / "meta.json"

    if x_path.exists() and y_path.exists() and not force:
        log.info(
            ".npz files already present in %s (use --force to regenerate). Skipping.",
            processed_dir,
        )
        with open(meta_path) as f:
            return json.load(f)

    log.info("Downloading/parsing RCV1 (data_home=%s)...", data_home)
    t0 = time.time()
    rcv1 = fetch_rcv1(data_home=data_home, download_if_missing=True)
    download_time = time.time() - t0
    log.info("Download/parsing completed in %.1f s", download_time)

    X = rcv1.data.tocsr()
    y = rcv1.target.tocsr()

    log.info(
        "X: shape=%s, nnz=%d (%.3f%%), dtype=%s",
        X.shape, X.nnz, 100 * X.nnz / (X.shape[0] * X.shape[1]), X.dtype,
    )
    log.info(
        "y: shape=%s, nnz=%d (%.3f%%), dtype=%s",
        y.shape, y.nnz, 100 * y.nnz / (y.shape[0] * y.shape[1]), y.dtype,
    )

    log.info("Saving in .npz format...")
    t0 = time.time()
    sp.save_npz(x_path, X)
    sp.save_npz(y_path, y)
    save_time = time.time() - t0
    log.info("Save completed in %.1f s", save_time)

    meta = {
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "n_categories": y.shape[1],
        "x_nnz": int(X.nnz),
        "y_nnz": int(y.nnz),
        "x_path": str(x_path),
        "y_path": str(y_path),
        "x_size_mb": round(x_path.stat().st_size / 1024**2, 1),
        "y_size_mb": round(y_path.stat().st_size / 1024**2, 1),
        "download_time_s": round(download_time, 1),
        "save_time_s": round(save_time, 1),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Metadata saved to %s", meta_path)
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-home",
        type=Path,
        default=Path.home() / "scikit_learn_data",
        help="Raw scikit-learn cache directory (default: ~/scikit_learn_data)",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path.home() / "datasets" / "rcv1",
        help="Destination directory for the .npz files (default: ~/datasets/rcv1)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate the .npz files even if they already exist",
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_GB,
        help=f"Minimum required free disk space in GB (default: {MIN_FREE_GB})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        check_disk_space(args.processed_dir.parent, args.min_free_gb)
        meta = download_and_save(
            data_home=args.data_home,
            processed_dir=args.processed_dir,
            force=args.force,
        )
        log.info("Setup completed. Summary:\n%s", json.dumps(meta, indent=2))
        return 0
    except Exception:
        log.exception("Setup failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
