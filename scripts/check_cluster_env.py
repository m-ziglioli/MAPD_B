"""
check_cluster_env.py
====================
Verify the cluster WORKERS' environment in one shot: the versions of the
freeze packages (requirements.txt) must match on EVERY node.
Born from the 2026-08-24 incident (workers re-provisioned without sklearn
-> by-reference import in a task -> process crash -> KilledWorker, see
docs/CHANGES.md): with this check the problem surfaces in seconds.

Usage (with a Client already active, e.g. inside a notebook on the head VM):
    from scripts.check_cluster_env import check_workers
    check_workers(client)

Or standalone:
    python scripts/check_cluster_env.py            # uses the default Client()
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PACKAGES = ["dask", "distributed", "numpy", "pandas", "pyarrow",
            "scikit-learn", "scipy", "matplotlib"]


def _probe():
    """Versions installed IN the current process (client or worker)."""
    import importlib.metadata as im

    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            out[pkg] = "MISSING"
    return out


def check_workers(client):
    """Probe every worker via client.run (no dependency on get_worker,
    which is not available in that context). Returns (local, remote) where
    remote is {worker_address: {pkg: version}}. Prints a table and a
    verdict on differences against the client's environment (the head,
    the freeze reference)."""
    local = _probe()
    remote = client.run(_probe)

    print(f"packages: {', '.join(PACKAGES)}\n")
    print("client (head):")
    print("  " + "  ".join(f"{p}={local[p]}" for p in PACKAGES))
    print("workers:")
    mismatches = 0
    for addr, env in remote.items():
        diffs = [p for p in PACKAGES if env.get(p) != local[p]]
        if diffs:
            mismatches += 1
            flag = " <-- DIFFERENCES: " + ", ".join(
                f"{p}({env.get(p, '?')} vs {local[p]})" for p in diffs)
        else:
            flag = "OK"
        print(f"  {addr}:")
        print("    " + "  ".join(f"{p}={env.get(p, '?')}" for p in PACKAGES))
        print(f"    {flag}")

    if mismatches:
        print(f"\nVERDICT: {mismatches} workers MISALIGNED -> "
              "python scripts/sync_workers.py --install")
    else:
        print("\nVERDICT: environment aligned on all workers")
    return local, remote


def main():
    import argparse

    from dask.distributed import Client

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--address", default=None,
                    help="scheduler address (default: default Client())")
    args = ap.parse_args()

    client = Client(args.address) if args.address else Client()
    try:
        check_workers(client)
    finally:
        if args.address:
            client.close()


if __name__ == "__main__":
    main()
