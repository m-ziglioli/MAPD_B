"""
check_cluster_env.py
====================
Verifica l'ambiente dei WORKER del cluster in un colpo solo: le versioni dei
pacchetti freeze (requirements.txt) devono corrispondere su OGNI nodo.
Nato dall'incidente del 2026-08-24 (worker riprovisionati senza sklearn ->
import by-reference in task -> crash processo -> KilledWorker, vedi
docs/CHANGES.md): con questo check il problema emerge in secondi.

Uso (con un Client gia' attivo, es. dentro un notebook sul head VM):
    from scripts.check_cluster_env import check_workers
    check_workers(client)

Oppure standalone:
    python scripts/check_cluster_env.py            # usa Client() default
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PACKAGES = ["dask", "distributed", "numpy", "pandas", "pyarrow",
            "scikit-learn", "scipy", "matplotlib"]


def _probe():
    """Versioni installate NEL processo corrente (client o worker)."""
    import importlib.metadata as im

    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = im.version(pkg)
        except im.PackageNotFoundError:
            out[pkg] = "MANCANTE"
    return out


def check_workers(client):
    """Sonda ogni worker via client.run (nessuna dipendenza da get_worker,
    che non e' disponibile in quel contesto). Ritorna (local, remote) dove
    remote e' {worker_address: {pkg: versione}}. Stampa una tabella e un
    verdetto per differenze rispetto all'ambiente del client (il head,
    riferimento del freeze)."""
    local = _probe()
    remote = client.run(_probe)

    print(f"pacchetti: {', '.join(PACKAGES)}\n")
    print("client (head):")
    print("  " + "  ".join(f"{p}={local[p]}" for p in PACKAGES))
    print("worker:")
    mismatches = 0
    for addr, env in remote.items():
        diffs = [p for p in PACKAGES if env.get(p) != local[p]]
        if diffs:
            mismatches += 1
            flag = " <-- DIFFERENZE: " + ", ".join(
                f"{p}({env.get(p, '?')} vs {local[p]})" for p in diffs)
        else:
            flag = "OK"
        print(f"  {addr}:")
        print("    " + "  ".join(f"{p}={env.get(p, '?')}" for p in PACKAGES))
        print(f"    {flag}")

    if mismatches:
        print(f"\nVERDETTO: {mismatches} worker DISALLINEATI -> "
              "python scripts/sync_workers.py --install")
    else:
        print("\nVERDETTO: ambiente allineato su tutti i worker")
    return local, remote


def main():
    import argparse

    from dask.distributed import Client

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--address", default=None,
                    help="indirizzo dello scheduler (default: Client() default)")
    args = ap.parse_args()

    client = Client(args.address) if args.address else Client()
    try:
        check_workers(client)
    finally:
        if args.address:
            client.close()


if __name__ == "__main__":
    main()
