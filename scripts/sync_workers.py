"""
sync_workers.py
===============
Deploy di ``src/`` + ``requirements.txt`` + file .pth su TUTTI i worker del
cluster (vedi docs/CHANGES.md 2026-08-24).

Perche' serve
-------------
I worker sono VM separate dal head e NON hanno la repo: senza questo step i
task serializzati by-reference non riescono a risolvere ``import src`` sul
worker (e la catena di import di src.kmeans_parallel richiede sklearn, che
sui worker puo' mancare -> crash del processo -> KilledWorker). Il file .pth
inserisce la repo nel sys.path del pyvenv remoto qualunque sia la cwd.

DA RIESEGUIRE dopo ogni ``git pull`` di codice nuovo (il deploy copia i
sorgenti correnti del head).

Uso (sul HEAD VM, dalla checkout, shell del progetto):
    python scripts/sync_workers.py              # deploy + verifica import
    python scripts/sync_workers.py --install    # + pip install requirements
    python scripts/sync_workers.py --check      # solo verifica import
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.launch_cluster import WORKER_IPS  # lista IP centralizzata (AGENTS.md)

REPO_DIR = Path("/home/ubuntu/Project/libero_development")
REMOTE_PY = "/home/ubuntu/pyvenv/bin/python3"

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
SCP = ["scp", "-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]

IMPORT_CHECK = (
    "import sklearn, src.kmeans_parallel, src.data_loader, src.benchmark; "
    "print('OK', sklearn.__version__)"
)


def run(cmd):
    """Esegue un comando remoto; ritorna l'output o stampa l'errore."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [FAIL] {' '.join(cmd[:4])}...")
        print("  " + (r.stderr or r.stdout).strip()[-400:])
        return None
    return (r.stdout or r.stderr).strip()


def remote_site_dir(host):
    """Directory site-packages del pyvenv remoto (robusto rispetto alla
    versione di python, invece di hardcodare .../python3.10/)."""
    out = run(SSH + [host, f"{REMOTE_PY} -c 'import site; print(site.getsitepackages()[0])'"])
    return out.strip().splitlines()[-1] if out else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--install", action="store_true",
                    help="pip install -r requirements.txt su ogni worker")
    ap.add_argument("--check", action="store_true",
                    help="solo verifica import (nessun deploy)")
    args = ap.parse_args()

    n_ok = 0
    for ip in WORKER_IPS:
        host = f"ubuntu@{ip}"
        print(f"=== {ip} ===")

        if not args.check:
            run(SSH + [host, f"mkdir -p {REPO_DIR}"])
            run(SCP + ["-r", str(REPO_DIR / "src"), f"{host}:{REPO_DIR}/"])
            run(SCP + [str(REPO_DIR / "requirements.txt"), f"{host}:{REPO_DIR}/"])
            site = remote_site_dir(host)
            if site:
                run(SSH + [host, f"echo '{REPO_DIR}' > {site}/mapd_b_project.pth"])
            else:
                print("  [WARN] site-packages non trovato: .pth non creato")

        if args.install:
            r = subprocess.run(
                SSH + [host,
                       f"{REMOTE_PY} -m pip install -r {REPO_DIR}/requirements.txt"],
                capture_output=True, text=True,
            )
            lines = (r.stdout or r.stderr).strip().splitlines()
            print("  pip:", lines[-1] if lines else f"rc={r.returncode}")

        out = run(SSH + [host, f"cd /home/ubuntu && {REMOTE_PY} -c '{IMPORT_CHECK}'"])
        if out and "OK" in out:
            print("  ", out.strip())
            n_ok += 1

    print(f"\n{n_ok}/{len(WORKER_IPS)} worker con import OK")
    return 0 if n_ok == len(WORKER_IPS) else 1


if __name__ == "__main__":
    sys.exit(main())
