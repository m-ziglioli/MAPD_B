"""
sync_workers.py

The workers are VMs separate from the head and do NOT have the repo:
without this step, tasks serialized by-reference cannot resolve
``import src`` on the worker (and the import chain of src.kmeans_parallel
requires sklearn, which may be missing on the workers -> process crash ->
KilledWorker). The .pth file puts the repo on the remote pyvenv's sys.path
regardless of the cwd.

RE-RUN after every ``git pull`` of new code (the deploy copies the head's
current sources).

Usage (on the HEAD VM, from the checkout, project shell):
    python scripts/sync_workers.py              # deploy + import verification
    python scripts/sync_workers.py --install    # + pip install requirements
    python scripts/sync_workers.py --check      # import verification only
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.launch_cluster import WORKER_IPS  # centralized IP list (AGENTS.md)

REPO_DIR = Path("/home/ubuntu/Project/libero_development")
REMOTE_PY = "/home/ubuntu/pyvenv/bin/python3"

SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]
SCP = ["scp", "-q", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no"]

IMPORT_CHECK = (
    "import sklearn, src.kmeans_parallel, src.data_loader, src.benchmark; "
    "print('OK', sklearn.__version__)"
)


def run(cmd):
    """Run a remote command; return the output or print the error."""
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [FAIL] {' '.join(cmd[:4])}...")
        print("  " + (r.stderr or r.stdout).strip()[-400:])
        return None
    return (r.stdout or r.stderr).strip()


def remote_site_dir(host):
    """site-packages directory of the remote pyvenv (robust to the python
    version, instead of hardcoding .../python3.10/)."""
    out = run(SSH + [host, f"{REMOTE_PY} -c 'import site; print(site.getsitepackages()[0])'"])
    return out.strip().splitlines()[-1] if out else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--install", action="store_true",
                    help="pip install -r requirements.txt on every worker")
    ap.add_argument("--check", action="store_true",
                    help="import verification only (no deploy)")
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
                print("  [WARN] site-packages not found: .pth not created")

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

    print(f"\n{n_ok}/{len(WORKER_IPS)} workers with import OK")
    return 0 if n_ok == len(WORKER_IPS) else 1


if __name__ == "__main__":
    sys.exit(main())
