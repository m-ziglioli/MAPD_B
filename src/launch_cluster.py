"""
launch_cluster.py
=================
Dask cluster management via SSH.

Designed to be imported from a notebook:

    from src.launch_cluster import launch_cluster, shutdown_cluster

    cluster, client = launch_cluster(n_workers=2)
    ...
    shutdown_cluster(cluster, client)

It can also be launched standalone from a terminal to keep the cluster
alive independently of a notebook:

    python launch_cluster.py -n 2
"""

import argparse
import time
from dask.distributed import Client, SSHCluster

# ==========================================
# CENTRALIZED CLUSTER CONFIGURATION
# ==========================================
HEAD_IP = "10.67.22.194"
WORKER_IPS = [
    "10.67.22.254",
    "10.67.22.34",
    "10.67.22.145",
    "10.67.22.121",
    "10.67.22.192",
    "10.67.22.18",
    "10.67.22.187",
    "10.67.22.48"
]
SCHEDULER_PORT = 8786
DASHBOARD_PORT = 8787
SSH_CONNECT_OPTIONS = {"known_hosts": None}
SCHEDULER_OPTIONS = {
    "port": SCHEDULER_PORT,
    "dashboard_address": f":{DASHBOARD_PORT}",
}

# Default timeout (seconds) for waiting for all requested workers to
# connect to the scheduler. Some nodes take longer because of heavy imports
# in the remote virtualenv or SSH latency.
DEFAULT_STARTUP_TIMEOUT = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Dask cluster via SSH with a configurable number of workers."
    )
    parser.add_argument(
        "-n", "--n-workers",
        type=int,
        default=len(WORKER_IPS),
        help=f"Number of workers to activate, between 1 and {len(WORKER_IPS)} "
             f"(default: all, {len(WORKER_IPS)}).",
    )
    parser.add_argument(
        "-t", "--startup-timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT,
        help=f"Maximum seconds to wait for all workers to connect "
             f"(default: {DEFAULT_STARTUP_TIMEOUT}).",
    )
    return parser.parse_args()


def _enable_pickle_by_value():
    """The project tasks reference functions of the src.* modules: by
    default cloudpickle serializes them BY REFERENCE (module+name), but
    scheduler and workers start via SSH with cwd=home and do not have the
    repo in sys.path -> 'ModuleNotFoundError: No module named src' when
    the graph is deserialized. Registering the modules BY-VALUE makes the
    code travel inside the graph itself and no remote process has to
    import the project.

    Idempotent; fails soft (warning) if something goes wrong, so it does
    not block the cluster startup."""
    import warnings

    try:
        import cloudpickle

        import src  # noqa: F401
        import src.benchmark  # noqa: F401
        import src.data_loader  # noqa: F401
        import src.kmeans_parallel  # noqa: F401
        import src.kmeans_serial  # noqa: F401

        for mod in (src, src.kmeans_parallel, src.data_loader,
                    src.benchmark, src.kmeans_serial):
            try:
                cloudpickle.register_pickle_by_value(mod)
            except Exception:
                pass  # already registered or non-serializable module: do not block
    except Exception as e:
        warnings.warn(
            f"pickle-by-value NOT enabled ({e}): if tasks use src.* "
            "functions, deserialization on the cluster may fail."
        )


def ensure_pickle_by_value():
    """Public alias for _enable_pickle_by_value, for use from notebooks or
    data_loader when the cluster was started standalone (Client(SCHEDULER_ADDRESS)
    path). Idempotent, soft-failing — safe to call repeatedly."""
    return _enable_pickle_by_value()


def launch_cluster(n_workers: int, block: bool = False, startup_timeout: float = DEFAULT_STARTUP_TIMEOUT):
    """Start the SSHCluster with the first n_workers nodes of WORKER_IPS
    and connect a Client.

    Returns (cluster, client), so it can be called repeatedly from a
    notebook (e.g. inside a loop varying n_workers), explicitly closing
    cluster/client between iterations with shutdown_cluster().

    If block=True (used only when the script is launched from a terminal
    with `python launch_cluster.py`), it keeps running until a CTRL+C
    arrives -- useful to keep the cluster alive standalone, but it must
    NOT be used from a notebook (it would block the cell forever).

    startup_timeout: maximum seconds to wait for all requested workers to
    connect (passed to Client.wait_for_workers). If not all workers are
    connected at the timeout, an explicit (non-silent) warning is printed
    and execution continues with the available ones.
    """

    if not (1 <= n_workers <= len(WORKER_IPS)):
        raise ValueError(
            f"n_workers must be between 1 and {len(WORKER_IPS)} "
            f"(available worker nodes), got: {n_workers}"
        )

    active_worker_ips = WORKER_IPS[:n_workers]
    # SSHCluster requires a flat list: the first element is the Scheduler,
    # the following ones are the Workers.
    ssh_hosts = [HEAD_IP] + active_worker_ips

    print(f"Initializing the SSH cluster with {n_workers} workers...")
    print(f"Selected workers: {active_worker_ips}")

    cluster = SSHCluster(
        hosts=ssh_hosts,
        connect_options=SSH_CONNECT_OPTIONS,
        scheduler_options=SCHEDULER_OPTIONS,
        remote_python="/home/ubuntu/pyvenv/bin/python3",
    )

    client = Client(cluster)

    # Explicit wait for the requested workers: before this check was
    # implemented, the startup_timeout parameter was accepted but never
    # used, and the notebook continued (sometimes mid-sweep) without all
    # the nodes. On timeout execution continues, with a visible warning.
    try:
        client.wait_for_workers(n_workers, timeout=startup_timeout)
        print("Cluster started and connection established successfully!\n")
    except TimeoutError:
        connected = len(client.scheduler_info().get("workers", {}))
        print(
            f"\n[WARNING] Timeout ({startup_timeout:.0f}s): connected "
            f"{connected}/{n_workers} workers. Continuing with the available ones."
        )

    # The client must be able to serialize the src.* functions by value:
    # see _enable_pickle_by_value for why (scheduler/worker via SSH do not
    # have the repo in sys.path).
    _enable_pickle_by_value()

    if block:
        print("\n[INFO] The cluster will stay active while this script is running.")
        print("Press CTRL+C in the terminal to shut down the cluster and exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterruption detected. Shutting down the cluster...")
            shutdown_cluster(cluster, client)

    return cluster, client


def shutdown_cluster(cluster, client):
    """Orderly close of client and cluster. Call between iterations of a
    notebook loop, before starting a cluster with a different n_workers,
    or at the end of a session."""
    try:
        client.close()
    finally:
        cluster.close()
    print("Cluster and client closed.")


if __name__ == "__main__":
    args = parse_args()
    launch_cluster(args.n_workers, block=True, startup_timeout=args.startup_timeout)
