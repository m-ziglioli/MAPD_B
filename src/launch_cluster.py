"""
launch_cluster.py
==================
Gestione del cluster Dask via SSH.

Pensato per essere importato da notebook:

    from src.launch_cluster import launch_cluster, shutdown_cluster

    cluster, client = launch_cluster(n_workers=2)
    ...
    shutdown_cluster(cluster, client)

Può anche essere lanciato standalone da terminale per tenere il cluster
vivo indipendentemente da un notebook:

    python launch_cluster.py -n 2
"""

import argparse
import time
from dask.distributed import Client, SSHCluster

# ==========================================
# CONFIGURAZIONE CENTRALIZZATA DEL CLUSTER
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

# Timeout di default (in secondi) per l'attesa che tutti i worker richiesti
# si connettano allo scheduler. Alcuni nodi possono impiegare più tempo per
# via di import pesanti nell'ambiente virtuale remoto o latenza SSH.
DEFAULT_STARTUP_TIMEOUT = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avvia il cluster Dask via SSH con un numero di worker configurabile."
    )
    parser.add_argument(
        "-n", "--n-workers",
        type=int,
        default=len(WORKER_IPS),
        help=f"Numero di worker da attivare, tra 1 e {len(WORKER_IPS)} "
             f"(default: tutti, {len(WORKER_IPS)}).",
    )
    parser.add_argument(
        "-t", "--startup-timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT,
        help=f"Secondi massimi di attesa per la connessione di tutti i worker "
             f"(default: {DEFAULT_STARTUP_TIMEOUT}).",
    )
    return parser.parse_args()


def launch_cluster(n_workers: int, block: bool = False, startup_timeout: float = DEFAULT_STARTUP_TIMEOUT):
    """Avvia lo SSHCluster con i primi n_workers nodi da WORKER_IPS e si
    connette con un Client.

    Ritorna (cluster, client), così può essere chiamata ripetutamente da
    un notebook (es. dentro un ciclo che varia n_workers), chiudendo
    esplicitamente cluster/client tra un'iterazione e l'altra con
    shutdown_cluster().

    Se block=True (usato solo quando lo script è lanciato da terminale con
    `python launch_cluster.py`), resta in esecuzione finché non arriva un
    CTRL+C -- utile per tenere il cluster vivo standalone, ma NON va usato
    da notebook (bloccherebbe la cella per sempre).

    startup_timeout: secondi massimi di attesa per la connessione di tutti
    i worker richiesti (passati a Client.wait_for_workers). Se allo scadere
    del timeout non tutti sono connessi, viene stampato un avviso esplicito
    (non silenzioso) e si procede comunque con quelli disponibili.
    """

    if not (1 <= n_workers <= len(WORKER_IPS)):
        raise ValueError(
            f"n_workers deve essere tra 1 e {len(WORKER_IPS)} "
            f"(nodi worker disponibili), ricevuto: {n_workers}"
        )

    active_worker_ips = WORKER_IPS[:n_workers]
    # SSHCluster richiede una lista piatta: il primo elemento è lo Scheduler,
    # i successivi sono i Worker.
    ssh_hosts = [HEAD_IP] + active_worker_ips

    print(f"Inizializzazione del cluster SSH con {n_workers} worker...")
    print(f"Worker selezionati: {active_worker_ips}")

    cluster = SSHCluster(
        hosts=ssh_hosts,
        connect_options=SSH_CONNECT_OPTIONS,
        scheduler_options=SCHEDULER_OPTIONS,
        remote_python="/home/ubuntu/pyvenv/bin/python3",
    )

    client = Client(cluster)

    # Attesa esplicita dei worker richiesti: prima dell'implementazione di
    # questo controllo il parametro startup_timeout era accettato ma mai
    # usato, e il notebook proseguiva (a volte a metà sweep) senza tutti
    # i nodi. Al timeout si prosegue comunque, ma con avviso visibile.
    try:
        client.wait_for_workers(n_workers, timeout=startup_timeout)
        print("Cluster avviato e connessione stabilita con successo!\n")
    except TimeoutError:
        connected = len(client.scheduler_info().get("workers", {}))
        print(
            f"\n[AVVISO] Timeout ({startup_timeout:.0f}s): connessi "
            f"{connected}/{n_workers} worker. Si procede con quelli disponibili."
        )

    if block:
        print("\n[INFO] Il cluster rimarrà attivo finché questo script è in esecuzione.")
        print("Premi CTRL+C nel terminale per spegnere il cluster e terminare lo script.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterruzione rilevata. Chiusura del cluster in corso...")
            shutdown_cluster(cluster, client)

    return cluster, client


def shutdown_cluster(cluster, client):
    """Chiude ordinatamente client e cluster. Da chiamare tra un'iterazione
    e l'altra di un ciclo notebook, prima di avviare un cluster con un
    n_workers diverso, oppure a fine lavoro."""
    try:
        client.close()
    finally:
        cluster.close()
    print("Cluster e client chiusi.")


if __name__ == "__main__":
    args = parse_args()
    launch_cluster(args.n_workers, block=True, startup_timeout=args.startup_timeout)