"""
paper_experiments.py
====================
Driver per la riproduzione di Bahmani et al. (2012), "Scalable K-Means++"
(docs/1203.6402v1.pdf), secondo il piano congelato in docs/ANALYSIS_PLAN.md.

Artefatti coperti (baseline Partition esclusa ovunque):
- run_fig51  : Fig 5.1 — costo finale vs round r, KDD 10%, campionamento
               ESATTO di l punti per round (sampling="exact"),
               k in {17,33,65,129}, l/k in {1,2,4};
- run_fig52  : Fig 5.2 — costo finale vs round r (da 0) su GaussMixture
               sintetico, k=50, riferimento orizzontale k-means++;
               r=0 degrada a k centri uniformi (percorso random-baseline);
- run_table34: Tabelle 3 e 4 — KDD full, k in {500,1000}, r=5,
               l/k in {0.1,...,10} + baseline Random; costo seed/final
               (grezzo e scala x1e-10 in analisi) e tempi (Table 4).

Protocollo sperimentale (come dall'articolo e da ANALYSIS_PLAN.md):
- mediana su n_runs ripetizioni (default 11), seed della run i = seed + i;
- costi "final" dopo Lloyd's a convergenza (max_iter_fit/tol condivisi);
- ATTENZIONE ai policy: Tabelle 3/4 usano policy="fixed", r=5 — la regola
  automatica "l/k<=0.1 -> 15 round" NON si applica qui (l'articolo usa r=5
  anche per l/k=0.1); Fig 5.2 spazia esplicitamente su r=0..15 quindi usa
  sempre policy="fixed"; Fig 5.1 ha l/k>=1 quindi il valore di r non viene
  mai riscritto dalla regola automatica.
- Il caso noto "pool candidati < k" (puo' capitare con l/k piccoli ed r
  bassi: pool atteso ~ 1 + r*l) NON interrompe le sweep: la run viene
  registrata con costi NaN e flag failed=True (robustezza per i run
  notturni), come gia' fatto in benchmark.run_single_test.

I CSV finiscono in results/ (gitignored), le figure in figures/.
Tutte le funzioni accettano client=None: girano sullo scheduler locale
(utile per validazioni su griglie ridotte prima delle sessioni cluster).
"""

import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.benchmark import RESULTS_DIR
from src.data_loader import array_to_dask, make_gauss_mixture
from src.kmeans_parallel import kmeans_parallel, inertia_of_bag
from src.kmeans_serial import kmeans_serial


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _save_results(df, label):
    """Salva il DataFrame dei risultati in results/{label}_{timestamp}.csv."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"{label}_{timestamp}.csv")
    df.to_csv(path, index=False)
    print(f"\nRisultati salvati in: {path}")
    return path


def _load_csv(path):
    return pd.read_csv(path)


def _inertia_numpy(X, C):
    """Inertia lato client (array piccoli: GaussMixture 10k x 15)."""
    C = np.asarray(C, dtype=np.float64)
    m_sq = np.einsum("ij,ij->i", X, X)
    c_sq = np.einsum("ij,ij->i", C, C)
    d2 = m_sq[:, None] + c_sq[None, :] - 2.0 * (X @ C.T)
    np.maximum(d2, 0.0, out=d2)
    return float(d2[np.arange(X.shape[0]), d2.argmin(axis=1)].sum())


def _run_one_parallel(X_bag, k, l, r, run_seed, policy="auto",
                      sampling="bernoulli", max_iter_fit=100, tol=1e-4):
    """Singola run k-means|| (+ Lloyd's) con gestione benigna del caso noto
    'pool candidati < k': sklearn solleva ValueError nel reclustering e la
    sweep continua registrando la riga con failed=True e costi NaN.
    Qualsiasi ALTRO ValueError e' un vero errore e propaga.

    Ritorna un dict con: r_effective, cost_seed, cost_final,
    n_lloyd_iters, time_seed, time_fit, failed.
    """
    clf = kmeans_parallel(k=k, l=l if l is not None else 1, r=r)
    out = {"r_effective": None, "cost_seed": np.nan, "cost_final": np.nan,
           "n_lloyd_iters": np.nan, "time_seed": np.nan, "time_fit": np.nan,
           "failed": True}
    t0 = time.time()
    try:
        clf.compute_starting_centroids(X_bag, seed=run_seed,
                                       policy=policy, sampling=sampling)
    except ValueError as e:
        # stesso criterio di benchmark.run_single_test: solo l'errore noto
        # ("n_samples=X should be >= n_clusters=Y") e' benigno
        if "n_clusters" not in str(e):
            raise
        print(f" -> SEEDING FAILED (candidati < k, k={k}, l={l}, r={r}): {e}")
        return out
    out["failed"] = False
    out["r_effective"] = clf.n_rounds_
    out["time_seed"] = time.time() - t0
    clf.fit(X_bag, max_iter=max_iter_fit, tol=tol, track_convergence=True)
    out["time_fit"] = time.time() - t0 - out["time_seed"]
    out["cost_seed"] = inertia_of_bag(X_bag, clf.starting_centroids)
    out["cost_final"] = inertia_of_bag(X_bag, clf.final_centroids)
    out["n_lloyd_iters"] = len(clf.iter_times_)
    return out


# ---------------------------------------------------------------------------
# Fig 5.1 — KDD 10%, exact-l sampling
# ---------------------------------------------------------------------------

def run_fig51(client, X_bag, k_values=(17, 33, 65, 129),
              l_over_k_values=(1, 2, 4), r_values=tuple(range(1, 11)),
              n_runs=11, seed=42, max_iter_fit=100, tol=1e-4,
              num_partitions=None, label="fig51"):
    """Costo finale vs numero di round r, campionando ESATTAMENTE l punti
    per round (sampling="exact", protocollo della Fig 5.1). Mediana su
    n_runs a tempo di analisi/plot; qui una riga per singola run.

    num_partitions è solo informativo (etichetta nei risultati): la bag
    arriva già costruita dal chiamante.
    """
    rows = []
    for k in k_values:
        for l_over_k in l_over_k_values:
            l = max(1, round(l_over_k * k))
            for r in r_values:
                # l/k >= 1: la regola automatica non tocca r; fisso il policy
                # per esplicitare che il protocollo richiede QUESTO r
                for i in range(n_runs):
                    res = _run_one_parallel(X_bag, k, l, r, seed + i,
                                            policy="fixed", sampling="exact",
                                            max_iter_fit=max_iter_fit, tol=tol)
                    rows.append({
                        "artifact": "fig51",
                        "method": "kmeans||",
                        "k": k, "l": l, "r": r, "l_over_k": l_over_k,
                        "partitions": num_partitions,
                        "sampling": "exact",
                        "run": i, "seed": seed + i,
                        **res,
                    })
                print(f"[fig51] k={k}, l={l} (l/k={l_over_k}), r={r}: "
                      f"{n_runs} run completate")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fig 5.2 — GaussMixture, r da 0, k-means++ come riferimento
# ---------------------------------------------------------------------------

def run_fig52(client=None, R_values=(1, 10, 100),
              l_over_k_values=(0.1, 0.5, 1, 2, 10), r_values=tuple(range(0, 16)),
              k=50, n=10_000, d=15, n_runs=11, seed=42,
              max_iter_fit=100, tol=1e-4, n_partitions=8,
              include_kmpp_reference=True, label="fig52"):
    """Costo finale vs r su GaussMixture (Fig 5.2). Non richiede cluster:
    con n=10^4 punti gira comodamente anche in locale (client=None).

    r=0 -> k centri uniformi (policy="fixed", percorso random-baseline):
    l'asse parte dal livello del baseline Random, come nell'articolo.
    Il dataset è rigenerato con seed deterministico per ogni R
    (gm_seed = seed + R).
    """
    rows = []
    for R in R_values:
        gm_seed = seed + int(R)
        X, _, _ = make_gauss_mixture(n=n, k=k, d=d, R=R, seed=gm_seed)
        X_bag = array_to_dask(X, n_partitions)

        if include_kmpp_reference:
            print(f"[fig52] R={R}: riferimento k-means++, {n_runs} run")
            for i in range(n_runs):
                run_seed = seed + i
                srl = kmeans_serial(k=k, init="k-means++")
                srl.compute_starting_centroids(X, seed=run_seed)
                srl.fit(X, max_iter=max_iter_fit, tol=tol)
                rows.append({
                    "artifact": "fig52",
                    "method": "k-means++",
                    "R": R, "n": n, "d": d, "k": k,
                    "l": np.nan, "r": np.nan, "l_over_k": np.nan,
                    "sampling": "bernoulli",
                    "run": i, "seed": run_seed,
                    "r_effective": np.nan,
                    "cost_seed": np.nan,
                    "cost_final": _inertia_numpy(X, srl.final_centroids),
                    "n_lloyd_iters": srl.n_iter_,
                    "time_seed": np.nan, "time_fit": np.nan,
                    "failed": False,
                })

        for l_over_k in l_over_k_values:
            l = max(1, round(l_over_k * k))
            for r in r_values:
                for i in range(n_runs):
                    res = _run_one_parallel(X_bag, k, l, r, seed + i,
                                            policy="fixed",
                                            max_iter_fit=max_iter_fit, tol=tol)
                    rows.append({
                        "artifact": "fig52",
                        "method": "kmeans||",
                        "R": R, "n": n, "d": d, "k": k,
                        "l": l, "r": r, "l_over_k": l_over_k,
                        "sampling": "bernoulli",
                        "run": i, "seed": seed + i,
                        **res,
                    })
            print(f"[fig52] R={R}, l={l} (l/k={l_over_k}): r sweep completato")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tabelle 3 e 4 — KDD full
# ---------------------------------------------------------------------------

def run_table34(client, X_bag_full, k_values=(500, 1000),
                l_over_k_values=(0.1, 0.5, 1, 2, 10), r_fixed=5,
                n_runs=11, seed=42, max_iter_fit=100, tol=1e-4,
                num_partitions=64, include_random=True, label="table34"):
    """Tabelle 3 (costo) e 4 (tempi) su KDD full.

    Protocollo critico: policy="fixed", r=r_fixed (=5 come nell'articolo).
    Con policy="auto" la configurazione l/k=0.1 verrebbe forzata a 15 round,
    che NON è il protocollo della tabella.

    include_random=True aggiunge il baseline Random come percorso r=0
    (k centri uniformi, nessun reclustering): stesso budget di Lloyd's.
    Registra cost_seed E cost_final (la tabella riporta entrambi, grezzi;
    la scala x1e-10 si applica in analisi) più i tempi separati seeding/
    Lloyd's per la Table 4.
    """
    rows = []
    for k in k_values:
        for l_over_k in l_over_k_values:
            l = max(1, round(l_over_k * k))
            for i in range(n_runs):
                res = _run_one_parallel(X_bag_full, k, l, r_fixed, seed + i,
                                        policy="fixed", sampling="bernoulli",
                                        max_iter_fit=max_iter_fit, tol=tol)
                rows.append({
                    "artifact": "table34",
                    "method": "kmeans||",
                    "k": k, "l": l, "r": r_fixed, "l_over_k": l_over_k,
                    "partitions": num_partitions, "sampling": "bernoulli",
                    "run": i, "seed": seed + i,
                    **res,
                })
            print(f"[table34] k={k}, l={l} (l/k={l_over_k}), r={r_fixed}: "
                  f"{n_runs} run completate")
        if include_random:
            for i in range(n_runs):
                res = _run_one_parallel(X_bag_full, k, None, 0, seed + i,
                                        policy="fixed", sampling="bernoulli",
                                        max_iter_fit=max_iter_fit, tol=tol)
                rows.append({
                    "artifact": "table34",
                    "method": "random",
                    "k": k, "l": np.nan, "r": 0, "l_over_k": np.nan,
                    "partitions": num_partitions, "sampling": "bernoulli",
                    "run": i, "seed": seed + i,
                    **res,
                })
            print(f"[table34] k={k}: baseline Random, {n_runs} run completate")

    df = pd.DataFrame(rows)
    df["cost_seed_e10"] = df["cost_seed"] / 1e10
    df["cost_final_e10"] = df["cost_final"] / 1e10
    return df


# ---------------------------------------------------------------------------
# Plotting (log-y, mediane: convenzioni dell'articolo)
# ---------------------------------------------------------------------------

def _agg_median(df, by, metric="cost_final"):
    return df.groupby(by)[metric].median().sort_index()


def plot_fig51(results, output_dir="figures", dpi=150):
    """Fig 5.1: griglia di pannelli (uno per k), curve mediane per l/k,
    asse y logaritmico. `results` è un DataFrame o il path del CSV."""
    df = _load_csv(results) if isinstance(results, str) else results.copy()
    sub = df[df["artifact"] == "fig51"]
    k_values = sorted(sub["k"].unique())
    l_over_k_values = sorted(sub["l_over_k"].unique())

    ncols = 2
    nrows = int(np.ceil(len(k_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.6 * nrows),
                             squeeze=False)
    for ax, k in zip(axes.ravel(), k_values):
        for l_over_k in l_over_k_values:
            sel = sub[(sub["k"] == k) & (sub["l_over_k"] == l_over_k)]
            if sel.empty:
                continue
            med = _agg_median(sel, "r")
            ax.plot(med.index, med.values, marker="o", label=f"ℓ/k={l_over_k:g}")
        ax.set_yscale("log")
        ax.set_xlabel("number of rounds $r$")
        ax.set_ylabel("final cost (median)")
        ax.set_title(f"$k={k}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in axes.ravel()[len(k_values):]:
        ax.axis("off")
    fig.tight_layout()

    outpath = os.path.join(output_dir, "fig51_cost_vs_rounds.png")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print("Saved", outpath)
    return outpath


def plot_fig52(results, output_dir="figures", dpi=150):
    """Fig 5.2: un pannello per R, curve mediane per l/k, linea orizzontale
    k-means++ (mediana delle run di riferimento), asse x da r=0, y log."""
    df = _load_csv(results) if isinstance(results, str) else results.copy()
    sub = df[df["artifact"] == "fig52"]
    par = sub[sub["method"] == "kmeans||"]
    R_values = sorted(par["R"].unique())
    l_over_k_values = sorted(par["l_over_k"].dropna().unique())

    ncols = len(R_values)
    fig, axes = plt.subplots(1, ncols, figsize=(5.6 * ncols, 4.6), squeeze=False)
    for ax, R in zip(axes.ravel(), R_values):
        for l_over_k in l_over_k_values:
            sel = par[(par["R"] == R) & (par["l_over_k"] == l_over_k)]
            if sel.empty:
                continue
            med = _agg_median(sel, "r")
            ax.plot(med.index, med.values, marker="o", label=f"ℓ/k={l_over_k:g}")
        ref = sub[(sub["R"] == R) & (sub["method"] == "k-means++")]["cost_final"]
        if not ref.empty:
            ax.axhline(ref.median(), linestyle="--", color="black",
                       label="k-means++")
        ax.set_yscale("log")
        ax.set_xlabel("number of rounds $r$")
        ax.set_ylabel("final cost (median)")
        ax.set_title(f"$R={R:g}$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()

    outpath = os.path.join(output_dir, "fig52_cost_vs_rounds.png")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)
    print("Saved", outpath)
    return outpath


def table34_cost_table(results, stat="median"):
    """Layout della Table 3: righe = variante algoritmo, colonne = blocchi
    per k con seed/final affiancati, costi in scala x1e-10 (come nel paper).
    Riusa il pivot di comparison_analysis.format_paper_table."""
    from src.comparison_analysis import format_paper_table

    df = _load_csv(results) if isinstance(results, str) else results.copy()
    piv = format_paper_table(df, stat=stat)
    return piv / 1e10


def table34_time_table(results, stat="median"):
    """Layout della Table 4: tempi medi (init + Lloyd) per metodo e k."""
    df = _load_csv(results) if isinstance(results, str) else results.copy()
    df = df.copy()
    df["label"] = df.apply(
        lambda r: r["method"] if r["method"] != "kmeans||"
        else f"kmeans|| (l/k={r['l_over_k']:g}, r={int(r['r'])})", axis=1)
    grouped = df.groupby(["label", "k"])[["time_seed", "time_fit"]].agg(stat)
    grouped["time_total"] = grouped["time_seed"] + grouped["time_fit"]
    return grouped
