"""
data_loader.py
===============
Caricamento e preprocessing del dataset usato per i test di k-means||.

Uso tipico (da notebook):

    from data_loader import load_dataset

    X = load_dataset("kddcup99", subset="SA", percent10=True)
"""

import pandas as pd
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler


def load_kdd99(subset="SA", percent10=True):
    """
    Carica il dataset KDD Cup 99, rimuove completamente le feature
    categoriche, scala le feature numeriche rimanenti e ritorna un
    array NumPy pulito.

    Parameters
    ----------
    subset : str
        Sottoinsieme del dataset da caricare (vedi sklearn.datasets.fetch_kddcup99).
    percent10 : bool
        Se True, usa la versione ridotta (10%) del dataset.

    Returns
    -------
    X : np.ndarray
        Matrice delle feature numeriche standardizzate.
    """
    print("Loading dataset...")
    dataset = fetch_kddcup99(subset=subset, percent10=percent10, as_frame=True, download_if_missing=True)
    df = dataset.frame

    # forza la conversione a numerico, trasformando le stringhe categoriche in NaN
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # elimina le colonne completamente non numeriche (ora tutte NaN)
    df_numeric = df_numeric.dropna(axis=1, how="all")

    # elimina le righe rimaste con NaN
    df_numeric = df_numeric.dropna()

    # standardizza le feature numeriche
    scaler = StandardScaler()
    X_numpy = scaler.fit_transform(df_numeric.values)

    print(f"Dataset preprocessed. Shape: {X_numpy.shape}")
    return X_numpy


def load_dataset(name="kddcup99", **kwargs):
    """
    Punto di ingresso unico per il caricamento dei dataset. Attualmente
    supporta solo 'kddcup99'; è pensato per essere esteso in futuro con
    altri loader senza dover cambiare il codice del notebook.

    Parameters
    ----------
    name : str
        Nome del dataset da caricare.
    **kwargs :
        Argomenti passati al loader specifico (es. subset, percent10 per kddcup99).
    """
    loaders = {
        "kddcup99": load_kdd99,
    }

    if name not in loaders:
        raise ValueError(f"Dataset '{name}' non supportato. Disponibili: {list(loaders.keys())}")

    return loaders[name](**kwargs)
