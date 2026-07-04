# Datasets

This document describes the datasets used to study the performance of the **parallel k-means** algorithm as execution parameters vary (number of cores, number of workers, etc.).

The datasets are downloaded via `scikit-learn` utilities, which automatically handle downloading, local caching, and parsing into the appropriate formats (sparse `scipy.sparse.csr_matrix` matrices).

---

## 1. RCV1 — Reuters Corpus Volume I

**Description**

The Reuters Corpus Volume I (RCV1) is an archive of over 800,000 manually categorized newswire stories, made available by Reuters, Ltd. for research purposes.

`sklearn.datasets.fetch_rcv1` loads the **RCV1-v2** version (vectors, full sets, multilabel topics).

**Download**

```python
from sklearn.datasets import fetch_rcv1

rcv1 = fetch_rcv1()
```

**Characteristics**

| Property | Value |
|---|---|
| Samples | 804,414 |
| Features | 47,236 |
| Feature matrix format | `scipy.sparse.csr_matrix` |
| Non-zero values (feature matrix) | ~0.16% |
| Value type | Cosine-normalized, log TF-IDF |
| Categories (target) | 103 |
| Target matrix format | `scipy.sparse.csr_matrix` |
| Non-zero values (target matrix) | ~3.15% |
| Official split (LYRL2004) | first 23,149 samples = training set, remaining 781,265 = test set |
| Sample ID | integer, non-contiguous, range [2286, 810596] |

**Notes for benchmarking**

- A **highly sparse**, **high-dimensional** dataset (47k features): useful for evaluating how the speedup of parallel k-means behaves on sparse matrices, where the cost of distance computation differs from the dense case.
- The large number of samples (~800k) makes it suitable for scalability tests as the number of workers varies.
- Since the data is multilabel text data, only the feature matrix should be used for k-means clustering (which requires no target, or a single target); the labels can optionally serve as *ground truth* for external clustering validation metrics (e.g., purity, NMI).

**References**

- [scikit-learn documentation — RCV1 dataset](https://scikit-learn.org/stable/datasets/real_world.html#rcv1-dataset)

---

## 2. KDD Cup '99

**Description**

The KDD Cup '99 dataset was created by processing the tcpdump portions of the "1998 DARPA Intrusion Detection System (IDS) Evaluation" dataset, produced by MIT Lincoln Lab. The original goal was to produce a large training set for supervised learning algorithms; as a result, the dataset contains a very high proportion (80.1%) of abnormal (attack) traffic, which is unrealistic in the real world and unsuitable for unsupervised anomaly detection tasks.

To address this, scikit-learn exposes two derived variants:

- **SA**: contains all normal data plus a small proportion of abnormal data, resulting in an anomaly ratio of 1%.
- **SF**: contains only records where the `logged_in` attribute is positive (focusing on intrusion attacks), resulting in an attack ratio of 0.3%.

**Download**

```python
from sklearn.datasets import fetch_kddcup99

# SA variant
kdd_sa = fetch_kddcup99(subset="SA")

# SF variant
kdd_sf = fetch_kddcup99(subset="SF")
```

**Characteristics**

| Variant | Anomaly ratio | Notes |
|---|---|---|
| SA | ~1% | All normal data + a small share of anomalies |
| SF | ~0.3% | Only records with `logged_in` positive, focused on intrusions |

**Notes for benchmarking**

- A tabular dataset (unlike RCV1), useful for comparing the behavior of parallel k-means on **dense** vs. **sparse** data.
- The strong class imbalance between normal and anomalous records makes it interesting for observing how the number of clusters (k) and initialization affect convergence time and clustering quality, in addition to raw parallelization performance.

**References**

- [scikit-learn documentation — KDD Cup '99 dataset](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset)

---

## General notes on usage in the benchmarking pipeline

- Both datasets are automatically downloaded and cached by `scikit-learn` in the `~/scikit_learn_data/` folder (configurable via the `data_home` parameter or the `SCIKIT_LEARN_DATA` environment variable).
- To run benchmarks directly on the VM, run the fetch script once to populate the local cache, avoiding re-downloading the data on every test run.
- Given the non-negligible size of RCV1 (a sparse but high-cardinality matrix), check available disk space on the VM before downloading (`df -h`).
