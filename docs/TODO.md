- [x] salvare costo ad ogni iterazione per vedere come converge k-means (fit
  track_convergence, CHANGES 2026-07-19)
- [x] numero di centroidi dopo le iterazioni di k-means|| (track_centroids,
  CHANGES 2026-07-19)
- [x] run_benchmark che prende una bag come input (CHANGES 2026-07-22)
- [x] confronto con k-means++ (src/kmeans_comparison.py + comparison.ipynb)
- [x] sistemare convergenza con soglia (stop stretta sulle etichette, fuse
  nell'assegnazione)
- [x] criticità calcoli punto-per-punto con bag.map (engine vettorizzato
  per partizione, CHANGES 2026-08-23)
- [x] ricontrollare bene il codice (review round 2, CHANGES 2026-08-23:
  media reale sui seed, resolve_rounds, igiene RNG, stato distribuito)
- [ ] run variando numero di worker e numero di partizioni in simultanea
- [ ] confronto con k-means standard nel dataset 100%
- [ ] questione unmanaged memory
- [ ] VALIDAZIONE SU CLUSTER del refactor stato-distribuito (2026-08-23):
  sweep piccolo k=500/1000, 4 combinazioni del notebook, confronto
  tempi/costi con i CSV storici in results/ — runbook: agents/vm_checklist.md
- [x] prerequisiti riproduzione articolo implementati in locale
  (CHANGES 2026-08-23 fase A: r=0 random baseline, sampling="exact",
  GaussMixture, src/paper_experiments.py, notebook flag-gated)
- [ ] ESECUZIONE articolo su cluster (ANALYSIS_PLAN sessioni A/B): Fig 5.1,
  Tables 3/4 (+sanity check), Fig 5.2 locale; poi write-up obtained-vs-paper
- [x] FASE D (shard Parquet per worker, CHANGES 2026-08-30): niente repliche
  del dataset, ogni shard sta su un worker; resta aperto il repartitioning
  nel grafo senza materializzazione client (run_benchmark ancora raccoglie
  X_arr una volta per ri-scatterare) + validazione costi identici su KDD 10%
- [ ] POSTPONATO→IN CORSO: voci sopra (il piano resta in docs/ANALYSIS_PLAN.md)
