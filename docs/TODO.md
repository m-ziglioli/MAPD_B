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
  tempi/costi con i CSV storici in results/
- [ ] POSTPONATO: riproduzione articolo (Tabelle 3/4/6, Fig 5.1/5.2) —
  piano completo e decisioni in docs/ANALYSIS_PLAN.md
