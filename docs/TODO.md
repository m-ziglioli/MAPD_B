- salvare costo ad ogni iterazione per vedere come converge k-means --> vogliamo capire come salvare i tempi di k-means/come gestirne la cvg
- numero di centroidi dopo le iterazioni di k-means||
- c'é da modificare anche run_benchmark che prenda una bag come input, esattamente come la funzione run_single_test
- confronto con k-means++ CAPIRE COME FARLO
- run variando numero di worker e numero di partizioni in simultanea
- sistemare convergenza con soglia
- confronto con k-means standard nel dataset 100%
- questione unmanaged memory
- ricontrollare bene code
- sistemare la criticità dei calcoli punto-per-punto con bag.map(lambda ...) in
  fit()/compute_starting_centroids (vettorizzare per partizione con
  map_partitions, es. sklearn.metrics.pairwise.euclidean_distances)


