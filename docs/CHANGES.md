# Changelog

## 2026-07-19 — Todo #1: tracking convergenza k-means (fit)

- `kmeans_parallel.py`: aggiunto parametro `track_convergence=False` a `fit()`.
  Se attivo, popola `self.cost_history_` (inertia ad ogni iterazione di Lloyd's)
  e `self.iter_times_` (tempo per iterazione), riusando lo stesso `foldby` già
  presente (nessun giro extra sui dati). A flag spento il comportamento è
  identico a prima, nessun overhead.
- `benchmark.py`: `run_single_test()` inoltra lo stesso flag a `fit()` e, se
  attivo, aggiunge `cost_history`/`iter_times` al dict di ritorno.
  `run_benchmark()` (griglia multi-run su CSV) non è stato toccato — resta
  per il todo #3.
- `analysis.ipynb`, sezione 5 (Single run): la chiamata a `run_single_test`
  ora passa `track_convergence=True`; aggiunta una cella subito dopo che
  plotta `result["cost_history"]` (costo per iterazione di Lloyd's fit) con
  matplotlib, per ispezionare la convergenza della run.

## 2026-07-19 — Todo #2: tracking numero di centroidi durante k-means||

- `kmeans_parallel.py`: aggiunto parametro `track_centroids=False` a
  `compute_starting_centroids()`. Se attivo, popola
  `self.n_centroids_history_` con il numero cumulativo di candidati
  centroidi dopo ogni round eseguito (nessun costo aggiuntivo, solo
  `len()` su una lista già mantenuta). A flag spento comportamento
  identico a prima.
- `benchmark.py`: `run_single_test()` inoltra lo stesso flag e, se attivo,
  aggiunge `n_centroids_history` al dict di ritorno.
- `analysis.ipynb`, sezione 5: la chiamata a `run_single_test` ora passa
  anche `track_centroids=True`; aggiunta una cella dopo il grafico di
  convergenza con una tabella pandas (`round`, `n_centroids`) per
  ispezionare la crescita del pool di candidati round per round.
- Nota emersa testando la modifica: il seed non è riproducibile in modo
  affidabile nemmeno in locale (scheduler threaded di Dask — le chiamate
  `np.random.uniform()` nel filtro Bernoulli avvengono da thread diversi
  in ordine non deterministico). Bug pre-esistente, non introdotto da
  questa modifica — corrisponde al punto già segnalato nella review
  iniziale ("riproducibilità del seed rotta in ambiente distribuito"), da
  affrontare a parte.

## 2026-07-22 — Todo #3: run_benchmark prende una bag come input

- `benchmark.py`: `run_benchmark()` ora accetta `X_bag` (Dask Bag, come
  `run_single_test`) invece di `X` (`np.ndarray`). La firma era rimasta
  disallineata da quando `load_dataset()` (Parquet + Dask DataFrame) ha
  sostituito il vecchio caricamento in un array numpy pieno lato client:
  `X` nel notebook non esisteva più (`NameError`), e la chiamata interna a
  `run_single_test(client, X, k=k, ...)` era diventata incompatibile con la
  firma di `run_single_test` (riordinata per i todo #1/#2), causando
  `TypeError: multiple values for argument 'k'`. Corretta anche questa
  chiamata (solo keyword args).
- Dentro `run_benchmark()`, `X_bag` viene materializzato in un array numpy
  lato client **una sola volta** (non più ad ogni cambio di
  `num_partitions`), poi ridistribuito ai worker con `client.scatter()`
  (`_build_bag`, invariata) per ogni combinazione — lo stesso meccanismo
  già usato dalla versione storica del 14/7 che funzionava bene. Scartate
  due alternative più dirette, entrambe testate su cluster reale e
  rivelatesi problematiche:
  - `Bag.repartition()` ad ogni cambio di partizioni: se applicato su
    `X_bag` non persistita rilegge/ripreprocessa da zero l'intera pipeline
    Parquet ad ogni combinazione (causa di un `FutureCancelledError` /
    scheduler-connection-lost, riprodotto 3 volte); se applicato su una
    bag persistita si è rivelato comunque un collo di bottiglia poco
    parallelizzato (task concentrati, tempi peggiori, visto in dashboard).
  - `X_bag.compute()` diretto: lo step di "finalize" che unisce le
    partizioni può finire schedulato su un solo worker, che arriva quasi
    al suo limite di memoria mentre gli altri restano vuoti (visto in
    dashboard — rischio OOM concreto, specialmente al 100% del dataset).
  - Anche il gather "via `to_delayed`" naive ha un'insidia: `data_loader.py`
    crea un array numpy per ogni singola riga (~500k oggetti per il 10%
    del dataset), quindi un `client.gather()` diretto sulle partizioni
    serializza mezzo milione di oggetti minuscoli invece di poche decine
    di blocchi, saturando la connessione client-scheduler
    (`CommClosedError: Stream is closed`). Fix: `vstack` di ogni
    partizione in un unico array 2D **dentro il grafo Dask** (calcolato
    sul worker) prima del gather.
  - Aggiunto `client.cancel(old_bag)` ad ogni cambio di `num_partitions`,
    per non accumulare in memoria sui worker le bag scatterate delle
    combinazioni precedenti (worker con poca RAM).
- `kmeans_parallel.py`: nello step 8 di `compute_starting_centroids()`
  (riduzione finale a k centroidi), `KMeans(n_clusters=self.k)` →
  `KMeans(n_clusters=self.k, n_init=1)`. Il paper originale (Bahmani et
  al., *Scalable K-Means++*, VLDB 2012) usa una singola inizializzazione
  k-means++ per il reclustering, non i 10 restart di default di sklearn;
  riduce anche il tempo di blocco locale (non distribuito) di questo step,
  che con k grandi poteva far scadere l'heartbeat client-scheduler.
- `analysis.ipynb`, sezione 6: la cella del benchmark ora passa `X_bag`
  invece della `X` inesistente; `combinations` ridotta da 25 punti quasi
  ridondanti (partitions 65→113, quasi tutti sullo stesso plateau di
  tempo) a 4 combinazioni rappresentative dei regimi rilevanti rispetto ai
  64 thread totali del cluster (8 worker × 8 thread): sotto-partizionato
  (32), bilanciato (64), sbilanciato di poco (65), sovra-partizionato
  (128).
- Nota operativa emersa debuggando: crash ripetuti senza mai un riavvio
  pulito del cluster (solo riconnessioni) lasciano su scheduler/worker
  task e futures orfani non cancellati, che rallentano silenziosamente le
  run successive anche a parità di codice/parametri (osservato: stessa
  configurazione ~7× più lenta dopo diversi crash rispetto alla prima run
  della giornata). Per numeri di benchmark affidabili da confrontare,
  usare `client.restart()` o un riavvio pulito del cluster prima del run
  definitivo — collegato al todo #8 ("questione unmanaged memory").
