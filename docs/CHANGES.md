# Changelog

## 2026-08-24 — Incidente cluster: worker senza freeze → KilledWorker. Causa, fix, prevenzione

### Catena causale completa

1. I worker erano stati **riprovisionati senza i pacchetti del freeze**
   (sklearn assente dal pyvenv dei worker; dask/numpy/pandas/pyarrow
   presenti — per questo il caricamento dati funzionava).
2. I task del motore referenziano funzioni di `src.kmeans_parallel` e la
   serializzazione di default è **by-reference**: per eseguire il task il
   worker deve importare il MODULO, e l'import di `src.kmeans_parallel`
   richiede sklearn (linea 39) → `ModuleNotFoundError` sul worker.
3. In distributed 2026.6.0 quell'errore durante l'esecuzione fa **crashare
   il processo worker** (non marca solo il task) → nanny riavvia, dask
   riprova su altri worker → `KilledWorker`, e churn della popolazione
   ("already forgotten", probe che vedeva 4 worker su 8).
4. Il `.pth` sul HEAD aveva sistemato solo lo SCHEDULER (che sta sul head):
   i worker sono VM separate e non avevano né la repo né sklearn.

Perché prima non si vedeva: fino al refactor i task erano lambda definite
nei notebook (by-value), mai import di `src` sul worker; il primo grafo che
referenzia `src` è arrivato con `_persist_matrices` (2026-08-23).

### Fix (due scudi indipendenti)

- **Scudo 1 — codice che viaggia col grafo** (commit `964359f`):
  `launch_cluster()` registra by-value i moduli `src.*` subito dopo la
  connessione: i task non richiedono più import di moduli sul worker.
  Verificato localmente: funzioni task unpicklano in un processo senza
  `src` importabile.
- **Scudo 2 — worker con repo e freeze allineati** (nuovo
  `scripts/sync_workers.py`): deploy di `src/` + `requirements.txt` su tutti
  gli 8 worker + file `.pth` nel site-packages del pyvenv remoto (path
  derivato a runtime, non hardcodato su python3.10) + opzione `--install`
  per allineare i pacchetti. Da rieseguire dopo ogni `git pull` di codice
  nuovo.

### Prevenzione

- Nuovo `scripts/check_cluster_env.py`: probe one-shot (via `client.run`,
  senza `get_worker`) delle versioni freeze su TUTTI i worker con verdetto
  di allineamento. Da eseguire subito dopo ogni avvio cluster, o standalone
  con `--address`.
- Regole operative documentate: non riavviare MAI il kernel con un cluster
  vivo (le sessioni SSH muoiono e portano giù i processi remoti); pulire
  scheduler orfani sulla porta 8786 prima di `launch_cluster`.

### Stato

B1 (validazione storica) in corso sul VM head una volta che
`check_cluster_env` è verde su tutti gli 8 worker.

## 2026-08-23 — Fase A (prerequisiti articolo): random baseline r=0, sampling esatto, driver paper

Prerequisiti del piano congelato `docs/ANALYSIS_PLAN.md`, implementati e
validati in locale (commit `78b0b21`, `9ba80f1`, `b3cd268`). Esecuzione su
cluster rimandata alle sessioni B/C.

### `kmeans_parallel`
- **r=0 → random baseline** (`policy="fixed"`): cortocircuito PRIMA di
  toccare i dati — k righe uniformi di X (ramo `ss_init`), niente round né
  reclustering, `n_rounds_=0`. È il punto r=0 della Fig 5.2 e il baseline
  Random della Table 3. `resolve_rounds` rifiuta ora r negativi.
- **`sampling="exact"`** (default resta `"bernoulli"` = Algorithm 2):
  ESATTAMENTE l punti per round senza reimmissione con probabilità ∝ d²
  (protocollo della sola Fig 5.1). Schema Efraimidis-Spirakis distribuito:
  ogni partizione restituisce solo il proprio top-l (chiave u^(1/d²),
  indice locale); il client fa il merge globale e una task per partizione
  recupera le sole righe scelte. Punti a d²=0 mai campionabili. Stesso
  schema RNG per (partizione, round) ⇒ determinismo preservato.
- Nuovo attributo `sampling_`.

### `data_loader`
- `make_gauss_mixture(n, k, d, R, seed)` (dataset Fig 5.2) e
  `array_to_bag(X, n_partitions)` (numpy → Bag di righe, formato identico
  a load_dataset).

### `paper_experiments` (nuovo modulo) + notebook
- `run_fig51` (KDD 10%, exact-ℓ), `run_fig52` (GaussMixture, r da 0,
  riferimento k-means++, girabile in locale), `run_table34` (KDD full).
- **Protocollo tabelle**: `policy="fixed", r=5` — la regola automatica
  l/k≤0.1→15 NON è quella delle tabelle dell'articolo (che usa r=5 anche a
  ℓ/k=0.1). Il baseline Random è il percorso r=0.
- **Robustezza notturna**: tutte le run passano da `_run_one_parallel`,
  che tratta il caso noto "pool candidati < k" come benigno (riga con
  failed=True e costi NaN, la sweep continua). Su griglie sintetiche
  compatte l/k=0.1 cade SISTEMATICAMENTE sotto k (pool ~0.5k+1); su KDD
  reale la coda pesata di d² fa campionare ≫ l al primo round (per questo
  la tabella del paper esiste) — verificare col sanity check previsto nel
  notebook prima della corsa completa.
- Plot con convenzioni articolo (mediane, log-y): `plot_fig51`,
  `plot_fig52`; tabelle `table34_cost_table` (×10⁻¹⁰, riusa
  `format_paper_table`) e `table34_time_table`.
- `notebooks/paper_reproduction.ipynb`: sezioni flag-gated (tiny/full
  Fig5.2 locale; sanity-run poi sweep piena su cluster), path VM identici
  agli altri notebook, scheletro obtained-vs-paper.

### `benchmark`
- `run_benchmark(..., X_arr=None)`: salta il gather-from-bag quando il
  client ha già l'array; chiamate storiche invariate (verificate
  bit-identiche sui costi).
- `run_worker_sweep(X_arr, workers_list, combinations_fn, ...)`: riavvia il
  cluster SSH per ogni conteggio di worker ed esegue la griglia — executor
  del todo "worker e partizioni in simultanea"; un array client-side solo
  per tutta la sweep, CSV per conteggio con suffisso `_w{n}` e colonna
  `workers_cfg`.

### Verifica e note operative
- smoke_test esteso (r=0 righe-esatte+determinismo+fit; exact +l/round;
  entrambi deterministici) — verde, golden regression intatta.
- Driver validati end-to-end su griglie ridotte locali (fig51/fig52/
  table34 incl. boundary l/k=0.1).
- **Windows/LocalCluster**: ogni script che crea un LocalCluster DEVE stare
  sotto `if __name__ == "__main__":` (spawn = re-import del modulo);
  senza guardia si va in ricorsione di processi e hang silenziosi. I
  notebook non sono interessati.
- Harness locale: `agents/vm_checklist.md` (read-only) per la fase B0 sul
  head VM (repo/env/ssh-workers/RAM/dataset/porte).

## 2026-08-23 — Review round 2: media reale sui seed, policy dei round esplicita, stato distribuito

Secondo giro di review critica (post-refactor vettorizzato). Tre difetti
strutturali e tre maggiori emersi e corretti. Commit `835dfb9` (fix+perf)
e `cf17263` (chore).

### Correttezza / metodologia
- **`run_benchmark`: media falsa corretta.** Ogni ripetizione chiamava
  `run_single_test` con lo STESSO seed ⇒ 10 run bit-identici, media e std
  privi di significato statistico (errorbar ~0 nei plot). Ora la
  ripetizione i-esima usa `seed + i`, come già faceva `run_comparison`;
  il CSV registra la colonna `seed`.
- **Numero di round risolto in UN punto (`resolve_rounds`).** La regola
  del paper "l/k <= 0.1 -> 15 round" era duplicata (engine + driver) e
  silenziamente scartava un `r` esplicito. Ora: `policy="auto"` (default,
  regola del paper, anche con r fornito — comportamento storico preservato)
  o `policy="fixed"` (escape hatch esplicito); il round effettivo è
  registrato come `clf.n_rounds_` e colonna `r_effective` nei CSV di
  entrambi i driver. Il costruttore accetta ora `r=None` come da docstring.
- **Igiene RNG.** Due `default_rng(seed_seq)` sulla STESSA SeedSequence
  duplicavano lo stream: il `random_state` del reclustering condivideva i
  bit dell'indice del centroide iniziale. Ora `SeedSequence(seed).spawn(3)`
  separa centroide iniziale / round di campionamento / reclustering; i
  seed per (partizione, round) sono pre-derivati con `spawn` (eliminato
  l'hack `entropy % 2**32`, che collideva per seed a distanza 2^32).
  NOTA: cambia il campionamento a parità di seed rispetto al giro prima
  (intenzionale; i golden di fit partono da centroidi fissi e non sono
  toccati).
- **`run_single_test`**: catturato SOIL il `ValueError` noto del
  reclustering con candidati < k (verifica sul messaggio); ogni altro
  errore propaga invece di comparire come riga "FAILED".

### Scalabilità (stato fuori dal client)
- **Partizioni materializzate una sola volta** (`_persist_matrices`):
  seeding e fit riusano gli stessi futures; niente ri-vstack delle
  partizioni del bag a ogni round/iterazione (~2r+2 passaggi evitati).
- **Stato k-means|| (m,2) sempre lato worker**: catena di Delayed tra i
  round; `_update_state` fonde lo scalare di costo del round. Al client
  arrivano solo scalari e i k-vettori dei pesi finali. Prima: la matrice
  di stato completa viaggiava client↔cluster ad ogni round (decine di MB
  su KDD full).
- **Etichette di Lloyd's nel grafo**: `fit()` calcola solo le riduzioni
  `t[:4]` (somme k×d, conteggi, costo, cambiamenti); le etichette `t[4]`
  restano nel grafo come input dell'iterazione successiva. Prima: vettori
  (m,) int32 su e giù dal client ad ogni iterazione.
- Guardie d'uso (`RuntimeError` chiaro se `fit`/`classify`/`inertia`
  chiamate fuori ordine) e deduplicazione inertia (`inertia_of_bag`
  condivisa da classe e benchmark).

### Altro (chore)
- `launch_cluster`: `startup_timeout` era accettato ma mai usato; ora
  passato a `Client.wait_for_workers`, con avviso esplicito se si procede
  con meno worker. Rimosso `POLL_INTERVAL` morto.
- `data_loader`: download saltato se il .gz esiste già
  (`force_download=True` per rifarlo); meta dtype come stringhe
  `'float64'` invece del tipo python.
- `comparison_analysis.plot_cost_vs_rounds`: aggregazione `stat="median"`
  di default (protocollo del paper; `"mean"` disponibile).
- `kmeans_comparison`: nota su differenza di semantica di `tol` fra
  sklearn e il nostro Lloyd's (irrilevante grazie alla stop stretta sulle
  etichette, documentata per onestà).
- Harness locale: smoke_test esteso (variazione seed, policy dei round,
  guardie d'uso); rimossa la nota stale sulla non-riproducibilità del seed.

### Verifica
- smoke_test verde: regression golden bit-identica, determinismo intatto,
  costo finale invariato (672313.9799 su bench_timing locale).
- Driver verificati end-to-end su LocalCluster (seed variati, colonne
  nuove, guardie).
- **Pendente validazione su cluster SSH**: sweep piccolo (k=500/1000,
  4 combinazioni del notebook) per confermare il guadagno temporale del
  nuovo schema di stato su dati/partizioni reali.

## 2026-08-23 — Determinismo del seeding + engine vettorizzato per-partizione

Il bug di non-riproducibilità del seed (vedi 2026-07-19) è **risolto**, e
l'intero engine di `kmeans_parallel` è stato riscritto attorno a calcoli
vettorizzati per partizione. Riepilogo dei commit:

### `bc66c19` — determinismo + correzioni di correttezza
- Campionamento Bernoulli dei round senza più RNG globale: uniformi
  pre-estratte da un `Generator` locale costruito dal `seed` (flusso
  deterministico, indipendente dall'ordine dei thread dello scheduler).
- Centroide iniziale estratto via indice uniforme dal RNG locale
  (niente più `dask.bag.random`, che usava `random` globale).
- `KMeans` del reclustering pesato (Step 8) ora riceve un `random_state`
  derivato dalla stessa `SeedSequence`: era la causa residua di
  non-riproducibilità a parità di candidati (bug pre-esistente).
- `fit()`: warning sui cluster vuoti e nuovo attributo `n_iter_`
  (parità con `kmeans_serial`); rimossa la side effect globale
  `np.random.seed(...)`.
- Early-return `psi == 0`: `starting_centroids` ora ha sempre k righe.

### `5f3371e` — refactor vettorizzato (motore su matrici per partizione)
- La Bag viene scomposta in partizioni ritardate (`to_delayed`), ognuna
  impilata in una matrice `(m, d)`: ogni passaggio è un task NumPy per
  partizione. Eliminate le lambda punto-per-punto, lo shuffle di righe
  del foldby e il passaggio extra di controllo convergenza (ora fuso
  nell'assegnazione via conteggio delle etichette cambiate).
- Distanze con espansione quadratica `||x||² + ||c||² − 2x·c`: una
  matmul BLAS per passaggio. Nota: durante lo sviluppo un einsum con
  asse sbagliato (`->j` invece di `->i`) sulle norme dei centroidi ha
  causato un ValueError — corretto con regression test.
- Seeding: stato come array `(m, 2)` per partizione; RNG per
  (partizione, round) da `SeedSequence.spawn`.
- `classify`/`inertia`/`benchmark.calculate_inertia` sullo stesso pattern;
  rimosso `min_dists` (nessun consumer).
- **Timing locale** (n=20k, d=15, k=20, 8 partizioni): seed 12.80s → 0.42s,
  fit 1.24s → 0.04s, inertia 0.62s → 0.02s (~30x); costo finale identico
  (672313.9799) e golden regression bit-identica su centroidi fissi.
- `agents/smoke_test.py` ora verifica anche determinismo (due run stesso
  seed ⇒ bit-identici) e regressione golden (`agents/fixtures/`,
  generati da `agents/make_fixtures.py` prima del refactor).

### Altro (non in src/)
- `docs/ANALYSIS_PLAN.md`: piano di riproduzione articolo (Tab 3/4,
  Fig 5.1/5.2) congelato e posticipato.
- Harness locale: comando opencode `/grill-me` per la review
  adversarial dei design (`.opencode/`, gitignored).

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
