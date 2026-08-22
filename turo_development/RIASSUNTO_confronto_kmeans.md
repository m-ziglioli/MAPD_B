# Riassunto — Confronto k-means|| vs k-means++ vs Random (todo #4)

## Obiettivo

Capire come confrontare la nostra implementazione di **k-means||** (Bahmani
et al., VLDB 2012) con l'inizializzazione classica **k-means++** (seriale) e
con **Random**, seguendo per quanto possibile la metodologia sperimentale
del paper originale.

## Decisioni prese e motivazioni

- **Dataset**: KDDCup99 al **10%**, già usato nel resto del progetto — è lo
  stesso dataset del paper, ed è l'unica scala su cui ha senso far girare
  k-means++ seriale (nel paper stesso, sul 100% con k grandi, gli autori
  dichiarano k-means++ seriale troppo lento e lo escludono dal confronto —
  quello rimane il nostro todo #7, separato, solo metodi paralleli).
- **Tre metodi a confronto**: k-means|| (nostro), k-means++ classico, Random
  — gli stessi baseline principali del paper (escluso "Partition", il terzo
  baseline, più complesso e non richiesto).
- **Metriche in stile paper**: costo "seed" (subito dopo l'inizializzazione)
  e costo "final" (dopo Lloyd's a convergenza) tenuti **separati**, non un
  solo numero; numero di iterazioni di Lloyd's come proxy di velocità
  indipendente dall'hardware — **mai** un confronto diretto di wall-clock
  tra seriale e parallelo, perché girano su substrati diversi (client
  single-thread vs cluster Dask).
- Le combinazioni k-means|| da testare (l, r, numero di partizioni)
  restano **da scegliere noi**, configurabili nel notebook.

## File nuovi creati

(nessun file esistente toccato, tranne un fix — vedi sotto)

- `kmeans_serial.py` — k-means++ e Random seriali via scikit-learn
- `kmeans_comparison.py` — orchestratore che fa girare tutti e tre i metodi
  e salva i risultati in CSV
- `comparison_analysis.py` — tabelle (stile Tabelle 1/2 del paper) e
  grafici dedicati al confronto
- `comparison.ipynb` — notebook dedicato, a specchio di `analysis.ipynb`,
  con una cella di verifica dei tempi prima del run completo

Tutti testati singolarmente e in un run end-to-end su un mini-cluster
locale con dati sintetici (non ancora sul cluster reale) — funzionano
correttamente.

## Bug trovato e corretto durante i test

k-means|| non si fermava mai da solo e consumava sempre tutte le iterazioni
disponibili, mentre k-means++/Random convergevano in pochi secondi.

**Causa**: il criterio di stop in `kmeans_parallel.fit()` confrontava lo
spostamento totale dei centroidi con una soglia fissa non proporzionata al
numero di cluster k — con k=500-1000 quella soglia è quasi impossibile da
raggiungere anche a clustering già stabile. Sklearn invece si ferma appena
l'assegnazione punto→cluster smette di cambiare, usando la soglia numerica
solo come rete di sicurezza.

**Fix applicato**: aggiunto lo stesso controllo a `kmeans_parallel.py`.
Testato — ora k-means|| converge in un numero di iterazioni comparabile
agli altri due metodi. Risolve anche il nostro todo #6 ("sistemare
convergenza con soglia"), che era esattamente questo problema.

## Da decidere/fare in chiamata

- Scegliere i valori di `k` e le combinazioni k-means|| (l, r, partizioni)
  da testare davvero in `comparison.ipynb`
- Far girare il notebook sul cluster reale (dopo la cella di verifica dei
  tempi)
- Decidere se includere anche uno sweep su `r` per il grafico
  costo-vs-round (stile Figura 5.2/5.3 del paper) o solo il confronto
  diretto a k fissato
- Todo #7 resta a parte: confronto sul 100% del dataset, solo metodi
  paralleli, niente k-means++ seriale
