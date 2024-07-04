# Iris Flower Classification with Logistic Regression

Questo codice implementa una pipeline di machine learning per classificare le specie di fiori di Iris utilizzando la Regressione Logistica. Utilizza la libreria scikit-learn per la manipolazione dei dati, il preprocessing, l'addestramento del modello e la valutazione.

## Funzionalità
### Caricamento dei Dati
Carica il dataset dei fiori di Iris dal modulo datasets di scikit-learn.
Converte i dati e le etichette target in un DataFrame di pandas per una manipolazione più semplice.
### Preprocessing dei Dati
Verifica la presenza di righe duplicate utilizzando la funzione check_duplicated.
Identifica e gestisce i valori mancanti imputando la media di ogni colonna (può essere personalizzato per casi d'uso specifici).
Rimuove le righe duplicate dopo aver gestito i valori mancanti.
### Scalatura delle Feature e Addestramento del Modello
Divide i dati in set di addestramento e test (70% addestramento, 30% test) utilizzando train_test_split.
Applica una scalatura standard alle feature di addestramento e test utilizzando StandardScaler.
Crea e addestra un modello di Regressione Logistica con un random state di 42 per la riproducibilità.
### Valutazione del Modello
Effettua predizioni sul set di test.
Calcola e stampa l'accuracy score utilizzando accuracy_score.
Genera un report di classificazione utilizzando classification_report, fornendo metriche dettagliate per classe.
Calcola e visualizza la matrice di confusione utilizzando confusion_matrix per visualizzare le prestazioni del modello.
### Predizione su Nuovi Dati
Crea un DataFrame di esempio con nuovi dati da prevedere.
Trasforma i nuovi dati utilizzando lo scaler addestrato (scaler).
Predice l'etichetta di classe per i nuovi dati utilizzando il modello addestrato (model).
Stampa la specie di Iris prevista.
## Personalizzazione
- Gestione dei Valori Mancanti: La funzione pre_process_data attualmente imputa i valori mancanti con la media. Puoi modificarla per gestire i valori mancanti in modo diverso, come eliminando le righe con valori mancanti o utilizzando tecniche di imputazione più sofisticate.
- Selezione del Modello: Anche se qui viene utilizzata la Regressione Logistica, puoi sperimentare con altri algoritmi di machine learning di scikit-learn per migliorare le prestazioni della classificazione.
- Ottimizzazione dei Parametri: Il modello di Regressione Logistica in classify_iris utilizza i parametri di default. Puoi esplorare tecniche di ottimizzazione degli iperparametri per ottimizzare le prestazioni del modello per il tuo dataset specifico.
