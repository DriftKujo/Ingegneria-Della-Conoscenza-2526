# ML+OntoBK: Integrazione di Semantic Background Knowledge nel Machine Learning
Repository per il progetto di Ingegneria della Conoscenza realizzato da:
- [Tuo Nome Cognome] (mat.[Tua Matricola])

# Esecuzione 
## Fase iniziale
Installare Java (Requisito di sistema fondamentale per il motore di ragionamento logico HermiT)

<code>https://adoptium.net/</code>

Clonare il progetto 

<code>git clone https://github.com/[TuoUsername]/[NomeRepository].git</code>

Creare l'ambiente virtuale ed attivarlo

<code> cd [NomeRepository] </code>

Su Windows:
<code> python -m venv venv </code>
<code> venv\Scripts\activate </code>

Su macOS/Linux:
<code> python3 -m venv venv </code>
<code> source venv/bin/activate </code>

Installare le dipendenze:

<code>pip install -r requirements.txt</code>

*Nota: L'ontologia reale della Symptom Ontology (symp.owl) necessaria per il progetto è già inclusa all'interno della repository nella directory `data/ontologies/`.*

## Esecuzione del codice

**Importante eseguire i run nell'ordine in cui sono posti almeno per la prima volta. Assicurarsi di lanciare i comandi dalla cartella radice del progetto.**<br>

 Fase 1: Caricamento Ontologia reale, iniezione regole logiche e Reasoning (HermiT) <br>

<code> python src/1_ontology_reasoner.py </code>
 
 Fase 2: Ricerca su Grafo, estrazione feature (Shortest Path) e salvataggio dataset <br>

<code> python src/2_semantic_search.py </code>
 
 Fase 3: Apprendimento Supervisionato e valutazione (Nested Cross-Validation) <br>

<code> python src/3_supervised_ml.py </code>
