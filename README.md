# ML+OntoBK: Integrazione di Semantic Background Knowledge nel Machine Learning
Repository per il progetto di Ingegneria della Conoscenza realizzato da:
- Matthias Andriola (mat.797413)

# Esecuzione 
## Fase iniziale
Installare Java (Requisito di sistema fondamentale per il motore di ragionamento logico HermiT)

<code>[https://www.oracle.com/java/technologies/downloads/](https://www.oracle.com/java/technologies/downloads/)</code>

Clonare il progetto 

<code>git clone https://github.com/DriftKujo/Ingegneria-Della-Conoscenza-2526.git</code>

Creare l'ambiente virtuale ed attivarlo

<code> cd Ingegneria-Della-Conoscenza-2526 </code>

Su Windows:
<code> python -m venv Ingegneria-Della-Conoscenza-2526 </code>

Su macOS/Linux:
<code> python3 -m venv Ingegneria-Della-Conoscenza-2526 </code>

Installare le dipendenze:

<code>pip install -r requirements.txt</code>

*Nota: L'ontologia reale della Symptom Ontology (symp.owl) necessaria per il progetto è già inclusa all'interno della repository nella directory `data/ontologies/`.*

## Esecuzione del codice

**Importante eseguire i run nell'ordine in cui sono posti almeno per la prima volta. Assicurarsi di lanciare i comandi dalla cartella radice del progetto.**<br>

 Fase 1: Caricamento Ontologia reale, iniezione regole logiche e Reasoning (HermiT) <br>

<code> python src/1.KB_Reasoning.py </code>
 
 Fase 2: Ricerca su Grafo, estrazione feature (Shortest Path) e salvataggio dataset <br>

<code> python src/2.Ricerca_Grafo.py </code>
 
 Fase 3: Apprendimento Supervisionato e valutazione (Nested Cross-Validation) <br>

<code> python src/3.Machine_Learning.py </code>
