import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from owlready2 import get_ontology
import os


def create_knowledge_graph(onto):
    """
    Converte la TBox dell'ontologia in uno Spazio degli Stati (Grafo NetworkX).
    Cicla su tutte le classi e crea un arco tra ogni classe e le sue sottoclassi.
    """
    G = nx.Graph()
    # ciclo su ogni classe presente nell'ontologia inferita
    for classe in list(onto.classes()):
        for sub in classe.subclasses():
            # collego il nodo genitore al nodo figlio (relazione SubClassOf)
            G.add_edge(classe.name, sub.name)
    return G

def calculate_semantic_distance(G, source_node, target_node):
    """
    Calcola la distanza minima (Shortest Path) sul grafo tra sintomo del paziente e sintomo target.
    """
    try:
        # calcolo del cammino minimo usando l'algoritmo di ricerca su grafo
        return nx.shortest_path_length(G, source=source_node, target=target_node)
    except nx.NetworkXNoPath:
        # se i due nodi non sono collegati sul grafo, imposto una penalità alta
        return 10
    except nx.NodeNotFound:
        # gestione di sicurezza nel caso un nodo non fosse presente nel grafo
        return 10

def generate_datasets():
    print("--- FASE 2: Ricerca su Grafo ed Estrazione Feature ---")

    # ricreo le cartelle se per caso qualcuno le ha cancellate nel frattempo
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    onto_file = "data/processed/inferred_ontology.owl"
    
    # controllo preventivo: se manca il file dello script 1, blocco l'esecuzione con un avviso chiaro
    if not os.path.exists(onto_file):
        print(f"Esegui prima lo script 1. File non trovato: {onto_file}")
        return

    print("1. Caricamento Ontologia Reale e Costruzione Grafo...")
    onto = get_ontology("file://" + os.path.abspath(onto_file)).load()
    kg = create_knowledge_graph(onto)

    # estrazione dinamica delle classi vere dall'ontologia (evito di scriverle a mano per non fare errori di battitura)
    classi_reali = [c.name for c in list(onto.classes()) if c.name != "Thing" and c.name != "Paziente"]
    
    # prendo 5 sintomi reali per costruire i dati dei pazienti e 1 sintomo come riferimento target
    sintomi_possibili = classi_reali[1:6]
    target_symptom = classi_reali[0]

    print("Generazione immagine del sotto-grafo per il report PDF...")
    try:
        # per evitare l'effetto "palla di pelo" nell'immagine, estraggo SOLO i nodi coinvolti nei cammini minimi
        nodi_da_disegnare = set([target_symptom])

        for sintomo in sintomi_possibili:
            try:
                # ricavo la lista dei nodi che formano il percorso esatto fino al target
                path = nx.shortest_path(kg, source=sintomo, target=target_symptom)
                nodi_da_disegnare.update(path)
            except nx.NetworkXNoPath:
                nodi_da_disegnare.add(sintomo)

        # creo il sotto-grafo pulito contenente solo i percorsi utili
        sub_kg = kg.subgraph(nodi_da_disegnare)

        plt.figure(figsize=(10, 6))

        # kamada_kawai_layout distribuisce i nodi in modo ben distanziato e leggibile
        pos = nx.kamada_kawai_layout(sub_kg)

        # disegno gli archi grigi e i nodi standard azzurri
        nx.draw_networkx_edges(sub_kg, pos, edge_color='gray', width=2, alpha=0.7)
        nx.draw_networkx_nodes(sub_kg, pos, node_color='lightblue', node_size=2500)

        # il nodo target lo evidenzio in rosso per farlo staccare visivamente nella relazione
        nx.draw_networkx_nodes(sub_kg, pos, nodelist=[target_symptom],
                               node_color='lightcoral', node_size=3000)

        # inserisco le etichette con il nome dei nodi
        nx.draw_networkx_labels(sub_kg, pos, font_size=9, font_weight='bold')

        plt.title(f"Percorsi di Distanza Semantica verso '{target_symptom}'")
        plt.axis('off')

        # salvo in alta risoluzione (dpi=300) 
        plt.savefig("data/processed/grafo_semantico.png", bbox_inches='tight', dpi=300)
        print("Immagine ottimizzata salvata in: data/processed/grafo_semantico.png")
    except Exception as e:
        print(f"Impossibile salvare l'immagine (proseguo comunque): {e}")

    print(f"\nSintomi usati per il dataset: {sintomi_possibili}")
    print(f"Sintomo target per il calcolo della distanza: {target_symptom}")

    print("\n2. Creazione del dataset originale (RAW)...")
    # imposto il seed randomico per garantire la riproducibilità esatta dei dati alla commissione
    np.random.seed(42)
    n_samples = 200

    # creo la struttura tabulare base con pazienti e caratteristiche biometriche
    data = {
        'Eta': np.random.randint(18, 90, n_samples),
        'BMI': np.random.uniform(18.5, 35.0, n_samples),
        'Sintomo_Principale': np.random.choice(sintomi_possibili, n_samples),
        'Malattia_Grave': np.random.randint(0, 2, n_samples)
    }
    df_raw = pd.DataFrame(data)

    # salvo il dataset grezzo non arricchito
    raw_csv = "data/raw/dataset_base.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"Dataset RAW salvato in: {raw_csv}")

    print("\n3. Iniezione Background Knowledge (Shortest Path)...")
    df_enriched = df_raw.copy()

    # trasformo la relazione semantica qualitativa in una feature numerica per il ML
    df_enriched['Distanza_Semantica'] = df_enriched['Sintomo_Principale'].apply(
        lambda x: calculate_semantic_distance(kg, x, target_symptom)
    )

    # salvo il dataset arricchito pronto per essere dato in pasto al modulo di ML (Fase 3)
    processed_csv = "data/processed/dataset_arricchito.csv"
    df_enriched.to_csv(processed_csv, index=False)
    print(f"Dataset PROCESSED salvato in: {processed_csv}")

if __name__ == "__main__":
    generate_datasets()
    # passa al terzo script