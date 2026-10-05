import pandas as pd
from owlready2 import *
import os

# 1. Creiamo il dataset tabellare espanso (28 righe)
data = {
    'animale': [
        'Leone', 'Zebra', 'Ghepardo', 'Elefante', 'Tigre', 'Giraffa', 'Lupo', 'Cervo',
        'Rinoceronte', 'Ippopotamo', 'Orso_Bruno', 'Orso_Polare', 'Gazzella', 'Antilope',
        'Iena', 'Leopardo', 'Bufalo', 'Alce', 'Bisonte', 'Puma', 
        'Cinghiale', 'Volpe', 'Lepre', 'Lince', 'Gorilla', 'Gnu', 
        'Coccodrillo', 'Orca'
    ],
    'peso_kg': [
        190, 300, 50, 4000, 220, 1200, 40, 150,
        2500, 1500, 400, 450, 25, 40,
        60, 30, 600, 500, 900, 70,
        100, 7, 4, 20, 160, 200,
        400, 4000
    ],
    'velocita_kmh': [
        80, 65, 120, 40, 65, 50, 60, 70,
        50, 30, 50, 40, 97, 90,
        60, 58, 50, 72, 65, 80,
        40, 50, 70, 50, 40, 80,
        32, 56
    ],
    'rischio_estinzione': [
        1, 0, 1, 1, 1, 0, 0, 0,
        1, 1, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0,
        0, 0
    ]
}

df_base = pd.DataFrame(data)
# Creiamo la cartella data se non esiste
os.makedirs('data', exist_ok=True)
df_base.to_csv('data/dataset_originale.csv', index=False)
print("Dataset di base (28 righe) salvato con successo.")

# 2. Creazione dell'Ontologia e popolamento
onto = get_ontology("http://esame_icon.org/animal_kb.owl")

with onto:
    # Definiamo le Classi Base
    class Animale(Thing): pass
    class Dieta(Thing): pass
    
    # Sottoclassi per la dieta
    class Carne(Dieta): pass
    class Vegetali(Dieta): pass
    
    # Object Property per legare l'animale alla sua dieta
    class mangia(ObjectProperty):
        domain = [Animale]
        range = [Dieta]
        
    # --- RAGIONAMENTO LOGICO ---
    # Assiomi di equivalenza: deduciamo la classe in base a ciò che mangiano
    class Carnivoro(Animale):
        equivalent_to = [Animale & mangia.some(Carne)]
        
    class Erbivoro(Animale):
        equivalent_to = [Animale & mangia.some(Vegetali)]
        
    # Background knowledge esplicita gerarchica
    class PredatoreAlfa(Carnivoro): pass

    # Istanze del cibo
    cibo_carne = Carne("PorzioneCarne")
    cibo_vegetali = Vegetali("PorzioneVegetali")

    # Mappatura della dieta per il popolamento automatico
    dieta_animali = {
        'Leone': cibo_carne, 'Zebra': cibo_vegetali, 'Ghepardo': cibo_carne, 
        'Elefante': cibo_vegetali, 'Tigre': cibo_carne, 'Giraffa': cibo_vegetali, 
        'Lupo': cibo_carne, 'Cervo': cibo_vegetali, 'Rinoceronte': cibo_vegetali, 
        'Ippopotamo': cibo_vegetali, 'Orso_Bruno': cibo_carne, 'Orso_Polare': cibo_carne, 
        'Gazzella': cibo_vegetali, 'Antilope': cibo_vegetali, 'Iena': cibo_carne, 
        'Leopardo': cibo_carne, 'Bufalo': cibo_vegetali, 'Alce': cibo_vegetali, 
        'Bisonte': cibo_vegetali, 'Puma': cibo_carne, 'Cinghiale': cibo_vegetali, 
        'Volpe': cibo_carne, 'Lepre': cibo_vegetali, 'Lince': cibo_carne, 
        'Gorilla': cibo_vegetali, 'Gnu': cibo_vegetali, 'Coccodrillo': cibo_carne, 
        'Orca': cibo_carne
    }

    # Creiamo gli animali nell'ontologia dinamicamente
    istanze_animali = {}
    for nome_animale, cibo in dieta_animali.items():
        istanze_animali[nome_animale] = Animale(nome_animale, mangia=[cibo])
    
    # Dichiariamo esplicitamente chi è all'apice della catena alimentare
    predatori_alfa = ['Leone', 'Tigre', 'Orso_Polare', 'Coccodrillo', 'Orca']
    for p in predatori_alfa:
        PredatoreAlfa.instances().append(istanze_animali[p])

# 3. Esecuzione del Reasoner (HermiT)
print("Avvio il reasoner per le deduzioni logiche...")
sync_reasoner(infer_property_values=True)

# 4. Estrazione delle Feature Semantiche dedotte
inferenze = []
for index, row in df_base.iterrows():
    nome_animale = row['animale']
    individuo = onto.search_one(iri=f"*#{nome_animale}")
    
    # --- CORREZIONE ---
    # Raccogliamo tutte le classi (dirette, inferite dal reasoner e i loro antenati)
    nomi_classi = set()
    for classe_assegnata in individuo.is_a:
        if hasattr(classe_assegnata, "ancestors"):
            for genitore in classe_assegnata.ancestors():
                if hasattr(genitore, "name") and genitore.name not in ['Thing', 'Animale']:
                    nomi_classi.add(genitore.name)
    
    # Costruisce la riga con le feature arricchite cercando i nomi nel set
    inferenze.append({
        'animale': nome_animale,
        'is_carnivoro': 1 if 'Carnivoro' in nomi_classi else 0,
        'is_erbivoro': 1 if 'Erbivoro' in nomi_classi else 0,
        'is_predatore_alfa': 1 if 'PredatoreAlfa' in nomi_classi else 0
    })

# Salvataggio del dataset arricchito
df_inferito = pd.DataFrame(inferenze)
df_arricchito = pd.merge(df_base, df_inferito, on='animale')
df_arricchito.to_csv('data/dataset_arricchito.csv', index=False)

print("\nFase 1 completata! Anteprima dei dati arricchiti (prime 5 righe):")
print(df_arricchito.head())

os.makedirs('kb', exist_ok=True)
onto.save(file="kb/animal_kb.owl", format="rdfxml")
print("Ontologia OWL salvata con successo nella cartella kb/")