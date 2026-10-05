import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

# Pulisce l'output del terminale da eventuali avvisi di scikit-learn
warnings.filterwarnings('ignore') 

# 1. Caricamento Dati dalla cartella 'data' (creata nello step 1)
df = pd.read_csv('data/dataset_arricchito.csv')

# Scenario A: Baseline (Solo dati fisici)
X_base = df[['peso_kg', 'velocita_kmh']]

# Scenario B: OntoBK (Dati fisici + Background Knowledge Semantica dedotta)
X_onto = df[['peso_kg', 'velocita_kmh', 'is_carnivoro', 'is_erbivoro', 'is_predatore_alfa']]

y = df['rischio_estinzione']

# 2. Configurazione della Nested Cross-Validation (aggiornata a 5-Fold per dataset di 28 righe)
inner_cv = KFold(n_splits=5, shuffle=True, random_state=42) # Loop interno: Scelta parametri ottimali
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42) # Loop esterno: Valutazione non distorta delle performance

# 3. Definizione Modelli e Spazio di Ricerca Iperparametri
modelli = {
    'Support Vector Machine': {
        # La Pipeline garantisce che i dati vengano standardizzati ad ogni fold per non falsare l'addestramento
        'estimator': Pipeline([('scaler', StandardScaler()), ('svm', SVC())]),
        'params': {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf']
        }
    },
    'Random Forest': {
        'estimator': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [10, 50, 100],
            'max_depth': [None, 3, 5]
        }
    },
    'Naive Bayes': {
        # Modello probabilistico (Cap. 10) usato come baseline avanzata, non necessita di tuning
        'estimator': GaussianNB(),
        'params': {} 
    }
}

# 4. Esecuzione Esperimento e Raccolta Metriche
risultati = []

print("Addestramento con Nested Cross-Validation (5-Fold) in corso...\n")

for nome, info in modelli.items():
    # Loop Interno gestito in automatico
    grid = GridSearchCV(estimator=info['estimator'], 
                        param_grid=info['params'], 
                        cv=inner_cv, 
                        scoring='accuracy')
    
    # Loop Esterno: Valutazione Baseline
    scores_base = cross_val_score(grid, X_base, y, cv=outer_cv, scoring='accuracy')
    
    # Loop Esterno: Valutazione OntoBK
    scores_onto = cross_val_score(grid, X_onto, y, cv=outer_cv, scoring='accuracy')
    
    risultati.append({
        'Modello': nome,
        'Accuracy Baseline': f"{np.mean(scores_base):.2f} ± {np.std(scores_base):.2f}",
        'Accuracy OntoBK': f"{np.mean(scores_onto):.2f} ± {np.std(scores_onto):.2f}"
    })

# 5. Stampa Tabella Riassuntiva in formato Markdown (da copiare nel README)
df_risultati = pd.DataFrame(risultati)
print("=== TABELLA DEI RISULTATI (Media ± Dev. Standard) ===\n")
print(df_risultati.to_markdown(index=False))