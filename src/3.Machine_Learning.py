import pandas as pd
import numpy as np
import os
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# imposto il seed globale per rendere l'esperimento completamente riproducibile ad ogni esecuzione
RANDOM_SEED = 42

def evaluate_model_nested_cv(X, y, model_type, description):
    """
    Esegue una Nested Cross-Validation (CV annidata):
    - Inner Loop (3 fold): per la ricerca degli iperparametri ottimali (GridSearch).
    - Outer Loop (5 fold): per stimare le prestazioni di generalizzazione senza bias o data leakage.
    """
    print(f"\nValutazione Modello: {description} ({model_type})")

    # KFold interno per la validazione degli iperparametri e KFold esterno per la valutazione finale
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    if model_type == 'SVM':
        # la Pipeline applica lo StandardScaler SOLO sui dati di train di ciascun fold,
        # evitando il Data Leakage dai dati di test 
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=RANDOM_SEED))
        ])
        
        # griglia di iperparametri da testare per la SVM (C e tipo di Kernel)
        param_grid = {
            'svm__C': [0.1, 1, 10],
            'svm__kernel': ['linear', 'rbf']
        }
    else:
        # per la Random Forest non serve lo scaler, ma incapsulo comunque in Pipeline per omogeneità
        pipeline = Pipeline([
            ('rf', RandomForestClassifier(random_state=RANDOM_SEED))
        ])
        
        # iperparametri per la Random Forest (numero di alberi e profondità massima)
        param_grid = {
            'rf__n_estimators': [50, 100],
            'rf__max_depth': [None, 10]
        }

    # la GridSearch lavora sul cv interno (3-fold) per trovare la combinazione migliore di parametri
    clf = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=inner_cv, scoring='accuracy')

    # la cross_validate esegue il cv esterno (5-fold) per valutare il modello ottimizzato
    nested_scores = cross_validate(clf, X, y, cv=outer_cv, scoring=('accuracy', 'f1_macro'))

    # calcolo medie e deviazioni standard come espressamente richiesto dalle linee guida d'esame
    acc_mean = nested_scores['test_accuracy'].mean()
    acc_std = nested_scores['test_accuracy'].std()
    f1_mean = nested_scores['test_f1_macro'].mean()
    f1_std = nested_scores['test_f1_macro'].std()

    # stampiamo i risultati nel formato (Media +/- Dev.Std) evitando la trappola del singolo run
    print(f"Accuracy : {acc_mean:.3f} +/- {acc_std:.3f}")
    print(f"F1-Macro : {f1_mean:.3f} +/- {f1_std:.3f}")

def run_ml_pipeline():
    print("--- FASE 3: Apprendimento Supervisionato (Nested CV) ---")

    dataset_file = "data/processed/dataset_arricchito.csv"
    
    # controllo che il dataset arricchito dallo script 2 sia stato generato correttamente
    if not os.path.exists(dataset_file):
        print(f"Esegui prima 2_semantic_search.py. File non trovato: {dataset_file}")
        return

    df = pd.read_csv(dataset_file)
    y = df['Malattia_Grave'].values

    # Baseline: il modello "cieco" che usa solo le feature tabulari standard (Eta, BMI)
    X_baseline = df[['Eta', 'BMI']].values

    # Onto-Model: il modello istruito che sfrutta anche la Distanza Semantica estratta dal Grafo
    X_onto = df[['Eta', 'BMI', 'Distanza_Semantica']].values

    # confronto prestazionale su SVM
    print("\n--- Test su Support Vector Machine ---")
    evaluate_model_nested_cv(X_baseline, y, 'SVM', "Baseline")
    evaluate_model_nested_cv(X_onto, y, 'SVM', "Onto-Model")

    # confronto prestazionale su Random Forest
    print("\n--- Test su Random Forest ---")
    evaluate_model_nested_cv(X_baseline, y, 'RF', "Baseline")
    evaluate_model_nested_cv(X_onto, y, 'RF', "Onto-Model")

if __name__ == "__main__":
    run_ml_pipeline()
    # fine della pipeline