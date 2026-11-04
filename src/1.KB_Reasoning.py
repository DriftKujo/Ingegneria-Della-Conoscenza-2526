import os
from owlready2 import get_ontology, sync_reasoner, ObjectProperty, Thing


def load_real_ontology():
    print("--- FASE 1: Caricamento Ontologia Reale e Reasoning ---")
    
    # creo le cartelle al volo se non ci sono
    os.makedirs('data/ontologies', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    onto_path = "data/ontologies/symp.owl"
    
    # controllo se il file esiste veramente o se ci siamo dimenticati di metterlo
    if not os.path.exists(onto_path):
        print(f"ERRORE: File non trovato in {onto_path}.")
        print("Scarica un'ontologia reale (es. symp.owl) e mettila in data/ontologies/")
        return
        
    print(f"Caricamento dell'ontologia reale da: {onto_path} ...")
    # carico il file owl usando il formato URI file://
    onto = get_ontology("file://" + os.path.abspath(onto_path)).load()
    
    # conto le classi giusto per vedere se ha caricato roba sensata
    num_classes = len(list(onto.classes()))
    print(f"Ontologia caricata! Trovate {num_classes} classi reali.")
    
    print("Iniezione di una Object Property...")
    
    # aggiungo le mie cose custom all'ontologia caricata
    with onto:
        class Paziente(Thing): 
            pass
            
        # Definiamo la proprietà relazionale imponendo vincoli logici
        class ha_sintomo(ObjectProperty):
            domain = [Paziente]
            range = [Thing] # Accetta qualsiasi sintomo dell'ontologia reale
            
        # Creiamo un'istanza per dimostrare il funzionamento
        classi_reali = list(onto.classes())
        
        # piccolo controllo per evitare IndexError se l'ontologia è vuota o quasi
        if len(classi_reali) > 1:
            sintomo_target = classi_reali[1] # Prendiamo una classe vera a caso
            paziente_test = Paziente("Paziente_Test_001")
            
            # glielo assegno come sintomo
            paziente_test.ha_sintomo = [sintomo_target]
            print(f"Istanza creata: Paziente_Test_001 ha_sintomo {sintomo_target.name}")
        else:
            print("Attenzione: poche classi trovate, salto la creazione dell'istanza di test.")
    
    print("Esecuzione del reasoner (HermiT) sull'ontologia ibrida...")
    
    # faccio partire il reasoner
    with onto:
        try:
            sync_reasoner()
        except Exception as e:
            print(f"Mhh, qualcosa è andato storto col reasoner: {e}")
            # lascio questo blocco per sicurezza nel caso fallisca
    
    output_file = "data/processed/inferred_ontology.owl"
    
    # salvo tutto nel formato rdfxml originale
    onto.save(file=output_file, format="rdfxml")
    print(f"Ontologia inferita salvata in: {output_file}")

if __name__ == "__main__":
    load_real_ontology()
    # passa al secondo script