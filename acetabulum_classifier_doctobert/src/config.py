"""
config.py — Configuration centrale du projet.

Tous les chemins et constantes sont définis ici pour qu'un seul fichier
gouverne le comportement de toute la chaîne (extraction, dataset, modèles, app).
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Chemins
# --------------------------------------------------------------------------
# config.py est dans src/ ; la racine du projet est donc un niveau au-dessus.
ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = ROOT / "data" / "raw"          # CRO/CRR (.pdf) + Excel d'annotation
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

EXCEL_NAME = "Excel_labels_tous_fractures.xlsx"  # nom du fichier Excel dans data/raw
DATASET_CSV = DATA_PROCESSED / "dataset.csv"
EXTRACTION_REPORT_CSV = DATA_PROCESSED / "extraction_report.csv"
RULES_REPORT_CSV = DATA_PROCESSED / "rules_report.csv"

TFIDF_MODEL = MODELS_DIR / "tfidf_logreg.joblib"
DOCTOBERT_DIR = MODELS_DIR / "doctobert"    # modèle DoctoBERT fine-tuné (HuggingFace)

# --------------------------------------------------------------------------
# Colonnes du fichier Excel
# --------------------------------------------------------------------------
COL_ID = "ID_local"               # identifiant du dossier == nom du PDF
COL_OBS = "Observation(s) CR"      # extraction manuelle de l'expert
COL_LABEL = "Label"               # classe attendue (1, 2 ou 3)
COL_SOURCE = "source_observation"  # CRO ou CRR : où l'expert a trouvé l'info

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
LABELS = {
    1: "Fracture colonne antérieure",
    2: "Fracture bicolonne",
    3: "Fracture colonne antérieure + hémi-transverse postérieure",
    4: "Autre fracture / hors périmètre",
}

# --------------------------------------------------------------------------
# LABEL_IDS — les classes APPRISES par le modèle (têtes de la softmax)
# --------------------------------------------------------------------------
# CHANGEMENT MAJEUR par rapport aux versions précédentes du projet.
#
# Historiquement, LABEL_IDS = [1, 2, 3] et la classe 4 n'était JAMAIS apprise :
# la cohorte initiale (99 dossiers, 50/36/13) ne contenait aucun exemple de
# classe 4. Ajouter un 4e neurone sans exemple positif aurait été inutile et
# trompeur — le modèle n'aurait jamais pu le prédire.
#
# La cohorte comporte désormais de VRAIS cas de classe 4 (paroi postérieure,
# colonne postérieure isolée, fracture transverse, fracture en T, fractures
# du bassin, luxations associées...). La classe 4 devient donc une classe
# apprise à part entière, au même titre que 1, 2 et 3.
#
# Conséquences à garder en tête :
#   - le modèle possède maintenant 4 sorties ; num_labels suit automatiquement
#     len(LABEL_IDS) dans train_*.py ;
#   - les métriques (macro-F1, matrices de confusion, bootstrap, McNemar) sont
#     calculées sur 4 classes : elles NE SONT PAS comparables numériquement aux
#     résultats publiés sur 3 classes. Toute comparaison avec les chiffres
#     antérieurs doit être explicitement signalée comme portant sur des tâches
#     différentes ;
#   - la classe 4 reste hétérogène par nature (elle regroupe plusieurs types
#     de fractures distincts) : un macro-F1 plus faible sur cette classe est
#     attendu et doit être interprété comme tel, pas comme un défaut du modèle.
LABEL_IDS = [1, 2, 3, 4]     # classes apprises par le modèle (4 incluse)

# --------------------------------------------------------------------------
# OUT_OF_SCOPE_ID — désormais un GARDE-FOU, plus une classe « non apprise »
# --------------------------------------------------------------------------
# La valeur reste 4, mais son rôle a changé. Elle est utilisée par :
#   (a) rules.classify_out_of_scope() : la règle hors-périmètre continue de
#       détecter les types nommés explicitement (paroi postérieure, transverse,
#       en T, colonne postérieure). Elle reste testée APRÈS les règles 1/2/3 ;
#   (b) classifier.py : si le contenu n'est pas acétabulaire, la proposition du
#       modèle est écartée au profit de la classe 4. Ce garde-fou reste utile —
#       il empêche le modèle d'affirmer une classe 1/2/3 sur un document qui ne
#       relève pas du tout de l'acétabulum.
#
# À ne pas confondre avec OTHER_LABEL ci-dessous : prédire la classe 4 signifie
# « ce document décrit une fracture d'un type non couvert par 1/2/3 », alors que
# OTHER_LABEL signifie « le système refuse de se prononcer ». Ce sont deux
# situations différentes qui doivent rester distinctes dans l'interface et dans
# l'analyse des erreurs.
OUT_OF_SCOPE_ID = 4          # garde-fou ; désormais AUSSI une classe apprise

# Statut hors des 3 classes : document non acétabulaire, ou classification
# trop incertaine pour être affirmée. Évite de forcer une des 3 classes.
OTHER_LABEL = "Non déterminé / Autre fracture"

# En l'absence de règle, on n'affirme une classe DoctoBERT que si sa confiance
# atteint ce seuil ; en dessous, on renvoie « Non déterminé / Autre fracture ».
ML_MIN_CONFIDENCE = 0.45

# --------------------------------------------------------------------------
# Modèle DoctoBERT
# --------------------------------------------------------------------------
# Identifiant HuggingFace du modèle pré-entraîné médical français (Doctolib
# Lab, licence Apache-2.0). Choisi comme modèle par défaut après comparaison :
# 0.960 d'accuracy / 0.939 de macro-F1 en validation croisée (modèle seul,
# sans les règles), contre 0.677 / 0.644 pour Dr-BERT/DrBERT-7GB sur ce même
# jeu de données et ce même pipeline d'extraction.
DOCTOBERT_BASE = "doctolib-lab/doctobert-fr-base"
MAX_LEN = 512        # longueur de séquence (la zone diagnostique est courte)
CV_FOLDS = 5           # validation croisée stratifiée pour des métriques honnêtes
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Data Augmentation (DA) — même stratégie que le fork DrBERT
# --------------------------------------------------------------------------
# Appliquée ici pour permettre les 4 expériences comparables demandées :
# DrBERT / DrBERT+DA / DoctoBERT / DoctoBERT+DA. Sur DoctoBERT (déjà ~0,97),
# on ne s'attend pas forcément à un gain — c'est précisément ce que la
# comparaison honnête (--compare-baseline) permet de vérifier plutôt que de
# supposer. Voir src/augmentation.py pour la « RÈGLE D'OR » (aucune technique
# ne touche au terme médical décisif).
AUGMENT_ENABLED = True
# Multiplicateurs d'augmentation, calibrés pour rapprocher les effectifs
# d'entraînement des 4 classes :
#   classe 1 : 50 x2 = 100     classe 3 : 13 x5 =  65
#   classe 2 : 36 x3 = 108     classe 4 : 26 x4 = 104
# Rapport majoritaire/minoritaire : 3,8 -> 1,66 après augmentation.
AUGMENT_MULTIPLIER = {1: 2, 2: 3, 3: 5, 4: 4}   # occurrences totales par classe
AUGMENT_VERIFY_RULE_CONSISTENCY = True
AUGMENT_PARAMS = dict(
    p_boilerplate=0.6,
    p_eda=0.8,
    p_ocr=0.5,
    eda_p_delete=0.10,
    eda_p_swap=0.15,
    ocr_p=0.02,
)
