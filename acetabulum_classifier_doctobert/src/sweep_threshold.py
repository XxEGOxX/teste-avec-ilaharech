"""
sweep_threshold.py — Effet du seuil de confiance sur le nombre de « autre ».

À quoi ça sert. Le système renvoie « classe 4 / autre » quand aucune règle ne
tranche ET que le modèle n'est pas assez confiant (seuil config.ML_MIN_CONFIDENCE).
Ce seuil est un CURSEUR : plus il est haut, plus le système exige d'être sûr
avant d'affirmer 1/2/3, donc plus il bascule de dossiers en « autre ».

Ce script ne change RIEN au code de décision : il réutilise le vrai
FractureClassifier (celui de l'app), et fait seulement varier
config.ML_MIN_CONFIDENCE pour montrer, sur VOS 99 dossiers réels, le
compromis :
  - combien de dossiers restent classés 1/2/3 (par règle ou par modèle),
  - combien basculent en « autre » (classe 4),
  - et surtout : combien de VRAIS 1/2/3 basculent à tort en « autre »
    (car vos 99 dossiers sont tous des vrais 1/2/3 — donc tout « autre »
    est ici une PERTE de précision, le prix de la prudence).

Comment lire le résultat. Vous cherchez le seuil qui reflète votre priorité :
  - priorité « ne jamais forcer une mauvaise classe » -> seuil haut, quitte à
    marquer plus de dossiers « autre / à vérifier » ;
  - priorité « ne jamais déranger sur un vrai 1/2/3 » -> seuil bas.
Il n'y a pas de bonne réponse universelle : le tableau vous donne les faits,
la décision clinique vous appartient.

⚠ Nécessite un modèle entraîné (models/drbert/ ou models/doctobert/). Sans
modèle, tous les cas non couverts par les règles tombent en « autre » quel que
soit le seuil (rien à arbitrer). Lancez d'abord train_*.py.

Exécution :
    python -m src.sweep_threshold
    python -m src.sweep_threshold --seuils 0.34 0.45 0.60 0.75 0.90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from . import config
    from .classifier import FractureClassifier
    from . import classifier as classifier_module
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.classifier import FractureClassifier
    from src import classifier as classifier_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seuils", type=float, nargs="+",
                        default=[0.34, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                        help="seuils de confiance à tester (défaut : 0.34 … 0.95)")
    cli = parser.parse_args()

    if not config.DATASET_CSV.exists():
        raise SystemExit(
            f"{config.DATASET_CSV} introuvable. Lancez d'abord : python -m src.build_dataset"
        )

    clf = FractureClassifier()
    if clf.ml is None:
        print("⚠ Aucun modèle entraîné trouvé. Sans modèle, tout cas non couvert")
        print("  par les règles tombe en « autre » quel que soit le seuil —")
        print("  il n'y a rien à arbitrer. Entraînez d'abord (train_*.py).\n")

    df = pd.read_csv(config.DATASET_CSV).fillna(
        {"cro_text": "", "crr_text": "", "label": 0})
    true_labels = df["label"].astype(int).tolist()
    cro_texts = df["cro_text"].astype(str).tolist()
    crr_texts = df["crr_text"].astype(str).tolist()
    n = len(df)

    original_threshold = config.ML_MIN_CONFIDENCE
    rows = []
    OOS = config.OUT_OF_SCOPE_ID

    try:
        for seuil in cli.seuils:
            # On fait varier UNIQUEMENT le seuil ; tout le reste de la logique
            # de décision est celle de FractureClassifier (non modifiée).
            config.ML_MIN_CONFIDENCE = seuil
            # le module classifier lit config.ML_MIN_CONFIDENCE au moment de
            # l'appel, donc pas besoin de recréer le classifieur.
            by_rule = by_ml = as_other = 0
            true123_became_other = 0
            correct_123 = 0

            for yt, cro, crr in zip(true_labels, cro_texts, crr_texts):
                pred = clf.predict(cro_text=cro, crr_text=crr)
                # « autre » = tout ce qui n'affirme PAS 1/2/3. Le système a deux
                # façons de ne pas affirmer : label=4 (hors périmètre / contenu
                # non acétabulaire) OU label=None (acétabulaire mais confiance
                # insuffisante -> OTHER_LABEL). Les deux comptent comme « autre »,
                # ce qui correspond au besoin « tout ce qui n'est pas clairement
                # 1/2/3 va dans autre ».
                is_123 = pred.label in config.LABEL_IDS
                if not is_123:
                    as_other += 1
                    if yt in config.LABEL_IDS:      # vrai 1/2/3 basculé en « autre »
                        true123_became_other += 1
                else:
                    if pred.source == "regle":
                        by_rule += 1
                    else:
                        by_ml += 1
                    if pred.label == yt:
                        correct_123 += 1

            rows.append({
                "seuil": seuil,
                "classes_1_2_3": by_rule + by_ml,
                "  dont_regle": by_rule,
                "  dont_modele": by_ml,
                "autre_(cl.4)": as_other,
                "vrais_123_en_autre": true123_became_other,
                "acc_sur_99": round(correct_123 / n, 3),
            })
    finally:
        config.ML_MIN_CONFIDENCE = original_threshold   # on restaure toujours

    results = pd.DataFrame(rows)
    print("=" * 78)
    print("EFFET DU SEUIL DE CONFIANCE SUR LA RÉPARTITION (vos 99 dossiers)")
    print("=" * 78)
    print(results.to_string(index=False))
    print()
    print("Colonnes :")
    print("  classes_1_2_3       = dossiers classés 1, 2 ou 3 (règle OU modèle)")
    print("    dont_regle        = tranchés par une règle (indépendant du seuil)")
    print("    dont_modele       = tranchés par le modèle (dépend du seuil)")
    print("  autre_(cl.4)        = dossiers basculés en « autre / à vérifier »")
    print("  vrais_123_en_autre  = parmi eux, ceux qui étaient en réalité 1/2/3")
    print("                        (= le PRIX de la prudence : ce sont des")
    print("                         signalements en trop sur cette cohorte)")
    print("  acc_sur_99          = exactitude globale (un « autre » sur un vrai")
    print("                        1/2/3 compte comme une erreur)")
    print()
    print(f"Seuil actuellement configuré (config.ML_MIN_CONFIDENCE) : {original_threshold}")
    print()
    print("Lecture : « dont_regle » ne bouge jamais (les règles sont sûres à 100 %).")
    print("Seule la part « modèle » se déplace vers « autre » quand le seuil monte.")
    print("Choisissez le seuil selon votre priorité : sur-signaler les hors-")
    print("périmètre (seuil haut) vs ne pas déranger sur les vrais 1/2/3 (seuil bas).")


if __name__ == "__main__":
    main()
