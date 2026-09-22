"""
predict_file.py — Prédiction en ligne de commande sur un dossier.

Donne le type de fracture pour un CRO et/ou un CRR fournis en arguments.
Fonctionne en mode « règles seules » si aucun modèle n'est entraîné, et
utilise automatiquement le modèle ML (DoctoBERT ou TF-IDF) s'il est présent.

Exemples :
    python -m src.predict_file --cro chemin/vers/CRO.pdf
    python -m src.predict_file --cro CRO.pdf --crr CRR.pdf
    python -m src.predict_file --crr CRR_seul.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from .classifier import FractureClassifier
    from .document_reader import read_document
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.classifier import FractureClassifier
    from src.document_reader import read_document


def main():
    ap = argparse.ArgumentParser(description="Classer une fracture de l'acétabulum.")
    ap.add_argument("--cro", help="chemin du compte rendu opératoire (PDF/ZIP)")
    ap.add_argument("--crr", help="chemin du compte rendu radiologique (PDF/ZIP)")
    args = ap.parse_args()

    if not args.cro and not args.crr:
        ap.error("fournir au moins --cro ou --crr")

    cro_text = read_document(args.cro) if args.cro else ""
    crr_text = read_document(args.crr) if args.crr else ""

    clf = FractureClassifier()
    pred = clf.predict(cro_text, crr_text)

    print("=" * 60)
    print(f"CLASSE PRÉDITE : {pred.label}  —  {pred.label_name}")
    print("=" * 60)
    print(f"Décidé par .......... {pred.source}")
    print(f"Confiance ........... {pred.confidence}")
    if pred.diagnostic_zone:
        print(f"Zone diagnostique ... {pred.diagnostic_zone[:200]}")
    if pred.source == "regle":
        print(f"Règle ............... zone={pred.rule_zone}  terme='{pred.rule_term}'")
    if pred.ml_label is not None:
        print(f"Modèle {pred.ml_name} ... classe {pred.ml_label} "
              f"(confiance {pred.ml_confidence:.2f})")
    if pred.disagreement:
        print("⚠ DÉSACCORD règle/modèle : décision = règle, à faire vérifier.")
    if pred.needs_review and not pred.disagreement:
        print("⚠ Confiance limitée : à faire vérifier par un expert.")
    if pred.label is None:
        print("Aucune règle déclenchée et aucun modèle disponible : à classer manuellement.")


if __name__ == "__main__":
    main()
