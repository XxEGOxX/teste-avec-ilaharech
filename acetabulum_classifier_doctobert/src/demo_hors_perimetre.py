"""
demo_hors_perimetre.py — Démonstration qualitative du mécanisme « autre ».

⚠ CE N'EST PAS UNE ÉVALUATION. Les comptes rendus ci-dessous sont SYNTHÉTIQUES
(écrits à la main pour la démonstration), pas des données patient. Ce script
ne mesure aucune performance : il MONTRE, cas par cas, comment le système
réagit à des CR qui ne relèvent pas des 3 classes cibles — pour vérifier de
visu que le mécanisme d'abstention / classe « autre » se déclenche bien.

Une vraie évaluation de ce mécanisme nécessiterait une cohorte de CR
hors-périmètre réels et annotés (travaux futurs). À utiliser uniquement comme
illustration qualitative (p. ex. une figure d'exemple dans l'article).

Les cas couvrent les DIFFÉRENTS chemins qui mènent à « autre » :
  A. Type acétabulaire non couvert, nommé explicitement (paroi postérieure,
     transverse, en T, colonne postérieure) -> attrapé par la RÈGLE
     hors-périmètre.
  B. Fracture d'un autre os (fémur, bassin non acétabulaire) -> attrapé par le
     garde-fou « contenu non acétabulaire » (abstention).
  C. Contre-exemples IN-périmètre (classe 1/2/3) -> doivent rester classés
     1/2/3, y compris quand un terme hors-périmètre coexiste avec un terme
     couvert (la règle in-périmètre prime).

Exécution (nécessite un modèle entraîné pour les cas où le modèle intervient) :
    python -m src.demo_hors_perimetre
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    from . import config
    from .classifier import FractureClassifier
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.classifier import FractureClassifier


# (description, texte_CRO, attendu_hors_perimetre?) — attendu sert juste à
# afficher un ✓/✗ indicatif, PAS à mesurer quoi que ce soit.
CAS = [
    # --- A. Types acétabulaires non couverts, nommés explicitement ---
    ("A1 · Paroi postérieure isolée (type non couvert)",
     "DIAGNOSTIC: Fracture de la paroi posterieure du cotyle droit avec "
     "fragment marginal.", True),
    ("A2 · Fracture transverse pure (type non couvert)",
     "DIAGNOSTIC: Fracture transverse de l'acetabulum gauche, trait unique "
     "juxta-tectal.", True),
    ("A3 · Fracture en T (type associé non couvert)",
     "DIAGNOSTIC: Fracture en T du cotyle droit.", True),
    ("A4 · Colonne postérieure isolée (type non couvert)",
     "DIAGNOSTIC: Fracture de la colonne posterieure isolee du cotyle gauche.",
     True),

    # --- B. Fracture d'un autre os : pas d'anatomie acétabulaire ---
    ("B1 · Fracture du col fémoral (autre os)",
     "DIAGNOSTIC: Fracture du col du femur gauche, Garden IV. "
     "INDICATION: Arthroplastie.", True),
    ("B2 · Fracture diaphysaire du fémur (autre os)",
     "DIAGNOSTIC: Fracture diaphysaire du femur droit, comminutive.", True),

    # --- C. Contre-exemples IN-périmètre : doivent RESTER 1/2/3 ---
    ("C1 · Bicolonne (classe 2) — ne doit PAS partir en autre",
     "DIAGNOSTIC: Fracture bicolonne du cotyle droit.", False),
    ("C2 · Colonne antérieure (classe 1) — ne doit PAS partir en autre",
     "DIAGNOSTIC: Fracture de la colonne anterieure du cotyle gauche.", False),
    ("C3 · CA + hémi-transverse (classe 3) — ne doit PAS partir en autre",
     "DIAGNOSTIC: Fracture de la colonne anterieure avec hemi-transverse "
     "posterieure.", False),
    ("C4 · PIÈGE : colonne antérieure ET paroi postérieure coexistent",
     "DIAGNOSTIC: Fracture de la colonne anterieure et de la paroi posterieure "
     "du cotyle droit.", False),  # in-périmètre (classe 1) doit primer
]


def _is_other(pred) -> bool:
    """« Autre » = le système n'affirme pas une classe 1/2/3. Deux formes :
    label == OUT_OF_SCOPE_ID (hors périmètre / non acétabulaire) OU label is
    None (acétabulaire mais confiance insuffisante)."""
    return pred.label not in config.LABEL_IDS


def main():
    clf = FractureClassifier()
    model_status = clf.ml_name or "AUCUN (mode règles seules)"

    print("=" * 78)
    print("DÉMONSTRATION QUALITATIVE — mécanisme « autre » (hors périmètre)")
    print("=" * 78)
    print(f"Modèle chargé : {model_status}")
    print("⚠ CR synthétiques — illustration, PAS une évaluation.\n")
    if clf.ml is None:
        print("Note : sans modèle, les cas B (contenu non acétabulaire) et les cas")
        print("nécessitant le modèle tomberont en « autre » par défaut. Pour la")
        print("démonstration complète, entraînez d'abord le modèle.\n")

    n_ok = 0
    for desc, cro, expect_other in CAS:
        pred = clf.predict(cro_text=cro)
        got_other = _is_other(pred)
        ok = (got_other == expect_other)
        n_ok += ok
        flag = "✓" if ok else "✗"

        if got_other:
            verdict = "AUTRE / à vérifier"
            via = {"regle": "règle hors-périmètre",
                   "abstention": "abstention (non acétabulaire)",
                   "incertain": "abstention (confiance insuffisante)",
                   "autre": "abstention (non acétabulaire)",
                   "aucune": "règles seules, pas de modèle"}.get(pred.source, pred.source)
        else:
            verdict = f"classe {pred.label} ({config.LABELS.get(pred.label,'?')[:32]})"
            via = {"regle": "règle in-périmètre", "ml": "modèle"}.get(pred.source, pred.source)

        print(f"[{flag}] {desc}")
        print(f"      -> {verdict}   [via {via}]")
        if pred.rule_term:
            print(f"         terme déclencheur : « {pred.rule_term} »")
        print()

    print("-" * 78)
    print(f"Comportement conforme à l'attendu sur {n_ok}/{len(CAS)} cas de démonstration.")
    print("(Attendu indicatif — ces CR sont synthétiques, pas une cohorte de test.)")
    print()
    print("Ce que ça illustre :")
    print("  • un type acétabulaire non couvert nommé explicitement -> « autre »")
    print("    via une règle dédiée (cas A) ;")
    print("  • une fracture d'un autre os -> « autre » via le garde-fou de")
    print("    pertinence acétabulaire (cas B) ;")
    print("  • un CR in-périmètre reste classé 1/2/3, même si un terme")
    print("    hors-périmètre coexiste (cas C, la règle in-périmètre prime).")


if __name__ == "__main__":
    main()
