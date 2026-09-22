"""
evaluate_rules.py — Rapport de couverture/précision de la couche de règles.

À lancer APRÈS build_dataset.py. Mesure, sur les dossiers annotés :
  - la couverture  : part des dossiers où une règle se déclenche ;
  - la précision   : part des décisions correctes parmi les dossiers couverts ;
  - le détail par classe et la liste des cas non couverts (délégués au ML).

Une règle qui se déclenche doit être (quasi) toujours correcte : c'est la
condition pour qu'elle ait priorité sur le ML dans la fusion finale.

Exécution :
    python -m src.evaluate_rules
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config


def main():
    if not config.DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{config.DATASET_CSV} introuvable. Lancez d'abord : python -m src.build_dataset"
        )
    df = pd.read_csv(config.DATASET_CSV)
    n = len(df)

    covered = df[df["rule_pred"].notna() & (df["rule_pred"].astype(str) != "")].copy()
    covered["rule_pred"] = covered["rule_pred"].astype(float).astype(int)
    correct = (covered["rule_pred"] == covered["label"]).sum()
    n_cov = len(covered)

    print("=" * 60)
    print("ÉVALUATION DE LA COUCHE DE RÈGLES")
    print("=" * 60)
    print(f"Dossiers ....................... {n}")
    print(f"Couverture ..................... {n_cov}/{n} ({100*n_cov/n:.1f}%)")
    if n_cov:
        print(f"Précision (sur couverts) ....... {correct}/{n_cov} ({100*correct/n_cov:.1f}%)")

    # Détail par classe réelle
    print("\nCouverture par classe réelle :")
    for lab in config.LABEL_IDS:
        sub = df[df["label"] == lab]
        sub_cov = covered[covered["label"] == lab]
        sub_ok = (sub_cov["rule_pred"] == lab).sum() if len(sub_cov) else 0
        print(
            f"  classe {lab} : couverts {len(sub_cov)}/{len(sub)}"
            f"  | corrects {sub_ok}/{len(sub_cov) if len(sub_cov) else 0}"
        )

    # Déclencheurs utilisés
    if n_cov:
        print("\nZones de déclenchement :")
        for zone, cnt in covered["rule_zone"].value_counts().items():
            print(f"  {zone} : {cnt}")

    # Erreurs éventuelles (doivent rester nulles)
    errors = covered[covered["rule_pred"] != covered["label"]]
    if len(errors):
        print(f"\n⚠ {len(errors)} ERREUR(S) de règle (à inspecter) :")
        for _, r in errors.iterrows():
            print(
                f"  ID {r['id']} : vrai={r['label']} prédit={r['rule_pred']} "
                f"zone={r['rule_zone']} terme={r['rule_term']}"
            )
    else:
        print("\n✓ Aucune erreur parmi les cas couverts.")

    # Cas non couverts -> délégués au ML
    uncov = df[~(df["rule_pred"].notna() & (df["rule_pred"].astype(str) != ""))]
    print(f"\nCas non couverts (délégués au ML) : {len(uncov)}")
    for _, r in uncov.iterrows():
        print(f"  ID {r['id']} (classe réelle {r['label']})")


if __name__ == "__main__":
    main()
