"""
advanced_metrics.py — AUC, sensibilité et spécificité par classe.

POURQUOI CES TROIS MÉTRIQUES
-----------------------------
L'exactitude et le macro-F1 résument la décision finale du modèle. Ces trois
métriques apportent un éclairage complémentaire, attendu en publication
médicale :

  • SENSIBILITÉ (= rappel) : parmi les patients qui ont réellement cette
    fracture, quelle proportion le système détecte-t-il ? Une sensibilité
    faible signifie que le système « rate » des cas.

  • SPÉCIFICITÉ : parmi les patients qui n'ont PAS cette fracture, quelle
    proportion le système écarte-t-il correctement ? Une spécificité faible
    signifie que le système « sur-diagnostique » cette classe.

  • AUC (aire sous la courbe ROC) : contrairement aux deux précédentes, elle
    ne dépend PAS du seuil de décision. Elle mesure la qualité du CLASSEMENT
    des probabilités : le modèle attribue-t-il des scores plus élevés aux vrais
    cas qu'aux autres ? Une AUC de 0,5 correspond au hasard, 1,0 à une
    séparation parfaite.

CALCUL EN MULTI-CLASSES
-----------------------
Ces trois métriques sont définies pour un problème binaire. En multi-classes,
on applique le schéma « un contre tous » (one-vs-rest) : pour chaque classe,
on considère « cette classe » contre « toutes les autres réunies », puis on
moyenne (macro-moyenne, chaque classe comptant autant, cohérent avec le
macro-F1 déjà employé dans le projet).

⚠ PRÉREQUIS POUR L'AUC
L'AUC exige les PROBABILITÉS prédites, pas seulement les classes. Les fichiers
out-of-fold doivent donc contenir des colonnes proba_1, proba_2, ... produites
par train_*.py après la mise à jour de _predict/_save_oof. Les anciens fichiers
OOF (sans ces colonnes) permettent le calcul de la sensibilité et de la
spécificité, mais pas celui de l'AUC : le script le signale explicitement
plutôt que d'inventer une valeur.

UTILISATION
-----------
    python -m src.advanced_metrics
    python -m src.advanced_metrics --oof doctobert

Sortie : affichage détaillé + data/processed/metrics/advanced_metrics.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config


OOF_DIR = config.DATA_PROCESSED / "oof"
OUT_DIR = config.DATA_PROCESSED / "metrics"


def sensibilite_specificite(y_true, y_pred, classe):
    """Sensibilité et spécificité d'une classe, en « un contre tous ».

    VP : vrais positifs   — cas de `classe` correctement identifiés
    FN : faux négatifs    — cas de `classe` manqués
    VN : vrais négatifs   — cas d'autres classes correctement écartés
    FP : faux positifs    — cas d'autres classes attribués à tort à `classe`
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    vp = int(np.sum((y_true == classe) & (y_pred == classe)))
    fn = int(np.sum((y_true == classe) & (y_pred != classe)))
    vn = int(np.sum((y_true != classe) & (y_pred != classe)))
    fp = int(np.sum((y_true != classe) & (y_pred == classe)))

    sens = vp / (vp + fn) if (vp + fn) > 0 else float("nan")
    spec = vn / (vn + fp) if (vn + fp) > 0 else float("nan")
    return sens, spec, dict(VP=vp, FN=fn, VN=vn, FP=fp)


def auc_une_classe(y_true, proba_classe, classe):
    """AUC « un contre tous » pour une classe, calculée à la main.

    Méthode : l'AUC équivaut à la statistique de Mann-Whitney U normalisée,
    c'est-à-dire la probabilité qu'un cas positif tiré au hasard reçoive un
    score plus élevé qu'un cas négatif tiré au hasard. On la calcule via les
    rangs, ce qui gère correctement les scores ex aequo.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(proba_classe, dtype=float)

    positifs = (y_true == classe)
    n_pos = int(positifs.sum())
    n_neg = int((~positifs).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # rangs moyens (gestion des ex aequo)
    ordre = np.argsort(scores, kind="mergesort")
    rangs = np.empty(len(scores), dtype=float)
    rangs[ordre] = np.arange(1, len(scores) + 1)
    tries = scores[ordre]
    i = 0
    while i < len(tries):
        j = i
        while j + 1 < len(tries) and tries[j + 1] == tries[i]:
            j += 1
        if j > i:
            rangs[ordre[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    somme_rangs_pos = rangs[positifs].sum()
    u = somme_rangs_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def analyser(df: pd.DataFrame, nom: str):
    """Calcule toutes les métriques pour un fichier out-of-fold."""
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()

    colonnes_proba = [f"proba_{c}" for c in config.LABEL_IDS]
    a_proba = all(c in df.columns for c in colonnes_proba)

    lignes = []
    for c in config.LABEL_IDS:
        sens, spec, cm = sensibilite_specificite(y_true, y_pred, c)
        auc = (auc_une_classe(y_true, df[f"proba_{c}"].to_numpy(), c)
               if a_proba else float("nan"))
        lignes.append({
            "config": nom,
            "classe": c,
            "libelle": config.LABELS.get(c, str(c))[:30],
            "n": int(np.sum(y_true == c)),
            "sensibilite": round(sens, 3) if sens == sens else None,
            "specificite": round(spec, 3) if spec == spec else None,
            "AUC": round(auc, 3) if auc == auc else None,
            **cm,
        })

    res = pd.DataFrame(lignes)

    print("\n" + "=" * 74)
    print(f"MÉTRIQUES AVANCÉES — {nom}")
    print("=" * 74)
    if not a_proba:
        print("⚠ Colonnes de probabilité absentes de ce fichier : AUC non calculable.")
        print("  Relancez train_*.py --save-oof avec la version à jour du code.\n")

    entete = f"{'Classe':<34}{'n':>4}{'Sens.':>9}{'Spéc.':>9}{'AUC':>9}"
    print(entete)
    print("-" * 74)
    for _, r in res.iterrows():
        auc_txt = f"{r['AUC']:.3f}" if r["AUC"] is not None else "   n/a"
        print(f"{r['classe']}:{r['libelle']:<31}{r['n']:>4}"
              f"{r['sensibilite']:>9.3f}{r['specificite']:>9.3f}{auc_txt:>9}")
    print("-" * 74)

    # macro-moyennes (chaque classe pèse autant)
    macro_sens = res["sensibilite"].astype(float).mean()
    macro_spec = res["specificite"].astype(float).mean()
    print(f"{'MACRO-MOYENNE':<38}{macro_sens:>9.3f}{macro_spec:>9.3f}", end="")
    if a_proba:
        macro_auc = res["AUC"].astype(float).mean()
        print(f"{macro_auc:>9.3f}")
    else:
        print(f"{'   n/a':>9}")

    print("\nLecture :")
    print("  Sensibilité — parmi les cas réels de la classe, proportion détectée.")
    print("  Spécificité — parmi les cas des autres classes, proportion écartée.")
    print("  AUC         — qualité du classement des probabilités, indépendante")
    print("                du seuil de décision (0,5 = hasard, 1,0 = parfait).")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", default=None,
                        help="nom d'une configuration précise (ex. doctobert). "
                             "Par défaut : toutes celles présentes.")
    cli = parser.parse_args()

    if not OOF_DIR.exists():
        raise SystemExit(
            f"{OOF_DIR} introuvable. Lancez d'abord train_*.py --save-oof.")

    fichiers = ([OOF_DIR / f"{cli.oof}.csv"] if cli.oof
                else sorted(OOF_DIR.glob("*.csv")))
    fichiers = [f for f in fichiers if f.exists()]
    if not fichiers:
        raise SystemExit(f"Aucun fichier out-of-fold trouvé dans {OOF_DIR}.")

    toutes = []
    for f in fichiers:
        df = pd.read_csv(f)
        if not {"y_true", "y_pred"} <= set(df.columns):
            print(f"[ignoré] {f.name} : colonnes y_true/y_pred manquantes")
            continue
        toutes.append(analyser(df, f.stem))

    if toutes:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        sortie = OUT_DIR / "advanced_metrics.csv"
        pd.concat(toutes, ignore_index=True).to_csv(sortie, index=False)
        print(f"\nDétails enregistrés -> {sortie}")


if __name__ == "__main__":
    main()
