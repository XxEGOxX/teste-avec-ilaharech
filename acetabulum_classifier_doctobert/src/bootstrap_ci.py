"""
bootstrap_ci.py — Intervalles de confiance par bootstrap sur les métriques.

Pourquoi. Une accuracy de 0,99 mesurée sur seulement 99 dossiers n'est pas un
nombre exact : c'est une estimation entourée d'incertitude. Publier « 0,990 »
sans intervalle de confiance n'est pas interprétable — un relecteur ne peut
pas savoir si l'écart avec 0,96 est réel ou dû au petit effectif. Le bootstrap
quantifie cette incertitude sans hypothèse de distribution.

Méthode (bootstrap par percentiles, Efron & Tibshirani 1993). On rééchantillonne
les 99 dossiers AVEC remise, B fois (défaut 10 000). Pour chaque rééchantillon,
on recalcule accuracy et macro-F1. Les percentiles 2,5 % et 97,5 % de ces B
valeurs donnent l'intervalle de confiance à 95 %. C'est un rééchantillonnage
des PRÉDICTIONS déjà obtenues (fichiers out-of-fold), pas un réentraînement :
rapide (quelques secondes) et cohérent avec les chiffres de evaluate_fusion /
train_*.

⚠ Le bootstrap capture l'incertitude d'ÉCHANTILLONNAGE (petit n), pas
l'incertitude d'entraînement (variabilité selon la graine). Pour cette
dernière, voir multiseed_eval.py. Les deux sont complémentaires : idéalement,
rapporter l'IC bootstrap ET l'écart-type inter-graines.

Entrée : les CSV out-of-fold dans data/processed/oof/ (colonnes id, y_true,
y_pred), produits par train_*.py --save-oof. Chaque fichier = une configuration.

Sortie : data/processed/bootstrap/bootstrap_ci.csv — une ligne par
configuration avec accuracy et macro-F1, chacune assortie de son IC à 95 %.

Exécution :
    python -m src.bootstrap_ci
    python -m src.bootstrap_ci --n-boot 20000 --alpha 0.05
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
OUT_DIR = config.DATA_PROCESSED / "bootstrap"


def _macro_f1(y_true, y_pred, classes):
    """Macro-F1 calculé à la main (pas de dépendance sklearn ici) : moyenne
    non pondérée des F1 par classe. Une classe absente du rééchantillon a un
    F1 de 0 (support nul) — cohérent avec zero_division=0."""
    f1s = []
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f1s))


def _bootstrap_metric(y_true, y_pred, metric_fn, n_boot, alpha, rng):
    """IC bootstrap par percentiles pour une métrique donnée.

    Renvoie (valeur_observée, borne_basse, borne_haute)."""
    n = len(y_true)
    observed = metric_fn(y_true, y_pred)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)          # tirage avec remise
        stats[b] = metric_fn(y_true[idx], y_pred[idx])
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return observed, lo, hi


def _accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def bootstrap_file(df, classes, n_boot, alpha, seed):
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    rng = np.random.default_rng(seed)

    acc, acc_lo, acc_hi = _bootstrap_metric(
        y_true, y_pred, _accuracy, n_boot, alpha, rng)
    # nouveau rng réamorcé pour que l'IC macro-F1 ne dépende pas du tirage acc
    rng = np.random.default_rng(seed + 1)
    f1, f1_lo, f1_hi = _bootstrap_metric(
        y_true, y_pred, lambda yt, yp: _macro_f1(yt, yp, classes),
        n_boot, alpha, rng)
    return {
        "n": len(y_true),
        "accuracy": round(acc, 3),
        "accuracy_CI_bas": round(acc_lo, 3),
        "accuracy_CI_haut": round(acc_hi, 3),
        "macroF1": round(f1, 3),
        "macroF1_CI_bas": round(f1_lo, 3),
        "macroF1_CI_haut": round(f1_hi, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="nombre de rééchantillons bootstrap (défaut 1000)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="niveau (défaut 0.05 -> IC à 95%%)")
    cli = parser.parse_args()

    if not OOF_DIR.exists():
        raise SystemExit(
            f"{OOF_DIR} introuvable. Générez d'abord les prédictions out-of-fold "
            f"avec train_*.py --save-oof (voir en-tête)."
        )
    files = sorted(OOF_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"Aucun fichier OOF dans {OOF_DIR}.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    classes = config.LABEL_IDS

    rows = []
    for csv in files:
        df = pd.read_csv(csv)
        if not {"y_true", "y_pred"} <= set(df.columns):
            print(f"[ignoré] {csv.name} : colonnes y_true/y_pred manquantes")
            continue
        res = bootstrap_file(df, classes, cli.n_boot, cli.alpha,
                             seed=config.RANDOM_STATE)
        res = {"config": csv.stem, **res}
        rows.append(res)

    if not rows:
        raise SystemExit("Aucun fichier OOF exploitable.")

    results = pd.DataFrame(rows)
    out = OUT_DIR / "bootstrap_ci.csv"
    results.to_csv(out, index=False)

    conf = int(round((1 - cli.alpha) * 100))
    print("=" * 72)
    print(f"INTERVALLES DE CONFIANCE À {conf}% PAR BOOTSTRAP "
          f"({cli.n_boot} rééchantillons)")
    print("=" * 72)
    for _, r in results.iterrows():
        print(f"\n{r['config']}  (n={r['n']})")
        print(f"  Accuracy : {r['accuracy']:.3f}  "
              f"[IC{conf}% {r['accuracy_CI_bas']:.3f} – {r['accuracy_CI_haut']:.3f}]")
        print(f"  Macro-F1 : {r['macroF1']:.3f}  "
              f"[IC{conf}% {r['macroF1_CI_bas']:.3f} – {r['macroF1_CI_haut']:.3f}]")
    print(f"\nDétails : {out}")
    print("\nLecture : l'IC indique la plage plausible de la vraie performance,")
    print("compte tenu du petit effectif. Deux configurations dont les IC se")
    print("recouvrent largement ne sont pas nettement distinguables sur ce jeu.")


if __name__ == "__main__":
    main()
