"""
mcnemar_compare.py — Test de McNemar apparié entre configurations de modèle.

Répond à une exigence de publication : « le +0,14 de la data augmentation est-il
statistiquement significatif, ou dans le bruit ? » L'accuracy seule ne le dit
pas ; le test de McNemar, oui.

Référence méthodologique :
  Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing
  Supervised Classification Learning Algorithms. Neural Computation, 10(7),
  1895–1923. https://doi.org/10.1162/089976698300017197
  -> Ce papier montre que, parmi cinq tests, McNemar est le SEUL à avoir un
     taux d'erreur de type I acceptable quand chaque algorithme n'est
     entraîné qu'UNE fois (notre cas : une validation croisée par
     configuration). C'est donc le test adapté ici.

Principe. McNemar ne regarde QUE les cas où les deux modèles diffèrent :
  - n01 : A se trompe, B a raison
  - n10 : A a raison, B se trompe
Les cas où les deux ont raison (ou tort) n'apportent aucune information sur
*laquelle* des deux configurations est meilleure. Hypothèse nulle : n01 == n10
(les deux configurations ont le même taux d'erreur). On utilise la version à
correction de continuité, ou le test binomial exact quand n01+n10 est petit
(< 25), plus fiable sur peu de données appariées — c'est notre cas.

Ce script NE réentraîne rien. Il consomme les prédictions out-of-fold (une
par dossier, issues de la validation croisée) que les scripts d'entraînement
sauvegardent. Chaque configuration doit avoir été évaluée sur EXACTEMENT les
mêmes plis (même graille, garanti par config.RANDOM_STATE) pour que
l'appariement dossier-par-dossier soit valide.

Entrée attendue : un CSV par configuration dans data/processed/oof/, colonnes
  id, y_true, y_pred
généré par train_*.py via --save-oof (voir ce flag). Les 4 configurations
comparées :
  drbert, drbert_aug, doctobert, doctobert_aug
Comme les deux forks sont indépendants, lancez ce script depuis le dossier où
se trouvent les 4 CSV (copiez-y les CSV de l'autre fork si besoin).

Sortie (dans data/processed/mcnemar/) :
  - mcnemar_results.csv  : une ligne par paire comparée (statistique, p-value,
                            n01, n10, test utilisé, significativité à 0,05)
  - contingency_<A>_vs_<B>.csv : le tableau 2x2 apparié de chaque paire

Exécution :
  # 1) générer les prédictions out-of-fold des 4 configurations :
  #    (fork DrBERT)    python -m src.train_drbert --save-oof --oof-tag drbert
  #                     python -m src.train_drbert --no-augment --save-oof --oof-tag drbert_noaug
  #    (fork DoctoBERT) python -m src.train_doctobert --save-oof --oof-tag doctobert
  #                     python -m src.train_doctobert --no-augment --save-oof --oof-tag doctobert_noaug
  # 2) rassembler les 4 CSV dans data/processed/oof/ d'un même dossier, puis :
  python -m src.mcnemar_compare
"""
from __future__ import annotations

import itertools
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
OUT_DIR = config.DATA_PROCESSED / "mcnemar"

# Ordre de préférence des paires à comparer si les fichiers existent.
# On compare surtout : effet de la DA sur chaque modèle, et modèle vs modèle.
CANDIDATE_TAGS = ["drbert_noaug", "drbert", "doctobert_noaug", "doctobert"]


def _mcnemar(y_true, pred_a, pred_b):
    """Renvoie un dict avec n01, n10, la statistique, la p-value et le test
    employé (exact binomial si n01+n10 < 25, sinon chi2 corrigé)."""
    correct_a = pred_a == y_true
    correct_b = pred_b == y_true
    n01 = int(np.sum(~correct_a & correct_b))   # A faux, B juste
    n10 = int(np.sum(correct_a & ~correct_b))   # A juste, B faux
    n = n01 + n10

    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": float("nan"),
                "p_value": 1.0, "test": "aucun cas discordant"}

    if n < 25:
        # Test binomial exact (recommandé sur petits effectifs discordants).
        try:
            from scipy.stats import binomtest
            p = binomtest(min(n01, n10), n=n, p=0.5, alternative="two-sided").pvalue
        except ImportError:
            from scipy.stats import binom_test  # scipy < 1.7
            p = binom_test(min(n01, n10), n=n, p=0.5, alternative="two-sided")
        return {"n01": n01, "n10": n10, "statistic": float(min(n01, n10)),
                "p_value": float(p), "test": "binomial exact"}

    # Chi2 de McNemar avec correction de continuité.
    stat = (abs(n01 - n10) - 1) ** 2 / n
    from scipy.stats import chi2
    p = float(chi2.sf(stat, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(stat),
            "p_value": p, "test": "chi2 corrigé (continuité)"}


def _load_oof():
    if not OOF_DIR.exists():
        raise FileNotFoundError(
            f"{OOF_DIR} introuvable. Générez d'abord les prédictions out-of-fold "
            f"avec train_*.py --save-oof --oof-tag <nom> (voir en-tête du script)."
        )
    tables = {}
    for csv in sorted(OOF_DIR.glob("*.csv")):
        tag = csv.stem
        df = pd.read_csv(csv)
        missing = {"id", "y_true", "y_pred"} - set(df.columns)
        if missing:
            print(f"[ignoré] {csv.name} : colonnes manquantes {missing}")
            continue
        tables[tag] = df.set_index("id").sort_index()
    if len(tables) < 2:
        raise SystemExit(
            f"Il faut au moins 2 fichiers OOF valides dans {OOF_DIR} pour comparer. "
            f"Trouvés : {list(tables)}"
        )
    return tables


def main():
    tables = _load_oof()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tags = [t for t in CANDIDATE_TAGS if t in tables] + \
           [t for t in tables if t not in CANDIDATE_TAGS]

    rows = []
    for a, b in itertools.combinations(tags, 2):
        da, db = tables[a], tables[b]
        common = da.index.intersection(db.index)
        if len(common) == 0:
            print(f"[ignoré] {a} vs {b} : aucun id commun")
            continue
        # Vérif d'appariement : mêmes vérités terrain sur les id communs.
        yt_a = da.loc[common, "y_true"].values
        yt_b = db.loc[common, "y_true"].values
        if not np.array_equal(yt_a, yt_b):
            print(f"[attention] {a} vs {b} : y_true diffère sur les id communs — "
                  f"les deux configs n'ont pas vu les mêmes plis ? Comparaison ignorée.")
            continue

        y_true = yt_a
        pred_a = da.loc[common, "y_pred"].values
        pred_b = db.loc[common, "y_pred"].values
        res = _mcnemar(y_true, pred_a, pred_b)

        acc_a = float(np.mean(pred_a == y_true))
        acc_b = float(np.mean(pred_b == y_true))
        rows.append({
            "config_A": a, "config_B": b, "n_communs": len(common),
            "acc_A": round(acc_a, 3), "acc_B": round(acc_b, 3),
            "n01_A_faux_B_juste": res["n01"], "n10_A_juste_B_faux": res["n10"],
            "statistic": round(res["statistic"], 4) if res["statistic"] == res["statistic"] else None,
            "p_value": round(res["p_value"], 4),
            "significatif_0.05": res["p_value"] < 0.05,
            "test": res["test"],
        })

        # Tableau de contingence apparié 2x2, sauvegardé par paire.
        correct_a = pred_a == y_true
        correct_b = pred_b == y_true
        cont = pd.DataFrame(
            [[int(np.sum(correct_a & correct_b)), res["n10"]],
             [res["n01"], int(np.sum(~correct_a & ~correct_b))]],
            index=[f"{a} juste", f"{a} faux"],
            columns=[f"{b} juste", f"{b} faux"],
        )
        cont_path = OUT_DIR / f"contingency_{a}_vs_{b}.csv"
        cont.to_csv(cont_path)

    if not rows:
        raise SystemExit("Aucune paire comparable (voir messages ci-dessus).")

    results = pd.DataFrame(rows)
    res_path = OUT_DIR / "mcnemar_results.csv"
    results.to_csv(res_path, index=False)

    print("=" * 70)
    print("TEST DE McNEMAR — comparaisons appariées (Dietterich, 1998)")
    print("=" * 70)
    print(results.to_string(index=False))
    print()
    print(f"Résultats détaillés : {res_path}")
    print(f"Tableaux de contingence 2x2 : {OUT_DIR}/contingency_*.csv")
    print()
    print("Lecture : p < 0,05 => différence de taux d'erreur statistiquement")
    print("significative entre les deux configurations. n01/n10 = nombre de")
    print("dossiers où une seule des deux configs se trompe (les seuls")
    print("informatifs pour le test).")


if __name__ == "__main__":
    main()
