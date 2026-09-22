"""
evaluate_fusion.py — Évaluation honnête du système COMPLET (règles + DoctoBERT).

`evaluate_rules.py` mesure les règles seules. `train_doctobert.py` mesure DoctoBERT
seul (hors-fold, out-of-fold). Aucun des deux ne mesure ce qu'un utilisateur
voit RÉELLEMENT : la fusion implémentée dans `classifier.py` (la règle prime
si elle se déclenche ; sinon DoctoBERT, s'il est assez confiant ; sinon
« indéterminé »). Ce script comble ce manque.

Comment, sans dupliquer la logique de fusion (source d'incohérences si elle
divergeait de la production) :
  1. Mêmes 5 plis que train_doctobert.py (même graine).
  2. Même entraînement par pli (_train_one, _set_reproducible, augmentation
     optionnelle) — réutilisés directement depuis train_doctobert.py.
  3. Le modèle fraîchement entraîné du pli est enveloppé dans un petit
     adaptateur (_FoldBackend) qui lui donne l'interface attendue par
     `classifier.FractureClassifier` (.name, .predict(text)). On construit
     alors un FractureClassifier RÉEL avec ce modèle, et on appelle SA
     méthode .predict(cro_text, crr_text) — c'est very exactement le code
     qui tourne en production (app/streamlit_app.py, predict_file.py).

Coût : comme il réentraîne un modèle par pli (comme train_doctobert.py), la
durée est comparable — plusieurs minutes sur GPU.

Les cas « indéterminé » (ni règle ni ML assez confiant) sont comptés comme
des erreurs dans l'accuracy/macro-F1 globales (ils nécessiteraient une
relecture manuelle) — jamais silencieusement ignorés. La matrice de
confusion a une colonne dédiée « Indéterminé » pour que ça reste visible.

Exécution :
    python -m src.evaluate_fusion                # avec DA (comportement par défaut de train_doctobert)
    python -m src.evaluate_fusion --no-augment    # sans DA, pour comparer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
    from .classifier import FractureClassifier
    from .train_doctobert import IDX2LABEL, _check_deps, _maybe_augment, _train_one
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.classifier import FractureClassifier
    from src.train_doctobert import IDX2LABEL, _check_deps, _maybe_augment, _train_one


class _FoldBackend:
    """Adaptateur : donne au modèle fraîchement entraîné d'un pli la même
    interface que `_DoctobertBackend` (classifier.py) — .name + .predict(text)
    -> (label, confiance, proba_dict) — pour pouvoir construire un VRAI
    FractureClassifier et réutiliser sa logique de fusion telle quelle."""

    name = "DoctoBERT"

    def __init__(self, model, tokenizer):
        import torch

        self._torch = torch
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def predict(self, text: str):
        enc = self.tokenizer([text], truncation=True, padding=True,
                             max_length=config.MAX_LEN, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            logits = self.model(**enc).logits
            proba = self._torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(proba.argmax())
        label = IDX2LABEL[idx]
        return label, float(proba[idx]), {IDX2LABEL[i]: float(p) for i, p in enumerate(proba)}


def _confusion_with_undetermined(true_labels, pred_labels) -> pd.DataFrame:
    """pred_labels peut contenir 0 (= indéterminé) en plus de 1/2/3."""
    mat = pd.DataFrame(
        0,
        index=[f"Vrai {i}" for i in config.LABEL_IDS],
        columns=[f"Prédit {i}" for i in config.LABEL_IDS] + ["Indéterminé"],
    )
    for t, p in zip(true_labels, pred_labels):
        col = f"Prédit {p}" if p in config.LABEL_IDS else "Indéterminé"
        mat.loc[f"Vrai {t}", col] += 1
    return mat


def _prf_report(mat: pd.DataFrame) -> pd.DataFrame:
    """Précision/rappel/F1 par classe, calculés explicitement à partir de la
    matrice de confusion (un « indéterminé » compte comme un faux négatif
    pour sa vraie classe — jamais retiré du dénominateur)."""
    rows = []
    for i in config.LABEL_IDS:
        tp = mat.loc[f"Vrai {i}", f"Prédit {i}"]
        support = mat.loc[f"Vrai {i}"].sum()
        fn = support - tp
        fp = mat[f"Prédit {i}"].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"classe": i, "precision": round(prec, 3), "recall": round(rec, 3),
                     "f1": round(f1, 3), "support": int(support)})
    return pd.DataFrame(rows).set_index("classe")


def cross_validate_fusion(df: pd.DataFrame, use_augment: bool = True):
    from sklearn.model_selection import StratifiedKFold

    texts = df["ml_text"].astype(str).values
    labels = df["label"].astype(int).values
    cro_texts = df["cro_text"].fillna("").astype(str).values
    crr_texts = df["crr_text"].fillna("").astype(str).values

    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_STATE)

    oof_fusion = np.zeros(len(labels), dtype=int)   # 0 = indéterminé
    oof_ml = np.zeros(len(labels), dtype=int)       # 0 = indéterminé / vide
    oof_source = np.array([""] * len(labels), dtype=object)

    for k, (tr, te) in enumerate(skf.split(texts, labels), 1):
        print(f"\n--- Pli {k}/{config.CV_FOLDS} ---")
        train_texts, train_labels = texts[tr], labels[tr]
        if use_augment:
            train_texts, train_labels = _maybe_augment(
                train_texts, train_labels, seed=config.RANDOM_STATE + k, tag=f"pli {k}"
            )
        model, tokenizer = _train_one(train_texts, train_labels)
        clf = FractureClassifier(ml_backend=_FoldBackend(model, tokenizer))

        for i in te:
            pred = clf.predict(cro_text=cro_texts[i], crr_text=crr_texts[i])
            oof_fusion[i] = pred.label if pred.label is not None else 0
            oof_ml[i] = pred.ml_label if pred.ml_label is not None else 0
            oof_source[i] = pred.source

        del model, tokenizer, clf
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    return oof_ml, oof_fusion, oof_source


def _print_block(title: str, true_labels, pred_labels_with_zero):
    mat = _confusion_with_undetermined(true_labels, pred_labels_with_zero)
    n = len(true_labels)
    n_undet = int((pred_labels_with_zero == 0).sum())
    n_correct = int(sum(t == p for t, p in zip(true_labels, pred_labels_with_zero)))
    acc = n_correct / n
    prf = _prf_report(mat)
    macro_f1 = prf["f1"].mean()

    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print(f"Couverture (classe affirmée) ... {n - n_undet}/{n} ({100 * (n - n_undet) / n:.1f}%)")
    if n - n_undet:
        print(f"Précision (sur affirmés) ....... {n_correct}/{n - n_undet} "
              f"({100 * n_correct / (n - n_undet):.1f}%)")
    print(f"Accuracy (indéterminé = faux) .. {acc:.3f}")
    print(f"Macro-F1 (indéterminé = faux) .. {macro_f1:.3f}")
    print()
    print(prf)
    print()
    print("Matrice de confusion (lignes = vraie classe) :")
    print(mat)
    return acc, macro_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-augment", action="store_true",
                        help="désactive la Data Augmentation pour l'entraînement DoctoBERT de chaque pli")
    cli = parser.parse_args()

    _check_deps()
    if not config.DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{config.DATASET_CSV} introuvable. Lancez d'abord : python -m src.build_dataset"
        )
    df = pd.read_csv(config.DATASET_CSV).fillna({"ml_text": "", "cro_text": "", "crr_text": ""})
    labels = df["label"].astype(int).values
    rule_pred_raw = pd.to_numeric(df["rule_pred"], errors="coerce")
    oof_rules = rule_pred_raw.fillna(0).astype(int).values

    oof_ml, oof_fusion, oof_source = cross_validate_fusion(df, use_augment=not cli.no_augment)

    print("\n" + "#" * 60)
    print("# RÉCAPITULATIF — mêmes 5 plis pour les 3 lignes ci-dessous")
    print("#" * 60)
    acc_r, f1_r = _print_block("RÈGLES SEULES (rappel — voir aussi evaluate_rules.py)",
                                labels, oof_rules)
    acc_m, f1_m = _print_block(f"DoctoBERT SEUL ({'avec' if not cli.no_augment else 'sans'} augmentation)",
                                labels, oof_ml)
    acc_f, f1_f = _print_block("SYSTÈME FUSIONNÉ (règles + DoctoBERT) — ce que voit l'utilisateur",
                                labels, oof_fusion)

    print("\n" + "=" * 60)
    print("COMPARATIF FINAL")
    print("=" * 60)
    print(f"{'':22s}{'Accuracy':>12s}{'Macro-F1':>12s}")
    print(f"{'Règles seules':22s}{acc_r:12.3f}{f1_r:12.3f}")
    print(f"{'DoctoBERT seul':22s}{acc_m:12.3f}{f1_m:12.3f}")
    print(f"{'Fusion (production)':22s}{acc_f:12.3f}{f1_f:12.3f}")

    print("\nRépartition des décisions de la fusion, par source :")
    print(pd.Series(oof_source).value_counts())


if __name__ == "__main__":
    main()
