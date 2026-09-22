"""
train_doctobert.py — Fine-tuning DoctoBERT + Data Augmentation (DA).

⚠ À EXÉCUTER SUR VOTRE MACHINE (pas dans un environnement sans accès à
HuggingFace). Nécessite `transformers` et `torch`, et le téléchargement du
modèle pré-entraîné DoctoBERT depuis huggingface.co.

Place dans l'architecture : DoctoBERT est le filet de sécurité pour les cas que
les règles ne couvrent pas (~5/99). Sans augmentation, DoctoBERT SEUL obtient
~0,97 d accuracy / ~0,95 de macro-F1 en validation croisée. Ce script ajoute
la DA (src/augmentation.py) pour tenter d'améliorer ce chiffre, sans changer
le reste de l'architecture ni les hyperparamètres déjà choisis (10 époques,
lr 2e-5, etc.) : on isole l'effet de la DA pour pouvoir l'évaluer honnêtement.

RÈGLE ANTI-FUITE (la plus importante de ce fichier) : l'augmentation ne doit
JAMAIS voir les exemples de test d'un pli. Sinon, un modèle pourrait être
évalué sur une variante quasi identique à un exemple qu'il a vu à
l'entraînement -> métriques artificiellement gonflées. Ici, l'augmentation
est appliquée APRÈS le split StratifiedKFold, uniquement sur `texts[tr]` /
`labels[tr]` ; `texts[te]` / `labels[te]` restent TOUJOURS les originaux.

Le script fait deux choses :
  1. ÉVALUATION HONNÊTE : validation croisée stratifiée à 5 plis (réentraîne
     un modèle par pli, augmentation appliquée seulement sur le train de
     chaque pli) pour estimer la vraie performance, jamais un split unique.
  2. MODÈLE FINAL : réentraîne sur 100 % des données (augmentées, puisqu'il
     n'y a plus de test à protéger à ce stade) et le sauvegarde dans
     models/doctobert/ (format HuggingFace), prêt pour l'inférence par
     classifier.py.

Le texte d'entrée est la colonne `ml_text` du dataset, identique à
l'inférence (l'augmentation, elle, n'existe qu'à l'entraînement : en
inférence, classifier.py utilise toujours le texte réel, jamais augmenté).

Exécution (sur votre machine) :
    pip install "transformers>=4.40" "torch" "scikit-learn" "pandas" "accelerate"
    python -m src.train_doctobert
    # pour sauter la validation croisée (plus rapide) :
    python -m src.train_doctobert --no-cv
    # pour désactiver la DA et comparer à l'ancien comportement :
    python -m src.train_doctobert --no-augment
    # pour mesurer HONNÊTEMENT le gain de la DA (mêmes plis, avec vs sans) :
    python -m src.train_doctobert --compare-baseline
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
    from .augmentation import augment_dataset
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.augmentation import augment_dataset


# label (1,2,3) <-> index modèle (0,1,2)
LABEL2IDX = {lab: i for i, lab in enumerate(config.LABEL_IDS)}
IDX2LABEL = {i: lab for lab, i in LABEL2IDX.items()}


def _check_deps():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "transformers et torch sont requis pour DoctoBERT.\n"
            "  pip install \"transformers>=4.40\" torch accelerate\n"
            f"(détail : {e})"
        )


def _make_dataset(texts, labels, tokenizer):
    import torch
    from torch.utils.data import Dataset

    enc = tokenizer(
        list(texts), truncation=True, padding="max_length",
        max_length=config.MAX_LEN,
    )

    class DS(Dataset):
        def __len__(self):
            return len(labels)

        def __getitem__(self, i):
            item = {k: torch.tensor(v[i]) for k, v in enc.items()}
            item["labels"] = torch.tensor(int(labels[i]))
            return item

    return DS()


def _set_reproducible(seed: int) -> None:
    """Fixe toutes les sources d'aléatoire AVANT le chargement du modèle.

    `TrainingArguments(seed=...)` ne suffit PAS à lui seul : il ne prend effet
    qu'à la création du `Trainer`, donc APRÈS `from_pretrained`. Or c'est
    justement `from_pretrained` qui initialise aléatoirement la tête de
    classification (`classifier.dense` / `classifier.out_proj`, absente du
    checkpoint DoctoBERT pré-entraîné — d'où le message « newly initialized »
    dans les logs). Sans cet appel AVANT `from_pretrained`, deux exécutions
    avec exactement les mêmes données et hyperparamètres peuvent donner des
    scores différents : observé en pratique 0.667 puis 0.697 d'accuracy sur
    la MÊME configuration --no-augment. Ce n'est pas de l'instabilité du
    modèle, c'est un ordonnancement de la graine aléatoire trop tardif.
    """
    import random as _random

    import numpy as _np
    import torch as _torch
    from transformers import set_seed as _hf_set_seed

    _random.seed(seed)
    _np.random.seed(seed)
    _hf_set_seed(seed)                       # couvre random / numpy / torch (CPU + CUDA)
    _torch.backends.cudnn.deterministic = True
    _torch.backends.cudnn.benchmark = False


def _train_one(train_texts, train_labels, eval_texts=None, eval_labels=None,
               output_dir=None, seed=None):
    """Entraîne un DoctoBERT sur un pli ou sur tout le jeu. Renvoie (model, tokenizer).

    Ne sait rien de l'augmentation : reçoit déjà des textes/labels prêts à
    l'emploi (augmentés ou non selon l'appelant). Les poids de classe sont
    recalculés ICI à partir de train_labels -> si train_labels a été
    rééquilibré par la DA, la pondération de la perte s'adapte automatiquement
    (moins de poids sur la classe 3 puisqu'elle est déjà mieux représentée).

    `seed` : si None (défaut), utilise config.RANDOM_STATE — c'est le
    comportement reproductible normal (deux runs identiques -> mêmes poids).
    Si une valeur est passée (par multiseed_eval.py), elle fixe l'aléatoire, y
    compris l'initialisation de la tête de classification : c'est ce qui permet
    de mesurer la VRAIE variabilité inter-graines (init + ordre des données +
    split), pas seulement celle du découpage en plis.
    """
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments)

    effective_seed = config.RANDOM_STATE if seed is None else seed
    _set_reproducible(effective_seed)   # AVANT from_pretrained : voir _set_reproducible

    tokenizer = AutoTokenizer.from_pretrained(config.DOCTOBERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.DOCTOBERT_BASE, num_labels=len(config.LABEL_IDS),
    )

    train_ds = _make_dataset(train_texts, [LABEL2IDX[l] for l in train_labels], tokenizer)
    eval_ds = None
    if eval_texts is not None:
        eval_ds = _make_dataset(eval_texts, [LABEL2IDX[l] for l in eval_labels], tokenizer)

    # Poids de classe (déséquilibre résiduel après DA éventuelle) -> perte pondérée
    counts = np.bincount([LABEL2IDX[l] for l in train_labels], minlength=len(config.LABEL_IDS))
    weights = torch.tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(output_dir or (config.MODELS_DIR / "_doctobert_tmp")),
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=effective_seed,
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=train_ds,
                              eval_dataset=eval_ds)
    trainer.train()
    return model, tokenizer


def _predict(model, tokenizer, texts, return_proba: bool = False):
    """Prédit les classes des textes fournis.

    Par défaut, renvoie uniquement le tableau des classes prédites (comportement
    historique, utilisé partout dans le projet).

    Si return_proba=True, renvoie le couple (classes, probabilités) où
    `probabilités` est un tableau (n_textes, n_classes) issu du softmax des
    logits. Ces probabilités sont INDISPENSABLES au calcul de l'AUC, qui ne peut
    pas être obtenue à partir des seules classes prédites : l'AUC mesure la
    qualité du CLASSEMENT des scores, pas celle de la décision finale.

    L'ordre des colonnes de probabilités suit config.LABEL_IDS
    (colonne 0 -> classe 1, colonne 1 -> classe 2, etc.).
    """
    import torch

    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    preds, probas = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            batch = list(texts[i:i + 16])
            enc = tokenizer(batch, truncation=True, padding=True,
                            max_length=config.MAX_LEN, return_tensors="pt").to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            idx = logits.argmax(-1).cpu().numpy()
            preds.extend(IDX2LABEL[int(j)] for j in idx)
            probas.append(p)
    preds = np.array(preds)
    if not return_proba:
        return preds
    return preds, np.vstack(probas)


def _augment_summary(before_labels, after_labels) -> str:
    """Résumé lisible de l'effet de la DA sur la distribution des classes."""
    b = collections.Counter(int(l) for l in before_labels)
    a = collections.Counter(int(l) for l in after_labels)
    return " | ".join(
        f"classe {lab} : {b.get(lab, 0)} -> {a.get(lab, 0)}" for lab in config.LABEL_IDS
    )


def _maybe_augment(texts, labels, seed, tag=""):
    """Applique la DA. Ne consulte PAS config.AUGMENT_ENABLED : c'est à
    l'appelant de décider s'il faut augmenter (voir cross_validate/main), pas
    à cette fonction. Avant ce correctif, --compare-baseline pouvait
    silencieusement comparer « sans DA » à « sans DA » si AUGMENT_ENABLED
    valait False dans config.py — bug réel, corrigé ici : le master switch
    AUGMENT_ENABLED ne s'applique désormais qu'au run par défaut (sans flag),
    jamais à --compare-baseline qui doit rester une comparaison honnête.

    `tag` sert seulement à préfixer les logs (ex. "pli 1", "modèle final").
    """
    aug_texts, aug_labels, report = augment_dataset(
        list(texts), list(labels),
        multiplier=config.AUGMENT_MULTIPLIER,
        seed=seed,
        verify_rule_consistency=config.AUGMENT_VERIFY_RULE_CONSISTENCY,
        **config.AUGMENT_PARAMS,
    )
    prefix = f"  [{tag}] " if tag else "  "
    print(f"{prefix}Augmentation : {_augment_summary(labels, aug_labels)}")
    if report["rejected"]:
        print(f"{prefix}Variantes rejetées (incohérence règle) : {report['rejected']}")
    return aug_texts, aug_labels


def cross_validate(texts, labels, use_augment=True, return_extras=False):
    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    from sklearn.model_selection import StratifiedKFold

    import time

    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                          random_state=config.RANDOM_STATE)
    oof = np.zeros(len(labels), dtype=int)
    # Probabilités out-of-fold : nécessaires au calcul de l'AUC.
    oof_proba = np.zeros((len(labels), len(config.LABEL_IDS)), dtype=float)
    # Chronométrage : temps d'entraînement de chaque pli, pour comparer le coût
    # de fine-tuning des deux encodeurs à protocole identique.
    fold_times = []
    for k, (tr, te) in enumerate(skf.split(texts, labels), 1):
        print(f"\n--- Pli {k}/{config.CV_FOLDS} ---")
        train_texts, train_labels = texts[tr], labels[tr]

        # L'augmentation ne voit QUE le train de ce pli : jamais texts[te].
        # C'est ce qui garantit qu'aucune variante d'un exemple de test ne
        # se retrouve à l'entraînement (pas de fuite -> métriques honnêtes).
        if use_augment:
            train_texts, train_labels = _maybe_augment(
                train_texts, train_labels, seed=config.RANDOM_STATE + k, tag=f"pli {k}"
            )

        t0 = time.perf_counter()
        model, tok = _train_one(train_texts, train_labels)
        dt = time.perf_counter() - t0
        fold_times.append(dt)
        print(f"  Temps d'entraînement du pli : {dt:.1f} s")

        # texts[te] = toujours les originaux (jamais augmentés)
        oof[te], oof_proba[te] = _predict(model, tok, texts[te], return_proba=True)

    total_t = sum(fold_times)
    print(f"\n[Temps] Fine-tuning : {total_t:.1f} s au total "
          f"({total_t/len(fold_times):.1f} s par pli, {len(fold_times)} plis)")

    acc = (oof == labels).mean()
    mf1 = f1_score(labels, oof, average="macro")
    mode = "avec augmentation" if use_augment else "sans augmentation"
    print("\n" + "=" * 60)
    print(f"DoctoBERT ({config.DOCTOBERT_BASE}) — validation croisée {config.CV_FOLDS} plis [{mode}]")
    print("=" * 60)
    print(f"Accuracy ....... {acc:.3f}")
    print(f"Macro-F1 ....... {mf1:.3f}")
    print(classification_report(
        labels, oof, labels=config.LABEL_IDS,
        target_names=[f"{i}:{config.LABELS[i][:28]}" for i in config.LABEL_IDS],
        digits=3, zero_division=0,
    ))
    cm = confusion_matrix(labels, oof, labels=config.LABEL_IDS)

    print(f"\nMatrice de confusion — DoctoBERT seul [{mode}]")
    print("Lignes = vraies classes, colonnes = classes prédites")
    print(pd.DataFrame(
        cm,
        index=[f"Vrai {i}" for i in config.LABEL_IDS],
        columns=[f"Prédit {i}" for i in config.LABEL_IDS],
    ))
    if return_extras:
        return oof, oof_proba, total_t
    return oof


def _save_oof(ids, y_true, y_pred, tag, y_proba=None):
    """Sauvegarde les prédictions out-of-fold pour le test de McNemar.

    Une ligne par dossier (id, y_true, y_pred). Le tag nomme la configuration
    (ex. « doctobert », « doctobert_noaug ») ; mcnemar_compare.py apparie ensuite les
    configurations dossier par dossier. Voir src/mcnemar_compare.py.
    """
    out_dir = config.DATA_PROCESSED / "oof"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.csv"
    data = {"id": ids, "y_true": y_true, "y_pred": y_pred}
    # Colonnes de probabilité (proba_1, proba_2, ...) : requises pour l'AUC.
    # Absentes -> l'AUC ne peut pas être calculée (voir advanced_metrics.py).
    if y_proba is not None:
        for j, lab in enumerate(config.LABEL_IDS):
            data[f"proba_{lab}"] = np.asarray(y_proba)[:, j]
    pd.DataFrame(data).to_csv(path, index=False)
    extra = " + probabilités (AUC)" if y_proba is not None else ""
    print(f"Prédictions out-of-fold sauvegardées -> {path}  (McNemar{extra})")


def _print_comparison(labels, oof_base, oof_aug):
    """Tableau comparatif HONNÊTE : mêmes 5 plis, seule la DA change."""
    from sklearn.metrics import f1_score

    acc_b, f1_b = (oof_base == labels).mean(), f1_score(labels, oof_base, average="macro")
    acc_a, f1_a = (oof_aug == labels).mean(), f1_score(labels, oof_aug, average="macro")
    print("\n" + "=" * 60)
    print("COMPARAISON — mêmes 5 plis de validation croisée, seule la DA change")
    print("=" * 60)
    print(f"{'':22s}{'Accuracy':>12s}{'Macro-F1':>12s}")
    print(f"{'Sans augmentation':22s}{acc_b:12.3f}{f1_b:12.3f}")
    print(f"{'Avec augmentation':22s}{acc_a:12.3f}{f1_a:12.3f}")
    print(f"{'Delta':22s}{acc_a - acc_b:+12.3f}{f1_a - f1_b:+12.3f}")
    if acc_a <= acc_b and f1_a <= f1_b:
        print("\n⚠ La DA n'améliore pas le modèle sur cette mesure : ajustez "
              "config.AUGMENT_MULTIPLIER / AUGMENT_PARAMS, ou conservez "
              "--no-augment. Ne pas forcer une technique qui ne marche pas.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cv", action="store_true",
                        help="sauter la validation croisée (entraîne seulement le modèle final)")
    parser.add_argument("--no-augment", action="store_true",
                        help="désactive la Data Augmentation (comportement d'origine)")
    parser.add_argument("--compare-baseline", action="store_true",
                        help="lance la validation croisée AVEC puis SANS augmentation, "
                             "sur les mêmes plis, et affiche le delta honnête")
    parser.add_argument("--save-oof", action="store_true",
                        help="sauvegarde les prédictions out-of-fold (pour le test de McNemar)")
    parser.add_argument("--oof-tag", default=None,
                        help="nom de la configuration pour le fichier OOF "
                             "(défaut : 'doctobert' ou 'doctobert_noaug' selon --no-augment)")
    cli = parser.parse_args()

    _check_deps()
    if not config.DATASET_CSV.exists():
        raise FileNotFoundError(
            f"{config.DATASET_CSV} introuvable. Lancez d'abord : python -m src.build_dataset"
        )
    df = pd.read_csv(config.DATASET_CSV).fillna({"ml_text": ""})
    texts = df["ml_text"].astype(str).values
    labels = df["label"].astype(int).values
    ids = df["id"].values if "id" in df.columns else np.arange(len(labels))

    if not cli.no_cv:
        if cli.compare_baseline:
            print("\n" + "#" * 60)
            print("# 1/2 — VALIDATION CROISÉE SANS augmentation (référence)")
            print("#" * 60)
            oof_base, proba_base, temps_base = cross_validate(
                texts, labels, use_augment=False, return_extras=True)

            print("\n" + "#" * 60)
            print("# 2/2 — VALIDATION CROISÉE AVEC augmentation")
            print("#" * 60)
            oof_aug, proba_aug, temps_aug = cross_validate(
                texts, labels, use_augment=True, return_extras=True)

            _print_comparison(labels, oof_base, oof_aug)
            if cli.save_oof:
                _save_oof(ids, labels, oof_base, "doctobert_noaug", y_proba=proba_base)
                _save_oof(ids, labels, oof_aug, "doctobert", y_proba=proba_aug)
        else:
            # AUGMENT_ENABLED n'agit QUE sur ce run par défaut (sans flag) :
            # --compare-baseline ci-dessus l'ignore volontairement pour rester
            # une comparaison honnête quelle que soit la valeur du switch.
            use_aug = config.AUGMENT_ENABLED and not cli.no_augment
            oof, oof_proba_run, temps_run = cross_validate(
                texts, labels, use_augment=use_aug, return_extras=True)
            if cli.save_oof:
                tag = cli.oof_tag or ("doctobert" if use_aug else "doctobert_noaug")
                _save_oof(ids, labels, oof, tag, y_proba=oof_proba_run)

    print("\nEntraînement du modèle final sur 100 % des données...")
    final_texts, final_labels = list(texts), list(labels)
    if config.AUGMENT_ENABLED and not cli.no_augment:
        # Pas de fuite possible ici : il n'y a plus de jeu de test à protéger
        # à ce stade (c'est le modèle définitif, celui qui sera déployé).
        final_texts, final_labels = _maybe_augment(
            final_texts, final_labels, seed=config.RANDOM_STATE, tag="modèle final"
        )

    model, tokenizer = _train_one(
        np.array(final_texts), np.array(final_labels), output_dir=config.DOCTOBERT_DIR
    )
    config.DOCTOBERT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.DOCTOBERT_DIR)
    tokenizer.save_pretrained(config.DOCTOBERT_DIR)
    # mémorise la correspondance label<->index pour l'inférence
    import json
    (config.DOCTOBERT_DIR / "label_map.json").write_text(
        json.dumps({"idx2label": {str(i): l for i, l in IDX2LABEL.items()}},
                   ensure_ascii=False, indent=2)
    )
    print(f"Modèle DoctoBERT sauvegardé -> {config.DOCTOBERT_DIR}")


if __name__ == "__main__":
    main()
