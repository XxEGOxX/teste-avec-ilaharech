"""
multiseed_eval.py — Robustesse aux graines : moyenne ± écart-type sur N graines.

Pourquoi. Le fine-tuning de BERT/RoBERTa sur petits jeux est notoirement
instable : la même configuration, réentraînée avec une graine différente, peut
donner un score sensiblement différent (Dodge et al. 2020 ; Mosbach et al.
2021). Rapporter un seul chiffre issu d'une seule graine peut donc induire en
erreur — soit trop optimiste, soit trop pessimiste selon le tirage. La pratique
attendue en publication (et suivie par l'article DoctoBERT lui-même, qui moyenne
sur 5 graines) est de rapporter moyenne ± écart-type sur PLUSIEURS graines.

Différence avec bootstrap_ci.py — les deux mesurent des incertitudes DISTINCTES
et complémentaires :
  - bootstrap_ci : incertitude d'ÉCHANTILLONNAGE (« et si on avait d'autres
                   patients ? »), à modèle figé. Rapide, sur prédictions.
  - multiseed    : incertitude d'ENTRAÎNEMENT (« et si on relançait
                   l'entraînement ? »), à jeu de données figé. Coûteux,
                   réentraîne tout.
Un article solide rapporte les deux.

Ce que fait ce script. Pour chaque graine de --seeds, il relance la validation
croisée complète (réentraînement par pli) et enregistre accuracy + macro-F1
du modèle SEUL. Il agrège ensuite en moyenne, écart-type, min, max. Comme il
réentraîne autant de fois qu'il y a de graines × plis, c'est LONG (plusieurs
dizaines de minutes sur GPU pour 5 graines). À lancer quand la configuration
est figée, pour produire le chiffre final de l'article.

⚠ Réutilise directement la machinerie d'entraînement du fork courant
(train_drbert / train_doctobert) : mêmes _train_one, _set_reproducible,
augmentation. La seule chose qui varie d'une exécution à l'autre est la graine
passée à StratifiedKFold ET à l'entraînement.

Sortie : data/processed/multiseed/multiseed_<config>.csv (une ligne par graine)
et un récapitulatif agrégé imprimé + multiseed_summary.csv.

Exécution (fork DrBERT) :
    python -m src.multiseed_eval --seeds 42 43 44 45 46            # avec DA (défaut)
    python -m src.multiseed_eval --seeds 42 43 44 45 46 --no-augment
(fork DoctoBERT : idem avec python -m src.multiseed_eval)
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config


def _load_train_module():
    """Importe le module d'entraînement du fork courant (drbert ou doctobert),
    sans coder en dur lequel : on tente les deux."""
    for name in ("train_drbert", "train_doctobert"):
        try:
            return importlib.import_module(f"src.{name}")
        except ImportError:
            try:
                return importlib.import_module(f".{name}", package="src")
            except ImportError:
                continue
    raise SystemExit("Impossible d'importer train_drbert ni train_doctobert.")


def _cross_validate_one_seed(train_mod, texts, labels, seed, use_augment):
    """Une validation croisée complète pour UNE graine. Renvoie (accuracy,
    macro_f1). Reproduit la logique de train_*.cross_validate mais en
    paramétrant la graine sur TOUTES les sources d'aléatoire : le découpage
    StratifiedKFold, l'augmentation, ET l'initialisation du modèle (via
    seed= passé à _train_one). C'est indispensable pour mesurer la vraie
    variabilité inter-graines : si on ne faisait varier que le découpage en
    plis (graine du split) sans l'initialisation, un modèle très stable
    donnerait un écart-type artificiellement nul, masquant l'incertitude
    d'entraînement réelle."""
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(labels), dtype=int)
    for k, (tr, te) in enumerate(skf.split(texts, labels), 1):
        train_texts, train_labels = texts[tr], labels[tr]
        if use_augment:
            # graine d'augmentation dérivée de la graine courante -> variantes
            # différentes d'une graine à l'autre (vraie variabilité)
            train_texts, train_labels = train_mod._maybe_augment(
                train_texts, train_labels, seed=seed + k, tag=f"graine {seed} pli {k}"
            )
        model, tok = train_mod._train_one(train_texts, train_labels, seed=seed)
        oof[te] = train_mod._predict(model, tok, texts[te])
        # libère la mémoire GPU entre plis
        del model, tok
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    acc = float((oof == labels).mean())
    f1 = float(f1_score(labels, oof, average="macro"))
    return acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="liste des graines (défaut : 42 43 44 45 46)")
    parser.add_argument("--no-augment", action="store_true",
                        help="désactive la Data Augmentation (sinon suit AUGMENT_ENABLED)")
    parser.add_argument("--tag", default=None,
                        help="nom de la configuration pour les fichiers de sortie")
    cli = parser.parse_args()

    train_mod = _load_train_module()
    train_mod._check_deps()

    if not config.DATASET_CSV.exists():
        raise SystemExit(
            f"{config.DATASET_CSV} introuvable. Lancez d'abord : python -m src.build_dataset"
        )
    df = pd.read_csv(config.DATASET_CSV).fillna({"ml_text": ""})
    texts = df["ml_text"].astype(str).values
    labels = df["label"].astype(int).values

    use_augment = config.AUGMENT_ENABLED and not cli.no_augment
    model_name = getattr(config, "DRBERT_BASE", None) or getattr(config, "DOCTOBERT_BASE", "modèle")
    tag = cli.tag or ("avecDA" if use_augment else "sansDA")

    print(f"Modèle    : {model_name}")
    print(f"Data augmentation : {'OUI' if use_augment else 'NON'}")
    print(f"Graines   : {cli.seeds}")
    print(f"(chaque graine = 1 validation croisée {config.CV_FOLDS} plis "
          f"réentraînée intégralement — soyez patient)\n")

    rows = []
    for seed in cli.seeds:
        print(f"===== Graine {seed} =====")
        acc, f1 = _cross_validate_one_seed(train_mod, texts, labels, seed, use_augment)
        print(f"  -> accuracy={acc:.3f}  macro-F1={f1:.3f}\n")
        rows.append({"seed": seed, "accuracy": round(acc, 4), "macroF1": round(f1, 4)})

    per_seed = pd.DataFrame(rows)
    out_dir = config.DATA_PROCESSED / "multiseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = out_dir / f"multiseed_{tag}.csv"
    per_seed.to_csv(per_seed_path, index=False)

    # Agrégats. ddof=1 = écart-type d'échantillon (le bon pour « ± » sur
    # quelques graines) ; nan si une seule graine.
    def agg(col):
        v = per_seed[col].to_numpy()
        return {
            "moyenne": round(float(v.mean()), 4),
            "ecart_type": round(float(v.std(ddof=1)), 4) if len(v) > 1 else float("nan"),
            "min": round(float(v.min()), 4),
            "max": round(float(v.max()), 4),
        }

    summary = pd.DataFrame({"accuracy": agg("accuracy"), "macroF1": agg("macroF1")}).T
    summary_path = out_dir / f"multiseed_summary_{tag}.csv"
    summary.to_csv(summary_path)

    print("=" * 66)
    print(f"ROBUSTESSE AUX GRAINES — {len(cli.seeds)} graines "
          f"({'avec' if use_augment else 'sans'} DA)")
    print("=" * 66)
    print(per_seed.to_string(index=False))
    print()
    a, f = agg("accuracy"), agg("macroF1")
    if len(cli.seeds) > 1:
        print(f"Accuracy : {a['moyenne']:.3f} ± {a['ecart_type']:.3f}  "
              f"(min {a['min']:.3f}, max {a['max']:.3f})")
        print(f"Macro-F1 : {f['moyenne']:.3f} ± {f['ecart_type']:.3f}  "
              f"(min {f['min']:.3f}, max {f['max']:.3f})")
        print("\nÀ rapporter dans l'article sous la forme « moyenne ± écart-type ».")
    else:
        print(f"Accuracy : {a['moyenne']:.3f}   Macro-F1 : {f['moyenne']:.3f}")
        print("\n(Une seule graine : relancez avec plusieurs --seeds pour un écart-type.)")
    print(f"\nDétails par graine : {per_seed_path}")
    print(f"Récapitulatif      : {summary_path}")


if __name__ == "__main__":
    main()
