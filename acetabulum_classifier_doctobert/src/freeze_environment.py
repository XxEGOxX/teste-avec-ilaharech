"""
freeze_environment.py — Capture l'environnement EXACT pour la reproductibilité.

Pourquoi ce script plutôt qu'un requirements-lock.txt fourni tout fait : les
versions qui comptent sont celles de VOTRE machine, pas celles de la machine
qui a généré le code. Un exemple concret vécu sur ce projet : pypdf 5.x et
pypdf 6.x extraient certains PDF différemment (espaces parasites au milieu de
mots), ce qui change les résultats. Coder en dur une version dans un fichier
livré serait donc trompeur. Ce script lit VOS versions réellement installées
et les fige.

Ce qu'il produit (dans le dossier courant) :
  - requirements-lock.txt : les versions EXACTES de toutes les dépendances
    directes du projet, telles qu'installées chez vous (format « paquet==x.y.z »).
  - environment-info.txt  : contexte matériel/logiciel (Python, OS, CUDA, GPU,
    versions clés) — à joindre en annexe de l'article pour la reproductibilité.

À lancer une fois, sur la machine qui a produit les résultats de l'article,
APRÈS avoir installé toutes les dépendances et vérifié que tout tourne :
    python -m src.freeze_environment

Puis committez requirements-lock.txt et environment-info.txt avec le code.
Pour reproduire à l'identique : pip install -r requirements-lock.txt
"""
from __future__ import annotations

import platform
import sys
from importlib import metadata
from pathlib import Path

# Dépendances directes du projet dont la version doit être figée. On NE fige
# PAS ici toute la sortie de « pip freeze » (qui inclut des centaines de
# paquets transitifs) : on veut un lock lisible et centré sur ce qui compte.
# pip install -r requirements-lock.txt réinstallera les transitifs compatibles.
_DIRECT_DEPS = [
    # cœur (extraction, règles, interface, dataset)
    "streamlit", "altair", "pandas", "numpy", "openpyxl", "pypdf",
    # modèle (fine-tuning + inférence)
    "transformers", "torch", "accelerate",
    # évaluation statistique (bootstrap, McNemar)
    "scipy", "scikit-learn",
    # optionnels s'ils sont présents
    "joblib", "pypdfium2", "pytesseract", "Pillow",
]


def _version_or_none(pkg: str):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _write_lock(path: Path):
    lines = [
        "# requirements-lock.txt — versions EXACTES ayant produit les résultats.",
        "# Généré par src/freeze_environment.py sur la machine de l'auteur.",
        "# Reproduire à l'identique : pip install -r requirements-lock.txt",
        "",
    ]
    found, missing = [], []
    for pkg in _DIRECT_DEPS:
        v = _version_or_none(pkg)
        if v is None:
            missing.append(pkg)
        else:
            found.append(f"{pkg}=={v}")
    lines.extend(found)
    if missing:
        lines.append("")
        lines.append("# Non installés au moment du gel (optionnels ou non utilisés) :")
        lines.extend(f"# {pkg}" for pkg in missing)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return found, missing


def _cuda_info():
    """Infos GPU/CUDA si torch est présent — sans planter s'il ne l'est pas."""
    try:
        import torch
    except ImportError:
        return ["torch non installé — pas d'info CUDA/GPU"]
    info = [
        f"torch                 : {torch.__version__}",
        f"CUDA disponible       : {torch.cuda.is_available()}",
    ]
    if torch.cuda.is_available():
        info.append(f"version CUDA (torch)  : {torch.version.cuda}")
        info.append(f"nombre de GPU         : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            info.append(f"  GPU {i}               : {torch.cuda.get_device_name(i)}")
    else:
        info.append("Exécution sur CPU (aucun GPU CUDA détecté).")
    return info


def _write_env_info(path: Path):
    lines = [
        "environment-info.txt — contexte de reproductibilité (annexe article).",
        "=" * 60,
        "",
        "## Système",
        f"OS                    : {platform.platform()}",
        f"Architecture          : {platform.machine()}",
        f"Processeur            : {platform.processor() or 'inconnu'}",
        "",
        "## Python",
        f"Version Python        : {sys.version.splitlines()[0]}",
        f"Implémentation        : {platform.python_implementation()}",
        f"Exécutable            : {sys.executable}",
        "",
        "## Bibliothèques clés",
    ]
    for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "transformers",
                "torch", "pypdf"]:
        v = _version_or_none(pkg)
        lines.append(f"{pkg:22s}: {v if v else 'NON INSTALLÉ'}")
    lines.append("")
    lines.append("## GPU / CUDA")
    lines.extend(_cuda_info())
    lines.append("")
    lines.append("## Graine aléatoire du projet")
    try:
        from . import config
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src import config
    lines.append(f"RANDOM_STATE          : {config.RANDOM_STATE}")
    lines.append(f"CV_FOLDS              : {config.CV_FOLDS}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    lock_path = Path("requirements-lock.txt")
    env_path = Path("environment-info.txt")

    found, missing = _write_lock(lock_path)
    _write_env_info(env_path)

    print("Environnement figé pour la reproductibilité.")
    print(f"  {lock_path}  ({len(found)} paquets figés"
          + (f", {len(missing)} optionnels absents)" if missing else ")"))
    print(f"  {env_path}  (Python, OS, CUDA/GPU, graine)")
    print()
    print("Versions figées :")
    for line in found:
        print(f"  {line}")
    print()
    print("À joindre au dépôt et à l'article. Reproduire : "
          "pip install -r requirements-lock.txt")


if __name__ == "__main__":
    main()
