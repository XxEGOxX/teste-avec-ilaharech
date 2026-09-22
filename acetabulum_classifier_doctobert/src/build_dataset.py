"""
build_dataset.py — Construit le dataset d'entraînement à partir des sources.

Pour chaque ligne du fichier Excel annoté :
  1. localise le CRO et/ou le CRR correspondant (zéro-padding variable géré) ;
  2. lit leur texte (OCR déjà présent dans les archives ZIP) ;
  3. extrait les zones propres (diagnostic CRO, phrases acétabulaires) ;
  4. construit le texte ML centralisé (build_ml_text) ;
  5. applique les règles (pour mesurer leur couverture, pas pour « tricher »).

Sorties (dans data/processed/) :
  - dataset.csv            : une ligne par dossier, prête pour l'entraînement ;
  - extraction_report.csv  : diagnostic de couverture (fichiers manquants, etc.).

Exécution :
    python -m src.build_dataset
  (depuis la racine du projet, après avoir placé les données dans data/raw/)
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import pandas as pd

# Imports compatibles « python -m src.build_dataset » ET « python src/build_dataset.py »
try:
    from . import config
    from .document_reader import file_id, read_document
    from .rules import classify_case
    from .text_zones import acetabular_sentences, build_ml_text, cro_diagnostic_zone
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.document_reader import file_id, read_document
    from src.rules import classify_case
    from src.text_zones import acetabular_sentences, build_ml_text, cro_diagnostic_zone


def _index_documents(raw_dir: Path):
    """Construit deux index id -> chemin : un pour les CRO, un pour les CRR."""
    cro, crr = {}, {}
    for f in glob.glob(str(raw_dir / "*.pdf")):
        name = os.path.basename(f)
        fid = file_id(name)
        if name.lower().endswith("_crr.pdf"):
            crr[fid] = f
        else:
            cro[fid] = f
    return cro, crr


def build():
    raw_dir = config.DATA_RAW
    excel_path = raw_dir / config.EXCEL_NAME
    if not excel_path.exists():
        raise FileNotFoundError(
            f"Fichier Excel introuvable : {excel_path}\n"
            f"Placez les données (PDF + Excel) dans {raw_dir}"
        )

    df = pd.read_excel(excel_path)
    # On garde uniquement les lignes étiquetées
    df = df.dropna(subset=[config.COL_LABEL]).copy()
    df[config.COL_LABEL] = df[config.COL_LABEL].astype(int)

    cro_idx, crr_idx = _index_documents(raw_dir)

    rows, report = [], []
    for _, r in df.iterrows():
        fid = str(int(r[config.COL_ID]))
        label = int(r[config.COL_LABEL])
        source_obs = str(r.get(config.COL_SOURCE, "") or "")
        observation = str(r.get(config.COL_OBS, "") or "")

        cro_path = cro_idx.get(fid)
        crr_path = crr_idx.get(fid)
        cro_text = read_document(cro_path) if cro_path else ""
        crr_text = read_document(crr_path) if crr_path else ""

        diag_zone = cro_diagnostic_zone(cro_text) if cro_text else ""
        ml_text = build_ml_text(cro_text, crr_text)

        rule = classify_case(cro_text, crr_text)

        rows.append({
            "id": fid,
            "has_cro": bool(cro_path),
            "has_crr": bool(crr_path),
            "source_observation": source_obs,
            "observation": observation,
            "label": label,
            "cro_text": cro_text,
            "crr_text": crr_text,
            "diag_zone": diag_zone,
            "ml_text": ml_text,
            "rule_pred": rule.label if rule.label is not None else "",
            "rule_conf": rule.confidence,
            "rule_zone": rule.zone,
            "rule_term": rule.term or "",
        })

        report.append({
            "id": fid,
            "label": label,
            "has_cro": bool(cro_path),
            "has_crr": bool(crr_path),
            "ml_text_len": len(ml_text),
            "rule_pred": rule.label if rule.label is not None else "",
            "rule_correct": (rule.label == label) if rule.label is not None else "",
            "ml_text_empty": len(ml_text.strip()) == 0,
        })

    out_df = pd.DataFrame(rows)
    rep_df = pd.DataFrame(report)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(config.DATASET_CSV, index=False)
    rep_df.to_csv(config.EXTRACTION_REPORT_CSV, index=False)

    # ----- Récapitulatif console -----
    n = len(out_df)
    missing_cro = (~out_df["has_cro"]).sum()
    missing_crr_only = ((~out_df["has_cro"]) & out_df["has_crr"]).sum()
    empty_ml = (out_df["ml_text"].str.strip() == "").sum()
    covered = (out_df["rule_pred"] != "").sum()
    correct = sum(
        1 for _, r in out_df.iterrows()
        if r["rule_pred"] != "" and int(r["rule_pred"]) == r["label"]
    )

    print(f"Dossiers étiquetés ............. {n}")
    print(f"  sans CRO ..................... {missing_cro}")
    print(f"  CRR uniquement (sans CRO) .... {missing_crr_only}")
    print(f"  texte ML vide ................ {empty_ml}")
    print("Distribution des classes :")
    for lab, cnt in out_df["label"].value_counts().sort_index().items():
        print(f"  classe {lab} : {cnt}")
    print("Règles :")
    print(f"  couverture ................... {covered}/{n} ({100*covered/n:.1f}%)")
    if covered:
        print(f"  précision (sur couverts) ..... {correct}/{covered} ({100*correct/covered:.1f}%)")
    print(f"\nÉcrit : {config.DATASET_CSV}")
    print(f"Écrit : {config.EXTRACTION_REPORT_CSV}")
    return out_df


if __name__ == "__main__":
    build()
