"""
registry.py — Construit le registre de TOUS les dossiers présents dans data/raw.

Pour chaque identifiant de dossier, on apparie le CRO et/ou le CRR, on lit les
textes, on extrait les informations structurées, on prédit la classe (règles +
DoctoBERT) et on prépare un résumé. Le résultat alimente :
  - le tableau global (page 2) ;
  - la fiche détaillée (clic sur une ligne) ;
  - la page statistiques (page 3).

Ce module n'importe pas Streamlit : il est testable seul et réutilisable.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from . import config
    from .classifier import FractureClassifier, Prediction
    from .document_reader import detect_doc_type, file_id, read_document
    from .extraction import ExtractedInfo, extract
    from .summarize import build_summary
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.classifier import FractureClassifier, Prediction
    from src.document_reader import detect_doc_type, file_id, read_document
    from src.extraction import ExtractedInfo, extract
    from src.summarize import build_summary


@dataclass
class DossierRecord:
    dossier_id: str
    has_cro: bool
    has_crr: bool
    cro_path: str = ""
    crr_path: str = ""
    cro_text: str = ""
    crr_text: str = ""
    cro_info: Optional[ExtractedInfo] = None
    crr_info: Optional[ExtractedInfo] = None
    prediction: Optional[Prediction] = None
    summary: str = ""

    # raccourcis pour le tableau
    @property
    def info(self) -> ExtractedInfo:
        """Infos « patient » : on privilégie le CRO, sinon le CRR."""
        if self.cro_info and (self.cro_info.nom or self.cro_info.prenom):
            return self.cro_info
        return self.crr_info or self.cro_info or ExtractedInfo()


def scan_dossiers(raw_dir) -> Dict[str, Dict[str, str]]:
    """Apparie les fichiers par identifiant de dossier (zéro-padding géré)."""
    raw_dir = Path(raw_dir)
    out: Dict[str, Dict[str, str]] = {}
    for f in glob.glob(str(raw_dir / "*.pdf")):
        name = os.path.basename(f)
        fid = file_id(name)
        rec = out.setdefault(fid, {"cro": "", "crr": ""})
        if name.lower().endswith("_crr.pdf"):
            rec["crr"] = f
        else:
            rec["cro"] = f
    return out


def build_records(classifier: FractureClassifier, raw_dir=None) -> List[DossierRecord]:
    """Construit la liste complète des dossiers (lecture + extraction + prédiction)."""
    raw_dir = raw_dir or config.DATA_RAW
    pairs = scan_dossiers(raw_dir)
    records: List[DossierRecord] = []

    for fid, paths in sorted(pairs.items()):
        cro_path, crr_path = paths.get("cro", ""), paths.get("crr", "")
        cro_text = read_document(cro_path) if cro_path else ""
        crr_text = read_document(crr_path) if crr_path else ""

        cro_info = extract(cro_text, "CRO") if cro_text else None
        crr_info = extract(crr_text, "CRR") if crr_text else None

        pred = classifier.predict(cro_text, crr_text)
        summary = build_summary(cro_info, crr_info, pred.label_name, pred.source)

        records.append(DossierRecord(
            dossier_id=fid,
            has_cro=bool(cro_path),
            has_crr=bool(crr_path),
            cro_path=cro_path,
            crr_path=crr_path,
            cro_text=cro_text,
            crr_text=crr_text,
            cro_info=cro_info,
            crr_info=crr_info,
            prediction=pred,
            summary=summary,
        ))
    return records


def records_to_dataframe(records: List[DossierRecord]) -> pd.DataFrame:
    """Tableau synthétique (une ligne par dossier) pour l'affichage et les stats."""
    rows = []
    for r in records:
        info = r.info
        if r.has_cro and r.has_crr:
            src_docs = "CRO+CRR"
        elif r.has_cro:
            src_docs = "CRO"
        else:
            src_docs = "CRR"
        rows.append({
            "Dossier": r.dossier_id,
            "Nom": info.nom or "",
            "Prénom": info.prenom or "",
            "Naissance": info.date_naissance or "",
            "Date CR": (r.cro_info.date_cr if r.cro_info and r.cro_info.date_cr
                        else (r.crr_info.date_cr if r.crr_info else "")) or "",
            "Type de fracture": r.prediction.label_name if r.prediction else "",
            "Classe": str(r.prediction.label) if (r.prediction and r.prediction.label) else "",
            "Opérateur": (r.cro_info.operateur if r.cro_info else "") or "",
            "Codes CCAM": (r.cro_info.codes_ccam if r.cro_info else "") or "",
            "Documents": src_docs,
            "Méthode": r.prediction.source if r.prediction else "",
            "Confiance": f"{r.prediction.confidence:.2f}" if r.prediction else "",
            "À vérifier": "⚠" if (r.prediction and r.prediction.needs_review) else "",
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Test rapide en CLI (mode règles seules si DoctoBERT absent)
    clf = FractureClassifier()
    recs = build_records(clf)
    df = records_to_dataframe(recs)
    print(f"{len(recs)} dossiers.")
    print(df.to_string(index=False, max_colwidth=28))
