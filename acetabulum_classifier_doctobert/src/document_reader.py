"""
document_reader.py — Lecture du texte des comptes rendus, quelle que soit la forme.

Trois cas gérés (dans cet ordre de tentative) :
  (a) ARCHIVE ZIP (format des données fournies) : le « .pdf » est en réalité une
      archive contenant des images scannées (1.jpeg...), un texte OCR déjà extrait
      (1.txt...) et un manifest.json donnant l'ordre des pages. On lit les .txt
      dans l'ordre du manifest. => l'OCR est déjà fait, qualité bonne.
  (b) VRAI PDF avec couche texte : extraction via pypdf puis, en secours,
      pdftotext (poppler). Utile pour les fichiers chargés dans l'app Streamlit.
  (c) PDF SCANNÉ sans texte : OCR optionnel via pytesseract si disponible
      (sinon on renvoie une chaîne vide plutôt que de planter).

Détection du type de document (CRO vs CRR) :
  - par le nom de fichier : suffixe « _CRR » -> CRR ;
  - sinon par le contenu  : marqueurs radiologiques -> CRR, opératoires -> CRO.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Optional

from .text_norm import normalize


# --------------------------------------------------------------------------
# (a) Lecture des archives ZIP (format des données du projet)
# --------------------------------------------------------------------------
def _read_zip_ocr(path: Path) -> Optional[str]:
    """Lit le texte OCR d'une archive ZIP déguisée en .pdf. None si pas un ZIP."""
    if not zipfile.is_zipfile(path):
        return None
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        # Ordre des pages via manifest.json si présent
        ordered_txt = []
        if "manifest.json" in names:
            try:
                manifest = json.loads(z.read("manifest.json"))
                for page in manifest.get("pages", []):
                    tp = page.get("text", {}).get("path")
                    if tp and tp in names:
                        ordered_txt.append(tp)
            except (json.JSONDecodeError, KeyError):
                ordered_txt = []
        # Repli : tous les .txt triés naturellement (1.txt, 2.txt, ...)
        if not ordered_txt:
            txts = [n for n in names if n.lower().endswith(".txt")]
            ordered_txt = sorted(txts, key=_natural_key)

        parts = []
        for tp in ordered_txt:
            try:
                parts.append(z.read(tp).decode("utf-8", "ignore"))
            except KeyError:
                continue
        return "\n".join(parts)


def _natural_key(name: str):
    """Tri naturel : 2.txt avant 10.txt."""
    stem = Path(name).stem
    return (int(stem) if stem.isdigit() else 10**9, name)


# --------------------------------------------------------------------------
# (b) Vrai PDF avec couche texte
# --------------------------------------------------------------------------
def _read_pdf_text(path: Path) -> str:
    """Extrait le texte d'un vrai PDF. Essaie pypdf puis pdftotext (poppler)."""
    text = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        text = ""

    if len(text.strip()) >= 20:
        return text

    # Secours : pdftotext (poppler-utils)
    try:
        import subprocess

        out = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True, timeout=60,
        )
        if out.returncode == 0:
            cand = out.stdout.decode("utf-8", "ignore")
            if len(cand.strip()) > len(text.strip()):
                text = cand
    except Exception:
        pass
    return text


# --------------------------------------------------------------------------
# (c) OCR optionnel pour PDF purement scannés
# --------------------------------------------------------------------------
def _ocr_pdf(path: Path) -> str:
    """OCR de secours (pytesseract). Renvoie '' si l'outil n'est pas installé."""
    try:
        import pypdfium2 as pdfium
        import pytesseract
        from PIL import Image  # noqa: F401

        pdf = pdfium.PdfDocument(str(path))
        chunks = []
        for i in range(len(pdf)):
            bitmap = pdf[i].render(scale=2.0)
            img = bitmap.to_pil()
            chunks.append(pytesseract.image_to_string(img, lang="fra"))
        return "\n".join(chunks)
    except Exception:
        return ""


# --------------------------------------------------------------------------
# API publique
# --------------------------------------------------------------------------
def read_document(path) -> str:
    """Renvoie le texte brut d'un compte rendu, quel que soit son format.

    Tente : (a) ZIP+OCR -> (b) PDF texte -> (c) OCR. Ne lève pas d'exception
    pour un format inattendu ; renvoie '' en dernier recours.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # (a) Archive ZIP (cas des données du projet)
    zip_text = _read_zip_ocr(path)
    if zip_text and zip_text.strip():
        return zip_text

    # (b) Vrai PDF avec texte
    pdf_text = _read_pdf_text(path)
    if pdf_text and len(pdf_text.strip()) >= 20:
        return pdf_text

    # (c) OCR de secours
    ocr_text = _ocr_pdf(path)
    if ocr_text.strip():
        return ocr_text

    return pdf_text or ""


def read_document_from_bytes(data: bytes, filename: str = "") -> str:
    """Variante pour l'app Streamlit : lit depuis des octets en mémoire.

    Écrit dans un tampon ; gère ZIP et PDF. Le nom de fichier sert seulement
    à la détection ZIP/PDF en cas d'ambiguïté.
    """
    buf = io.BytesIO(data)
    # ZIP ?
    if zipfile.is_zipfile(buf):
        buf.seek(0)
        with zipfile.ZipFile(buf) as z:
            names = z.namelist()
            ordered = []
            if "manifest.json" in names:
                try:
                    manifest = json.loads(z.read("manifest.json"))
                    for page in manifest.get("pages", []):
                        tp = page.get("text", {}).get("path")
                        if tp and tp in names:
                            ordered.append(tp)
                except Exception:
                    ordered = []
            if not ordered:
                ordered = sorted(
                    [n for n in names if n.lower().endswith(".txt")], key=_natural_key
                )
            parts = [z.read(tp).decode("utf-8", "ignore") for tp in ordered]
            if parts:
                return "\n".join(parts)

    # Sinon, vrai PDF : on s'appuie sur pypdf depuis les octets
    try:
        from pypdf import PdfReader

        buf.seek(0)
        reader = PdfReader(buf)
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        if text.strip():
            return text
    except Exception:
        pass
    return ""


# Marqueurs de détection du type de document (sur texte normalisé)
_CRR_MARKERS = (
    "bodyscanner", "body scanner", "scanner", "tomodensitometr", "imagerie",
    "radiologi", "examen tdm", "reconstruction", "coupes", "injection de produit",
)
_CRO_MARKERS = (
    "compte rendu operatoire", "operatoire", "diagnostic :", "intervention",
    "installation", "voie d abord", "abord", "osteosynthese", "champ operatoire",
)


def detect_doc_type(text: str, filename: str = "") -> str:
    """Renvoie 'CRR', 'CRO' ou 'UNKNOWN'.

    Priorité au nom de fichier (suffixe _CRR), puis au contenu.
    """
    fn = (filename or "").lower()
    if fn.endswith("_crr.pdf") or "_crr" in fn:
        return "CRR"

    n = normalize(text)
    crr_hits = sum(1 for m in _CRR_MARKERS if m in n)
    cro_hits = sum(1 for m in _CRO_MARKERS if m in n)
    if crr_hits == 0 and cro_hits == 0:
        return "UNKNOWN"
    return "CRR" if crr_hits > cro_hits else "CRO"


def file_id(filename: str) -> str:
    """Normalise un nom de fichier en identifiant de dossier.

    Gère le zéro-padding variable : '0089311.pdf' et Excel 89311 -> '89311'.
    Retire aussi le suffixe '_CRR'.
    """
    base = Path(filename).name
    base = re.sub(r"_CRR\.pdf$|\.pdf$", "", base, flags=re.I)
    try:
        return str(int(base))
    except ValueError:
        return base
