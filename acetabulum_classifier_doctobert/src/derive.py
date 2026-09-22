"""
derive.py — Informations dérivées des champs extraits.

Pour l'instant : calcul automatique de l'âge à partir de la date de naissance
(et, si possible, de la date du compte rendu).

Fonctions séparées et testables, sans dépendance à Streamlit.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

_MONTHS = {
    "janvier": 1, "fevrier": 2, "février": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "aout": 8, "août": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "decembre": 12, "décembre": 12,
}


def _pivot_year(yy: int) -> int:
    """Année sur 2 chiffres -> 4 chiffres (00-25 -> 20xx, 26-99 -> 19xx)."""
    return 2000 + yy if yy <= 25 else 1900 + yy


def parse_date(s: str) -> Optional[date]:
    """Analyse une date française en plusieurs formats.

    Gère : jj/mm/aaaa, jj/mm/aa, jj-mm-aaaa, et le format long
    « (lundi) 7 mars 2017 ». Renvoie None si rien d'exploitable.
    """
    if not s or not str(s).strip():
        return None
    s = str(s).strip()

    # jj/mm/aaaa ou jj/mm/aa (séparateurs / . -)
    m = re.search(r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})\b", s)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y = _pivot_year(y)
        try:
            return date(y, mo, d)
        except ValueError:
            return None

    # format long « 7 mars 2017 » (avec éventuel jour de semaine devant)
    m = re.search(r"\b(\d{1,2})\s+([a-zàâäéèêëîïôöùûüç]+)\s+(\d{4})\b", s, re.IGNORECASE)
    if m:
        d = int(m.group(1))
        mo = _MONTHS.get(m.group(2).lower())
        y = int(m.group(3))
        if mo:
            try:
                return date(y, mo, d)
            except ValueError:
                return None
    return None


def compute_age(date_naissance: str, reference: Optional[str] = None) -> Optional[int]:
    """Âge en années à partir de la date de naissance.

    `reference` : date du compte rendu si disponible ; sinon, date du jour.
    Renvoie None si la date de naissance est inexploitable ou si l'âge obtenu
    est aberrant (< 0 ou > 120, signe d'une date mal lue).
    """
    dob = parse_date(date_naissance)
    if dob is None:
        return None
    ref = parse_date(reference) if reference else None
    ref = ref or date.today()
    age = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    if age < 0 or age > 120:
        return None
    return age


def age_label(extracted_age: str, date_naissance: str,
              reference: Optional[str] = None) -> str:
    """Libellé d'âge prêt à afficher (« 54 ans »), ou "" si indisponible.

    Priorité : l'âge explicitement présent dans le CR ; sinon, l'âge calculé
    depuis la date de naissance. Le caller affiche « Non renseigné » si "".
    """
    if extracted_age and str(extracted_age).strip():
        s = str(extracted_age).strip()
        # ne garder que le nombre si présent, et normaliser le suffixe « ans »
        m = re.search(r"\d{1,3}", s)
        if m:
            return f"{m.group(0)} ans"
        return s
    age = compute_age(date_naissance, reference)
    return f"{age} ans" if age is not None else ""
