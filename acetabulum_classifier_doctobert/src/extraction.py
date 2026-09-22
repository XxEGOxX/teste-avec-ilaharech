"""
extraction.py — Extraction structurée des informations d'un CRO ou d'un CRR.

Problème principal (CRO) : la mise en page est en DEUX COLONNES sur une même
ligne, p. ex. « Opérateur : Dr X    Nom : Y ». On ne peut donc pas se contenter
de « tout ce qui suit l'étiquette jusqu'à la fin de ligne ».

Solution robuste : on repère TOUTES les étiquettes connues dans le document,
puis pour chacune on prend le texte jusqu'à l'étiquette SUIVANTE (par position).
Comme les étiquettes de gauche et de droite s'entrelacent dans l'ordre de
lecture, la valeur de « Opérateur » s'arrête naturellement à « Nom : », etc.

Aucune information n'est inventée : un champ absent vaut "" (l'interface
affiche alors « Non trouvé »). Le module ne lève jamais d'exception sur un
document mal formé.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------
# Étiquettes connues -> clé canonique.
# L'ordre n'importe pas (on travaille par position dans le texte), mais les
# motifs doivent tolérer les variantes d'OCR (espaces, apostrophes, accents).
# --------------------------------------------------------------------------
# Champs « en ligne » (valeur courte, souvent en deux colonnes)
INLINE_LABELS = {
    r"operateur": "operateur",
    r"assistant": "assistant",
    r"aide operatoire|aide": "aide",
    r"interne": "interne",
    r"instrumentiste": "instrumentiste",
    r"panseuse": "panseuse",
    r"anesthesiste": "anesthesiste",
    r"type ?d ?'?anesthesie|type d anesthesie": "type_anesthesie",
    r"nom et prenom|nom/?prenom d ?'?usages?|nom/?prenom": "nom_prenom",
    r"nom de naissance|nom naissance": "nom_naissance",
    r"nom marital|nom d ?'?epouse|nom d ?'?usage": "nom_usage",
    r"prenom\(s\) de naissance|prenoms? de naissance": "prenom_naissance",
    r"(?:1er|1\s*er|premier) prenom(?: de naissance)?": "premier_prenom",
    r"lieu de naissance": "lieu_naissance",
    r"nom": "nom",
    r"prenom(?:\(s\))?": "prenom",
    r"date de naissance|date naissance": "date_naissance",
    r"n[°o] de s\.?s\.?|n[°o] de ss": "num_ss",
    r"dossier n[°o]": "dossier",
    r"adresse": "adresse",
    r"medecin traitant": "medecin_traitant",
    r"rhumatologue": "rhumatologue",
    r"medecin reeducateur": "medecin_reeducateur",
    r"kinesitherapeute": "kinesitherapeute",
    r"n[°o] de tel|n[°o] de tel\.?|n[°o] de telephone": "tel",
    r"courriel securise de l ?'?usager|courriel|e-?mail": "email",
    r"codes? diagnostic": "codes_diagnostic",
    r"codes? ccam": "codes_ccam",
    # spécifiques CRR
    r"sexe": "sexe",
    r"age": "age",
    r"ipp": "ipp",
    r"n[°o] episode": "episode",
    r"compte rendu de": "compte_rendu_de",
    r"uf demandeuse": "uf_demandeuse",
    r"date de l ?'?acte": "date_acte",
}

# Sections « bloc » (valeur multi-lignes jusqu'à la section suivante)
SECTION_LABELS = {
    r"diagnostic": "diagnostic",
    r"indication": "indication",
    r"voies? d ?'?abords?": "voies_abord",
    r"table": "table",
    r"installation": "installation",
    r"intervention": "intervention",
    r"consignes? post ?-? ?operatoires?": "consignes_post_op",
    r"suites? operatoires?": "consignes_post_op",
    r"technique ?(?:& ?| et )?resultat|technique": "technique",
    r"resultat": "resultat",
    r"conclusion": "conclusion",
}

# Fabricants connus (matériel chirurgical / imagerie) — pour repérage best effort
KNOWN_MANUFACTURERS = [
    "stryker", "serf", "zimmer", "depuy", "smith & nephew", "smith and nephew",
    "medacta", "medartis", "newclip", "synthes", "biomet", "amplitude",
    "tornier", "wright", "ge medical systems", "ge healthcare", "siemens",
    "philips", "canon", "toshiba", "fh orthopedics", "lepine", "evolutis",
]

_LABELS_ARE_SECTION = {v: True for v in SECTION_LABELS.values()}


def _strip_accents_lower(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _searchable(s: str) -> str:
    """Forme de recherche SANS changer le nombre de caractères (positions
    préservées vis-à-vis du texte original) : minuscule, accents retirés,
    espaces spéciaux (\\xa0, \\r, \\t, fines) ramenés à un espace simple.
    Indispensable car l'OCR colle des espaces insécables dans les étiquettes
    (« Date\\xa0de\\xa0naissance »)."""
    out = []
    for ch in s:
        c = ch
        if c in "\xa0\r\t\u202f\u2009\f\v":
            c = " "
        c = c.lower()
        d = unicodedata.normalize("NFD", c)
        base = "".join(x for x in d if unicodedata.category(x) != "Mn")
        out.append(base[0] if base else " ")
    return "".join(out)


def _build_master_regex():
    """Construit un regex alternant toutes les étiquettes (sur texte sans accent).

    Renvoie (regex_compilé, mapping motif->clé, set des clés de section).
    """
    entries = []  # (pattern, key, is_section)
    for pat, key in SECTION_LABELS.items():
        entries.append((pat, key, True))
    for pat, key in INLINE_LABELS.items():
        entries.append((pat, key, False))
    # On numérote les groupes pour retrouver la clé.
    parts = []
    meta = []
    for i, (pat, key, is_sec) in enumerate(entries):
        parts.append(f"(?P<g{i}>{pat})")
        meta.append((key, is_sec))
    # Une étiquette = un des motifs, suivi d'un éventuel espace puis « : »
    master = re.compile(
        r"(?:^|\n|\r|\t| )(?:" + "|".join(parts) + r")\s*:",
        re.IGNORECASE,
    )
    return master, meta


_MASTER, _META = _build_master_regex()


@dataclass
class ExtractedInfo:
    doc_type: str = ""
    # patient / administratif
    nom: str = ""
    prenom: str = ""
    date_naissance: str = ""
    sexe: str = ""
    age: str = ""
    ipp: str = ""
    date_cr: str = ""
    # acteurs
    operateur: str = ""
    assistant: str = ""
    interne: str = ""
    instrumentiste: str = ""
    anesthesiste: str = ""
    type_anesthesie: str = ""
    radiologue: str = ""
    chirurgien: str = ""
    # codes
    codes_diagnostic: str = ""
    codes_ccam: str = ""
    # matériel
    materiel: str = ""
    fabricant: str = ""
    # sections (multi-lignes)
    sections: Dict[str, str] = field(default_factory=dict)
    # tous les champs en ligne bruts (pour debug/affichage complémentaire)
    raw_fields: Dict[str, str] = field(default_factory=dict)

    def get_section(self, key: str) -> str:
        return self.sections.get(key, "")


def _normalize_apostrophes(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u02bc", "'")


def _clean_inline(value: str) -> str:
    """Nettoie une valeur en ligne : retire retours chariot et espaces multiples."""
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    # retire des résidus du genre « Dr » seul ou ponctuation isolée
    if value in {".", "()", "() .", "( )", "-", ":"}:
        return ""
    return value


def _clean_block(value: str) -> str:
    """Nettoie une section bloc : conserve la structure en lignes."""
    value = value.replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in value.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines).strip()


def _clean_dob(value: str) -> str:
    """Garde la date de naissance « propre » : si une date jj/mm/aa(aa) est
    présente on ne garde qu'elle (retire les suffixes du type « (92 ans) »)."""
    if not value:
        return ""
    m = re.search(r"\b([0-3]?\d[/.\-][01]?\d[/.\-]\d{2,4})\b", value)
    if m:
        return m.group(1)
    # sinon, retirer une éventuelle parenthèse en fin
    return re.sub(r"\s*\(.*$", "", value).strip()


def _segment(text: str):
    """Découpe le texte en (clé, is_section, valeur) via le balisage par position.

    On travaille sur une copie sans accents pour repérer les étiquettes, mais on
    extrait les valeurs sur le texte ORIGINAL (pour l'affichage).
    """
    original = _normalize_apostrophes(text)
    flat = _searchable(original)
    assert len(flat) == len(original)  # positions alignées

    matches = list(_MASTER.finditer(flat))
    results = []
    for i, m in enumerate(matches):
        # Identifier la clé via le groupe nommé qui a matché
        key = None
        is_sec = False
        for gi, (k, s) in enumerate(_META):
            if m.group(f"g{gi}") is not None:
                key, is_sec = k, s
                break
        if key is None:
            continue
        value_start = m.end()
        value_end = matches[i + 1].start() if i + 1 < len(matches) else len(original)
        value = original[value_start:value_end]
        results.append((key, is_sec, value))
    return results


def _extract_first(segments, key: str, is_section: bool) -> str:
    for k, s, v in segments:
        if k == key:
            return _clean_block(v) if is_section else _clean_inline(v)
    return ""


def section_text(text: str, key: str) -> str:
    """API publique : texte complet d'UNE section (ex. « diagnostic »),
    du début de l'étiquette jusqu'à l'étiquette suivante (section OU champ
    en ligne), quel que soit le nombre de lignes.

    Réutilise EXACTEMENT le même découpage par position que `extract()`
    (_segment) : c'est le même mécanisme qui gère déjà la mise en page deux
    colonnes des CRO. Utilisé par `text_zones.cro_diagnostic_zone` pour ne
    jamais dupliquer la logique de détection des limites de section — un
    seul endroit décide où une section commence et se termine.
    """
    if not text or not text.strip():
        return ""
    segments = _segment(text)
    return _extract_first(segments, key, is_section=True)


def _extract_date_cr(text: str, doc_type: str, fields_acte: str) -> str:
    """Date du compte rendu : CRO -> « <Ville>, le ... » ; CRR -> DATE DE L'ACTE."""
    t = _normalize_apostrophes(text)
    if doc_type == "CRR":
        if fields_acte:
            # garde la partie date (jj/mm/aaaa) si présente
            mdate = re.search(r"\d{2}/\d{2}/\d{4}", fields_acte)
            return mdate.group(0) if mdate else fields_acte
        # repli : date longue française du type « mardi 7 mars 2017 »
        m = re.search(
            r"(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+\d{1,2}\s+\w+\s+\d{4}",
            t, re.IGNORECASE,
        )
        return m.group(0) if m else ""
    # CRO : ligne « Nice, le 03/01/19 » ou « Nice, le 5 décembre 2025 ».
    # La date doit être sur la MÊME ligne que « le » (pas de saut de ligne) et
    # commencer par un chiffre — sinon la ligne « Nice, le » est vide.
    m = re.search(r"\n\s*[A-Za-zÉéÈè\-]+\s*,\s*le[ \t]+(\d[^\n\r]*)", t)
    if m:
        return _clean_inline(m.group(1))
    return ""


def _extract_materiel_fabricant(text: str, doc_type: str, sections: dict):
    """Best effort : matériel chirurgical / d'imagerie + fabricant."""
    t = _normalize_apostrophes(text)
    fabricants = set()
    low = t.lower()
    for man in KNOWN_MANUFACTURERS:
        if man in low:
            fabricants.add(man.upper())

    materiel = ""
    if doc_type == "CRR":
        # « Examen réalisé sur l'équipement de marque X, Y »
        m = re.search(r"equipement de marque\s*([^\n\r]*(?:\n[^\n\r:]*)?)", t, re.IGNORECASE)
        if m:
            materiel = _clean_inline(m.group(1))
    else:
        # CRO : on liste les implants entre parenthèses dans INDICATION+INTERVENTION
        body = " ".join([sections.get("indication", ""), sections.get("intervention", "")])
        # phrases contenant un fabricant connu ou une parenthèse de marque
        paren = re.findall(r"[A-Za-zÀ-ÿ0-9 \-/]+\([A-Za-zÀ-ÿ &]+\)", body)
        materiel = "; ".join(p.strip() for p in paren[:6])

    return materiel, ", ".join(sorted(fabricants))


def _extract_radiologue(text: str) -> str:
    """CRR : le radiologue signataire, souvent « Docteur NOM- Prénom » en fin."""
    t = _normalize_apostrophes(text)
    m = re.search(r"(?:Docteur|Dr)\s+([A-ZÉÈÀ][\w\-']+(?:[ \-][A-ZÉÈÀ][\w\-']+)*)", t)
    return _clean_inline(m.group(1)) if m else ""


def extract(text: str, doc_type: str = "") -> ExtractedInfo:
    """Extrait toutes les informations utiles d'un CRO ou d'un CRR."""
    info = ExtractedInfo(doc_type=doc_type or "")
    if not text or not text.strip():
        return info

    segments = _segment(text)

    # --- sections bloc ---
    section_keys = set(SECTION_LABELS.values())
    for key in section_keys:
        val = _extract_first(segments, key, is_section=True)
        if val:
            info.sections[key] = val

    # --- champs en ligne ---
    inline_keys = set(INLINE_LABELS.values())
    raw = {}
    for key in inline_keys:
        val = _extract_first(segments, key, is_section=False)
        if val:
            raw[key] = val
    info.raw_fields = raw

    info.nom = raw.get("nom", "")
    info.prenom = raw.get("prenom", "")
    # Format CRR combiné « NOM et PRENOM : X Y » -> si nom absent, on l'utilise
    if not info.nom and raw.get("nom_prenom"):
        info.nom = raw["nom_prenom"]
    info.date_naissance = _clean_dob(raw.get("date_naissance", ""))
    info.sexe = raw.get("sexe", "")
    info.age = raw.get("age", "")
    info.ipp = raw.get("ipp", "")
    info.operateur = raw.get("operateur", "")
    info.assistant = raw.get("assistant", "")
    info.interne = raw.get("interne", "")
    info.instrumentiste = raw.get("instrumentiste", "")
    info.anesthesiste = raw.get("anesthesiste", "")
    info.type_anesthesie = raw.get("type_anesthesie", "")
    info.codes_diagnostic = raw.get("codes_diagnostic", "")
    info.codes_ccam = raw.get("codes_ccam", "")
    info.chirurgien = info.operateur

    # --- dérivés ---
    info.date_cr = _extract_date_cr(text, doc_type, raw.get("date_acte", ""))
    info.materiel, info.fabricant = _extract_materiel_fabricant(text, doc_type, info.sections)
    if doc_type == "CRR":
        info.radiologue = _extract_radiologue(text)

    return info


# Libellés lisibles des sections (pour l'affichage dans l'app)
SECTION_DISPLAY = {
    "diagnostic": "Diagnostic",
    "indication": "Indication",
    "voies_abord": "Voies d'abord",
    "table": "Table",
    "installation": "Installation",
    "intervention": "Intervention",
    "consignes_post_op": "Consignes post-opératoires",
    "technique": "Technique",
    "resultat": "Résultat",
    "conclusion": "Conclusion",
}

# Champs administratifs lisibles (clé -> libellé)
FIELD_DISPLAY = [
    ("nom", "Nom"),
    ("prenom", "Prénom"),
    ("date_naissance", "Date de naissance"),
    ("sexe", "Sexe"),
    ("age", "Âge"),
    ("date_cr", "Date du compte rendu"),
    ("operateur", "Opérateur / Chirurgien"),
    ("assistant", "Assistant"),
    ("interne", "Interne"),
    ("instrumentiste", "Instrumentiste"),
    ("anesthesiste", "Anesthésiste"),
    ("type_anesthesie", "Type d'anesthésie"),
    ("radiologue", "Radiologue"),
    ("codes_diagnostic", "Codes Diagnostic"),
    ("codes_ccam", "Codes CCAM"),
    ("materiel", "Matériel"),
    ("fabricant", "Fabricant / Société"),
    ("ipp", "IPP"),
]
