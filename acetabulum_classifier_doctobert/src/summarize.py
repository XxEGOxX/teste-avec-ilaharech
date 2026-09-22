"""
summarize.py — Résumé automatique, clair et médicalement utile, d'un CRO/CRR.

⚠ Choix technique assumé : DoctoBERT est un modèle ENCODEUR (type BERT). Il sert à
classer du texte, il NE GÉNÈRE PAS de texte libre. Le résumé est donc EXTRACTIF :
on sélectionne et réagence des phrases réellement présentes dans le compte rendu.
Aucune information n'est inventée.

Améliorations :
  • découpage des phrases sur la PONCTUATION (pas sur les retours à la ligne),
    pour ne pas couper une phrase que l'OCR a répartie sur plusieurs lignes ;
  • sélection de la phrase qui DÉCRIT VRAIMENT la fracture (score par mots-clés
    de fracture, pénalité sur le bruit technique) plutôt que la 1re venue ;
  • pour l'imagerie : conclusion -> résultat -> meilleure phrase acétabulaire ;
  • coupe propre en fin de phrase, jamais au milieu d'un mot ;
  • message explicite si aucune phrase fiable n'est trouvée.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from .extraction import ExtractedInfo


# --------------------------------------------------------------------------
# Outils de phrases
# --------------------------------------------------------------------------
def _sa(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").lower()


def _sentences(text: str):
    """Découpe en phrases sur la ponctuation de fin, après avoir recollé les
    lignes (l'OCR coupe les phrases au fil des lignes)."""
    t = re.sub(r"\s+", " ", (text or "").replace("\r", " ").replace("\n", " ")).strip()
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = []
    for p in parts:
        p = p.strip()
        # garder les fragments contenant assez de lettres (évite "06001", "PDL=...")
        if len(re.sub(r"[^a-zA-Zà-ÿ]", "", p)) >= 6:
            out.append(p)
    return out


def _strip_label_prefix(s: str) -> str:
    """Retire un préfixe d'étiquette résiduel (« RESULTATS : », « CONCLUSION : »…)."""
    return re.sub(r"^\s*(resultats?|conclusion|technique|indication|impression|"
                  r"au total)\s*:?\s*", "", s, flags=re.IGNORECASE).strip()


def _clip(text: str, hard: int = 300) -> str:
    """Nettoie et borne une phrase. Ne coupe jamais au milieu d'un mot ; si la
    phrase est trop longue, coupe à une frontière de proposition (« … »)."""
    text = re.sub(r"\s+", " ", text or "").strip().strip(" .;,–-")
    if not text:
        return ""
    if len(text) <= hard:
        return text
    cut = text[:hard]
    for sep in (". ", " ; ", "; ", ", "):
        i = cut.rfind(sep)
        if i > hard * 0.5:
            return cut[:i].strip(" .;,") + "…"
    return cut.rsplit(" ", 1)[0].strip(" .;,") + "…"


# mots décrivant une fracture / pénalités « bruit technique »
_DESCR = ("fracture", "colonne anter", "colonne poster", "paroi anter", "paroi poster",
          "transvers", "bicolo", "bi-colo", "bi colo", "hemi", "cotyle", "acetabul",
          "cotyloid", "deplac", "comminu", "plurifrag", "luxation", "enfonc",
          "impaction", "fragment", "subluxation", "incongru", "marche d'escalier",
          "trait de fracture")
_NOISE = ("acquisition", "injection", "produit de contraste", "reconstruction",
          "coupes", "helicoidal", "equipement", "protocole", "dosimetrie", "pdl",
          "mgy", "millimetr", "matrice", "topogramme", "secretariat", "mot de passe",
          "code d", "valide par", "https", "page ", "scanographique")


def _score(s: str) -> int:
    sl = _sa(s)
    sc = sum(2 for t in _DESCR if t in sl) - sum(2 for t in _NOISE if t in sl)
    if "fracture" in sl:
        sc += 2
    return sc


def _best_fracture_sentence(text: str) -> str:
    """Meilleure phrase décrivant la fracture (score le plus élevé)."""
    best, best_sc = "", 0
    for s in _sentences(text):
        s2 = _strip_label_prefix(s)
        sc = _score(s2)
        if sc > best_sc:
            best, best_sc = s2, sc
    return best if best_sc >= 2 else ""


def _first_useful(text: str) -> str:
    """Première phrase complète utile d'un bloc (diagnostic, indication, suites)."""
    ss = _sentences(text)
    return _clip(_strip_label_prefix(ss[0])) if ss else ""


def _imaging_line(crr_info: ExtractedInfo) -> str:
    """Phrase d'imagerie : conclusion -> résultat -> meilleure phrase acétabulaire.

    On ne retient qu'une phrase qui décrit réellement la fracture (score ≥ 2),
    afin d'éviter de remonter un en-tête ou une ligne technique. Si rien de
    fiable n'est trouvé, on renvoie "" (le caller affiche un message dédié)."""
    for section in ("conclusion", "resultat"):
        s = _best_fracture_sentence(crr_info.get_section(section))
        if s:
            return _clip(s)
    body = " ".join(x for x in (crr_info.get_section("resultat"),
                                crr_info.get_section("technique"),
                                crr_info.get_section("conclusion")) if x)
    s = _best_fracture_sentence(body)
    return _clip(s) if s else ""


# --------------------------------------------------------------------------
# Composition du résumé
# --------------------------------------------------------------------------
def _fmt_patient(info: ExtractedInfo) -> str:
    nom = " ".join(x for x in [info.nom, info.prenom] if x).strip()
    bits = []
    if nom:
        bits.append(nom)
    if info.date_naissance:
        bits.append(f"né(e) le {info.date_naissance}")
    return ", ".join(bits)


def _end(s: str) -> str:
    return s + ("" if s.endswith(("…", ".", "!", "?")) else ".")


def build_summary(
    cro_info: Optional[ExtractedInfo],
    crr_info: Optional[ExtractedInfo],
    predicted_label_name: str,
    method: str,
    anonymize: bool = False,
    dossier_id: str = "",
) -> str:
    """Résumé extractif structuré. `method` est ignoré ici (réservé à l'audit)."""
    lines = []

    # En-tête patient (CRO prioritaire, sinon CRR)
    patient_src = cro_info if (cro_info and (cro_info.nom or cro_info.prenom)) else crr_info
    date_cr = ((cro_info.date_cr if cro_info and cro_info.date_cr
                else (crr_info.date_cr if crr_info else "")) or "")
    if anonymize:
        ident = f"Patient {dossier_id}".strip() or "Patient"
        tail = f" — compte rendu du {date_cr}" if date_cr else ""
        lines.append(_end(f"{ident}{tail}"))
    elif patient_src:
        head = _fmt_patient(patient_src)
        if head or date_cr:
            tail = f" — compte rendu du {date_cr}" if date_cr else ""
            lines.append(_end(f"Patient : {head}{tail}" if head else f"Compte rendu du {date_cr}"))

    # Volet opératoire (CRO)
    if cro_info:
        diag = _first_useful(cro_info.get_section("diagnostic"))
        if diag:
            lines.append(_end(f"Diagnostic opératoire : {diag}"))
        geste = []
        ind = _first_useful(cro_info.get_section("indication"))
        if ind:
            geste.append(ind)
        abord = cro_info.get_section("voies_abord").strip()
        if abord:
            geste.append(f"voie d'abord : {abord}")
        if geste:
            lines.append(_end("Geste : " + " ; ".join(geste)))
        post = _first_useful(cro_info.get_section("consignes_post_op"))
        if post:
            lines.append(_end(f"Suites : {post}"))

    # Volet radiologique (CRR)
    if crr_info:
        ind = _first_useful(crr_info.get_section("indication"))
        if ind:
            lines.append(_end(f"Indication (imagerie) : {ind}"))
        img = _imaging_line(crr_info)
        if img:
            lines.append(_end(f"Imagerie : {img}"))
        else:
            lines.append("Imagerie : Information radiologique non clairement identifiée.")

    # Conclusion : classe retenue (la méthode reste dans l'analyse avancée)
    lines.append(f"➜ Type de fracture retenu : {predicted_label_name}.")
    return "\n".join(lines)
