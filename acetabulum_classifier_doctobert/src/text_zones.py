"""
text_zones.py — Extraction des ZONES DIAGNOSTIQUES PROPRES.

Découverte essentielle du projet : on ne doit JAMAIS lire le corps opératoire
d'un CRO pour décider du type de fracture. Le corps contient des pièges :
  - « vis bi-colonne 3.5 mm »      -> c'est une VIS, pas la fracture ;
  - « ostéosynthèse des 2 colonnes » -> c'est une TECHNIQUE, pas la fracture.

On restreint donc l'analyse à des zones fiables :
  - CRO : la ligne « DIAGNOSTIC : ... » (+ « INDICATION : ... »), très propres ;
  - CRR : uniquement les phrases mentionnant l'acétabulum (le diagnostic est dans
          le corps du compte rendu, souvent pas dans la CONCLUSION).

`build_ml_text` assemble le texte vu par le modèle ML. Il est CENTRALISÉ ici
pour garantir que l'entraînement et l'inférence utilisent strictement la même
représentation (pas de décalage train/inférence).
"""
from __future__ import annotations

import re

from .text_norm import normalize
from .extraction import section_text

# Mots-clés délimitant une phrase « acétabulaire » pertinente (texte normalisé).
_ACETABULAR_KW = re.compile(
    r"cotyle|acetabul|colo[nl]|paroi|transvers|bicolo|judet|letournel|"
    r"anterieur|posterieur|hemi"
)


def cro_diagnostic_zone(raw_text: str) -> str:
    """Texte normalisé de la zone diagnostique d'un CRO.

    Prend les sections DIAGNOSTIC et INDICATION dans leur INTÉGRALITÉ (pas
    seulement leur première ligne), en réutilisant le découpage par position
    de extraction.py (`section_text`) : la valeur d'une section s'arrête à
    l'étiquette suivante (autre section OU champ en ligne comme « Voies
    d'abords », « Table »...), pas au premier retour à la ligne.

    Pourquoi ce changement : l'OCR/la mise en page coupe parfois la phrase
    diagnostique sur 2 lignes physiques avant la vraie fin de section — le
    terme décisif peut alors être sur la 2e ligne. En ne lisant que la 1re
    ligne, deux cas réels du projet passaient à côté d'un terme décisif
    (« bi-colonne », « hémi-transverse ») situé juste après le saut de ligne.
    Toujours restreint à DIAGNOSTIC + INDICATION : on ne lit jamais le corps
    opératoire (VOIES D'ABORD, INTERVENTION...), donc les pièges connus
    (« vis bi-colonne 3.5 mm », « ostéosynthèse des 2 colonnes ») restent hors
    de cette zone comme avant.
    """
    zones = []
    diag = section_text(raw_text, "diagnostic")
    if diag:
        zones.append(diag)
    indic = section_text(raw_text, "indication")
    if indic:
        zones.append(indic)
    return normalize(" ".join(zones))


def acetabular_sentences(raw_text: str) -> str:
    """Concatène (normalisées) les phrases mentionnant l'acétabulum.

    Utilisé pour le CRR (et en complément du CRO). On segmente sur . \\n ;
    puis on conserve uniquement les phrases contenant un mot-clé acétabulaire.
    """
    norm_text = normalize(raw_text)
    sentences = re.split(r"[.\n;]", norm_text)
    kept = [s.strip() for s in sentences if _ACETABULAR_KW.search(s)]
    return " ".join(kept).strip()


def build_ml_text(cro_text: str = "", crr_text: str = "") -> str:
    """Représentation unique vue par le modèle ML (train ET inférence).

    Composition (zones propres uniquement, jamais le corps opératoire brut) :
        phrases acétabulaires du CRO
      + ligne DIAGNOSTIC du CRO
      + phrases acétabulaires du CRR
    """
    parts = []
    if cro_text:
        parts.append(acetabular_sentences(cro_text))
        parts.append(cro_diagnostic_zone(cro_text))
    if crr_text:
        parts.append(acetabular_sentences(crr_text))
    merged = " ".join(p for p in parts if p)
    return normalize(merged)


# --------------------------------------------------------------------------
# Pertinence : le compte rendu décrit-il bien une fracture de l'acétabulum ?
# --------------------------------------------------------------------------
# Termes DÉFINISSANT l'acétabulum (volontairement précis pour éviter de
# confondre avec d'autres régions : « colonne vertébrale », « paroi
# abdominale », « face postérieure » d'un autre os, etc.).
_ACETABULAR_STRICT = re.compile(
    r"cotyle|cotyloid|acetabul|"
    r"colonne\s+(?:anterieure|posterieure)|"
    r"paroi\s+(?:anterieure|posterieure)|"
    r"bi[\s\-]?colo[nl]|judet|letournel|"
    r"hemi[\s\-]?transvers|transverse"
)


def has_acetabular_content(cro_text: str = "", crr_text: str = "") -> bool:
    """Vrai si les comptes rendus mentionnent l'anatomie acétabulaire.

    Sert de garde-fou : si AUCUN terme acétabulaire n'apparaît, le document
    concerne probablement un autre type de fracture -> on ne force pas une des
    trois classes (statut « Non déterminé / Autre fracture »).
    """
    full = normalize((cro_text or "") + " " + (crr_text or ""))
    return bool(full.strip() and _ACETABULAR_STRICT.search(full))
