"""
rules.py — Classifieur médical à RÈGLES haute précision.

Principe : ne se prononcer QUE lorsqu'un terme diagnostique non ambigu est
présent dans une zone propre (cf. text_zones.py). Sinon, renvoyer None et
laisser le modèle ML décider.

Mesuré sur les 99 dossiers annotés : 94 % de couverture à 100 % de précision
(0 erreur parmi les cas couverts). Les ~5 cas restants sont délégués au ML.

Priorité de décision (l'ordre compte) :
    1. « bicolonne / bi-colonne / bi colonne »  -> classe 2
    2. « hémi-transverse » ou « HTP / CA+HTP »   -> classe 3
    3. « colonne antérieure »                    -> classe 1
    4. sinon                                     -> None (indéterminé)

Pourquoi cet ordre protège des pièges :
  - « paroi antérieure + colonne antérieure » : ne matche jamais HÉMI/HTP,
    donc reste classe 1 (le « + » seul n'a aucun effet).
  - « deux colonnes / des 2 colonnes / colonne antérieure et postérieure » :
    NE déclenchent PAS la classe 2 (l'expert les classe en 1). Seules les
    formes contractées « bicolonne / bi-colonne / bi colonne » déclenchent 2.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .text_norm import normalize
from .text_zones import acetabular_sentences, cro_diagnostic_zone

# --------------------------------------------------------------------------
# Motifs (s'appliquent sur du texte NORMALISÉ : minuscules, sans accents)
# --------------------------------------------------------------------------
# Classe 2 — uniquement la forme contractée « bicolonne » et ses variantes
# d'orthographe/OCR (bi colonne, bi-colonne, bicolonnes, bicolone, bicolonna).
RE_BICOL = re.compile(r"bi[\s\-]?colo[nl]{1,2}[ae]?s?")

# Classe 3 — « hémi-transverse » DOIT être qualifiée de « postérieure » à
# proximité immédiate. Sans cette vérification, un texte disant « hémi-
# transverse ANTÉRIEURE » (orientation opposée, hors périmètre) déclenchait
# la classe 3 à tort — piège réel rencontré : « fracture luxation
# hémi-transverse antérieure et de la paroi postérieure du cotyle » contient
# à la fois « hemi transvers » et « posterieur » (dans « paroi postérieure »)
# mais ne décrit PAS le patron classe 3 (colonne antérieure + hémi-transverse
# POSTÉRIEURE) : c'est l'orientation inverse, associée à une paroi postérieure,
# un patron non couvert relevant de la classe 4.
# On exige donc « postérieure » DANS LA MÊME EXPRESSION que « hémi-transverse »
# (jusqu'à 2 mots d'écart, pour tolérer « hémi-transverse postérieure du
# cotyle »), et on rejette explicitement le cas où le mot immédiatement après
# « hémi-transverse » est « antérieure ».
RE_HEMI = re.compile(
    r"hemi[\s\-]?transvers[a-z]*\s+posteri[eu]*r?e?s?"
    r"|posteri[eu]*r?e?s?\s+hemi[\s\-]?transvers"
)
_RE_HEMI_ANT_PIEGE = re.compile(r"hemi[\s\-]?transvers[a-z]*\s+anteri[eu]*r?e?s?")
# Classe 3 — abréviations expertes « HTP » et « CA + HTP ».
RE_HTP = re.compile(r"\bhtp\b|\bca\s*\+\s*htp\b")

# Classe 1 — « colonne antérieure » (tolère typos OCR : colonna, colone,
# anteriur, anterieur sans 'e' final).
RE_COLANT = re.compile(r"colo[nl]{1,2}[ae]?s? anteri[eu]*r?e?s?")
# Exclusion piège réel : « fragment ... venant s'interposer entre [X] et la
# colonne antérieure » utilise « colonne antérieure » comme repère spatial
# (où se loge le fragment), PAS comme site fracturé. Rejette le match si l'un
# de ces verbes/tournures de repère apparaît juste avant dans la phrase.
_RE_COLANT_REPERE = re.compile(
    r"interpos|s.interpose|venant se loger|venant s.intercaler"
    r"|coince[e]? entre|pince[e]? entre"
)

# --------------------------------------------------------------------------
# Classe 4 — HORS PÉRIMÈTRE (types Judet-Letournel non couverts)
# --------------------------------------------------------------------------
# Judet-Letournel décrit 10 types de fractures acétabulaires ; ce système n'en
# couvre que 3. Sans ces motifs, un CR décrivant p. ex. une fracture de paroi
# postérieure isolée serait forcé dans l'une des 3 classes par le modèle —
# défaut de conception pour un outil de SCREENING.
#
# ⚠ Ces motifs ne sont testés QU'APRÈS l'échec des règles 1/2/3 (voir
# classify_case) : si un terme décisif in-périmètre est présent, il l'emporte
# toujours. Cet ordre est ce qui rend les motifs ci-dessous sûrs.
#
# Pièges réels rencontrés dans les données, qui dictent la prudence :
#   - « apophyses transverses L2-L3 » : vertébral, PAS acétabulaire -> le motif
#     transverse exclut explicitement « apophyse ».
#   - « paroi antérieure + colonne antérieure » : reste classe 1 (RE_COLANT
#     se déclenche avant, donc jamais évalué ici).
RE_OOS_PAROI_POST = re.compile(r"paroi[s]?\s+posteri[eu]*r?e?s?")
RE_OOS_COL_POST = re.compile(r"colo[nl]{1,2}[ae]?s?\s+posteri[eu]*r?e?s?")
# « transverse » SANS « hemi » (déjà capté classe 3) et SANS « apophyse »
# (piège vertébral) : fracture transverse pure / transversale.
RE_OOS_TRANSVERSE = re.compile(
    r"(?<!hemi )(?<!hemi-)(?<!apophyse )(?<!apophyses )\b(?:fracture|trait)\s+transvers"
)
# Fracture « en T » (type associé Judet-Letournel, hors périmètre).
RE_OOS_EN_T = re.compile(r"\bfracture\s+en\s+t\b")

_OOS_PATTERNS = [
    (RE_OOS_PAROI_POST, "paroi posterieure (hors perimetre)"),
    (RE_OOS_COL_POST, "colonne posterieure isolee (hors perimetre)"),
    (RE_OOS_TRANSVERSE, "fracture transverse (hors perimetre)"),
    (RE_OOS_EN_T, "fracture en T (hors perimetre)"),
]


@dataclass
class RuleResult:
    """Résultat d'une décision par règles."""
    label: Optional[int]          # 1, 2, 3, 4 (hors périmètre) ou None
    confidence: float             # 1.0 si une règle se déclenche, 0.0 sinon
    zone: str                     # zone propre où le terme a été trouvé
    term: Optional[str]           # libellé du déclencheur (traçabilité)
    source: str                   # 'regle' (toujours, ici)


def classify_out_of_scope(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Détecte un type de fracture acétabulaire HORS des 3 classes couvertes.

    À n'appeler QU'APRÈS échec de classify_text (les 3 classes couvertes ont
    toujours la priorité). Renvoie (4, terme) ou (None, None).
    """
    t = normalize(text)
    for pattern, term in _OOS_PATTERNS:
        if pattern.search(t):
            return 4, term
    return None, None


def classify_text(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Applique la cascade de règles à un texte DÉJÀ restreint à une zone propre.

    Renvoie (label, terme) ou (None, None) si aucun terme non ambigu.
    Le texte est normalisé par sécurité (idempotent).
    """
    t = normalize(text)
    if RE_BICOL.search(t):
        return 2, "bicolonne"
    m_hemi = RE_HEMI.search(t)
    if m_hemi:
        # La classe 3 est « colonne antérieure + hémi-transverse POSTÉRIEURE » :
        # les DEUX composants doivent être présents. Vérifié sur la cohorte :
        # 12 des 13 cas de classe 3 mentionnent explicitement « colonne
        # antérieure », le 13e utilise l'abréviation « CA+HTP » (où CA = colonne
        # antérieure), captée par RE_HTP plus bas.
        # Sans cette exigence, un texte décrivant une hémi-transverse
        # postérieure associée à une AUTRE lésion (p. ex. paroi postérieure,
        # sans atteinte de la colonne antérieure) serait classé 3 à tort :
        # il relève en réalité de la classe 4.
        if RE_COLANT.search(t):
            return 3, "colonne anterieure + hemi-transverse posterieure"
        # hémi-transverse postérieure SANS colonne antérieure -> patron non
        # couvert : on laisse la main aux règles hors-périmètre / au modèle.
        return None, None
    if RE_HTP.search(t):
        return 3, "HTP / CA+HTP"
    m = RE_COLANT.search(t)
    if m:
        # Piège réel rencontré : « ... la tête fémorale luxée en arrière et
        # la colonne antérieure du cotyle » (fragment qui s'interpose ENTRE
        # deux repères anatomiques). Ici « colonne antérieure » sert de point
        # de repère spatial, PAS d'assertion qu'elle est fracturée — le
        # diagnostic réel portait sur le toit et la colonne postérieure.
        # On rejette donc le match si un verbe d'interposition/repère apparaît
        # juste avant, dans la même phrase (fenêtre de 80 caractères).
        contexte_avant = t[max(0, m.start() - 80):m.start()]
        if not _RE_COLANT_REPERE.search(contexte_avant):
            return 1, "colonne anterieure"
    return None, None


def classify_case(cro_text: str = "", crr_text: str = "") -> RuleResult:
    """Décision par règles sur un dossier complet (CRO et/ou CRR).

    Stratégie : zone DIAGNOSTIC du CRO d'abord (la plus fiable), puis phrases
    acétabulaires du CRR. On ne scanne jamais le corps opératoire du CRO. On
    s'arrête au premier déclenchement.
    """
    # 1) Zone DIAGNOSTIC/INDICATION du CRO (seule zone fiable du CRO :
    #    on ne scanne JAMAIS le corps opératoire, qui contient des pièges
    #    « vis bi-colonne », « ostéosynthèse des 2 colonnes », etc.)
    if cro_text:
        diag = cro_diagnostic_zone(cro_text)
        label, term = classify_text(diag)
        if label is not None:
            return RuleResult(label, 1.0, "CRO/diagnostic", term, "regle")

    # 2) Phrases acétabulaires du CRR (le CRR n'a pas de corps opératoire :
    #    ses phrases acétabulaires sont du registre diagnostique)
    if crr_text:
        crr_sents = acetabular_sentences(crr_text)
        label, term = classify_text(crr_sents)
        if label is not None:
            return RuleResult(label, 1.0, "CRR/phrases", term, "regle")

    # 3) HORS PÉRIMÈTRE (classe 4) — testé en DERNIER, uniquement si aucune des
    #    3 classes couvertes ne s'est déclenchée. Cet ordre est ce qui garantit
    #    qu'un CR mentionnant à la fois « colonne antérieure » et « paroi
    #    postérieure » reste classé 1 : la règle in-périmètre a déjà tranché.
    if cro_text:
        label, term = classify_out_of_scope(cro_diagnostic_zone(cro_text))
        if label is not None:
            return RuleResult(label, 1.0, "CRO/diagnostic", term, "regle")
    if crr_text:
        label, term = classify_out_of_scope(acetabular_sentences(crr_text))
        if label is not None:
            return RuleResult(label, 1.0, "CRR/phrases", term, "regle")

    return RuleResult(None, 0.0, "", None, "regle")
