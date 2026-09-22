"""
text_norm.py — Normalisation de texte robuste pour l'OCR médical français.

La normalisation est CENTRALISÉE ici : règles et modèles ML doivent voir
exactement le même texte, sinon on introduit un décalage train/inférence
(« train/serve skew »).

Choix de normalisation (justifiés par les données OCR observées) :
  - minuscules               : « Bicolonne » == « bicolonne »
  - suppression des accents  : l'OCR produit « antérieure » ET « anteriure »
  - \xa0 (espace insécable) et \r -> espace simple (très fréquents dans l'OCR)
  - compression des espaces multiples
  - recollage des mots-clés médicaux coupés par un espace parasite (voir
    _defragment_keywords) : observé avec pypdf >= 6 sur certains PDF, où le
    texte extrait contient p. ex. « ante rieure » au lieu de « anterieure »
    (changement de détection d'espace entre glyphes selon la version de la
    bibliothèque, pas un problème des données sources).
"""
import re
import unicodedata


def strip_accents(text: str) -> str:
    """Retire les diacritiques (é -> e, ê -> e, ç -> c...)."""
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


# --------------------------------------------------------------------------
# Recollage de mots-clés fragmentés par l'extraction PDF
# --------------------------------------------------------------------------
# Observé concrètement (dossier 1803531, pypdf 6.13.3) : « frac ture »,
# « ante rieure » au lieu de « fracture », « anterieure ». Certaines versions
# de pypdf insèrent un espace entre deux glyphes quand leur espacement dans
# le PDF dépasse un seuil interne — ça n'a rien à voir avec un OCR de
# mauvaise qualité, c'est la bibliothèque d'extraction elle-même.
#
# On ne recolle QUE les mots de cette liste explicite (pas n'importe quelle
# paire de syllabes) : reste auditable, et les mots sont tous assez longs
# (6+ lettres) pour qu'un recollage accidentel sur du texte normal soit
# pratiquement impossible. Liste = vocabulaire qui apparaît dans les zones
# lues par les règles (text_zones.py) — inutile d'être exhaustif au-delà.
#
# Volontairement AU SINGULIER (pas de « s » final) : un « s » de pluriel est
# un caractère isolé très fréquent en début d'un autre mot français (« s'étendant »,
# « sans »...) — l'inclure avec un espace optionnel devant risquait d'avaler
# le début du mot suivant. Le « s » du pluriel, quand il existe dans le texte
# original, reste naturellement collé au radical recollé (les règles de
# rules.py tolèrent déjà le pluriel via leurs propres `s?`).
_DEFRAG_KEYWORDS = [
    "fracture", "anterieure", "anterieur", "posterieure", "posterieur",
    "colonne", "bicolonne", "hemitransverse", "hemitransversale",
    "transverse", "transversale", "cotyle", "acetabulum", "acetabulaire",
    "paroi", "judet", "letournel", "comminutive", "comminutif", "luxation",
    "articulaire", "impaction", "subluxation",
]
_DEFRAG_PATTERNS = [
    (re.compile(r"\s?".join(re.escape(c) for c in kw) + r"\b", re.IGNORECASE), kw)
    for kw in sorted(_DEFRAG_KEYWORDS, key=len, reverse=True)
]


def _defragment_keywords(text: str) -> str:
    """Recolle les mots-clés de _DEFRAG_KEYWORDS coupés par un espace
    parasite inséré au milieu du mot (voir note ci-dessus)."""
    for pattern, kw in _DEFRAG_PATTERNS:
        text = pattern.sub(kw, text)
    return text


def normalize(text: str) -> str:
    """Normalise un texte pour la comparaison par règles et la vectorisation ML.

    Idempotente : normalize(normalize(x)) == normalize(x).
    """
    if not text:
        return ""
    text = text.replace("\xa0", " ").replace("\r", " ")
    text = strip_accents(text)
    text = text.lower()
    text = _defragment_keywords(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
