"""
augmentation.py — Data Augmentation (DA) pour l'entraînement de DrBERT.

RÈGLE D'OR (à ne jamais enfreindre) : aucune technique ne doit jamais toucher
au terme médical qui a justifié le label (le "terme décisif"), ni à son
contexte anatomique immédiat. Une augmentation qui abîmerait "bicolonne" ou
"hémi-transverse postérieure" créerait un exemple d'entraînement au texte
incohérent avec son label -> c'est le pire type d'erreur pour un si petit
jeu de données (99 dossiers), donc chaque technique ci-dessous est conçue
pour être "protection-first" : on calcule d'abord les zones protégées, puis
on n'agit qu'en dehors.

Techniques implémentées :
  1. strip_boilerplate  : retire du bruit administratif non discriminant
                          (chef de pôle/service, téléphone, adresse). Ce bruit
                          est institutionnel et constant -> sur 99 exemples il
                          peut créer une fausse corrélation avec la classe.
  2. eda_word_level      : suppression / permutation aléatoires de MOTS NON
                          protégés (technique EDA, Wei & Zou 2019), adaptée
                          avec la contrainte de protection ci-dessus.
  3. ocr_noise           : bruit caractère par caractère imitant les artefacts
                          OCR déjà présents dans les données sources (le jeu
                          de données vient de PDF/OCR réels) ; entraîne DrBERT
                          à être robuste au MÊME type de bruit qu'il verra en
                          inférence.
  4. augment_text        : combine aléatoirement les trois techniques pour
                          produire UNE variante.
  5. augment_dataset     : point d'entrée principal. Génère un nombre de
                          variantes par classe (sur-échantillonnage ciblé sur
                          les classes minoritaires) et VÉRIFIE, pour chaque
                          variante, que le texte augmenté déclenche toujours
                          la même décision de règle que l'original quand
                          l'original en déclenchait une (garde-fou anti-
                          incohérence). Une variante qui échoue au contrôle
                          est rejetée (pas de retry infini, juste ignorée).

⚠ Ne jamais appeler augment_dataset sur les données de VALIDATION d'un pli de
validation croisée : l'augmentation ne doit s'appliquer qu'au TRAIN de chaque
pli (sinon fuite de données -> métriques artificiellement gonflées). Voir
train_drbert.py, qui applique cette règle.
"""
from __future__ import annotations

import random
import re
from typing import List, Sequence, Tuple

from .rules import RE_BICOL, RE_COLANT, RE_HEMI, RE_HTP, classify_text
from .text_norm import normalize

# --------------------------------------------------------------------------
# 1) ZONES PROTÉGÉES
# --------------------------------------------------------------------------
# Les mêmes motifs que rules.py (le terme qui décide la classe), + quelques
# ancres anatomiques proches (contexte clinique qu'on préfère aussi ne pas
# perturber, même si elles ne décident pas seules de la classe).
_PROTECTED_PATTERNS = [
    RE_BICOL, RE_HEMI, RE_HTP, RE_COLANT,
    re.compile(r"cotyle|cotyloid|acetabul|paroi\s+(?:anterieure?|posterieure?)|"
               r"judet|letournel|transvers"),
]


def find_protected_spans(text: str, margin: int = 4) -> List[Tuple[int, int]]:
    """Renvoie les intervalles [start, end) à ne jamais modifier.

    `margin` élargit chaque zone de quelques caractères pour ne pas tronquer
    un mot collé au terme protégé (utile pour le bruit caractère par
    caractère).
    """
    spans = []
    for pat in _PROTECTED_PATTERNS:
        for m in pat.finditer(text):
            s, e = m.start(), m.end()
            spans.append((max(0, s - margin), min(len(text), e + margin)))
    return _merge_spans(spans)


def _merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def _is_protected(pos: int, spans) -> bool:
    return any(s <= pos < e for s, e in spans)


# --------------------------------------------------------------------------
# 2) BOILERPLATE ADMINISTRATIF
# --------------------------------------------------------------------------
# Motifs purement institutionnels/administratifs. Construits pour ne JAMAIS
# recouper le vocabulaire médical (aucun ne contient de terme anatomique),
# donc pas besoin de vérifier les zones protégées ici.
_BOILERPLATE_PATTERNS = [
    re.compile(r"institut universitaire locomoteur et du sport\s*"),
    re.compile(r"chef de (?:pole|service)\s*:\s*[a-z][a-z .\-]{2,40}"),
    re.compile(r"service de [a-z][a-z '\-]{4,60}?(?=\s(?:chef|compte|diagnostic|indication|nom\s*:))"),
    re.compile(r"n[°o]?\s*de\s*tel\s*:?\s*[\d .]{6,}"),
    re.compile(r"adresse\s*:\s*[^:]{5,60}?(?=\s(?:n[°o]?\s*de\s*tel|anesthesiste|type\s*d|diagnostic|indication|$))"),
]


def strip_boilerplate(text: str) -> str:
    """Retire le bruit administratif non discriminant (voir motifs ci-dessus).

    Ne prend pas de spans protégés en argument : par construction, ces motifs
    sont uniquement administratifs et ne recoupent jamais un terme médical.
    """
    out = text
    for pat in _BOILERPLATE_PATTERNS:
        out = pat.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------
# 3) EDA — suppression / permutation de mots NON protégés
# --------------------------------------------------------------------------
def eda_word_level(text: str, protected_spans, rng: random.Random,
                    p_delete: float = 0.10, p_swap: float = 0.15) -> str:
    """Random deletion + random swap (EDA, Wei & Zou 2019), version protégée.

    Un mot qui chevauche une zone protégée n'est JAMAIS supprimé ni utilisé
    dans une permutation.
    """
    tokens = list(re.finditer(r"\S+", text))
    if len(tokens) < 8:
        return text  # texte trop court pour toucher sans risque

    words = [t.group(0) for t in tokens]
    is_protected = [
        _is_protected(t.start(), protected_spans) or _is_protected(t.end() - 1, protected_spans)
        for t in tokens
    ]

    # --- random deletion ---
    new_words, new_prot = [], []
    for w, p in zip(words, is_protected):
        if p or rng.random() > p_delete:
            new_words.append(w)
            new_prot.append(p)
    if len(new_words) < 8:          # garde-fou : jamais un texte trop vidé
        new_words, new_prot = words, is_protected
    words, is_protected = new_words, new_prot

    # --- random swap (mots adjacents, tous deux non protégés) ---
    n_swaps = max(1, int(len(words) * p_swap / 2))
    for _ in range(n_swaps):
        movable = [i for i in range(len(words) - 1)
                   if not is_protected[i] and not is_protected[i + 1]]
        if not movable:
            break
        i = rng.choice(movable)
        words[i], words[i + 1] = words[i + 1], words[i]

    return " ".join(words)


# --------------------------------------------------------------------------
# 4) Bruit caractère-par-caractère imitant l'OCR
# --------------------------------------------------------------------------
def ocr_noise(text: str, protected_spans, rng: random.Random,
              p: float = 0.02) -> str:
    """Bruit léger (suppression / duplication / permutation de caractère,
    insertion d'espace) hors des zones protégées, à faible probabilité `p`
    par caractère alphabétique éligible."""
    chars = list(text)
    out = []
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        if _is_protected(i, protected_spans) or not c.isalpha() or rng.random() > p:
            out.append(c)
            i += 1
            continue
        op = rng.choice(("drop", "dup", "swap", "space"))
        if op == "drop":
            pass
        elif op == "dup":
            out.append(c)
            out.append(c)
        elif op == "swap" and i + 1 < n and not _is_protected(i + 1, protected_spans):
            out.append(chars[i + 1])
            out.append(c)
            i += 1
        elif op == "space":
            out.append(c)
            out.append(" ")
        else:
            out.append(c)
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------
# 5) Orchestrateur : une variante = combinaison aléatoire des techniques
# --------------------------------------------------------------------------
def augment_text(text: str, rng: random.Random,
                  p_boilerplate: float = 0.6,
                  p_eda: float = 0.8,
                  p_ocr: float = 0.5,
                  eda_p_delete: float = 0.10,
                  eda_p_swap: float = 0.15,
                  ocr_p: float = 0.02) -> str:
    """Produit UNE variante augmentée de `text`.

    Les zones protégées sont recalculées après CHAQUE opération qui modifie
    la longueur du texte (le stripping change les offsets), pour que les
    étapes suivantes protègent les bonnes positions.
    """
    out = text
    if rng.random() < p_boilerplate:
        out = strip_boilerplate(out)
    if rng.random() < p_eda:
        spans = find_protected_spans(out)
        out = eda_word_level(out, spans, rng, p_delete=eda_p_delete, p_swap=eda_p_swap)
    if rng.random() < p_ocr:
        spans = find_protected_spans(out)
        out = ocr_noise(out, spans, rng, p=ocr_p)
    return normalize(out)


# --------------------------------------------------------------------------
# 6) Point d'entrée principal
# --------------------------------------------------------------------------
def augment_dataset(
    texts: Sequence[str],
    labels: Sequence[int],
    multiplier: dict,
    seed: int = 42,
    verify_rule_consistency: bool = True,
    **augment_kwargs,
) -> Tuple[List[str], List[int], dict]:
    """Construit un jeu (texts, labels) augmenté.

    `multiplier[classe]` = nombre total d'occurrences souhaité par exemple
    original de cette classe (1 = pas d'augmentation ; 3 = original + 2
    variantes). Les originaux sont TOUJOURS conservés intégralement.

    `verify_rule_consistency` (par défaut True) : si le texte original
    déclenchait une règle (bicolonne / hémi-transverse / colonne antérieure),
    on vérifie que la variante déclenche TOUJOURS la même règle. Sinon, la
    variante est rejetée silencieusement (comptée dans le rapport renvoyé)
    plutôt qu'utilisée — c'est le garde-fou anti-incohérence texte/label.

    Renvoie (texts_aug, labels_aug, rapport) où `rapport` détaille combien de
    variantes ont été gardées / rejetées, par classe (pour audit).
    """
    rng = random.Random(seed)
    out_texts: List[str] = list(texts)
    out_labels: List[int] = list(labels)
    report = {"kept": {}, "rejected": {}}

    for text, label in zip(texts, labels):
        n_total = int(multiplier.get(int(label), 1))
        n_variants = max(0, n_total - 1)
        kept = 0
        # on tente jusqu'à 3x le nombre demandé pour compenser les rejets
        attempts = 0
        max_attempts = n_variants * 3 + 3
        original_rule_label, _ = classify_text(text) if verify_rule_consistency else (None, None)
        while kept < n_variants and attempts < max_attempts:
            attempts += 1
            variant = augment_text(text, rng, **augment_kwargs)
            if not variant.strip():
                continue
            if verify_rule_consistency and original_rule_label is not None:
                new_rule_label, _ = classify_text(variant)
                if new_rule_label != original_rule_label:
                    report["rejected"][label] = report["rejected"].get(label, 0) + 1
                    continue
            out_texts.append(variant)
            out_labels.append(int(label))
            kept += 1
            report["kept"][label] = report["kept"].get(label, 0) + 1

    return out_texts, out_labels, report
