"""
classifier.py — Le cerveau d'inférence : fusion RÈGLES + ML.

Logique de fusion (priorité aux règles, car elles sont à 100 % de précision
sur les cas qu'elles couvrent) :

    1. On applique les règles sur les zones propres (CRO/diagnostic, CRR).
    2. On calcule aussi la prédiction du modèle ML (DoctoBERT si présent, sinon
       TF-IDF), si un modèle est disponible.
    3. Décision finale :
         - règle déclenchée            -> on prend la règle (confiance haute) ;
                                          si le ML est en désaccord, on signale
                                          « à vérifier » sans changer la décision.
         - règle silencieuse + ML       -> on prend le ML (confiance = proba ML).
         - règle silencieuse + pas de ML-> « indéterminé » (à classer à la main).

Dégradation gracieuse : sans aucun modèle entraîné, l'objet fonctionne en mode
« règles seules » (couvre ~94 % des cas). Avec TF-IDF, il couvre tout. DoctoBERT
remplace TF-IDF s'il a été entraîné.

L'objet renvoie un dictionnaire riche et traçable (label, nom, source,
confiance, zone, terme déclencheur, détail règle/ML, drapeau de désaccord),
pour que l'app Streamlit puisse tout expliquer à l'utilisateur.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

try:
    from . import config
    from .rules import classify_case as rule_classify
    from .text_zones import build_ml_text, cro_diagnostic_zone, has_acetabular_content
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import config
    from src.rules import classify_case as rule_classify
    from src.text_zones import build_ml_text, cro_diagnostic_zone, has_acetabular_content


# --------------------------------------------------------------------------
# Backends ML (chargés à la demande)
# --------------------------------------------------------------------------
class _TfidfBackend:
    name = "TF-IDF"

    def __init__(self, path: Path):
        import joblib
        self.pipe = joblib.load(path)

    def predict(self, text: str):
        proba = self.pipe.predict_proba([text])[0]
        classes = list(self.pipe.classes_)
        idx = int(proba.argmax())
        label = int(classes[idx])
        return label, float(proba[idx]), {int(c): float(p) for c, p in zip(classes, proba)}


class _DoctobertBackend:
    name = "DoctoBERT"

    def __init__(self, model_dir: Path):
        import torch
        from transformers import (AutoModelForSequenceClassification,
                                  AutoTokenizer)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        lm = model_dir / "label_map.json"
        if lm.exists():
            data = json.loads(lm.read_text())
            self.idx2label = {int(k): int(v) for k, v in data["idx2label"].items()}
        else:
            self.idx2label = {i: lab for i, lab in enumerate(config.LABEL_IDS)}

    def predict(self, text: str):
        enc = self.tokenizer([text], truncation=True, padding=True,
                             max_length=config.MAX_LEN, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
            proba = self.torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(proba.argmax())
        label = self.idx2label[idx]
        return label, float(proba[idx]), {
            self.idx2label[i]: float(p) for i, p in enumerate(proba)
        }


def _load_ml_backend(prefer: str = "doctobert"):
    """Charge le modèle ML. Par défaut : DoctoBERT local uniquement.

    Conformément au cahier des charges, TF-IDF n'est PLUS utilisé comme moteur
    de l'interface. Si DoctoBERT n'est pas disponible (modèle absent, ou
    transformers/torch non installés), on retourne None : le système fonctionne
    alors en mode « règles seules » (et le signale), sans repli silencieux.
    """
    if prefer == "tfidf":
        # chemin explicite, hors interface (réservé à l'expérimentation)
        if config.TFIDF_MODEL.exists():
            try:
                return _TfidfBackend(config.TFIDF_MODEL)
            except Exception as e:
                print(f"[info] TF-IDF non chargé ({e}).")
        return None

    # DoctoBERT local
    if config.DOCTOBERT_DIR.exists() and (config.DOCTOBERT_DIR / "config.json").exists():
        try:
            return _DoctobertBackend(config.DOCTOBERT_DIR)
        except Exception as e:
            print(f"[info] DoctoBERT présent mais non chargé ({e}). "
                  f"Vérifiez l'installation de transformers/torch.")
            return None
    print(f"[info] Aucun modèle DoctoBERT trouvé dans {config.DOCTOBERT_DIR}. "
          f"Mode règles seules.")
    return None


# --------------------------------------------------------------------------
# Résultat
# --------------------------------------------------------------------------
@dataclass
class Prediction:
    label: Optional[int]            # 1, 2, 3 ou None
    label_name: str                 # libellé lisible
    source: str                     # 'regle' | 'ml' | 'aucune'
    confidence: float               # 0..1
    rule_label: Optional[int]       # ce que la règle a dit (ou None)
    rule_zone: str                  # zone du déclenchement
    rule_term: Optional[str]        # terme déclencheur
    ml_label: Optional[int]         # ce que le ML a dit (ou None)
    ml_confidence: Optional[float]
    ml_name: Optional[str]          # 'TF-IDF' ou 'DoctoBERT'
    ml_proba: Optional[dict]        # proba par classe
    disagreement: bool              # règle ≠ ML (informatif)
    needs_review: bool              # à vérifier par un humain
    diagnostic_zone: str            # zone diagnostique extraite (affichage)

    def as_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Classifieur
# --------------------------------------------------------------------------
class FractureClassifier:
    """Fusionne règles et ML. Réutilisable (charge le modèle une seule fois)."""

    def __init__(self, ml_backend="auto", prefer="doctobert"):
        if ml_backend == "auto":
            self.ml = _load_ml_backend(prefer=prefer)
        else:
            self.ml = ml_backend

    @property
    def ml_name(self) -> Optional[str]:
        return self.ml.name if self.ml else None

    def predict(self, cro_text: str = "", crr_text: str = "") -> Prediction:
        # --- 1) Règles (zones propres uniquement) ---
        rule = rule_classify(cro_text, crr_text)

        # --- 2) ML (si disponible) sur le texte centralisé ---
        ml_label = ml_conf = ml_proba = None
        ml_name = self.ml.name if self.ml else None
        if self.ml is not None:
            ml_text = build_ml_text(cro_text, crr_text)
            if ml_text.strip():
                ml_label, ml_conf, ml_proba = self.ml.predict(ml_text)

        # zone diagnostique pour l'affichage (CRO prioritaire, sinon vide)
        diag_zone = cro_diagnostic_zone(cro_text) if cro_text else ""

        # --- 3) Fusion ---
        disagreement = (
            rule.label is not None and ml_label is not None and rule.label != ml_label
        )

        if rule.label is not None:
            # La règle prime (haute précision démontrée).
            return Prediction(
                label=rule.label,
                label_name=config.LABELS[rule.label],
                source="regle",
                confidence=1.0,
                rule_label=rule.label,
                rule_zone=rule.zone,
                rule_term=rule.term,
                ml_label=ml_label,
                ml_confidence=ml_conf,
                ml_name=ml_name,
                ml_proba=ml_proba,
                disagreement=disagreement,
                needs_review=disagreement,  # signaler le désaccord sans changer la décision
                diagnostic_zone=diag_zone,
            )

        # Pas de règle déclenchée : le document décrit-il bien une fracture de
        # l'acétabulum ? (garde-fou contre une classification forcée)
        relevant = has_acetabular_content(cro_text, crr_text)

        # --- GARDE-FOU CONTENU NON ACÉTABULAIRE ---
        # Le modèle apprend désormais la classe 4 (voir config.LABEL_IDS), il
        # peut donc prédire lui-même « autre fracture ». Ce garde-fou reste
        # néanmoins utile : si le document ne relève pas de l'acétabulum, on
        # écarte la proposition du modèle plutôt que de le laisser affirmer une
        # classe 1/2/3. Quand le modèle prédit déjà 4, la réponse est identique.
        # Note : certains cas de classe 4 de la cohorte (fractures du bassin,
        # trans-iliaques) ne contiennent aucun terme acétabulaire et passent donc
        # par ce chemin plutôt que par le modèle — le résultat affiché reste
        # correct, mais la source indiquée sera « abstention » et non « ml ».
        if ml_label is not None and not relevant:
            return Prediction(
                label=config.OUT_OF_SCOPE_ID,
                label_name=config.LABELS[config.OUT_OF_SCOPE_ID],
                source="abstention",
                confidence=0.0,
                rule_label=None,
                rule_zone="",
                rule_term="contenu non acetabulaire",
                ml_label=ml_label,
                ml_confidence=ml_conf,
                ml_name=ml_name,
                ml_proba=ml_proba,
                disagreement=False,
                needs_review=True,
                diagnostic_zone=diag_zone,
            )

        if ml_label is not None and (ml_conf is None or ml_conf >= config.ML_MIN_CONFIDENCE):
            # Pas de règle, mais contenu acétabulaire pertinent et confiance
            # suffisante : on s'en remet au ML (confiance = proba ML).
            return Prediction(
                label=ml_label,
                label_name=config.LABELS[ml_label],
                source="ml",
                confidence=round(ml_conf, 3),
                rule_label=None,
                rule_zone="",
                rule_term=None,
                ml_label=ml_label,
                ml_confidence=ml_conf,
                ml_name=ml_name,
                ml_proba=ml_proba,
                disagreement=False,
                needs_review=(ml_conf < 0.60),  # faible confiance -> à vérifier
                diagnostic_zone=diag_zone,
            )

        # --- Aucune classe affirmée : « Non déterminé / Autre fracture » ---
        # Cas distincts pour l'audit : document non acétabulaire, confiance ML
        # insuffisante, ou aucun modèle disponible.
        if not relevant:
            why = "autre"          # le CR ne décrit pas une fracture de l'acétabulum
        elif ml_label is not None:
            why = "incertain"      # acétabulaire mais le modèle n'est pas assez sûr
        else:
            why = "aucune"         # pas de modèle (mode règles seules)
        return self._undetermined(why, diag_zone, ml_name, ml_label, ml_conf, ml_proba)

    def _undetermined(self, why, diag_zone, ml_name,
                      ml_label=None, ml_conf=None, ml_proba=None) -> "Prediction":
        """Construit une prédiction « Non déterminé / Autre fracture ».

        `why` ∈ {'autre', 'incertain', 'aucune'} explique l'absence de décision
        (visible seulement dans l'analyse avancée). On conserve la proposition du
        modèle (le cas échéant) pour information, sans l'affirmer.
        """
        return Prediction(
            label=None,
            label_name=config.OTHER_LABEL,
            source=why,
            confidence=round(ml_conf, 3) if ml_conf is not None else 0.0,
            rule_label=None,
            rule_zone="",
            rule_term=None,
            ml_label=ml_label,
            ml_confidence=ml_conf,
            ml_name=ml_name,
            ml_proba=ml_proba,
            disagreement=False,
            needs_review=True,
            diagnostic_zone=diag_zone,
        )
