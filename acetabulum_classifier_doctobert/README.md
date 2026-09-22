# Classification des fractures de l'acétabulum

Application hospitalière qui lit des comptes rendus médicaux — opératoires
(**CRO**) et/ou radiologiques (**CRR**) — extrait les informations utiles,
prédit le type de fracture et présente le tout dans une interface claire.

| Classe | Libellé |
|:------:|---------|
| 1 | Fracture colonne antérieure |
| 2 | Fracture bicolonne |
| 3 | Fracture colonne antérieure + hémi-transverse postérieure |

Le système utilise automatiquement la meilleure information disponible : CRO
seul, CRR seul, ou les deux combinés.

---

## 1. Architecture

**Des règles médicales de haute précision décident en priorité ; DoctoBERT
(chargé localement) tranche les cas que les règles ne couvrent pas.** TF-IDF
n'est pas utilisé par l'interface.

```
        CRO (.pdf)            CRR (.pdf)
            │                     │
            ▼                     ▼
   Lecture du texte (OCR déjà présent)        document_reader.py
            │
   ┌────────┴───────────────────────────┐
   ▼                                     ▼
 EXTRACTION STRUCTURÉE            ZONES PROPRES                 extraction.py
 (patient, acteurs, codes,        (CRO : ligne DIAGNOSTIC ;     text_zones.py
  sections, matériel…)             CRR : phrases acétabulaires)
            │                                     │
            │                        ┌────────────┴───────────┐
            │                        ▼                        ▼
            │                     RÈGLES                DoctoBERT (local)
            │                  (haute précision)        models/doctobert/     classifier.py
            │                        │                        │
            │                        └───────────┬────────────┘
            │                                    ▼
            │                   FUSION : la règle prime ; sinon DoctoBERT ;
            │                   désaccord → drapeau « à vérifier »
            ▼                                    │
   Résumé extractif  ◄───────────────────────────┘                            summarize.py
            │
            ▼
   Interface Streamlit en 3 pages                                            streamlit_app.py
```

### Pourquoi « règles d'abord »

Mesures sur les **99 dossiers annotés** (répartition 50 / 36 / 13), en
validation croisée stratifiée à 5 plis :

| Approche | Couverture / Accuracy | Précision / Macro-F1 |
|----------|:---------------------:|:--------------------:|
| Règles seules (zones propres) | **98,0 %** de couverture (97/99) | **100 %** de précision (0 erreur) |
| DoctoBERT seul | **0,960–0,970** d'accuracy | **0,939–0,947** de macro-F1 |
| Système fusionné (règles + DoctoBERT) | à mesurer sur votre machine ⚠ | à mesurer sur votre machine ⚠ |

⚠ `evaluate_rules.py` mesure les règles seules ; `train_doctobert.py` mesure
DoctoBERT seul. Le chiffre de fusion (ce qu'un utilisateur voit réellement)
n'était mesuré par aucun script — un chiffre figurait ici auparavant sans
script pour le reproduire, il a été retiré. **`evaluate_fusion.py`** comble
ce manque (voir section 9) : il réentraîne un modèle par pli comme
`train_doctobert.py`, mais fait ensuite passer chaque prédiction par la VRAIE
logique de fusion de `classifier.py` (pas une réimplémentation séparée).

DoctoBERT (`doctolib-lab/doctobert-fr-base`) a été retenu comme modèle par
défaut après comparaison directe, sur ce même pipeline d'extraction : ~0,96 /
~0,94 contre 0,68–0,70 / 0,64–0,67 pour Dr-BERT/DrBERT-7GB (modèle seul, sans
les règles, sans augmentation de données). Voir le dossier
`acetabulum_classifier_drbert/` pour la version DrBERT (avec Data
Augmentation, qui porte son score à 0,788 / 0,76 — voir son README).

Les règles, restreintes aux **zones diagnostiques propres**, ne se trompent
jamais sur les ~97/99 cas qu'elles couvrent. DoctoBERT n'intervient que sur
les 2 cas restants (formulations « des 2 colonnes » non contractées, que
l'expert classe en 1 mais que la règle ne déclenche jamais volontairement).
On ne lit **jamais le corps opératoire** d'un CRO : il contient des pièges
(« vis bi-colonne 3.5 mm » = une vis ; « ostéosynthèse des 2 colonnes » = une
technique) qui feraient chuter la précision.

### Pièges médicaux gérés

- « bicolonne / bi-colonne / bi colonne » → classe 2 ; mais « deux colonnes »,
  « des 2 colonnes », « colonne antérieure **et** postérieure » → restent classe 1.
- « hémi-transverse postérieure » / « HTP » / « CA+HTP » → classe 3.
- « paroi antérieure + colonne antérieure » → reste classe 1 ; le « + » seul ne
  déclenche jamais la classe 3.

### Robustesse à l'extraction PDF (pypdf)

`text_norm.py` recolle certains mots-clés médicaux (fracture, colonne,
antérieure, bicolonne, hémi-transverse…) quand un espace parasite les a
coupés en deux — observé concrètement avec `pypdf>=6` sur certains PDF
(« ante rieure » au lieu de « anterieure »). Cette version de pypdf insère
parfois un espace entre deux glyphes selon leur position dans le PDF,
indépendamment de la qualité de l'OCR source. Liste de mots explicite,
volontairement au singulier (voir `text_norm.py`).

### Zone diagnostique : bloc complet, pas la 1ère ligne

`cro_diagnostic_zone` (`text_zones.py`) lit les sections DIAGNOSTIC et
INDICATION dans leur intégralité — jusqu'à l'étiquette suivante (autre
section ou champ en ligne), pas seulement leur première ligne. Raison : la
mise en page/OCR coupe parfois la phrase diagnostique sur 2-3 lignes
physiques avant la vraie fin de section ; le terme décisif peut être sur la
2e ou 3e ligne. Ce changement fait passer la couverture des règles de 94/99 à
**97/99** (vérifié dossier par dossier : 0 régression, 0 nouvelle erreur sur
les 99 cas). Toujours restreint à DIAGNOSTIC + INDICATION : le corps
opératoire n'est toujours jamais lu.

---

## 2. Les trois pages

1. **Analyse d'un nouveau patient** — chargez un CRO et/ou un CRR. L'app affiche
   la classe prédite, la confiance, la méthode (règle / DoctoBERT), les éléments
   justifiant la décision (zone diagnostique + terme surligné), les informations
   extraites, les sections du CR, et un résumé automatique.
2. **Tableau global des dossiers** — un tableau de tous les dossiers présents
   dans `data/raw/` (nom, naissance, date du CR, type de fracture, opérateur,
   codes CCAM…). Sélectionnez un dossier pour ouvrir sa **fiche détaillée**.
3. **Statistiques** — répartition par type de fracture, CRO / CRR / CRO+CRR,
   activité par opérateur, codes CCAM, interventions par année.

### Extraction des informations

Le module `extraction.py` lit la mise en page **deux colonnes** des CRO
(`Opérateur : … Nom : …` sur une même ligne) en repérant toutes les étiquettes
connues et en prenant, pour chacune, le texte jusqu'à l'étiquette suivante.
Champs couverts : patient (nom, prénom, naissance, sexe, âge), date du CR,
opérateur/chirurgien, assistant, interne, anesthésiste, type d'anesthésie,
radiologue (CRR), codes Diagnostic et CCAM, matériel et fabricant, et les
sections (Diagnostic, Indication, Voies d'abord, Table, Installation,
Intervention, Consignes post-opératoires ; Technique / Résultat / Conclusion
pour le CRR). **Un champ absent est affiché « Non trouvé »** — jamais d'erreur.

### Résumé automatique

DoctoBERT est un modèle **encodeur** : il classe, il ne génère pas de texte.
Le résumé est donc **extractif** (sélection de phrases réellement présentes
dans le CR). C'est volontaire : cela garantit qu'**aucune information n'est
inventée**.

---

## 3. Installation

```bash
cd acetabulum_classifier
python -m venv .venv && source .venv/bin/activate   # optionnel
pip install -r requirements.txt
```

---

## 4. Utilisation

Placez vos données dans **`data/raw/`** : les `.pdf` (CRO et CRR, le suffixe
`_CRR` identifie les CRR) et le fichier `Excel_labels_fractures.xlsx`.

### a) Entraîner DoctoBERT (sur votre machine, accès à huggingface.co requis)

```bash
python -m src.build_dataset      # data/processed/dataset.csv (+ rapport)
python -m src.evaluate_rules     # vérifie les règles : ~98 % couverture, 100 % précision
python -m src.train_doctobert    # -> models/doctobert/  (+ métriques en validation croisée)
python -m src.evaluate_fusion    # score du système COMPLET (règles + DoctoBERT) — section 9
```

### b) Lancer l'application

```bash
streamlit run app/streamlit_app.py
```

L'app charge **DoctoBERT depuis `models/doctobert/`**. Si le modèle est
absent, elle passe en **mode « règles seules »** (couvre ~95 % des cas) et
l'indique clairement dans la barre latérale — sans repli silencieux sur TF-IDF.

### Prédire en ligne de commande

```bash
python -m src.predict_file --cro chemin/CRO.pdf --crr chemin/CRR.pdf
python -m src.predict_file --cro chemin/CRO.pdf      # CRO seul
python -m src.predict_file --crr chemin/CRR.pdf      # CRR seul
```

---

## 5. Vérifier que tout est correct

- `python -m src.evaluate_rules` → **0 erreur** sur les cas couverts (~97/99) et
  liste des 2 cas délégués à DoctoBERT.
- La barre latérale de l'app indique **« DoctoBERT · local »** si le modèle
  est bien chargé depuis `models/doctobert/`.
- Chaque prédiction montre la **zone diagnostique**, le **terme surligné**, la
  décision de la règle ET de DoctoBERT, et un **drapeau « à vérifier »** en
  cas de désaccord.
- `python -m src.evaluate_fusion` → score et matrice de confusion du système
  **complet** (règles + DoctoBERT), pas juste ses deux moitiés séparément
  (section 9).

---

## 6. Structure du projet

```
acetabulum_classifier/
├── README.md  ·  requirements.txt
├── .streamlit/config.toml        # thème clinique
├── src/
│   ├── config.py                 # chemins, colonnes Excel, classes, hyperparamètres
│   ├── text_norm.py              # normalisation
│   ├── document_reader.py        # lecture ZIP+OCR / PDF ; détection CRO vs CRR
│   ├── text_zones.py             # zones propres + texte ML centralisé
│   ├── rules.py                  # règles médicales haute précision
│   ├── extraction.py             # extraction structurée (champs + sections)
│   ├── summarize.py              # résumé extractif (sans hallucination)
│   ├── classifier.py             # fusion règles + DoctoBERT (TF-IDF non utilisé)
│   ├── registry.py               # registre de tous les dossiers (tableau global)
│   ├── build_dataset.py          # Excel + PDF -> dataset.csv
│   ├── evaluate_rules.py         # rapport couverture/précision des règles SEULES
│   ├── train_doctobert.py        # fine-tuning DoctoBERT + validation croisée
│   ├── evaluate_fusion.py        # score du système COMPLET (règles+DoctoBERT), voir section 9
│   ├── mcnemar_compare.py        # test de McNemar (significativité), voir section 11
│   ├── bootstrap_ci.py           # intervalles de confiance bootstrap, voir section 12
│   ├── multiseed_eval.py         # robustesse multi-graines, voir section 12
│   ├── freeze_environment.py     # fige les versions exactes, voir section 12
│   ├── train_tfidf.py            # (optionnel, expérimental — non utilisé par l'app)
│   └── predict_file.py           # prédiction en ligne de commande
├── app/streamlit_app.py          # interface 3 pages + fiche détail
├── data/raw/                     # <- y placer les .pdf et l'Excel
├── data/processed/               # généré : dataset.csv, rapports
└── models/                       # généré : doctobert/
```

---

## 7. Limites et honnêteté

Ces limites sont formulées pour être reprises dans la section *Limitations*
d'un article. Voir aussi `JUSTIFICATION_HYPERPARAMETRES.md`.

- **Cohorte de développement, pas validation externe.** Les 99 dossiers ont
  servi à la fois à concevoir les règles et à mesurer la performance ; les
  règles ont été affinées sur les cas qui échouaient parmi ces 99. Le 97/99 à
  100 % de précision est une performance **de développement**, pas une
  estimation de généralisation. Faute de nouveaux CRO indépendants disponibles,
  aucune validation externe n'a pu être conduite — à présenter comme tel ; une
  validation prospective/externe est un travail futur nécessaire.
- **Annotation par un seul expert.** Vérité terrain établie par un Professeur de
  chirurgie orthopédique, sans double annotation ni kappa inter-annotateurs.
- **Pas de comparaison à un LLM généraliste.** Choix délibéré d'encodeurs
  spécialisés du français médical (adaptation domaine, déploiement local /
  confidentialité RGPD, reproductibilité, pas de dépendance API, coût réduit).
  Une comparaison à des LLM (zero/few-shot) fait partie des travaux futurs — ce
  n'est pas un jugement sur l'inutilité des LLM.
- **Classe 4 (hors périmètre) non évaluable sur cette cohorte** : aucun exemple
  positif dans les 99 dossiers (voir section 10). Mécanisme de sécurité, pas une
  4ᵉ classe apprise.
- **Hyperparamètres non optimisés sur ce jeu** : valeurs conventionnelles
  BERT/RoBERTa (voir `JUSTIFICATION_HYPERPARAMETRES.md`).
- **Significativité statistique** : gain de la DA testé par McNemar
  (`mcnemar_compare.py`, section 11).
- **Reproductibilité** : graine fixée avant `from_pretrained` ; multi-graines
  recommandé pour l'article vu l'instabilité du fine-tuning sur petits jeux.
- L'outil assiste le clinicien, il ne le remplace pas : décisions à faible
  confiance ou en désaccord règle/DoctoBERT marquées « à vérifier ».

---

## 8. Ce dossier vs. `acetabulum_classifier_drbert/`

Les deux dossiers sont des copies **indépendantes** du même pipeline
(extraction, règles, fusion, app) : seul le modèle ML change. Aucune
dépendance entre les deux ; chacun a ses propres `data/`, `models/`.

| | Ce dossier | `acetabulum_classifier_drbert/` |
|---|---|---|
| Modèle | `doctolib-lab/doctobert-fr-base` | `Dr-BERT/DrBERT-7GB` |
| Accuracy seul (CV) | 0,960–0,970 | 0,68–0,70 sans DA · **0,788 avec DA** |
| Macro-F1 seul (CV) | 0,939–0,947 | 0,64–0,67 sans DA · **0,758–0,765 avec DA** |
| Data Augmentation | non (pas nécessaire au vu du score) | oui, `src/augmentation.py` |
| Script d'entraînement | `train_doctobert.py` | `train_drbert.py` |

---

## 9. Score du système complet (règles + DoctoBERT)

`evaluate_rules.py` mesure les règles seules. La validation croisée de
`train_doctobert.py` mesure DoctoBERT seul. **Aucun des deux ne mesure ce
qu'un utilisateur voit réellement** : la fusion de `classifier.py` (la règle
prime si elle se déclenche ; sinon DoctoBERT s'il est assez confiant ; sinon
« indéterminé »).

```bash
python -m src.evaluate_fusion
```

### Comment, sans dupliquer la logique de fusion

Le script réentraîne un modèle par pli (mêmes 5 plis, même graine que
`train_doctobert.py`), enveloppe ce modèle dans un petit adaptateur qui lui
donne l'interface attendue par `FractureClassifier`, puis construit un
**vrai** `FractureClassifier` avec ce modèle et appelle sa méthode
`.predict(cro_text, crr_text)` — exactement le code qui tourne dans l'app et
`predict_file.py`. Aucune règle de fusion n'est réécrite séparément.

### Ce qui est affiché

Trois blocs mesurés sur les **mêmes 5 plis** (règles seules / DoctoBERT seul
/ fusion), puis un tableau récapitulatif. Pour chaque bloc : couverture,
précision sur les cas couverts, accuracy et macro-F1 globales (un cas
**« indéterminé » compte comme une erreur**, jamais retiré silencieusement du
calcul), et une matrice de confusion avec une colonne dédiée « Indéterminé ».

### Coût

Comme il réentraîne un modèle par pli (comme `train_doctobert.py`), la durée
est comparable — plusieurs minutes sur GPU pour 99 dossiers.

---

## 10. Classe 4 — « Autre fracture / hors périmètre »

Judet-Letournel décrit **10 types** de fractures acétabulaires ; ce système
n'en couvre que 3. La classe 4 empêche le modèle de forcer une des 3 classes
sur un CR décrivant une fracture non couverte (paroi postérieure isolée,
transverse pure, fracture en T).

**Ce n'est PAS un 4ᵉ neurone du modèle** : la cohorte contient 0 exemple de
classe 4, donc rien à apprendre. Elle est décidée par le **système**
(`config.LABEL_IDS = [1,2,3]`, `OUT_OF_SCOPE_ID = 4`) via :
1. **Règle hors-périmètre** (`rules.classify_out_of_scope`), testée après les
   règles 1/2/3 (un terme in-périmètre prime toujours) ;
2. **Abstention** (`classifier.py`) : contenu non acétabulaire ou confiance
   insuffisante → classe 4 au lieu d'une classe forcée.

Vérifié sur les 99 dossiers : 0 basculement à tort, couverture/précision
inchangées. Sensibilité non chiffrable ici (aucun exemple positif — voir
Limites).

---

## 11. Significativité statistique — test de McNemar

`mcnemar_compare.py` teste si le gain de la DA est significatif plutôt que
constaté à l'œil, en comparant les prédictions **appariées** (mêmes dossiers,
mêmes plis) : DoctoBERT / DoctoBERT+DA / DrBERT / DrBERT+DA.

**Méthode** : McNemar (Dietterich, 1998), avec bascule automatique sur le test
binomial exact quand les cas discordants sont peu nombreux (< 25).

```bash
python -m src.train_doctobert --save-oof --oof-tag doctobert                 # avec DA
python -m src.train_doctobert --no-augment --save-oof --oof-tag doctobert_noaug
# rassembler les CSV oof/ (des deux forks) dans un même dossier, puis :
python -m src.mcnemar_compare
```

**Sorties** (`data/processed/mcnemar/`) : `mcnemar_results.csv` + un tableau de
contingence 2×2 par paire. Copiez les 4 CSV `oof/` des deux forks dans un même
dossier pour la comparaison croisée modèle-vs-modèle.

---

## 12. Reproductibilité et robustesse statistique (pour publication)

Trois outils complémentaires, identiques au fork DrBERT. Ils s'appuient sur les
prédictions out-of-fold produites par `train_doctobert.py --save-oof` (sauf
`multiseed_eval.py`, qui réentraîne).

### 12.1 Versions figées — `freeze_environment.py`

Capture les versions EXACTES installées sur votre machine (pypdf 5.x vs 6.x
change les résultats — d'où l'importance de figer VOS versions) :

```bash
python -m src.freeze_environment
```

Produit `requirements-lock.txt` + `environment-info.txt`. Reproduire :
`pip install -r requirements-lock.txt`.

### 12.2 Intervalles de confiance — `bootstrap_ci.py`

Bootstrap (10 000 rééchantillons) → IC à 95 % de l'accuracy et du macro-F1.

```bash
python -m src.bootstrap_ci
python -m src.bootstrap_ci --n-boot 20000
```

Produit `data/processed/bootstrap/bootstrap_ci.csv`. **Incertitude
d'échantillonnage**, à modèle figé (rapide).

### 12.3 Robustesse aux graines — `multiseed_eval.py`

Relance la validation croisée sur plusieurs graines → moyenne ± écart-type
(comme l'article DoctoBERT, sur 5 graines).

```bash
python -m src.multiseed_eval --seeds 42 43 44 45 46
python -m src.multiseed_eval --seeds 42 43 44 45 46 --no-augment
```

Produit `data/processed/multiseed/`. **Incertitude d'entraînement**, à jeu figé.
⚠ LONG (réentraîne graines × plis modèles).

### Lequel utiliser

| Question | Outil | Coût |
|---|---|---|
| « Mes versions sont-elles figées ? » | `freeze_environment` | instantané |
| « Quelle est la plage plausible du score ? » | `bootstrap_ci` | secondes |
| « Le score est-il stable si je relance ? » | `multiseed_eval` | long (GPU) |
| « Le gain de la DA est-il significatif ? » | `mcnemar_compare` (section 11) | secondes |

Note DoctoBERT : `AUGMENT_ENABLED = False` recommandé pour le déploiement (la DA
dégrade légèrement la fusion, voir section 11). Le `--compare-baseline` et le
`--no-augment` de `multiseed_eval` restent utilisables pour l'analyse.
