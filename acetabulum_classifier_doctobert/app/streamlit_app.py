"""
streamlit_app.py — Outil clinique d'aide à la classification des fractures de
l'acétabulum (interface « produit »).

UX :
  • Vue médecin épurée ; détails techniques uniquement en mode Expert / onglet
    « Analyse avancée ».
  • Parcours guidé « Nouveau patient » (Documents → Analyse → Résultat).
  • Tableau de bord riche, fiche patient moderne, historique des analyses.
Moteur (inchangé) : règle médicale fiable d'abord, puis DoctoBERT local.

Lancement :  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import glob
import io
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ui
from src import config
from src.classifier import FractureClassifier
from src.derive import age_label
from src.document_reader import detect_doc_type, file_id, read_document_from_bytes
from src.extraction import SECTION_DISPLAY, ExtractedInfo, extract
from src.registry import build_records, records_to_dataframe
from src.rules import RE_BICOL, RE_COLANT, RE_HEMI, RE_HTP
from src.summarize import build_summary

st.set_page_config(
    page_title="Acétabule — Aide à la décision",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)
ui.inject_css()

SOURCE_LABEL = {
    "regle": "Règle médicale",
    "ml": "Modèle DoctoBERT",
    "abstention": "Hors périmètre / abstention",
    "autre": "Hors périmètre acétabulaire",
    "incertain": "Confiance insuffisante",
    "aucune": "Indéterminé (règles seules)",
}


# ==========================================================================
# Ressources mises en cache
# ==========================================================================
@st.cache_resource(show_spinner="Chargement du modèle…")
def get_classifier() -> FractureClassifier:
    return FractureClassifier()


def _data_signature():
    files = sorted(glob.glob(str(config.DATA_RAW / "*.pdf")))
    return tuple((os.path.basename(f), int(os.path.getmtime(f))) for f in files)


@st.cache_data(show_spinner="Analyse des dossiers… (mise en cache)")
def get_records(_clf, signature):
    recs = build_records(_clf)
    return recs, records_to_dataframe(recs), {r.dossier_id: r for r in recs}


def has_data() -> bool:
    return bool(glob.glob(str(config.DATA_RAW / "*.pdf")))


# ==========================================================================
# Helpers d'affichage / logique
# ==========================================================================
def _g(info, attr):
    return getattr(info, attr, "") if info else ""


def _date_cr(cro_info, crr_info):
    if cro_info and cro_info.date_cr:
        return cro_info.date_cr
    return crr_info.date_cr if crr_info else ""


def _age(main_info, cro_info, crr_info):
    """Âge affichable : extrait du CR si présent, sinon calculé depuis la
    date de naissance (référence = date du compte rendu, sinon date du jour)."""
    return age_label(_g(main_info, "age"), _g(main_info, "date_naissance"),
                     _date_cr(cro_info, crr_info))


def display_name(info, dossier_id, anon):
    if anon:
        return f"Patient {dossier_id}".strip() if dossier_id else "Patient"
    name = " ".join(x for x in [_g(info, "nom"), _g(info, "prenom")] if x).strip()
    return name or (f"Dossier {dossier_id}" if dossier_id else "Patient")


def attention(pred, has_cro, has_crr):
    if pred is None or pred.label is None:
        src = pred.source if pred else "aucune"
        if src == "autre":
            msg = ("Ce compte rendu ne semble pas décrire une fracture de "
                   "l'acétabulum (aucune mention de l'anatomie cotyloïdienne). "
                   "Vérifiez qu'il s'agit du bon document.")
        elif src == "incertain":
            msg = ("Le type de fracture n'a pas pu être déterminé avec une "
                   "certitude suffisante. Relecture par un clinicien nécessaire.")
        else:
            msg = ("Le type de fracture n'a pas pu être déterminé automatiquement. "
                   "Relecture par un clinicien nécessaire.")
        return ("Non déterminé", "warn"), [(msg, "amber", "alert")]
    alerts, status = [], ("Analyse cohérente", "ok")
    if pred.disagreement:
        status = ("À confirmer", "warn")
        alerts.append(("Les éléments du compte rendu et l'analyse automatique ne "
                       "concordent pas entièrement. Vérification recommandée.", "amber", "alert"))
    elif pred.needs_review:
        status = ("À confirmer", "warn")
        alerts.append(("Niveau de certitude limité sur ce dossier. "
                       "Vérification recommandée.", "amber", "alert"))
    if not has_crr and pred.rule_label is None:
        alerts.append(("Aucun compte rendu radiologique (CRR) fourni et le compte "
                       "rendu opératoire ne précise pas explicitement le type. "
                       "L'ajout du CRR fiabiliserait l'analyse.", "blue", "info"))
    return status, alerts


def highlight(text):
    if not text or not text.strip():
        return "<span style='color:#94a3b8;font-style:italic'>(zone vide)</span>"
    import html as _h
    safe = _h.escape(text)
    for rx, color in [(RE_BICOL, ui.CLASS_META[2]["color"]),
                      (RE_HEMI, ui.CLASS_META[3]["color"]),
                      (RE_HTP, ui.CLASS_META[3]["color"]),
                      (RE_COLANT, ui.CLASS_META[1]["color"])]:
        safe = rx.sub(lambda m, c=color: f"<mark style='background:{c}22;padding:0 3px;"
                      f"border-radius:4px'>{m.group(0)}</mark>", safe)
    return safe


# ==========================================================================
# Fiche patient (réutilisée : nouveau patient, détail dossier, historique)
# ==========================================================================
def render_fiche(pred, cro_info, crr_info, cro_text, crr_text,
                 dossier_id, expert, anon):
    has_cro, has_crr = bool(cro_text), bool(crr_text)
    main_info = cro_info if (cro_info and (cro_info.nom or cro_info.prenom)) else (crr_info or cro_info)
    status, alerts = attention(pred, has_cro, has_crr)

    name = display_name(main_info, dossier_id, anon)
    meta = " · ".join(x for x in [
        ("" if anon else (f"Né(e) le {main_info.date_naissance}" if _filled(_g(main_info, "date_naissance")) else "")),
        (f"{main_info.sexe}" if _filled(_g(main_info, "sexe")) else ""),
        _age(main_info, cro_info, crr_info),
        (f"CR du {_date_cr(cro_info, crr_info)}" if _filled(_date_cr(cro_info, crr_info)) else ""),
        f"Dossier {dossier_id}",
    ] if x)
    ui.identity_card(name, pred.label if pred else None, meta, ui.doc_badges(has_cro, has_crr))

    tabs = st.tabs(["Synthèse", "Informations extraites", "Documents", "Analyse avancée"])

    # ---- Synthèse (clinique pure, sans doublon) ----
    with tabs[0]:
        ui.hero(pred.label if pred else None, status, ui.doc_badges(has_cro, has_crr))
        for msg, kind, ic in alerts:
            ui.alert(msg, kind, ic)
        summary = build_summary(cro_info, crr_info,
                                pred.label_name if pred else "", pred.source if pred else "",
                                anonymize=anon, dossier_id=dossier_id)
        ui.text_card("Résumé médical", summary, "ai")

    # ---- Informations extraites ----
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            if anon:
                ui.field_card("Patient", [("Identifiant", f"Dossier {dossier_id}"),
                                          ("Sexe", _g(main_info, "sexe")),
                                          ("Âge", _age(main_info, cro_info, crr_info))], "patient", True)
            else:
                ui.field_card("Patient", [
                    ("Nom", _g(main_info, "nom")), ("Prénom", _g(main_info, "prenom")),
                    ("Date de naissance", _g(main_info, "date_naissance")),
                    ("Sexe", _g(main_info, "sexe")), ("Âge", _age(main_info, cro_info, crr_info)),
                    ("IPP", _g(crr_info, "ipp"))], "patient", True)
        with c2:
            ui.field_card("Équipe médicale", [
                ("Opérateur / Chirurgien", _g(cro_info, "operateur")),
                ("Assistant", _g(cro_info, "assistant")),
                ("Interne", _g(cro_info, "interne")),
                ("Instrumentiste", _g(cro_info, "instrumentiste")),
                ("Anesthésiste", _g(cro_info, "anesthesiste")),
                ("Type d'anesthésie", _g(cro_info, "type_anesthesie")),
                ("Radiologue", _g(crr_info, "radiologue"))], "users", True)
        ui.field_card("Acte, codes & matériel", [
            ("Date du compte rendu", _date_cr(cro_info, crr_info)),
            ("Codes CCAM", _g(cro_info, "codes_ccam")),
            ("Codes diagnostic", _g(cro_info, "codes_diagnostic")),
            ("Matériel", _g(cro_info, "materiel")),
            ("Fabricant / Société", _g(cro_info, "fabricant") or _g(crr_info, "fabricant"))],
            "tag", True)

    # ---- Documents ----
    with tabs[2]:
        st.markdown(f"<div class='chip-row' style='margin-bottom:12px'>"
                    f"{_doc_chip('Compte rendu opératoire', has_cro)}"
                    f"{_doc_chip('Compte rendu radiologique', has_crr)}</div>",
                    unsafe_allow_html=True)
        blocks = []
        for info, tag in [(cro_info, "CRO"), (crr_info, "CRR")]:
            if info:
                for key, label in SECTION_DISPLAY.items():
                    blocks.append((label, info.sections.get(key, ""), tag))
        ui.sections_blocks(blocks)
        if expert:
            with st.expander("Texte brut extrait (OCR)"):
                if cro_text:
                    st.text_area("CRO", cro_text, height=220)
                if crr_text:
                    st.text_area("CRR", crr_text, height=220)
        else:
            st.markdown("<div class='muted'>Le texte brut OCR est disponible en mode Expert.</div>",
                        unsafe_allow_html=True)

    # ---- Analyse avancée ----
    with tabs[3]:
        if not expert:
            ui.alert("Section réservée à l'analyse technique du raisonnement de l'outil "
                     "(méthode, score, règle, probabilités du modèle). Activez le « Mode "
                     "Expert » dans la barre latérale pour y accéder.", "blue", "ai")
        else:
            render_audit(pred, cro_text, crr_text)


def render_audit(pred, cro_text, crr_text):
    if pred is None:
        st.info("Aucune analyse disponible.")
        return
    method = SOURCE_LABEL.get(pred.source, pred.source)
    m = ui.class_meta(pred.label)

    c1, c2 = st.columns([1, 1.3])
    with c1:
        ui.gauge(pred.confidence * 100, m["color"], method)
    with c2:
        ui.field_card("Décision", [
            ("Décision finale", pred.label_name),
            ("Méthode retenue", method),
            ("Concordance règle / modèle", "Divergence" if pred.disagreement else "Concordant"),
            ("Vérification conseillée", "Oui" if pred.needs_review else "Non")],
            "shield", True)

    with st.expander("Preuves textuelles (zones analysées)", expanded=True):
        if cro_text:
            st.caption("CRO — ligne diagnostic / indication")
            st.markdown(f"<div class='med-card'>{highlight(pred.diagnostic_zone)}</div>",
                        unsafe_allow_html=True)
        if crr_text:
            from src.text_zones import acetabular_sentences
            st.caption("CRR — phrases acétabulaires")
            st.markdown(f"<div class='med-card'>{highlight(acetabular_sentences(crr_text))}</div>",
                        unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Règle médicale**")
        if pred.rule_label is not None:
            st.markdown(f"Classe : **{pred.rule_label}**  \nZone : `{pred.rule_zone}`  \n"
                        f"Terme : « **{pred.rule_term}** »")
        else:
            st.markdown("Aucun terme non ambigu déclenché.")
    with cc2:
        st.markdown("**Modèle DoctoBERT**")
        if pred.ml_label is not None:
            st.markdown(f"Classe : **{pred.ml_label}** ({pred.ml_name}, "
                        f"confiance {pred.ml_confidence:.2f})")
            if pred.ml_proba:
                st.bar_chart(pd.DataFrame({"probabilité": {
                    f"Classe {k}": v for k, v in sorted(pred.ml_proba.items())}}))
        else:
            st.markdown("Non sollicité (règle suffisante) ou indisponible.")


def _filled(v):
    return v is not None and str(v).strip() not in ("", "None")


def _doc_chip(label, present):
    fg, bg = ("#166534", "#f0fdf4") if present else ("#94a3b8", "#f4f6f9")
    return ui.badge(f"{label} : {'présent' if present else 'absent'}", fg, bg)


# ==========================================================================
# Page — Tableau de bord
# ==========================================================================
def page_dashboard(clf):
    ui.topbar("Tableau de bord", "Vue d'ensemble de l'activité et des fractures de l'acétabulum")
    if not has_data():
        ui.alert(f"Aucun compte rendu dans « {config.DATA_RAW} ». Ajoutez vos fichiers, "
                 "ou utilisez « Nouveau patient » pour analyser un document.", "blue", "info")
        return

    recs, df, _ = get_records(clf, _data_signature())
    classified = df[df["Classe"] != ""]
    n = len(df)
    n_crr = int((df["Documents"] != "CRO").sum())
    n_review = int((df["À vérifier"] == "⚠").sum())
    pct_crr = f"{round(100*n_crr/n)} % des dossiers" if n else ""

    cols = st.columns(4)
    with cols[0]:
        ui.kpi("folder", n, "Dossiers analysés", ui.PRIMARY)
    with cols[1]:
        ui.kpi("scan", n_crr, "Avec imagerie (CRR)", "#b45309", pct_crr)
    with cols[2]:
        sub = "Aucun" if n_review == 0 else "À relire"
        ui.kpi("alert", n_review, "Dossiers à confirmer", "#e11d48", sub,
               "#16a34a" if n_review == 0 else "#e11d48")
    with cols[3]:
        ui.kpi("users", df[df["Opérateur"] != ""]["Opérateur"].nunique(), "Opérateurs", "#2563eb")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    a, b = st.columns([1.1, 1])
    with a:
        st.markdown("**Répartition par type de fracture**")
        counts = classified["Classe"].value_counts().sort_index()
        if len(counts):
            data = pd.DataFrame({
                "Type": [ui.CLASS_META[int(k)]["short"] for k in counts.index],
                "Nombre": counts.values,
                "c": [ui.CLASS_META[int(k)]["color"] for k in counts.index]})
            _donut(data)
        else:
            st.caption("Aucun dossier classé.")
    with b:
        st.markdown("**Documents disponibles**")
        dc = df["Documents"].value_counts()
        data = pd.DataFrame({"Documents": dc.index, "Nombre": dc.values})
        _vbar(data, "Documents", "Nombre",
              [ui.DOC_BADGE.get(k, (ui.PRIMARY,))[0] for k in dc.index])

    c, d = st.columns(2)
    with c:
        st.markdown("**Activité par opérateur** (top 8)")
        op = df[df["Opérateur"] != ""]["Opérateur"].value_counts().head(8)
        if len(op):
            _hbar(pd.DataFrame({"n": op.values}, index=op.index), ui.PRIMARY)
        else:
            st.caption("Non disponible.")
    with d:
        st.markdown("**Codes CCAM fréquents** (top 8)")
        codes = []
        for x in df["Codes CCAM"]:
            if x:
                codes += [t.strip() for t in re.split(r"[,;]", x) if t.strip()]
        if codes:
            cc = pd.Series(codes).value_counts().head(8)
            _hbar(pd.DataFrame({"n": cc.values}, index=cc.index), "#2563eb")
        else:
            st.caption("Non disponible.")

    st.markdown("**Interventions par année**")
    years = [y for y in (_year(v) for v in df["Date CR"]) if y]
    if years:
        ys = pd.Series(years).value_counts().sort_index()
        _area(pd.DataFrame({"Année": ys.index.astype(str), "Interventions": ys.values}))
    else:
        st.caption("Dates non exploitables.")


# ==========================================================================
# Page — Nouveau patient (parcours guidé)
# ==========================================================================
def _sig(cro_file, crr_file):
    def one(f):
        return (f.name, getattr(f, "size", len(f.getvalue()))) if f else None
    return (one(cro_file), one(crr_file))


def page_new_patient(clf, expert, anon):
    ui.topbar("Nouveau patient", "Analyse guidée d'un compte rendu opératoire et/ou radiologique")

    prev = (st.session_state.get("np_cro"), st.session_state.get("np_crr"))
    have_prev = bool(prev[0] or prev[1])
    analyzed = st.session_state.get("np_sig")
    cur_step = 2 if (analyzed == _sig(*prev) and have_prev) else (1 if have_prev else 0)
    ui.step_indicator(["Documents", "Analyse", "Résultat"], cur_step)

    st.markdown("<div class='med-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{ui.icon('folder',16)}Étape 1 — Importer les documents</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    cro_file = c1.file_uploader("Compte rendu opératoire (CRO)", type=["pdf"], key="np_cro")
    crr_file = c2.file_uploader("Compte rendu radiologique (CRR)", type=["pdf"], key="np_crr")
    st.markdown("<div class='dz-note'>" + ui.icon("info", 15) +
                " Formats PDF. Vous pouvez fournir un seul document ; les deux ensemble "
                "donnent l'analyse la plus fiable.</div></div>", unsafe_allow_html=True)

    if not cro_file and not crr_file:
        return

    sig = _sig(cro_file, crr_file)
    if st.session_state.get("np_sig") != sig:
        colA, _ = st.columns([1, 2])
        if colA.button("▶  Lancer l'analyse", type="primary", width="stretch"):
            st.session_state["np_sig"] = sig
            st.rerun()
        st.caption("Documents prêts. Lancez l'analyse pour obtenir la classification.")
        return

    # --- Analyse ---
    cro_text = read_document_from_bytes(cro_file.getvalue(), cro_file.name) if cro_file else ""
    crr_text = read_document_from_bytes(crr_file.getvalue(), crr_file.name) if crr_file else ""
    if cro_file and detect_doc_type(cro_text, cro_file.name) == "CRR":
        ui.alert(f"« {cro_file.name} » ressemble à un CRR. Vérifiez la zone de dépôt.", "amber", "alert")
    if not (cro_text.strip() or crr_text.strip()):
        ui.alert("Aucun texte exploitable (PDF scanné sans couche texte ?).", "red", "alert")
        return

    cro_info = extract(cro_text, "CRO") if cro_text else None
    crr_info = extract(crr_text, "CRR") if crr_text else None
    pred = clf.predict(cro_text, crr_text)
    did = file_id((cro_file or crr_file).name).replace("_CRR", "")

    _save_history(did, pred, cro_info, crr_info, cro_text, crr_text, sig)
    render_fiche(pred, cro_info, crr_info, cro_text, crr_text, did, expert, anon)


def _save_history(did, pred, cro_info, crr_info, cro_text, crr_text, sig):
    hist = st.session_state.setdefault("history", [])
    if hist and hist[0].get("sig") == sig:
        return  # déjà enregistré (re-run sur les mêmes fichiers)
    hist.insert(0, {
        "sig": sig, "id": did, "ts": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "label": pred.label, "label_name": pred.label_name,
        "docs": ("CRO+CRR" if (cro_text and crr_text) else ("CRO" if cro_text else "CRR")),
        "pred": pred, "cro_info": cro_info, "crr_info": crr_info,
        "cro_text": cro_text, "crr_text": crr_text,
    })
    del hist[50:]


# ==========================================================================
# Page — Historique
# ==========================================================================
def page_history(clf, expert, anon):
    ui.topbar("Historique des analyses", "Analyses de nouveaux patients réalisées durant la session")
    hist = st.session_state.get("history", [])

    open_idx = st.session_state.get("hist_open")
    if open_idx is not None and 0 <= open_idx < len(hist):
        e = hist[open_idx]
        if st.button("← Retour à l'historique", width="stretch"):
            st.session_state.pop("hist_open", None)
            st.rerun()
        render_fiche(e["pred"], e["cro_info"], e["crr_info"], e["cro_text"],
                     e["crr_text"], e["id"], expert, anon)
        return

    if not hist:
        ui.alert("Aucune analyse pour le moment. Rendez-vous dans « Nouveau patient » "
                 "pour analyser un compte rendu : il apparaîtra ici.", "blue", "clock")
        return

    cols = st.columns([3, 1])
    cols[0].caption(f"{len(hist)} analyse(s) durant cette session.")
    if cols[1].button("Vider l'historique", width="stretch"):
        st.session_state.pop("history", None)
        st.rerun()

    for i, e in enumerate(hist):
        with st.container(border=True):
            a, b, c = st.columns([2.6, 1.4, 1])
            name = display_name(e["cro_info"] or e["crr_info"], e["id"], anon)
            a.markdown(f"<div class='tl-time'>🕑 {e['ts']}</div>"
                       f"<div style='font-weight:700;font-size:1.02rem'>{ui._esc(name)}</div>"
                       f"<div class='muted'>Dossier {ui._esc(e['id'])}</div>",
                       unsafe_allow_html=True)
            b.markdown(ui.class_badge(e["label"]) + " " +
                       ui.doc_badges(e["docs"] != "CRR", e["docs"] != "CRO"),
                       unsafe_allow_html=True)
            if c.button("Ouvrir", key=f"hist_{i}", width="stretch"):
                st.session_state["hist_open"] = i
                st.rerun()


# ==========================================================================
# Page — Dossiers
# ==========================================================================
def page_dossiers(clf, expert, anon):
    ui.topbar("Dossiers", "Tous les comptes rendus du centre — recherche, tri et export")
    if not has_data():
        ui.alert(f"Aucun document dans « {config.DATA_RAW} ».", "blue", "info")
        return

    recs, df, by_id = get_records(clf, _data_signature())

    sel = st.session_state.get("selected_dossier")
    if sel and sel in by_id:
        r = by_id[sel]
        if st.button("← Retour à la liste", width="stretch"):
            st.session_state["tbl_nonce"] = st.session_state.get("tbl_nonce", 0) + 1
            st.session_state.pop("selected_dossier", None)
            st.rerun()
        render_fiche(r.prediction, r.cro_info, r.crr_info, r.cro_text, r.crr_text,
                     r.dossier_id, expert, anon)
        return

    f1, f2, f3 = st.columns([2, 1, 1])
    query = f1.text_input("Rechercher", "", placeholder="Nom ou numéro de dossier…",
                          label_visibility="collapsed")
    classes = f2.multiselect("Type", [1, 2, 3, 4], format_func=lambda c: ui.CLASS_META[c]["short"],
                             placeholder="Type de fracture")
    docs = f3.multiselect("Documents", ["CRO", "CRR", "CRO+CRR"], placeholder="Documents")

    view = df.copy()
    if classes:
        view = view[view["Classe"].isin({str(c) for c in classes})]
    if docs:
        view = view[view["Documents"].isin(docs)]
    if query:
        q = query.lower()
        view = view[view["Nom"].str.lower().str.contains(q, na=False)
                    | view["Prénom"].str.lower().str.contains(q, na=False)
                    | view["Dossier"].astype(str).str.contains(q, na=False)]

    disp = _display_df(view, expert, anon)
    st.caption(f"{len(disp)} dossier(s). Cliquez une ligne pour la fiche ; triez via les en-têtes.")

    selected, nonce = None, st.session_state.get("tbl_nonce", 0)
    try:
        ev = st.dataframe(disp, width="stretch", hide_index=True,
                          on_select="rerun", selection_mode="single-row", key=f"dt_{nonce}")
        rows = getattr(getattr(ev, "selection", None), "rows", []) or []
        if rows:
            selected = str(disp.iloc[rows[0]]["Dossier"])
    except TypeError:
        st.dataframe(disp, width="stretch", hide_index=True)
        pick = st.selectbox("Consulter un dossier", ["—"] + disp["Dossier"].astype(str).tolist())
        if pick != "—":
            selected = pick

    e1, e2, _ = st.columns([1, 1, 3])
    e1.download_button("⬇  CSV", disp.to_csv(index=False).encode("utf-8-sig"),
                       "dossiers_acetabulum.csv", "text/csv", width="stretch")
    e2.download_button("⬇  Excel", _to_excel(disp), "dossiers_acetabulum.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                       width="stretch")

    if selected:
        st.session_state["selected_dossier"] = selected
        st.rerun()


def _display_df(view, expert, anon):
    d = view.copy()
    if anon:
        d["Patient"] = "Patient " + d["Dossier"].astype(str)
        base = ["Dossier", "Patient", "Date CR", "Type de fracture", "Opérateur",
                "Codes CCAM", "Documents"]
    else:
        d["Patient"] = (d["Nom"].fillna("") + " " + d["Prénom"].fillna("")).str.strip().replace("", "—")
        base = ["Dossier", "Patient", "Naissance", "Date CR", "Type de fracture",
                "Opérateur", "Codes CCAM", "Documents"]
    if expert:
        d["Méthode"] = d["Méthode"].map(lambda s: SOURCE_LABEL.get(s, s))
        base = base + ["Méthode", "Confiance", "À vérifier"]
    return d[base]


def _to_excel(df):
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="Dossiers")
    except Exception:
        return df.to_csv(index=False).encode("utf-8-sig")
    return buf.getvalue()


# ==========================================================================
# Graphiques (Altair, repli st.bar_chart)
# ==========================================================================
def _year(s):
    if not s:
        return None
    m = re.search(r"\b(19|20)\d{2}\b", s)
    if m:
        return int(m.group(0))
    m = re.search(r"\b\d{2}/\d{2}/(\d{2})\b", s)
    if m:
        yy = int(m.group(1))
        return 2000 + yy if yy < 50 else 1900 + yy
    return None


def _donut(data):
    try:
        import altair as alt
        base = alt.Chart(data).encode(
            theta=alt.Theta("Nombre:Q", stack=True),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=list(data["Type"]), range=list(data["c"])),
                legend=alt.Legend(orient="bottom", title=None, labelLimit=220)),
            tooltip=["Type", "Nombre"])
        arc = base.mark_arc(innerRadius=62, cornerRadius=4, stroke="#fff", strokeWidth=2)
        total = int(data["Nombre"].sum())
        txt = alt.Chart(pd.DataFrame({"t": [total]})).mark_text(
            fontSize=30, fontWeight=800, color="#0f172a", dy=-6).encode(text="t:Q")
        sub = alt.Chart(pd.DataFrame({"t": ["dossiers"]})).mark_text(
            fontSize=12, color="#64748b", dy=18).encode(text="t:N")
        st.altair_chart((arc + txt + sub).properties(height=260), width="stretch")
    except Exception:
        st.bar_chart(data.set_index("Type")["Nombre"])


def _vbar(data, x, y, colors=None):
    try:
        import altair as alt
        color = alt.Color(f"{x}:N", legend=None, scale=alt.Scale(range=colors)) if colors \
            else alt.Color(f"{x}:N", legend=None)
        ch = (alt.Chart(data).mark_bar(cornerRadiusEnd=6, size=46)
              .encode(x=alt.X(f"{x}:N", sort=None, title=None),
                      y=alt.Y(f"{y}:Q", title=None), color=color,
                      tooltip=list(data.columns)).properties(height=260))
        st.altair_chart(ch, width="stretch")
    except Exception:
        st.bar_chart(data.set_index(x))


def _hbar(df_indexed, color):
    try:
        import altair as alt
        d = df_indexed.reset_index()
        d.columns = ["cat", "n"]
        ch = (alt.Chart(d).mark_bar(cornerRadiusEnd=5, color=color)
              .encode(y=alt.Y("cat:N", sort="-x", title=None, axis=alt.Axis(labelLimit=180)),
                      x=alt.X("n:Q", title=None), tooltip=["cat", "n"])
              .properties(height=max(150, 32 * len(d))))
        st.altair_chart(ch, width="stretch")
    except Exception:
        st.bar_chart(df_indexed)


def _area(data):
    try:
        import altair as alt
        ch = (alt.Chart(data).mark_area(
            line={"color": ui.PRIMARY}, color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#ffffff", offset=0),
                       alt.GradientStop(color=ui.PRIMARY, offset=1)],
                x1=1, x2=1, y1=1, y2=0))
            .encode(x=alt.X("Année:N", title=None),
                    y=alt.Y("Interventions:Q", title=None),
                    tooltip=["Année", "Interventions"]).properties(height=220))
        st.altair_chart(ch, width="stretch")
    except Exception:
        st.bar_chart(data.set_index("Année"))


# ==========================================================================
# Barre latérale + routage
# ==========================================================================
NAV = [("Tableau de bord", "🏠"), ("Nouveau patient", "🩺"),
       ("Dossiers", "🗂️"), ("Historique", "🕘")]


def _sidebar(clf):
    with st.sidebar:
        st.markdown(
            f"<div class='sb-brand'><div class='sb-logo'>{ui.icon('shield',22,'#fff')}</div>"
            f"<div><div class='sb-name'>Acétabule</div>"
            f"<div class='sb-tag'>Aide à la décision clinique</div></div></div>",
            unsafe_allow_html=True)

        st.markdown("<div class='sb-sec'>Navigation</div>", unsafe_allow_html=True)
        page = st.session_state.setdefault("page", "Tableau de bord")
        for label, ic in NAV:
            if st.button(f"{ic}\u2002{label}", key=f"nav_{label}",
                         type="primary" if page == label else "secondary", width="stretch"):
                st.session_state.page = label
                st.rerun()
        page = st.session_state.page

        st.markdown("<div class='sb-sec'>Affichage</div>", unsafe_allow_html=True)
        expert = st.toggle("Mode expert", value=False,
                           help="Affiche les détails techniques du raisonnement de l'IA.")
        anon = st.toggle("Anonymiser les patients", value=False,
                         help="Masque l'identité ; affiche l'identifiant de dossier.")

        st.markdown("<div class='sb-sec'>État du système</div>", unsafe_allow_html=True)
        if clf.ml_name:
            st.markdown(
                f"<div class='sb-status'><span class='dot' style='background:#34d399'></span>"
                f"<div><div class='s-main'>{ui._esc(clf.ml_name)} · local</div>"
                f"<div class='s-sub'>Modèle opérationnel</div></div></div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                "<div class='sb-status'><span class='dot' style='background:#fbbf24'></span>"
                "<div><div class='s-main'>Mode règles seules</div>"
                "<div class='s-sub'>DoctoBERT introuvable dans models/doctobert/</div></div></div>",
                unsafe_allow_html=True)

        #st.markdown("<div class='sb-sec'>Types de fracture</div>", unsafe_allow_html=True)
        #leg = ""
        #for i in (1, 2, 3):
         #   mm = ui.CLASS_META[i]
          #  leg += (f"<div class='sb-leg'><span class='dot' style='background:{mm['color']}'>"
           #         f"</span><b>{i}</b>&nbsp;— {mm['short']}</div>")
        #none_m = ui.CLASS_META[None]
        #leg += (f"<div class='sb-leg'><span class='dot' style='background:{none_m['color']}'>"
         #       f"</span>{none_m['short']} / autre</div>")
        #st.markdown(leg, unsafe_allow_html=True)

        st.markdown("<div class='sb-foot'>Acétabule v2 · Prototype d'aide à la décision. "
                    "Ne remplace pas le jugement clinique.</div>", unsafe_allow_html=True)

    return page, expert, anon


def main():
    clf = get_classifier()
    page, expert, anon = _sidebar(clf)

    if page != "Dossiers":
        st.session_state.pop("selected_dossier", None)
    if page != "Historique":
        st.session_state.pop("hist_open", None)

    if page == "Tableau de bord":
        page_dashboard(clf)
    elif page == "Nouveau patient":
        page_new_patient(clf, expert, anon)
    elif page == "Dossiers":
        page_dossiers(clf, expert, anon)
    else:
        page_history(clf, expert, anon)


if __name__ == "__main__":
    main()
