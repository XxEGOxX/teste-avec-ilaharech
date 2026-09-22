"""
ui.py — Système de design clinique « premium » et composants réutilisables.

Tout le visuel de l'application vit ici : palette, typographie, icônes SVG,
cartes, KPI, jauge, parcours par étapes, fiche patient. La logique métier
(extraction, règles, modèle) est ailleurs.

Principe : vue médecin épurée et lisible ; les détails techniques ne sont
rendus que dans l'onglet « Analyse avancée » (mode Expert).
"""
from __future__ import annotations

import html
import math
from typing import List, Optional, Sequence, Tuple

import streamlit as st

# --------------------------------------------------------------------------
# Couleurs sémantiques des classes (centralisées — utilisées partout)
# --------------------------------------------------------------------------
CLASS_META = {
    1: {"name": "Fracture colonne antérieure", "short": "Colonne antérieure",
        "color": "#2563eb", "soft": "#eef4ff"},
    2: {"name": "Fracture bicolonne", "short": "Bicolonne",
        "color": "#7c3aed", "soft": "#f5f0ff"},
    3: {"name": "Colonne antérieure + hémi-transverse postérieure",
        "short": "Col. antérieure + hémi-transverse post.",
        "color": "#e11d48", "soft": "#fff0f3"},
    4: {"name": "Autre fracture / hors périmètre", "short": "Autre fracture",
        "color": "#b45309", "soft": "#fff7ec"},
    None: {"name": "Non déterminé / Autre fracture", "short": "Non déterminé",
           "color": "#64748b", "soft": "#f4f6f9"},
}
DOC_BADGE = {
    "CRO": ("#0d9488", "#eafaf6"),
    "CRR": ("#b45309", "#fff7ec"),
    "CRO+CRR": ("#2563eb", "#eef4ff"),
}
PRIMARY = "#0d9488"


def class_meta(label: Optional[int]) -> dict:
    return CLASS_META.get(label, CLASS_META[None])


# --------------------------------------------------------------------------
# Icônes SVG (décoratives ; un libellé texte accompagne toujours l'icône)
# --------------------------------------------------------------------------
_ICONS = {
    "home": "<path d='M3 11l9-8 9 8'/><path d='M5 10v10h14V10'/>",
    "patient": "<path d='M2 12h4l2-6 4 12 2-6h4'/>",
    "folder": "<path d='M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/>",
    "clock": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/>",
    "users": "<circle cx='9' cy='8' r='3'/><path d='M3.5 19a5.5 5.5 0 0 1 11 0'/><path d='M16 8.5a3 3 0 0 1 0 5'/><path d='M20 19a5 5 0 0 0-3-4.5'/>",
    "chart": "<line x1='4' y1='20' x2='4' y2='12'/><line x1='10' y1='20' x2='10' y2='5'/><line x1='16' y1='20' x2='16' y2='9'/><line x1='2' y1='20' x2='22' y2='20'/>",
    "tag": "<path d='M20 11l-8 8-8-8V4h8z'/><circle cx='8' cy='8' r='1.2' fill='currentColor' stroke='none'/>",
    "doc": "<path d='M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z'/><path d='M14 3v5h5'/><path d='M8 13h8M8 17h6'/>",
    "alert": "<path d='M12 3l9 16H3z'/><line x1='12' y1='10' x2='12' y2='14'/><circle cx='12' cy='17' r='.7' fill='currentColor' stroke='none'/>",
    "check": "<circle cx='12' cy='12' r='9'/><path d='M8 12.5l3 3 5-6'/>",
    "info": "<circle cx='12' cy='12' r='9'/><line x1='12' y1='11' x2='12' y2='16'/><circle cx='12' cy='8' r='.7' fill='currentColor' stroke='none'/>",
    "ai": "<path d='M12 3l1.7 4.8L18.5 9.5l-4.8 1.7L12 16l-1.7-4.8L5.5 9.5l4.8-1.7z'/>",
    "search": "<circle cx='11' cy='11' r='6'/><line x1='16' y1='16' x2='21' y2='21'/>",
    "activity": "<path d='M2 12h4l2-6 4 12 2-6h4'/>",
    "scan": "<path d='M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2'/><line x1='4' y1='12' x2='20' y2='12'/>",
    "shield": "<path d='M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z'/>",
}


def icon(name: str, size: int = 18, color: str = "currentColor", sw: float = 1.8) -> str:
    inner = _ICONS.get(name, "")
    return (f"<svg width='{size}' height='{size}' viewBox='0 0 24 24' fill='none' "
            f"stroke='{color}' stroke-width='{sw}' stroke-linecap='round' "
            f"stroke-linejoin='round' style='vertical-align:-3px;flex:none'>{inner}</svg>")


# --------------------------------------------------------------------------
# CSS global
# --------------------------------------------------------------------------
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#f4f7fb; --surface:#ffffff; --border:#e8edf4; --border-2:#f0f3f8;
  --ink:#0f172a; --ink-2:#334155; --muted:#64748b; --muted-2:#9aa7b8;
  --primary:#0d9488; --primary-d:#0f766e;
  --shadow-sm:0 1px 2px rgba(16,24,40,.04);
  --shadow:0 1px 3px rgba(16,24,40,.06),0 6px 18px rgba(16,24,40,.05);
  --shadow-lg:0 10px 30px rgba(16,24,40,.10);
  --r:16px;
}
.stApp{background:
  radial-gradient(1200px 600px at 100% -5%, #eaf3f1 0, rgba(234,243,241,0) 55%),
  radial-gradient(1000px 500px at -10% 0%, #eef2fb 0, rgba(238,242,251,0) 50%),
  var(--bg);}
html,body,[class*="css"]{font-family:'Inter',system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  color:var(--ink);}
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1240px;}
#MainMenu,footer{display:none;}
header[data-testid="stHeader"]{background:transparent;}
h1,h2,h3{letter-spacing:-.02em;}

/* ---------- Topbar ---------- */
.topbar{display:flex;align-items:center;justify-content:space-between;
  background:linear-gradient(110deg,#0f766e,#0d9488 60%,#14b8a6);color:#fff;
  border-radius:18px;padding:16px 22px;box-shadow:var(--shadow-lg);margin-bottom:18px;}
.topbar .brand{display:flex;align-items:center;gap:13px;}
.topbar .logo{width:42px;height:42px;border-radius:12px;background:rgba(255,255,255,.16);
  display:grid;place-items:center;}
.topbar h1{font-size:1.15rem;margin:0;color:#fff;font-weight:800;}
.topbar .sub{font-size:.8rem;opacity:.85;margin-top:1px;}
.topbar .tb-right{display:flex;align-items:center;gap:8px;font-size:.8rem;}
.tb-pill{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.22);
  padding:5px 11px;border-radius:999px;font-weight:600;}

/* ---------- Cards ---------- */
.med-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--r);
  padding:18px 20px;box-shadow:var(--shadow);margin-bottom:14px;}
.card-title{font-size:.72rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;
  color:var(--muted);margin-bottom:13px;display:flex;align-items:center;gap:8px;}
.card-title svg{color:var(--primary);}

/* ---------- Hero résultat ---------- */
.hero{position:relative;overflow:hidden;border-radius:20px;padding:24px 26px;
  background:linear-gradient(180deg,var(--soft),#fff 75%);border:1px solid var(--c);
  box-shadow:var(--shadow);margin-bottom:16px;}
.hero::before{content:'';position:absolute;left:0;top:0;bottom:0;width:7px;background:var(--c);}
.hero-row{display:flex;justify-content:space-between;align-items:center;gap:18px;}
.hero-eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);}
.hero-title{font-size:1.75rem;font-weight:800;color:var(--c);line-height:1.12;margin:6px 0 13px;}
.hero-num{width:74px;height:74px;border-radius:20px;background:var(--c);color:#fff;
  font-size:2rem;font-weight:800;display:grid;place-items:center;box-shadow:var(--shadow);flex:none;}

/* ---------- Badges ---------- */
.badge{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:999px;
  font-size:.78rem;font-weight:650;line-height:1;border:1px solid transparent;white-space:nowrap;}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none;}
.chip-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;}

/* ---------- Champs ---------- */
.fgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:6px 20px;}
.field{padding:9px 0;border-bottom:1px solid var(--border-2);}
.field:last-child{border-bottom:none;}
.f-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px;}
.f-value{font-size:.96rem;font-weight:600;color:var(--ink);word-break:break-word;}
.f-empty{color:var(--muted-2);font-style:italic;font-weight:500;}

/* ---------- KPI ---------- */
.kpi{position:relative;background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:18px;box-shadow:var(--shadow);height:100%;}
.kpi-ic{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;margin-bottom:14px;}
.kpi-val{font-size:2rem;font-weight:800;color:var(--ink);line-height:1;}
.kpi-lab{font-size:.82rem;color:var(--muted);margin-top:6px;}
.kpi-sub{font-size:.74rem;margin-top:7px;font-weight:600;}

/* ---------- Avatar / identité ---------- */
.idcard{display:flex;align-items:center;gap:16px;background:var(--surface);
  border:1px solid var(--border);border-radius:var(--r);padding:16px 20px;box-shadow:var(--shadow);margin-bottom:14px;}
.avatar{width:54px;height:54px;border-radius:16px;display:grid;place-items:center;
  font-weight:800;font-size:1.25rem;color:#fff;flex:none;}
.idcard .nm{font-size:1.2rem;font-weight:800;line-height:1.1;}
.idcard .mt{font-size:.82rem;color:var(--muted);margin-top:3px;}

/* ---------- Sections médicales ---------- */
.sec{border:1px solid var(--border);border-radius:13px;overflow:hidden;margin-bottom:11px;
  background:var(--surface);box-shadow:var(--shadow-sm);}
.sec-h{padding:11px 16px;font-weight:700;color:var(--ink);background:#fbfcfe;
  border-bottom:1px solid var(--border-2);font-size:.92rem;display:flex;justify-content:space-between;align-items:center;}
.sec-b{padding:13px 16px;color:var(--ink-2);font-size:.92rem;white-space:pre-line;line-height:1.55;}

/* ---------- Alertes ---------- */
.alert{display:flex;gap:11px;align-items:flex-start;border-radius:13px;padding:13px 16px;
  font-size:.9rem;margin-bottom:13px;border:1px solid;line-height:1.45;}
.alert svg{margin-top:1px;flex:none;}
.alert-amber{background:#fffbeb;border-color:#fde68a;color:#92400e;}
.alert-red{background:#fef2f2;border-color:#fecaca;color:#991b1b;}
.alert-ok{background:#f0fdf4;border-color:#bbf7d0;color:#166534;}
.alert-blue{background:#eff6ff;border-color:#bfdbfe;color:#1e40af;}

/* ---------- Jauge (CSS conic) ---------- */
.gauge-wrap{display:flex;align-items:center;gap:18px;}
.gauge{position:relative;width:118px;height:118px;border-radius:50%;display:grid;place-items:center;flex:none;}
.gauge::after{content:'';position:absolute;width:84px;height:84px;border-radius:50%;background:var(--surface);box-shadow:inset 0 0 0 1px var(--border);}
.gauge .g-val{position:relative;font-size:1.5rem;font-weight:800;color:var(--ink);}

/* ---------- Stepper ---------- */
.stepper{display:flex;align-items:center;margin:6px 0 18px;}
.step{display:flex;flex-direction:column;align-items:center;gap:7px;min-width:96px;}
.s-dot{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font-weight:700;
  font-size:.9rem;border:2px solid var(--border);background:#fff;color:var(--muted-2);}
.s-lab{font-size:.76rem;color:var(--muted);font-weight:600;}
.step.active .s-dot{background:var(--primary);border-color:var(--primary);color:#fff;box-shadow:0 0 0 4px #0d948822;}
.step.active .s-lab{color:var(--primary);}
.step.done .s-dot{background:#ecfdf5;border-color:var(--primary);color:var(--primary);}
.s-line{flex:1;height:2px;background:var(--border);margin:0 4px 24px;border-radius:2px;}
.s-line.done{background:var(--primary);}

/* ---------- Dropzone ---------- */
.dz-note{display:flex;align-items:center;gap:10px;color:var(--muted);font-size:.88rem;
  padding:6px 2px 2px;}

/* ---------- Timeline (historique) ---------- */
.tl-time{font-size:.74rem;color:var(--muted);font-weight:600;letter-spacing:.03em;}

.muted{color:var(--muted);font-size:.9rem;}
hr{margin:.8rem 0;border-color:var(--border);}

/* ---------- Onglets ---------- */
.stTabs [data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--border);}
.stTabs [data-baseweb="tab"]{border-radius:11px 11px 0 0;padding:9px 18px;font-weight:600;}
.stTabs [aria-selected="true"]{background:var(--surface);box-shadow:var(--shadow-sm);}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0b3b39,#072523);
  border-right:1px solid rgba(255,255,255,.05);}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] .stMarkdown *,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p{color:#dCEFEC !important;}
.sb-brand{display:flex;align-items:center;gap:12px;padding:6px 4px 10px;}
.sb-logo{width:42px;height:42px;border-radius:13px;
  background:linear-gradient(135deg,#0d9488,#2dd4bf);display:grid;place-items:center;
  box-shadow:0 6px 16px rgba(13,148,136,.4);}
.sb-name{font-weight:800;font-size:1.18rem;line-height:1;color:#fff !important;}
.sb-tag{font-size:.72rem;opacity:.62;margin-top:3px;}
.sb-sec{font-size:.66rem;text-transform:uppercase;letter-spacing:.13em;opacity:.5;
  margin:18px 0 8px;font-weight:700;}

/* navigation en boutons */
section[data-testid="stSidebar"] .stButton{margin-bottom:3px;}
section[data-testid="stSidebar"] .stButton>button{width:100%;justify-content:flex-start;
  gap:11px;text-align:left;background:transparent;border:1px solid transparent;
  color:#bfe0db !important;font-weight:600;font-size:.94rem;padding:10px 13px;
  border-radius:12px;box-shadow:none;transition:.15s;}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.07);
  color:#fff !important;}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{
  background:linear-gradient(100deg,#0d9488,#14b8a6);color:#fff !important;
  box-shadow:0 7px 18px rgba(13,148,136,.4);}

/* toggles plus lisibles sur fond sombre */
section[data-testid="stSidebar"] [data-testid="stToggle"]{margin:2px 0;}

/* statut système */
.sb-status{display:flex;align-items:center;gap:10px;padding:11px 13px;border-radius:13px;
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.07);}
.sb-status .dot{width:9px;height:9px;box-shadow:0 0 0 3px rgba(255,255,255,.06);}
.sb-status .s-main{font-weight:700;font-size:.9rem;color:#fff !important;}
.sb-status .s-sub{font-size:.72rem;opacity:.65;}

/* légende */
.sb-leg{display:flex;align-items:center;gap:9px;margin:5px 0;font-size:.84rem;}
.sb-leg .dot{width:9px;height:9px;}
.sb-leg b{color:#fff !important;}
.sb-foot{margin-top:18px;padding-top:12px;border-top:1px solid rgba(255,255,255,.08);
  font-size:.72rem;opacity:.5;}

/* boutons */
.stButton>button, .stDownloadButton>button{border-radius:11px;font-weight:600;}
.stDownloadButton>button{border:1px solid var(--border);}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Briques
# --------------------------------------------------------------------------
def _esc(v) -> str:
    return html.escape(str(v)) if v is not None else ""


def _filled(v) -> bool:
    return v is not None and str(v).strip() not in ("", "None")


def badge(text: str, fg: str, bg: str, dot: bool = True) -> str:
    d = f"<span class='dot' style='background:{fg}'></span>" if dot else ""
    return (f"<span class='badge' style='color:{fg};background:{bg};"
            f"border-color:{fg}33'>{d}{_esc(text)}</span>")


def doc_badges(has_cro: bool, has_crr: bool) -> str:
    key = "CRO+CRR" if (has_cro and has_crr) else ("CRO" if has_cro else "CRR")
    fg, bg = DOC_BADGE[key]
    return badge(key, fg, bg, dot=False)


def kv(label: str, value) -> str:
    if _filled(value):
        val = f"<span class='f-value'>{_esc(value)}</span>"
    else:
        val = "<span class='f-value f-empty'>Non renseigné</span>"
    return f"<div class='field'><div class='f-label'>{_esc(label)}</div>{val}</div>"


# --------------------------------------------------------------------------
# Composants
# --------------------------------------------------------------------------
def topbar(title: str, subtitle: str, right_pills: Sequence[str] = ()):
    pills = "".join(f"<span class='tb-pill'>{_esc(p)}</span>" for p in right_pills)
    st.markdown(
        f"<div class='topbar'><div class='brand'>"
        f"<div class='logo'>{icon('shield', 22, '#fff')}</div>"
        f"<div><h1>{_esc(title)}</h1><div class='sub'>{_esc(subtitle)}</div></div></div>"
        f"<div class='tb-right'>{pills}</div></div>",
        unsafe_allow_html=True)


def hero(label: Optional[int], status: Optional[Tuple[str, str]], docs_html: str):
    m = class_meta(label)
    chips = [docs_html]
    if status:
        txt, level = status
        if level == "ok":
            chips.append(f"<span class='badge' style='color:#166534;background:#f0fdf4;"
                         f"border-color:#16653433'>{icon('check',14,'#166534')}{_esc(txt)}</span>")
        else:
            chips.append(f"<span class='badge' style='color:#92400e;background:#fffbeb;"
                         f"border-color:#92400e33'>{icon('alert',14,'#92400e')}{_esc(txt)}</span>")
    mark = (f"<div class='hero-num'>{label}</div>" if label
            else "<div class='hero-num' style='font-size:1.4rem'>?</div>")
    st.markdown(
        f"<div class='hero' style='--c:{m['color']};--soft:{m['soft']}'>"
        f"<div class='hero-row'><div>"
        f"<div class='hero-eyebrow'>Type de fracture</div>"
        f"<div class='hero-title'>{m['name']}</div>"
        f"<div class='chip-row'>{''.join(chips)}</div></div>"
        f"{mark}</div></div>",
        unsafe_allow_html=True)


def identity_card(name: str, label: Optional[int], meta_line: str, docs_html: str):
    m = class_meta(label)
    initials = "".join(w[0] for w in name.split()[:2]).upper() or "P"
    st.markdown(
        f"<div class='idcard'>"
        f"<div class='avatar' style='background:linear-gradient(135deg,{m['color']},{m['color']}cc)'>{_esc(initials)}</div>"
        f"<div style='flex:1'><div class='nm'>{_esc(name)}</div>"
        f"<div class='mt'>{_esc(meta_line)}</div></div>"
        f"<div class='chip-row'>{docs_html}</div></div>",
        unsafe_allow_html=True)


def alert(message: str, kind: str = "amber", ic: str = "alert"):
    colors = {"amber": "#92400e", "red": "#991b1b", "ok": "#166534", "blue": "#1e40af"}
    st.markdown(f"<div class='alert alert-{kind}'>{icon(ic,18,colors.get(kind,'#1e40af'))}"
                f"<div>{_esc(message)}</div></div>", unsafe_allow_html=True)


def field_card(title: str, items: List[Tuple[str, object]], ic: str = "",
               show_empty: bool = False):
    rows = [(l, v) for l, v in items if show_empty or _filled(v)]
    if not rows:
        if not show_empty:
            return  # carte vide -> on n'affiche rien
        body = "<div class='muted'>Aucune information renseignée.</div>"
    else:
        body = "<div class='fgrid'>" + "".join(kv(l, v) for l, v in rows) + "</div>"
    head = f"<div class='card-title'>{icon(ic,16) if ic else ''}{_esc(title)}</div>" if title else ""
    st.markdown(f"<div class='med-card'>{head}{body}</div>", unsafe_allow_html=True)


def text_card(title: str, content: str, ic: str = ""):
    if not content or not content.strip():
        return
    head = f"<div class='card-title'>{icon(ic,16) if ic else ''}{_esc(title)}</div>" if title else ""
    st.markdown(f"<div class='med-card'>{head}"
                f"<div style='white-space:pre-line;line-height:1.6;color:var(--ink-2)'>"
                f"{_esc(content)}</div></div>", unsafe_allow_html=True)


def sections_blocks(blocks: List[Tuple[str, str, str]]):
    shown = False
    for label, content, tag in blocks:
        if not content or not content.strip():
            continue
        shown = True
        st.markdown(
            f"<div class='sec'><div class='sec-h'><span>{_esc(label)}</span>"
            f"<span class='muted'>{_esc(tag)}</span></div>"
            f"<div class='sec-b'>{_esc(content)}</div></div>", unsafe_allow_html=True)
    if not shown:
        st.markdown("<div class='muted'>Aucune section structurée détectée.</div>",
                    unsafe_allow_html=True)


def kpi(ic: str, value, label: str, accent: str = PRIMARY, sub: str = "", sub_color: str = ""):
    sub_html = (f"<div class='kpi-sub' style='color:{sub_color or 'var(--muted)'}'>{_esc(sub)}</div>"
                if sub else "")
    st.markdown(
        f"<div class='kpi'><div class='kpi-ic' style='background:{accent}1a'>"
        f"{icon(ic,20,accent)}</div>"
        f"<div class='kpi-val'>{_esc(value)}</div>"
        f"<div class='kpi-lab'>{_esc(label)}</div>{sub_html}</div>",
        unsafe_allow_html=True)


def gauge(percent: float, color: str, caption: str = ""):
    pct = max(0, min(100, round(percent)))
    cap = f"<div><div style='font-weight:700;font-size:1.02rem'>{_esc(caption)}</div>" \
          f"<div class='muted'>Indice de confiance</div></div>" if caption else ""
    st.markdown(
        f"<div class='gauge-wrap'><div class='gauge' style='background:"
        f"conic-gradient({color} {pct*3.6}deg, #eef2f7 0)'>"
        f"<span class='g-val'>{pct}%</span></div>{cap}</div>",
        unsafe_allow_html=True)


def step_indicator(steps: Sequence[str], current: int):
    """current : index 0-based de l'étape active."""
    parts = []
    for i, label in enumerate(steps):
        cls = "active" if i == current else ("done" if i < current else "")
        mark = "✓" if i < current else str(i + 1)
        parts.append(f"<div class='step {cls}'><div class='s-dot'>{mark}</div>"
                     f"<div class='s-lab'>{_esc(label)}</div></div>")
        if i < len(steps) - 1:
            parts.append(f"<div class='s-line {'done' if i < current else ''}'></div>")
    st.markdown(f"<div class='stepper'>{''.join(parts)}</div>", unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    sub = f"<div class='muted' style='margin-top:-4px'>{_esc(subtitle)}</div>" if subtitle else ""
    st.markdown(f"<h2 style='margin:.2rem 0 .2rem'>{_esc(title)}</h2>{sub}",
                unsafe_allow_html=True)


def class_badge(label: Optional[int]) -> str:
    m = class_meta(label)
    txt = f"{label} · {m['short']}" if label else m["short"]
    return badge(txt, m["color"], m["soft"])
