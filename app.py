"""
Group 17 -- Llama-based customer-service chatbot (no RAG)
MSBA 610 Advanced Text Analytics

Presentation build. Every number and chart is read from `artifacts/`, produced by
the Colab notebook. Nothing is hard-coded; missing files show a clear empty state.

Run locally:   streamlit run app.py
"""
import html
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import g17_text as T

ART = Path(__file__).parent / "artifacts"

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
INK = "#17233B"        # text
MUTED = "#5B6678"      # secondary text
LINE = "#E3E7EE"       # hairlines
SURFACE = "#F4F6F9"    # quiet panels
BASE = "#8C96A8"       # Model 1 (baseline) -- always slate
FT = "#0B7A83"         # Model 2 (fine-tuned) -- always deep teal
AGENT = "#C98A12"      # human agent / reference -- always amber
ALERT = "#B83A2A"      # flagged details
DATA = "#4A5F86"       # dataset facts (not a model)
CATS = ["#4A5F86", "#7C9A47", "#9A5B86", "#3F8C77", "#B8693E", "#6A63A6", "#8E7B3F", "#3D7EA6", "#A0525A", "#5C8C8C", "#6E7480"]

st.set_page_config(page_title="Group 17 · Llama customer-service chatbot", page_icon="💬", layout="wide")

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,750&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button, table, td, th {{
  font-family: 'Instrument Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
.stApp {{ color: {INK}; }}
.block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }}
.stMarkdown p, .stMarkdown li {{ font-size: 1.04rem; line-height: 1.6; }}
h1, h2, h3, .g-display {{ font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif; color: {INK}; letter-spacing: -0.015em; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ visibility: hidden; height: 0; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] .g-brand {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.35rem; font-weight: 750; line-height: 1.15; margin: 0.2rem 0 0.25rem; }}
[data-testid="stSidebar"] .g-brand-sub {{ color: {MUTED}; font-size: 0.9rem; margin-bottom: 1.2rem; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ padding: 0.32rem 0.2rem; font-size: 0.98rem; }}
.g-legend {{ margin-top: 1.6rem; font-size: 0.88rem; color: {MUTED}; line-height: 1.9; }}
.g-dot {{ display: inline-block; width: 0.7rem; height: 0.7rem; border-radius: 50%; margin-right: 0.45rem; vertical-align: -0.05rem; }}

/* page header */
.g-step {{ color: {FT}; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.15rem; }}
.g-title {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 750; font-size: 2.35rem; line-height: 1.08; margin: 0 0 0.55rem; }}
.g-lead {{ color: {MUTED}; font-size: 1.12rem; line-height: 1.55; max-width: 68ch; margin-bottom: 1.6rem; }}

/* hero */
.g-hero {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 750; font-size: clamp(2.3rem, 4.6vw, 3.6rem);
           line-height: 1.02; letter-spacing: -0.03em; margin: 0.4rem 0 0.9rem; max-width: 17ch; }}
.g-hero-sub {{ font-size: 1.18rem; color: {MUTED}; max-width: 62ch; line-height: 1.55; margin-bottom: 1.8rem; }}

/* stat strip: hairline separated, not cards */
.g-stats {{ display: flex; flex-wrap: wrap; border-top: 1px solid {LINE}; border-bottom: 1px solid {LINE}; margin: 0.4rem 0 1.8rem; }}
.g-stat {{ flex: 1 1 150px; padding: 0.9rem 1.1rem 0.9rem 0; }}
.g-stat + .g-stat {{ border-left: 1px solid {LINE}; padding-left: 1.1rem; }}
.g-stat-v {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.85rem; font-weight: 750; line-height: 1.1; }}
.g-stat-l {{ color: {MUTED}; font-size: 0.9rem; margin-top: 0.15rem; }}

/* transcript */
.g-chat {{ display: flex; flex-direction: column; gap: 0.75rem; margin: 0.4rem 0 1rem; }}
.g-who {{ font-size: 0.86rem; font-weight: 600; margin-bottom: 0.3rem; color: {MUTED}; }}
.g-bubble {{ padding: 0.95rem 1.1rem; border-radius: 14px; line-height: 1.58; font-size: 1.02rem; white-space: normal; overflow-wrap: anywhere; }}
.g-customer {{ background: {SURFACE}; border: 1px solid {LINE}; border-top-left-radius: 4px; max-width: 760px; }}
.g-m-base {{ background: #FFFFFF; border: 1px solid {LINE}; border-left: 5px solid {BASE}; border-top-left-radius: 4px; }}
.g-m-ft {{ background: #F1F8F8; border: 1px solid #CFE6E8; border-left: 5px solid {FT}; border-top-left-radius: 4px; }}
.g-m-agent {{ background: #FFFBF2; border: 1px solid #F1E2C2; border-left: 5px solid {AGENT}; border-top-left-radius: 4px; }}
.g-slot {{ background: #E4F0F1; color: #065C63; border-radius: 6px; padding: 0.05rem 0.35rem; font-weight: 600; white-space: nowrap; }}
.g-flag {{ background: #FBE9E6; color: {ALERT}; border-radius: 6px; padding: 0.05rem 0.35rem; font-weight: 600; }}
.g-chips {{ margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.4rem; }}
.g-chip {{ font-size: 0.82rem; color: {MUTED}; border: 1px solid {LINE}; border-radius: 999px; padding: 0.12rem 0.6rem; background: #fff; }}

/* verdict + table */
.g-verdict {{ border-left: 6px solid {FT}; background: #F1F8F8; padding: 1.1rem 1.3rem; border-radius: 4px 12px 12px 4px; margin: 0.2rem 0 1.6rem; }}
.g-verdict.tie {{ border-left-color: {BASE}; background: {SURFACE}; }}
.g-verdict h3 {{ margin: 0 0 0.3rem; font-size: 1.45rem; }}
.g-verdict p {{ margin: 0; color: {MUTED}; }}
.g-table {{ width: 100%; border-collapse: collapse; font-size: 0.97rem; }}
.g-table th {{ text-align: left; color: {MUTED}; font-weight: 600; font-size: 0.86rem; padding: 0.5rem 0.6rem; border-bottom: 2px solid {INK}; }}
.g-table td {{ padding: 0.55rem 0.6rem; border-bottom: 1px solid {LINE}; vertical-align: top; }}
.g-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.g-table tr.grp td {{ font-family: 'Bricolage Grotesque', sans-serif; font-weight: 750; font-size: 1.02rem; padding-top: 1.05rem; border-bottom: none; }}
.g-dir {{ color: {MUTED}; font-size: 0.85rem; }}
.g-pill {{ display: inline-block; border-radius: 999px; padding: 0.1rem 0.65rem; font-size: 0.84rem; font-weight: 600; white-space: nowrap; }}
.g-pill.ft {{ background: {FT}; color: #fff; }}
.g-pill.base {{ background: {BASE}; color: #fff; }}
.g-pill.tie {{ background: {SURFACE}; color: {MUTED}; border: 1px solid {LINE}; }}
.g-wrap {{ overflow-x: auto; }}
.g-table, .g-table th, .g-table td {{ border-left: none !important; border-right: none !important; border-top: none !important; }}
.g-table {{ border: none !important; }}
.g-before {{ background: {SURFACE}; border: 1px solid {LINE}; }}
.g-after {{ background: #FFFFFF; border: 1px solid {LINE}; border-left: 5px solid {DATA}; }}

/* empty state */
.g-empty {{ border: 1.5px dashed #C9D1DD; border-radius: 12px; padding: 1.3rem 1.4rem; background: #FBFCFD; margin: 0.6rem 0 1.4rem; }}
.g-empty b {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 1.12rem; }}
.g-empty p {{ margin: 0.35rem 0 0; color: {MUTED}; }}

/* pipeline sequence on overview */
.g-flow {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0; border-top: 1px solid {LINE}; margin-top: 0.6rem; }}
.g-flow div {{ padding: 0.8rem 0.9rem 0.8rem 0; }}
.g-flow span {{ display: block; color: {FT}; font-weight: 600; font-size: 0.9rem; }}
.g-flow b {{ font-weight: 600; }}
.g-note {{ color: {MUTED}; font-size: 0.92rem; }}

/* ===== assistant (chat product) ===== */
.st-key-chatwin {{ max-width: 860px; margin: 0 auto; }}
.c-head {{ display: flex; align-items: center; gap: 0.9rem; padding: 0.95rem 1.15rem; border: 1px solid {LINE}; border-radius: 18px;
          background: #fff; box-shadow: 0 1px 2px rgba(23,35,59,.05), 0 10px 30px rgba(23,35,59,.07); margin: 0.2rem 0 0.6rem; }}
.c-avatar {{ width: 46px; height: 46px; flex: 0 0 46px; border-radius: 50%; background: {FT}; color: #fff; display: grid; place-items: center;
            font-family: 'Bricolage Grotesque', sans-serif; font-weight: 750; font-size: 1.05rem; }}
.c-avatar.sm {{ width: 30px; height: 30px; flex: 0 0 30px; font-size: 0.74rem; }}
.c-avatar.base {{ background: {BASE}; }}
.c-avatar.sys {{ background: {INK}; }}
.c-row.bot .c-msg.sys {{ background: #fff; border: 1px solid {LINE}; }}
.st-key-chips button {{ justify-content: flex-start; }}
.st-key-chips button p {{ text-align: left; }}
.st-key-chips button > div, .st-key-chips button [data-testid="stMarkdownContainer"] {{ justify-content: flex-start; text-align: left; width: 100%; }}
.c-msg li, .c-msg ul {{ font-size: 0.97rem; line-height: 1.5; font-family: inherit; }}
.c-name {{ font-weight: 600; font-size: 1.1rem; line-height: 1.2; }}
.c-sub {{ color: {MUTED}; font-size: 0.88rem; }}
.c-status {{ margin-left: auto; font-size: 0.84rem; color: {MUTED}; display: flex; align-items: center; gap: 0.45rem; white-space: nowrap; }}
.c-status i {{ width: 8px; height: 8px; border-radius: 50%; background: #2E9E5B; display: inline-block; }}
.c-disclose {{ color: {MUTED}; font-size: 0.86rem; margin: 0 0.2rem 1rem; line-height: 1.5; }}
.c-row {{ display: flex; gap: 0.6rem; margin: 0.7rem 0; align-items: flex-end; }}
.c-row.user {{ justify-content: flex-end; }}
.c-msg {{ max-width: 80%; padding: 0.78rem 1.05rem; border-radius: 18px; line-height: 1.58; font-size: 1.01rem; overflow-wrap: anywhere; }}
.c-row.user .c-msg {{ background: {INK}; color: #fff; border-bottom-right-radius: 6px; }}
.c-row.user .g-slot {{ background: rgba(255,255,255,.16); color: #fff; }}
.c-row.bot .c-msg {{ background: {SURFACE}; border: 1px solid {LINE}; border-bottom-left-radius: 6px; }}
.c-row.bot .c-msg.ft {{ background: #F1F8F8; border-color: #CFE6E8; }}
.c-who {{ font-size: 0.8rem; font-weight: 600; color: {MUTED}; margin-bottom: 0.25rem; }}
.c-meta {{ font-size: 0.8rem; color: {MUTED}; margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 0.3rem 0.5rem; }}
.c-tag {{ border: 1px solid {LINE}; background: #fff; border-radius: 999px; padding: 0.05rem 0.55rem; }}
.c-tag.warn {{ border-color: #F2C9C2; background: #FBE9E6; color: {ALERT}; }}
.c-msg details {{ margin-top: 0.55rem; }}
.c-msg details summary {{ cursor: pointer; color: {FT}; font-size: 0.85rem; font-weight: 600; }}
.c-msg details div {{ margin-top: 0.4rem; padding: 0.6rem 0.8rem; border-left: 4px solid {AGENT}; background: #FFFBF2; border-radius: 4px 10px 10px 4px; font-size: 0.95rem; }}
.c-typing {{ display: inline-flex; gap: 5px; padding: 0.2rem 0.1rem; }}
.c-typing span {{ width: 8px; height: 8px; border-radius: 50%; background: {MUTED}; animation: cblink 1.2s infinite both; }}
.c-typing span:nth-child(2) {{ animation-delay: .2s; }} .c-typing span:nth-child(3) {{ animation-delay: .4s; }}
@keyframes cblink {{ 0%, 80%, 100% {{ opacity: .25; }} 40% {{ opacity: 1; }} }}
@media (prefers-reduced-motion: reduce) {{ .c-typing span {{ animation: none; opacity: .6; }} }}
.c-suggest {{ font-size: 0.86rem; color: {MUTED}; font-weight: 600; margin: 1.1rem 0 0.35rem; }}
.st-key-chips button {{ border-radius: 999px; border: 1px solid #CFE6E8; background: #fff; color: #065C63; font-size: 0.9rem;
                       padding: 0.28rem 0.85rem; min-height: 0; text-align: left; white-space: normal; height: auto; }}
.st-key-chips button:hover {{ background: #F1F8F8; border-color: {FT}; color: #065C63; }}
.st-key-chattools button {{ border-radius: 999px; font-size: 0.86rem; padding: 0.2rem 0.8rem; min-height: 0; }}
.st-key-who [role="radiogroup"] {{ gap: 0.4rem; }}
.st-key-who [role="radiogroup"] label {{ border: 1px solid {LINE}; border-radius: 999px; padding: 0.22rem 0.8rem 0.22rem 0.5rem; background: #fff; }}
[data-testid="stChatInput"] textarea {{ font-size: 1rem; }}
[data-testid="stChatInput"] {{ border-radius: 16px; }}
[data-testid="stBottomBlockContainer"] {{ max-width: 900px; margin: 0 auto; }}
[data-testid="stSidebar"] .g-mark {{ width: 38px; height: 38px; border-radius: 12px; background: {FT}; color: #fff; display: grid; place-items: center;
                                    font-family: 'Bricolage Grotesque', sans-serif; font-weight: 750; margin-bottom: 0.6rem; }}
@media (max-width: 640px) {{ .g-title {{ font-size: 1.8rem; }} .g-stat + .g-stat {{ border-left: none; padding-left: 0; }} .c-msg {{ max-width: 92%; }} .c-status {{ display: none; }} }}
</style>
""",
    unsafe_allow_html=True,
)

# shared chart look
pio.templates["g17"] = go.layout.Template(layout=dict(
    font=dict(family="Instrument Sans, Segoe UI, sans-serif", size=14, color=INK),
    title=dict(font=dict(family="Bricolage Grotesque, Segoe UI, sans-serif", size=18, color=INK), x=0, xanchor="left"),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE), yaxis=dict(gridcolor=LINE, zeroline=False, linecolor=LINE),
    colorway=CATS,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
    margin=dict(l=10, r=10, t=56, b=10), hoverlabel=dict(font_family="Instrument Sans, sans-serif"),
))
pio.templates.default = "g17"
PLOT_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(name):
    p = ART / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = ART / name
    return pd.read_csv(p) if p.exists() else None


def model_label(summ, key):
    default = {"baseline": "Baseline (zero-shot)", "finetuned": "Fine-tuned (QLoRA)"}[key]
    return (summ or {}).get("models", {}).get(key, {}).get("label", default)


def model_name(cfg):
    mid = (cfg or {}).get("model_id", "meta-llama/Llama-3.2-1B-Instruct")
    return mid.split("/")[-1].replace("-", " ").replace("Llama 3.2", "Llama 3.2")


def fmt(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:,.{d}f}"


# ----------------------------------------------------------------------------
# UI building blocks
# ----------------------------------------------------------------------------
def header(step, title, lead):
    st.markdown(
        (f'<div class="g-step">Stage {step}</div>' if step else "")
        + f'<div class="g-title">{html.escape(title)}</div><div class="g-lead">{lead}</div>',
        unsafe_allow_html=True,
    )


def stats(items):
    cells = "".join(f'<div class="g-stat"><div class="g-stat-v">{v}</div><div class="g-stat-l">{html.escape(l)}</div></div>'
                    for v, l in items)
    st.markdown(f'<div class="g-stats">{cells}</div>', unsafe_allow_html=True)


def empty(what):
    st.markdown(
        f'<div class="g-empty"><b>{html.escape(what)} not loaded yet</b>'
        "<p>Run the Colab notebook, unzip <code>group17_artifacts.zip</code> into the <code>artifacts</code> folder next to "
        "<code>app.py</code>, then reload this page.</p></div>",
        unsafe_allow_html=True,
    )


def rich(text, flags=()):
    """Escape text, highlight {{slots}} and any flagged invented details."""
    t = html.escape(str(text) if isinstance(text, str) else "(empty answer)")
    for _, val in flags:
        v = html.escape(val)
        if v:
            t = t.replace(v, f'<span class="g-flag" title="Not in the question or reference">{v}</span>')
    t = re.sub(r"\{\{\s*([^{}]+?)\s*\}\}", lambda m: f'<span class="g-slot">{{{{{m.group(1)}}}}}</span>', t)
    return t.replace("\n", "<br>")


def bubble(who, text, css, chips=(), flags=()):
    chip_html = "".join(f'<span class="g-chip">{html.escape(c)}</span>' for c in chips)
    return (f'<div><div class="g-who">{html.escape(who)}</div><div class="g-bubble {css}">{rich(text, flags)}</div>'
            + (f'<div class="g-chips">{chip_html}</div>' if chip_html else "") + "</div>")


def row_chips(r, m):
    out = []
    for col, lab in (("rougeL", "ROUGE-L"), ("bertscore_f1", "BERTScore"), ("relevance_query", "Relevance")):
        k = f"{col}_{m}"
        if k in r and pd.notna(r[k]):
            out.append(f"{lab} {r[k]:.2f}")
    return out


def row_flags(r, m):
    raw = r.get(f"unsupported_detail_{m}")
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [tuple(x.split(":", 1)) for x in raw.split("; ") if ":" in x]


def transcript(r, summ, show_reference=True):
    st.markdown(f'<div class="g-chat">{bubble("Customer", r["instruction"], "g-customer")}</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")
    fb, ff = row_flags(r, "baseline"), row_flags(r, "finetuned")
    c1.markdown(bubble(f"Model 1 · {model_label(summ, 'baseline')}", r["resp_baseline"], "g-m-base", row_chips(r, "baseline"), fb),
                unsafe_allow_html=True)
    c2.markdown(bubble(f"Model 2 · {model_label(summ, 'finetuned')}", r["resp_finetuned"], "g-m-ft", row_chips(r, "finetuned"), ff),
                unsafe_allow_html=True)
    if fb or ff:
        st.markdown('<p class="g-note">Red highlights are concrete details (phone numbers, links, amounts, IDs) that appear in '
                    'neither the question nor the reference answer.</p>', unsafe_allow_html=True)
    if show_reference:
        with st.expander("What the human agent wrote (reference answer)"):
            st.markdown(bubble("Human agent · Bitext dataset", r["reference"], "g-m-agent"), unsafe_allow_html=True)


def chart(fig, height=None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config=PLOT_CFG)



# ----------------------------------------------------------------------------
# Assistant: chat product over the recorded held-out answers
# ----------------------------------------------------------------------------
MATCH_MIN = 0.30   # below this cosine similarity the bot says it has no close recorded answer
WHO = ["Model 2 · fine-tuned", "Model 1 · baseline", "Compare both"]


@st.cache_resource(show_spinner=False)
def build_index(texts: tuple):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
    X = vec.fit_transform([T.strip_placeholders(t, " ") for t in texts])
    return vec, X


def match(gen, q, k=1):
    vec, X = build_index(tuple(gen["instruction"].astype(str)))
    sims = (X @ vec.transform([T.strip_placeholders(q, " ")]).T).toarray().ravel()
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def user_html(text):
    return f'<div class="c-row user"><div class="c-msg">{rich(text)}</div></div>'


def bot_html(body_html, who="Model 2 · fine-tuned", css="ft", meta="", extra=""):
    av = {"base": "base", "sys": "sys"}.get(css, "")
    initials = {"base": "M1", "sys": "G17"}.get(css, "M2")
    return (f'<div class="c-row bot"><div class="c-avatar sm {av}">{initials}</div><div class="c-msg {css}">'
            f'<div class="c-who">{html.escape(who)}</div>{body_html}{extra}'
            + (f'<div class="c-meta">{meta}</div>' if meta else "") + "</div></div>")


def answer_html(gen, summ, turn, model):
    r = gen.iloc[turn["row"]]
    resp = r[f"resp_{model}"]
    flags = row_flags(r, model)
    tags = [f'<span class="c-tag">{html.escape(str(r.get("intent", "")).replace("_", " "))}</span>']
    k = f"rougeL_{model}"
    if k in r and pd.notna(r[k]):
        tags.append(f'<span class="c-tag">ROUGE-L {r[k]:.2f}</span>')
    if flags:
        tags.append(f'<span class="c-tag warn">{len(flags)} unverified detail{"s" if len(flags) > 1 else ""}</span>')
    extra = ""
    if turn["sim"] < 0.995:
        extra += (f'<details><summary>Answered from the closest recorded question ({turn["sim"]:.0%} match)</summary>'
                  f'<div>{rich(r["instruction"])}</div></details>')
    extra += f'<details><summary>Compare with the human agent\'s answer</summary><div>{rich(r["reference"])}</div></details>'
    who = f"Model 2 · {model_label(summ, 'finetuned')}" if model == "finetuned" else f"Model 1 · {model_label(summ, 'baseline')}"
    return bot_html(rich(resp, flags), who, "ft" if model == "finetuned" else "base", " ".join(tags), extra)


def render_turn(gen, summ, turn):
    if turn["role"] == "user":
        return user_html(turn["text"])
    if turn["kind"] == "nomatch":
        sug = "".join(f"<li>{rich(gen.iloc[i]['instruction'])}</li>" for i in turn["alts"])
        return bot_html("I don't have a recorded answer for anything close to that, so I won't guess. "
                        f"The nearest questions I was tested on are:<ul style='margin:.4rem 0 0'>{sug}</ul>",
                        "Assistant", "sys")
    models = {"Model 2 · fine-tuned": ["finetuned"], "Model 1 · baseline": ["baseline"]}.get(turn["who"], ["baseline", "finetuned"])
    return "".join(answer_html(gen, summ, turn, m) for m in models)


def _queue(text):
    st.session_state.queued = text


def page_assistant():
    gen, summ, cfg = load_csv("generations.csv"), load_json("results_summary.json"), load_json("config.json")
    prompt = st.chat_input("Message the support assistant", disabled=gen is None)   # pinned to the bottom of the page

    with st.container(key="chatwin"):
        sub = (f"{model_name(cfg)}, fine-tuned on {cfg['n_train']:,} support conversations" if cfg and cfg.get("n_train")
               else "Llama 3.2 1B Instruct, fine-tuned with QLoRA")
        st.markdown(
            '<div class="c-head"><div class="c-avatar">G17</div><div><div class="c-name">Customer support assistant</div>'
            f'<div class="c-sub">{html.escape(sub)}</div></div>'
            '<div class="c-status"><i></i>Replaying test answers</div></div>'
            '<div class="c-disclose">Every reply is the model\'s real output on a held-out test question. The free hosting tier cannot run '
            'Llama live, so a typed message is matched to the closest recorded question, and the reply shows the match.</div>',
            unsafe_allow_html=True)
        if gen is None:
            return empty("Recorded answers")

        st.session_state.setdefault("thread", [])
        st.session_state.setdefault("sugg_seed", 17)
        who = st.radio("Who answers", WHO, horizontal=True, label_visibility="collapsed", key="who")

        st.markdown(bot_html("Hello. I can help with orders, refunds, payments, delivery and your account. "
                             "Type a message below or pick one of the suggestions.", "Assistant", "sys"), unsafe_allow_html=True)
        for turn in st.session_state.thread:
            st.markdown(render_turn(gen, summ, turn), unsafe_allow_html=True)

        q = (prompt or st.session_state.pop("queued", None) or "").strip()
        if q:
            st.session_state.thread.append({"role": "user", "text": q})
            st.markdown(user_html(q), unsafe_allow_html=True)
            slot = st.empty()
            slot.markdown(bot_html('<div class="c-typing"><span></span><span></span><span></span></div>', "Assistant", "sys"),
                          unsafe_allow_html=True)
            hits = match(gen, q, k=3)
            idx, sim = hits[0]
            turn = ({"role": "bot", "kind": "answer", "row": idx, "sim": sim, "who": who} if sim >= MATCH_MIN
                    else {"role": "bot", "kind": "nomatch", "alts": [i for i, _ in hits]})
            time.sleep(0.55)
            slot.markdown(render_turn(gen, summ, turn), unsafe_allow_html=True)
            st.session_state.thread.append(turn)

        st.markdown('<div class="c-suggest">Try asking</div>', unsafe_allow_html=True)
        rng = np.random.default_rng(st.session_state.sugg_seed)
        if "category" in gen.columns and gen["category"].nunique() >= 4:
            cats = rng.choice(gen["category"].dropna().unique(), size=4, replace=False)
            picks = [int(rng.choice(np.flatnonzero(gen["category"].values == c))) for c in cats]
        else:
            picks = [int(i) for i in rng.choice(len(gen), size=min(4, len(gen)), replace=False)]
        with st.container(key="chips"):
            cols = st.columns(2)
            for j, i in enumerate(picks):
                text = str(gen.iloc[i]["instruction"])
                cols[j % 2].button(text if len(text) <= 70 else text[:67] + "…", key=f"chip_{i}_{j}", on_click=_queue, args=(text,),
                                   width="stretch")
        with st.container(key="chattools"):
            c1, c2, _ = st.columns([1.3, 1.3, 3])
            if c1.button("More suggestions"):
                st.session_state.sugg_seed += 1
                st.rerun()
            if c2.button("New conversation", disabled=not st.session_state.thread):
                st.session_state.thread = []
                st.rerun()


# ----------------------------------------------------------------------------
# Report pages
# ----------------------------------------------------------------------------
def page_overview():
    eda, summ, gen, cfg = load_json("eda_stats.json"), load_json("results_summary.json"), load_csv("generations.csv"), load_json("config.json")
    n_tr = f"{cfg['n_train']:,} " if cfg and cfg.get("n_train") else ""
    st.markdown('<div class="g-hero">Teaching a small Llama to answer customers</div>'
                f'<div class="g-hero-sub">We took {html.escape(model_name(cfg))}, asked it customer-service questions, then fine-tuned it on '
                f'{n_tr}support conversations and asked the same questions again. No retrieval, no external knowledge: everything the bot '
                'knows has to live in its weights.</div>', unsafe_allow_html=True)

    if gen is not None and len(gen):
        st.markdown("#### One held-out question, both models")
        if "hero_idx" not in st.session_state:
            st.session_state.hero_idx = int(np.random.default_rng(17).integers(len(gen)))
        if st.button("Show another question"):
            st.session_state.hero_idx = int(np.random.default_rng().integers(len(gen)))
        transcript(gen.iloc[st.session_state.hero_idx % len(gen)], summ)
    else:
        empty("Example answers")

    items = []
    if eda:
        items += [(f"{eda['n_rows']:,}", "support conversations after cleaning"), (eda["n_intents"], "customer intents"),
                  (eda["n_categories"], "service categories")]
    if summ:
        items.append((summ["n_test"], "held-out test questions"))
        b = summ.get("best_model")
        if b in ("baseline", "finetuned"):
            items.append((f"{summ['wins'][b]} of {sum(v for k, v in summ['wins'].items())}", f"metrics won by {'Model 2' if b == 'finetuned' else 'Model 1'}"))
    if items:
        stats(items)

    st.markdown("#### How the project runs")
    steps = [("Clean", "regex, slots kept"), ("Explore", "intents, lengths, n-grams"), ("Represent", "Llama tokens and embeddings"),
             ("Model 1", "zero-shot baseline"), ("Model 2", "QLoRA fine-tuning"), ("Evaluate", "five metric families"),
             ("Decide", "paired comparison")]
    st.markdown('<div class="g-flow">' + "".join(f"<div><span>{i}</span><b>{a}</b><br><span class='g-note' style='display:inline;color:{MUTED};font-weight:400'>{b}</span></div>"
                                                for i, (a, b) in enumerate(steps, 1)) + "</div>", unsafe_allow_html=True)


def page_data():
    header(1, "Cleaning the conversations",
           "Each record pairs a customer message with an agent's answer, an intent and a category. Personal details appear as "
           "slots such as <span class='g-slot'>{{Order Number}}</span>. We keep the slots, because a good bot should write "
           "the slot instead of inventing an order number.")
    log = load_json("cleaning_log.json")
    if log:
        stats([(f"{log['rows_raw']:,}", "raw records"), (log["dropped_empty"], "empty records removed"),
               (log["dropped_duplicate_questions"], "duplicate questions removed"), (f"{log['rows_clean']:,}", "clean records")])
        if log.get("regex_examples"):
            st.markdown("#### What cleaning changed in the real data")
            for ex in log["regex_examples"][:3]:
                c1, c2 = st.columns(2, gap="medium")
                c1.markdown(bubble("Before", ex["raw"], "g-before"), unsafe_allow_html=True)
                c2.markdown(bubble("After", ex["cleaned"], "g-after"), unsafe_allow_html=True)
    else:
        empty("Cleaning log")

    st.markdown("#### Try the cleaner")
    st.markdown('<p class="g-note">This runs the same regular expressions that cleaned the dataset.</p>', unsafe_allow_html=True)
    sample = "  I\u2019m trying to cancel order {{ Order Number }}  \u2014 email me at jane@shop.com or call +233 24 123 4567 "
    txt = st.text_area("Customer message", sample, height=96, label_visibility="collapsed")
    rep = T.clean_report(txt)
    c1, c2 = st.columns([3, 2], gap="medium")
    c1.markdown(bubble("Cleaned text", rep["cleaned"], "g-after"), unsafe_allow_html=True)
    slots = ", ".join(rep["placeholders"]) or "none"
    specs = "".join(f"<li><b>{html.escape(k)}</b>: {html.escape(v)}</li>" for k, v in rep["specifics"]) or "<li>none</li>"
    c2.markdown(f"<div class='g-who'>Slots found</div><p>{html.escape(slots)}</p>"
                f"<div class='g-who'>Details the hallucination check would flag if a bot made them up</div><ul>{specs}</ul>",
                unsafe_allow_html=True)


def page_eda():
    header(2, "What customers ask about",
           "Volume by category and intent, how long questions and answers are, and the words and slots that recur.")
    eda = load_json("eda_stats.json")
    if not eda:
        return empty("EDA statistics")
    stats([(f"{eda['len_words']['instruction']['mean']:.1f}", "words per question, on average"),
           (f"{eda['len_words']['response']['mean']:.0f}", "words per answer, on average"),
           (f"{eda['pct_responses_with_slot']:.0f}%", "of answers contain a slot"),
           (f"{eda['pct_instructions_with_slot']:.0f}%", "of questions contain a slot")])

    c1, c2 = st.columns([2, 3], gap="large")
    cc = pd.Series(eda["category_counts"]).sort_values().rename_axis("category").reset_index(name="records")
    with c1:
        chart(px.bar(cc, x="records", y="category", orientation="h", title="Records per category",
                     color_discrete_sequence=[DATA]).update_layout(yaxis_title="", xaxis_title=""), 520)
    ic = pd.Series(eda["intent_counts"]).sort_values().rename_axis("intent").reset_index(name="records")
    ic["category"] = ic["intent"].map(eda.get("intent_to_category", {}))
    with c2:
        chart(px.bar(ic, x="records", y="intent", color="category", orientation="h", title="Records per intent, coloured by category")
              .update_layout(yaxis_title="", xaxis_title="", showlegend=False), 640)

    fig = go.Figure()
    for col, color, name in (("instruction", DATA, "Customer questions"), ("response", AGENT, "Agent answers")):
        h = eda["len_hist"][col]; e = np.array(h["bins"])
        fig.add_bar(x=(e[:-1] + e[1:]) / 2, y=h["counts"], name=name, marker_color=color, opacity=0.85)
    fig.update_layout(barmode="overlay", title="Answers are several times longer than questions", xaxis_title="words", yaxis_title="records")
    chart(fig, 380)

    c1, c2 = st.columns(2, gap="large")
    sp = pd.DataFrame(eda["placeholder_top"][:12], columns=["slot", "count"]).iloc[::-1]
    with c1:
        chart(px.bar(sp, x="count", y="slot", orientation="h", title="Most frequent slots", color_discrete_sequence=[DATA])
              .update_layout(yaxis_title="", xaxis_title=""), 430)
    with c2:
        who = st.radio("Bigrams from", ["Questions", "Answers"], horizontal=True, label_visibility="collapsed")
        key = "instruction" if who == "Questions" else "response"
        ng = pd.DataFrame(eda["top_bigrams"][key][:12], columns=["bigram", "count"]).iloc[::-1]
        chart(px.bar(ng, x="count", y="bigram", orientation="h", title=f"Most common word pairs in {who.lower()}",
                     color_discrete_sequence=[DATA if key == "instruction" else AGENT]).update_layout(yaxis_title="", xaxis_title=""), 390)
    if "splits" in eda:
        sp = eda["splits"]
        st.markdown(f'<p class="g-note">Split 80/10/10, stratified by intent: {sp.get("train", 0):,} train, {sp.get("val", 0):,} validation, '
                    f'{sp.get("test", 0):,} test records. Duplicate questions were removed first, so no test question appears in training.</p>',
                    unsafe_allow_html=True)


def page_repr():
    header(3, "How Llama reads the text",
           "Llama splits text into sub-word tokens and turns each into a learned vector. Averaging a message's vectors gives "
           "one point per message, which lets us check whether questions with the same intent sit together.")
    ts = load_json("token_stats.json")
    if not ts:
        return empty("Token statistics")
    stats([(f"{ts['vocab_size']:,}", "tokens in Llama's vocabulary"), (f"{ts['tokens_per_word_response']:.2f}", "tokens per word in answers"),
           (ts["max_len_chosen"], "token limit per training example"), (f"{ts['pct_truncated_at_max_len']:.1f}%", "of examples cut by that limit")])
    ex = ts.get("example")
    if ex:
        st.markdown("#### One sentence, as Llama sees it")
        toks = "".join(f'<span class="g-chip" style="margin:0 .25rem .35rem 0;display:inline-block;color:{INK}">{html.escape(t.replace("Ġ", "·"))}</span>'
                       for t in ex["tokens"])
        st.markdown(f'<p>{rich(ex["text"])}</p><div>{toks}</div>'
                    f'<p class="g-note">A dot marks a token that starts with a space. {len(ex["tokens"])} tokens for this sentence.</p>',
                    unsafe_allow_html=True)
    pca = load_csv("embedding_pca.csv")
    if pca is not None:
        st.markdown("#### Customer messages in Llama's embedding space")
        color = st.radio("Colour by", ["category", "intent"], horizontal=True)
        fig = px.scatter(pca, x="x", y="y", color=color, hover_data=["intent", "category"])
        fig.update_traces(marker=dict(size=7, opacity=0.82, line=dict(width=0)))
        fig.update_layout(xaxis_title="principal component 1", yaxis_title="principal component 2",
                          legend=dict(orientation="v", x=1.01, y=1, yanchor="top"))
        chart(fig, 560)
        v = ts.get("embedding_pca_variance_explained", [])
        sil = ts.get("embedding_silhouette_by_intent")
        st.markdown(f'<p class="g-note">Silhouette score by intent: <b>{fmt(sil)}</b> (1 means perfectly separated groups, 0 means overlapping). '
                    f'These two axes keep {sum(v) * 100:.0f}% of the variation, so the full space separates intents better than this flat view suggests.</p>',
                    unsafe_allow_html=True)


def page_models():
    header(4, "Two models, one difference",
           "Both models share the same weights, system prompt, test questions and greedy decoding. The only change is "
           "supervised fine-tuning, so any gap in the results comes from it.")
    cfg, tl = load_json("config.json"), load_json("training_log.json")
    c1, c2 = st.columns(2, gap="medium")
    c1.markdown(bubble("Model 1 · Baseline",
                       "Llama 3.2 1B Instruct, loaded in 4-bit. Answers from a system prompt only, with no training on support data.",
                       "g-m-base"), unsafe_allow_html=True)
    c2.markdown(bubble("Model 2 · Fine-tuned",
                       "The same model with small LoRA adapters trained on support conversations. Only the answer tokens count "
                       "towards the loss, so it learns how to reply rather than to repeat the question.", "g-m-ft"),
                unsafe_allow_html=True)
    if cfg:
        st.markdown("")
        stats([(f"{cfg['n_train']:,}", "training conversations"), (f"{cfg['lora']['r']} / {cfg['lora']['alpha']}", "LoRA rank / alpha"),
               (f"{cfg['lr']:g}", "learning rate"), (cfg["epochs"], "epoch(s)")])
        if cfg.get("system_prompt"):
            with st.expander("System prompt used for both models"):
                st.markdown(bubble("System", cfg["system_prompt"], "g-customer"), unsafe_allow_html=True)
    if not tl:
        return empty("Training log")
    lh = pd.DataFrame(tl["log_history"])
    fig = go.Figure()
    if "loss" in lh:
        d = lh.dropna(subset=["loss"]); fig.add_scatter(x=d["step"], y=d["loss"], name="training loss", mode="lines", line=dict(color=FT, width=2.5))
    if "eval_loss" in lh:
        d = lh.dropna(subset=["eval_loss"]); fig.add_scatter(x=d["step"], y=d["eval_loss"], name="validation loss", mode="lines+markers",
                                                             line=dict(color=FT, width=2, dash="dot"), marker=dict(size=8, symbol="circle-open"))
    fig.update_layout(title="Loss falls as the model learns the support style", xaxis_title="optimizer step", yaxis_title="loss")
    chart(fig, 400)
    st.markdown(f'<p class="g-note">{tl["trainable_params"]:,} of {tl["total_params"]:,} parameters trained '
                f'({100 * tl["trainable_params"] / tl["total_params"]:.2f}%), in {tl["train_seconds"] / 60:.0f} minutes on one free T4 GPU.</p>',
                unsafe_allow_html=True)


def _diff_cell(m, d):
    if d["delta"] is None:
        return "n/a"
    if m == "log_perplexity":   # difference of logs = ratio of geometric means
        return f"×{math.exp(d['delta']):.2f} <span class='g-dir'>[{math.exp(d['ci_low']):.2f}, {math.exp(d['ci_high']):.2f}]</span>"
    return f"{d['delta']:+.3f} <span class='g-dir'>[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]</span>"


def scorecard_html(summ):
    groups = {}
    for m, meta in summ["metric_meta"].items():
        groups.setdefault(meta["group"], []).append(m)
    rows = []
    for grp, ms in groups.items():
        rows.append(f'<tr class="grp"><td colspan="5">{html.escape(grp)}</td></tr>')
        for m in ms:
            meta = summ["metric_meta"][m]
            a = summ["models"]["baseline"]["metrics"][m]["mean"]
            b = summ["models"]["finetuned"]["metrics"][m]["mean"]
            if m == "log_perplexity":
                a, b = (math.exp(x) if x is not None else None for x in (a, b))
                name = "Perplexity under GPT-2"
            else:
                name = meta["label"]
            direction = {"higher": "higher is better", "lower": "lower is better", None: "descriptive"}[meta["direction"]]
            w = summ["scorecard"][m]
            pill = {"finetuned": '<span class="g-pill ft">Model 2</span>', "baseline": '<span class="g-pill base">Model 1</span>',
                    "tie": '<span class="g-pill tie">No clear difference</span>', None: ""}[w]
            rows.append(f'<tr><td>{html.escape(name)}<br><span class="g-dir">{direction}</span></td>'
                        f'<td class="num">{fmt(a)}</td><td class="num">{fmt(b)}</td>'
                        f'<td class="num">{_diff_cell(m, summ["tests"][m])}</td><td>{pill}</td></tr>')
    head = ("<tr><th>Metric</th><th style='text-align:right'>Model 1</th><th style='text-align:right'>Model 2</th>"
            "<th style='text-align:right'>Difference [95% interval]</th><th>Better model</th></tr>")
    return f'<div class="g-wrap"><table class="g-table"><thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>'


def verdict(summ):
    best, w = summ["best_model"], summ["wins"]
    total = w["baseline"] + w["finetuned"] + w["ties"]
    if best == "tie":
        st.markdown(f'<div class="g-verdict tie"><h3>No overall winner</h3><p>Model 1 won {w["baseline"]} metrics, Model 2 won '
                    f'{w["finetuned"]}, and {w["ties"]} showed no clear difference.</p></div>', unsafe_allow_html=True)
    else:
        other = "baseline" if best == "finetuned" else "finetuned"
        st.markdown(f'<div class="g-verdict"><h3>{html.escape(model_label(summ, best))} is the better chatbot</h3>'
                    f'<p>It won {w[best]} of {total} scored metrics; the other model won {w[other]}, and {w["ties"]} showed no clear difference.</p></div>',
                    unsafe_allow_html=True)


def page_eval():
    header(5, "Which model answers better",
           "Both models answered the same held-out questions. For each metric we take the difference per question and put a "
           "bootstrap 95% interval around the average. A model wins a metric only when that interval sits entirely on its side of zero.")
    summ = load_json("results_summary.json")
    if not summ:
        return empty("Results summary")
    verdict(summ)
    st.markdown(scorecard_html(summ), unsafe_allow_html=True)
    if not summ.get("judge_available"):
        st.markdown('<p class="g-note">LLM-judge scores were not collected in this run; the automatic metrics cover each criterion.</p>',
                    unsafe_allow_html=True)

    keys = [k for k in ("rougeL", "bertscore_f1", "sim_reference", "relevance_query", "slot_recall", "halluc_proxy", "degenerate")
            if k in summ["metric_meta"]]
    short = {"rougeL": "ROUGE-L", "bertscore_f1": "BERTScore", "sim_reference": "Similarity to reference", "relevance_query": "Relevance to question",
             "slot_recall": "Slot recall", "halluc_proxy": "Hallucination rate", "degenerate": "Broken answers"}
    fig = go.Figure()
    for mod, color in (("baseline", BASE), ("finetuned", FT)):
        mm = [summ["models"][mod]["metrics"][k] for k in keys]
        fig.add_bar(name=model_label(summ, mod), x=[short[k] for k in keys], y=[m["mean"] for m in mm], marker_color=color,
                    error_y=dict(type="data", symmetric=False, color=INK, thickness=1.2, width=4,
                                 array=[m["ci_high"] - m["mean"] for m in mm], arrayminus=[m["mean"] - m["ci_low"] for m in mm]))
    fig.update_layout(barmode="group", bargap=0.28, title="Headline metrics with 95% intervals", yaxis_title="score or rate")
    chart(fig, 430)

    if summ.get("by_category"):
        metric = st.selectbox("Break down by category", [m for m in ("rougeL", "bertscore_f1", "halluc_proxy") if m in summ["metric_meta"]],
                              format_func=lambda m: summ["metric_meta"][m]["label"])
        rows = [{"category": c, "model": model_label(summ, mod), "value": v[metric][mod]}
                for c, v in summ["by_category"].items() if metric in v for mod in ("baseline", "finetuned")]
        fig = px.bar(pd.DataFrame(rows), x="category", y="value", color="model", barmode="group",
                     color_discrete_map={model_label(summ, "baseline"): BASE, model_label(summ, "finetuned"): FT})
        fig.update_layout(xaxis_title="", yaxis_title=summ["metric_meta"][metric]["label"])
        chart(fig, 400)
        st.markdown('<p class="g-note">Each category has only a few test questions, so treat gaps between categories as indicative.</p>',
                    unsafe_allow_html=True)

    hum = load_json("human_ratings_summary.json")
    if hum and hum.get("n_raters"):
        st.markdown(f"#### Blind ratings from {hum['n_raters']} group member(s) on {hum['n_items']} answers")
        rows = [{"criterion": c.replace("_0or1", " (share flagged)"), "model": model_label(summ, m), "mean": v["mean"]}
                for m, cs in hum["means"].items() for c, v in cs.items()]
        fig = px.bar(pd.DataFrame(rows), x="criterion", y="mean", color="model", barmode="group",
                     color_discrete_map={model_label(summ, "baseline"): BASE, model_label(summ, "finetuned"): FT})
        fig.update_layout(xaxis_title="", yaxis_title="mean rating")
        chart(fig, 380)


def page_best():
    header(6, "What we conclude",
           "The verdict in plain language, where the better model still fails, and what these results can and cannot show.")
    summ = load_json("results_summary.json")
    if not summ:
        return empty("Results summary")
    verdict(summ)
    groups = {}
    for m, meta in summ["metric_meta"].items():
        if summ["scorecard"][m] is not None:
            groups.setdefault(meta["group"], []).append(m)
    c1, c2 = st.columns(2, gap="large")
    for i, (grp, ms) in enumerate(groups.items()):
        lines = []
        for m in ms:
            win, d = summ["scorecard"][m], summ["tests"][m]
            name = "Perplexity" if m == "log_perplexity" else summ["metric_meta"][m]["label"]
            res = "no clear difference" if win == "tie" else f"{'Model 2' if win == 'finetuned' else 'Model 1'} is better"
            lines.append(f"<li><b>{html.escape(name)}</b>: {res} ({_diff_cell(m, d)})</li>")
        (c1 if i % 2 == 0 else c2).markdown(f"<h4>{html.escape(grp)}</h4><ul>{''.join(lines)}</ul>", unsafe_allow_html=True)

    desc = [m for m in ("n_words", "latency_s") if m in summ["metric_meta"]]
    if desc:
        st.markdown('<p class="g-note">' + " ".join(
            f"{summ['metric_meta'][m]['label']}: Model 1 {summ['models']['baseline']['metrics'][m]['mean']:.2f}, "
            f"Model 2 {summ['models']['finetuned']['metrics'][m]['mean']:.2f}." for m in desc) + "</p>", unsafe_allow_html=True)

    worst = load_csv("error_analysis_worst.csv")
    if worst is not None and len(worst):
        st.markdown("#### Where Model 2 still struggles")
        st.markdown('<p class="g-note">The fine-tuned answers that matched the reference least, by ROUGE-L.</p>', unsafe_allow_html=True)
        pick = st.selectbox("Example", worst.index, format_func=lambda i: str(worst.at[i, "instruction"])[:100])
        r = worst.loc[pick]
        st.markdown(f'<div class="g-chat">{bubble("Customer", r["instruction"], "g-customer")}</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2, gap="medium")
        c1.markdown(bubble("Model 2 · Fine-tuned", r["resp_finetuned"], "g-m-ft", [f"ROUGE-L {r['rougeL_finetuned']:.2f}"]), unsafe_allow_html=True)
        c2.markdown(bubble("Human agent · reference", r["reference"], "g-m-agent"), unsafe_allow_html=True)
    hall = load_csv("hallucination_examples_finetuned.csv")
    if hall is not None and len(hall):
        with st.expander(f"Model 2 answers flagged by the hallucination check ({len(hall)})"):
            for _, r in hall.iterrows():
                flags = [tuple(x.split(":", 1)) for x in str(r["unsupported_detail_finetuned"]).split("; ") if ":" in x]
                st.markdown(bubble(str(r["instruction"])[:120], r["resp_finetuned"], "g-m-ft", flags=flags), unsafe_allow_html=True)

    st.markdown("#### Limits of this study")
    st.markdown(
        "- ROUGE and BERTScore compare against a single reference answer, so a good answer worded differently scores lower.\n"
        "- The hallucination check only catches invented concrete details such as phone numbers, links, amounts and IDs. It is not a full fact-check.\n"
        "- One training run and one test sample: the intervals reflect which questions were sampled, not run-to-run variation.\n"
        "- Bitext conversations are templated and clean; real customers write messier messages.\n"
        "- Without retrieval the bot cannot know a company's real policies, prices or order data, which is why the slots matter."
    )


PAGES = {
    "Assistant": page_assistant,
    "About the project": page_overview,
    "1. Cleaning": page_data,
    "2. Exploring the data": page_eda,
    "3. Representation": page_repr,
    "4. The two models": page_models,
    "5. Evaluation": page_eval,
    "6. Conclusions": page_best,
}

with st.sidebar:
    st.markdown('<div class="g-mark">G17</div><div class="g-brand">Customer support assistant</div>'
                '<div class="g-brand-sub">Group 17, MSBA 610 Advanced Text Analytics. A Llama chatbot with no retrieval.</div>',
                unsafe_allow_html=True)
    choice = st.radio("Go to", list(PAGES), label_visibility="collapsed")
    st.markdown(f'<div class="g-legend"><span class="g-dot" style="background:{BASE}"></span>Model 1, baseline<br>'
                f'<span class="g-dot" style="background:{FT}"></span>Model 2, fine-tuned<br>'
                f'<span class="g-dot" style="background:{AGENT}"></span>Human agent reference</div>', unsafe_allow_html=True)

PAGES[choice]()
