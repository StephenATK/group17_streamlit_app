"""
g17_text.py  --  Group 17 (Llama customer-service chatbot, no RAG)

Shared regular-expression and text-cleaning helpers.
Used by BOTH the Colab notebook and the Streamlit app, so the cleaning
shown in the app is exactly the cleaning applied to the data.

Pure standard library: no heavy dependencies.
"""
import re
import unicodedata
from collections import Counter

# --------------------------------------------------------------------------
# 1. Regular expressions
# --------------------------------------------------------------------------
# Bitext marks variable slots as {{Order Number}}, {{Person Name}}, etc.
# The chatbot should learn to PRODUCE these slots, so cleaning keeps them
# (it only normalises their spacing).
PLACEHOLDER_RE = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')]+", re.I)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# international-ish phone numbers: +233 24 123 4567, (555) 123-4567, 555-123-4567
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3}[\s.-]?\d{3,4}(?!\w)")
MONEY_RE = re.compile(r"(?:[$€£₵]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|GHS|dollars?|euros?|cedis?)\b)", re.I)
LONGNUM_RE = re.compile(r"\b\d{5,}\b")          # order/account/tracking style numbers
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
MULTISPACE_RE = re.compile(r"[ \t\u00a0]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")
CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")

_SMART = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
}


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def normalise_placeholders(text: str) -> str:
    """'{{ Order Number }}' -> '{{Order Number}}' (consistent slot spelling)."""
    return PLACEHOLDER_RE.sub(lambda m: "{{" + m.group(1).strip() + "}}", text)


def clean_text(text) -> str:
    """Light-touch cleaning that PRESERVES meaning and {{placeholders}}.

    Steps: NFKC unicode normalisation, smart-quote/dash folding, control
    character removal, whitespace collapsing, placeholder spacing fix.
    Case is deliberately NOT changed: the chatbot must learn natural casing.
    """
    if text is None:
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    for k, v in _SMART.items():
        text = text.replace(k, v)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = MULTISPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    text = normalise_placeholders(text)
    return text.strip()


def extract_placeholders(text: str):
    """Return the list of slot names found, e.g. ['Order Number', 'Person Name']."""
    return [m.group(1).strip() for m in PLACEHOLDER_RE.finditer(text or "")]


def strip_placeholders(text: str, token: str = "<SLOT>") -> str:
    """Replace slots with a neutral token (used for word statistics only)."""
    return PLACEHOLDER_RE.sub(token, text or "")


def words(text: str, lower: bool = True):
    """Word tokens for EDA (slots removed first so they do not skew counts)."""
    t = PLACEHOLDER_RE.sub(" ", text or "")
    toks = WORD_RE.findall(t)
    return [w.lower() for w in toks] if lower else toks


# --------------------------------------------------------------------------
# 3. Regex-based quality checks (used for the hallucination PROXY)
# --------------------------------------------------------------------------
def concrete_specifics(text: str):
    """Concrete, checkable details a customer-service bot might invent."""
    t = PLACEHOLDER_RE.sub(" ", text or "")
    found = []
    found += [("url", m.group(0).rstrip(".,;")) for m in URL_RE.finditer(t)]
    found += [("email", m.group(0)) for m in EMAIL_RE.finditer(t)]
    found += [("money", m.group(0)) for m in MONEY_RE.finditer(t)]
    found += [("phone", m.group(0).strip()) for m in PHONE_RE.finditer(t)
              if len(re.sub(r"\D", "", m.group(0))) >= 7]
    found += [("number", m.group(0)) for m in LONGNUM_RE.finditer(t)]
    return found


def _norm(s: str) -> str:
    return re.sub(r"[\s\-().,]", "", s.lower())


def unsupported_specifics(response: str, *sources: str):
    """Specifics present in `response` but absent from every source text.

    sources = the customer's message and the reference answer. A specific that
    appears in neither was invented by the model. This is a conservative,
    regex-based PROXY for hallucination, not a full factuality check.
    """
    pool = " ".join(_norm(s) for s in sources if s)
    return [(kind, val) for kind, val in concrete_specifics(response)
            if _norm(val) not in pool]


def placeholder_scores(response: str, reference: str, known_slots=None):
    """Slot-level quality of a generated answer vs the reference answer.

    recall   : share of the reference's slots the model reproduced
    invalid  : slots in the response that do not exist in the dataset's slot
               inventory (i.e. made-up slots)
    """
    ref = set(extract_placeholders(reference))
    got = extract_placeholders(response)
    recall = (len(ref & set(got)) / len(ref)) if ref else None
    invalid = 0
    if known_slots is not None:
        invalid = sum(1 for g in got if g not in known_slots)
    return {"slot_recall": recall, "invalid_slots": invalid, "n_slots": len(got)}


# --------------------------------------------------------------------------
# 4. Fluency / degeneration helpers
# --------------------------------------------------------------------------
def distinct_n(text: str, n: int = 2) -> float:
    toks = words(text)
    if len(toks) < n:
        return 1.0 if toks else 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


def is_degenerate(text: str) -> bool:
    """Empty, extremely short, or heavily repetitive output counts as an error."""
    toks = words(text)
    if len(toks) < 3:
        return True
    return distinct_n(text, 2) < 0.5


def top_ngrams(texts, n: int = 2, k: int = 20, stop=None):
    stop = stop or set()
    c = Counter()
    for t in texts:
        toks = [w for w in words(t) if w not in stop]
        c.update(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return [(" ".join(g), v) for g, v in c.most_common(k)]


def clean_report(raw: str) -> dict:
    """Used by the app's live regex demo: show what the cleaner does."""
    cleaned = clean_text(raw)
    return {
        "raw": raw,
        "cleaned": cleaned,
        "placeholders": extract_placeholders(cleaned),
        "specifics": concrete_specifics(cleaned),
        "n_words": len(words(cleaned)),
    }
