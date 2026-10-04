"""
g17_eval.py  --  Group 17 (Llama customer-service chatbot, no RAG)

GPU-free helpers: dataset statistics, stratified sampling, paired bootstrap
comparison of two models, scorecard / best-model rule, and human-rating
aggregation. Depends only on numpy + pandas (+ g17_text).
"""
import json
import numpy as np
import pandas as pd

import g17_text as T

MODELS = ("baseline", "finetuned")

# direction: "higher" / "lower" is better; None = descriptive only (no winner)
METRIC_META = {
    # answer correctness
    "rougeL":            {"label": "ROUGE-L F1 vs reference",          "direction": "higher", "group": "Correctness"},
    "bertscore_f1":      {"label": "BERTScore F1 vs reference",        "direction": "higher", "group": "Correctness"},
    "slot_recall":       {"label": "Slot ({{...}}) recall",            "direction": "higher", "group": "Correctness"},
    "judge_correctness": {"label": "Judge: correctness (1-5)",         "direction": "higher", "group": "Correctness"},
    # relevance
    "relevance_query":   {"label": "Relevance: cos(query, answer)",    "direction": "higher", "group": "Relevance"},
    "sim_reference":     {"label": "Similarity: cos(reference, answer)", "direction": "higher", "group": "Relevance"},
    "judge_relevance":   {"label": "Judge: relevance (1-5)",           "direction": "higher", "group": "Relevance"},
    # helpfulness
    "judge_helpfulness": {"label": "Judge: helpfulness (1-5)",         "direction": "higher", "group": "Helpfulness"},
    # fluency
    "log_perplexity":    {"label": "Fluency: log-perplexity (GPT-2)",  "direction": "lower",  "group": "Fluency"},
    "distinct2":         {"label": "Distinct-2 (variety)",             "direction": "higher", "group": "Fluency"},
    "judge_fluency":     {"label": "Judge: fluency (1-5)",             "direction": "higher", "group": "Fluency"},
    # hallucination / error rate
    "halluc_proxy":      {"label": "Hallucination proxy rate (regex)", "direction": "lower",  "group": "Hallucination / error"},
    "invalid_slot_rate": {"label": "Invalid-slot rate",                "direction": "lower",  "group": "Hallucination / error"},
    "degenerate":        {"label": "Degenerate-output rate",           "direction": "lower",  "group": "Hallucination / error"},
    "judge_halluc":      {"label": "Judge: hallucination rate",        "direction": "lower",  "group": "Hallucination / error"},
    # descriptive
    "n_words":           {"label": "Answer length (words)",            "direction": None,     "group": "Descriptive"},
    "latency_s":         {"label": "Latency per answer (s)",           "direction": None,     "group": "Descriptive"},
}


# --------------------------------------------------------------------------
# Sampling / splitting
# --------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, n: int, by: str = "intent", seed: int = 42):
    """Sample exactly n rows (largest-remainder allocation) keeping intent proportions."""
    if n is None or n >= len(df):
        return df.copy()
    sizes = df.groupby(by).size()
    exact = sizes / len(df) * n
    alloc = np.floor(exact).astype(int)
    short = int(n - alloc.sum())
    if short > 0:
        alloc[(exact - alloc).sort_values(ascending=False).index[:short]] += 1
    parts = [g.sample(min(int(alloc[k]), len(g)), random_state=seed) for k, g in df.groupby(by)]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def split_stratified(df: pd.DataFrame, seed: int = 42, val: float = 0.1, test: float = 0.1, by: str = "intent"):
    """80/10/10 split, stratified by intent. Returns (train, val, test)."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for _, g in df.groupby(by):
        idx = rng.permutation(len(g))
        n_te = max(1, int(round(len(g) * test)))
        n_va = max(1, int(round(len(g) * val)))
        te.append(g.iloc[idx[:n_te]])
        va.append(g.iloc[idx[n_te:n_te + n_va]])
        tr.append(g.iloc[idx[n_te + n_va:]])
    cat = lambda xs: pd.concat(xs).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return cat(tr), cat(va), cat(te)


# --------------------------------------------------------------------------
# Cleaning + EDA statistics
# --------------------------------------------------------------------------
def clean_dataframe(raw: pd.DataFrame):
    """Apply g17_text.clean_text, drop empties and duplicate questions.

    Returns (clean_df, cleaning_log). Duplicates are judged on the lower-cased
    question text so the same question cannot appear in train AND test.
    """
    log = {"rows_raw": int(len(raw)), "columns": list(raw.columns)}
    df = raw.copy()
    for col in ("instruction", "response"):
        before = df[col].astype(str)
        df[col] = before.map(T.clean_text)
        log[f"changed_{col}"] = int((before != df[col]).sum())
    empty = (df["instruction"] == "") | (df["response"] == "")
    log["dropped_empty"] = int(empty.sum())
    df = df[~empty]
    key = df["instruction"].str.lower()
    dup = key.duplicated(keep="first")
    log["dropped_duplicate_questions"] = int(dup.sum())
    df = df[~dup].reset_index(drop=True)
    log["rows_clean"] = int(len(df))
    ex = []
    for i in raw.index:
        r, c = str(raw.at[i, "instruction"]), T.clean_text(raw.at[i, "instruction"])
        if r != c:
            ex.append({"raw": r, "cleaned": c})
        if len(ex) >= 5:
            break
    log["regex_examples"] = ex
    return df, log


def _len_summary(s: pd.Series):
    return {"mean": float(s.mean()), "median": float(s.median()),
            "p95": float(s.quantile(0.95)), "min": float(s.min()), "max": float(s.max())}


def _hist(s: pd.Series, bins: int = 30):
    hi = float(s.quantile(0.99))
    counts, edges = np.histogram(s.clip(upper=hi), bins=bins, range=(0, hi))
    return {"bins": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def build_eda_stats(df: pd.DataFrame, splits: dict | None = None, stop=None):
    """Everything the EDA page needs, as plain JSON-serialisable data."""
    stop = stop or set()
    d = df.copy()
    d["n_words_instruction"] = d["instruction"].map(lambda x: len(T.words(x)))
    d["n_words_response"] = d["response"].map(lambda x: len(T.words(x)))
    slots_resp = d["response"].map(T.extract_placeholders)
    slots_inst = d["instruction"].map(T.extract_placeholders)
    from collections import Counter
    slot_c = Counter(s for xs in list(slots_resp) + list(slots_inst) for s in xs)
    stats = {
        "n_rows": int(len(d)),
        "n_intents": int(d["intent"].nunique()),
        "n_categories": int(d["category"].nunique()),
        "category_counts": d["category"].value_counts().to_dict(),
        "intent_counts": d["intent"].value_counts().to_dict(),
        "intent_to_category": d.drop_duplicates("intent").set_index("intent")["category"].to_dict(),
        "len_words": {"instruction": _len_summary(d["n_words_instruction"]),
                      "response": _len_summary(d["n_words_response"])},
        "len_hist": {"instruction": _hist(d["n_words_instruction"]),
                     "response": _hist(d["n_words_response"])},
        "placeholder_top": slot_c.most_common(25),
        "placeholder_inventory": sorted(slot_c),
        "pct_instructions_with_slot": float((slots_inst.map(len) > 0).mean() * 100),
        "pct_responses_with_slot": float((slots_resp.map(len) > 0).mean() * 100),
        "top_bigrams": {"instruction": T.top_ngrams(d["instruction"], 2, 20, stop),
                        "response": T.top_ngrams(d["response"], 2, 20, stop)},
        "top_words": {},
    }
    for col in ("instruction", "response"):
        c = Counter(w for t in d[col] for w in T.words(t) if w not in stop and len(w) > 2)
        stats["top_words"][col] = c.most_common(25)
    if "flags" in d.columns:
        fc = Counter(ch for f in d["flags"].astype(str) for ch in f if ch.isalpha())
        stats["flag_counts"] = dict(fc.most_common())
    if splits:
        stats["splits"] = splits
    return stats


# --------------------------------------------------------------------------
# Paired comparison
# --------------------------------------------------------------------------
def boot_ci(x, n_boot: int = 2000, seed: int = 42, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()),
            "ci_low": float(np.quantile(means, alpha / 2)),
            "ci_high": float(np.quantile(means, 1 - alpha / 2)),
            "n": int(len(x))}


def paired_delta(a, b, n_boot: int = 2000, seed: int = 42):
    """delta = b - a with a paired bootstrap CI (rows where both are valid)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    r = boot_ci(b[ok] - a[ok], n_boot, seed)
    return {"delta": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"]}


def summarise(gen: pd.DataFrame, models=MODELS, n_boot: int = 2000, seed: int = 42):
    """Build results_summary.json content from row-level generations."""
    a, b = models
    metrics = [m for m in METRIC_META if f"{m}_{a}" in gen.columns and f"{m}_{b}" in gen.columns
               and gen[[f"{m}_{a}", f"{m}_{b}"]].notna().any().any()]
    out = {"n_test": int(len(gen)), "models": {a: {"metrics": {}}, b: {"metrics": {}}},
           "tests": {}, "scorecard": {}, "metric_meta": {m: METRIC_META[m] for m in metrics}}
    for m in metrics:
        for mod in models:
            out["models"][mod]["metrics"][m] = boot_ci(gen[f"{m}_{mod}"], n_boot, seed)
        d = paired_delta(gen[f"{m}_{a}"], gen[f"{m}_{b}"], n_boot, seed)
        out["tests"][m] = d
        direction = METRIC_META[m]["direction"]
        if direction is None or d["delta"] is None:
            winner = None
        else:
            sig = d["ci_low"] > 0 or d["ci_high"] < 0
            better_b = (d["delta"] > 0) == (direction == "higher")
            winner = (b if better_b else a) if sig else "tie"
        out["scorecard"][m] = winner
    wins = {mod: sum(1 for w in out["scorecard"].values() if w == mod) for mod in models}
    ties = sum(1 for w in out["scorecard"].values() if w == "tie")
    out["wins"] = {**wins, "ties": ties}
    out["best_model"] = (b if wins[b] > wins[a] else a if wins[a] > wins[b] else "tie")
    # by-category breakdown on the headline metrics
    cats = {}
    for cat, g in gen.groupby("category"):
        row = {"n": int(len(g))}
        for m in ("rougeL", "bertscore_f1", "halluc_proxy"):
            if m in metrics:
                row[m] = {mod: float(g[f"{m}_{mod}"].mean()) for mod in models}
        cats[cat] = row
    out["by_category"] = cats
    return out


# --------------------------------------------------------------------------
# Human ratings (blind A/B sheet) -- never auto-filled
# --------------------------------------------------------------------------
def make_human_sheet(gen: pd.DataFrame, n: int = 40, seed: int = 42, models=MODELS):
    """Blind, shuffled rating sheet + a separate answer key."""
    a, b = models
    s = gen.sample(min(n, len(gen)), random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    flip = rng.random(len(s)) < 0.5
    sheet = pd.DataFrame({
        "id": s["id"], "customer_message": s["instruction"], "reference_answer": s["reference"],
        "answer_A": np.where(flip, s[f"resp_{b}"], s[f"resp_{a}"]),
        "answer_B": np.where(flip, s[f"resp_{a}"], s[f"resp_{b}"]),
    })
    for who in ("A", "B"):
        for crit in ("correctness", "relevance", "helpfulness", "fluency", "hallucination_0or1"):
            sheet[f"{crit}_{who}"] = ""
    key = pd.DataFrame({"id": s["id"], "A_is": np.where(flip, b, a), "B_is": np.where(flip, a, b)})
    return sheet, key


def aggregate_human(rating_files, key: pd.DataFrame):
    """Mean human score per model/criterion from filled sheets (one file per rater)."""
    rows = []
    for f in rating_files:
        r = pd.read_csv(f)
        r = r.merge(key, on="id")
        for who in ("A", "B"):
            for crit in ("correctness", "relevance", "helpfulness", "fluency", "hallucination_0or1"):
                col = f"{crit}_{who}"
                if col in r:
                    tmp = pd.DataFrame({"id": r["id"], "model": r[f"{who}_is"], "criterion": crit,
                                        "score": pd.to_numeric(r[col], errors="coerce"), "rater": str(f)})
                    rows.append(tmp)
    if not rows:
        return {"n_raters": 0}
    long = pd.concat(rows).dropna(subset=["score"])
    res = {"n_raters": int(long["rater"].nunique()), "n_items": int(long["id"].nunique()), "means": {}}
    for (mod, crit), g in long.groupby(["model", "criterion"]):
        res["means"].setdefault(mod, {})[crit] = boot_ci(g["score"].values)
    return res


def to_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
