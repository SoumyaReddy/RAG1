"""Evaluation metrics: rule-based hallucination detector, format adherence, ROUGE/BLEU, router JSON parsing."""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher

import numpy as np

# ---- Hallucination detector -------------------------------------------------------------------------------------
# Flagged when an output contains (a) a numeric claim with a unit (days/hours/%/$...) NOT in the supporting SOP text
# or the customer query, (b) a URL/e-mail/phone number not in the SOP, or (c) a promise the SOPs forbid.
_WORD2NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "fourteen": 14, "fifteen": 15, "twenty": 20, "thirty": 30,
             "forty-eight": 48, "sixty": 60, "seventy-two": 72, "ninety": 90}
_WORDS_RE = "|".join(sorted(map(re.escape, _WORD2NUM), key=len, reverse=True))
_NUM = rf"(?:\d+(?:\.\d+)?|\b(?:{_WORDS_RE})\b)"
_UNIT = (r"(?:%|percent\b|(?:business |calendar |working )?(?:days?|hours?|hrs?|minutes?|mins?|weeks?|months?|years?)\b"
         r"|dollars?\b|usd\b)")
_CLAIM_RE = re.compile(rf"(?:\$\s?\d+(?:\.\d+)?|{_NUM}(?:\s*(?:-|–|—|to|or)\s*{_NUM})?\s*-?\s*{_UNIT})", re.I)
_CONTACT_RE = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b|\b1-800[-\d]*", re.I)
_FORBIDDEN = [re.compile(p, re.I) for p in [
    r"\b(?:we|i)(?:'ll| will)? guarantee\b",
    r"\bguaranteed (?:delivery|refund|date)\b",
    r"\b(?:we|i)(?:'ll| will| can)? (?:offer|provide|issue|give)[^.]{0,30}\bcash\b",
]]


def numbers_in(text: str) -> set[float]:
    t = re.sub(r"^\s*\d+[.)]\s+", "", text.lower(), flags=re.M)      # ignore list markers such as "1." / "2)"
    nums = {float(m.group()) for m in re.finditer(r"\d+(?:\.\d+)?", t)}
    for w, v in _WORD2NUM.items():
        if re.search(rf"\b{re.escape(w)}\b", t):
            nums.add(float(v))
    return nums


def check_hallucination(output: str, support_text: str, query_text: str = "") -> dict:
    support_nums = numbers_in(support_text) | numbers_in(query_text)
    ok_text = (support_text + " " + query_text).lower()
    bad_nums = [m.group().strip() for m in _CLAIM_RE.finditer(output)
                if any(n not in support_nums for n in numbers_in(m.group()))]
    bad_contacts = [m.group() for m in _CONTACT_RE.finditer(output) if m.group().lower() not in ok_text]
    violations = [p.pattern for p in _FORBIDDEN if p.search(output)]
    return {"hallucinated": bool(bad_nums or bad_contacts or violations),
            "unsupported_numbers": bad_nums, "unsupported_contacts": bad_contacts, "violations": violations}


def hallu_detail(h: dict) -> str:
    return str(h["unsupported_numbers"] + h["unsupported_contacts"] + h["violations"])


def fact_coverage(output: str, key_fact_patterns: list[str]) -> float:
    """Functional-correctness proxy: share of required SOP facts (regex patterns) present in the output."""
    return float(np.mean([bool(re.search(p, output, re.I)) for p in key_fact_patterns]))


def is_well_formed(text: str) -> bool:
    """Format adherence for free-text answers: non-empty, >= 8 words, no leaked chat tokens, not a raw JSON blob."""
    t = (text or "").strip()
    return bool(len(t.split()) >= 8 and "<|im_" not in t and not t.startswith("{"))


# ---- ROUGE / BLEU --------------------------------------------------------------------------------------------------
_rouge = None
_smooth = None


def score_text(hyp: str, ref: str) -> dict:
    global _rouge, _smooth
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from rouge_score import rouge_scorer
    if _rouge is None:
        _rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
        _smooth = SmoothingFunction().method1
    r = _rouge.score(ref, hyp)
    toks = lambda s: re.findall(r"\w+", s.lower())
    bleu = sentence_bleu([toks(ref)], toks(hyp), smoothing_function=_smooth) if toks(hyp) else 0.0
    return {"rouge1": r["rouge1"].fmeasure, "rougeL": r["rougeL"].fmeasure, "bleu": bleu}


# ---- Router JSON ---------------------------------------------------------------------------------------------------
def parse_router(raw: str):
    """Returns (parsed_dict_or_None, strict_valid, salvaged). strict = WHOLE output is valid JSON with both keys."""
    try:
        obj = json.loads(raw.strip())
        if isinstance(obj, dict) and {"intent", "category"} <= set(obj):
            return obj, True, False
    except Exception:
        pass
    m = re.search(r"\{.*?\}", raw, re.S)                      # safety net for stray preamble / trailing text
    if m:
        try:
            obj = json.loads(m.group())
            if isinstance(obj, dict) and "intent" in obj:
                return obj, False, True
        except Exception:
            pass
    return None, False, False


def fuzzy_match(a, b, thr: float = 0.80) -> bool:
    """Tolerates word-order differences, e.g. 'order_tracking' ~ 'track_order'."""
    norm = lambda s: " ".join(sorted(str(s).lower().replace("-", "_").split("_")))
    return SequenceMatcher(None, norm(a), norm(b)).ratio() >= thr


ADV_RE = re.compile(r"\b(still|never|terrible|frustrat\w*|ridiculous|awful|worst|useless|unacceptable|angry|furious|"
                    r"horrible|disappointed|nobody|again)\b", re.I)


def pct(new, old) -> float:
    return float("nan") if old == 0 or old != old else (new - old) / old * 100
