"""Shared pipeline pieces used by several stages: CLI, data loading, the three systems (Baseline / V1 / V2)."""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from . import config as C
from .kb import INTENT_SEARCH
from .llm import baseline_messages, rag_messages, router_messages
from .metrics import parse_router


def common_args(desc: str) -> tuple[argparse.Namespace, C.Paths]:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--dry-run", action="store_true",
                    help="CPU-only wiring check: synthetic data + stand-in model + TF-IDF retrieval. Not real results.")
    ap.add_argument("--n-eval", type=int, default=None, help="evaluate only N test rows (default: full test split)")
    ap.add_argument("--bitext-csv", default=None, help="local CSV with instruction,intent,category (skips HF download)")
    args = ap.parse_args()
    paths = C.get_paths(args.dry_run)
    print(f"[{'DRY-RUN' if args.dry_run else 'REAL'}] artefacts -> {paths.root}")
    return args, paths


def require(paths: C.Paths, *names: str):
    missing = [n for n in names if not paths.f(n).exists()]
    if missing:
        sys.exit(f"Missing {missing} in {paths.root}. Run the earlier stage(s) first (see README run order).")


def load_raw_bitext(args) -> pd.DataFrame:
    if args.dry_run:
        from .synthetic import make_synthetic_bitext
        return make_synthetic_bitext(seed=C.SEED)
    if args.bitext_csv:
        return pd.read_csv(args.bitext_csv)
    from datasets import load_dataset
    return load_dataset(C.BITEXT_ID, split="train").to_pandas()


def read_json(paths: C.Paths, name: str) -> dict:
    return json.loads(paths.f(name).read_text())


def write_json(paths: C.Paths, name: str, obj) -> None:
    paths.f(name).write_text(json.dumps(obj, indent=2, default=str))


# ---- the three systems --------------------------------------------------------------------------------------------
def generate_baseline(llm, queries: list[str]) -> list[str]:
    return llm.chat([baseline_messages(q) for q in queries], C.MAX_NEW_TOKENS, desc="baseline")


def generate_naive_rag(llm, retrieve, queries: list[str]):
    """RAW query -> top-K chunks -> prompt augmentation."""
    chunks = [retrieve(q, C.K) for q in queries]
    outs = llm.chat([rag_messages(q, c) for q, c in zip(queries, chunks)], C.MAX_NEW_TOKENS, desc="naive RAG")
    return outs, chunks


def make_zs_router_system(df_train: pd.DataFrame) -> str:
    intent_cat = df_train.drop_duplicates("intent").set_index("intent")["category"].to_dict()
    shots = df_train.sample(4, random_state=C.SEED)
    return (C.ROUTER_SYSTEM + "\nValid intents (intent -> category): "
            + "; ".join(f"{i} -> {c}" for i, c in sorted(intent_cat.items()))
            + "\nExamples:\n" + "\n".join(f"Customer: {t}\nJSON: {j}" for t, j in zip(shots["text"], shots["target_json"])))


def route(llm, queries: list[str], finetuned: bool = True, zs_system: str | None = None) -> list[str]:
    if finetuned:   # trained template -> the exact ROUTER_SYSTEM used when the training data was built
        return llm.chat([router_messages(q) for q in queries], C.ROUTER_MAX_NEW_TOKENS, batch_size=16,
                        use_adapter=True, desc="router (fine-tuned)")
    return llm.chat([router_messages(q, zs_system) for q in queries], C.ROUTER_MAX_NEW_TOKENS, batch_size=8,
                    use_adapter=False, desc="router (zero-shot)")


def to_search_string(parsed, fallback_query: str):
    """Map the extracted JSON intent to a targeted vector-search string; fall back to the raw query if unusable."""
    if parsed and parsed.get("intent") in INTENT_SEARCH:
        return INTENT_SEARCH[parsed["intent"]], True
    return fallback_query, False


def hybrid_rag(llm, retrieve, queries: list[str]) -> pd.DataFrame:
    """Fine-tuned router -> JSON intent -> search string -> retrieval -> synthesis with the SAME base model as V1."""
    raws = route(llm, queries, finetuned=True)
    parsed = [parse_router(r) for r in raws]
    searches = [to_search_string(p[0], q) for p, q in zip(parsed, queries)]
    chunks = [retrieve(s, C.K) for s, _ in searches]
    answers = llm.chat([rag_messages(q, c) for q, c in zip(queries, chunks)], C.MAX_NEW_TOKENS, batch_size=8,
                       use_adapter=False, desc="synthesis")
    return pd.DataFrame({
        "router_raw": raws, "router_json": [p[0] for p in parsed], "router_strict": [p[1] for p in parsed],
        "router_salvaged": [p[2] for p in parsed], "search_string": [s for s, _ in searches],
        "used_intent_search": [u for _, u in searches], "v2_sources": ["|".join(c["source"] for c in cs) for cs in chunks],
        "v2_chunks": chunks, "v2_output": answers})
