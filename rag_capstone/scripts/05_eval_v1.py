"""Stage 3.3/3.4 - Evaluate Baseline vs Solution V1 (naive RAG) on the held-out test split.
Creates: v1_raw_outputs.csv, v1_metrics.csv, v1_summary.json, eval_config.json, v1_vs_baseline.png"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

from capstone import config as C
from capstone.kb import INTENT_SEARCH, INTENT_TO_SOP, load_sops, sop_reference
from capstone.llm import make_llm
from capstone.metrics import check_hallucination, hallu_detail, is_well_formed, pct, score_text
from capstone.pipeline import common_args, generate_baseline, generate_naive_rag, require, write_json
from capstone.retrieval import Retriever, make_embedder

args, P = common_args(__doc__)
require(P, "df_test.csv", "outputs.json", "retrieval_chunks.json")
import json
sops = load_sops(P.sop_dir)
chunks = json.load(open(P.f("retrieval_chunks.json")))
retrieve = Retriever(make_embedder(args.dry_run, [c["text"] for c in chunks]), P.chroma_dir)
llm = make_llm(args.dry_run)

df_test = pd.read_csv(P.f("df_test.csv"))
n_eval = args.n_eval
df_eval = (df_test if n_eval is None else df_test.sample(min(n_eval, len(df_test)), random_state=C.SEED)).reset_index(drop=True)
df_eval["gold_sop"] = df_eval["intent"].map(INTENT_TO_SOP)
print(f"Evaluation rows: {len(df_eval)} (held-out test split)")

# 3.3.1 automated testing
queries = df_eval["text"].tolist()
base_out = generate_baseline(llm, queries)
rag_out, rag_chunks = generate_naive_rag(llm, retrieve, queries)
res = df_eval.rename(columns={"text": "query"}).copy()
res["b_output"], res["v1_output"] = base_out, rag_out
res["v1_sources"] = ["|".join(c["source"] for c in cs) for cs in rag_chunks]
res["v1_hit_at_k"] = [g in {c["source"] for c in cs} for g, cs in zip(res["gold_sop"], rag_chunks)]
res["v1_top1_hit"] = [g == cs[0]["source"] for g, cs in zip(res["gold_sop"], rag_chunks)]
res.to_csv(P.f("v1_raw_outputs.csv"), index=False)

# 3.3.2 format adherence
res["b_format"], res["v1_format"] = res["b_output"].map(is_well_formed), res["v1_output"].map(is_well_formed)

# 3.3.3 ROUGE / BLEU vs SOP-grounded references
res["reference"] = [sop_reference(sops, i, q) for i, q in zip(res["intent"], res["query"])]
b_sc = pd.DataFrame([score_text(h, r) for h, r in zip(res["b_output"], res["reference"])]).add_prefix("b_")
v_sc = pd.DataFrame([score_text(h, r) for h, r in zip(res["v1_output"], res["reference"])]).add_prefix("v1_")
res = pd.concat([res, b_sc, v_sc], axis=1)

# 3.3.4 consistency: 3 repeated deterministic runs
N_CONS = min(20, len(df_eval))
cons_q = df_eval["text"].sample(N_CONS, random_state=C.SEED).tolist()
def consistency(fn):
    runs = [fn(cons_q) for _ in range(3)]
    runs = [r[0] if isinstance(r, tuple) else r for r in runs]
    return float(np.mean([len({runs[k][i] for k in range(3)}) == 1 for i in range(N_CONS)]) * 100)
cons_base = consistency(lambda qs: generate_baseline(llm, qs))
cons_v1 = consistency(lambda qs: generate_naive_rag(llm, retrieve, qs))

# 3.3.5 hallucination frequency
hb = [check_hallucination(o, sops[g], q) for o, g, q in zip(res["b_output"], res["gold_sop"], res["query"])]
hv = [check_hallucination(o, sops[g], q) for o, g, q in zip(res["v1_output"], res["gold_sop"], res["query"])]
res["b_hallu"], res["v1_hallu"] = [h["hallucinated"] for h in hb], [h["hallucinated"] for h in hv]
res["b_hallu_detail"], res["v1_hallu_detail"] = [hallu_detail(h) for h in hb], [hallu_detail(h) for h in hv]

# 3.4.1 compare
S = {"format_adherence_baseline": res["b_format"].mean() * 100, "format_adherence_v1": res["v1_format"].mean() * 100,
     "rouge1_baseline": res["b_rouge1"].mean(), "rouge1_v1": res["v1_rouge1"].mean(),
     "rougeL_baseline": res["b_rougeL"].mean(), "rougeL_v1": res["v1_rougeL"].mean(),
     "bleu_baseline": res["b_bleu"].mean(), "bleu_v1": res["v1_bleu"].mean(),
     "consistency_baseline": cons_base, "consistency_v1": cons_v1,
     "hallucination_baseline": res["b_hallu"].mean() * 100, "hallucination_v1": res["v1_hallu"].mean() * 100,
     "retrieval_hit_at_k_v1": res["v1_hit_at_k"].mean() * 100, "retrieval_top1_v1": res["v1_top1_hit"].mean() * 100,
     "n_eval": len(res)}
rows = [{"Metric": n, "Baseline": S[b], "Solution V1": S[v], "Change %": pct(S[v], S[b]), "Better when": d}
        for n, b, v, d in [("Format adherence (%)", "format_adherence_baseline", "format_adherence_v1", "up"),
                           ("ROUGE-1", "rouge1_baseline", "rouge1_v1", "up"), ("ROUGE-L", "rougeL_baseline", "rougeL_v1", "up"),
                           ("BLEU", "bleu_baseline", "bleu_v1", "up"), ("Consistency (%)", "consistency_baseline", "consistency_v1", "up"),
                           ("Hallucination (%)", "hallucination_baseline", "hallucination_v1", "down")]]
print(pd.DataFrame(rows).round(4).to_string(index=False))
print(f"\nNaive retrieval: gold SOP in top-{C.K}: {S['retrieval_hit_at_k_v1']:.1f}% | top-1: {S['retrieval_top1_v1']:.1f}%")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
for a, (t, k) in zip(ax, [("ROUGE-L", "rougeL"), ("BLEU", "bleu")]):
    a.bar(["Baseline", "V1"], [S[f"{k}_baseline"], S[f"{k}_v1"]]); a.set_title(t)
ax[2].bar(["Baseline", "V1"], [S["hallucination_baseline"], S["hallucination_v1"]]); ax[2].set_title("Hallucination %")
plt.tight_layout(); plt.savefig(P.f("v1_vs_baseline.png"), dpi=150); plt.close()

res.to_csv(P.f("v1_metrics.csv"), index=False)
write_json(P, "v1_summary.json", S)
write_json(P, "eval_config.json", {"model_id": C.MODEL_ID, "embedding_model": retrieve.embedder.name, "k": C.K,
                                   "max_new_tokens": C.MAX_NEW_TOKENS, "gen_kwargs": C.GEN_KW, "n_eval": len(res), "seed": C.SEED,
                                   "baseline_system": C.BASELINE_SYSTEM, "rag_system": C.RAG_SYSTEM})
print("Saved v1_metrics.csv", res.shape)
