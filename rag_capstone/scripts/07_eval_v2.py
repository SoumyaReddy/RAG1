"""Stage 4.3-4.5 - Hybrid RAG (fine-tuned router + retrieval) evaluation and 3-way comparison.
Creates: v2_raw_outputs.csv, Comparative_Results_Full.csv, Comparative_Results_Summary.csv, router_eval_summary.csv,
         fine_tuning_impact.csv, comparison_chart.png"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

from capstone import config as C
from capstone.kb import INTENT_TO_SOP, load_sops, sop_reference
from capstone.llm import make_llm
from capstone.metrics import (ADV_RE, check_hallucination, fuzzy_match, hallu_detail, is_well_formed, parse_router, pct,
                              score_text)
from capstone.pipeline import common_args, hybrid_rag, make_zs_router_system, read_json, require, route, write_json
from capstone.retrieval import Retriever, make_embedder

args, P = common_args(__doc__)
require(P, "v1_metrics.csv", "v1_summary.json", "eval_config.json", "df_train.csv", "df_test.csv", "outputs.json", "retrieval_chunks.json")
if not args.dry_run:
    require(P, "intent_lora_best")
sops = load_sops(P.sop_dir)
chunks = json.load(open(P.f("retrieval_chunks.json")))
retrieve = Retriever(make_embedder(args.dry_run, [c["text"] for c in chunks]), P.chroma_dir)
df_train, df_test = pd.read_csv(P.f("df_train.csv")), pd.read_csv(P.f("df_test.csv"))
llm = make_llm(args.dry_run, adapter_dir=None if args.dry_run else P.lora_best, train_df=df_train)

# --- evaluation rows = EXACTLY the rows scored in Stage 5, same config (guards against config drift) ---
v1 = pd.read_csv(P.f("v1_metrics.csv")); cfg = read_json(P, "eval_config.json"); v1s = read_json(P, "v1_summary.json")
assert cfg["model_id"] == C.MODEL_ID and cfg["k"] == C.K and cfg["max_new_tokens"] == C.MAX_NEW_TOKENS, "config drift between stages!"
df_eval = df_test[df_test["row_id"].isin(v1["row_id"])].set_index("row_id").loc[v1["row_id"]].reset_index()
assert set(df_eval["row_id"]).isdisjoint(set(df_train["row_id"])), "LEAKAGE: evaluation rows found in training data"
assert not (set(df_eval["text"].str.lower().str.strip()) & set(df_train["text"].str.lower().str.strip())), "LEAKAGE: duplicate prompt"
print(f"Evaluating {len(df_eval)} held-out rows (zero leakage verified vs df_train)")
df_eval["adversarial"] = df_eval["text"].map(lambda s: bool(ADV_RE.search(s)))
print("Adversarial subset:", int(df_eval["adversarial"].sum()), f"({df_eval['adversarial'].mean() * 100:.1f}%)")

# 4.3 integrated pipeline demo
outputs = read_json(P, "outputs.json")
demo = hybrid_rag(llm, retrieve, [outputs["test_query"]]).iloc[0]
print("\n1) query:", outputs["test_query"], "\n2) router:", demo["router_raw"], "| valid JSON:", demo["router_strict"],
      "\n3) search:", demo["search_string"], "\n4) SOPs:", [(c["source"], c["section"]) for c in demo["v2_chunks"]],
      "\n5) answer:", demo["v2_output"])
outputs.update(hybrid_rag_router_json=demo["router_raw"], hybrid_rag_search_string=demo["search_string"], hybrid_rag_output=demo["v2_output"])
write_json(P, "outputs.json", outputs)

# 4.4.1 evaluation
v2 = hybrid_rag(llm, retrieve, df_eval["text"].tolist())
v2 = pd.concat([df_eval[["row_id", "text", "intent", "category", "adversarial"]].reset_index(drop=True), v2], axis=1)
v2["pred_intent"] = [j.get("intent") if isinstance(j, dict) else None for j in v2["router_json"]]
v2["pred_category"] = [j.get("category") if isinstance(j, dict) else None for j in v2["router_json"]]
v2["exact_match"] = v2["pred_intent"] == v2["intent"]
v2["fuzzy_match"] = [p is not None and fuzzy_match(p, g) for p, g in zip(v2["pred_intent"], v2["intent"])]
v2["category_match"] = v2["pred_category"] == v2["category"]

def router_report(d, name):
    return {"subset": name, "n": len(d), "format_adherence_%": d["router_strict"].mean() * 100,
            "format_incl_salvage_%": (d["router_strict"] | d["router_salvaged"]).mean() * 100,
            "intent_exact_%": d["exact_match"].mean() * 100, "intent_fuzzy_%": d["fuzzy_match"].mean() * 100,
            "category_acc_%": d["category_match"].mean() * 100}
rep = pd.DataFrame([router_report(v2, "full test split"), router_report(v2[v2["adversarial"]], "adversarial subset")]).round(1)
print("\nFINE-TUNED ROUTER\n", rep.to_string(index=False))

zs_raw = route(llm, df_eval["text"].tolist(), finetuned=False, zs_system=make_zs_router_system(df_train))
zs = [parse_router(r) for r in zs_raw]
zs_fmt = float(np.mean([z[1] for z in zs]) * 100)
zs_exact = float(np.mean([(z[0] or {}).get("intent") == g for z, g in zip(zs, df_eval["intent"])]) * 100)
print(f"Base model as few-shot router: format {zs_fmt:.1f}% | intent exact {zs_exact:.1f}%")

v2["gold_sop"] = v2["intent"].map(INTENT_TO_SOP)
v2["reference"] = [sop_reference(sops, i, q) for i, q in zip(v2["intent"], v2["text"])]
v2 = pd.concat([v2, pd.DataFrame([score_text(h, r) for h, r in zip(v2["v2_output"], v2["reference"])]).add_prefix("v2_")], axis=1)
v2["v2_format"] = v2["v2_output"].map(is_well_formed)
v2["v2_hit_at_k"] = [g in s.split("|") for g, s in zip(v2["gold_sop"], v2["v2_sources"])]
v2["v2_top1_hit"] = [g == s.split("|")[0] for g, s in zip(v2["gold_sop"], v2["v2_sources"])]
v2.drop(columns=["v2_chunks"]).to_csv(P.f("v2_raw_outputs.csv"), index=False)

# 4.4.2 fine-tuning impact (V1 -> V2)
m = v1.merge(v2[["row_id", "adversarial", "v2_rouge1", "v2_rougeL", "v2_bleu", "v2_hit_at_k", "v2_top1_hit"]], on="row_id")
rows = []
for name, c1, c2 in [("ROUGE-1", "v1_rouge1", "v2_rouge1"), ("ROUGE-L", "v1_rougeL", "v2_rougeL"), ("BLEU", "v1_bleu", "v2_bleu"),
                     ("Retrieval hit@K (%)", "v1_hit_at_k", "v2_hit_at_k"), ("Retrieval top-1 (%)", "v1_top1_hit", "v2_top1_hit")]:
    mult = 100 if "%" in name else 1
    for subset, d in [("full test", m), ("adversarial", m[m["adversarial"]])]:
        a, b = d[c1].mean() * mult, d[c2].mean() * mult
        rows.append({"Metric": name, "Subset": subset, "V1": a, "V2": b, "Abs. change": b - a, "Change %": pct(b, a)})
impact = pd.DataFrame(rows).round(4); print("\nFINE-TUNING IMPACT (V1 -> V2)\n", impact.to_string(index=False))

# 4.5 three-way comparison
full = v1.merge(v2[["row_id", "adversarial", "router_raw", "router_strict", "pred_intent", "exact_match", "fuzzy_match", "category_match",
                    "search_string", "v2_sources", "v2_output", "v2_format", "v2_rouge1", "v2_rougeL", "v2_bleu", "v2_hit_at_k",
                    "v2_top1_hit"]], on="row_id")
hv2 = [check_hallucination(o, sops[g], q) for o, g, q in zip(full["v2_output"], full["gold_sop"], full["query"])]
full["v2_hallu"], full["v2_hallu_detail"] = [h["hallucinated"] for h in hv2], [hallu_detail(h) for h in hv2]
cons_q = df_eval["text"].sample(min(20, len(df_eval)), random_state=C.SEED).tolist()
runs = [hybrid_rag(llm, retrieve, cons_q) for _ in range(3)]
cons_v2 = float(np.mean([len({(runs[k].loc[i, "router_raw"], runs[k].loc[i, "v2_output"]) for k in range(3)}) == 1
                         for i in range(len(cons_q))]) * 100)
S = lambda col, mult=1: full[col].mean() * mult
nan = np.nan
summary = pd.DataFrame([
    ["Format adherence - answers (%)", S("b_format", 100), S("v1_format", 100), S("v2_format", 100)],
    ["Router JSON validity (%)", nan, nan, S("router_strict", 100)],
    ["Intent accuracy - exact match (%)", nan, nan, S("exact_match", 100)],
    ["Intent accuracy - fuzzy match (%)", nan, nan, S("fuzzy_match", 100)],
    ["Intent accuracy - adversarial subset (%)", nan, nan, full[full["adversarial"]]["exact_match"].mean() * 100],
    ["ROUGE-1", S("b_rouge1"), S("v1_rouge1"), S("v2_rouge1")], ["ROUGE-L", S("b_rougeL"), S("v1_rougeL"), S("v2_rougeL")],
    ["BLEU", S("b_bleu"), S("v1_bleu"), S("v2_bleu")],
    ["Consistency rate (%)", v1s["consistency_baseline"], v1s["consistency_v1"], cons_v2],
    ["Hallucination frequency (%)", S("b_hallu", 100), S("v1_hallu", 100), S("v2_hallu", 100)],
    ["Retrieval hit@K (%)", nan, S("v1_hit_at_k", 100), S("v2_hit_at_k", 100)],
    ["Retrieval top-1 (%)", nan, S("v1_top1_hit", 100), S("v2_top1_hit", 100)],
], columns=["Metric", "Baseline", "Solution_V1", "Solution_V2"])
summary["V1_vs_Baseline_%"] = [pct(b, a) for a, b in zip(summary["Baseline"], summary["Solution_V1"])]
summary["V2_vs_V1_%"] = [pct(b, a) for a, b in zip(summary["Solution_V1"], summary["Solution_V2"])]
summary["V2_vs_Baseline_%"] = [pct(b, a) for a, b in zip(summary["Baseline"], summary["Solution_V2"])]
summary = summary.round(4); print("\nTHREE-WAY COMPARISON\n", summary.to_string(index=False))

sy = summary.set_index("Metric"); cols3 = ["Baseline", "Solution_V1", "Solution_V2"]
fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
for a, met in zip(ax[:3], ["ROUGE-1", "ROUGE-L", "BLEU"]):
    a.bar(["Baseline", "V1", "V2"], sy.loc[met, cols3]); a.set_title(met)
ax[3].bar(["Baseline", "V1", "V2"], sy.loc["Hallucination frequency (%)", cols3]); ax[3].set_title("Hallucination (%)")
plt.tight_layout(); plt.savefig(P.f("comparison_chart.png"), dpi=150); plt.close()

full.to_csv(P.f("Comparative_Results_Full.csv"), index=False)
summary.to_csv(P.f("Comparative_Results_Summary.csv"), index=False)
rep.to_csv(P.f("router_eval_summary.csv"), index=False); impact.to_csv(P.f("fine_tuning_impact.csv"), index=False)
write_json(P, "v2_extra.json", {"zero_shot_router_format_%": zs_fmt, "zero_shot_router_intent_exact_%": zs_exact,
                                "n_eval": len(full), "adversarial_n": int(full["adversarial"].sum())})
print("\nSaved Comparative_Results_Full.csv / Comparative_Results_Summary.csv and supporting files")
