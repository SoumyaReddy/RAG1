"""Stage 3.2 - Embeddings, Chroma index, naive retrieval workflow.  Creates: chroma_db/, sop_embeddings.npy, updates outputs.json"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np, pandas as pd
from capstone import config as C
from capstone.kb import INTENT_TO_SOP, load_sops
from capstone.llm import make_llm, rag_messages
from capstone.metrics import check_hallucination
from capstone.pipeline import common_args, read_json, require, write_json
from capstone.retrieval import Retriever, build_index, make_embedder

args, P = common_args(__doc__)
require(P, "retrieval_chunks.json", "outputs.json")
chunks = json.load(open(P.f("retrieval_chunks.json")))
sops = load_sops(P.sop_dir)

# 3.2.1 embeddings + validation
embedder = make_embedder(args.dry_run, [c["text"] for c in chunks])
col, emb = build_index(chunks, embedder, P.chroma_dir)          # 3.2.2 index (asserts shape / NaN / count)
assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3), "embeddings should be unit-norm"
np.save(P.f("sop_embeddings.npy"), emb)
print(f"Embeddings {emb.shape} via {embedder.name} | Chroma '{C.COLLECTION}': {col.count()} vectors in {P.chroma_dir}")

retrieve = Retriever(embedder, P.chroma_dir)                      # reopen from disk = proves persistence
hits = sum(retrieve.collection.query(query_embeddings=[emb[i].tolist()], n_results=1)["ids"][0][0] == c["chunk_id"]
           for i, c in enumerate(chunks))
print(f"Self-retrieval accuracy: {hits}/{len(chunks)}")
assert hits == len(chunks), "index is inconsistent with the embeddings"
for q in ["I want my money back", "forgot my password", "where is my parcel"]:
    top = retrieve(q, 1)[0]; print(f"  '{q}' -> {top['source']} / {top['section']}")

# 3.2.3 naive RAG on the demo query
outputs = read_json(P, "outputs.json")
q = outputs["test_query"]
retrieved = retrieve(q, C.K)
print(f"\nTop-{C.K} for the raw query:", [(c["source"], c["section"], round(c["distance"], 3)) for c in retrieved])
gold = "shipping_delays"
print("Gold SOP retrieved:", gold in [c["source"] for c in retrieved])
llm = make_llm(args.dry_run)
naive = llm.chat([rag_messages(q, retrieved)], C.MAX_NEW_TOKENS, batch_size=1, desc="naive RAG")[0]
print("\nBaseline :", outputs["baseline_output"], "\nNaive RAG:", naive)
print("Hallucination check:", check_hallucination(naive, sops[gold], q))

# optional: retrieval-only quality on the whole test split (no LLM needed)
if P.f("df_test.csv").exists():
    dft = pd.read_csv(P.f("df_test.csv"))
    hk = [INTENT_TO_SOP[i] in {c["source"] for c in retrieve(t, C.K)} for t, i in zip(dft["text"], dft["intent"])]
    h1 = [INTENT_TO_SOP[i] == retrieve(t, 1)[0]["source"] for t, i in zip(dft["text"], dft["intent"])]
    print(f"Naive retrieval on test split: gold SOP in top-{C.K}: {np.mean(hk) * 100:.1f}% | top-1: {np.mean(h1) * 100:.1f}%")

outputs["naive_rag_output"] = naive
outputs["naive_rag_retrieved"] = [{k: c[k] for k in ("source", "section", "distance")} for c in retrieved]
outputs["retrieval_config"] = {"embedding_model": embedder.name, "k": C.K, "collection": C.COLLECTION, "chunks": len(chunks)}
write_json(P, "outputs.json", outputs)
print("outputs.json updated")
