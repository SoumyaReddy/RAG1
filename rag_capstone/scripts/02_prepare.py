"""Stage 2 - Data preparation (Tasks 2.1-2.4).
Creates: df_train/valid/test.csv, retrieval_chunks.json, tokenized_{train,valid,test}/ (real mode), data_prep_config.json"""
import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np, pandas as pd, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from capstone import config as C
from capstone.kb import build_chunks
from capstone.pipeline import common_args, require

args, P = common_args(__doc__)
require(P, "sampled_data.csv")
df = pd.read_csv(P.f("sampled_data.csv"))
audit = {"start_rows": len(df)}

# 2.1.1 remove incomplete / invalid / duplicate records -------------------------------------------------------------
df = df.dropna(subset=["instruction", "intent", "category"]); audit["incomplete"] = audit["start_rows"] - len(df)
for c in ["instruction", "intent", "category"]:
    df[c] = df[c].astype(str)
n = len(df)
bad = (df["instruction"].str.strip().str.len() < 3) | df["intent"].str.strip().eq("") | df["category"].str.strip().eq("")
bad |= df["instruction"].str.contains("�|Ã|â€", regex=True)
bad |= df["instruction"].str.contains(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", regex=True)
df = df[~bad]; audit["invalid/corrupted"] = n - len(df)
n = len(df)
df["_norm"] = df["instruction"].str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
conflict = df.groupby("_norm")["intent"].transform("nunique") > 1       # same prompt, different intent = label noise
audit["conflicting-label dupes"] = int(conflict.sum()); df = df[~conflict]
df = df.drop_duplicates("_norm"); audit["normalised dupes removed"] = n - len(df) - audit["conflicting-label dupes"]
n = len(df)
df["_n2"] = df["_norm"].str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
df = df.drop_duplicates("_n2"); audit["punctuation near-dupes removed"] = n - len(df)
df = df.drop(columns=["_norm", "_n2"]).reset_index(drop=True); audit["final_rows"] = len(df)
print(pd.Series(audit, name="rows").to_string())
assert 1000 <= len(df) <= 5000

# 2.1.2 standardise (strip BEFORE casing) ---------------------------------------------------------------------------
df["instruction"] = df["instruction"].str.strip().str.replace(r"\s+", " ", regex=True)
df["intent"] = df["intent"].str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
df["category"] = df["category"].str.strip().str.upper()
df = df.rename(columns={"instruction": "text"})[["text", "intent", "category"]].reset_index(drop=True)
df.insert(0, "row_id", range(len(df)))
assert df["intent"].str.fullmatch(r"[a-z0-9_]+").all() and df["category"].str.fullmatch(r"[A-Z0-9_]+").all()
assert (df.groupby("intent")["category"].nunique() == 1).all()
print("Schema:", list(df.columns), "| rows:", len(df), "| intents:", df["intent"].nunique())

# 2.2 ChatML with strict JSON targets (identical to Qwen's chat template) -------------------------------------------
def chatml_prompt(text):
    return (f"<|im_start|>system\n{C.ROUTER_SYSTEM}<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")

df["target_json"] = [json.dumps({"intent": i, "category": c}, ensure_ascii=False) for i, c in zip(df["intent"], df["category"])]
df["prompt_text"] = [chatml_prompt(t) for t in df["text"]]
df["chatml"] = [p + tj + "<|im_end|>" for p, tj in zip(df["prompt_text"], df["target_json"])]
fails = []
for i, r in df.iterrows():
    try:
        assert r["chatml"].count("<|im_start|>") == 3
        obj = json.loads(r["chatml"].split("<|im_start|>assistant\n")[1].replace("<|im_end|>", "").strip())
        assert set(obj) == {"intent", "category"} and obj["intent"] == r["intent"] and obj["category"] == r["category"]
    except Exception as e:
        fails.append((i, repr(e)))
assert not fails, fails[:5]
print(f"Validated {len(df)} ChatML templates -> 0 failures")

# 2.4 stratified split + leakage control ------------------------------------------------------------------------------
meta = df[["row_id", "text", "intent", "category", "target_json"]].copy()
train_df, temp_df = train_test_split(meta, test_size=0.20, stratify=meta["intent"], random_state=C.SEED)
valid_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["intent"], random_state=C.SEED)
vec = TfidfVectorizer(ngram_range=(1, 2)).fit(meta["text"].str.lower())
held = pd.concat([valid_df, test_df])
sim = cosine_similarity(vec.transform(train_df["text"].str.lower()), vec.transform(held["text"].str.lower()))
leaky = sim.max(axis=1) >= 0.98
print(f"Train rows near-identical (cos>=0.98) to a val/test prompt, removed from train: {int(leaky.sum())}")
train_df = train_df[~leaky]

parts = {"train": train_df, "valid": valid_df, "test": test_df}
ids = {n: set(d["row_id"]) for n, d in parts.items()}
txt = {n: set(d["text"].str.lower().str.strip()) for n, d in parts.items()}
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    assert not (ids[a] & ids[b]), f"row_id leakage {a}/{b}"
    assert not (txt[a] & txt[b]), f"duplicate prompt across {a}/{b}"
    print(f"{a:5s} ∩ {b:5s}: row_ids=0  duplicate prompts=0")
for n, d in parts.items():
    assert set(d["intent"]) == set(meta["intent"]), f"{n} split is missing intents"
print("All", meta["intent"].nunique(), "intents in every split | sizes:", {n: len(d) for n, d in parts.items()})
for n, d in parts.items():
    d.reset_index(drop=True).to_csv(P.f(f"df_{n}.csv"), index=False)

# 2.3.3 retrieval corpus ------------------------------------------------------------------------------------------------
chunks = build_chunks(P.sop_dir)
cdf = pd.DataFrame(chunks); cdf["chars"] = cdf["text"].str.len()
assert cdf["chars"].min() > 40 and cdf["chunk_id"].is_unique and not cdf["text"].duplicated().any()
json.dump(chunks, open(P.f("retrieval_chunks.json"), "w"), indent=1)
print(f"{cdf['source'].nunique()} SOPs -> {len(chunks)} chunks (mean {cdf['chars'].mean():.0f} chars)")

# 2.3.1 / 2.3.2 tokenisation + label masking (needs the Qwen tokenizer -> real mode only) ---------------------------
if args.dry_run:
    print("[dry-run] skipping tokenisation (needs Hugging Face tokenizer)")
else:
    from datasets import Dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # make sure our hand-built ChatML == the tokenizer's own template (guards against template drift)
    ref = tok.apply_chat_template([{"role": "system", "content": C.ROUTER_SYSTEM}, {"role": "user", "content": df["text"][0]}],
                                  tokenize=False, add_generation_prompt=True)
    assert ref == df["prompt_text"][0], "hand-built ChatML differs from tokenizer chat template"
    P_IDS = [tok(p, add_special_tokens=False)["input_ids"] for p in df["prompt_text"]]
    A_IDS = [tok(t + "<|im_end|>", add_special_tokens=False)["input_ids"] for t in df["target_json"]]
    full_len = np.array([len(p) + len(a) for p, a in zip(P_IDS, A_IDS)])
    trunc = (full_len > C.MAX_LEN).mean() * 100
    print(f"Token length: mean {full_len.mean():.1f} | p99 {np.percentile(full_len, 99):.0f} | max {full_len.max()} | "
          f"> MAX_LEN={C.MAX_LEN}: {trunc:.2f}%")
    assert trunc < 5, "increase MAX_LEN"
    plt.figure(figsize=(7, 3.5)); plt.hist(full_len, bins=40); plt.axvline(C.MAX_LEN, color="red", ls="--")
    plt.title("Full sequence length (tokens)"); plt.tight_layout(); plt.savefig(P.f("token_length_distribution.png"), dpi=150); plt.close()
    cols = {"row_id": [], "input_ids": [], "attention_mask": [], "labels": []}
    for rid, p, a in zip(df["row_id"], P_IDS, A_IDS):
        ids_ = (p + a)[:C.MAX_LEN]; lab = ([-100] * len(p) + a)[:C.MAX_LEN]; pad = C.MAX_LEN - len(ids_)
        cols["row_id"].append(int(rid)); cols["attention_mask"].append([1] * len(ids_) + [0] * pad)
        cols["input_ids"].append(ids_ + [tok.pad_token_id] * pad); cols["labels"].append(lab + [-100] * pad)
    tok_ds = Dataset.from_dict(cols)
    for k in np.random.RandomState(C.SEED).choice(len(df), 50, replace=False):       # verify masking
        ex = tok_ds[int(k)]
        sup = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
        assert tok.decode(sup) == df["target_json"][int(k)] + "<|im_end|>", "label masking is wrong"
    rid2pos = {r: i for i, r in enumerate(tok_ds["row_id"])}
    for n, d in parts.items():
        out = str(P.f(f"tokenized_{n}")); shutil.rmtree(out, ignore_errors=True)
        tok_ds.select([rid2pos[r] for r in d["row_id"]]).save_to_disk(out)
    print("Saved tokenized_train/valid/test")

json.dump({"model_id": C.MODEL_ID, "max_len": C.MAX_LEN, "seed": C.SEED, "split": C.SPLIT,
           "rows": {n: len(d) for n, d in parts.items()}}, open(P.f("data_prep_config.json"), "w"), indent=1)
