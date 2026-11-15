"""Stage 1 - Data understanding & EDA (Tasks 1.2, 1.3).  Creates: sampled_data.csv + EDA plots."""
import os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from capstone import config as C
from capstone.kb import load_sops, validate_sops
from capstone.pipeline import common_args, load_raw_bitext

args, P = common_args(__doc__)

# 1.2.1 inspect structure ------------------------------------------------------------------------------------------
sops = load_sops(P.sop_dir); validate_sops(sops)
sop_struct = pd.DataFrame([{"file": k + ".md", "words": len(v.split()), "sections(##)": len(re.findall(r"^## ", v, flags=re.M))}
                           for k, v in sops.items()])
print(sop_struct.to_string(index=False))
df_raw = load_raw_bitext(args)
print("\nBitext rows:", len(df_raw), "| intents:", df_raw["intent"].nunique(), "| categories:", df_raw["category"].nunique())
print("Nulls:\n", df_raw[["instruction", "intent", "category"]].isnull().sum().to_string())

# 1.2.2 sample into the 1,000-5,000 range ---------------------------------------------------------------------------
df = df_raw.sample(min(C.SAMPLE_SIZE, len(df_raw)), random_state=C.SEED).reset_index(drop=True)
df["n_words"] = df["instruction"].astype(str).str.split().str.len()
print(f"\nSampled {len(df)} rows | words/prompt: {df['n_words'].describe().round(1).to_dict()}")

# 1.2.3 sample quality: nulls + normalised dedup + alignment --------------------------------------------------------
norm = df["instruction"].astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
print(f"Exact dupes: {int(df['instruction'].duplicated().sum())} | normalised dupes: {int(norm.duplicated().sum())}")
df["_norm"] = norm
clean = df.dropna(subset=["instruction", "intent", "category"]).drop_duplicates("_norm").drop(columns="_norm")
clean = clean.reset_index(drop=True)
print(f"Rows removed: {len(df) - len(clean)} -> {len(clean)} clean rows")
assert (clean.groupby("intent")["category"].nunique() == 1).all(), "an intent maps to >1 category"
assert 1000 <= len(clean) <= 5000, "Dataset must contain 1,000-5,000 valid records"

# 1.3.1 univariate ---------------------------------------------------------------------------------------------------
counts = clean["intent"].value_counts()
plt.figure(figsize=(9, 4.5))
sns.countplot(data=clean, y="intent", order=counts.index[:10], hue="intent", palette="viridis", legend=False)
plt.title("Top 10 most frequent customer intents"); plt.tight_layout(); plt.savefig(P.f("intent_distribution_top10.png"), dpi=150); plt.close()
print(f"Top intent = {counts.iloc[0] / len(clean) * 100:.1f}% of data | imbalance ratio {counts.max() / counts.min():.2f}")

# 1.3.2 / 1.3.3 SOP TF-IDF similarity ------------------------------------------------------------------------------
names, texts = list(sops), list(sops.values())
sim = cosine_similarity(TfidfVectorizer(stop_words="english").fit_transform(texts))
plt.figure(figsize=(12, 9.5))
sns.heatmap(pd.DataFrame(sim, index=names, columns=names), annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1)
plt.xticks(rotation=60, ha="right"); plt.title("TF-IDF cosine similarity between SOP documents"); plt.tight_layout()
plt.savefig(P.f("sop_similarity_heatmap.png"), dpi=150); plt.close()
off = sim[np.triu_indices_from(sim, k=1)]
print(f"SOP similarity: mean {off.mean():.3f} | max {off.max():.3f} | pairs > 0.5: {(off > 0.5).sum()}")

clean[["instruction", "intent", "category"]].to_csv(P.f("sampled_data.csv"), index=False)
print("Saved sampled_data.csv", clean.shape)
