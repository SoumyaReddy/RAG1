# Hybrid RAG & Fine-Tuning for Customer Support — runnable project

Three systems, measured against each other on one held-out test split:

| System | What it does |
|---|---|
| **Baseline** | Qwen2.5-1.5B-Instruct (4-bit) answers with no context |
| **V1 – Naive RAG** | raw customer query → top-3 SOP chunks (MiniLM + ChromaDB) → answer |
| **V2 – Hybrid RAG** | QLoRA-tuned **intent router** → JSON `{"intent","category"}` → targeted search string → top-3 chunks → answer |

## Run order

```
pip install -r requirements.txt
python run_all.py --dry-run            # 1) CPU-only wiring check, ~20 s, no downloads (NOT real results)
python run_all.py                      # 2) real run: needs a CUDA GPU + Hugging Face access (Colab T4 is enough)
python run_all.py --from 5             #    resume from a stage
python run_all.py --n-eval 100         #    quicker evaluation on 100 test rows
pytest -q                              # offline tests
```

| # | Script | Notebook it mirrors | Needs | Creates |
|---|---|---|---|---|
| 1 | `01_eda.py` | Data_Understanding_and_EDA | Bitext (HF) | `sampled_data.csv`, EDA plots |
| 2 | `02_prepare.py` | Data_Preparation | stage 1 | `df_train/valid/test.csv`, `tokenized_*`, `retrieval_chunks.json` |
| 3 | `03_baseline.py` | Baseline_Model_Evaluation | GPU | `outputs.json` |
| 4 | `04_build_index.py` | RAG_Implementation | stage 2–3 | `chroma_db/`, naive RAG output |
| 5 | `05_eval_v1.py` | Solution_V1_RAG_Evaluation | GPU | `v1_metrics.csv`, `v1_summary.json` |
| 6 | `06_finetune.py` | Fine_Tuning_Pipeline | GPU | `intent_lora_best/`, `training_log.csv`, `training_curves.png` |
| 7 | `07_eval_v2.py` | Solution_V2_FineTuned_RAG_Evaluation | GPU | `Comparative_Results_*.csv`, charts |

All artefacts go to `artifacts/` (`artifacts_dry_run/` for `--dry-run`, so the two never mix).
Each script stops with a clear message if the previous stage's files are missing.

## Layout

```
Dataset/sop_documents/*.md   13 SOP files (knowledge base)
src/capstone/config.py       ONE place for MODEL_ID, EMB_MODEL, K, decoding, prompts, LoRA/training settings
src/capstone/kb.py           SOP chunking, intent→SOP routing table, intent→search strings, SOP-grounded references
src/capstone/metrics.py      hallucination detector, format adherence, ROUGE/BLEU, router JSON parsing, fuzzy match
src/capstone/retrieval.py    embeddings + ChromaDB build/query
src/capstone/llm.py          HFLLM (real 4-bit Qwen + LoRA adapter) and FakeLLM (dry-run only)
src/capstone/pipeline.py     Baseline / Naive RAG / Hybrid RAG functions
scripts/                     the 7 stages        tests/   offline tests        run_all.py   runner
```

## What `--dry-run` is (and is not)
It swaps in synthetic Bitext-shaped data, a scripted stand-in for the LLM and a TF-IDF embedder so the **entire
pipeline, file hand-offs, leakage checks and metric code run on a laptop**. Its numbers mean nothing about model
quality (the synthetic data is template-generated, so the stand-in router scores ~100%). Never report them.
Tokenisation, `HFLLM`, MiniLM and fine-tuning only run in real mode.

## Using your own Bitext copy
If Hugging Face is blocked: download the CSV once elsewhere, then `python run_all.py --bitext-csv bitext.csv`
(columns: `instruction, intent, category`).

## Design rules enforced in code
- Same `MODEL_ID`/embedding/`K`/decoding everywhere (`config.py`); stage 7 asserts the config matches stage 5's.
- Greedy decoding via `do_sample=False, temperature=None` (never `temperature=0.0`).
- Normalised dedup (lower+strip+whitespace, then punctuation-insensitive), conflicting-label rows dropped.
- Stratified 80/10/10 split, near-duplicate (TF-IDF ≥ 0.98) rows removed from train, set-intersection leakage asserts.
- Loss only on the JSON answer (prompt + padding masked to −100); best checkpoint saved by validation loss; early stopping.
- Router and answer synthesis share ONE quantised base model (adapter on = router, off = same base as Baseline/V1).
