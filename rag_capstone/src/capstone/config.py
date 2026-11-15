"""Single source of truth for every setting shared across the pipeline.

The workflow requires IDENTICAL model / embedding / K / decoding settings in every stage,
so they live here once instead of being copy-pasted into each script.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BITEXT_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
SEED = 42
COLLECTION = "sop_chunks"
K = 3                       # chunks retrieved per query (Naive RAG and Hybrid RAG)
MAX_NEW_TOKENS = 150        # answer length (Baseline, V1, V2)
ROUTER_MAX_NEW_TOKENS = 40  # JSON router output length
SAMPLE_SIZE = 4000          # must stay within 1,000-5,000 after cleaning
MAX_LEN = 192               # tokenisation length for fine-tuning
SPLIT = (0.80, 0.10, 0.10)

# do_sample=False with temperature/top_p/top_k=None (NOT temperature=0.0) -> pure greedy decoding.
GEN_KW = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)

BASELINE_SYSTEM = ("You are a helpful customer support assistant for an online retail company. "
                   "Answer the customer's message concisely (maximum 4 sentences).")
RAG_SYSTEM = ("You are a customer support assistant. Answer strictly using the SOP context below. "
              "Do not invent policies, timelines, numbers, links or contact details. "
              "If the SOP does not cover the question, say you will escalate to a human agent. "
              "Be concise (maximum 4 sentences).\n\nSOP context:\n{context}")
# Router prompt: used for fine-tuning data AND inference - must match exactly.
ROUTER_SYSTEM = ('You are an intent router for a customer support system. Read the customer message and reply '
                 'with ONLY a JSON object of the form {"intent": "<snake_case_intent>", "category": "<UPPERCASE_CATEGORY>"}. '
                 'Do not add any explanation.')

# Fine-tuning hyper-parameters (QLoRA)
LORA_CFG = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"],
                bias="none", task_type="CAUSAL_LM")
TRAIN = dict(lr=2e-4, batch_size=4, eval_batch_size=16, max_steps=600, warmup_steps=30,
             val_every=25, patience=3, min_delta=1e-4, weight_decay=0.01, grad_clip=1.0)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """All artefact locations. Dry-run writes to a separate folder so it can never pollute real results."""
    root: Path

    @property
    def sop_dir(self) -> Path:
        env = os.environ.get("SOP_DIR")
        cands = [Path(env)] if env else []
        cands += [PROJECT_ROOT / "Dataset" / "sop_documents", PROJECT_ROOT / "sop_documents",
                  PROJECT_ROOT / "corporate_policies", Path("Dataset/sop_documents")]
        for c in cands:
            if c.is_dir() and list(c.glob("*.md")):
                return c
        raise FileNotFoundError("SOP folder not found. Put the *.md files in Dataset/sop_documents or set SOP_DIR.")

    def f(self, name: str) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / name

    @property
    def chroma_dir(self) -> str:
        return str(self.f("chroma_db"))

    @property
    def lora_best(self) -> str:
        return str(self.f("intent_lora_best"))

    @property
    def lora_final(self) -> str:
        return str(self.f("intent_lora"))


def get_paths(dry_run: bool = False) -> Paths:
    base = os.environ.get("CAPSTONE_OUT")
    root = Path(base) if base else PROJECT_ROOT / ("artifacts_dry_run" if dry_run else "artifacts")
    return Paths(root)
