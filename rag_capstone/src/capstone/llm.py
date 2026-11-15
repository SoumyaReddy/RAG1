"""LLM access. One interface, two backends:

* HFLLM   - Qwen2.5-1.5B-Instruct in 4-bit NF4 (+ optional LoRA router adapter). Needs a GPU.
* FakeLLM - deterministic stand-in for --dry-run: lets the WHOLE pipeline run on CPU so the wiring, metrics and
            file hand-offs can be verified. Its outputs are NOT real model results.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import re

from . import config as C


def baseline_messages(q: str) -> list[dict]:
    return [{"role": "system", "content": C.BASELINE_SYSTEM}, {"role": "user", "content": q}]


def build_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['source']} / {c['section']}]\n{c['text']}" for c in chunks)


def rag_messages(q: str, chunks: list[dict]) -> list[dict]:
    return [{"role": "system", "content": C.RAG_SYSTEM.format(context=build_context(chunks))},
            {"role": "user", "content": q}]


def router_messages(q: str, system: str = C.ROUTER_SYSTEM) -> list[dict]:
    return [{"role": "system", "content": system}, {"role": "user", "content": q}]


class HFLLM:
    def __init__(self, adapter_dir: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        assert torch.cuda.is_available(), "A CUDA GPU is required (Colab: Runtime > Change runtime type > T4 GPU)."
        torch.manual_seed(C.SEED)
        self.torch = torch
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.tok = AutoTokenizer.from_pretrained(C.MODEL_ID, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(C.MODEL_ID, quantization_config=bnb, device_map="auto")
        self.has_adapter = adapter_dir is not None
        if adapter_dir:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_dir, adapter_name="router")
        self.model.eval()

    def chat(self, messages_list, max_new_tokens, batch_size=8, use_adapter=False, desc="generating"):
        """Greedy, batched, length-sorted generation. Returns completions in the ORIGINAL order.
        use_adapter=True -> LoRA router ON; False -> the plain base model (adapter disabled)."""
        from tqdm.auto import tqdm
        prompts = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list]
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs = [None] * len(prompts)
        ctx = contextlib.nullcontext() if (use_adapter or not self.has_adapter) else self.model.disable_adapter()
        with ctx, self.torch.no_grad():
            for s in tqdm(range(0, len(order), batch_size), desc=desc):
                idx = order[s:s + batch_size]
                enc = self.tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(self.model.device)
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id,
                                          **C.GEN_KW)
                texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i, t in zip(idx, texts):
                    outs[i] = t.strip()
        return outs


class FakeLLM:
    """Deterministic stand-in. Behaviour depends on which system prompt it receives:

    baseline  -> generic advice with INVENTED numbers (so the hallucination detector has something to catch)
    rag       -> first sentences of the retrieved context (grounded)
    router    -> adapter ON: nearest-centroid classifier fitted on the TRAIN split, emits strict JSON
                 adapter OFF: chatty non-JSON text (like an un-tuned base model)
    """

    def __init__(self, train_df=None):
        self.has_adapter = True
        self.clf = None
        if train_df is not None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(train_df["text"])
            self.clf = LogisticRegression(max_iter=300).fit(self.vec.transform(train_df["text"]), train_df["intent"])
            self.cat = train_df.drop_duplicates("intent").set_index("intent")["category"].to_dict()

    @staticmethod
    def _h(s: str, mod: int) -> int:
        return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod

    def chat(self, messages_list, max_new_tokens, batch_size=8, use_adapter=False, desc=""):
        out = []
        for m in messages_list:
            system, user = m[0]["content"], m[-1]["content"]
            if system.startswith("You are an intent router"):
                out.append(self._route(user) if use_adapter else
                           f"Sure! The customer seems to be asking about: {user[:50]}. I would classify this as a support request.")
            elif system.startswith("You are a customer support assistant. Answer strictly"):
                ctx = system.split("SOP context:\n", 1)[1]
                body = re.sub(r"\[[^\]]+\]\n", "", ctx).split("\n\n")[0]
                body = re.sub(r"\s+", " ", body)
                sents = re.split(r"(?<=[.!?])\s+", body)
                out.append(" ".join(sents[:3]))
            else:
                a, b = 1 + self._h(user, 4), 5 + self._h(user[::-1], 9)
                out.append(f"I'm sorry for the trouble. Your request should be resolved within {a}-{b} business days. "
                           f"If you still need help, call us on 1-800-555-0199 and we will guarantee a quick fix.")
        return out

    def _route(self, text: str) -> str:
        if self.clf is None:
            return json.dumps({"intent": "track_order", "category": "ORDER"})
        intent = self.clf.predict(self.vec.transform([text]))[0]
        return json.dumps({"intent": intent, "category": self.cat[intent]})


def make_llm(dry_run: bool, adapter_dir: str | None = None, train_df=None):
    return FakeLLM(train_df) if dry_run else HFLLM(adapter_dir)
