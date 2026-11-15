"""Stage 4.1/4.2 - QLoRA fine-tuning of the intent router with a custom AdamW loop, validation-loss early stopping and
best-checkpoint saving.  Needs a CUDA GPU (Colab T4 is enough).
Creates: intent_lora_best/, intent_lora/, training_log.csv, training_curves.png, training_summary.json"""
import os, sys, json, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from capstone import config as C
from capstone.pipeline import common_args, require

args, P = common_args(__doc__)
if args.dry_run:
    sys.exit("[dry-run] fine-tuning needs a GPU + Hugging Face. Skipped: the stand-in router is used by 07_eval_v2.py.")
require(P, "tokenized_train", "tokenized_valid", "df_valid.csv")

import torch, transformers, peft
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

assert torch.cuda.is_available(), "Enable a GPU runtime (T4 minimum)"
random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED); torch.cuda.manual_seed_all(C.SEED)
T = C.TRAIN
tokenizer = AutoTokenizer.from_pretrained(C.MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4.1.1 PEFT (QLoRA: 4-bit NF4 frozen base + trainable low-rank adapters)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(C.MODEL_ID, quantization_config=bnb, device_map="auto")
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
model = get_peft_model(base, LoraConfig(**C.LORA_CFG)); model.print_trainable_parameters()

# 4.1.2 training parameters
cap = torch.cuda.get_device_capability(0)[0]
AMP = torch.bfloat16 if cap >= 8 else torch.float16
device = torch.device("cuda")
cols = ["input_ids", "attention_mask", "labels"]
train_ds, valid_ds = load_from_disk(str(P.f("tokenized_train"))), load_from_disk(str(P.f("tokenized_valid")))
train_ds.set_format("torch", columns=cols); valid_ds.set_format("torch", columns=cols)
print("train/valid rows:", len(train_ds), len(valid_ds))

def collate(batch):
    ids = torch.stack([b["input_ids"] for b in batch]); att = torch.stack([b["attention_mask"] for b in batch])
    lab = torch.stack([b["labels"] for b in batch]); L = int(att.sum(1).max())
    return {"input_ids": ids[:, :L], "attention_mask": att[:, :L], "labels": lab[:, :L]}

g = torch.Generator().manual_seed(C.SEED)
train_loader = DataLoader(train_ds, batch_size=T["batch_size"], shuffle=True, collate_fn=collate, generator=g)
valid_loader = DataLoader(valid_ds, batch_size=T["eval_batch_size"], shuffle=False, collate_fn=collate)
trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(trainable, lr=T["lr"], weight_decay=T["weight_decay"])
sched = get_linear_schedule_with_warmup(opt, T["warmup_steps"], T["max_steps"])
scaler = torch.amp.GradScaler("cuda", enabled=(AMP == torch.float16))
HP = dict(**T, amp_dtype=str(AMP), optimizer="AdamW", quantisation="4-bit NF4 + double-quant (QLoRA)", seed=C.SEED, lora=C.LORA_CFG, model_id=C.MODEL_ID)

@torch.no_grad()
def run_validation():
    model.eval(); tot, n = 0.0, 0
    for b in valid_loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=AMP):
            loss = model(**b).loss
        k = int((b["labels"][:, 1:] != -100).sum()); tot += loss.item() * k; n += k
    model.train()
    return tot / n

def batches():
    while True:
        yield from train_loader

# 4.2.1 training loop
model.config.use_cache = False; model.train()
log = [{"step": 0, "train_loss": np.nan, "val_loss": run_validation()}]
print(f"step    0 | val {log[0]['val_loss']:.4f} (untrained adapter)")
best_val, best_step, bad, stopped = float("inf"), 0, 0, False
recent, step, t0 = [], 0, time.time()
for batch in batches():
    step += 1
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast("cuda", dtype=AMP):
        loss = model(**batch).loss
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(trainable, T["grad_clip"])
    scaler.step(opt); scaler.update(); sched.step(); recent.append(loss.item())
    if step % T["val_every"] == 0:
        tr, va = float(np.mean(recent)), run_validation(); recent = []
        log.append({"step": step, "train_loss": tr, "val_loss": va})
        improved = va < best_val - T["min_delta"]
        if improved:
            best_val, best_step, bad = va, step, 0
            model.save_pretrained(P.lora_best)                  # best checkpoint BY VALIDATION LOSS
        else:
            bad += 1
        note = "*saved*" if improved else f"(no improvement {bad}/{T['patience']})"
        print(f"step {step:4d} | train {tr:.4f} | val {va:.4f} | best {best_val:.4f}@{best_step} {note} | {time.time() - t0:.0f}s")
        if bad >= T["patience"]:
            stopped = True; print(f"Early stopping at step {step}"); break
    if step >= T["max_steps"]:
        break
print(f"Done: {step} steps in {(time.time() - t0) / 60:.1f} min | best val {best_val:.4f} @ {best_step} | early_stopped={stopped}")

# curves + 4.2.2 outputs
log_df = pd.DataFrame(log)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(log_df["step"], log_df["train_loss"], marker="o", label="train loss")
ax.plot(log_df["step"], log_df["val_loss"], marker="s", label="validation loss")
ax.axvline(best_step, color="green", ls="--", label=f"best checkpoint (step {best_step})")
ax.set_xlabel("optimizer step"); ax.set_ylabel("cross-entropy"); ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend()
ax.set_title("QLoRA intent-router: train vs validation loss"); plt.tight_layout(); plt.savefig(P.f("training_curves.png"), dpi=150)
log_df.to_csv(P.f("training_log.csv"), index=False)
model.save_pretrained(P.lora_final)
assert os.path.exists(os.path.join(P.lora_best, "adapter_config.json")), "best checkpoint missing"
json.dump({"best_val_loss": best_val, "best_step": best_step, "steps_run": step, "early_stopped": stopped,
           "train_seconds": round(time.time() - t0), "hyperparameters": HP,
           "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}},
          open(P.f("training_summary.json"), "w"), indent=2, default=str)
print("Saved:", P.lora_best, P.lora_final, "training_log.csv training_curves.png training_summary.json")
