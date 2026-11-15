#!/usr/bin/env python
"""Run the whole pipeline in order.

    python run_all.py --dry-run     # CPU-only wiring check (synthetic data, stand-in model) - minutes, no downloads
    python run_all.py               # real run: needs a CUDA GPU + Hugging Face access (Colab T4 is enough)
    python run_all.py --from 5      # resume from stage 5
"""
import argparse, subprocess, sys, time
from pathlib import Path

STAGES = ["01_eda", "02_prepare", "03_baseline", "04_build_index", "05_eval_v1", "06_finetune", "07_eval_v2"]

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--from", dest="start", type=int, default=1, help="first stage number to run (1-7)")
ap.add_argument("--only", type=int, default=None, help="run a single stage number")
ap.add_argument("--n-eval", type=int, default=None, help="evaluate only N test rows (faster)")
ap.add_argument("--bitext-csv", default=None, help="local Bitext CSV (instruction,intent,category) instead of HF download")
a = ap.parse_args()

here = Path(__file__).parent / "scripts"
for i, name in enumerate(STAGES, 1):
    if (a.only and i != a.only) or i < a.start:
        continue
    if a.dry_run and name == "06_finetune":
        print(f"\n===== {name}: skipped in dry-run (needs GPU) ====="); continue
    cmd = [sys.executable, str(here / f"{name}.py")] + (["--dry-run"] if a.dry_run else [])
    if a.n_eval and name in ("05_eval_v1", "07_eval_v2"):
        cmd += ["--n-eval", str(a.n_eval)]
    if a.bitext_csv and name == "01_eda":
        cmd += ["--bitext-csv", a.bitext_csv]
    print(f"\n===== Stage {i}: {name} =====", flush=True)
    t0 = time.time()
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"\nStage {name} FAILED - fix it before continuing.")
    print(f"----- {name} done in {time.time() - t0:.0f}s")
print("\nAll requested stages completed.")
