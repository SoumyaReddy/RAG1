"""Offline tests (no GPU / Hugging Face). Run:  pytest -q"""
import json, os, subprocess, sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from capstone import config as C
from capstone.kb import INTENT_SEARCH, INTENT_TO_SOP, build_chunks, load_sops, sop_reference, validate_sops
from capstone.metrics import check_hallucination, fuzzy_match, is_well_formed, parse_router, score_text
from capstone.synthetic import INTENT_CATEGORY, make_synthetic_bitext


def test_sops_valid_and_routing_table_complete():
    sops = load_sops(C.get_paths(True).sop_dir)
    validate_sops(sops)
    assert set(INTENT_TO_SOP) == set(INTENT_SEARCH) == set(INTENT_CATEGORY) and len(INTENT_TO_SOP) == 27


def test_chunking_covers_every_sop_without_duplicates():
    ch = build_chunks(C.get_paths(True).sop_dir)
    assert {c["source"] for c in ch} == set(load_sops(C.get_paths(True).sop_dir))
    assert len({c["chunk_id"] for c in ch}) == len(ch) and len({c["text"] for c in ch}) == len(ch)
    assert min(len(c["text"]) for c in ch) > 40


def test_references_exist_for_every_intent():
    sops = load_sops(C.get_paths(True).sop_dir)
    for intent in INTENT_TO_SOP:
        assert len(sop_reference(sops, intent, "my order is late").split()) > 10


def test_hallucination_detector():
    sop = "Refunds may be requested within 30 days. Released within 5-7 business days."
    assert not check_hallucination("You can request a refund within 30 days and it arrives in 5-7 business days.", sop)["hallucinated"]
    assert check_hallucination("Refunds arrive in 2 days.", sop)["hallucinated"]
    assert check_hallucination("Call 1-800-555-0100 for help.", sop)["hallucinated"]
    assert check_hallucination("We guarantee delivery.", sop)["hallucinated"]


def test_router_parsing_and_format_checks():
    assert parse_router('{"intent": "get_refund", "category": "REFUND"}')[1:] == (True, False)
    obj, strict, salvaged = parse_router('Sure! {"intent": "get_refund", "category": "REFUND"} hope it helps')
    assert obj["intent"] == "get_refund" and not strict and salvaged
    assert parse_router("no json here") == (None, False, False)
    assert fuzzy_match("order_tracking", "track_order") and not fuzzy_match("get_refund", "delete_account")
    assert is_well_formed("This is a long enough answer to count as well formed.") and not is_well_formed("{}")
    assert score_text("refund within 30 days", "refund within 30 days")["rouge1"] == pytest.approx(1.0)


def test_synthetic_data_shape():
    df = make_synthetic_bitext()
    assert {"instruction", "intent", "category"} <= set(df.columns) and df["intent"].nunique() == 27
    assert df["instruction"].isna().sum() > 0     # messy on purpose so cleaning has work to do


@pytest.fixture(scope="module")
def dry_run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("dry")
    env = {**os.environ, "CAPSTONE_OUT": str(out)}
    r = subprocess.run([sys.executable, str(ROOT / "run_all.py"), "--dry-run", "--n-eval", "150"],
                       env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return out


def test_dry_run_end_to_end_artifacts_and_no_leakage(dry_run_dir):
    for f in ["sampled_data.csv", "df_train.csv", "df_valid.csv", "df_test.csv", "retrieval_chunks.json", "outputs.json",
              "v1_metrics.csv", "Comparative_Results_Full.csv", "Comparative_Results_Summary.csv", "sop_similarity_heatmap.png"]:
        assert (dry_run_dir / f).exists(), f
    tr, va, te = (pd.read_csv(dry_run_dir / f"df_{n}.csv") for n in ("train", "valid", "test"))
    norm = lambda d: set(d["text"].str.lower().str.strip())
    assert not (norm(tr) & norm(va)) and not (norm(tr) & norm(te)) and not (norm(va) & norm(te))
    assert set(tr["intent"]) == set(va["intent"]) == set(te["intent"])
    assert 1000 <= len(tr) + len(va) + len(te) <= 5000
    out = json.loads((dry_run_dir / "outputs.json").read_text())
    assert {"baseline_output", "naive_rag_output", "hybrid_rag_output"} <= set(out)


def test_dry_run_metrics_are_sane(dry_run_dir):
    s = pd.read_csv(dry_run_dir / "Comparative_Results_Summary.csv").set_index("Metric")
    assert s.loc["Hallucination frequency (%)", "Baseline"] > s.loc["Hallucination frequency (%)", "Solution_V1"]
    assert s.loc["Router JSON validity (%)", "Solution_V2"] == 100
    assert s.loc["Consistency rate (%)", "Solution_V2"] == 100
    assert s.loc["Retrieval hit@K (%)", "Solution_V2"] >= s.loc["Retrieval hit@K (%)", "Solution_V1"]
