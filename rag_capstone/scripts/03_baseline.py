"""Stage 3.1 - Baseline model evaluation (zero-shot, no context).  Creates: outputs.json"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from capstone import config as C
from capstone.llm import make_llm
from capstone.metrics import check_hallucination, fact_coverage, hallu_detail
from capstone.pipeline import common_args, generate_baseline, write_json

args, P = common_args(__doc__)
llm = make_llm(args.dry_run)

test_query = ("My package still hasn't arrived and it's way past the date they quoted me. "
              "This is ridiculous, what is going on with my delivery?")
ground_truth = ("Domestic orders deliver within 3-7 business days; international orders within 10-21 business days. "
                "An order is only considered delayed once it has passed the upper bound of its quoted window. "
                "If the parcel shows no movement for more than 5 business days, treat it as potentially lost and offer "
                "a replacement shipment or a full refund. If it is in transit but late, open a carrier trace and tell the "
                "customer to expect an update within 24-48 hours.")
baseline_output = generate_baseline(llm, [test_query])[0]
print("Query   :", test_query, "\nBaseline:", baseline_output)

FACTS_SHIP = [r"3\s*(?:-|–|to)\s*7\s*business", r"10\s*(?:-|–|to)\s*21\s*business"]
print("Key SOP facts present:", f"{fact_coverage(baseline_output, FACTS_SHIP) * 100:.0f}%")
print("Hallucination check  :", check_hallucination(baseline_output, ground_truth, test_query))

PROBES = [
    dict(q=test_query, facts=FACTS_SHIP, sop=ground_truth),
    dict(q="How long do I have to ask for a refund and when does the money come back to me?",
         facts=[r"30\s*days", r"5\s*(?:-|–|to)\s*7\s*business"],
         sop="Refunds may be requested within 30 days of purchase or delivery. Approved refunds are released within 5-7 business days; bank posting can add 3-5 business days, up to 12 business days in total."),
    dict(q="I forgot my password and the reset link in my email doesn't work any more, what now?",
         facts=[r"60\s*minutes", r"single[- ]use|used once|already used|once"],
         sop="The password reset link is valid for 60 minutes and is single use. If expired or used, generate a fresh reset. Minimum 10 characters."),
    dict(q="When can I actually talk to someone on the phone?",
         facts=[r"08:00|8(?::00)?\s*(?:am|a\.m\.)?\s*(?:-|–|to)", r"10:00|10\s*(?:am)?\s*(?:-|–|to)\s*(?:6|18)"],
         sop="Live chat and phone support operate Monday to Friday 08:00-20:00 and weekends 10:00-18:00. Phone average wait under 5 minutes; live chat under 2 minutes; email first reply within one business day."),
    dict(q="My account got locked after I typed the wrong password a few times.", facts=[r"30\s*minutes", r"five|5"],
         sop="Accounts lock for 30 minutes after five failed sign-in attempts. If the customer did not make those attempts, verify identity, force a password reset and recommend two-factor authentication."),
    dict(q="I got a shipping confirmation yesterday but the tracking page shows nothing. Is my parcel lost?", facts=[r"24\s*hours"],
         sop="Tracking can take up to 24 hours to show the first scan after a label is created, so a label with no movement on day one is expected, not lost."),
]
probe_out = generate_baseline(llm, [p["q"] for p in PROBES])
rows = []
for p, o in zip(PROBES, probe_out):
    c = check_hallucination(o, p["sop"], p["q"])
    rows.append({"query": p["q"][:60] + "...", "fact_coverage_%": round(fact_coverage(o, p["facts"]) * 100),
                 "hallucinated": c["hallucinated"], "unsupported": hallu_detail(c)})
probe_df = pd.DataFrame(rows); print(probe_df.to_string(index=False))

runs = [generate_baseline(llm, [test_query])[0] for _ in range(3)]
consistent = len(set(runs)) == 1
print("Consistency over 3 runs:", "IDENTICAL" if consistent else "DIFFERENT")

assessment = {"functional_correctness_%": round(fact_coverage(baseline_output, FACTS_SHIP) * 100),
              "probe_mean_fact_coverage_%": round(float(probe_df["fact_coverage_%"].mean()), 1),
              "probe_hallucination_rate_%": round(float(probe_df["hallucinated"].mean() * 100), 1),
              "consistent_3_runs": consistent}
write_json(P, "outputs.json", {"test_query": test_query, "ground_truth": ground_truth, "baseline_output": baseline_output,
                               "baseline_assessment": assessment, "model_id": C.MODEL_ID})
print("Baseline assessment:", assessment, "\nSaved outputs.json")
