"""Knowledge-base helpers: SOP loading + chunking, intent->SOP routing table, SOP-grounded references."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Gold SOP per Bitext intent (business routing table). Intents without a dedicated SOP map to the closest one.
INTENT_TO_SOP = {
    "cancel_order": "order_tracking", "change_order": "order_tracking", "place_order": "payment_methods",
    "track_order": "order_tracking", "change_shipping_address": "order_tracking",
    "set_up_shipping_address": "order_tracking", "delivery_options": "shipping_delays",
    "delivery_period": "shipping_delays", "check_refund_policy": "refund_policy", "get_refund": "refund_policy",
    "track_refund": "refund_policy", "check_cancellation_fee": "subscription_cancellation",
    "check_invoice": "billing_disputes", "get_invoice": "billing_disputes",
    "check_payment_methods": "payment_methods", "payment_issue": "payment_methods",
    "complaint": "escalation_matrix", "review": "escalation_matrix", "contact_human_agent": "escalation_matrix",
    "contact_customer_service": "working_hours", "create_account": "account_recovery",
    "switch_account": "account_recovery", "registration_problems": "technical_troubleshooting",
    "recover_password": "password_reset", "edit_account": "data_privacy", "delete_account": "data_privacy",
    "newsletter_subscription": "data_privacy",
}

# Natural-language vector-search string per intent (what Hybrid RAG sends to the vector DB instead of the noisy query).
INTENT_SEARCH = {
    "cancel_order": "cancel an order that has not shipped yet, order states processing and preparing to ship",
    "change_order": "change or modify an existing order before it ships, order states",
    "place_order": "checkout payment authorization when placing an order, accepted payment methods",
    "track_order": "track order status, tracking number, carrier link, in transit, delivered",
    "change_shipping_address": "change the delivery address on an order, address correction with the carrier",
    "set_up_shipping_address": "set up the shipping address for delivery of an order",
    "delivery_options": "delivery options and standard delivery timelines, domestic and international shipping",
    "delivery_period": "how many business days delivery takes, domestic and international timelines",
    "check_refund_policy": "refund eligibility window, refund methods and processing time",
    "get_refund": "request a refund for an order, eligibility, processing time and agent steps",
    "track_refund": "where is my refund, refund processing time and bank posting delay",
    "check_cancellation_fee": "cancellation options, fees and refunds for the billed period of a subscription",
    "check_invoice": "check an invoice or charge, amount, tax and shipping components of a charge",
    "get_invoice": "obtain a copy of an invoice for a charge, billing reference",
    "check_payment_methods": "accepted payment methods, credit and debit cards, wallets, bank transfer",
    "payment_issue": "payment failed or declined, authorization hold, failed payments",
    "complaint": "customer complaint, escalation severity, supervisor, handing off a case",
    "review": "customer feedback and reviews, communication and escalation",
    "contact_human_agent": "escalate to a human agent or supervisor, routing and severity levels",
    "contact_customer_service": "customer support availability, live chat and phone hours, response targets",
    "create_account": "create a new account, identity verification and registered email",
    "switch_account": "switch to another account, identity verification and account access",
    "registration_problems": "sign-up or registration errors, app and login failures, troubleshooting",
    "recover_password": "forgot password, self-service reset link sent by email",
    "edit_account": "update or correct account and personal details through the verified-change workflow",
    "delete_account": "delete my account and personal data, data retention and privacy rights",
    "newsletter_subscription": "marketing emails, newsletter preferences and opt-out",
}

REQUIRED_SOPS = ["refund_policy", "shipping_delays", "password_reset", "technical_troubleshooting", "escalation_matrix"]


def load_sops(sop_dir: Path) -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(Path(sop_dir).glob("*.md"))}


def validate_sops(sops: dict[str, str]) -> None:
    missing = [r for r in REQUIRED_SOPS if r not in sops]
    assert not missing, f"required SOP files missing: {missing}"
    missing_targets = sorted(set(INTENT_TO_SOP.values()) - set(sops))
    assert not missing_targets, f"INTENT_TO_SOP points at SOPs that do not exist: {missing_targets}"
    empty = [k for k, v in sops.items() if not v.strip()]
    assert not empty, f"empty SOP files: {empty}"


def build_chunks(sop_dir: Path) -> list[dict]:
    """One chunk per Markdown '##' section; long sections are split further (700 chars, 80 overlap)."""
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    md_split = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=True)
    size_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=80)
    chunks = []
    for stem, text in load_sops(sop_dir).items():
        n = 0
        for sec in md_split.split_text(text):
            title, section = sec.metadata.get("title", stem), sec.metadata.get("section", "Overview")
            for piece in size_split.split_text(sec.page_content):
                chunks.append({"chunk_id": f"{stem}::{n:02d}", "source": stem, "title": title, "section": section,
                               "text": f"{title} - {section}\n{piece.strip()}"})
                n += 1
    return chunks


def md_to_plain(md: str) -> str:
    t = re.sub(r"^#+\s*", "", md, flags=re.M)
    t = re.sub(r"\*\*|__|`", "", t)
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.M)
    t = re.sub(r"^\s*\d+[.)]\s+", "", t, flags=re.M)
    return re.sub(r"\s+", " ", t).strip()


def sop_sections(sop_text: str) -> list[str]:
    parts = re.split(r"\n(?=## )", sop_text.strip())
    return [p for p in parts if p.startswith("## ")]


def sop_reference(sops: dict[str, str], intent: str, query: str = "", n_sections: int = 2) -> str:
    """SOP-grounded reference: the n sections of the GOLD SOP (from the intent label) most similar to the query."""
    stem = INTENT_TO_SOP[intent]
    all_secs = sop_sections(sops[stem])
    secs = [s for s in all_secs if not s.startswith("## Agent Steps")] or all_secs
    docs = [md_to_plain(s) for s in secs]
    q = f"{query} {INTENT_SEARCH[intent]}"
    vec = TfidfVectorizer(stop_words="english").fit(docs + [q])
    sims = cosine_similarity(vec.transform([q]), vec.transform(docs))[0]
    top = sorted(np.argsort(-sims)[:n_sections])
    return " ".join(docs[i] for i in top)
