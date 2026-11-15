"""Synthetic Bitext-shaped data used ONLY by --dry-run (the real Bitext download needs Hugging Face access).

Same schema as Bitext (instruction, intent, category) and the same 27 intents / 11 categories, so every downstream
stage behaves exactly as it would on the real data. The text itself is template-generated - never report metrics
from it as real results.
"""
from __future__ import annotations

import itertools
import random

import pandas as pd

INTENT_CATEGORY = {
    "cancel_order": "ORDER", "change_order": "ORDER", "place_order": "ORDER", "track_order": "ORDER",
    "change_shipping_address": "SHIPPING_ADDRESS", "set_up_shipping_address": "SHIPPING_ADDRESS",
    "delivery_options": "DELIVERY", "delivery_period": "DELIVERY",
    "check_refund_policy": "REFUND", "get_refund": "REFUND", "track_refund": "REFUND",
    "check_cancellation_fee": "CANCELLATION_FEE", "check_invoice": "INVOICE", "get_invoice": "INVOICE",
    "check_payment_methods": "PAYMENT", "payment_issue": "PAYMENT", "complaint": "FEEDBACK", "review": "FEEDBACK",
    "contact_human_agent": "CONTACT", "contact_customer_service": "CONTACT",
    "create_account": "ACCOUNT", "switch_account": "ACCOUNT", "registration_problems": "ACCOUNT",
    "recover_password": "ACCOUNT", "edit_account": "ACCOUNT", "delete_account": "ACCOUNT",
    "newsletter_subscription": "NEWSLETTER",
}

CORE = {
    "cancel_order": ["cancel my order", "stop my order from shipping", "call off the purchase I just made", "cancel the order I placed yesterday"],
    "change_order": ["change something in my order", "swap an item in my order", "modify my order before it ships", "update the size on my order"],
    "place_order": ["place an order for a new item", "buy a product from your store", "check out my cart", "order two of these"],
    "track_order": ["track my order", "see where my parcel is", "find the status of my package", "get the tracking number for my order"],
    "change_shipping_address": ["change the shipping address on my order", "send my order to a different address", "correct the delivery address I typed", "update where my package is going"],
    "set_up_shipping_address": ["set up a shipping address", "add a delivery address to my profile", "save a new address for deliveries", "register my home address for shipping"],
    "delivery_options": ["know the delivery options", "see which shipping methods you offer", "choose express or standard delivery", "learn about delivery choices"],
    "delivery_period": ["know how long delivery takes", "find out when my order will arrive", "ask about the delivery time", "see how many days shipping takes"],
    "check_refund_policy": ["read your refund policy", "know if I can get my money back", "understand your return and refund rules", "check what the refund policy says"],
    "get_refund": ["get a refund", "ask for my money back", "request a refund for a damaged item", "be refunded for my last order"],
    "track_refund": ["track my refund", "find out when my refund will arrive", "check the status of my refund", "see where my money is"],
    "check_cancellation_fee": ["know the cancellation fee", "find out what it costs to cancel my subscription", "check fees for ending my plan", "see if there is a charge to cancel"],
    "check_invoice": ["check my invoice", "view the charges on my bill", "see the details of my invoice", "review the amount I was billed"],
    "get_invoice": ["get a copy of my invoice", "download my invoice", "receive the invoice for my purchase", "have the bill sent to me"],
    "check_payment_methods": ["know which payment methods you accept", "see if you take credit cards", "find out if I can pay with a wallet", "check the available ways to pay"],
    "payment_issue": ["report a problem with my payment", "fix a failed payment", "understand why my card was declined", "sort out a payment error"],
    "complaint": ["file a complaint", "complain about the service I got", "report a bad experience", "make a formal complaint"],
    "review": ["leave a review", "give feedback about a product", "write a review of your service", "share my opinion on my purchase"],
    "contact_human_agent": ["speak to a human agent", "talk to a real person", "be transferred to a live agent", "reach a human, not a bot"],
    "contact_customer_service": ["contact customer service", "know your support hours", "find the support phone number", "reach your help desk"],
    "create_account": ["create an account", "sign up with you", "open a new account", "register as a customer"],
    "switch_account": ["switch to another account", "change to my other profile", "use a different account", "log into my second account"],
    "registration_problems": ["fix a problem registering", "solve a sign-up error", "figure out why I cannot register", "get past the registration failure"],
    "recover_password": ["recover my password", "reset my forgotten password", "get back into my account, I lost the password", "change my password because I forgot it"],
    "edit_account": ["edit my account details", "update my profile information", "change my email on file", "correct my account name"],
    "delete_account": ["delete my account", "close my account for good", "remove my profile and data", "erase my account"],
    "newsletter_subscription": ["manage the newsletter", "unsubscribe from your emails", "sign up for the newsletter", "stop the marketing emails"],
}
OPEN = ["I want to", "I need to", "Help me", "Hi, I'd like to", "Could you help me", "Can you help me", "I am trying to",
        "Please help, I have to", "Hello, I want to", "Quick question - how do I"]
TAIL = ["", " please", " as soon as possible", " today", " if possible", " - thanks", " ASAP", " and it is urgent"]
ADV_OPEN = ["This is ridiculous, I still can't", "I have never been able to", "Terrible service, I am frustrated, I just want to",
            "Again?! I only want to", "Nobody helps me, I need to"]


def make_synthetic_bitext(n: int = 4400, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    per_intent = -(-n // len(INTENT_CATEGORY))
    rows = []
    for intent, cores in CORE.items():
        combos = [(o, c, t) for o, c, t in itertools.product(OPEN + ADV_OPEN, cores, TAIL)]
        rng.shuffle(combos)
        for o, c, t in combos[:per_intent]:
            rows.append({"instruction": f"{o} {c}{t}", "category": INTENT_CATEGORY[intent], "intent": intent})
    rng.shuffle(rows)
    df = pd.DataFrame(rows).iloc[:n].reset_index(drop=True)
    # Bitext-style messiness so the cleaning stage has real work: casing/whitespace dupes, a few nulls.
    dup = df.sample(25, random_state=seed).copy()
    dup["instruction"] = dup["instruction"].str.upper().radd("  ") + " "
    df = pd.concat([df, dup], ignore_index=True)
    df.loc[df.sample(3, random_state=seed + 1).index, "instruction"] = None
    return df
