"""Deterministic two-pass reconciliation matcher.

Pass 1 pairs rows that agree on amount to the cent and on date within a day.
Pass 2 scores what is left with an explicit, auditable confidence formula.

No LLM, no network, no fuzzy-matching dependency -- difflib only, so the same
inputs always produce the same output and every score can be recomputed by hand.
"""
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd

# --- tunables: the whole behaviour of the engine lives in these six numbers ---
EXACT_DATE_DAYS = 1          # pass 1: dates may differ by this much
FUZZY_DATE_DAYS = 5          # pass 2: widest date window considered
AMOUNT_TOLERANCE_ABS = 2.00  # pass 2: absolute amount tolerance, whichever is larger
AMOUNT_TOLERANCE_PCT = 0.005 # pass 2: 0.5% of the ledger amount
AUTO_MATCH_CONFIDENCE = 0.70 # at or above this a fuzzy pair is auto-matched

# Confidence = weighted mean of three sub-scores, each in 0..1. Weights sum to 1.
W_AMOUNT, W_DATE, W_TEXT = 0.35, 0.20, 0.45
# Text score is itself a mix: the invoice number carries more signal than the name.
W_REFERENCE, W_NAME = 0.60, 0.40


@dataclass
class Match:
    ledger_id: str
    bank_id: str
    method: str  # "exact" | "fuzzy"
    confidence: float
    reasons: list = field(default_factory=list)


@dataclass
class Exception_:
    """Best candidate found, but not confident enough to post without a human."""
    ledger_id: str
    bank_id: str
    confidence: float
    reasons: list = field(default_factory=list)


@dataclass
class Unmatched:
    side: str  # "ledger" | "bank"
    transaction_id: str
    reason: str


# --------------------------------------------------------------------------- #
# sub-scores
# --------------------------------------------------------------------------- #

def _norm(s):
    return "".join(c for c in str(s).upper() if c.isalnum() or c == " ").strip()


def _digits(s):
    return "".join(c for c in str(s) if c.isdigit())


def reference_score(a, b):
    """An invoice number is an identifier: either it is the same one or it is not.

    Three rules, in order:
      equal digits                -> 1.00  'INV-1023', 'INV1023', 'PMT INV 1023'
      one run contains the other  -> 0.85  'INV-1023' against a truncated '023'
      anything else               -> 0.20

    The last rule is the point. Invoice numbers are issued sequentially, so
    INV-1024 is not "75% of" INV-1023 -- it is the single most likely wrong pair
    in the file. Scoring near misses by character overlap rewards exactly the
    collision a reconciliation system must never make.

    A containing run has to be at least three digits: '1' inside '1023' is a
    coincidence, not a truncated reference.

    References with no digits at all (free-text memos) fall back to comparing the
    normalised text, where there is no identifier to reason about.
    """
    da, db = _digits(a), _digits(b)
    if da and db:
        if da == db:
            return 1.0
        short, long = sorted((da, db), key=len)
        return 0.85 if len(short) >= 3 and short in long else 0.20
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def name_score(vendor, description):
    """How much of the vendor name survives inside the bank description.

    Banks truncate ('CASCADE OFFICE SUP'), so we measure the longest run of the
    vendor name that appears in the description rather than comparing wholesale.
    """
    v, d = _norm(vendor), _norm(description)
    if not v or not d:
        return 0.0
    block = SequenceMatcher(None, v, d).find_longest_match(0, len(v), 0, len(d))
    return block.size / len(v)


def _decay(distance, tolerance):
    """1.0 at zero distance, 0.5 at the edge of tolerance, 0.0 beyond it.

    Being inside tolerance is itself evidence, so the score floors at 0.5 rather
    than collapsing to nothing the moment a bank fee shifts an amount by $2.
    """
    if tolerance <= 0:
        return 1.0 if distance == 0 else 0.0
    if distance > tolerance:
        return 0.0
    return 1.0 - 0.5 * (distance / tolerance)


def amount_tolerance(amount):
    return max(AMOUNT_TOLERANCE_ABS, abs(amount) * AMOUNT_TOLERANCE_PCT)


def score_pair(ledger, bank):
    """Confidence in 0..1 for one candidate pair, plus the reasons behind it."""
    amount_diff = round(abs(ledger["amount"] - bank["amount"]), 2)
    day_diff = abs((ledger["date"] - bank["date"]).days)
    tolerance = amount_tolerance(ledger["amount"])

    amount = _decay(amount_diff, tolerance)
    date = _decay(day_diff, FUZZY_DATE_DAYS)
    reference = reference_score(ledger["reference"], bank["reference"])
    name = max(
        name_score(ledger["vendor"], bank["description"]),
        name_score(ledger["description"], bank["description"]),
    )
    text = W_REFERENCE * reference + W_NAME * name
    confidence = W_AMOUNT * amount + W_DATE * date + W_TEXT * text

    reasons = [
        "amount matches exactly" if amount_diff == 0 else f"amount differs by ${amount_diff:,.2f}",
        "same date" if day_diff == 0 else f"date offset {day_diff} day{'s' if day_diff != 1 else ''}",
        f"reference similarity {reference:.2f}",
        f"vendor name similarity {name:.2f}",
    ]
    return round(confidence, 4), reasons


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #

def _text(value):
    """A blank cell is a blank string, never the literal 'nan'.

    Bank memos routinely carry no reference; how the caller happened to read the
    CSV must not change what a missing reference scores.
    """
    return "" if value is None or pd.isna(value) else str(value)


def _rows(df, has_vendor):
    out = []
    for r in df.to_dict("records"):
        out.append({
            "transaction_id": str(r["transaction_id"]),
            "date": pd.to_datetime(r["date"]),
            "amount": round(float(r["amount"]), 2),
            "reference": _text(r.get("reference", "")),
            "description": _text(r.get("description", "")),
            "vendor": _text(r.get("vendor", "")) if has_vendor else "",
        })
    return out


def _consume(candidates, taken_ledger, taken_bank):
    """Greedy one-to-one assignment, best candidate first.

    ponytail: greedy, not Hungarian -- swap in scipy.optimize.linear_sum_assignment
    only if real data shows contested pairs being assigned suboptimally.
    """
    paired = []
    for _sort_key, ledger, bank, confidence, reasons in candidates:
        if ledger["transaction_id"] in taken_ledger or bank["transaction_id"] in taken_bank:
            continue
        taken_ledger.add(ledger["transaction_id"])
        taken_bank.add(bank["transaction_id"])
        paired.append((ledger, bank, confidence, reasons))
    return paired


def match_transactions(ledger_df, bank_df):
    """Reconcile a ledger against a bank statement.

    Returns (matches, exceptions, unmatched):
      matches    -- Match, either an exact pair or a fuzzy pair we trust
      exceptions -- Exception_, a plausible pair a human should confirm
      unmatched  -- Unmatched, rows on either side with no plausible partner
    """
    ledger = _rows(ledger_df, has_vendor="vendor" in ledger_df.columns)
    bank = _rows(bank_df, has_vendor="vendor" in bank_df.columns)
    taken_ledger, taken_bank = set(), set()

    # --- pass 1: exact amount, date within a day -------------------------- #
    exact_candidates = []
    for l in ledger:
        for b in bank:
            if l["amount"] != b["amount"]:
                continue
            day_diff = abs((l["date"] - b["date"]).days)
            if day_diff > EXACT_DATE_DAYS:
                continue
            reference = reference_score(l["reference"], b["reference"])
            sort_key = (day_diff, -reference, l["transaction_id"], b["transaction_id"])
            reasons = ["amount matches exactly",
                       "same date" if day_diff == 0 else f"date offset {day_diff} day",
                       f"reference similarity {reference:.2f}"]
            exact_candidates.append((sort_key, l, b, 1.0, reasons))
    exact_candidates.sort(key=lambda c: c[0])

    matches = [
        Match(l["transaction_id"], b["transaction_id"], "exact", conf, reasons)
        for l, b, conf, reasons in _consume(exact_candidates, taken_ledger, taken_bank)
    ]

    # --- pass 2: fuzzy over the leftovers --------------------------------- #
    fuzzy_candidates = []
    for l in ledger:
        if l["transaction_id"] in taken_ledger:
            continue
        for b in bank:
            if b["transaction_id"] in taken_bank:
                continue
            if abs(round(l["amount"] - b["amount"], 2)) > amount_tolerance(l["amount"]):
                continue
            if abs((l["date"] - b["date"]).days) > FUZZY_DATE_DAYS:
                continue
            confidence, reasons = score_pair(l, b)
            sort_key = (-confidence, l["transaction_id"], b["transaction_id"])
            fuzzy_candidates.append((sort_key, l, b, confidence, reasons))
    fuzzy_candidates.sort(key=lambda c: c[0])

    exceptions = []
    for l, b, confidence, reasons in _consume(fuzzy_candidates, taken_ledger, taken_bank):
        if confidence >= AUTO_MATCH_CONFIDENCE:
            matches.append(Match(l["transaction_id"], b["transaction_id"], "fuzzy", confidence, reasons))
        else:
            exceptions.append(Exception_(l["transaction_id"], b["transaction_id"], confidence, reasons))

    # --- whatever is left had no plausible partner ------------------------ #
    unmatched = [Unmatched("ledger", l["transaction_id"], "no bank row within amount and date tolerance")
                 for l in ledger if l["transaction_id"] not in taken_ledger]
    unmatched += [Unmatched("bank", b["transaction_id"], "no ledger row within amount and date tolerance")
                  for b in bank if b["transaction_id"] not in taken_bank]

    matches.sort(key=lambda m: m.ledger_id)
    exceptions.sort(key=lambda e: e.ledger_id)
    unmatched.sort(key=lambda u: (u.side, u.transaction_id))
    return matches, exceptions, unmatched
