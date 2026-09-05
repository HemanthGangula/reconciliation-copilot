"""Score src/matcher.py against data/ground_truth.json.

    python tests/test_matcher.py     # prints the full report
    pytest tests/test_matcher.py     # same run, asserted thresholds
"""
import json
import os
import sys
from collections import Counter

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from matcher import AUTO_MATCH_CONFIDENCE, match_transactions  # noqa: E402

OUTCOMES = ("auto_correct", "auto_wrong", "exception_correct", "exception_wrong", "unmatched")


def score():
    """Run the matcher over the fixture and compare every ledger row to truth."""
    ledger = pd.read_csv(os.path.join(ROOT, "data", "ledger.csv"))
    bank = pd.read_csv(os.path.join(ROOT, "data", "bank_statement.csv"))
    with open(os.path.join(ROOT, "data", "ground_truth.json")) as f:
        truth = json.load(f)

    matches, exceptions, unmatched = match_transactions(ledger, bank)
    got = {m.ledger_id: ("auto", m.bank_id, m.confidence) for m in matches}
    got.update({e.ledger_id: ("exception", e.bank_id, e.confidence) for e in exceptions})
    got.update({u.transaction_id: ("unmatched", None, 0.0) for u in unmatched if u.side == "ledger"})

    # A duplicate whose bank row is claimed by two ledger rows is a ledger-side
    # duplicate: one row may legitimately match, the other must be flagged.
    claims = Counter(v["bank_transaction_id"] for v in truth.values() if v["bank_transaction_id"])

    rows = {}
    for lid, want in truth.items():
        kind, bank_id, confidence = got.get(lid, ("missing", None, 0.0))
        right = bank_id is not None and bank_id == want["bank_transaction_id"]
        if kind == "auto":
            outcome = "auto_correct" if right else "auto_wrong"
        elif kind == "exception":
            outcome = "exception_correct" if right else "exception_wrong"
        elif kind == "unmatched":
            outcome = "unmatched"
        else:
            outcome = "missing"
        rows[lid] = {
            "match_type": want["match_type"],
            "expected_bank_id": want["bank_transaction_id"],
            "shared_bank_row": claims[want["bank_transaction_id"]] > 1 if want["bank_transaction_id"] else False,
            "outcome": outcome,
            "got_bank_id": bank_id,
            "confidence": confidence,
        }
    return rows, matches, exceptions, unmatched


def metrics(rows):
    by_type = {}
    for r in rows.values():
        by_type.setdefault(r["match_type"], Counter())[r["outcome"]] += 1

    # Rows that should end up on a bank row: everything ground truth gives an id for.
    payable = [r for r in rows.values() if r["expected_bank_id"]]
    solo = [r for r in payable if not r["shared_bank_row"]]
    ledger_only = [r for r in rows.values() if not r["expected_bank_id"]]

    # Ledger-side duplicate pairs: exactly one should match, the sibling be flagged.
    pairs = {}
    for r in rows.values():
        if r["shared_bank_row"]:
            pairs.setdefault(r["expected_bank_id"], []).append(r["outcome"])
    pairs_split = sum(1 for outs in pairs.values() if sum(o == "auto_correct" for o in outs) == 1)

    return {
        "by_type": by_type,
        "n": len(rows),
        "auto_correct": sum(r["outcome"] == "auto_correct" for r in rows.values()),
        "false_positives": sum(r["outcome"] == "auto_wrong" for r in rows.values()),
        "solo_recall": sum(r["outcome"] == "auto_correct" for r in solo) / len(solo),
        "right_candidate": sum(r["outcome"] in ("auto_correct", "exception_correct") for r in payable) / len(payable),
        "unmatched_recall": sum(r["outcome"] == "unmatched" for r in ledger_only) / len(ledger_only),
        "n_solo": len(solo),
        "n_payable": len(payable),
        "n_ledger_only": len(ledger_only),
        "pairs_split": pairs_split,
        "n_pairs": len(pairs),
    }


def report():
    rows, matches, exceptions, unmatched = score()
    m = metrics(rows)
    bank_unmatched = [u for u in unmatched if u.side == "bank"]

    print(f"\nRECONCILIATION ACCURACY  ({m['n']} ledger rows, auto-match threshold {AUTO_MATCH_CONFIDENCE})\n")
    head = f"{'ground truth':<12}" + "".join(f"{o:>18}" for o in OUTCOMES)
    print(head)
    print("-" * len(head))
    totals = Counter()
    for mt in sorted(m["by_type"]):
        counts = m["by_type"][mt]
        totals.update(counts)
        print(f"{mt:<12}" + "".join(f"{counts[o]:>18}" for o in OUTCOMES))
    print("-" * len(head))
    print(f"{'total':<12}" + "".join(f"{totals[o]:>18}" for o in OUTCOMES))

    print(f"\nFALSE POSITIVES (auto-matched to the wrong bank row): {m['false_positives']}"
          f"  ({m['false_positives'] / m['n']:.1%})")
    print(f"auto-matched to the correct bank row      {m['auto_correct']:>3}/{m['n']}"
          f"  ({m['auto_correct'] / m['n']:.1%} of all rows)")
    print(f"  of rows with an unshared bank partner   {m['solo_recall']:.1%}  (n={m['n_solo']})")
    print(f"correct candidate surfaced (auto or exc)  {m['right_candidate']:.1%}  (n={m['n_payable']})")
    print(f"ledger-only rows reported unmatched       {m['unmatched_recall']:.1%}  (n={m['n_ledger_only']})")
    print(f"ledger-side duplicate pairs split 1+1     {m['pairs_split']}/{m['n_pairs']}")
    print(f"\nengine totals: {len(matches)} matched "
          f"({sum(x.method == 'exact' for x in matches)} exact / {sum(x.method == 'fuzzy' for x in matches)} fuzzy), "
          f"{len(exceptions)} exceptions, {len(unmatched)} unmatched "
          f"({len(unmatched) - len(bank_unmatched)} ledger / {len(bank_unmatched)} bank)")

    if exceptions:
        print("\nsample exceptions (what an auditor would read):")
        for e in exceptions[:5]:
            print(f"  {e.ledger_id} -> {e.bank_id}  confidence {e.confidence:.2f}  |  {'; '.join(e.reasons)}")
    return rows, m


def test_no_false_positives():
    _rows, m = report()
    assert m["false_positives"] == 0, f"{m['false_positives']} rows auto-matched to the wrong bank row"


def test_recall_on_unshared_partners():
    m = metrics(score()[0])
    assert m["solo_recall"] >= 0.95, m["solo_recall"]


def test_correct_candidate_surfaced():
    m = metrics(score()[0])
    assert m["right_candidate"] >= 0.95, m["right_candidate"]


def test_ledger_only_rows_reported_unmatched():
    m = metrics(score()[0])
    assert m["unmatched_recall"] >= 0.80, m["unmatched_recall"]


def test_every_row_accounted_for_once():
    rows, matches, exceptions, unmatched = score()
    ids = [x.ledger_id for x in matches] + [x.ledger_id for x in exceptions] + \
          [u.transaction_id for u in unmatched if u.side == "ledger"]
    assert len(ids) == len(set(ids)) == len(rows)
    assert not any(r["outcome"] == "missing" for r in rows.values())


# --- the fixture is clean enough that nothing lands in the exception bucket, so
# --- the below-threshold path gets its own hand-built case.

def _frames(bank_amount, bank_date, bank_reference, bank_description):
    ledger = pd.DataFrame([{
        "transaction_id": "L1", "date": "2025-07-10", "amount": 500.00,
        "reference": "INV-2001", "vendor": "Northwind Trading Co",
        "description": "Inventory purchase invoice INV-2001",
    }])
    bank = pd.DataFrame([{
        "transaction_id": "B1", "date": bank_date, "amount": bank_amount,
        "reference": bank_reference, "description": bank_description,
    }])
    return ledger, bank


def test_weak_candidate_becomes_an_exception_with_readable_reasons():
    """Inside tolerance on amount and date, but the reference and vendor disagree."""
    matches, exceptions, unmatched = match_transactions(
        *_frames(498.50, "2025-07-14", "INV-2087", "ACH DEBIT ORCHARD DATA SERV INV-2087"))
    assert matches == []
    assert len(exceptions) == 1 and not [u for u in unmatched if u.side == "ledger"]
    exc = exceptions[0]
    assert exc.ledger_id == "L1" and exc.bank_id == "B1"
    assert exc.confidence < AUTO_MATCH_CONFIDENCE
    assert "amount differs by $1.50" in exc.reasons
    assert "date offset 4 days" in exc.reasons
    assert any(r.startswith("reference similarity") for r in exc.reasons)


def test_strong_candidate_clears_the_threshold():
    """Same tolerances, but the invoice number and vendor line up."""
    matches, exceptions, _unmatched = match_transactions(
        *_frames(498.50, "2025-07-14", "PMT INV 2001", "ACH DEBIT NORTHWIND TRADIN INV2001"))
    assert exceptions == [] and len(matches) == 1
    assert matches[0].method == "fuzzy" and matches[0].confidence >= AUTO_MATCH_CONFIDENCE


def test_outside_tolerance_is_unmatched_not_matched():
    for amount, date in ((450.00, "2025-07-10"), (500.00, "2025-07-20")):
        matches, exceptions, unmatched = match_transactions(
            *_frames(amount, date, "INV-2001", "ACH DEBIT NORTHWIND TRADIN INV2001"))
        assert matches == [] and exceptions == []
        assert {u.side for u in unmatched} == {"ledger", "bank"}


def test_engine_leaves_exactly_the_unreferenced_bank_rows_unmatched():
    """The 12 bank rows ground truth never points at are the 12 the engine rejects."""
    _rows, _m, _e, unmatched = score()
    with open(os.path.join(ROOT, "data", "ground_truth.json")) as f:
        truth = json.load(f)
    referenced = {v["bank_transaction_id"] for v in truth.values() if v["bank_transaction_id"]}
    bank = pd.read_csv(os.path.join(ROOT, "data", "bank_statement.csv"))
    stray = set(bank["transaction_id"]) - referenced
    assert {u.transaction_id for u in unmatched if u.side == "bank"} == stray


if __name__ == "__main__":
    report()
