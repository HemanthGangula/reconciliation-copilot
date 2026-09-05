"""Score src/matcher.py against data/ground_truth.json.

    python tests/test_matcher.py     # prints the full report
    pytest tests/test_matcher.py     # same run, asserted thresholds

The expectation for each ledger row is derived from ground_truth.json, never
from a hardcoded id: exact and near_match rows should be auto-matched, messy
rows should be held back as exceptions, ledger-only rows should be reported
unmatched, and a duplicate whose bank row two ledger rows claim should end up
matched once and flagged once.
"""
import json
import os
import sys
from collections import Counter

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from matcher import AUTO_MATCH_CONFIDENCE, match_transactions, reference_score  # noqa: E402

OUTCOMES = ("auto_correct", "auto_wrong", "exception_correct", "exception_wrong", "unmatched")


def _fixture():
    ledger = pd.read_csv(os.path.join(ROOT, "data", "ledger.csv"), keep_default_na=False)
    bank = pd.read_csv(os.path.join(ROOT, "data", "bank_statement.csv"), keep_default_na=False)
    with open(os.path.join(ROOT, "data", "ground_truth.json")) as f:
        truth = json.load(f)
    return ledger, bank, truth


def score():
    """Run the matcher over the fixture and compare every ledger row to truth."""
    ledger, bank, truth = _fixture()
    matches, exceptions, unmatched = match_transactions(ledger, bank)

    got = {m.ledger_id: ("auto", m.bank_id, m.confidence) for m in matches}
    got.update({e.ledger_id: ("exception", e.bank_id, e.confidence) for e in exceptions})
    got.update({u.transaction_id: ("unmatched", None, 0.0) for u in unmatched if u.side == "ledger"})

    # A bank row claimed by two ledger rows is a ledger-side duplicate: one row
    # may legitimately match, the other has to be held back.
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
        shared = bool(want["bank_transaction_id"]) and claims[want["bank_transaction_id"]] > 1
        if not want["bank_transaction_id"]:
            expected = "unmatched"
        elif want["match_type"] == "messy":
            expected = "exception"
        elif shared:
            expected = "one_of_pair"
        else:
            expected = "auto"
        rows[lid] = {
            "match_type": want["match_type"],
            "expected_bank_id": want["bank_transaction_id"],
            "expected": expected,
            "outcome": outcome,
            "got_bank_id": bank_id,
            "confidence": confidence,
        }
    return rows, matches, exceptions, unmatched


def _pct(hits, total):
    return hits / total if total else 0.0


def metrics(rows, exceptions=()):
    by_type = {}
    for r in rows.values():
        by_type.setdefault(r["match_type"], Counter())[r["outcome"]] += 1

    payable = [r for r in rows.values() if r["expected_bank_id"]]
    expect_auto = [r for r in payable if r["expected"] == "auto"]
    messy = [r for r in payable if r["expected"] == "exception"]
    ledger_only = [r for r in rows.values() if r["expected"] == "unmatched"]

    pairs = {}
    for r in rows.values():
        if r["expected"] == "one_of_pair":
            pairs.setdefault(r["expected_bank_id"], []).append(r["outcome"])
    pairs_split = sum(1 for outs in pairs.values() if sum(o == "auto_correct" for o in outs) == 1)

    # Of everything the engine put in the review queue, how much points a
    # reviewer at the right bank row. A queue full of wrong candidates is worse
    # than an empty one.
    queue_correct = sum(1 for r in rows.values() if r["outcome"] == "exception_correct")
    queue_total = len(exceptions)

    return {
        "by_type": by_type,
        "n": len(rows),
        "auto_correct": sum(r["outcome"] == "auto_correct" for r in rows.values()),
        "false_positives": sum(r["outcome"] == "auto_wrong" for r in rows.values()),
        "auto_recall": _pct(sum(r["outcome"] == "auto_correct" for r in expect_auto), len(expect_auto)),
        "messy_flagged": _pct(sum(r["outcome"].startswith("exception") for r in messy), len(messy)),
        "messy_right_candidate": _pct(sum(r["outcome"] == "exception_correct" for r in messy), len(messy)),
        "messy_auto": sum(r["outcome"].startswith("auto") for r in messy),
        "queue_precision": _pct(queue_correct, queue_total),
        "queue_total": queue_total,
        "right_candidate": _pct(
            sum(r["outcome"] in ("auto_correct", "exception_correct") for r in payable), len(payable)),
        "unmatched_recall": _pct(sum(r["outcome"] == "unmatched" for r in ledger_only), len(ledger_only)),
        "n_auto": len(expect_auto),
        "n_messy": len(messy),
        "n_payable": len(payable),
        "n_ledger_only": len(ledger_only),
        "pairs_split": pairs_split,
        "n_pairs": len(pairs),
    }


def report():
    rows, matches, exceptions, unmatched = score()
    m = metrics(rows, exceptions)
    bank_unmatched = [u for u in unmatched if u.side == "bank"]

    print(f"\nRECONCILIATION ACCURACY  ({m['n']} ledger rows, auto-match threshold {AUTO_MATCH_CONFIDENCE})\n")
    head = f"{'ground truth':<14}" + "".join(f"{o:>18}" for o in OUTCOMES)
    print(head)
    print("-" * len(head))
    totals = Counter()
    for mt in sorted(m["by_type"]):
        counts = m["by_type"][mt]
        totals.update(counts)
        print(f"{mt:<14}" + "".join(f"{counts[o]:>18}" for o in OUTCOMES))
    print("-" * len(head))
    print(f"{'total':<14}" + "".join(f"{totals[o]:>18}" for o in OUTCOMES))

    print(f"\nFALSE POSITIVES (auto-matched to the wrong bank row): {m['false_positives']}"
          f"  ({_pct(m['false_positives'], m['n']):.1%})")
    print(f"  of which messy rows posted without review        {m['messy_auto']}")
    print(f"\nauto-matched to the correct bank row       {m['auto_correct']:>3}/{m['n']}"
          f"  ({_pct(m['auto_correct'], m['n']):.1%} of all rows)")
    print(f"  of rows that should auto-match          {m['auto_recall']:.1%}  (n={m['n_auto']})")
    print(f"messy rows held back as exceptions        {m['messy_flagged']:.1%}  (n={m['n_messy']})")
    print(f"  ...pointing at the right bank row       {m['messy_right_candidate']:.1%}")
    print(f"review queue top-candidate accuracy       {m['queue_precision']:.1%}  (n={m['queue_total']})")
    print(f"correct candidate surfaced (auto or exc)  {m['right_candidate']:.1%}  (n={m['n_payable']})")
    print(f"ledger-only rows reported unmatched       {m['unmatched_recall']:.1%}  (n={m['n_ledger_only']})")
    print(f"ledger-side duplicate pairs split 1+1     {m['pairs_split']}/{m['n_pairs']}")
    print(f"\nengine totals: {len(matches)} matched "
          f"({sum(x.method == 'exact' for x in matches)} exact / {sum(x.method == 'fuzzy' for x in matches)} fuzzy), "
          f"{len(exceptions)} exceptions, {len(unmatched)} unmatched "
          f"({len(unmatched) - len(bank_unmatched)} ledger / {len(bank_unmatched)} bank)")

    if exceptions:
        print("\nreview queue (what an auditor would read):")
        for e in sorted(exceptions, key=lambda e: -e.confidence):
            ok = "ok " if rows.get(e.ledger_id, {}).get("outcome") == "exception_correct" else "BAD"
            print(f"  [{ok}] {e.ledger_id} -> {e.bank_id}  {e.confidence:.2f}  |  {'; '.join(e.reasons)}")
    return rows, m


def test_no_false_positives():
    """The regression that matters: nothing posted against the wrong bank row."""
    _rows, m = report()
    assert m["false_positives"] == 0, f"{m['false_positives']} rows auto-matched to the wrong bank row"


def test_clean_rows_auto_match():
    rows, _matches, exceptions, _unmatched = score()
    m = metrics(rows, exceptions)
    assert m["auto_recall"] >= 0.95, m["auto_recall"]


def test_messy_rows_are_held_back_for_review():
    rows, _matches, exceptions, _unmatched = score()
    m = metrics(rows, exceptions)
    assert m["messy_auto"] == 0, f"{m['messy_auto']} ambiguous rows posted without review"
    assert m["messy_flagged"] >= 0.85, m["messy_flagged"]


def test_review_queue_points_at_the_right_row():
    rows, _matches, exceptions, _unmatched = score()
    m = metrics(rows, exceptions)
    assert m["queue_precision"] >= 0.75, m["queue_precision"]


def test_correct_candidate_surfaced():
    rows, _matches, exceptions, _unmatched = score()
    assert metrics(rows, exceptions)["right_candidate"] >= 0.95


def test_ledger_only_rows_reported_unmatched():
    rows, _matches, exceptions, _unmatched = score()
    assert metrics(rows, exceptions)["unmatched_recall"] >= 0.80


def test_every_row_accounted_for_once():
    rows, matches, exceptions, unmatched = score()
    ids = [x.ledger_id for x in matches] + [x.ledger_id for x in exceptions] + \
          [u.transaction_id for u in unmatched if u.side == "ledger"]
    assert len(ids) == len(set(ids)) == len(rows)
    assert not any(r["outcome"] == "missing" for r in rows.values())


def test_engine_leaves_exactly_the_unreferenced_bank_rows_unmatched():
    """The bank rows ground truth never points at are the ones the engine rejects."""
    _rows, _matches, _exceptions, unmatched = score()
    _ledger, bank, truth = _fixture()
    referenced = {v["bank_transaction_id"] for v in truth.values() if v["bank_transaction_id"]}
    stray = set(bank["transaction_id"]) - referenced
    assert {u.transaction_id for u in unmatched if u.side == "bank"} == stray


# --- the below-threshold path also gets hand-built cases, independent of the fixture ---

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



# --- ranking when a decoy competes with the true partner --------------------- #
# Everything above measures thresholding: is this pair good enough to post. These
# two measure ranking: when two bank rows are both inside tolerance, does the
# engine prefer the right one. The fixture cannot ask that question, because no
# messy row happens to have a competitor.

def _decoy_frames(decoy_reference, decoy_description):
    """One ledger row; two bank rows inside tolerance.

    B_TRUE is the real partner but wears its evidence badly: a partial reference,
    a clipped vendor, $4 off and five days late. B_DECOY is a different vendor's
    payment that merely happens to sit closer on amount and date.
    """
    ledger = pd.DataFrame([{
        "transaction_id": "L1", "date": "2025-07-10", "amount": 1000.00,
        "reference": "INV-1023", "vendor": "Northwind Trading Co",
        "description": "Inventory purchase invoice INV-1023",
    }])
    bank = pd.DataFrame([
        {"transaction_id": "B_TRUE", "date": "2025-07-15", "amount": 996.00,
         "reference": "023", "description": "CHECK #4521 NORTHW"},
        {"transaction_id": "B_DECOY", "date": "2025-07-11", "amount": 999.00,
         "reference": decoy_reference, "description": decoy_description},
    ])
    return ledger, bank


@pytest.mark.parametrize("decoy_reference, why", [
    ("INV-7742", "an unrelated invoice number"),
    ("INV-1024", "the next invoice number, one digit off"),
    ("INV-1032", "the same digits transposed"),
    ("PO54210", "the bank's own PO number"),
])
def test_a_decoy_never_beats_the_true_partner(decoy_reference, why):
    """Better amount and date must not outweigh an identifier that disagrees.

    Invoice numbers are issued sequentially, so the one-digit-off decoy is the
    single most likely wrong pair in a real ledger -- it must lose, not nearly win.
    """
    matches, exceptions, unmatched = match_transactions(
        *_decoy_frames(decoy_reference, f"ACH DEBIT MERIDIAN SOFTWAR {decoy_reference}"))
    assert matches == [], f"{why}: nothing here is strong enough to post unreviewed"
    assert [e.bank_id for e in exceptions] == ["B_TRUE"], f"{why}: the queue must point at the real partner"
    assert [u.transaction_id for u in unmatched if u.side == "bank"] == ["B_DECOY"]


def test_reference_score_treats_an_invoice_number_as_an_identifier():
    """The three rules, stated as the auditor would check them."""
    assert reference_score("INV-1023", "INV-1023") == 1.0
    assert reference_score("INV-1023", "PMT INV 1023") == 1.0   # reformatted, same identifier
    assert reference_score("INV-1023", "023") == 0.85           # truncated reference
    assert reference_score("INV-1023", "INV-1024") == 0.20      # a different invoice
    assert reference_score("INV-1023", "INV-1032") == 0.20      # transposed digits
    assert reference_score("INV-1023", "3") == 0.20             # too short to be a truncation
    assert reference_score("INV-1023", "") == 0.0               # nothing to compare


if __name__ == "__main__":
    report()
