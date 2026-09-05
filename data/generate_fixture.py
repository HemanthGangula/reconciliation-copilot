"""Deterministic reconciliation test fixture.

Writes ledger.csv (150 rows), bank_statement.csv (150 rows) and ground_truth.json
(150 entries keyed by ledger transaction_id) next to this file.

    python data/generate_fixture.py
"""
import csv, json, os, random
from datetime import date, timedelta

random.seed(20250701)

HERE = os.path.dirname(os.path.abspath(__file__))
Q_START = date(2025, 7, 1)
Q_DAYS = 92  # 2025-07-01 .. 2025-09-30

# ledger rows per match_type, plus bank-only strays. Sums to 150 on each side.
N_EXACT, N_NEAR, N_DUP_BANK, N_DUP_LEDGER, N_LEDGER_ONLY, N_BANK_ONLY = 105, 23, 5, 5, 7, 7

VENDORS = [
    ("Northwind Trading Co", "Inventory purchase"),
    ("Acme Industrial Supply", "Maintenance parts"),
    ("Cascade Office Supplies", "Office supplies"),
    ("Bluepeak Logistics LLC", "Freight"),
    ("Meridian Software Inc", "Software subscription"),
    ("Harborview Properties", "Office rent"),
    ("Sterling Legal Partners", "Legal fees"),
    ("Pacific Power & Light", "Utilities"),
    ("Ridgeline Staffing Group", "Contract labor"),
    ("Copperfield Printing", "Marketing materials"),
    ("Vantage Insurance Brokers", "Insurance premium"),
    ("Ironclad Security Systems", "Facility security"),
    ("Greenfield Janitorial", "Cleaning services"),
    ("Lakeshore Telecom", "Phone and internet"),
    ("Summit Equipment Rental", "Equipment rental"),
    ("Delta Freight Systems", "Freight"),
    ("Orchard Data Services", "Data hosting"),
    ("Fairmont Catering Co", "Staff catering"),
]

refs = list(range(1000, 1400))
random.shuffle(refs)
refs = iter(refs)


def workday(d):
    return d + timedelta(days={5: 2, 6: 1}.get(d.weekday(), 0))


def txn():
    vendor, category = random.choice(VENDORS)
    n = next(refs)
    return {
        "vendor": vendor,
        "category": category,
        "ref": f"INV-{n}",
        "num": n,
        "date": workday(Q_START + timedelta(days=random.randrange(Q_DAYS))),
        "amount": round(random.uniform(45, 18500), 2),
    }


def ledger_row(t):
    return {
        "date": t["date"].isoformat(),
        "amount": f"{t['amount']:.2f}",
        "reference": t["ref"],
        "vendor": t["vendor"],
        "description": f"{t['category']} invoice {t['ref']}",
    }


def bank_row(t, ref=None, days=0, amount=None):
    ref = t["ref"] if ref is None else ref
    trunc = "".join(c for c in t["vendor"].upper() if c.isalnum() or c == " ")[:18].strip()
    style = random.choice(("ACH", "ACH", "WIRE", "CHECK"))
    if style == "ACH":
        desc = f"ACH DEBIT {trunc} {ref}"
    elif style == "WIRE":
        desc = f"WIRE OUT {trunc}"
    else:
        desc = f"CHECK #{random.randrange(4000, 5200)} {trunc}"
    return {
        "date": workday(t["date"] + timedelta(days=days)).isoformat(),
        "amount": f"{t['amount'] if amount is None else amount:.2f}",
        "reference": ref,
        "description": desc,
    }


def reformat(t):
    return random.choice((f"INV{t['num']}", f"PMT INV {t['num']}", f"{t['num']}", f"INV #{t['num']}"))


pairs, bank_rows = [], []  # pairs: (ledger dict, bank dict or None, match_type)

for _ in range(N_EXACT):
    t = txn()
    b = bank_row(t, days=random.choice((0, 0, 1)))
    bank_rows.append(b)
    pairs.append((ledger_row(t), b, "exact"))

for i in range(N_NEAR):
    t = txn()
    kind = i % 3
    if kind == 0:  # posted 2-3 days later
        b = bank_row(t, days=random.choice((2, 3)))
    elif kind == 1:  # rounding difference or bank fee
        drift = random.choice((-0.03, -0.01, 0.02, 0.25, 1.50, 2.00, 0.75))
        b = bank_row(t, days=random.choice((0, 1)), amount=round(t["amount"] + drift, 2))
    else:  # reference written differently
        b = bank_row(t, ref=reformat(t), days=random.choice((0, 1)))
    bank_rows.append(b)
    pairs.append((ledger_row(t), b, "near_match"))

for _ in range(N_DUP_BANK):  # paid twice on the bank side
    t = txn()
    first = bank_row(t, days=random.choice((0, 1)))
    second = bank_row(t, ref=random.choice((t["ref"], reformat(t))), days=random.choice((2, 5, 9)))
    bank_rows += [first, second]
    pairs.append((ledger_row(t), first, "duplicate"))

for _ in range(N_DUP_LEDGER):  # invoice keyed twice in the ledger, paid once
    t = txn()
    b = bank_row(t, days=random.choice((0, 1)))
    bank_rows.append(b)
    pairs.append((ledger_row(t), b, "duplicate"))
    pairs.append((ledger_row(t), b, "duplicate"))

for _ in range(N_LEDGER_ONLY):
    pairs.append((ledger_row(txn()), None, "unmatched"))

for _ in range(N_BANK_ONLY):
    bank_rows.append(bank_row(txn(), days=random.randrange(3)))

random.shuffle(pairs)
random.shuffle(bank_rows)

for i, b in enumerate(bank_rows, 1):
    b["transaction_id"] = f"B{i:04d}"
for i, (l, _b, _m) in enumerate(pairs, 1):
    l["transaction_id"] = f"L{i:04d}"

ground_truth = {
    l["transaction_id"]: {"bank_transaction_id": b["transaction_id"] if b else None, "match_type": m}
    for l, b, m in pairs
}

LEDGER_COLS = ["transaction_id", "date", "amount", "reference", "vendor", "description"]
BANK_COLS = ["transaction_id", "date", "amount", "reference", "description"]


def write_csv(name, cols, rows):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r[c] for c in cols} for r in rows)


write_csv("ledger.csv", LEDGER_COLS, [l for l, _b, _m in pairs])
write_csv("bank_statement.csv", BANK_COLS, sorted(bank_rows, key=lambda r: (r["date"], r["transaction_id"])))
with open(os.path.join(HERE, "ground_truth.json"), "w") as f:
    json.dump(ground_truth, f, indent=2, sort_keys=True)
    f.write("\n")

# ponytail: asserts instead of a test file -- the fixture is only useful if these hold
counts = {}
for e in ground_truth.values():
    counts[e["match_type"]] = counts.get(e["match_type"], 0) + 1
bank_ids = {b["transaction_id"] for b in bank_rows}
assert len(pairs) == 150 and len(bank_rows) == 150, (len(pairs), len(bank_rows))
assert len(ground_truth) == 150, len(ground_truth)
assert all(e["bank_transaction_id"] in bank_ids for e in ground_truth.values() if e["bank_transaction_id"])
assert counts == {"exact": 105, "near_match": 23, "duplicate": 15, "unmatched": 7}, counts
for mt, n in sorted(counts.items()):
    print(f"{mt:11} {n:4}  {n / 150:6.1%}")
