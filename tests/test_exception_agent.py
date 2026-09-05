"""Tests for src/exception_agent.py.

    pytest tests/test_exception_agent.py                 # offline only
    TENSORMUX_API_KEY=... pytest tests/test_exception_agent.py

The offline tests pin the failure contract (retries, graceful degradation,
enrichment shape) with a stubbed transport. The `live` tests are the sanity
check that the model actually discriminates -- one pair it should wave through
and one it should not -- and are skipped when no key is present.
"""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import exception_agent as agent  # noqa: E402
from matcher import Exception_  # noqa: E402

LIVE = pytest.mark.skipif(not os.environ.get(agent.API_KEY_ENV),
                          reason=f"{agent.API_KEY_ENV} not set")

LEDGER = {"transaction_id": "L1", "date": "2025-07-10", "amount": 4820.00,
          "reference": "INV-2001", "vendor": "Northwind Trading Co",
          "description": "Inventory purchase invoice INV-2001"}


def _body(content):
    """An OpenAI-shaped response, with reasoning present to prove we ignore it."""
    return {"choices": [{"message": {"reasoning": "Let me think... approve? no...",
                                     "content": content}}]}


def _stub(monkeypatch, *bodies):
    """Replace the transport with a scripted sequence. Exceptions are raised."""
    calls = []

    def fake_post(prompt, api_key):
        calls.append(prompt)
        result = bodies[min(len(calls) - 1, len(bodies) - 1)]
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(agent, "_post", fake_post)
    monkeypatch.setattr(agent.time, "sleep", lambda _s: None)
    monkeypatch.setenv(agent.API_KEY_ENV, "test-key")
    return calls


GOOD = json.dumps({"explanation": "Same invoice and vendor, four-day settlement lag.",
                   "confidence": 0.91, "recommended_action": "approve"})


# --- parsing --------------------------------------------------------------- #

def test_parses_the_content_field_and_ignores_reasoning(monkeypatch):
    _stub(monkeypatch, _body(GOOD))
    result = agent.explain_match(LEDGER, None, 0.62, "reference mismatch")
    assert result == {"explanation": "Same invoice and vendor, four-day settlement lag.",
                      "confidence": 0.91, "recommended_action": "approve"}


def test_parses_json_wrapped_in_markdown_fences(monkeypatch):
    _stub(monkeypatch, _body(f"Here is my verdict:\n```json\n{GOOD}\n```\n"))
    assert agent.explain_match(LEDGER, None, 0.62)["recommended_action"] == "approve"


def test_confidence_is_clamped_into_zero_one(monkeypatch):
    _stub(monkeypatch, _body(json.dumps(
        {"explanation": "e", "confidence": 4.2, "recommended_action": "reject"})))
    assert agent.explain_match(LEDGER)["confidence"] == 1.0


def test_post_sends_the_documented_request(monkeypatch):
    """The one test that exercises the real transport, not the stub."""
    seen = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def read(self):
            return json.dumps(_body(GOOD)).encode()

    def fake_urlopen(request, timeout=None):
        seen.update(url=request.full_url, headers=dict(request.header_items()),
                    payload=json.loads(request.data), method=request.get_method(),
                    timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv(agent.API_KEY_ENV, "sk-secret")
    assert agent.explain_match(LEDGER, None, 0.5)["recommended_action"] == "approve"

    assert seen["url"] == "https://api.tensormux.com/v1/chat/completions"
    assert seen["method"] == "POST"
    assert seen["headers"]["Authorization"] == "Bearer sk-secret"
    assert seen["headers"]["Content-type"] == "application/json"
    assert seen["timeout"] == agent.TIMEOUT
    assert seen["payload"]["model"] == "glm-4-7-flash"
    assert seen["payload"]["max_tokens"] >= 700, "reasoning eats the budget below this"
    assert [m["role"] for m in seen["payload"]["messages"]] == ["system", "user"]
    assert "Northwind Trading Co" in seen["payload"]["messages"][1]["content"]


# --- retries --------------------------------------------------------------- #

def test_retries_when_content_is_null_then_succeeds(monkeypatch):
    calls = _stub(monkeypatch, _body(None), _body(GOOD))
    assert agent.explain_match(LEDGER, None, 0.5)["recommended_action"] == "approve"
    assert len(calls) == 2


def test_retries_when_content_is_not_json_then_gives_up(monkeypatch):
    calls = _stub(monkeypatch, _body("I cannot help with that."))
    result = agent.explain_match(LEDGER, None, 0.5)
    assert result["recommended_action"] == agent.NEEDS_REVIEW
    assert len(calls) == agent.RETRIES + 1, "should use every retry before giving up"


def test_rejects_an_action_outside_the_enum(monkeypatch):
    _stub(monkeypatch, _body(json.dumps(
        {"explanation": "e", "confidence": 0.5, "recommended_action": "maybe"})))
    assert agent.explain_match(LEDGER)["recommended_action"] == agent.NEEDS_REVIEW


# --- failure never escapes ------------------------------------------------- #

@pytest.mark.parametrize("error", [
    TimeoutError("timed out"),
    OSError("HTTP Error 429: Too Many Requests"),
    ValueError("Expecting value: line 1 column 1"),
])
def test_api_errors_degrade_to_manual_review(monkeypatch, error):
    _stub(monkeypatch, error)
    result = agent.explain_match(LEDGER, None, 0.5, "amount differs by $1.50")
    assert result["recommended_action"] == agent.NEEDS_REVIEW
    assert result["confidence"] == 0.0
    assert "human" in result["explanation"].lower()


def test_missing_api_key_does_not_call_the_api(monkeypatch):
    calls = _stub(monkeypatch, _body(GOOD))
    monkeypatch.delenv(agent.API_KEY_ENV)
    assert agent.explain_match(LEDGER)["recommended_action"] == agent.NEEDS_REVIEW
    assert calls == []


# --- prompt ---------------------------------------------------------------- #

def test_prompt_carries_both_sides_the_score_and_the_reason():
    prompt = agent.build_prompt(
        LEDGER,
        {"transaction_id": "B1", "amount": 4820.00, "description": "ACH DEBIT ORCHARD DATA"},
        0.58,
        ["amount matches exactly", "date offset 4 days"],
    )
    assert "Northwind Trading Co" in prompt and "ACH DEBIT ORCHARD DATA" in prompt
    assert "0.58" in prompt
    assert "amount matches exactly; date offset 4 days" in prompt


def test_prompt_says_so_when_there_is_no_candidate():
    assert "none" in agent.build_prompt(LEDGER, None, 0.0, "").lower()


# --- batch ----------------------------------------------------------------- #

def test_batch_enriches_each_matcher_exception(monkeypatch):
    calls = _stub(monkeypatch, _body(GOOD))
    ledger = pd.DataFrame([LEDGER, {**LEDGER, "transaction_id": "L2", "reference": "INV-2002"}])
    bank = pd.DataFrame([{"transaction_id": "B1", "date": "2025-07-14", "amount": 4818.50,
                          "reference": "INV-2087", "description": "ACH DEBIT ORCHARD DATA"}])
    exceptions = [Exception_("L1", "B1", 0.58, ["amount differs by $1.50"]),
                  Exception_("L2", "B9", 0.41, ["date offset 4 days"])]

    reviewed = agent.explain_exceptions(exceptions, ledger, bank)

    assert len(reviewed) == 2 and len(calls) == 2
    assert [r["ledger_id"] for r in reviewed] == ["L1", "L2"]
    assert reviewed[0]["confidence"] == 0.58, "matcher confidence must survive enrichment"
    assert reviewed[0]["ai_confidence"] == 0.91
    assert reviewed[0]["recommended_action"] == "approve"
    assert reviewed[0]["reasons"] == ["amount differs by $1.50"]
    assert "Northwind" in calls[0] and "ORCHARD" in calls[0]
    # L2's bank_id B9 is not in the frame -- the review still happens, one-sided.
    assert "none" in calls[1].lower()


def test_batch_of_nothing_is_nothing(monkeypatch):
    calls = _stub(monkeypatch, _body(GOOD))
    assert agent.explain_exceptions([], None, None) == [] and calls == []


def test_one_bad_item_does_not_stop_the_batch(monkeypatch):
    _stub(monkeypatch, OSError("down"), OSError("down"), OSError("down"), _body(GOOD))
    reviewed = agent.explain_exceptions(
        [Exception_("L1", "B1", 0.5, []), Exception_("L2", "B2", 0.5, [])],
        pd.DataFrame([LEDGER]), None)
    assert [r["recommended_action"] for r in reviewed] == [agent.NEEDS_REVIEW, "approve"]


# --------------------------------------------------------------------------- #
# live: does the model actually discriminate, or does it approve everything?
#
# The bundled fixture is clean enough that the matcher produces no exceptions at
# all, so the two scenarios below are hand-built: one an auditor would wave
# through, one they would not. If both come back "approve", the model is
# rubber-stamping and the whole agent is worthless.
# --------------------------------------------------------------------------- #

# same vendor, same invoice, same amount to the cent -- only the date moved
TIMING_DELAY = (
    {"transaction_id": "L0042", "date": "2025-07-10", "amount": 4820.00,
     "reference": "INV-2001", "vendor": "Northwind Trading Co",
     "description": "Inventory purchase invoice INV-2001"},
    {"transaction_id": "B0117", "date": "2025-07-14", "amount": 4820.00,
     "reference": "INV 2001", "description": "ACH DEBIT NORTHWIND TRADIN INV2001"},
    0.68,
    "amount matches exactly; date offset 4 days; reference similarity 1.00; "
    "vendor name similarity 0.89",
)

# only the amount agrees: different vendor, different invoice, days apart
SUSPICIOUS_DUPLICATE = (
    {"transaction_id": "L0088", "date": "2025-08-03", "amount": 7342.37,
     "reference": "INV-1301", "vendor": "Harborview Properties",
     "description": "Office rent invoice INV-1301"},
    {"transaction_id": "B0450", "date": "2025-08-08", "amount": 7342.37,
     "reference": "INV-1877", "description": "WIRE OUT SUMMIT AUTO PARTS INV-1877"},
    0.44,
    "amount matches exactly; date offset 5 days; reference similarity 0.25; "
    "vendor name similarity 0.15",
)


@LIVE
def test_live_timing_delay_is_approved():
    result = agent.explain_match(*TIMING_DELAY)
    assert result["recommended_action"] == "approve", result
    assert result["confidence"] >= 0.6, result
    assert len(result["explanation"]) > 20, result


@LIVE
def test_live_duplicate_amount_different_vendor_is_not_approved():
    result = agent.explain_match(*SUSPICIOUS_DUPLICATE)
    assert result["recommended_action"] in ("reject", "investigate"), result
    assert len(result["explanation"]) > 20, result


@LIVE
def test_live_batch_reviews_both_and_splits_them():
    """The real batch path, over both scenarios, in one call."""
    scenarios = [TIMING_DELAY, SUSPICIOUS_DUPLICATE]
    ledger = pd.DataFrame([s[0] for s in scenarios])
    bank = pd.DataFrame([s[1] for s in scenarios])
    exceptions = [Exception_(s[0]["transaction_id"], s[1]["transaction_id"], s[2], [s[3]])
                  for s in scenarios]

    reviewed = agent.explain_exceptions(exceptions, ledger, bank)

    for r in reviewed:
        print(f"\n{r['ledger_id']} -> {r['bank_id']}  {r['recommended_action']} "
              f"({r['ai_confidence']:.2f})\n  {r['explanation']}")
    assert [r["ledger_id"] for r in reviewed] == ["L0042", "L0088"]
    assert [r["confidence"] for r in reviewed] == [0.68, 0.44]
    assert reviewed[0]["recommended_action"] == "approve", reviewed[0]
    assert reviewed[1]["recommended_action"] != "approve", reviewed[1]
