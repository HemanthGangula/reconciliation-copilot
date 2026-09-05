"""Tests for src/exception_agent.py.

The default run never touches the network: every HTTP call is mocked, so this
passes in CI with no key and no spend. The cases that actually exercise the
model's judgement are gated behind RUN_LIVE_API_TESTS=1 and skip cleanly.

    pytest tests/test_exception_agent.py                    # mocked only
    RUN_LIVE_API_TESTS=1 pytest tests/test_exception_agent.py  # + real API
"""
import json
import os
import socket
import sys
import urllib.error

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import exception_agent as agent  # noqa: E402

FAKE_KEY = "sk-test-not-a-real-key-0000"

LEDGER = {"transaction_id": "L0007", "date": "2025-07-10", "amount": "1250.00",
          "reference": "INV-1023", "vendor": "Acme Industrial Supply",
          "description": "Maintenance parts invoice INV-1023"}
BANK = {"transaction_id": "B0031", "date": "2025-07-14", "amount": "1250.00",
        "reference": "INV1023", "description": "ACH DEBIT ACME INDUSTRIAL INV1023"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch, request):
    """No real key, no real sleeping, and no socket a mocked test could escape through.

    The socket block is the part that matters: a future test that forgets to mock
    fails loudly here instead of quietly spending money against the live API.
    """
    monkeypatch.setenv("TENSORMUX_API_KEY", FAKE_KEY)
    monkeypatch.setattr(agent, "RETRY_SLEEP_SECONDS", 0)
    if request.node.name.startswith("test_live_"):
        return  # the opt-in cases are supposed to reach the network

    def blocked(*args, **kwargs):
        raise AssertionError("this test tried to open a socket -- mock the HTTP layer")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


def reply(content, reasoning="the model's private scratch work"):
    """A response body shaped like the API's."""
    return {"choices": [{"message": {"content": content, "reasoning": reasoning}}]}


def good_json(action="approve", confidence=0.9):
    return json.dumps({"explanation": "Same invoice, bank posted four days later.",
                       "confidence": confidence, "recommended_action": action})


def fake_post(*responses):
    """Replace the HTTP layer; each call returns (or raises) the next item."""
    calls = []

    def _post(payload, api_key, timeout):
        calls.append({"payload": payload, "api_key": api_key, "timeout": timeout})
        item = responses[min(len(calls) - 1, len(responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    _post.calls = calls
    return _post


def run(monkeypatch, *responses, ledger=LEDGER, bank=BANK, confidence=0.53, reason="date offset 4 days"):
    poster = fake_post(*responses)
    monkeypatch.setattr(agent, "_post", poster)
    result = agent.explain_exception(ledger, bank, confidence, reason)
    return result, poster.calls


# --- the happy path and the shapes the model actually emits ------------------ #

def test_valid_response_is_returned_with_only_the_contracted_keys(monkeypatch):
    result, calls = run(monkeypatch, reply(good_json()))
    assert len(calls) == 1
    assert result == {"explanation": "Same invoice, bank posted four days later.",
                      "confidence": 0.9, "recommended_action": "approve"}


def test_fenced_json_is_parsed(monkeypatch):
    result, _ = run(monkeypatch, reply(f"```json\n{good_json('investigate')}\n```"))
    assert result["recommended_action"] == "investigate"


def test_json_wrapped_in_prose_is_parsed(monkeypatch):
    content = f"Here is my assessment.\n{good_json('reject', 0.2)}\nLet me know if you need more."
    result, _ = run(monkeypatch, reply(content))
    assert result["recommended_action"] == "reject" and result["confidence"] == 0.2


def test_reasoning_field_is_never_used_as_the_answer(monkeypatch):
    """content is the answer; reasoning is scratch work, even when content is null."""
    result, calls = run(monkeypatch, reply(None, reasoning=good_json("approve")))
    assert len(calls) == agent.MAX_ATTEMPTS
    assert result["recommended_action"] == agent.NEEDS_REVIEW


# --- retries ---------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [None, "", "   "])
def test_empty_content_retries_then_succeeds(monkeypatch, bad):
    result, calls = run(monkeypatch, reply(bad), reply(bad), reply(good_json()))
    assert len(calls) == 3
    assert result["recommended_action"] == "approve"


def test_garbage_content_exhausts_retries(monkeypatch):
    result, calls = run(monkeypatch, reply("I cannot answer that."))
    assert len(calls) == agent.MAX_ATTEMPTS
    assert result["recommended_action"] == agent.NEEDS_REVIEW
    assert "error" in result


# --- well-formed JSON of the wrong shape must not flow downstream ------------ #

@pytest.mark.parametrize("payload", [
    {"explanation": "ok", "confidence": 0.5},                                    # missing action
    {"confidence": 0.5, "recommended_action": "approve"},                        # missing explanation
    {"explanation": "ok", "recommended_action": "approve"},                      # missing confidence
    {"explanation": "ok", "confidence": 0.5, "recommended_action": "escalate"},  # action not allowed
    {"explanation": "ok", "confidence": 1.5, "recommended_action": "approve"},   # out of range
    {"explanation": "ok", "confidence": -0.1, "recommended_action": "approve"},  # out of range
    {"explanation": "ok", "confidence": "0.8", "recommended_action": "approve"}, # string, not number
    {"explanation": "ok", "confidence": True, "recommended_action": "approve"},  # bool is not a number
    {"explanation": "", "confidence": 0.5, "recommended_action": "approve"},     # empty explanation
    {"explanation": None, "confidence": 0.5, "recommended_action": "approve"},   # wrong type
    ["approve", 0.9],                                                            # not an object
])
def test_wrong_shaped_json_is_treated_as_a_failure(monkeypatch, payload):
    result, calls = run(monkeypatch, reply(json.dumps(payload)))
    assert len(calls) == agent.MAX_ATTEMPTS, "a wrong shape should be retried like a parse failure"
    assert result["recommended_action"] == agent.NEEDS_REVIEW


def test_extra_keys_are_dropped(monkeypatch):
    body = json.dumps({"explanation": "ok", "confidence": 0.5, "recommended_action": "approve",
                       "post_it": True, "internal_note": "ignore me"})
    result, _ = run(monkeypatch, reply(body))
    assert set(result) == {"explanation", "confidence", "recommended_action"}


# --- transport failures ------------------------------------------------------ #

def test_timeout_does_not_crash(monkeypatch):
    result, calls = run(monkeypatch, TimeoutError("timed out"))
    assert len(calls) == agent.MAX_ATTEMPTS
    assert result["recommended_action"] == agent.NEEDS_REVIEW


def test_rate_limit_is_retried_then_falls_back(monkeypatch):
    err = urllib.error.HTTPError(agent.API_URL, 429, "Too Many Requests", None, None)
    result, calls = run(monkeypatch, err)
    assert len(calls) == agent.MAX_ATTEMPTS
    assert "429" in result["error"]


def test_rate_limit_then_success(monkeypatch):
    err = urllib.error.HTTPError(agent.API_URL, 429, "Too Many Requests", None, None)
    result, calls = run(monkeypatch, err, reply(good_json()))
    assert len(calls) == 2 and result["recommended_action"] == "approve"


@pytest.mark.parametrize("code", [400, 401, 500, 503])
def test_non_200_does_not_crash(monkeypatch, code):
    err = urllib.error.HTTPError(agent.API_URL, code, "boom", None, None)
    result, _ = run(monkeypatch, err)
    assert result["recommended_action"] == agent.NEEDS_REVIEW
    assert str(code) in result["error"]


def test_connection_error_does_not_crash(monkeypatch):
    result, _ = run(monkeypatch, urllib.error.URLError("name resolution failed"))
    assert result["recommended_action"] == agent.NEEDS_REVIEW


def test_missing_api_key_returns_manual_review_without_calling(monkeypatch):
    monkeypatch.delenv("TENSORMUX_API_KEY", raising=False)
    poster = fake_post(reply(good_json()))
    monkeypatch.setattr(agent, "_post", poster)
    result = agent.explain_exception(LEDGER, BANK, 0.5, "whatever")
    assert poster.calls == []
    assert result["recommended_action"] == agent.NEEDS_REVIEW


# --- the key must not leak --------------------------------------------------- #

def test_api_key_never_appears_in_output_or_logs(monkeypatch, caplog):
    caplog.set_level("DEBUG")
    err = urllib.error.HTTPError(agent.API_URL, 401, "Unauthorized", None, None)
    result, _ = run(monkeypatch, err)
    assert FAKE_KEY not in json.dumps(result)
    assert FAKE_KEY not in caplog.text


# --- the prompt -------------------------------------------------------------- #

def test_prompt_without_a_bank_row_never_says_none():
    prompt = agent.build_prompt(LEDGER, None, 0.0, "no bank row within amount and date tolerance")
    assert "None" not in prompt
    assert "no bank row fell inside" in prompt
    assert LEDGER["reference"] in prompt


def test_prompt_accepts_the_matchers_list_of_reasons():
    """matcher.Exception_.reasons is a list; it must not render as a Python repr."""
    prompt = agent.build_prompt(LEDGER, BANK, 0.53,
                                ["amount differs by $1.40", "date offset 4 days"])
    assert "amount differs by $1.40; date offset 4 days" in prompt
    assert "[" not in prompt and "'" not in prompt.split("Why the engine flagged it:")[1]


def test_prompt_with_a_bank_row_shows_both_sides():
    prompt = agent.build_prompt(LEDGER, BANK, 0.53, "date offset 4 days")
    assert BANK["transaction_id"] in prompt and LEDGER["transaction_id"] in prompt
    assert "0.53" in prompt


def test_missing_bank_row_still_reaches_the_model(monkeypatch):
    result, calls = run(monkeypatch, reply(good_json("investigate")), bank=None, confidence=0.0)
    assert result["recommended_action"] == "investigate"
    assert "none" in calls[0]["payload"]["messages"][1]["content"]


# --- the request itself, mocked one layer lower ------------------------------ #

class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_request_is_built_correctly(monkeypatch):
    seen = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["method"] = request.get_method()
        seen["headers"] = dict(request.header_items())
        seen["body"] = json.loads(request.data.decode())
        seen["timeout"] = timeout
        return _FakeResponse(reply(good_json()))

    monkeypatch.setattr(agent.urllib.request, "urlopen", fake_urlopen)
    result = agent.explain_exception(LEDGER, BANK, 0.53, "date offset 4 days", timeout=12)

    assert result["recommended_action"] == "approve"
    assert seen["url"] == "https://api.tensormux.com/v1/chat/completions"
    assert seen["method"] == "POST"
    assert seen["timeout"] == 12, "the timeout must be explicit, never urllib's default"
    assert seen["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert seen["body"]["model"] == "glm-4-7-flash"
    assert seen["body"]["max_tokens"] >= 700, "too low and this model returns empty content"
    assert [m["role"] for m in seen["body"]["messages"]] == ["system", "user"]


# --- process_exceptions ------------------------------------------------------ #

def test_process_exceptions_enriches_every_item(monkeypatch):
    monkeypatch.setattr(agent, "_post", fake_post(reply(good_json())))
    items = [{"ledger": LEDGER, "bank": BANK, "confidence": 0.53, "reason": "date offset 4 days"},
             {"ledger": LEDGER, "bank": None, "confidence": 0.0, "reason": "no candidate"}]
    out = agent.process_exceptions(items)
    assert len(out) == 2
    assert all(o["ai_review"]["recommended_action"] == "approve" for o in out)
    assert out[0]["confidence"] == 0.53, "the engine's confidence must survive the model's"
    assert out[0]["ai_review"]["confidence"] == 0.9


def test_process_exceptions_survives_one_item_failing(monkeypatch):
    calls = []

    def flaky(payload, api_key, timeout):
        calls.append(payload)
        if "L0007" in payload["messages"][1]["content"]:
            raise TimeoutError("timed out")
        return reply(good_json())

    monkeypatch.setattr(agent, "_post", flaky)
    other = {**LEDGER, "transaction_id": "L0099"}
    out = agent.process_exceptions([
        {"ledger": LEDGER, "bank": BANK, "confidence": 0.5, "reason": "x"},
        {"ledger": other, "bank": BANK, "confidence": 0.5, "reason": "y"},
    ])
    assert out[0]["ai_review"]["recommended_action"] == agent.NEEDS_REVIEW
    assert out[1]["ai_review"]["recommended_action"] == "approve"


def test_process_exceptions_survives_a_malformed_item(monkeypatch):
    monkeypatch.setattr(agent, "_post", fake_post(reply(good_json())))
    out = agent.process_exceptions([{"ledger": None, "bank": None}])
    assert len(out) == 1 and "ai_review" in out[0]


# --- live model judgement, opt-in only --------------------------------------- #

live = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_API_TESTS") != "1" or not os.environ.get("TENSORMUX_API_KEY"),
    reason="set RUN_LIVE_API_TESTS=1 and TENSORMUX_API_KEY to exercise the real model",
)

TIMING_DELAY = (
    {"transaction_id": "L0007", "date": "2025-07-10", "amount": "1250.00", "reference": "INV-1023",
     "vendor": "Acme Industrial Supply", "description": "Maintenance parts invoice INV-1023"},
    {"transaction_id": "B0031", "date": "2025-07-14", "amount": "1250.00", "reference": "INV1023",
     "description": "ACH DEBIT ACME INDUSTRIAL INV1023"},
    0.63, "amount matches exactly; date offset 4 days; reference similarity 1.00",
)

SUSPICIOUS = (
    {"transaction_id": "L0042", "date": "2025-08-04", "amount": "8400.00", "reference": "INV-1188",
     "vendor": "Harborview Properties", "description": "Office rent invoice INV-1188"},
    {"transaction_id": "B0077", "date": "2025-08-06", "amount": "8400.00", "reference": "PO54210",
     "description": "WIRE OUT MERIDIAN SOFTWARE"},
    0.44, "amount matches exactly; date offset 2 days; reference similarity 0.15; vendor name similarity 0.05",
)


@live
def test_live_timing_delay_is_approved():
    """Same invoice, same amount, bank posted four days later -- this is not fraud."""
    result = agent.explain_exception(*TIMING_DELAY)
    print("\nlive timing-delay case ->", json.dumps(result, indent=2))
    assert result["recommended_action"] == "approve"


@live
def test_live_suspicious_case_is_not_rubber_stamped():
    """Same amount, completely different vendor -- a reviewer must look at this."""
    result = agent.explain_exception(*SUSPICIOUS)
    print("\nlive suspicious case ->", json.dumps(result, indent=2))
    assert result["recommended_action"] in ("investigate", "reject")


@live
def test_live_missing_bank_row_is_handled():
    ledger, _bank, _conf, _reason = TIMING_DELAY
    result = agent.explain_exception(ledger, None, 0.0, "no bank row within amount and date tolerance")
    print("\nlive no-candidate case ->", json.dumps(result, indent=2))
    assert result["recommended_action"] in agent.ACTIONS
