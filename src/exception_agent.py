"""AI reviewer for the exceptions the matcher cannot decide on its own.

src/matcher.py decides *whether* a pair is confident enough to post. This module
explains *why* a pair is ambiguous, in words an auditor can read, and recommends
approve / investigate / reject.

Two deliberate constraints:

* stdlib HTTP only -- the pipeline gains no new dependency for one POST.
* nothing raises. A timeout, a rate limit or junk JSON degrades that one item to
  "needs_manual_review"; a flaky API call must never take down a reconciliation
  run that has already produced good matches.
"""
import json
import logging
import os
import time
import urllib.request

API_URL = "https://api.tensormux.com/v1/chat/completions"
MODEL = "glm-4-7-flash"

# This model emits its chain of thought before the answer. Budget too small and
# the reasoning eats the whole allowance, leaving `content` empty -- 700 is the
# floor, 900 buys headroom for a wordy explanation.
MAX_TOKENS = 900
TIMEOUT = 60          # seconds; reasoning models are slow, not broken
RETRIES = 2           # attempts = RETRIES + 1
TEMPERATURE = 0.2     # a money path wants repeatable verdicts, not variety

ACTIONS = ("approve", "investigate", "reject")
NEEDS_REVIEW = "needs_manual_review"
API_KEY_ENV = "TENSORMUX_API_KEY"

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a financial reconciliation exception reviewer at an accounting firm.

A deterministic matcher paired a ledger transaction with a candidate bank transaction but was not confident enough to post it automatically. Judge the pair.

recommended_action:
  "approve"     - the differences are benign (settlement or timing delay, bank fee, truncated vendor name, reformatted invoice reference) and this is almost certainly the same payment.
  "investigate" - plausible, but the evidence is thin, partial or contradictory.
  "reject"      - the evidence points at two different payments: a different vendor, a different invoice, or a likely duplicate.

Be skeptical. Approving a wrong pair puts a bad entry in the books, so a matching amount alone is not evidence -- vendor and invoice reference must agree too.

Respond with ONLY a JSON object. No prose, no markdown fences:
{"explanation": "<2-3 sentences an auditor can act on>", "confidence": <float between 0 and 1>, "recommended_action": "approve" | "investigate" | "reject"}"""


# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #

def _describe(row, label):
    """One line per side, whatever columns that side happens to carry."""
    if row is None:
        return f"{label}: none -- the matcher found no candidate on this side"
    fields = ", ".join(f"{k}={v}" for k, v in dict(row).items() if str(v).strip() not in ("", "nan"))
    return f"{label}: {fields}"


def build_prompt(ledger, bank, confidence, reason):
    if isinstance(reason, (list, tuple)):
        reason = "; ".join(str(r) for r in reason)
    return (
        f"{_describe(ledger, 'LEDGER TRANSACTION')}\n"
        f"{_describe(bank, 'CANDIDATE BANK TRANSACTION')}\n\n"
        f"Matcher confidence: {float(confidence):.2f} (below the auto-post threshold)\n"
        f"Why it did not match cleanly: {reason or 'not stated'}"
    )


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

def _post(prompt, api_key):
    """One POST. Raises on anything that is not a 2xx JSON body."""
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read())


def _content(body):
    """Only `content` -- `reasoning` is the model thinking aloud, never the answer."""
    message = body["choices"][0]["message"]
    return (message.get("content") or "").strip()


def _parse(content):
    """Extract the verdict, tolerating markdown fences or a stray closing note."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in content")
    data = json.loads(content[start:end + 1])

    action = str(data.get("recommended_action", "")).strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"recommended_action {action!r} is not one of {ACTIONS}")
    explanation = str(data.get("explanation", "")).strip()
    if not explanation:
        raise ValueError("explanation is empty")
    return {
        "explanation": explanation,
        "confidence": min(1.0, max(0.0, float(data.get("confidence", 0.0)))),
        "recommended_action": action,
    }


def _needs_review(message):
    log.error("exception agent falling back to manual review: %s", message)
    return {
        "explanation": f"AI review unavailable ({message}). Route to a human reviewer.",
        "confidence": 0.0,
        "recommended_action": NEEDS_REVIEW,
    }


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def explain_match(ledger, bank=None, confidence=0.0, reason="", api_key=None, retries=RETRIES):
    """Review one ambiguous pair.

    Returns {"explanation", "confidence", "recommended_action"} always -- on any
    failure the action is "needs_manual_review" rather than an exception.
    """
    api_key = api_key or os.environ.get(API_KEY_ENV)
    if not api_key:
        return _needs_review(f"{API_KEY_ENV} is not set")

    prompt = build_prompt(ledger, bank, confidence, reason)
    failure = "no attempts made"
    for attempt in range(retries + 1):
        try:
            content = _content(_post(prompt, api_key))
            if not content:
                raise ValueError("content field was null or empty")
            return _parse(content)
        # ponytail: bare Exception on purpose -- the contract is "never crash the
        # pipeline", and a hand-written tuple always misses one transport quirk.
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            log.warning("exception agent attempt %d/%d failed: %s", attempt + 1, retries + 1, failure)
            if attempt < retries:
                time.sleep(1 + attempt)
    return _needs_review(failure)


def _index(rows):
    """transaction_id -> row, from a DataFrame or a list of dicts."""
    if rows is None:
        return {}
    records = rows.to_dict("records") if hasattr(rows, "to_dict") else rows
    return {str(r["transaction_id"]): r for r in records}


def explain_exceptions(exceptions, ledger_df=None, bank_df=None, **kwargs):
    """Review the matcher's whole exception list, one API call per item.

    Each result is the original exception plus `explanation`, `ai_confidence`
    and `recommended_action`. The matcher's own `confidence` is left untouched.
    """
    ledger_rows, bank_rows = _index(ledger_df), _index(bank_df)
    reviewed = []
    for exception in exceptions:
        item = dict(exception) if isinstance(exception, dict) else dict(vars(exception))
        review = explain_match(
            ledger_rows.get(str(item.get("ledger_id"))),
            bank_rows.get(str(item.get("bank_id"))),
            item.get("confidence", 0.0),
            item.get("reasons", ""),
            **kwargs,
        )
        reviewed.append({
            **item,
            "explanation": review["explanation"],
            "ai_confidence": review["confidence"],
            "recommended_action": review["recommended_action"],
        })
    return reviewed
