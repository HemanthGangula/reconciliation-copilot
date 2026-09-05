"""AI second opinion on reconciliation exceptions.

The matcher decides what to match; this only explains what it refused to match.
Every call is best-effort: any failure marks the item needs_manual_review and the
pipeline keeps going, because a reconciliation run must never die on a flaky API.

The API key is read from TENSORMUX_API_KEY and only ever leaves this process in
an Authorization header -- it is never logged, returned or put in an error.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.request

LOG = logging.getLogger(__name__)

API_URL = "https://api.tensormux.com/v1/chat/completions"
MODEL = "glm-4-7-flash"
MAX_TOKENS = 700       # the model reasons before answering; too low and content comes back empty
TEMPERATURE = 0
TIMEOUT_SECONDS = 60   # explicit: urllib's default is "wait forever"
MAX_ATTEMPTS = 3       # one call plus the two retries
RETRY_SLEEP_SECONDS = 1.0  # only after a rate limit or server error; tests set this to 0

ACTIONS = ("approve", "investigate", "reject")
NEEDS_REVIEW = "needs_manual_review"  # our sentinel, never something the model may return

SYSTEM_PROMPT = (
    "You are a financial reconciliation exception reviewer. You are shown one ledger "
    "transaction, the bank transaction a matching engine proposed for it (if any), and "
    "why the engine was not confident. Judge whether they are the same real-world "
    "payment.\n"
    "Respond ONLY with valid JSON, no prose and no code fence, in exactly this shape:\n"
    '{"explanation": "<one or two sentences a reviewer can act on>", '
    '"confidence": <number between 0 and 1>, '
    '"recommended_action": "approve" | "investigate" | "reject"}\n'
    "approve = same payment, post it. investigate = plausible but needs a human to "
    "confirm. reject = not the same payment."
)


def _describe(txn, label):
    if not txn:
        return f"{label}: none"
    fields = ("transaction_id", "date", "amount", "reference", "vendor", "description")
    shown = ", ".join(f"{f}={txn[f]!r}" for f in fields if txn.get(f) not in (None, ""))
    return f"{label}: {shown}"


def build_prompt(ledger_txn, bank_txn, match_confidence, reason):
    """The user turn. Says plainly when there is no candidate rather than sending 'None'."""
    if isinstance(reason, (list, tuple)):
        reason = "; ".join(str(r) for r in reason)  # the matcher hands back a list of reasons
    lines = [_describe(ledger_txn, "Ledger transaction")]
    if bank_txn:
        lines.append(_describe(bank_txn, "Proposed bank transaction"))
        lines.append(f"Matching engine confidence: {match_confidence:.2f} (below its auto-post threshold)")
    else:
        lines.append("Proposed bank transaction: none -- no bank row fell inside the "
                     "amount and date tolerances, so there is no candidate to compare against.")
        lines.append("Matching engine confidence: 0.00 (nothing to score)")
    lines.append(f"Why the engine flagged it: {reason}")
    if not bank_txn:
        lines.append("Decide whether this ledger entry is simply unpaid or missing from the "
                     "statement, or whether it looks like an error worth chasing.")
    return "\n".join(lines)


def _post(payload, api_key, timeout):
    """The only place that touches the network. Tests replace this."""
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _content(body):
    """The answer lives in message.content. message.reasoning is scratch work -- never read it."""
    try:
        return body["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return None


def _strip_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    return text.strip()


def parse_content(content):
    """Text -> dict, tolerating a code fence or the model talking around its JSON."""
    if not content or not content.strip():
        raise ValueError("model returned empty content")
    text = _strip_fence(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(f"content was not JSON: {text[:120]!r}")
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            raise ValueError(f"content was not JSON: {text[:120]!r}")


def validate(obj):
    """A well-formed-but-wrong-shaped answer is a failure, not a result.

    Returns only the three contracted keys, so nothing the model invents on the
    side can travel downstream.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"expected a JSON object, got {type(obj).__name__}")
    missing = {"explanation", "confidence", "recommended_action"} - set(obj)
    if missing:
        raise ValueError(f"missing key(s): {', '.join(sorted(missing))}")

    action = obj["recommended_action"]
    if action not in ACTIONS:
        raise ValueError(f"recommended_action {action!r} is not one of {ACTIONS}")

    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(f"confidence must be a number, got {type(confidence).__name__}")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence {confidence} is outside 0-1")

    explanation = obj["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation must be a non-empty string")

    return {"explanation": explanation.strip(),
            "confidence": float(confidence),
            "recommended_action": action}


def _manual(why):
    LOG.warning("exception review fell back to manual: %s", why)
    return {"explanation": f"Automated review unavailable: {why}",
            "confidence": 0.0,
            "recommended_action": NEEDS_REVIEW,
            "error": why}


def explain_exception(ledger_txn, bank_txn, match_confidence, reason,
                      api_key=None, timeout=TIMEOUT_SECONDS):
    """Ask the model to explain one exception. Never raises.

    Returns {"explanation", "confidence", "recommended_action"}, where the action
    is approve/investigate/reject, or needs_manual_review (plus an "error" key)
    if the call or the response could not be trusted.
    """
    api_key = api_key or os.environ.get("TENSORMUX_API_KEY")
    if not api_key:
        return _manual("TENSORMUX_API_KEY is not set")

    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(ledger_txn, bank_txn, match_confidence, reason)},
        ],
    }

    problem = "no attempt was made"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        pause = False
        try:
            body = _post(payload, api_key, timeout)
        except urllib.error.HTTPError as exc:  # non-200, including 429
            problem = f"HTTP {exc.code}"
            pause = exc.code == 429 or exc.code >= 500
        except OSError as exc:  # timeout, DNS, connection reset -- URLError included
            problem = f"request failed ({type(exc).__name__})"
            pause = True
        except json.JSONDecodeError:
            problem = "response body was not JSON"
        else:
            try:
                return validate(parse_content(_content(body)))
            except ValueError as exc:
                problem = str(exc)

        LOG.warning("attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, problem)
        if pause and attempt < MAX_ATTEMPTS and RETRY_SLEEP_SECONDS:
            time.sleep(RETRY_SLEEP_SECONDS)

    return _manual(f"{problem} after {MAX_ATTEMPTS} attempts")


def process_exceptions(exceptions_list, api_key=None, timeout=TIMEOUT_SECONDS):
    """Review each exception in turn and attach the result under "ai_review".

    Each item is a dict: {"ledger": {...}, "bank": {...} or None,
                          "confidence": float, "reason": str}
    which is what you get by joining the matcher's exceptions back to their rows.
    The review is nested rather than merged so the model's confidence cannot be
    mistaken for the engine's.

    ponytail: sequential on purpose. Reconciliation runs are hundreds of rows,
    not millions; parallelise only when a real run is measurably too slow.
    """
    reviewed = []
    for item in exceptions_list:
        try:
            review = explain_exception(item.get("ledger"), item.get("bank"),
                                       item.get("confidence", 0.0), item.get("reason", ""),
                                       api_key=api_key, timeout=timeout)
        except Exception as exc:  # a bad item must not take the whole run down
            LOG.exception("exception review crashed for %s", item.get("ledger", {}).get("transaction_id"))
            review = _manual(f"unexpected {type(exc).__name__}")
        reviewed.append({**item, "ai_review": review})
    return reviewed
