"""Query harness for the revised Majority Illusion experiment.

For every entity x ratio x model condition the harness:
  1. sends one byte-identical primary prompt three times;
  2. classifies each answer MAJ/MIN/COM/FLAG/OTHER/UNSCORED;
  3. records modal-category self-consistency as a separate diagnostic;
  4. asks the same model for a post-hoc 0-100 probability that the modal answer
     gives the independently labeled true value.

Most entities also report a 0-100 probability inside each primary answer. A
deterministic, domain-stratified 10-entity control omits that inline request so
the study can measure whether confidence elicitation changes answer behavior.

The default roster is three explicit model slots in one invocation:
  - gemini:   gemini-3.5-flash via the native google-genai SDK;
  - deepseek: deepseek/deepseek-v4-flash via OpenRouter; and
  - claude:   anthropic/claude-haiku-4.5 via OpenRouter.

DeepSeek and Claude no longer share a mutable model slot. The raw CSV records
the slot, requested model id, and API-returned model id; a cross-family mismatch
is a hard row error. The condition CSV groups by model_id so the two OpenRouter
models cannot be silently pooled.

Every billed/mock call is flushed to the raw CSV. A second condition-level CSV
stores response categories, modal answer, raw post-hoc probability, and
self-consistency.

Config is read from the environment. Gemini has three auth modes:
  GEMINI_USE_VERTEX          1/true to use Vertex AI (either mode below) instead
                              of the AI Studio endpoint.
  -- Vertex PROJECT auth (paid tier, recommended) --
  GEMINI_VERTEX_PROJECT      GCP project id. When set (with GEMINI_USE_VERTEX),
                              the client uses project+location auth and does NOT
                              use GEMINI_API_KEY. Lifts Express Mode's ~5 req/min
                              free cap to the project's paid quota.
  GEMINI_VERTEX_LOCATION     default "global" (gemini-3.5-flash 404s in
                              us-central1 for our project -- use global).
  GOOGLE_APPLICATION_CREDENTIALS  path to the service-account JSON key (or rely
                              on ADC via `gcloud auth application-default login`).
  -- Vertex EXPRESS mode (limited free quota) --
  GEMINI_API_KEY             used only when GEMINI_USE_VERTEX is set but
                              GEMINI_VERTEX_PROJECT is NOT. Express Mode takes
                              only the api key; project/location + api_key
                              together are rejected by the SDK.
  -- AI Studio (default, GEMINI_USE_VERTEX unset) --
  GEMINI_API_KEY             a plain AI Studio key.
  -- OpenRouter models --
  OPENROUTER_API_KEY         shared gateway key for DeepSeek and Claude.
  DEEPSEEK_MODEL             optional DeepSeek model-id override.
  CLAUDE_MODEL               optional Claude model-id override.
  OPENROUTER_MODEL           legacy variable; deliberately ignored.
A .env file in the repo root is auto-loaded if present (.env is gitignored, so
secrets are never committed).

Usage:
    python harness/run_experiment.py --mock --entities 3   # pipeline test, no API calls
    python harness/run_experiment.py --entities 3          # small live pilot
    python harness/run_experiment.py                       # full current dataset
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
import uuid
import random as py_random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "entities.json"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"  # current frontier Flash (GA 2026-05-19)
DEFAULT_DEEPSEEK_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_CLAUDE_MODEL = "anthropic/claude-haiku-4.5"
# Kept as a compatibility import for token/probe utilities. Production model
# selection now uses the explicit DeepSeek and Claude constants above.
DEFAULT_OPENROUTER_MODEL = DEFAULT_DEEPSEEK_MODEL

DEFAULT_SAMPLES_PER_CONDITION = 3
DEFAULT_NO_INLINE_CONFIDENCE_ENTITIES = 10

# Models "think" before answering, and those reasoning/thinking tokens count
# against the output budget -- a tight cap gets consumed by reasoning and
# truncates the JSON mid-object. Give generous headroom well above the ~30
# tokens the visible JSON answer needs.
GEMINI_MAX_OUTPUT_TOKENS = 2048
OPENROUTER_MAX_TOKENS = 2048
MAX_RETRIES = 4  # exponential-backoff attempts on 429/5xx/connection errors

# Reasoning-depth controls, one per model, verified against each SDK/API before
# use rather than assumed (see UPDATES.md for how each was checked):
#   - Gemini: native google-genai ThinkingConfig.thinking_level. Confirmed via
#     the installed SDK's pydantic model_fields that this field exists and
#     ThinkingLevel.MEDIUM is a real enum member (not guessed).
#   - OpenRouter (explicit Claude Haiku 4.5 and DeepSeek V4 Flash slots): one
#     unified `reasoning` field OpenRouter translates
#     per-provider -- see openrouter_reasoning_param() below. Anthropic models
#     get reasoning.max_tokens (-> Claude's native budget_tokens server-side);
#     everything else gets reasoning.effort.
GEMINI_THINKING_LEVEL = "MEDIUM"          # Google's default-optimized setting
OPENROUTER_ANTHROPIC_BUDGET_TOKENS = 2048  # tighter than Claude's default budget
OPENROUTER_DEFAULT_EFFORT = "high"         # DeepSeek and any non-Anthropic model

# USD per 1M tokens, as (input, output). Token COUNTS reported by the counter are
# measured from each API response and are exact; the dollar figures are only as
# good as this table. Keys are matched as a prefix of the CSV's model_id, so both
# the OpenRouter form (anthropic/claude-haiku-4.5) and the direct form
# (claude-haiku-4-5) are listed.
#
#   gemini-3.5-flash      VERIFIED against Google's published rate.
#   claude-haiku-4.5      VERIFIED ($1/$5) against OpenRouter's + Anthropic's pages.
#   deepseek/*            DELIBERATELY ABSENT -- could not verify a rate, and this
#                         file does not guess prices. DeepSeek rows get exact token
#                         counts but no cost (shown as n/a). Add a verified rate
#                         here if you want its cost.
#
# A model_id absent from this table still gets exact token counts, just no cost.
PRICING = {
    "gemini-3.5-flash":           (1.50, 9.00),   # verified
    "anthropic/claude-haiku-4.5": (1.00, 5.00),   # verified (OpenRouter slug)
}
UNVERIFIED_PRICES = set()  # every rate above is verified; deepseek is left unpriced


def price_for(model_id):
    """(input_rate, output_rate) per 1M tokens, or None if unpriced.

    Mock ids are deliberately unpriced: mock runs never call an API, so their
    cost is zero and their token counts are chars/4 guesses, not real usage.
    """
    mid = str(model_id).lower()
    if mid.endswith("-mock"):
        return None
    for prefix, rates in PRICING.items():
        if mid.startswith(prefix):
            return rates
    return None


def summarize_usage(usage, stream=None):
    """Print a token + cost summary. `usage` maps model_id -> [calls, in, out]."""
    out = stream or sys.stdout
    if not usage:
        return
    print("\n=== token usage ===", file=out)
    print(f"{'model':24} {'calls':>5} {'input':>10} {'output':>10} {'total':>10} {'est. cost':>10}",
          file=out)
    t_calls = t_in = t_out = 0
    t_cost = 0.0
    any_unverified = False
    for mid, (calls, tin, tout) in sorted(usage.items()):
        rates = price_for(mid)
        if rates:
            cost = tin / 1e6 * rates[0] + tout / 1e6 * rates[1]
            unverified = any(mid.lower().startswith(p) for p in UNVERIFIED_PRICES)
            any_unverified |= unverified
            cost_s = f"${cost:,.4f}" + ("*" if unverified else "")
            t_cost += cost
        else:
            cost_s = "n/a"
        print(f"{mid:24} {calls:5} {tin:10,} {tout:10,} {tin+tout:10,} {cost_s:>10}", file=out)
        t_calls += calls; t_in += tin; t_out += tout
    print(f"{'TOTAL':24} {t_calls:5} {t_in:10,} {t_out:10,} {t_in+t_out:10,} "
          f"{'$' + format(t_cost, ',.4f'):>10}", file=out)
    if any_unverified:
        print("* cost uses an UNVERIFIED rate -- see PRICING in this file. Token "
              "counts are exact; verify rates before quoting costs.", file=out)

SYSTEM_PROMPT = (
    "You are an information assistant. Based only on the provided documents, "
    "answer the user's question."
)

SYSTEM_PROMPT_COT = (
    "You are an information assistant. Based only on the provided documents, "
    "analyze the evidence step-by-step and think out loud before answering the user's question."
)

POSTHOC_SYSTEM_PROMPT = (
    "You are a confidence evaluator. Based only on the supplied documents and "
    "the previous answer, estimate whether that answer gives the single true "
    "factual value. Do not revise or replace the answer."
)

PROMPT_CORE_TEMPLATE = """Below are documents retrieved for a research question. Read them, then answer the question.

{documents}

Question: {question}"""

PRIMARY_WITH_CONFIDENCE = """Respond with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>", "probability_correct": <integer 0-100>}}

probability_correct is the probability that your answer supplies the single
true factual value. It is not merely confidence that your answer follows the
claim repeated most often."""

PRIMARY_WITHOUT_CONFIDENCE = """Respond with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>"}}

Do not report, rate, or discuss your confidence."""

POSTHOC_TEMPLATE = """{prompt_core}

A model previously gave this answer:
{previous_answer}

Do not answer the question again and do not revise the previous answer.
Evaluate only whether it supplies the single true factual value. Return ONLY
one JSON object in exactly this format:
{{"probability_correct": <integer 0-100>}}"""


CSV_FIELDS = [
    "run_id", "timestamp_utc", "entity_id", "entity_name", "domain",
    "attribute", "ratio", "n_docs",
    "call_phase", "trial_index", "strategy", "prompt_hash", "doc_positions",
    "confidence_condition", "inline_confidence_requested", "model_slot",
    "model_provider", "model_id", "returned_model_id", "question",
    "majority_value", "minority_value", "response_category", "raw_response",
    "parsed_answer", "parsed_confidence", "confidence_scale",
    "format_error", "prompt_tokens",
    "completion_tokens", "error",
]

CONDITION_FIELDS = [
    "run_id", "timestamp_utc", "entity_id", "entity_name", "domain",
    "attribute", "ratio", "n_docs", "strategy",
    "prompt_hash", "doc_positions", "confidence_condition",
    "inline_confidence_requested", "model_slot", "model_provider", "model_id",
    "returned_model_ids", "question", "majority_value", "minority_value",
    "response_categories",
    "n_samples", "n_scored", "modal_category", "modal_count",
    "self_consistency", "self_consistency_all_samples", "modal_tie",
    "modal_answer", "inline_probabilities",
    "mean_inline_probability", "posthoc_probability", "posthoc_raw_response",
    "posthoc_prompt_tokens", "posthoc_completion_tokens", "posthoc_error",
]


def build_prompt(entity, ratio, strategy="standard", trial_idx=1,
                 run_seed=20260714, ask_inline_confidence=True,
                 return_core=False):
    """Build one primary prompt.

    ``trial_idx`` controls document layout only. The run loop builds this once
    per entity/ratio and sends the exact resulting string three times, so
    repeated-sampling agreement is not confounded by different document order.
    """
    docs = list(entity["documents"][ratio])
    
    # Shuffle documents using a seed combining trial index, entity name, and ratio.
    # Ensures different trials run with different document layouts to neutralize position bias.
    shuffle_rng = py_random.Random(f"{run_seed}-{entity['entity_id']}-{ratio}-trial-{trial_idx}")
    shuffle_rng.shuffle(docs)
    
    doc_blocks = []
    doc_positions = []
    for i, d in enumerate(docs, start=1):
        doc_blocks.append(f"Document {i} (source: {d['source']}):\n{d['text']}")
        if entity["majority_value"] in d["text"]:
            doc_positions.append("MAJ")
        elif entity["minority_value"] in d["text"]:
            doc_positions.append("MIN")
        else:
            doc_positions.append("UNK")
            
    prompt_core = PROMPT_CORE_TEMPLATE.format(
        documents="\n\n".join(doc_blocks), question=entity["question"]
    )
    instruction = (
        PRIMARY_WITH_CONFIDENCE
        if ask_inline_confidence
        else PRIMARY_WITHOUT_CONFIDENCE
    )
    if strategy == "cot":
        instruction = instruction.replace(
            "Respond with ONLY a JSON object",
            "Respond with your step-by-step reasoning, then conclude with ONLY "
            "a JSON object",
            1,
        )
    prompt = prompt_core + "\n\n" + instruction
    if return_core:
        return prompt, doc_positions, prompt_core
    return prompt, doc_positions


def build_posthoc_prompt(prompt_core, previous_answer):
    return POSTHOC_TEMPLATE.format(
        prompt_core=prompt_core,
        previous_answer=json.dumps(previous_answer, ensure_ascii=False),
    )


def prompt_digest(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def extract_json_object(raw):
    """Return the last valid JSON object in a response."""
    if not raw:
        return None
    decoder = json.JSONDecoder()
    found = []
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            found.append(value)
    return found[-1] if found else None


def _parse_probability(value):
    if isinstance(value, bool):
        raise ValueError("probability_correct must be numeric")
    number = float(value)
    if not number.is_integer() or not 0 <= number <= 100:
        raise ValueError("probability_correct must be an integer from 0 to 100")
    return int(number)


def parse_primary_response(raw, expect_inline_confidence=True):
    """Parse the answer and the new 0-100 self-report.

    A legacy 1-5 ``confidence`` field is retained only for compatibility and is
    never promoted to the primary probability field.
    """
    parsed = {
        "answer": "", "probability_correct": "",
        "format_error": "",
    }
    obj = extract_json_object(raw)
    if obj is None:
        parsed["format_error"] = "no valid JSON object"
        return parsed
    parsed["answer"] = str(obj.get("answer", "")).strip()
    if not parsed["answer"]:
        parsed["format_error"] = "JSON object has no answer"

    if "probability_correct" in obj:
        try:
            parsed["probability_correct"] = _parse_probability(
                obj["probability_correct"]
            )
        except (TypeError, ValueError) as exc:
            parsed["format_error"] = str(exc)
    if expect_inline_confidence and parsed["probability_correct"] == "":
        parsed["format_error"] = (
            parsed["format_error"] or "missing probability_correct"
        )
    if not expect_inline_confidence and parsed["probability_correct"] != "":
        parsed["format_error"] = "control response unexpectedly reported confidence"
    return parsed


def parse_posthoc_response(raw):
    obj = extract_json_object(raw)
    if obj is None:
        return "", "no valid JSON object"
    try:
        return _parse_probability(obj.get("probability_correct")), ""
    except (TypeError, ValueError) as exc:
        return "", str(exc)


def parse_response(raw):
    """Backward-compatible parser returning ``(answer, confidence)``."""
    parsed = parse_primary_response(raw, expect_inline_confidence=False)
    return parsed["answer"], parsed["probability_correct"]


def select_no_inline_confidence_ids(entities, count, seed):
    """Select a deterministic control stratified by domain."""
    if count < 0 or count > len(entities):
        raise ValueError(
            f"no-inline confidence count must be between 0 and {len(entities)}"
        )
    if count == 0:
        return set()
    strata = defaultdict(list)
    for entity in entities:
        key = entity.get("domain", "")
        strata[key].append(entity)

    exact = {
        stratum: count * len(group) / len(entities)
        for stratum, group in strata.items()
    }
    allocation = {stratum: int(value) for stratum, value in exact.items()}
    remaining = count - sum(allocation.values())
    for stratum in sorted(
        strata, key=lambda key: (-(exact[key] - allocation[key]), key)
    ):
        if remaining <= 0:
            break
        allocation[stratum] += 1
        remaining -= 1

    selected = set()
    for stratum, group in strata.items():
        ranked = sorted(
            group,
            key=lambda entity: hashlib.sha256(
                f"{seed}|no-inline-confidence|{entity['entity_id']}".encode("utf-8")
            ).digest(),
        )
        selected.update(
            entity["entity_id"] for entity in ranked[:allocation[stratum]]
        )
    return selected


def classify_primary_row(row):
    visualizations_dir = str(REPO_ROOT / "visualizations")
    if visualizations_dir not in sys.path:
        sys.path.insert(0, visualizations_dir)
    from common import classify
    return classify(row)


def summarize_repeats(rows):
    categories = [row["response_category"] for row in rows]
    scored = [category for category in categories if category != "UNSCORED"]
    if not scored:
        return {
            "categories": categories, "n_scored": 0, "modal_category": "",
            "modal_count": 0, "self_consistency": "",
            "self_consistency_all_samples": 0.0, "modal_tie": "",
            "representative": rows[0],
        }
    counts = Counter(scored)
    modal_count = max(counts.values())
    tied = {category for category, value in counts.items() if value == modal_count}
    modal_category = next(category for category in scored if category in tied)
    representative = next(
        row for row in rows if row["response_category"] == modal_category
    )
    return {
        "categories": categories,
        "n_scored": len(scored),
        "modal_category": modal_category,
        "modal_count": modal_count,
        "self_consistency": modal_count / len(scored),
        "self_consistency_all_samples": modal_count / len(rows),
        "modal_tie": int(len(tied) > 1),
        "representative": representative,
    }


# ---------------------------------------------------------------------------
# Model callers: each returns
# (raw_text, prompt_tokens, completion_tokens, API-returned model id)

def call_with_backoff(fn, max_retries=MAX_RETRIES, base_delay=1.0, max_delay=65.0):
    """Retry `fn` on transient errors (429 / 5xx / connection) with exponential
    backoff + jitter. Used for the Gemini path; the OpenAI SDK does this
    internally via max_retries. Non-transient errors (e.g. 400/404) raise
    immediately.

    When a 429 carries a server-supplied retryDelay (Gemini per-minute quota
    resets), honor it instead of the shorter exponential delay so the retry
    actually lands after the quota window rather than failing again."""
    from google.genai import errors as genai_errors

    # Network-layer failures (dropped WiFi, DNS blip, connect/read timeout) are
    # raised by the underlying httpx transport, NOT as google.genai errors, so
    # they bypass the genai_errors handlers below entirely. A 2026-07-15 run lost
    # 40 Gemini calls to un-retried ConnectTimeouts during a WiFi outage for
    # exactly this reason. These are transient by definition -- retry them.
    import httpx
    NETWORK_ERRORS = (
        httpx.TimeoutException,     # covers Connect/Read/Write/PoolTimeout
        httpx.NetworkError,         # covers ConnectError/ReadError/WriteError
        httpx.RemoteProtocolError,
        ConnectionError,            # stdlib, in case a lower layer raises it
        TimeoutError,
    )

    def _is_transient(exc):
        code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if code in (408, 409, 429) or (isinstance(code, int) and code >= 500):
            return True
        # ServerError is 5xx; connection/timeout errors have no HTTP code
        return isinstance(exc, genai_errors.ServerError) or code is None

    def _server_retry_delay(exc):
        m = re.search(r"retry.?delay['\"]?\s*:\s*['\"]?(\d+)", str(exc), re.I)
        return int(m.group(1)) if m else None

    last = None
    for attempt in range(max_retries):
        try:
            return fn()
        except genai_errors.ClientError as exc:
            if not _is_transient(exc):  # 400/404 etc. — don't retry
                raise
            last = exc
        except (genai_errors.ServerError, genai_errors.APIError) as exc:
            last = exc
        except NETWORK_ERRORS as exc:
            last = exc
        backoff = base_delay * (2 ** attempt)
        server_delay = _server_retry_delay(last) or 0
        # wait at least as long as the server asks (+small buffer), capped
        delay = min(max(backoff, server_delay + 1) + random.uniform(0, 0.5), max_delay)
        time.sleep(delay)
    raise last


def call_gemini(client, model_id, prompt, system_prompt=SYSTEM_PROMPT,
                temperature=1.0):
    from google.genai import types
    def _do():
        return client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel[GEMINI_THINKING_LEVEL],
                ),
            ),
        )
    resp = call_with_backoff(_do)
    text = resp.text or ""
    usage = resp.usage_metadata
    ptok = usage.prompt_token_count if usage else ""
    ctok = usage.candidates_token_count if usage else ""
    returned_model = getattr(resp, "model_version", "") or model_id
    return text, ptok, ctok, returned_model


def openrouter_reasoning_param(model_id):
    """Build OpenRouter's unified `reasoning` field for the configured model.

    OpenRouter exposes ONE field (`reasoning`) whose shape it translates to
    each provider's native mechanism server-side:
      - Anthropic models: reasoning.max_tokens -> Claude's native budget_tokens
      - Most other models (DeepSeek, Gemini-via-OpenRouter, etc.): reasoning.effort
    The explicit DeepSeek and Claude slots therefore receive the correct shape
    automatically without mutating a shared model-selection variable.
    """
    mid = model_id.lower()
    if "claude" in mid or "anthropic" in mid:
        return {"max_tokens": OPENROUTER_ANTHROPIC_BUDGET_TOKENS}
    return {"effort": OPENROUTER_DEFAULT_EFFORT}


def call_openrouter(client, model_id, prompt, system_prompt=SYSTEM_PROMPT,
                    temperature=1.0):
    # OpenRouter exposes an OpenAI-compatible chat completions endpoint, so this
    # uses the standard OpenAI SDK (which retries 429/5xx/connection errors
    # internally via max_retries) pointed at OpenRouter's base_url. `reasoning`
    # is an OpenRouter-specific field, not part of the OpenAI schema, so it goes
    # through extra_body rather than a typed SDK parameter.
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=OPENROUTER_MAX_TOKENS,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        extra_body={"reasoning": openrouter_reasoning_param(model_id)},
    )
    usage = resp.usage
    return (resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else "",
            usage.completion_tokens if usage else "",
            getattr(resp, "model", "") or model_id)


def model_family_matches(slot, returned_model_id):
    """Detect gateway-side substitution across the study's model families."""
    returned = str(returned_model_id).lower()
    if not returned:
        return False
    if slot == "gemini":
        return "gemini" in returned
    if slot == "deepseek":
        return "deepseek" in returned
    if slot == "claude":
        return "claude" in returned or "anthropic" in returned
    return False


class MockClient:
    """Simulates a model response so the full pipeline can be tested offline."""

    def __init__(self, provider):
        self.provider = provider

    def call(self, prompt, entity, strategy="standard", phase="primary",
             ask_inline_confidence=True):
        answer = entity["majority_value"]
        if phase == "posthoc":
            raw = json.dumps({"probability_correct": 80})
        elif strategy == "cot":
            payload = {"answer": answer}
            if ask_inline_confidence:
                payload["probability_correct"] = 90
            raw = (
                "Thinking Process:\n"
                f"1. The question asks about {entity['question']}.\n"
                f"2. Multiple documents state the answer is {answer}.\n"
                "3. Concluding with the JSON block.\n\n"
                + json.dumps(payload)
            )
        else:
            payload = {"answer": answer}
            if ask_inline_confidence:
                payload["probability_correct"] = 90
            raw = json.dumps(payload)
            
        sys_prompt = (
            POSTHOC_SYSTEM_PROMPT if phase == "posthoc"
            else SYSTEM_PROMPT_COT if strategy == "cot"
            else SYSTEM_PROMPT
        )
        return (raw, len(sys_prompt + prompt) // 4, len(raw) // 4,
                f"{self.provider}-MOCK")


def main():
    # Load the repository-local environment without overriding explicitly
    # exported variables. Model selection itself no longer uses the ambiguous
    # OPENROUTER_MODEL variable; DeepSeek and Claude have separate slots.
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entities", type=int, default=None,
                    help="limit to first N entities (default: all)")
    ap.add_argument("--entity-ids", nargs="*", default=None,
                    help="run only these entity IDs, e.g. E001 E002")
    ap.add_argument("--ratios", nargs="*", default=None,
                    help="subset of ratios, e.g. 3:1 2:2 (default: all)")
    ap.add_argument(
        "--models", nargs="*", choices=["gemini", "deepseek", "claude"],
        default=["gemini", "deepseek", "claude"],
        help="explicit model slots to run (default: all three)",
    )
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--deepseek-model", default=None,
                    help="DeepSeek OpenRouter id (default: DEEPSEEK_MODEL env "
                         "or deepseek/deepseek-v4-flash)")
    ap.add_argument("--claude-model", default=None,
                    help="Claude OpenRouter id (default: CLAUDE_MODEL env or "
                         "anthropic/claude-haiku-4.5)")
    ap.add_argument(
        "--trials", type=int, default=DEFAULT_SAMPLES_PER_CONDITION,
        help="identical answer samples per model/condition (production default: 3)",
    )
    ap.add_argument(
        "--no-inline-confidence-entities", type=int,
        default=DEFAULT_NO_INLINE_CONFIDENCE_ENTITIES,
        help="deterministic entities that omit inline confidence (default: 10)",
    )
    ap.add_argument(
        "--layout-index", type=int, default=1,
        help="document-order layout shared by all repeated samples (default: 1)",
    )
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="nonzero sampling temperature for repeated calls (default: 1.0)")
    ap.add_argument("--strategy", choices=["standard", "cot"], default="standard",
                    help="prompting strategy to use (default: standard)")
    ap.add_argument("--mock", action="store_true",
                    help="no API calls; canned responses (pipeline test)")
    ap.add_argument("--output", default=None,
                    help="raw call-log CSV (default: results/run_<timestamp>.csv)")
    ap.add_argument("--condition-output", default=None,
                    help="condition-level CSV with modal answers")
    args = ap.parse_args()

    if args.trials < 1:
        ap.error("--trials must be at least 1")
    if args.temperature <= 0:
        ap.error("--temperature must be greater than zero for repeated sampling")
    if not args.models:
        ap.error("--models must select at least one model")
    if args.trials != DEFAULT_SAMPLES_PER_CONDITION:
        print(
            f"WARNING: production design specifies exactly "
            f"{DEFAULT_SAMPLES_PER_CONDITION} identical samples; this run uses "
            f"{args.trials}."
        )

    dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    full_entities = dataset["entities"]

    run_seed = dataset.get("seed", 20260714)
    try:
        no_inline_ids = select_no_inline_confidence_ids(
            full_entities, args.no_inline_confidence_entities, run_seed
        )
    except ValueError as exc:
        ap.error(str(exc))

    entities = full_entities
    if args.entity_ids:
        requested_ids = set(args.entity_ids)
        entities = [e for e in entities if e["entity_id"] in requested_ids]
    if args.entities is not None:
        if args.entities < 1:
            ap.error("--entities must be at least 1")
        entities = entities[:args.entities]
    if not entities:
        ap.error("entity filters selected no entities")

    ratios = args.ratios or list(dataset["ratios"])
    unknown_ratios = sorted(set(ratios) - set(dataset["ratios"]))
    if unknown_ratios:
        ap.error(f"unknown ratios: {', '.join(unknown_ratios)}")

    deepseek_model = (
        args.deepseek_model or os.environ.get("DEEPSEEK_MODEL")
        or DEFAULT_DEEPSEEK_MODEL
    )
    claude_model = (
        args.claude_model or os.environ.get("CLAUDE_MODEL")
        or DEFAULT_CLAUDE_MODEL
    )
    if "deepseek" in args.models and "deepseek" not in deepseek_model.lower():
        ap.error(
            f"--deepseek-model must identify a DeepSeek model, got {deepseek_model!r}"
        )
    if "claude" in args.models and not any(
        marker in claude_model.lower() for marker in ("claude", "anthropic")
    ):
        ap.error(
            f"--claude-model must identify a Claude/Anthropic model, got {claude_model!r}"
        )
    if os.environ.get("OPENROUTER_MODEL"):
        print(
            "NOTE: legacy OPENROUTER_MODEL is ignored. The harness now runs "
            "explicit DeepSeek and Claude slots; use --deepseek-model or "
            "--claude-model to override them."
        )

    requested_models = {
        "gemini": args.gemini_model,
        "deepseek": deepseek_model,
        "claude": claude_model,
    }
    providers = {
        "gemini": "gemini",
        "deepseek": "openrouter",
        "claude": "openrouter",
    }

    def system_prompt_for(phase):
        if phase == "posthoc":
            return POSTHOC_SYSTEM_PROMPT
        return SYSTEM_PROMPT_COT if args.strategy == "cot" else SYSTEM_PROMPT

    # Each caller accepts (prompt, entity, phase, ask_inline_confidence).
    callers = {}
    if args.mock:
        for slot in args.models:
            requested = requested_models[slot] + "-MOCK"
            mock = MockClient(requested_models[slot])
            callers[slot] = {
                "provider": providers[slot],
                "model_id": requested,
                "call": lambda p, e, phase, ask, m=mock: m.call(
                    p, e, args.strategy, phase, ask
                ),
            }
    else:
        missing = []
        if "gemini" in args.models:
            vertex = os.environ.get("GEMINI_USE_VERTEX", "").lower() in (
                "1", "true", "yes"
            )
            project = os.environ.get("GEMINI_VERTEX_PROJECT")
            if vertex and project:
                credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                if credentials and not Path(credentials).is_file():
                    sys.exit(
                        "GOOGLE_APPLICATION_CREDENTIALS is set but the file "
                        f"does not exist:\n  {credentials}"
                    )
            elif not os.environ.get("GEMINI_API_KEY"):
                missing.append("GEMINI_API_KEY")
        if any(slot in args.models for slot in ("deepseek", "claude")) \
                and not os.environ.get("OPENROUTER_API_KEY"):
            missing.append("OPENROUTER_API_KEY")
        if missing:
            sys.exit(
                f"Missing environment variable(s): {', '.join(missing)}. "
                "Export them (or add to .env) or use --mock."
            )

        if "gemini" in args.models:
            from google import genai
            vertex = os.environ.get("GEMINI_USE_VERTEX", "").lower() in (
                "1", "true", "yes"
            )
            project = os.environ.get("GEMINI_VERTEX_PROJECT")
            if vertex and project:
                location = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
                gemini_client = genai.Client(
                    vertexai=True, project=project, location=location
                )
            elif vertex:
                gemini_client = genai.Client(
                    vertexai=True, api_key=os.environ["GEMINI_API_KEY"]
                )
            else:
                gemini_client = genai.Client()
            callers["gemini"] = {
                "provider": "gemini",
                "model_id": args.gemini_model,
                "call": lambda p, e, phase, ask, c=gemini_client,
                               m=args.gemini_model: call_gemini(
                    c, m, p, system_prompt_for(phase), args.temperature
                ),
            }

        if any(slot in args.models for slot in ("deepseek", "claude")):
            from openai import OpenAI
            for slot in ("deepseek", "claude"):
                if slot not in args.models:
                    continue
                model_id = requested_models[slot]
                # Separate clients let the two OpenRouter models run concurrently
                # without sharing mutable connection state.
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"],
                    max_retries=MAX_RETRIES,
                )
                callers[slot] = {
                    "provider": "openrouter",
                    "model_id": model_id,
                    "call": lambda p, e, phase, ask, c=client,
                                   m=model_id: call_openrouter(
                        c, m, p, system_prompt_for(phase), args.temperature
                    ),
                }

    print("\n=== resolved model manifest (fixed for this run) ===")
    for slot, spec in callers.items():
        reasoning = (
            f", reasoning={openrouter_reasoning_param(spec['model_id'])}"
            if spec["provider"] == "openrouter" else ""
        )
        print(
            f"  {slot:8s} provider={spec['provider']:10s} "
            f"requested_model={spec['model_id']}{reasoning}"
        )
    selected_controls = sorted(
        entity["entity_id"] for entity in entities
        if entity["entity_id"] in no_inline_ids
    )
    print(
        f"Inline-confidence control: {len(no_inline_ids)} of "
        f"{len(full_entities)} full-dataset entities; {len(selected_controls)} "
        "are present in this run."
    )
    print("  control IDs in this run: " + (", ".join(selected_controls) or "none"))

    run_id = uuid.uuid4().hex[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / f"run_{stamp}_{run_id}.csv"
    )
    suffix = out_path.stem[4:] if out_path.stem.startswith("run_") else out_path.stem
    condition_path = (
        Path(args.condition_output) if args.condition_output
        else out_path.with_name(f"conditions_{suffix}.csv")
    )
    if out_path.resolve() == condition_path.resolve():
        ap.error("--output and --condition-output must be different files")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    condition_path.parent.mkdir(parents=True, exist_ok=True)


    planned_total = (
        len(entities) * len(ratios) * len(callers) * (args.trials + 1)
    )
    done = 0
    hard_errors = 0
    usage = defaultdict(lambda: [0, 0, 0])
    condition_rows = []

    def base_raw_row(entity, ratio, prompt, doc_positions, confidence_condition,
                     ask_inline, slot, phase, trial_index=""):
        spec = callers[slot]
        return {
            "run_id": run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "entity_id": entity["entity_id"],
            "entity_name": entity["entity_name"],
            "domain": entity["domain"],
            "attribute": entity["attribute"],
            "ratio": ratio,
            "n_docs": len(entity["documents"][ratio]),
            "call_phase": phase,
            "trial_index": trial_index,
            "strategy": args.strategy,
            "prompt_hash": prompt_digest(prompt),
            "doc_positions": json.dumps(doc_positions),
            "confidence_condition": confidence_condition,
            "inline_confidence_requested": int(ask_inline),
            "model_slot": slot,
            "model_provider": spec["provider"],
            "model_id": spec["model_id"],
            "returned_model_id": "",
            "question": entity["question"],
            "majority_value": entity["majority_value"],
            "minority_value": entity["minority_value"],
            "response_category": "",
            "raw_response": "",
            "parsed_answer": "",
            "parsed_confidence": "",
            "confidence_scale": "",
            "format_error": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "error": "",
        }

    def record_usage(model_id, prompt_tokens, completion_tokens):
        values = usage[model_id]
        values[0] += 1
        values[1] += int(prompt_tokens or 0)
        values[2] += int(completion_tokens or 0)

    # Raw rows are flushed after every billed call. Condition rows are also
    # flushed as soon as each modal/post-hoc result is available; after the run,
    # that file is rewritten once with cross-validated Platt scores added.
    with out_path.open("w", newline="", encoding="utf-8") as raw_file, \
            condition_path.open("w", newline="", encoding="utf-8") as condition_file, \
            ThreadPoolExecutor(max_workers=max(len(callers), 1)) as pool:
        raw_writer = csv.DictWriter(raw_file, fieldnames=CSV_FIELDS)
        condition_writer = csv.DictWriter(
            condition_file, fieldnames=CONDITION_FIELDS
        )
        raw_writer.writeheader()
        condition_writer.writeheader()

        for entity in entities:
            ask_inline = entity["entity_id"] not in no_inline_ids
            confidence_condition = (
                "inline_plus_posthoc" if ask_inline
                else "posthoc_only_control"
            )
            for ratio in ratios:
                prompt, doc_positions, prompt_core = build_prompt(
                    entity, ratio, strategy=args.strategy,
                    trial_idx=args.layout_index, run_seed=run_seed,
                    ask_inline_confidence=ask_inline, return_core=True,
                )
                repeated = {slot: [] for slot in callers}

                for trial_index in range(1, args.trials + 1):
                    futures = {
                        slot: pool.submit(
                            spec["call"], prompt, entity, "primary", ask_inline
                        )
                        for slot, spec in callers.items()
                    }
                    for slot, spec in callers.items():
                        row = base_raw_row(
                            entity, ratio, prompt, doc_positions,
                            confidence_condition, ask_inline, slot, "primary",
                            trial_index,
                        )
                        try:
                            raw, ptok, ctok, returned_model = futures[slot].result()
                            row.update(
                                raw_response=raw, prompt_tokens=ptok,
                                completion_tokens=ctok,
                                returned_model_id=returned_model,
                            )
                            record_usage(spec["model_id"], ptok, ctok)
                            if not model_family_matches(slot, returned_model):
                                row["error"] = (
                                    "model_mismatch: requested "
                                    f"{spec['model_id']}, API returned {returned_model}"
                                )
                            parsed = parse_primary_response(raw, ask_inline)
                            row.update(
                                parsed_answer=parsed["answer"],
                                parsed_confidence=parsed["probability_correct"],
                                confidence_scale=(
                                    "0-100_probability"
                                    if parsed["probability_correct"] != "" else ""
                                ),
                                format_error=parsed["format_error"],
                            )
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        row["response_category"] = classify_primary_row(row)
                        if row["error"]:
                            hard_errors += 1
                        raw_writer.writerow(row)
                        raw_file.flush()
                        repeated[slot].append(row)
                        done += 1
                        print(
                            f"[{done}/{planned_total}] {entity['entity_id']} "
                            f"{ratio} sample {trial_index} {slot} "
                            f"({spec['model_id']})"
                            f"{' ERROR' if row['error'] else ''}"
                        )

                summaries = {
                    slot: summarize_repeats(rows)
                    for slot, rows in repeated.items()
                }
                posthoc_prompts = {}
                posthoc_futures = {}
                for slot, summary in summaries.items():
                    modal_answer = summary["representative"]["parsed_answer"]
                    if not summary["modal_category"] or not modal_answer:
                        continue
                    posthoc_prompt = build_posthoc_prompt(
                        prompt_core, modal_answer
                    )
                    posthoc_prompts[slot] = posthoc_prompt
                    posthoc_futures[slot] = pool.submit(
                        callers[slot]["call"], posthoc_prompt, entity,
                        "posthoc", False,
                    )

                for slot, spec in callers.items():
                    summary = summaries[slot]
                    posthoc_probability = ""
                    posthoc_raw = ""
                    posthoc_ptok = ""
                    posthoc_ctok = ""
                    posthoc_returned_model = ""
                    posthoc_error = ""
                    if slot not in posthoc_futures:
                        posthoc_error = "posthoc_skipped_no_modal_answer"
                    else:
                        posthoc_prompt = posthoc_prompts[slot]
                        row = base_raw_row(
                            entity, ratio, posthoc_prompt, doc_positions,
                            confidence_condition, ask_inline, slot, "posthoc",
                        )
                        try:
                            posthoc_raw, posthoc_ptok, posthoc_ctok, returned_model = (
                                posthoc_futures[slot].result()
                            )
                            row.update(
                                raw_response=posthoc_raw,
                                prompt_tokens=posthoc_ptok,
                                completion_tokens=posthoc_ctok,
                                returned_model_id=returned_model,
                            )
                            posthoc_returned_model = returned_model
                            record_usage(
                                spec["model_id"], posthoc_ptok, posthoc_ctok
                            )
                            if not model_family_matches(slot, returned_model):
                                row["error"] = (
                                    "model_mismatch: requested "
                                    f"{spec['model_id']}, API returned {returned_model}"
                                )
                            probability, format_error = parse_posthoc_response(
                                posthoc_raw
                            )
                            row.update(
                                parsed_confidence=probability,
                                confidence_scale=(
                                    "0-100_probability" if probability != "" else ""
                                ),
                                format_error=format_error,
                            )
                            if row["error"]:
                                posthoc_error = row["error"]
                            elif format_error:
                                posthoc_error = format_error
                            else:
                                posthoc_probability = probability
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            posthoc_error = row["error"]
                        if row["error"]:
                            hard_errors += 1
                        raw_writer.writerow(row)
                        raw_file.flush()
                        done += 1
                        print(
                            f"[{done}/{planned_total}] {entity['entity_id']} "
                            f"{ratio} posthoc {slot} ({spec['model_id']})"
                            f"{' ERROR' if posthoc_error else ''}"
                        )

                    inline_probabilities = [
                        int(row["parsed_confidence"])
                        for row in repeated[slot]
                        if ask_inline and row["parsed_confidence"] != ""
                    ]
                    modal_category = summary["modal_category"]
                    returned_model_ids = {
                        row["returned_model_id"] for row in repeated[slot]
                        if row["returned_model_id"]
                    }
                    if posthoc_returned_model:
                        returned_model_ids.add(posthoc_returned_model)
                    condition = {
                        "run_id": run_id,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "entity_id": entity["entity_id"],
                        "entity_name": entity["entity_name"],
                        "domain": entity["domain"],
                        "attribute": entity["attribute"],
                        "ratio": ratio,
                        "n_docs": len(entity["documents"][ratio]),
                        "strategy": args.strategy,
                        "prompt_hash": prompt_digest(prompt),
                        "doc_positions": json.dumps(doc_positions),
                        "confidence_condition": confidence_condition,
                        "inline_confidence_requested": int(ask_inline),
                        "model_slot": slot,
                        "model_provider": spec["provider"],
                        "model_id": spec["model_id"],
                        "returned_model_ids": json.dumps(
                            sorted(returned_model_ids)
                        ),
                        "question": entity["question"],
                        "majority_value": entity["majority_value"],
                        "minority_value": entity["minority_value"],
                        "response_categories": json.dumps(summary["categories"]),
                        "n_samples": len(repeated[slot]),
                        "n_scored": summary["n_scored"],
                        "modal_category": modal_category,
                        "modal_count": summary["modal_count"],
                        "self_consistency": summary["self_consistency"],
                        "self_consistency_all_samples": summary[
                            "self_consistency_all_samples"
                        ],
                        "modal_tie": summary["modal_tie"],
                        "modal_answer": summary["representative"]["parsed_answer"],
                        "inline_probabilities": json.dumps(inline_probabilities),
                        "mean_inline_probability": (
                            round(sum(inline_probabilities) / len(inline_probabilities), 2)
                            if inline_probabilities else ""
                        ),
                        "posthoc_probability": posthoc_probability,
                        "posthoc_raw_response": posthoc_raw,
                        "posthoc_prompt_tokens": posthoc_ptok,
                        "posthoc_completion_tokens": posthoc_ctok,
                        "posthoc_error": posthoc_error,
                    }
                    condition_rows.append(condition)
                    condition_writer.writerow(condition)
                    condition_file.flush()


    print(
        f"\nDone. {done} live/mock calls completed out of {planned_total} "
        f"planned; {hard_errors} hard/model-routing errors."
    )
    print(f"Raw call log: {out_path}")
    print(f"Condition results: {condition_path}")

    summarize_usage(usage)


if __name__ == "__main__":
    main()
