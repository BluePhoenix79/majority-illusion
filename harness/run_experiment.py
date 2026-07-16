"""Query harness for the Majority Illusion experiment.

For each entity x ratio combination, builds a prompt containing the
conflicting documents, queries both models, elicits a stated confidence,
and appends one row per (entity, ratio, trial, model) to a CSV with full metadata.

Providers (one CSV row per model, per condition):
  - gemini   : gemini-3.5-flash, via the native google-genai SDK.
               Key from GEMINI_API_KEY; override --gemini-model.
  - deepseek : the OpenRouter slot. Runs whatever OPENROUTER_MODEL points at --
               deepseek/deepseek-v4-flash by default, or anthropic/claude-haiku-4.5
               to test Claude. BOTH go through the same OPENROUTER_API_KEY; swap
               the model string, nothing else. (The slot is named "deepseek" for
               historical reasons; it serves any OpenRouter model.)
  - anthropic: OPTIONAL, opt-in only. Claude via a *direct* Anthropic API key.
               Not part of the current study -- Claude is tested through the
               OpenRouter slot above, no separate key needed. Kept as an escape
               hatch; runs only with an explicit --models anthropic.

The study models (gemini + the OpenRouter slot) run by default. gemini-3.5-flash
is Google's current frontier Flash model (GA 2026-05-19).

NOTE: gemini-3.5-flash reasons before answering; its thinking tokens count
against the output budget, so a tight cap truncates the JSON mid-object. Both
providers get generous output headroom (see the *_MAX_* constants). Per-model
reasoning depth is set explicitly -- see GEMINI_THINKING_LEVEL and
openrouter_reasoning_param().

Error handling: the OpenAI-compatible SDK (OpenRouter) retries transient errors
(429/5xx/timeouts) internally via max_retries; the Gemini path is wrapped in an
equivalent retry (see call_with_backoff, which also handles httpx network
errors). Any single-call failure is caught and recorded in the CSV's `error`
column so the run continues; rows are flushed after every call.

Config is read from the environment:
  GEMINI_API_KEY
  GEMINI_USE_VERTEX          set to 1/true to use Vertex AI Express Mode instead
                              of the AI Studio endpoint (needed for keys scoped
                              to Google Cloud / aiplatform.googleapis.com rather
                              than generativelanguage.googleapis.com -- a plain
                              AI Studio key gets a 403 API_KEY_SERVICE_BLOCKED
                              from Vertex, and vice versa; use whichever matches
                              how the key was provisioned). Express Mode takes
                              ONLY the API key -- no project/location needed.
  OPENROUTER_API_KEY         one key for the OpenRouter slot (DeepSeek or Claude)
  OPENROUTER_MODEL           which OpenRouter model to run (default:
                              deepseek/deepseek-v4-flash; also --openrouter-model)
  ANTHROPIC_API_KEY          only if you opt into the direct --models anthropic slot
A .env file in the repo root is auto-loaded if present (.env is gitignored, so
secrets are never committed).

Usage:
    python harness/run_experiment.py --mock --entities 3   # pipeline test, no API calls
    python harness/run_experiment.py --entities 3          # small live pilot
    python harness/run_experiment.py                       # full 50-entity run
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
import uuid
import random as py_random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "entities.json"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"  # current frontier Flash (GA 2026-05-19)
DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"  # DeepSeek via OpenRouter
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"  # only for the opt-in direct slot

# Models "think" before answering, and those reasoning/thinking tokens count
# against the output budget -- a tight cap gets consumed by reasoning and
# truncates the JSON mid-object. Give generous headroom well above the ~30
# tokens the visible JSON answer needs.
GEMINI_MAX_OUTPUT_TOKENS = 2048
ANTHROPIC_MAX_TOKENS = 2048
OPENROUTER_MAX_TOKENS = 2048
MAX_RETRIES = 4  # exponential-backoff attempts on 429/5xx/connection errors

# Reasoning-depth controls, one per model, verified against each SDK/API before
# use rather than assumed (see UPDATES.md for how each was checked):
#   - Gemini: native google-genai ThinkingConfig.thinking_level. Confirmed via
#     the installed SDK's pydantic model_fields that this field exists and
#     ThinkingLevel.MEDIUM is a real enum member (not guessed).
#   - OpenRouter (Claude Haiku 4.5 / DeepSeek V4 Flash, switched via
#     OPENROUTER_MODEL): one unified `reasoning` field OpenRouter translates
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
#   gemini-3.1-flash-lite NOT VERIFIED (only used for a few early pilot calls).
#
# A model_id absent from this table still gets exact token counts, just no cost.
PRICING = {
    "gemini-3.5-flash":           (1.50, 9.00),   # verified
    "gemini-3.1-flash-lite":      (0.10, 0.40),   # unverified estimate
    "anthropic/claude-haiku-4.5": (1.00, 5.00),   # verified (OpenRouter slug)
    "claude-haiku-4-5":           (1.00, 5.00),   # verified (direct-API id)
}
UNVERIFIED_PRICES = {"gemini-3.1-flash-lite"}


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

PROMPT_TEMPLATE = """Below are documents retrieved for a research question. Read them, then answer the question.

{documents}

Question: {question}

Respond with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>", "confidence": <integer 1-5 representing how confident you are in this answer: 1 = very uncertain, 5 = completely certain>}}"""

PROMPT_TEMPLATE_COT = """Below are documents retrieved for a research question. Read them, analyze the evidence step-by-step, then answer the question.

{documents}

Question: {question}

Respond with your step-by-step reasoning trace, and conclude with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>", "confidence": <integer 1-5 representing how confident you are in this answer: 1 = very uncertain, 5 = completely certain>}}"""

CSV_FIELDS = [
    "run_id", "timestamp_utc", "entity_id", "entity_name", "domain",
    "attribute", "ratio", "n_docs", "trial_index", "strategy", "doc_positions",
    "model_provider", "model_id", "question", "majority_value", "minority_value",
    "raw_response", "parsed_answer", "parsed_confidence", "prompt_tokens",
    "completion_tokens", "error",
]


def build_prompt(entity, ratio, strategy="standard", trial_idx=1, run_seed=20260714):
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
            
    template = PROMPT_TEMPLATE_COT if strategy == "cot" else PROMPT_TEMPLATE
    prompt = template.format(documents="\n\n".join(doc_blocks), question=entity["question"])
    return prompt, doc_positions


def parse_response(raw):
    """Best-effort extraction of {"answer", "confidence"} from model output."""
    if not raw:
        return "", ""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return "", ""
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "", ""
    answer = str(obj.get("answer", "")).strip()
    confidence = obj.get("confidence", "")
    try:
        confidence = int(confidence)
    except (TypeError, ValueError):
        confidence = ""
    return answer, confidence


# ---------------------------------------------------------------------------
# Model callers: each returns (raw_text, prompt_tokens, completion_tokens)

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


def call_gemini(client, model_id, prompt, strategy="standard"):
    from google.genai import types
    sys_prompt = SYSTEM_PROMPT_COT if strategy == "cot" else SYSTEM_PROMPT
    def _do():
        return client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
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
    return text, ptok, ctok


def call_anthropic(client, model_id, prompt, strategy="standard"):
    sys_prompt = SYSTEM_PROMPT_COT if strategy == "cot" else SYSTEM_PROMPT
    resp = client.messages.create(
        model=model_id,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=sys_prompt,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def openrouter_reasoning_param(model_id):
    """Build OpenRouter's unified `reasoning` field for the configured model.

    OpenRouter exposes ONE field (`reasoning`) whose shape it translates to
    each provider's native mechanism server-side:
      - Anthropic models: reasoning.max_tokens -> Claude's native budget_tokens
      - Most other models (DeepSeek, Gemini-via-OpenRouter, etc.): reasoning.effort
    This lets OPENROUTER_MODEL be swapped (e.g. deepseek/deepseek-v4-flash <->
    anthropic/claude-haiku-4.5) with no other code change -- the right shape is
    picked automatically from the model id.
    """
    mid = model_id.lower()
    if "claude" in mid or "anthropic" in mid:
        return {"max_tokens": OPENROUTER_ANTHROPIC_BUDGET_TOKENS}
    return {"effort": OPENROUTER_DEFAULT_EFFORT}


def call_openrouter(client, model_id, prompt, strategy="standard"):
    # OpenRouter exposes an OpenAI-compatible chat completions endpoint, so this
    # uses the standard OpenAI SDK (which retries 429/5xx/connection errors
    # internally via max_retries) pointed at OpenRouter's base_url. `reasoning`
    # is an OpenRouter-specific field, not part of the OpenAI schema, so it goes
    # through extra_body rather than a typed SDK parameter.
    sys_prompt = SYSTEM_PROMPT_COT if strategy == "cot" else SYSTEM_PROMPT
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=OPENROUTER_MAX_TOKENS,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        extra_body={"reasoning": openrouter_reasoning_param(model_id)},
    )
    usage = resp.usage
    return (resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else "",
            usage.completion_tokens if usage else "")


class MockClient:
    """Simulates a model response so the full pipeline can be tested offline."""

    def __init__(self, provider):
        self.provider = provider

    def call(self, prompt, entity, strategy="standard"):
        answer = entity["majority_value"]
        if strategy == "cot":
            raw = (
                "Thinking Process:\n"
                f"1. The question asks about {entity['question']}.\n"
                f"2. Multiple documents state the answer is {answer}.\n"
                "3. Concluding with the JSON block.\n\n"
                + json.dumps({"answer": answer, "confidence": 5})
            )
        else:
            raw = json.dumps({"answer": answer, "confidence": 5})
            
        sys_prompt = SYSTEM_PROMPT_COT if strategy == "cot" else SYSTEM_PROMPT
        return raw, len(sys_prompt + prompt) // 4, len(raw) // 4


def main():
    # Load a .env from the repo root if present (never required; keys can also
    # come straight from the environment). Silently skipped if python-dotenv
    # isn't installed.
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
    ap.add_argument("--models", nargs="*",
                    choices=["gemini", "deepseek", "anthropic"],
                    default=["gemini", "deepseek"],
                    help="providers to run (default: gemini + the OpenRouter "
                         "slot; 'anthropic' is the opt-in direct-API escape hatch)")
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--openrouter-model", default=None,
                    help="OpenRouter model id for the deepseek slot "
                         "(default: OPENROUTER_MODEL env or deepseek/deepseek-v4-flash; "
                         "set to anthropic/claude-haiku-4.5 to test Claude)")
    ap.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL,
                    help="model id for the opt-in direct Anthropic slot")
    ap.add_argument("--trials", type=int, default=1,
                    help="number of repetitions per condition (default: 1)")
    ap.add_argument("--strategy", choices=["standard", "cot"], default="standard",
                    help="prompting strategy to use (default: standard)")
    ap.add_argument("--mock", action="store_true",
                    help="no API calls; canned responses (pipeline test)")
    ap.add_argument("--output", default=None,
                    help="output CSV path (default: results/run_<timestamp>.csv)")
    args = ap.parse_args()

    dataset = json.loads(DATA_PATH.read_text())
    entities = dataset["entities"]
    if args.entity_ids:
        entities = [e for e in entities if e["entity_id"] in args.entity_ids]
    if args.entities:
        entities = entities[: args.entities]
    ratios = args.ratios or list(dataset["ratios"])
    run_seed = dataset.get("seed", 20260714)

    # Which OpenRouter model the deepseek slot runs: CLI flag > env var > default.
    openrouter_model = (args.openrouter_model
                        or os.environ.get("OPENROUTER_MODEL")
                        or DEFAULT_OPENROUTER_MODEL)

    # --- set up clients -----------------------------------------------------
    callers = {}
    if args.mock:
        for provider in args.models:
            mock = MockClient(provider)
            if provider == "gemini":
                model_id = args.gemini_model + "-MOCK"
            elif provider == "deepseek":
                model_id = openrouter_model + "-MOCK"
            elif provider == "anthropic":
                model_id = args.anthropic_model + "-MOCK"
            callers[provider] = (model_id, lambda p, e, m=mock, s=args.strategy: m.call(p, e, s))
    else:
        missing = []
        if "gemini" in args.models and not os.environ.get("GEMINI_API_KEY"):
            missing.append("GEMINI_API_KEY")
        if "deepseek" in args.models and not os.environ.get("OPENROUTER_API_KEY"):
            missing.append("OPENROUTER_API_KEY")
        if "anthropic" in args.models and not os.environ.get("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            sys.exit(f"Missing environment variable(s): {', '.join(missing)}. "
                     f"Export them (or add to a .env file) or use --mock.")
        if "gemini" in args.models:
            from google import genai
            # GEMINI_USE_VERTEX=1 -> Vertex AI Express Mode (Google Cloud
            # credits; key scoped to aiplatform.googleapis.com, not the AI
            # Studio generativelanguage.googleapis.com endpoint). Express Mode
            # takes ONLY vertexai=True + api_key -- passing project/location
            # alongside api_key is rejected by the SDK ("mutually exclusive").
            # Default path (unset) is the plain AI Studio client.
            if os.environ.get("GEMINI_USE_VERTEX", "").lower() in ("1", "true", "yes"):
                gm = genai.Client(vertexai=True, api_key=os.environ["GEMINI_API_KEY"])
            else:
                gm = genai.Client()  # reads GEMINI_API_KEY from the environment
            callers["gemini"] = (
                args.gemini_model,
                lambda p, e, c=gm, m=args.gemini_model, s=args.strategy: call_gemini(c, m, p, s))
        if "deepseek" in args.models:
            # OpenRouter gateway (OpenAI-compatible). Runs whatever OPENROUTER_MODEL
            # points at -- DeepSeek by default, or anthropic/claude-haiku-4.5 to
            # test Claude through the same key.
            from openai import OpenAI
            dr = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                max_retries=MAX_RETRIES,
            )
            callers["deepseek"] = (
                openrouter_model,
                lambda p, e, c=dr, m=openrouter_model, s=args.strategy: call_openrouter(c, m, p, s))
        if "anthropic" in args.models:
            # Opt-in only: Claude via a DIRECT Anthropic key, separate from the
            # OpenRouter slot above. Not part of the default study.
            import anthropic
            an = anthropic.Anthropic(max_retries=MAX_RETRIES)
            callers["anthropic"] = (
                args.anthropic_model,
                lambda p, e, c=an, m=args.anthropic_model, s=args.strategy: call_anthropic(c, m, p, s))

    # --- run ------------------------------------------------------------------
    run_id = uuid.uuid4().hex[:8]
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"run_{stamp}_{run_id}.csv"

    total = len(entities) * len(ratios) * args.trials * len(callers)
    done = 0
    errors = 0
    usage = defaultdict(lambda: [0, 0, 0])  # model_id -> [calls, input, output]

    # encoding="utf-8" is required: on Windows, open() without it defaults to
    # the system locale (cp1252), and model responses containing em-dashes or
    # other non-ASCII punctuation then fail to round-trip through pandas
    # (UnicodeDecodeError on 0x97 etc.) when the CSV is read back as UTF-8.
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for entity in entities:
            for ratio in ratios:
                for trial_idx in range(1, args.trials + 1):
                    prompt, doc_positions = build_prompt(entity, ratio, strategy=args.strategy, trial_idx=trial_idx, run_seed=run_seed)
                    for provider, (model_id, call) in callers.items():
                        row = {
                            "run_id": run_id,
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "entity_id": entity["entity_id"],
                            "entity_name": entity["entity_name"],
                            "domain": entity["domain"],
                            "attribute": entity["attribute"],
                            "ratio": ratio,
                            "n_docs": len(entity["documents"][ratio]),
                            "trial_index": trial_idx,
                            "strategy": args.strategy,
                            "doc_positions": json.dumps(doc_positions),
                            "model_provider": provider,
                            "model_id": model_id,
                            "question": entity["question"],
                            "majority_value": entity["majority_value"],
                            "minority_value": entity["minority_value"],
                            "raw_response": "", "parsed_answer": "",
                            "parsed_confidence": "", "prompt_tokens": "",
                            "completion_tokens": "", "error": "",
                        }
                        try:
                            raw, ptok, ctok = call(prompt, entity)
                            answer, confidence = parse_response(raw)
                            row.update(raw_response=raw, parsed_answer=answer,
                                       parsed_confidence=confidence,
                                       prompt_tokens=ptok, completion_tokens=ctok)
                            # Count tokens for any call the API actually billed,
                            # including ones whose JSON failed to parse -- those
                            # still cost money.
                            u = usage[model_id]
                            u[0] += 1
                            u[1] += int(ptok or 0)
                            u[2] += int(ctok or 0)
                            if not answer:
                                row["error"] = "parse_failure: no JSON answer extracted"
                        except Exception as exc:
                            errors += 1
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        writer.writerow(row)
                        f.flush()
                        done += 1
                        print(f"[{done}/{total}] {entity['entity_id']} {ratio} trial {trial_idx} "
                              f"{provider}{' ERROR' if row['error'] else ''}")

    print(f"\nDone. {done} calls, {errors} hard errors. Output: {out_path}")
    summarize_usage(usage)


if __name__ == "__main__":
    main()
