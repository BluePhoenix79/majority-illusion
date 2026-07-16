"""Query harness for the Majority Illusion experiment.

For each entity x ratio combination, builds a prompt containing the
conflicting documents, queries both models, elicits a stated confidence,
and appends one row per (entity, ratio, model) to a CSV with full metadata.

Models:
  - Gemini: gemini-3.5-flash       (key from GEMINI_API_KEY; override --gemini-model)
  - OpenAI: via Azure OpenAI Service (deployment from AZURE_OPENAI_DEPLOYMENT
            or --azure-deployment)

Both run on paid, pay-as-you-go tiers. Model B is served through Azure OpenAI
Service (Azure for Students credits) rather than the direct OpenAI API.

NOTE 1: gemini-3.5-flash is Google's current frontier Flash model (GA 2026-05-19).
An earlier free-tier key capped it at 20 requests/day, which forced a temporary
downgrade to gemini-3.1-flash-lite (500/day); that key has since been replaced
with one that has unrestricted 3.5 Flash access, so we are back on the frontier
Flash tier and the Flash-Lite downgrade no longer applies. Both 3.5 Flash and
gpt-5-mini are current-generation models, so the Research Brief can describe the
pair as such -- but see the thinking-token note below: 3.5 Flash reasons before
answering and needs output headroom.

NOTE 2: On Azure OpenAI, the `model` argument to the API is the *deployment name*
you created in the Azure AI Foundry portal, NOT the base model name. Record the
underlying base model (e.g. gpt-4o-mini) in the Research Brief since the CSV
stores the deployment name.

Error handling: the OpenAI SDK retries transient errors (429/5xx/timeouts) with
exponential backoff via max_retries; the Gemini path is wrapped in an equivalent
exponential-backoff retry (see call_with_backoff). On top of that, any failure
on a single call is caught and recorded in the CSV's `error` column so the run
continues. Rows are flushed to disk after every call, so a crash loses nothing.

Config is read from the environment:
  GEMINI_API_KEY
  AZURE_OPENAI_ENDPOINT      e.g. https://<resource>.openai.azure.com/
  AZURE_OPENAI_API_KEY
  AZURE_OPENAI_DEPLOYMENT    your deployment name (default: gpt-4o-mini)
  AZURE_OPENAI_API_VERSION   (default: 2024-10-21)
A .env file in the repo root is auto-loaded if present (.env is gitignored, so
secrets are never committed).

Usage:
    python harness/run_experiment.py --mock --entities 3           # pipeline test, no API calls
    python harness/run_experiment.py --entities 3                  # real pilot (needs both keys)
    python harness/run_experiment.py                               # full 50-entity run
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
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "entities.json"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"  # current frontier Flash (GA 2026-05-19)
DEFAULT_AZURE_DEPLOYMENT = "gpt-5-mini"     # Azure OpenAI deployment name
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"  # supports GPT-5 reasoning models
# Both models "think" before answering (Gemini thinking tokens; GPT-5
# reasoning tokens). Those count against the output budget, so give both
# generous headroom well above the ~30 tokens the visible JSON needs — a tight
# cap gets consumed by reasoning and truncates the JSON mid-object. GPT-5 also
# requires max_completion_tokens (not max_tokens) and only the default temperature.
GEMINI_MAX_OUTPUT_TOKENS = 2048
OPENAI_MAX_COMPLETION_TOKENS = 2000
MAX_RETRIES = 4  # exponential-backoff attempts on 429/5xx/connection errors

SYSTEM_PROMPT = (
    "You are a careful research assistant. Answer the user's question using "
    "the provided documents as your evidence."
)

PROMPT_TEMPLATE = """Below are documents retrieved for a research question. Read them, then answer the question.

{documents}

Question: {question}

Respond with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>", "confidence": <integer 0-100 representing how confident you are in this answer>}}"""

CSV_FIELDS = [
    "run_id", "timestamp_utc", "entity_id", "entity_name", "domain",
    "attribute", "ratio", "n_docs", "model_provider", "model_id", "question",
    "majority_value", "minority_value", "raw_response", "parsed_answer",
    "parsed_confidence", "prompt_tokens", "completion_tokens", "error",
]


def build_prompt(entity, ratio):
    docs = entity["documents"][ratio]
    doc_blocks = []
    for i, d in enumerate(docs, start=1):
        doc_blocks.append(f"Document {i} (source: {d['source']}):\n{d['text']}")
    return PROMPT_TEMPLATE.format(documents="\n\n".join(doc_blocks),
                                  question=entity["question"])


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
        backoff = base_delay * (2 ** attempt)
        server_delay = _server_retry_delay(last) or 0
        # wait at least as long as the server asks (+small buffer), capped
        delay = min(max(backoff, server_delay + 1) + random.uniform(0, 0.5), max_delay)
        time.sleep(delay)
    raise last


def call_gemini(client, model_id, prompt):
    from google.genai import types
    def _do():
        return client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )
    resp = call_with_backoff(_do)
    text = resp.text or ""
    usage = resp.usage_metadata
    ptok = usage.prompt_token_count if usage else ""
    ctok = usage.candidates_token_count if usage else ""
    return text, ptok, ctok


def call_openai(client, deployment, prompt):
    # On Azure OpenAI, `model` is the deployment name, not the base model name.
    # GPT-5 reasoning models require max_completion_tokens (not max_tokens) and
    # only accept the default temperature. The OpenAI SDK retries transient
    # errors (429/5xx/connection) internally via max_retries.
    resp = client.chat.completions.create(
        model=deployment,
        max_completion_tokens=OPENAI_MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    usage = resp.usage
    return (resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else "",
            usage.completion_tokens if usage else "")


class MockClient:
    """Simulates a model response so the full pipeline can be tested offline.

    Estimates token counts as chars/4 so cost projections are still possible.
    """

    def __init__(self, provider):
        self.provider = provider

    def call(self, prompt, entity):
        answer = entity["majority_value"]
        raw = json.dumps({"answer": answer, "confidence": 85})
        return raw, len(SYSTEM_PROMPT + prompt) // 4, len(raw) // 4


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
    ap.add_argument("--models", nargs="*", choices=["gemini", "openai"],
                    default=["gemini", "openai"])
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--azure-deployment", default=None,
                    help="Azure OpenAI deployment name "
                         "(default: AZURE_OPENAI_DEPLOYMENT env or gpt-4o-mini)")
    ap.add_argument("--azure-api-version", default=None,
                    help="Azure OpenAI API version "
                         "(default: AZURE_OPENAI_API_VERSION env or 2024-10-21)")
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

    # Resolve Azure OpenAI settings: CLI flag > env var > built-in default.
    azure_deployment = (args.azure_deployment
                        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                        or DEFAULT_AZURE_DEPLOYMENT)
    azure_api_version = (args.azure_api_version
                         or os.environ.get("AZURE_OPENAI_API_VERSION")
                         or DEFAULT_AZURE_API_VERSION)

    # --- set up clients -----------------------------------------------------
    callers = {}  # provider -> (model_id, fn(prompt, entity) -> (raw, ptok, ctok))
    if args.mock:
        for provider in args.models:
            mock = MockClient(provider)
            model_id = {"gemini": args.gemini_model,
                        "openai": azure_deployment}[provider] + "-MOCK"
            callers[provider] = (model_id, mock.call)
    else:
        missing = []
        if "gemini" in args.models and not os.environ.get("GEMINI_API_KEY"):
            missing.append("GEMINI_API_KEY")
        if "openai" in args.models:
            for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
                if not os.environ.get(var):
                    missing.append(var)
        if missing:
            sys.exit(f"Missing environment variable(s): {', '.join(missing)}. "
                     f"Export them (or add to a .env file) or use --mock.")
        if "gemini" in args.models:
            from google import genai
            gm = genai.Client()  # reads GEMINI_API_KEY from the environment
            callers["gemini"] = (
                args.gemini_model,
                lambda p, e, c=gm, m=args.gemini_model: call_gemini(c, m, p))
        if "openai" in args.models:
            from openai import AzureOpenAI
            az = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=azure_api_version,
                max_retries=MAX_RETRIES,
            )
            callers["openai"] = (
                azure_deployment,
                lambda p, e, c=az, m=azure_deployment: call_openai(c, m, p))

    # --- run ------------------------------------------------------------------
    run_id = uuid.uuid4().hex[:8]
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"run_{stamp}_{run_id}.csv"

    total = len(entities) * len(ratios) * len(callers)
    done = 0
    errors = 0

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for entity in entities:
            for ratio in ratios:
                prompt = build_prompt(entity, ratio)
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
                        if not answer:
                            row["error"] = "parse_failure: no JSON answer extracted"
                    except Exception as exc:  # one failed call must not stop the run
                        errors += 1
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    writer.writerow(row)
                    f.flush()
                    done += 1
                    print(f"[{done}/{total}] {entity['entity_id']} {ratio} "
                          f"{provider}{' ERROR' if row['error'] else ''}")

    print(f"\nDone. {done} calls, {errors} hard errors. Output: {out_path}")


if __name__ == "__main__":
    main()
