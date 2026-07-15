"""Query harness for the Majority Illusion experiment.

For each entity x ratio combination, builds a prompt containing the
conflicting documents, queries both models, elicits a stated confidence,
and appends one row per (entity, ratio, trial, model) to a CSV with full metadata.

Models:
  - Gemini: gemini-3.1-flash-lite  (key from GEMINI_API_KEY; override --gemini-model)
  - OpenAI: via Azure OpenAI Service (deployment from AZURE_OPENAI_DEPLOYMENT
            or --azure-deployment)

Both run on paid, pay-as-you-go tiers. Model B is served through Azure OpenAI
Service (Azure for Students credits) rather than the direct OpenAI API.

NOTE 1: gemini-3.5-flash (the frontier Flash tier) is capped at 20 requests/day
on this project's free quota -- unusable for a 200-call run (50 entities x 4
ratios x 1 model). gemini-3.1-flash-lite gets 500 req/day and 15 req/min on the
same free tier, which comfortably covers the full run. Confirmed against the
live models.list() response for this API key -- there is no plain
"gemini-3.1-flash" on this account, only the -lite variant (GA) plus preview/
image/tts variants. Document the quota-driven downgrade from 3.5 Flash to
3.1 Flash-Lite in the Research Brief.

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
import random as py_random
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "entities.json"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"  # 500 req/day, 15/min on free tier
DEFAULT_AZURE_DEPLOYMENT = "gpt-5-mini"     # Azure OpenAI deployment name
DEFAULT_AZURE_API_VERSION = "2024-12-01-preview"  # supports GPT-5 reasoning models
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"  # 3.5 Haiku retired 2026-02-19

# Both models "think" before answering (Gemini thinking tokens; GPT-5
# reasoning tokens). Those count against the output budget, so give both
# generous headroom well above the ~30 tokens the visible JSON needs — a tight
# cap gets consumed by reasoning and truncates the JSON mid-object. GPT-5 also
# requires max_completion_tokens (not max_tokens) and only the default temperature.
GEMINI_MAX_OUTPUT_TOKENS = 2048
OPENAI_MAX_COMPLETION_TOKENS = 2000
ANTHROPIC_MAX_TOKENS = 2048
MAX_RETRIES = 4  # exponential-backoff attempts on 429/5xx/connection errors

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
            ),
        )
    resp = call_with_backoff(_do)
    text = resp.text or ""
    usage = resp.usage_metadata
    ptok = usage.prompt_token_count if usage else ""
    ctok = usage.candidates_token_count if usage else ""
    return text, ptok, ctok


def call_openai(client, deployment, prompt, strategy="standard"):
    # On Azure OpenAI, `model` is the deployment name, not the base model name.
    # GPT-5 reasoning models require max_completion_tokens (not max_tokens) and
    # only accept the default temperature. The OpenAI SDK retries transient
    # errors (429/5xx/connection) internally via max_retries.
    sys_prompt = SYSTEM_PROMPT_COT if strategy == "cot" else SYSTEM_PROMPT
    resp = client.chat.completions.create(
        model=deployment,
        max_completion_tokens=OPENAI_MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    usage = resp.usage
    return (resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else "",
            usage.completion_tokens if usage else "")


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
    ap.add_argument("--models", nargs="*", choices=["gemini", "openai", "anthropic"],
                    default=["gemini", "openai"])
    ap.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    ap.add_argument("--azure-deployment", default=None,
                    help="Azure OpenAI deployment name "
                         "(default: AZURE_OPENAI_DEPLOYMENT env or gpt-5-mini)")
    ap.add_argument("--azure-api-version", default=None,
                    help="Azure OpenAI API version "
                         "(default: AZURE_OPENAI_API_VERSION env or 2024-12-01-preview)")
    ap.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    ap.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
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

    # Resolve Azure OpenAI settings: CLI flag > env var > built-in default.
    azure_deployment = (args.azure_deployment
                        or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
                        or DEFAULT_AZURE_DEPLOYMENT)
    azure_api_version = (args.azure_api_version
                         or os.environ.get("AZURE_OPENAI_API_VERSION")
                         or DEFAULT_AZURE_API_VERSION)

    # --- set up clients -----------------------------------------------------
    callers = {}
    if args.mock:
        for provider in args.models:
            mock = MockClient(provider)
            if provider == "gemini":
                model_id = args.gemini_model + "-MOCK"
            elif provider == "openai":
                model_id = (azure_deployment or args.openai_model) + "-MOCK"
            elif provider == "anthropic":
                model_id = args.anthropic_model + "-MOCK"
            callers[provider] = (model_id, lambda p, e, m=mock, s=args.strategy: m.call(p, e, s))
    else:
        missing = []
        if "gemini" in args.models and not os.environ.get("GEMINI_API_KEY"):
            missing.append("GEMINI_API_KEY")
        if "openai" in args.models:
            for var in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"):
                if not os.environ.get(var):
                    missing.append(var)
        if "anthropic" in args.models and not os.environ.get("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            sys.exit(f"Missing environment variable(s): {', '.join(missing)}. "
                     f"Export them (or add to a .env file) or use --mock.")
        if "gemini" in args.models:
            from google import genai
            gm = genai.Client()  # reads GEMINI_API_KEY from the environment
            callers["gemini"] = (
                args.gemini_model,
                lambda p, e, c=gm, m=args.gemini_model, s=args.strategy: call_gemini(c, m, p, s))
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
                lambda p, e, c=az, m=azure_deployment, s=args.strategy: call_openai(c, m, p, s))
        if "anthropic" in args.models:
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

    with out_path.open("w", newline="") as f:
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


if __name__ == "__main__":
    main()
