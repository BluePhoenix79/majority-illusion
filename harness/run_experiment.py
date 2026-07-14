"""Query harness for the Majority Illusion experiment.

For each entity x ratio combination, builds a prompt containing the
conflicting documents, queries both models, elicits a stated confidence,
and appends one row per (entity, ratio, model) to a CSV with full metadata.

Models (override with --openai-model / --anthropic-model):
  - OpenAI:    gpt-4o-mini            (key from OPENAI_API_KEY)
  - Anthropic: claude-haiku-4-5       (key from ANTHROPIC_API_KEY)

NOTE: the project plan named Claude 3.5 Haiku, but claude-3-5-haiku-20241022
was RETIRED by Anthropic on Feb 19, 2026 and now returns 404. claude-haiku-4-5
is the official drop-in replacement (fastest/cheapest current tier). Document
this substitution in the Research Brief.

Error handling: both SDKs retry transient errors (429/5xx/timeouts) with
exponential backoff via max_retries; on top of that, any failure on a single
call is caught and recorded in the CSV's `error` column so the run continues.
Rows are flushed to disk after every call, so a crash loses nothing.

Usage:
    python harness/run_experiment.py --mock --entities 3           # pipeline test, no API calls
    python harness/run_experiment.py --entities 3                  # real pilot (needs both keys)
    python harness/run_experiment.py                               # full 50-entity run
"""

import argparse
import csv
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_ROOT / "data" / "entities.json"
RESULTS_DIR = REPO_ROOT / "results"

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"  # 3.5 Haiku retired 2026-02-19
MAX_TOKENS = 300
MAX_RETRIES = 4  # SDK-level exponential backoff on 429/5xx/connection errors

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

def call_openai(client, model_id, prompt):
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    usage = resp.usage
    return (resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else "",
            usage.completion_tokens if usage else "")


def call_anthropic(client, model_id, prompt):
    resp = client.messages.create(
        model=model_id,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--entities", type=int, default=None,
                    help="limit to first N entities (default: all)")
    ap.add_argument("--entity-ids", nargs="*", default=None,
                    help="run only these entity IDs, e.g. E001 E002")
    ap.add_argument("--ratios", nargs="*", default=None,
                    help="subset of ratios, e.g. 3:1 2:2 (default: all)")
    ap.add_argument("--models", nargs="*", choices=["openai", "anthropic"],
                    default=["openai", "anthropic"])
    ap.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    ap.add_argument("--anthropic-model", default=DEFAULT_ANTHROPIC_MODEL)
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

    # --- set up clients -----------------------------------------------------
    callers = {}  # provider -> (model_id, fn(prompt, entity) -> (raw, ptok, ctok))
    if args.mock:
        for provider in args.models:
            mock = MockClient(provider)
            model_id = {"openai": args.openai_model,
                        "anthropic": args.anthropic_model}[provider] + "-MOCK"
            callers[provider] = (model_id, mock.call)
    else:
        missing = []
        if "openai" in args.models and not os.environ.get("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        if "anthropic" in args.models and not os.environ.get("ANTHROPIC_API_KEY"):
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            sys.exit(f"Missing environment variable(s): {', '.join(missing)}. "
                     f"Export them or use --mock.")
        if "openai" in args.models:
            from openai import OpenAI
            oa = OpenAI(max_retries=MAX_RETRIES)
            callers["openai"] = (
                args.openai_model,
                lambda p, e, c=oa, m=args.openai_model: call_openai(c, m, p))
        if "anthropic" in args.models:
            import anthropic
            an = anthropic.Anthropic(max_retries=MAX_RETRIES)
            callers["anthropic"] = (
                args.anthropic_model,
                lambda p, e, c=an, m=args.anthropic_model: call_anthropic(c, m, p))

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
