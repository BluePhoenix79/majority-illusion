"""Query harness for the revised Majority Illusion experiment.

For every entity x ratio x model condition the harness:
  1. sends one byte-identical primary prompt three times;
  2. classifies each answer MAJ/MIN/COM/FLAG/OTHER/UNSCORED;
  3. records modal-category self-consistency as a separate diagnostic;
  4. asks the same model for a post-hoc 0-100 probability that the modal answer
     is the best resolution of the supplied documents.

For conflict ratios, most entities also report a rich probability distribution
inside each primary answer: probabilities for counterbalanced Claim A, Claim B,
and indeterminate (summing to 100), plus an independent probability that the
sources conflict. A deterministic, domain-and-attribute-stratified 37-entity
control sees the same Claim A/B labels but only returns an answer, so the study
can measure whether distribution elicitation changes behavior. Every 4:0
condition is a pure answer-only unanimous-context control; post-hoc confidence
is still collected for it.

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
DEFAULT_NO_INLINE_CONFIDENCE_ENTITIES = 37
PROTOCOL_VERSION = "2026-07-17-rich-distribution-v3-balanced"

# Alternating this fixed order with entity number gives an exactly balanced
# Claim-A mapping across the full 75 x 6 design (225 majority-as-A and 225
# minority-as-A conditions). The run seed and layout index flip labels without
# changing that balance.
RATIO_COUNTERBALANCE_ORDER = ("4:0", "3:1", "2:2", "4:1", "2:1", "3:2")

# Models "think" before answering, and those reasoning/thinking tokens count
# against the output budget -- a tight cap gets consumed by reasoning and
# truncates the JSON mid-object. Give generous headroom well above the ~30
# tokens the visible JSON answer needs.
GEMINI_MAX_OUTPUT_TOKENS = 2048
OPENROUTER_DEFAULT_MAX_TOKENS = 2048
OPENROUTER_CLAUDE_MAX_TOKENS = 4096
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
    "the previous answer, estimate whether that answer is the best resolution "
    "of the supplied documents. Do not revise or replace the answer."
)

PROMPT_CORE_TEMPLATE = """Below are documents retrieved for a research question. Read them, then answer the question.

{documents}

Question: {question}"""

CLAIM_LABELS = """Use these counterbalanced labels:
Claim A: {claim_a_value}
Claim B: {claim_b_value}"""

PRIMARY_WITH_CONFIDENCE = CLAIM_LABELS + """

Respond with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>", "p_claim_a": <integer 0-100>, "p_claim_b": <integer 0-100>, "p_indeterminate": <integer 0-100>, "p_sources_conflict": <integer 0-100>}}

p_claim_a, p_claim_b, and p_indeterminate are your subjective probabilities
for the best resolution of the supplied documents and must sum exactly to 100.
p_sources_conflict is an independent 0-100 estimate that the supplied sources
conflict; it is not included in that sum."""

PRIMARY_WITHOUT_CONFIDENCE = CLAIM_LABELS + """

Respond with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>"}}"""

PRIMARY_UNANIMOUS_ANSWER_ONLY = """Respond with ONLY a JSON object in exactly this format:
{{"answer": "<your answer>"}}"""

POSTHOC_TEMPLATE = """{prompt_core}

A model previously gave this answer:
{previous_answer}

Do not answer the question again and do not revise the previous answer.
Estimate only whether it is the best resolution of the supplied documents.
Return ONLY one JSON object in exactly this format:
{{"confidence_best_resolution": <integer 0-100>}}"""


CSV_FIELDS = [
    "run_id", "protocol_version", "dataset_sha256", "run_seed",
    "layout_index", "timestamp_utc", "entity_id", "entity_name", "domain",
    "attribute", "ratio", "n_docs",
    "call_phase", "trial_index", "strategy", "prompt_hash", "doc_positions",
    "confidence_condition", "distribution_request_assigned",
    "inline_confidence_requested", "model_slot",
    "model_provider", "model_id", "returned_model_id", "question",
    "majority_value", "minority_value", "claim_a_value", "claim_b_value",
    "claim_a_side", "claim_b_side", "response_category", "mentions_conflict",
    "abstained", "raw_response",
    "parsed_answer", "p_claim_a", "p_claim_b", "p_indeterminate",
    "p_sources_conflict", "p_majority", "p_minority",
    "confidence_best_resolution", "confidence_scale",
    "format_error", "prompt_tokens", "completion_tokens", "reasoning_tokens",
    "error",
]

CONDITION_FIELDS = [
    "run_id", "protocol_version", "dataset_sha256", "run_seed",
    "layout_index", "timestamp_utc", "entity_id", "entity_name", "domain",
    "attribute", "ratio", "n_docs", "strategy",
    "prompt_hash", "doc_positions", "confidence_condition",
    "distribution_request_assigned", "inline_confidence_requested",
    "model_slot", "model_provider", "model_id",
    "returned_model_ids", "question", "majority_value", "minority_value",
    "claim_a_value", "claim_b_value", "claim_a_side", "claim_b_side",
    "response_categories",
    "n_samples", "n_scored", "n_primary_errors", "n_primary_format_errors",
    "n_valid_distributions", "distribution_compliance", "modal_category", "modal_count",
    "conflict_mention_count", "conflict_mention_rate", "abstention_count",
    "abstention_rate",
    "self_consistency", "self_consistency_all_samples", "modal_tie",
    "modal_answer", "inline_distributions", "inline_p_majority",
    "inline_p_minority", "inline_p_indeterminate", "inline_p_sources_conflict",
    "mean_p_majority", "mean_p_minority", "mean_p_indeterminate",
    "mean_p_sources_conflict", "confidence_best_resolution",
    "posthoc_raw_response", "primary_reasoning_tokens", "posthoc_reasoning_tokens",
    "reasoning_tokens", "posthoc_status", "posthoc_skipped",
    "posthoc_prompt_tokens", "posthoc_completion_tokens", "posthoc_error",
]


def dataset_sha256(path=DATA_PATH):
    """Return a stable fingerprint of the exact dataset bytes used."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_protocol_dataset(dataset):
    """Reject accidental dataset/protocol drift before any API call is made."""
    entities = dataset.get("entities", [])
    if len(entities) != 75:
        raise ValueError(f"protocol requires exactly 75 entities; found {len(entities)}")
    if len({entity.get("entity_id") for entity in entities}) != 75:
        raise ValueError("protocol requires 75 unique entity IDs")
    domains = Counter(entity.get("domain") for entity in entities)
    if domains != Counter({"banking": 30, "general": 45}):
        raise ValueError(
            "protocol requires 30 banking and 45 general entities; "
            f"found {dict(domains)}"
        )
    if tuple(dataset.get("ratios", [])) != RATIO_COUNTERBALANCE_ORDER:
        raise ValueError(
            "protocol ratio order must be " + ", ".join(RATIO_COUNTERBALANCE_ORDER)
        )
    for entity in entities:
        if set(entity.get("documents", {})) != set(RATIO_COUNTERBALANCE_ORDER):
            raise ValueError(
                f"{entity.get('entity_id')} does not contain exactly the six ratios"
            )
        for ratio in RATIO_COUNTERBALANCE_ORDER:
            expected_majority, expected_minority = map(int, ratio.split(":"))
            sides = Counter(
                document_side(
                    document["text"], entity["majority_value"],
                    entity["minority_value"],
                )
                for document in entity["documents"][ratio]
            )
            if sides != Counter(
                {"MAJ": expected_majority, "MIN": expected_minority}
            ):
                raise ValueError(
                    f"{entity['entity_id']} {ratio} document sides are malformed: "
                    f"{dict(sides)}"
                )


def claim_label_mapping(entity, ratio, run_seed=20260714, layout_index=1):
    """Deterministically counterbalance majority/minority across Claim A/B.

    Alternating both entity number and ratio index makes the complete 75 x 6
    design exactly balanced. Seed/layout parity allows reproducible alternate
    mappings while retaining that full-design balance.
    """
    match = re.search(r"(\d+)$", str(entity["entity_id"]))
    if match:
        entity_index = int(match.group(1))
    else:
        entity_index = int.from_bytes(
            hashlib.sha256(str(entity["entity_id"]).encode("utf-8")).digest()[:4],
            "big",
        )
    try:
        ratio_index = RATIO_COUNTERBALANCE_ORDER.index(ratio)
    except ValueError:
        ratio_index = int.from_bytes(
            hashlib.sha256(str(ratio).encode("utf-8")).digest()[:4], "big"
        )
    try:
        seed_parity = int(run_seed) & 1
    except (TypeError, ValueError):
        seed_parity = hashlib.sha256(str(run_seed).encode("utf-8")).digest()[0] & 1
    majority_is_a = (
        entity_index + ratio_index + int(layout_index) + seed_parity
    ) % 2 == 0
    if majority_is_a:
        return {
            "claim_a_value": str(entity["majority_value"]),
            "claim_b_value": str(entity["minority_value"]),
            "claim_a_side": "MAJ",
            "claim_b_side": "MIN",
        }
    return {
        "claim_a_value": str(entity["minority_value"]),
        "claim_b_value": str(entity["majority_value"]),
        "claim_a_side": "MIN",
        "claim_b_side": "MAJ",
    }


def _contains_value(text, value):
    """Match a value as a complete token/phrase, not inside another value."""
    value = str(value)
    if not value:
        return False
    prefix = r"(?<!\w)" if value[0].isalnum() else ""
    suffix = r"(?!\w)" if value[-1].isalnum() else ""
    return re.search(prefix + re.escape(value) + suffix, str(text)) is not None


def document_side(text, majority_value, minority_value):
    """Return MAJ/MIN/UNK using exact value matching.

    If one value is a phrase containing the other (for example ``$100`` and
    ``$1000`` or ``Kansas`` and ``Kansas City``), the longer matched value wins.
    """
    matches = []
    for side, value in (("MAJ", majority_value), ("MIN", minority_value)):
        if _contains_value(text, value):
            matches.append((side, len(str(value))))
    if not matches:
        return "UNK"
    if len(matches) == 1:
        return matches[0][0]
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[0][0] if matches[0][1] != matches[1][1] else "UNK"


def build_prompt(entity, ratio, strategy="standard", trial_idx=1,
                 run_seed=20260714, ask_inline_confidence=True,
                 return_core=False):
    """Build one primary prompt.

    ``trial_idx`` controls document layout only. The run loop builds this once
    per entity/ratio and sends the exact resulting string three times, so
    repeated-sampling agreement is not confounded by different document order.
    """
    docs = list(entity["documents"][ratio])
    
    # Shuffle documents using the separately logged layout index. All three
    # primary samples share this exact layout and byte-identical prompt.
    shuffle_rng = py_random.Random(f"{run_seed}-{entity['entity_id']}-{ratio}-trial-{trial_idx}")
    shuffle_rng.shuffle(docs)
    
    doc_blocks = []
    doc_positions = []
    for i, d in enumerate(docs, start=1):
        doc_blocks.append(f"Document {i} (source: {d['source']}):\n{d['text']}")
        doc_positions.append(document_side(
            d["text"], entity["majority_value"], entity["minority_value"]
        ))
            
    prompt_core = PROMPT_CORE_TEMPLATE.format(
        documents="\n\n".join(doc_blocks), question=entity["question"]
    )
    mapping = claim_label_mapping(entity, ratio, run_seed, trial_idx)
    if ratio == "4:0":
        # Preserve the unanimous-context control: do not expose the absent
        # minority value or request a distribution in this condition.
        instruction = PRIMARY_UNANIMOUS_ANSWER_ONLY
    elif ask_inline_confidence:
        instruction = PRIMARY_WITH_CONFIDENCE.format(
            claim_a_value=mapping["claim_a_value"],
            claim_b_value=mapping["claim_b_value"],
        )
    else:
        # Structurally matched answer-only control. It sees the same Claim A/B
        # labels as the treatment, with only the distribution request omitted.
        instruction = PRIMARY_WITHOUT_CONFIDENCE.format(
            claim_a_value=mapping["claim_a_value"],
            claim_b_value=mapping["claim_b_value"],
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


def _parse_probability(value, field_name="probability"):
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric")
    number = float(value)
    if not number.is_integer() or not 0 <= number <= 100:
        raise ValueError(f"{field_name} must be an integer from 0 to 100")
    return int(number)


def parse_primary_response(raw, expect_inline_confidence=True,
                           claim_mapping=None):
    """Parse and validate the primary answer/distribution JSON."""
    probability_fields = (
        "p_claim_a", "p_claim_b", "p_indeterminate", "p_sources_conflict"
    )
    parsed = {
        "answer": "", "p_claim_a": "", "p_claim_b": "",
        "p_indeterminate": "", "p_sources_conflict": "",
        "p_majority": "", "p_minority": "", "format_error": "",
    }
    obj = extract_json_object(raw)
    if obj is None:
        parsed["format_error"] = "no valid JSON object"
        return parsed
    parsed["answer"] = str(obj.get("answer", "")).strip()
    if not parsed["answer"]:
        parsed["format_error"] = "JSON object has no answer"

    supplied_probability_fields = [field for field in probability_fields if field in obj]
    if expect_inline_confidence:
        errors = [parsed["format_error"]] if parsed["format_error"] else []
        for field in probability_fields:
            if field not in obj:
                errors.append(f"missing {field}")
                continue
            try:
                parsed[field] = _parse_probability(obj[field], field)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        sum_fields = ("p_claim_a", "p_claim_b", "p_indeterminate")
        if all(parsed[field] != "" for field in sum_fields):
            total = sum(parsed[field] for field in sum_fields)
            if total != 100:
                errors.append(
                    "p_claim_a + p_claim_b + p_indeterminate must sum exactly to 100"
                )
        if errors:
            parsed["format_error"] = "; ".join(errors)
        if claim_mapping and all(
            parsed[field] != "" for field in ("p_claim_a", "p_claim_b")
        ):
            if claim_mapping["claim_a_side"] == "MAJ":
                parsed["p_majority"] = parsed["p_claim_a"]
                parsed["p_minority"] = parsed["p_claim_b"]
            else:
                parsed["p_majority"] = parsed["p_claim_b"]
                parsed["p_minority"] = parsed["p_claim_a"]
    elif supplied_probability_fields:
        control_error = (
            "control response unexpectedly reported a probability distribution"
        )
        parsed["format_error"] = "; ".join(
            error for error in (parsed["format_error"], control_error) if error
        )
    return parsed


def parse_posthoc_response(raw):
    obj = extract_json_object(raw)
    if obj is None:
        return "", "no valid JSON object"
    # ``probability_correct`` is accepted only to read interrupted legacy runs;
    # every new prompt and CSV column uses the truth-neutral field below.
    field = (
        "confidence_best_resolution"
        if "confidence_best_resolution" in obj else "probability_correct"
    )
    try:
        return _parse_probability(obj.get(field), field), ""
    except (TypeError, ValueError) as exc:
        return "", str(exc)


def parse_response(raw):
    """Backward-compatible parser returning ``(answer, confidence)``."""
    parsed = parse_primary_response(raw, expect_inline_confidence=False)
    obj = extract_json_object(raw) or {}
    legacy = obj.get("confidence_best_resolution", obj.get("probability_correct", ""))
    try:
        legacy = _parse_probability(legacy) if legacy != "" else ""
    except (TypeError, ValueError):
        legacy = ""
    return parsed["answer"], legacy


def select_no_inline_confidence_ids(entities, count, seed):
    """Select a deterministic control stratified by domain and attribute.

    With the production 30/45 domain split and count=10 this allocates four
    banking and six general controls. Each of the four attributes per domain
    receives at least one control; the two remaining general slots are assigned
    proportionally and deterministically.
    """
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
    for domain, group in strata.items():
        domain_slots = allocation[domain]
        attribute_groups = defaultdict(list)
        for entity in group:
            attribute_groups[entity.get("attribute", "")].append(entity)
        attribute_allocation = {attribute: 0 for attribute in attribute_groups}
        if domain_slots >= len(attribute_groups):
            attribute_allocation = {
                attribute: 1 for attribute in attribute_groups
            }
        remaining_slots = domain_slots - sum(attribute_allocation.values())
        targets = {
            attribute: domain_slots * len(attribute_group) / len(group)
            for attribute, attribute_group in attribute_groups.items()
        }
        while remaining_slots > 0:
            eligible = [
                attribute for attribute, attribute_group in attribute_groups.items()
                if attribute_allocation[attribute] < len(attribute_group)
            ]
            if not eligible:
                break
            chosen = min(
                eligible,
                key=lambda attribute: (
                    -(targets[attribute] - attribute_allocation[attribute]),
                    hashlib.sha256(
                        f"{seed}|attribute-allocation|{domain}|{attribute}".encode(
                            "utf-8"
                        )
                    ).digest(),
                ),
            )
            attribute_allocation[chosen] += 1
            remaining_slots -= 1

        for attribute, attribute_group in attribute_groups.items():
            ranked = sorted(
                attribute_group,
                key=lambda entity: hashlib.sha256(
                    f"{seed}|no-inline-confidence|{entity['entity_id']}".encode(
                        "utf-8"
                    )
                ).digest(),
            )
            selected.update(
                entity["entity_id"]
                for entity in ranked[:attribute_allocation[attribute]]
            )
    return selected


def score_primary_row(row):
    """Use the shared scorer for category and the two RQ4 diagnostics."""
    visualizations_dir = str(REPO_ROOT / "visualizations")
    if visualizations_dir not in sys.path:
        sys.path.insert(0, visualizations_dir)
    from common import score_response
    return score_response(row)


def classify_primary_row(row):
    """Backward-compatible category-only wrapper."""
    return score_primary_row(row)["category"]


def summarize_repeats(rows):
    categories = [row["response_category"] for row in rows]
    scored = [category for category in categories if category != "UNSCORED"]
    if not scored:
        return {
            "categories": categories, "n_scored": 0,
            "modal_category": "UNSCORED",
            "modal_count": 0, "self_consistency": "",
            "self_consistency_all_samples": 0.0, "modal_tie": 0,
            "representative": None,
        }
    counts = Counter(scored)
    modal_count = max(counts.values())
    tied = {category for category, value in counts.items() if value == modal_count}
    is_tie = len(tied) > 1
    modal_category = "TIE" if is_tie else next(iter(tied))
    representative = None if is_tie else next(
        row for row in rows if row["response_category"] == modal_category
    )
    return {
        "categories": categories,
        "n_scored": len(scored),
        "modal_category": modal_category,
        "modal_count": modal_count,
        "self_consistency": modal_count / len(scored),
        "self_consistency_all_samples": modal_count / len(rows),
        "modal_tie": int(is_tie),
        "representative": representative,
    }


# ---------------------------------------------------------------------------
# Model callers: each returns
# (raw_text, prompt_tokens, billed_completion_tokens, reasoning_tokens,
#  API-returned model id)

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


def gemini_usage_tokens(usage):
    """Split Gemini usage while keeping thoughts in billed completion tokens."""
    if not usage:
        return "", "", ""
    prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
    visible_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    reasoning_tokens = int(getattr(usage, "thoughts_token_count", 0) or 0)
    return prompt_tokens, visible_tokens + reasoning_tokens, reasoning_tokens


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
    ptok, ctok, rtok = gemini_usage_tokens(resp.usage_metadata)
    returned_model = getattr(resp, "model_version", "") or model_id
    return text, ptok, ctok, rtok, returned_model


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


def openrouter_max_tokens(model_id):
    """Leave Claude visible-output room beyond its reasoning-token budget."""
    mid = model_id.lower()
    if "claude" in mid or "anthropic" in mid:
        return OPENROUTER_CLAUDE_MAX_TOKENS
    return OPENROUTER_DEFAULT_MAX_TOKENS


def openrouter_reasoning_tokens(usage):
    """Read OpenRouter's optional reasoning-token detail without double-counting."""
    if not usage:
        return ""
    details = getattr(usage, "completion_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
    if details is None:
        return ""
    value = (
        details.get("reasoning_tokens")
        if isinstance(details, dict)
        else getattr(details, "reasoning_tokens", None)
    )
    return int(value) if value is not None else ""


def call_openrouter(client, model_id, prompt, system_prompt=SYSTEM_PROMPT,
                    temperature=1.0):
    # OpenRouter exposes an OpenAI-compatible chat completions endpoint, so this
    # uses the standard OpenAI SDK (which retries 429/5xx/connection errors
    # internally via max_retries) pointed at OpenRouter's base_url. `reasoning`
    # is an OpenRouter-specific field, not part of the OpenAI schema, so it goes
    # through extra_body rather than a typed SDK parameter.
    resp = client.chat.completions.create(
        model=model_id,
        max_tokens=openrouter_max_tokens(model_id),
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        extra_body={"reasoning": openrouter_reasoning_param(model_id)},
    )
    usage = resp.usage
    return (
        resp.choices[0].message.content or "",
        usage.prompt_tokens if usage else "",
        # OpenRouter completion_tokens already includes reasoning tokens.
        usage.completion_tokens if usage else "",
        openrouter_reasoning_tokens(usage),
        getattr(resp, "model", "") or model_id,
    )


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
        claim_a_match = re.search(r"^Claim A: (.*)$", prompt, re.MULTILINE)
        majority_is_a = bool(
            claim_a_match and claim_a_match.group(1).strip() == str(answer)
        )

        def add_mock_distribution(payload):
            payload.update(
                p_claim_a=90 if majority_is_a else 5,
                p_claim_b=5 if majority_is_a else 90,
                p_indeterminate=5,
                p_sources_conflict=80,
            )

        if phase == "posthoc":
            raw = json.dumps({"confidence_best_resolution": 80})
        elif strategy == "cot":
            payload = {"answer": answer}
            if ask_inline_confidence:
                add_mock_distribution(payload)
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
                add_mock_distribution(payload)
            raw = json.dumps(payload)
            
        sys_prompt = (
            POSTHOC_SYSTEM_PROMPT if phase == "posthoc"
            else SYSTEM_PROMPT_COT if strategy == "cot"
            else SYSTEM_PROMPT
        )
        return (
            raw, len(sys_prompt + prompt) // 4, len(raw) // 4, 0,
            f"{self.provider}-MOCK",
        )


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
        help="deterministic entity arm omitting conflict-ratio distributions "
             "(default: 37; all 4:0 prompts are answer-only)",
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
    try:
        validate_protocol_dataset(dataset)
    except ValueError as exc:
        ap.error(str(exc))
    full_entities = dataset["entities"]
    dataset_fingerprint = dataset_sha256()

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
        f"Distribution-request assignment control: {len(no_inline_ids)} of "
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


    planned_primary = len(entities) * len(ratios) * len(callers) * args.trials
    planned_posthoc_max = len(entities) * len(ratios) * len(callers)
    planned_total = planned_primary + planned_posthoc_max
    done = 0
    primary_done = 0
    posthoc_done = 0
    posthoc_skipped = 0
    hard_errors = 0
    primary_errors = 0
    primary_format_errors = 0
    distributions_expected = 0
    valid_distributions = 0
    usage = defaultdict(lambda: [0, 0, 0])
    condition_rows = []

    def base_raw_row(entity, ratio, prompt, doc_positions, confidence_condition,
                     distribution_assigned, ask_inline, slot, phase,
                     trial_index=""):
        spec = callers[slot]
        mapping = (
            claim_label_mapping(
                entity, ratio, run_seed=run_seed, layout_index=args.layout_index
            )
            if ratio != "4:0" else {
                "claim_a_value": "", "claim_b_value": "",
                "claim_a_side": "", "claim_b_side": "",
            }
        )
        return {
            "run_id": run_id,
            "protocol_version": PROTOCOL_VERSION,
            "dataset_sha256": dataset_fingerprint,
            "run_seed": run_seed,
            "layout_index": args.layout_index,
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
            "distribution_request_assigned": int(distribution_assigned),
            "inline_confidence_requested": int(ask_inline),
            "model_slot": slot,
            "model_provider": spec["provider"],
            "model_id": spec["model_id"],
            "returned_model_id": "",
            "question": entity["question"],
            "majority_value": entity["majority_value"],
            "minority_value": entity["minority_value"],
            "claim_a_value": mapping["claim_a_value"],
            "claim_b_value": mapping["claim_b_value"],
            "claim_a_side": mapping["claim_a_side"],
            "claim_b_side": mapping["claim_b_side"],
            "response_category": "",
            "mentions_conflict": "",
            "abstained": "",
            "raw_response": "",
            "parsed_answer": "",
            "p_claim_a": "",
            "p_claim_b": "",
            "p_indeterminate": "",
            "p_sources_conflict": "",
            "p_majority": "",
            "p_minority": "",
            "confidence_best_resolution": "",
            "confidence_scale": "",
            "format_error": "",
            "prompt_tokens": "",
            "completion_tokens": "",
            "reasoning_tokens": "",
            "error": "",
        }

    def record_usage(model_id, prompt_tokens, completion_tokens):
        values = usage[model_id]
        values[0] += 1
        values[1] += int(prompt_tokens or 0)
        values[2] += int(completion_tokens or 0)

    # Raw rows are flushed after every billed call. Condition rows are flushed
    # as soon as each modal/post-hoc result is available.
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
            distribution_assigned = entity["entity_id"] not in no_inline_ids
            for ratio in ratios:
                ask_inline = distribution_assigned and ratio != "4:0"
                if ratio == "4:0":
                    confidence_condition = "unanimous_answer_only_plus_posthoc"
                    claim_mapping = None
                else:
                    confidence_condition = (
                        "rich_distribution_plus_posthoc" if ask_inline
                        else "matched_answer_only_plus_posthoc_control"
                    )
                    claim_mapping = claim_label_mapping(
                        entity, ratio, run_seed=run_seed,
                        layout_index=args.layout_index,
                    )
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
                            confidence_condition, distribution_assigned,
                            ask_inline, slot, "primary", trial_index,
                        )
                        try:
                            raw, ptok, ctok, rtok, returned_model = (
                                futures[slot].result()
                            )
                            row.update(
                                raw_response=raw, prompt_tokens=ptok,
                                completion_tokens=ctok,
                                reasoning_tokens=rtok,
                                returned_model_id=returned_model,
                            )
                            record_usage(spec["model_id"], ptok, ctok)
                            if not model_family_matches(slot, returned_model):
                                row["error"] = (
                                    "model_mismatch: requested "
                                    f"{spec['model_id']}, API returned {returned_model}"
                                )
                            parsed = parse_primary_response(
                                raw, ask_inline, claim_mapping=claim_mapping
                            )
                            row.update(
                                parsed_answer=parsed["answer"],
                                p_claim_a=parsed["p_claim_a"],
                                p_claim_b=parsed["p_claim_b"],
                                p_indeterminate=parsed["p_indeterminate"],
                                p_sources_conflict=parsed["p_sources_conflict"],
                                p_majority=parsed["p_majority"],
                                p_minority=parsed["p_minority"],
                                confidence_scale=(
                                    "rich_distribution_0-100"
                                    if parsed["p_claim_a"] != "" else ""
                                ),
                                format_error=parsed["format_error"],
                            )
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                        score = score_primary_row(row)
                        row.update(
                            response_category=score["category"],
                            mentions_conflict=score["mentions_conflict"],
                            abstained=score["abstained"],
                        )
                        if row["error"]:
                            hard_errors += 1
                            primary_errors += 1
                        if row["format_error"]:
                            primary_format_errors += 1
                        if ask_inline:
                            distributions_expected += 1
                            if not row["error"] and not row["format_error"]:
                                valid_distributions += 1
                        raw_writer.writerow(row)
                        raw_file.flush()
                        repeated[slot].append(row)
                        done += 1
                        primary_done += 1
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
                    representative = summary["representative"]
                    modal_answer = (
                        representative["parsed_answer"] if representative else ""
                    )
                    if summary["modal_category"] in ("TIE", "UNSCORED") or not modal_answer:
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
                    confidence_best_resolution = ""
                    posthoc_raw = ""
                    posthoc_ptok = ""
                    posthoc_ctok = ""
                    posthoc_rtok = ""
                    posthoc_returned_model = ""
                    posthoc_error = ""
                    posthoc_status = ""
                    was_posthoc_skipped = 0
                    if slot not in posthoc_futures:
                        posthoc_skipped += 1
                        was_posthoc_skipped = 1
                        if summary["modal_category"] == "TIE":
                            posthoc_status = "skipped_modal_tie"
                        elif summary["modal_category"] == "UNSCORED":
                            posthoc_status = "skipped_all_unscored"
                        else:
                            posthoc_status = "skipped_no_modal_answer"
                    else:
                        posthoc_prompt = posthoc_prompts[slot]
                        row = base_raw_row(
                            entity, ratio, posthoc_prompt, doc_positions,
                            confidence_condition, distribution_assigned,
                            ask_inline, slot, "posthoc",
                        )
                        try:
                            (
                                posthoc_raw, posthoc_ptok, posthoc_ctok,
                                posthoc_rtok, returned_model,
                            ) = posthoc_futures[slot].result()
                            row.update(
                                raw_response=posthoc_raw,
                                prompt_tokens=posthoc_ptok,
                                completion_tokens=posthoc_ctok,
                                reasoning_tokens=posthoc_rtok,
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
                                confidence_best_resolution=probability,
                                confidence_scale=(
                                    "0-100_best_resolution"
                                    if probability != "" else ""
                                ),
                                format_error=format_error,
                            )
                            if row["error"]:
                                posthoc_error = row["error"]
                                posthoc_status = "error"
                            elif format_error:
                                posthoc_error = format_error
                                posthoc_status = "format_error"
                            else:
                                confidence_best_resolution = probability
                                posthoc_status = "completed"
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            posthoc_error = row["error"]
                            posthoc_status = "error"
                        if row["error"]:
                            hard_errors += 1
                        raw_writer.writerow(row)
                        raw_file.flush()
                        done += 1
                        posthoc_done += 1
                        print(
                            f"[{done}/{planned_total}] {entity['entity_id']} "
                            f"{ratio} posthoc {slot} ({spec['model_id']})"
                            f"{' ERROR' if posthoc_error else ''}"
                        )

                    distributions = [
                        {
                            "p_claim_a": int(primary_row["p_claim_a"]),
                            "p_claim_b": int(primary_row["p_claim_b"]),
                            "p_indeterminate": int(primary_row["p_indeterminate"]),
                            "p_sources_conflict": int(primary_row["p_sources_conflict"]),
                            "p_majority": int(primary_row["p_majority"]),
                            "p_minority": int(primary_row["p_minority"]),
                        }
                        for primary_row in repeated[slot]
                        if ask_inline and not primary_row["format_error"]
                        and not primary_row["error"] and all(
                            primary_row[field] != "" for field in (
                                "p_claim_a", "p_claim_b", "p_indeterminate",
                                "p_sources_conflict", "p_majority", "p_minority",
                            )
                        )
                    ]
                    inline_p_majority = [d["p_majority"] for d in distributions]
                    inline_p_minority = [d["p_minority"] for d in distributions]
                    inline_p_indeterminate = [d["p_indeterminate"] for d in distributions]
                    inline_p_sources_conflict = [d["p_sources_conflict"] for d in distributions]

                    n_primary_errors = sum(bool(r["error"]) for r in repeated[slot])
                    n_primary_format_errors = sum(
                        bool(r["format_error"]) for r in repeated[slot]
                    )
                    conflict_mention_count = sum(
                        int(r["mentions_conflict"] or 0) for r in repeated[slot]
                        if r["response_category"] != "UNSCORED"
                    )
                    abstention_count = sum(
                        int(r["abstained"] or 0) for r in repeated[slot]
                        if r["response_category"] != "UNSCORED"
                    )
                    diagnostic_denominator = summary["n_scored"]
                    primary_reasoning = [
                        int(r["reasoning_tokens"])
                        for r in repeated[slot] if r["reasoning_tokens"] != ""
                    ]
                    all_reasoning = list(primary_reasoning)
                    if posthoc_rtok != "":
                        all_reasoning.append(int(posthoc_rtok))

                    def mean_or_blank(values):
                        return round(sum(values) / len(values), 2) if values else ""

                    modal_category = summary["modal_category"]
                    modal_answer = (
                        summary["representative"]["parsed_answer"]
                        if summary["representative"] else ""
                    )
                    returned_model_ids = {
                        row["returned_model_id"] for row in repeated[slot]
                        if row["returned_model_id"]
                    }
                    if posthoc_returned_model:
                        returned_model_ids.add(posthoc_returned_model)
                    condition = {
                        "run_id": run_id,
                        "protocol_version": PROTOCOL_VERSION,
                        "dataset_sha256": dataset_fingerprint,
                        "run_seed": run_seed,
                        "layout_index": args.layout_index,
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
                        "distribution_request_assigned": int(distribution_assigned),
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
                        "claim_a_value": (
                            claim_mapping["claim_a_value"] if claim_mapping else ""
                        ),
                        "claim_b_value": (
                            claim_mapping["claim_b_value"] if claim_mapping else ""
                        ),
                        "claim_a_side": (
                            claim_mapping["claim_a_side"] if claim_mapping else ""
                        ),
                        "claim_b_side": (
                            claim_mapping["claim_b_side"] if claim_mapping else ""
                        ),
                        "response_categories": json.dumps(summary["categories"]),
                        "n_samples": len(repeated[slot]),
                        "n_scored": summary["n_scored"],
                        "n_primary_errors": n_primary_errors,
                        "n_primary_format_errors": n_primary_format_errors,
                        "n_valid_distributions": len(distributions),
                        "distribution_compliance": (
                            round(len(distributions) / len(repeated[slot]), 4)
                            if ask_inline else ""
                        ),
                        "modal_category": modal_category,
                        "modal_count": summary["modal_count"],
                        "conflict_mention_count": conflict_mention_count,
                        "conflict_mention_rate": (
                            round(conflict_mention_count / diagnostic_denominator, 4)
                            if diagnostic_denominator else ""
                        ),
                        "abstention_count": abstention_count,
                        "abstention_rate": (
                            round(abstention_count / diagnostic_denominator, 4)
                            if diagnostic_denominator else ""
                        ),
                        "self_consistency": summary["self_consistency"],
                        "self_consistency_all_samples": summary[
                            "self_consistency_all_samples"
                        ],
                        "modal_tie": summary["modal_tie"],
                        "modal_answer": modal_answer,
                        "inline_distributions": json.dumps(distributions),
                        "inline_p_majority": json.dumps(inline_p_majority),
                        "inline_p_minority": json.dumps(inline_p_minority),
                        "inline_p_indeterminate": json.dumps(inline_p_indeterminate),
                        "inline_p_sources_conflict": json.dumps(inline_p_sources_conflict),
                        "mean_p_majority": mean_or_blank(inline_p_majority),
                        "mean_p_minority": mean_or_blank(inline_p_minority),
                        "mean_p_indeterminate": mean_or_blank(inline_p_indeterminate),
                        "mean_p_sources_conflict": mean_or_blank(
                            inline_p_sources_conflict
                        ),
                        "confidence_best_resolution": confidence_best_resolution,
                        "posthoc_raw_response": posthoc_raw,
                        "primary_reasoning_tokens": json.dumps(primary_reasoning),
                        "posthoc_reasoning_tokens": posthoc_rtok,
                        "reasoning_tokens": (
                            sum(all_reasoning) if all_reasoning else ""
                        ),
                        "posthoc_status": posthoc_status,
                        "posthoc_skipped": was_posthoc_skipped,
                        "posthoc_prompt_tokens": posthoc_ptok,
                        "posthoc_completion_tokens": posthoc_ctok,
                        "posthoc_error": posthoc_error,
                    }
                    condition_rows.append(condition)
                    condition_writer.writerow(condition)
                    condition_file.flush()


    print(f"\nDone. {done} live/mock API calls completed.")
    print(f"Primary calls: {primary_done}/{planned_primary} completed")
    print(
        f"Post-hoc conditions: {posthoc_done} calls completed + "
        f"{posthoc_skipped} intentionally skipped = "
        f"{posthoc_done + posthoc_skipped}/{planned_posthoc_max} accounted for"
    )
    print(
        f"Maximum call plan accounted for: {done} actual + "
        f"{posthoc_skipped} intentional skips = {done + posthoc_skipped}/"
        f"{planned_total}"
    )
    print(
        f"Errors: {hard_errors} hard/model-routing total; "
        f"{primary_errors} primary API/routing; "
        f"{primary_format_errors} primary format"
    )
    if distributions_expected:
        print(
            f"Distribution compliance: {valid_distributions}/"
            f"{distributions_expected} "
            f"({valid_distributions / distributions_expected:.1%})"
        )
    else:
        print("Distribution compliance: n/a (no distributions requested)")
    print(f"Raw call log: {out_path}")
    print(f"Condition results: {condition_path}")

    summarize_usage(usage)


if __name__ == "__main__":
    main()
