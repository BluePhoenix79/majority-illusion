"""Safely resume and merge the interrupted final Standard experiment.

This utility is intentionally separate from :mod:`run_experiment`.  It treats
the committed partial CSVs as immutable source evidence, audits every retained
condition against its raw call rows, reruns only *whole* invalid/missing
conditions into fresh attempt files, and publishes a new pair of CSVs only
after the complete 75 x 6 x 3 factorial passes validation.

Typical use::

    python harness/salvage_standard_run.py --audit-only
    python harness/salvage_standard_run.py

An interrupted invocation is safe to repeat.  Completely valid chunk attempts
are reused; partial or invalid attempts are never appended to or overwritten.
The original source CSVs are hash-checked before and after recovery.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:  # Works both as ``python -m harness...`` and as a direct script.
    from harness import run_experiment as experiment
except ModuleNotFoundError:  # pragma: no cover - direct-script fallback
    import run_experiment as experiment


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_RAW = REPO_ROOT / "results" / "run_final_standard_raw.csv"
DEFAULT_BASE_CONDITIONS = REPO_ROOT / "results" / "conditions_final_standard.csv"
DEFAULT_OUTPUT_RAW = REPO_ROOT / "results" / "run_final_standard_salvaged_raw.csv"
DEFAULT_OUTPUT_CONDITIONS = (
    REPO_ROOT / "results" / "conditions_final_standard_salvaged.csv"
)
DEFAULT_MANIFEST = REPO_ROOT / "results" / "salvage_standard_manifest.json"
DEFAULT_WORK_DIR = REPO_ROOT / "results" / "salvage_standard_work"

TRIALS = 3
TEMPERATURE = 1.0
LAYOUT_INDEX = 1
NO_INLINE_CONFIDENCE_ENTITIES = 37
STRATEGY = "standard"
FORMAT_RETRIES = experiment.DEFAULT_FORMAT_RETRIES
MODEL_ORDER = ("gemini", "deepseek", "claude")


class SalvageError(RuntimeError):
    """Raised when recovery cannot be proven safe."""


@dataclass(frozen=True, order=True)
class ConditionKey:
    entity_id: str
    ratio: str
    model_slot: str

    def label(self) -> str:
        return f"{self.entity_id}/{self.ratio}/{self.model_slot}"


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    requested_id: str
    returned_id: str


@dataclass
class StudyDesign:
    dataset: dict
    entities: list[dict]
    entity_by_id: dict[str, dict]
    ratios: tuple[str, ...]
    run_seed: int
    dataset_sha256: str
    no_inline_ids: set[str]
    expected_keys: set[ConditionKey]


@dataclass
class AuditReport:
    raw_rows: list[dict]
    condition_rows: list[dict]
    valid_keys: set[ConditionKey]
    pending_keys: set[ConditionKey]
    issues_by_key: dict[ConditionKey, list[str]]
    unexpected_keys: set[ConditionKey]
    derived_corrections: list[dict]

    @property
    def valid_raw_rows(self) -> list[dict]:
        return [row for row in self.raw_rows if row_key(row) in self.valid_keys]

    @property
    def valid_condition_rows(self) -> list[dict]:
        return [
            row for row in self.condition_rows if row_key(row) in self.valid_keys
        ]


@dataclass(frozen=True)
class RecoveryChunk:
    index: int
    entity_id: str
    ratios: tuple[str, ...]
    model_slots: tuple[str, ...]
    keys: frozenset[ConditionKey]

    @property
    def slug(self) -> str:
        ratio_part = "-".join(ratio.replace(":", "x") for ratio in self.ratios)
        model_part = "-".join(self.model_slots)
        return f"{self.index:03d}_{self.entity_id}_{model_part}_{ratio_part}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(raw_path: Path, condition_path: Path) -> dict[str, str]:
    return {
        "raw_sha256": sha256_file(raw_path),
        "conditions_sha256": sha256_file(condition_path),
    }


def assert_source_hashes(
    raw_path: Path, condition_path: Path, expected: dict[str, str]
) -> None:
    actual = source_hashes(raw_path, condition_path)
    if actual != expected:
        raise SalvageError(
            "source CSVs changed during salvage; refusing to continue\n"
            f"expected={expected}\nactual={actual}"
        )


def load_design() -> StudyDesign:
    dataset = json.loads(experiment.DATA_PATH.read_text(encoding="utf-8"))
    experiment.validate_protocol_dataset(dataset)
    entities = dataset["entities"]
    ratios = tuple(dataset["ratios"])
    if len(entities) != 75 or len(ratios) != 6:
        raise SalvageError("salvage protocol requires exactly 75 entities and 6 ratios")
    run_seed = int(dataset.get("seed", 20260714))
    no_inline_ids = experiment.select_no_inline_confidence_ids(
        entities, NO_INLINE_CONFIDENCE_ENTITIES, run_seed
    )
    expected_keys = {
        ConditionKey(entity["entity_id"], ratio, slot)
        for entity in entities
        for ratio in ratios
        for slot in MODEL_ORDER
    }
    if len(expected_keys) != 75 * 6 * 3:
        raise SalvageError("dataset does not produce the expected 1,350 conditions")
    return StudyDesign(
        dataset=dataset,
        entities=entities,
        entity_by_id={entity["entity_id"]: entity for entity in entities},
        ratios=ratios,
        run_seed=run_seed,
        dataset_sha256=experiment.dataset_sha256(),
        no_inline_ids=no_inline_ids,
        expected_keys=expected_keys,
    )


def real_model_specs() -> dict[str, ModelSpec]:
    return {
        "gemini": ModelSpec(
            "gemini", experiment.DEFAULT_GEMINI_MODEL,
            experiment.DEFAULT_GEMINI_MODEL,
        ),
        "deepseek": ModelSpec(
            "openrouter", experiment.DEFAULT_DEEPSEEK_MODEL,
            experiment.DEFAULT_DEEPSEEK_MODEL,
        ),
        "claude": ModelSpec(
            "openrouter", experiment.DEFAULT_CLAUDE_MODEL,
            experiment.DEFAULT_CLAUDE_MODEL,
        ),
    }


def mock_model_specs() -> dict[str, ModelSpec]:
    return {
        slot: ModelSpec(
            spec.provider,
            spec.requested_id + "-MOCK",
            spec.returned_id + "-MOCK",
        )
        for slot, spec in real_model_specs().items()
    }


def _load_local_environment() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env", override=False)
    except ImportError:  # run_experiment will report missing live dependencies
        pass


def generation_contract(mock: bool) -> dict:
    """Fingerprint every non-row setting that can change model generation."""
    if mock:
        route = {"mode": "mock"}
    else:
        _load_local_environment()
        vertex = os.environ.get("GEMINI_USE_VERTEX", "").lower() in (
            "1", "true", "yes"
        )
        project_set = bool(os.environ.get("GEMINI_VERTEX_PROJECT"))
        location = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
        credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
        credentials_exist = bool(credentials and Path(credentials).is_file())
        openrouter_key_set = bool(os.environ.get("OPENROUTER_API_KEY"))
        if not (
            vertex and project_set and location == "global"
            and credentials_exist and openrouter_key_set
        ):
            raise SalvageError(
                "live salvage requires Gemini Vertex project auth at location="
                "global with an existing service-account credential file, plus "
                "an OpenRouter key"
            )
        route = {
            "mode": "vertex_project_auth",
            "project_configured": True,
            "location": "global",
            "service_account_file_exists": True,
            "openrouter_key_configured": True,
        }

    def text_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return {
        "run_experiment_sha256": sha256_file(Path(experiment.__file__).resolve()),
        "system_prompt_sha256": text_hash(experiment.SYSTEM_PROMPT),
        "posthoc_system_prompt_sha256": text_hash(
            experiment.POSTHOC_SYSTEM_PROMPT
        ),
        "gemini_max_output_tokens": experiment.GEMINI_MAX_OUTPUT_TOKENS,
        "openrouter_default_max_tokens": (
            experiment.OPENROUTER_DEFAULT_MAX_TOKENS
        ),
        "openrouter_claude_max_tokens": experiment.OPENROUTER_CLAUDE_MAX_TOKENS,
        "gemini_thinking_level": experiment.GEMINI_THINKING_LEVEL,
        "openrouter_anthropic_budget_tokens": (
            experiment.OPENROUTER_ANTHROPIC_BUDGET_TOKENS
        ),
        "openrouter_default_effort": experiment.OPENROUTER_DEFAULT_EFFORT,
        "gemini_max_total_attempts": experiment.MAX_RETRIES,
        "gemini_generic_400_extra_retries": (
            experiment.GENERIC_GEMINI_400_RETRIES
        ),
        "openrouter_sdk_retries": experiment.OPENROUTER_MAX_RETRIES,
        "route": route,
    }


def read_csv_exact(path: Path, expected_fields: list[str]) -> list[dict]:
    if not path.is_file():
        raise SalvageError(f"required CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise SalvageError(
                f"schema mismatch in {path}\n"
                f"expected={expected_fields}\nactual={reader.fieldnames}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise SalvageError(f"row has fields beyond the declared schema in {path}")
    return rows


def row_key(row: dict) -> ConditionKey:
    return ConditionKey(
        str(row.get("entity_id", "")),
        str(row.get("ratio", "")),
        str(row.get("model_slot", "")),
    )


def _integer(value, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not an integer: {value!r}") from exc
    if not number.is_integer():
        raise ValueError(f"{field} is not an integer: {value!r}")
    return int(number)


def _number(value, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc


def _json(value, field: str, expected_type):
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid JSON: {value!r}") from exc
    if not isinstance(parsed, expected_type):
        raise ValueError(f"{field} has the wrong JSON type")
    return parsed


def _same_number(actual, expected, tolerance=1e-9) -> bool:
    try:
        return math.isclose(float(actual), float(expected), abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def expected_condition_values(
    design: StudyDesign, key: ConditionKey, model_specs: dict[str, ModelSpec]
) -> dict:
    entity = design.entity_by_id[key.entity_id]
    distribution_assigned = key.entity_id not in design.no_inline_ids
    ask_inline = distribution_assigned and key.ratio != "4:0"
    prompt, doc_positions, prompt_core = experiment.build_prompt(
        entity,
        key.ratio,
        strategy=STRATEGY,
        trial_idx=LAYOUT_INDEX,
        run_seed=design.run_seed,
        ask_inline_confidence=ask_inline,
        return_core=True,
    )
    if key.ratio == "4:0":
        confidence_condition = "unanimous_answer_only_plus_posthoc"
        mapping = {
            "claim_a_value": "", "claim_b_value": "",
            "claim_a_side": "", "claim_b_side": "",
        }
    else:
        confidence_condition = (
            "rich_distribution_plus_posthoc" if ask_inline
            else "matched_answer_only_plus_posthoc_control"
        )
        mapping = experiment.claim_label_mapping(
            entity, key.ratio, design.run_seed, LAYOUT_INDEX
        )
    spec = model_specs[key.model_slot]
    return {
        "entity": entity,
        "distribution_assigned": distribution_assigned,
        "ask_inline": ask_inline,
        "prompt": prompt,
        "prompt_core": prompt_core,
        "prompt_hash": experiment.prompt_digest(prompt),
        "doc_positions": doc_positions,
        "confidence_condition": confidence_condition,
        "mapping": mapping,
        "spec": spec,
    }


def _validate_static_row(
    row: dict,
    key: ConditionKey,
    expected: dict,
    design: StudyDesign,
    *,
    is_condition: bool,
) -> list[str]:
    entity = expected["entity"]
    spec = expected["spec"]
    fields = {
        "protocol_version": experiment.PROTOCOL_VERSION,
        "dataset_sha256": design.dataset_sha256,
        "run_seed": str(design.run_seed),
        "layout_index": str(LAYOUT_INDEX),
        "entity_id": key.entity_id,
        "entity_name": str(entity["entity_name"]),
        "domain": str(entity["domain"]),
        "attribute": str(entity["attribute"]),
        "ratio": key.ratio,
        "n_docs": str(len(entity["documents"][key.ratio])),
        "strategy": STRATEGY,
        "doc_positions": json.dumps(expected["doc_positions"]),
        "confidence_condition": expected["confidence_condition"],
        "distribution_request_assigned": str(
            int(expected["distribution_assigned"])
        ),
        "inline_confidence_requested": str(int(expected["ask_inline"])),
        "model_slot": key.model_slot,
        "model_provider": spec.provider,
        "model_id": spec.requested_id,
        "question": str(entity["question"]),
        "majority_value": str(entity["majority_value"]),
        "minority_value": str(entity["minority_value"]),
        "claim_a_value": str(expected["mapping"]["claim_a_value"]),
        "claim_b_value": str(expected["mapping"]["claim_b_value"]),
        "claim_a_side": str(expected["mapping"]["claim_a_side"]),
        "claim_b_side": str(expected["mapping"]["claim_b_side"]),
    }
    issues = []
    for field, wanted in fields.items():
        if str(row.get(field, "")) != wanted:
            issues.append(f"{field}={row.get(field)!r}, expected {wanted!r}")
    if not str(row.get("run_id", "")).strip():
        issues.append("run_id is blank")
    timestamp = str(row.get("timestamp_utc", ""))
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        issues.append("timestamp_utc is not ISO-8601")
    try:
        positions = _json(row.get("doc_positions", ""), "doc_positions", list)
        if positions != expected["doc_positions"]:
            issues.append("doc_positions differs from deterministic layout")
    except ValueError as exc:
        issues.append(str(exc))
    if is_condition and row.get("prompt_hash") != expected["prompt_hash"]:
        issues.append("condition prompt_hash differs from deterministic primary prompt")
    return issues


def _compare_parsed_primary(
    row: dict, expected: dict, issues: list[str], prefix: str
) -> None:
    parsed = experiment.parse_primary_response(
        row.get("raw_response", ""),
        expected["ask_inline"],
        claim_mapping=(expected["mapping"] if expected["ask_inline"] else None),
    )
    for field in (
        "answer", "p_claim_a", "p_claim_b", "p_indeterminate",
        "p_sources_conflict", "p_majority", "p_minority", "format_error",
    ):
        csv_field = "parsed_answer" if field == "answer" else field
        if str(row.get(csv_field, "")) != str(parsed[field]):
            issues.append(f"{prefix} {csv_field} disagrees with reparsed response")
    if row.get("attempt_status") == "accepted" and parsed["format_error"]:
        issues.append(f"{prefix} accepted response does not satisfy JSON contract")


def validate_condition(
    condition: dict,
    raw_rows: list[dict],
    key: ConditionKey,
    design: StudyDesign,
    model_specs: dict[str, ModelSpec],
) -> list[str]:
    expected = expected_condition_values(design, key, model_specs)
    spec = expected["spec"]
    issues = _validate_static_row(
        condition, key, expected, design, is_condition=True
    )
    condition_run_id = str(condition.get("run_id", ""))

    for index, row in enumerate(raw_rows, start=1):
        prefix = f"raw row {index}"
        issues.extend(
            f"{prefix}: {message}" for message in _validate_static_row(
                row, key, expected, design, is_condition=False
            )
        )
        if str(row.get("run_id", "")) != condition_run_id:
            issues.append(f"{prefix}: run_id differs from condition row")
        if row.get("call_phase") not in {"primary", "posthoc"}:
            issues.append(f"{prefix}: invalid call_phase")
        expected_hash = expected["prompt_hash"]
        if row.get("call_phase") == "posthoc":
            expected_hash = experiment.prompt_digest(
                experiment.build_posthoc_prompt(
                    expected["prompt_core"], condition.get("modal_answer", "")
                )
            )
        if row.get("prompt_hash") != expected_hash:
            issues.append(f"{prefix}: prompt_hash is not deterministic")
        if row.get("error"):
            issues.append(f"{prefix}: API/model error present")
        if row.get("returned_model_id") != spec.returned_id:
            issues.append(
                f"{prefix}: returned_model_id={row.get('returned_model_id')!r}, "
                f"expected {spec.returned_id!r}"
            )

    primary = [row for row in raw_rows if row.get("call_phase") == "primary"]
    posthoc = [row for row in raw_rows if row.get("call_phase") == "posthoc"]
    terminal_primary = []
    for trial in range(1, TRIALS + 1):
        trial_rows = []
        for row in primary:
            try:
                if _integer(row.get("trial_index"), "trial_index") == trial:
                    trial_rows.append(row)
            except ValueError:
                pass
        try:
            trial_rows.sort(key=lambda row: _integer(row["attempt_index"], "attempt_index"))
        except ValueError as exc:
            issues.append(str(exc))
            continue
        attempts = []
        for row in trial_rows:
            try:
                attempts.append(_integer(row["attempt_index"], "attempt_index"))
            except ValueError as exc:
                issues.append(str(exc))
        if attempts != list(range(1, len(trial_rows) + 1)):
            issues.append(f"trial {trial}: attempt indexes are not contiguous from 1")
        if len(trial_rows) > FORMAT_RETRIES + 1:
            issues.append(
                f"trial {trial}: more than {FORMAT_RETRIES} format retries"
            )
        terminals = []
        for attempt_number, row in enumerate(trial_rows, start=1):
            prefix = f"trial {trial} attempt {attempt_number}"
            _compare_parsed_primary(row, expected, issues, prefix)
            try:
                trial_record = _integer(row.get("trial_record"), "trial_record")
            except ValueError as exc:
                issues.append(f"{prefix}: {exc}")
                continue
            if trial_record == 1:
                terminals.append(row)
            elif trial_record != 0:
                issues.append(f"{prefix}: trial_record is not 0/1")
            if attempt_number < len(trial_rows):
                if row.get("attempt_status") != "format_retry" or trial_record != 0:
                    issues.append(f"{prefix}: nonterminal attempt is not format_retry")
                if not row.get("format_error"):
                    issues.append(f"{prefix}: format_retry has no format_error")
            else:
                if row.get("attempt_status") != "accepted" or trial_record != 1:
                    issues.append(f"{prefix}: terminal primary is not accepted")
                if row.get("format_error") or row.get("error"):
                    issues.append(f"{prefix}: accepted primary has an error")
        if len(terminals) != 1:
            issues.append(f"trial {trial}: expected one terminal row, found {len(terminals)}")
        else:
            terminal_primary.append(terminals[0])

    for row in primary:
        try:
            trial = _integer(row.get("trial_index"), "trial_index")
        except ValueError:
            issues.append("primary row has non-integer trial_index")
            continue
        if trial not in range(1, TRIALS + 1):
            issues.append(f"primary row has unexpected trial_index {trial}")

    if len(terminal_primary) == TRIALS:
        terminal_primary.sort(key=lambda row: _integer(row["trial_index"], "trial_index"))
        for row in terminal_primary:
            if row.get("response_category") == "UNSCORED":
                issues.append("accepted primary response is UNSCORED")
            if row.get("response_category") == "OTHER":
                issues.append(
                    "accepted primary response is OTHER and requires manual review"
                )
            score = experiment.score_primary_row(row)
            if row.get("response_category") != score["category"]:
                issues.append("stored response_category disagrees with shared scorer")
            if str(row.get("mentions_conflict")) != str(score["mentions_conflict"]):
                issues.append("stored mentions_conflict disagrees with shared scorer")
            if str(row.get("abstained")) != str(score["abstained"]):
                issues.append("stored abstained disagrees with shared scorer")

        summary = experiment.summarize_repeats(terminal_primary)
        categories = [row["response_category"] for row in terminal_primary]
        try:
            recorded_categories = _json(
                condition.get("response_categories", ""),
                "response_categories", list,
            )
            if recorded_categories != categories:
                issues.append("response_categories disagrees with terminal raw rows")
        except ValueError as exc:
            issues.append(str(exc))
        integer_expectations = {
            "primary_api_attempts": len(primary),
            "primary_format_retries": sum(
                row.get("attempt_status") == "format_retry" for row in primary
            ),
            "n_samples": TRIALS,
            "n_scored": TRIALS,
            "n_primary_errors": 0,
            "n_primary_format_errors": 0,
            "modal_count": summary["modal_count"],
            "modal_tie": summary["modal_tie"],
            "conflict_mention_count": sum(
                _integer(row["mentions_conflict"], "mentions_conflict")
                for row in terminal_primary
            ),
            "abstention_count": sum(
                _integer(row["abstained"], "abstained")
                for row in terminal_primary
            ),
        }
        for field, wanted in integer_expectations.items():
            try:
                if _integer(condition.get(field), field) != wanted:
                    issues.append(f"{field} disagrees with raw rows")
            except ValueError as exc:
                issues.append(str(exc))
        if condition.get("modal_category") != summary["modal_category"]:
            issues.append("modal_category disagrees with raw rows")
        modal_answer = (
            summary["representative"]["parsed_answer"]
            if summary["representative"] else ""
        )
        # Preserve the answer that the post-hoc call actually evaluated.  A
        # later deterministic classifier correction can change which modal row
        # appears first without changing the modal category.  Repointing the
        # condition to that newly-first row would falsify the post-hoc prompt.
        if summary["modal_category"] == "TIE":
            if condition.get("modal_answer"):
                issues.append("modal-tie condition unexpectedly has a modal_answer")
        else:
            valid_modal_answers = {
                row["parsed_answer"] for row in terminal_primary
                if row["response_category"] == summary["modal_category"]
            }
            if condition.get("modal_answer") not in valid_modal_answers:
                issues.append(
                    "modal_answer is not an actually evaluated modal response"
                )
        for field, wanted in (
            ("self_consistency", summary["self_consistency"]),
            ("self_consistency_all_samples", summary["self_consistency_all_samples"]),
        ):
            if not _same_number(condition.get(field), wanted):
                issues.append(f"{field} disagrees with raw rows")
        conflict_count = integer_expectations["conflict_mention_count"]
        abstention_count = integer_expectations["abstention_count"]
        if not _same_number(
            condition.get("conflict_mention_rate"), round(conflict_count / TRIALS, 4)
        ):
            issues.append("conflict_mention_rate disagrees with raw rows")
        if not _same_number(
            condition.get("abstention_rate"), round(abstention_count / TRIALS, 4)
        ):
            issues.append("abstention_rate disagrees with raw rows")

        distributions = []
        if expected["ask_inline"]:
            for row in terminal_primary:
                distribution = {
                    field: _integer(row[field], field)
                    for field in (
                        "p_claim_a", "p_claim_b", "p_indeterminate",
                        "p_sources_conflict", "p_majority", "p_minority",
                    )
                }
                if (
                    distribution["p_claim_a"]
                    + distribution["p_claim_b"]
                    + distribution["p_indeterminate"]
                    != 100
                ):
                    issues.append("accepted distribution does not sum to 100")
                distributions.append(distribution)
                if row.get("confidence_scale") != "rich_distribution_0-100":
                    issues.append("rich response has wrong confidence_scale")
        else:
            for row in terminal_primary:
                for field in (
                    "p_claim_a", "p_claim_b", "p_indeterminate",
                    "p_sources_conflict", "p_majority", "p_minority",
                    "confidence_scale",
                ):
                    if row.get(field):
                        issues.append(f"answer-only response unexpectedly has {field}")
        try:
            recorded_distributions = _json(
                condition.get("inline_distributions", ""),
                "inline_distributions", list,
            )
            if recorded_distributions != distributions:
                issues.append("inline_distributions disagrees with raw rows")
        except ValueError as exc:
            issues.append(str(exc))
        distribution_fields = {
            "inline_p_majority": "p_majority",
            "inline_p_minority": "p_minority",
            "inline_p_indeterminate": "p_indeterminate",
            "inline_p_sources_conflict": "p_sources_conflict",
        }
        for csv_field, item_field in distribution_fields.items():
            values = [item[item_field] for item in distributions]
            try:
                if _json(condition.get(csv_field, ""), csv_field, list) != values:
                    issues.append(f"{csv_field} disagrees with raw rows")
            except ValueError as exc:
                issues.append(str(exc))
        expected_valid = TRIALS if expected["ask_inline"] else 0
        try:
            if _integer(condition.get("n_valid_distributions"), "n_valid_distributions") != expected_valid:
                issues.append("n_valid_distributions is wrong")
        except ValueError as exc:
            issues.append(str(exc))
        expected_compliance = "1.0" if expected["ask_inline"] else ""
        if condition.get("distribution_compliance") != expected_compliance:
            issues.append("distribution_compliance is wrong")
        mean_fields = {
            "mean_p_majority": "p_majority",
            "mean_p_minority": "p_minority",
            "mean_p_indeterminate": "p_indeterminate",
            "mean_p_sources_conflict": "p_sources_conflict",
        }
        for csv_field, item_field in mean_fields.items():
            if distributions:
                wanted = round(
                    sum(item[item_field] for item in distributions) / len(distributions),
                    2,
                )
                if not _same_number(condition.get(csv_field), wanted):
                    issues.append(f"{csv_field} disagrees with raw rows")
            elif condition.get(csv_field):
                issues.append(f"answer-only condition unexpectedly has {csv_field}")

        primary_reasoning = [
            _integer(row["reasoning_tokens"], "reasoning_tokens")
            for row in terminal_primary if row.get("reasoning_tokens") != ""
        ]
        try:
            if _json(
                condition.get("primary_reasoning_tokens", ""),
                "primary_reasoning_tokens", list,
            ) != primary_reasoning:
                issues.append("primary_reasoning_tokens disagrees with raw rows")
        except ValueError as exc:
            issues.append(str(exc))

    try:
        returned_ids = _json(
            condition.get("returned_model_ids", ""), "returned_model_ids", list
        )
        if returned_ids != [spec.returned_id]:
            issues.append("returned_model_ids does not equal the exact requested model")
    except ValueError as exc:
        issues.append(str(exc))

    modal_tie = condition.get("modal_category") == "TIE"
    if modal_tie:
        if posthoc:
            issues.append("modal-tie condition has post-hoc raw rows")
        required = {
            "posthoc_status": "skipped_modal_tie",
            "posthoc_skipped": "1",
            "posthoc_api_attempts": "0",
            "posthoc_format_retries": "0",
            "posthoc_error": "",
            "confidence_best_resolution": "",
        }
        for field, wanted in required.items():
            if condition.get(field) != wanted:
                issues.append(f"invalid modal-tie {field}")
        for field in (
            "posthoc_raw_response", "posthoc_prompt_tokens",
            "posthoc_completion_tokens", "posthoc_reasoning_tokens",
        ):
            if condition.get(field):
                issues.append(f"modal-tie {field} is not blank")
    else:
        try:
            posthoc.sort(key=lambda row: _integer(row["attempt_index"], "attempt_index"))
        except ValueError as exc:
            issues.append(str(exc))
        attempts = []
        for row in posthoc:
            try:
                attempts.append(_integer(row.get("attempt_index"), "attempt_index"))
            except ValueError as exc:
                issues.append(str(exc))
            if row.get("trial_index"):
                issues.append("post-hoc row has a trial_index")
        if attempts != list(range(1, len(posthoc) + 1)):
            issues.append("post-hoc attempt indexes are not contiguous from 1")
        if len(posthoc) > FORMAT_RETRIES + 1:
            issues.append(f"post-hoc has more than {FORMAT_RETRIES} format retries")
        terminals = [row for row in posthoc if row.get("trial_record") == "1"]
        if len(terminals) != 1:
            issues.append(f"expected one terminal post-hoc row, found {len(terminals)}")
        elif posthoc:
            terminal = terminals[0]
            for row in posthoc[:-1]:
                if row.get("attempt_status") != "format_retry" or row.get("trial_record") != "0":
                    issues.append("nonterminal post-hoc attempt is not format_retry")
            if terminal is not posthoc[-1]:
                issues.append("terminal post-hoc row is not the final attempt")
            if terminal.get("attempt_status") != "accepted":
                issues.append("terminal post-hoc row is not accepted")
            probability, format_error = experiment.parse_posthoc_response(
                terminal.get("raw_response", "")
            )
            if format_error or terminal.get("format_error"):
                issues.append("terminal post-hoc response is malformed")
            if str(terminal.get("confidence_best_resolution", "")) != str(probability):
                issues.append("post-hoc probability disagrees with reparsed response")
            checks = {
                "posthoc_status": "completed",
                "posthoc_skipped": "0",
                "posthoc_api_attempts": str(len(posthoc)),
                "posthoc_format_retries": str(
                    sum(row.get("attempt_status") == "format_retry" for row in posthoc)
                ),
                "posthoc_error": "",
                "confidence_best_resolution": str(probability),
                "posthoc_raw_response": terminal.get("raw_response", ""),
                "posthoc_prompt_tokens": terminal.get("prompt_tokens", ""),
                "posthoc_completion_tokens": terminal.get("completion_tokens", ""),
                "posthoc_reasoning_tokens": terminal.get("reasoning_tokens", ""),
            }
            for field, wanted in checks.items():
                if condition.get(field) != wanted:
                    issues.append(f"{field} disagrees with terminal post-hoc row")

    reasoning_values = [
        _integer(row["reasoning_tokens"], "reasoning_tokens")
        for row in terminal_primary if row.get("reasoning_tokens") != ""
    ]
    terminal_posthoc = [
        row for row in posthoc
        if row.get("trial_record") == "1"
        and row.get("attempt_status") == "accepted"
    ]
    if len(terminal_posthoc) == 1 and terminal_posthoc[0].get("reasoning_tokens") != "":
        reasoning_values.append(
            _integer(
                terminal_posthoc[0]["reasoning_tokens"], "reasoning_tokens"
            )
        )
    expected_total_reasoning = (
        str(sum(reasoning_values)) if reasoning_values else ""
    )
    if condition.get("reasoning_tokens") != expected_total_reasoning:
        issues.append("reasoning_tokens aggregate disagrees with terminal raw rows")

    return sorted(set(issues))


def audit_rows(
    raw_rows: list[dict],
    condition_rows: list[dict],
    design: StudyDesign,
    model_specs: dict[str, ModelSpec],
    expected_keys: set[ConditionKey] | frozenset[ConditionKey] | None = None,
) -> AuditReport:
    expected_keys = set(expected_keys or design.expected_keys)
    # Work on copies.  Re-scoring derived fields is allowed; source response
    # text, probabilities, tokens, timestamps, and provenance remain immutable.
    raw_rows = [dict(row) for row in raw_rows]
    condition_rows = [dict(row) for row in condition_rows]
    raw_by_key = defaultdict(list)
    conditions_by_key = defaultdict(list)
    for row in raw_rows:
        raw_by_key[row_key(row)].append(row)
    for row in condition_rows:
        conditions_by_key[row_key(row)].append(row)
    observed = set(raw_by_key) | set(conditions_by_key)
    unexpected = observed - expected_keys
    derived_corrections = []
    for key, rows in raw_by_key.items():
        conditions = conditions_by_key.get(key, [])
        if len(conditions) != 1:
            continue
        condition = conditions[0]
        terminal = []
        row_changes = []
        for row in rows:
            if (
                row.get("call_phase") != "primary"
                or row.get("trial_record") != "1"
                or row.get("attempt_status") != "accepted"
                or row.get("error")
                or row.get("format_error")
            ):
                continue
            score = experiment.score_primary_row(row)
            before = {
                field: row.get(field, "")
                for field in ("response_category", "mentions_conflict", "abstained")
            }
            after = {
                "response_category": str(score["category"]),
                "mentions_conflict": str(score["mentions_conflict"]),
                "abstained": str(score["abstained"]),
            }
            row.update(after)
            if before != after:
                row_changes.append({
                    "trial_index": row.get("trial_index", ""),
                    "before": before,
                    "after": after,
                })
            terminal.append(row)
        if not row_changes or len(terminal) != TRIALS:
            continue
        terminal.sort(key=lambda row: _integer(row["trial_index"], "trial_index"))
        summary = experiment.summarize_repeats(terminal)
        # Safe correction requires that the original post-hoc target still
        # belongs to the corrected modal category.  Otherwise the condition
        # must be rerun as a whole rather than retroactively changing its target.
        modal_answer = condition.get("modal_answer", "")
        modal_rows = [
            row for row in terminal
            if row["response_category"] == summary["modal_category"]
        ]
        can_preserve_posthoc = (
            summary["modal_category"] not in {"TIE", "UNSCORED"}
            and any(row["parsed_answer"] == modal_answer for row in modal_rows)
            and condition.get("posthoc_status") == "completed"
        )
        if not can_preserve_posthoc:
            continue
        conflict_count = sum(_integer(row["mentions_conflict"], "mentions_conflict") for row in terminal)
        abstention_count = sum(_integer(row["abstained"], "abstained") for row in terminal)
        condition.update({
            "response_categories": json.dumps(
                [row["response_category"] for row in terminal]
            ),
            "n_scored": str(summary["n_scored"]),
            "modal_category": str(summary["modal_category"]),
            "modal_count": str(summary["modal_count"]),
            "self_consistency": str(summary["self_consistency"]),
            "self_consistency_all_samples": str(
                summary["self_consistency_all_samples"]
            ),
            "modal_tie": str(summary["modal_tie"]),
            "conflict_mention_count": str(conflict_count),
            "conflict_mention_rate": str(round(conflict_count / TRIALS, 4)),
            "abstention_count": str(abstention_count),
            "abstention_rate": str(round(abstention_count / TRIALS, 4)),
        })
        derived_corrections.append({
            "key": key.label(),
            "raw_trial_corrections": row_changes,
            "preserved_modal_answer": modal_answer,
            "preserved_posthoc": True,
        })

    valid = set()
    issues_by_key = {}
    for key in sorted(expected_keys):
        issues = []
        conditions = conditions_by_key.get(key, [])
        if len(conditions) != 1:
            issues.append(f"expected one condition row, found {len(conditions)}")
        elif not raw_by_key.get(key):
            issues.append("condition has no raw rows")
        else:
            try:
                issues.extend(
                    validate_condition(
                        conditions[0], raw_by_key[key], key, design, model_specs
                    )
                )
            except Exception as exc:  # convert malformed evidence into a pending key
                issues.append(f"validation exception: {type(exc).__name__}: {exc}")
        if issues:
            issues_by_key[key] = sorted(set(issues))
        else:
            valid.add(key)
    for key in sorted(unexpected):
        issues_by_key[key] = ["unexpected condition key"]
    return AuditReport(
        raw_rows=raw_rows,
        condition_rows=condition_rows,
        valid_keys=valid,
        pending_keys=expected_keys - valid,
        issues_by_key=issues_by_key,
        unexpected_keys=unexpected,
        derived_corrections=derived_corrections,
    )


def audit_files(
    raw_path: Path,
    condition_path: Path,
    design: StudyDesign,
    model_specs: dict[str, ModelSpec],
    expected_keys: set[ConditionKey] | frozenset[ConditionKey] | None = None,
) -> AuditReport:
    raw_rows = read_csv_exact(raw_path, experiment.CSV_FIELDS)
    condition_rows = read_csv_exact(condition_path, experiment.CONDITION_FIELDS)
    return audit_rows(raw_rows, condition_rows, design, model_specs, expected_keys)


def plan_chunks(
    pending_keys: set[ConditionKey], design: StudyDesign
) -> list[RecoveryChunk]:
    """Create bounded entity-ratio chunks, grouping only the model slots.

    A chunk contains at most three complete conditions (roughly 12 logical
    calls).  If the harness exits nonzero, retrying a fresh chunk cannot waste
    an entity's other ratios and never cherry-picks individual samples.
    """
    chunks = []
    next_index = 1
    for entity in design.entities:
        entity_id = entity["entity_id"]
        for ratio in design.ratios:
            slots = tuple(
                slot for slot in MODEL_ORDER
                if ConditionKey(entity_id, ratio, slot) in pending_keys
            )
            if not slots:
                continue
            keys = frozenset(
                ConditionKey(entity_id, ratio, slot)
                for slot in slots
            )
            chunks.append(
                RecoveryChunk(
                    index=next_index,
                    entity_id=entity_id,
                    ratios=(ratio,),
                    model_slots=tuple(slots),
                    keys=keys,
                )
            )
            next_index += 1
    planned_keys = set().union(*(chunk.keys for chunk in chunks)) if chunks else set()
    if planned_keys != pending_keys:
        raise SalvageError("internal error: recovery chunks do not exactly cover pending keys")
    return chunks


def fixed_design_manifest(mock: bool) -> dict:
    return {
        "protocol_version": experiment.PROTOCOL_VERSION,
        "dataset_sha256": experiment.dataset_sha256(),
        "strategy": STRATEGY,
        "trials": TRIALS,
        "temperature": TEMPERATURE,
        "layout_index": LAYOUT_INDEX,
        "no_inline_confidence_entities": NO_INLINE_CONFIDENCE_ENTITIES,
        "format_retries": FORMAT_RETRIES,
        "mock": mock,
        "models": {
            slot: (mock_model_specs() if mock else real_model_specs())[slot].requested_id
            for slot in MODEL_ORDER
        },
        "generation_contract": generation_contract(mock),
    }


def chunk_command(
    chunk: RecoveryChunk,
    raw_path: Path,
    condition_path: Path,
    python_executable: Path | str,
    *,
    mock: bool,
) -> list[str]:
    command = [
        str(python_executable),
        str(REPO_ROOT / "harness" / "run_experiment.py"),
        "--entity-ids", chunk.entity_id,
        "--ratios", *chunk.ratios,
        "--models", *chunk.model_slots,
        "--gemini-model", experiment.DEFAULT_GEMINI_MODEL,
        "--deepseek-model", experiment.DEFAULT_DEEPSEEK_MODEL,
        "--claude-model", experiment.DEFAULT_CLAUDE_MODEL,
        "--trials", str(TRIALS),
        "--format-retries", str(FORMAT_RETRIES),
        "--no-inline-confidence-entities", str(NO_INLINE_CONFIDENCE_ENTITIES),
        "--layout-index", str(LAYOUT_INDEX),
        "--temperature", str(TEMPERATURE),
        "--strategy", STRATEGY,
        "--output", str(raw_path),
        "--condition-output", str(condition_path),
    ]
    if mock:
        command.append("--mock")
    return command


def attempt_paths(work_dir: Path, chunk: RecoveryChunk, attempt: int):
    prefix = work_dir / f"{chunk.slug}.attempt{attempt:03d}"
    return (
        Path(str(prefix) + ".raw.csv"),
        Path(str(prefix) + ".conditions.csv"),
        Path(str(prefix) + ".meta.json"),
    )


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp-{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_attempt_meta(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def validate_chunk_attempt(
    raw_path: Path,
    condition_path: Path,
    meta_path: Path,
    chunk: RecoveryChunk,
    design: StudyDesign,
    *,
    mock: bool,
    initial_source_hashes: dict[str, str],
) -> AuditReport | None:
    meta = _read_attempt_meta(meta_path)
    if not meta or meta.get("return_code") != 0:
        return None
    if meta.get("design") != fixed_design_manifest(mock):
        return None
    if meta.get("source_hashes") != initial_source_hashes:
        return None
    try:
        report = audit_files(
            raw_path,
            condition_path,
            design,
            mock_model_specs() if mock else real_model_specs(),
            set(chunk.keys),
        )
    except SalvageError:
        return None
    if report.unexpected_keys or report.pending_keys:
        return None
    return report


def manual_review_messages(
    raw_path: Path,
    condition_path: Path,
    chunk: RecoveryChunk,
    design: StudyDesign,
    *,
    mock: bool,
) -> list[str]:
    """Return semantic issues that must never trigger stochastic resampling."""
    try:
        report = audit_files(
            raw_path,
            condition_path,
            design,
            mock_model_specs() if mock else real_model_specs(),
            set(chunk.keys),
        )
    except SalvageError:
        return []
    return [
        f"{key.label()}: {issue}"
        for key, issues in report.issues_by_key.items()
        for issue in issues
        if "requires manual review" in issue
    ]


def recover_chunks(
    chunks: list[RecoveryChunk],
    design: StudyDesign,
    base_raw: Path,
    base_conditions: Path,
    initial_hashes: dict[str, str],
    work_dir: Path,
    python_executable: Path | str,
    *,
    mock: bool,
    max_new_attempts: int,
) -> list[tuple[RecoveryChunk, Path, Path, Path, AuditReport]]:
    mode_dir = work_dir / ("mock" if mock else "live")
    mode_dir.mkdir(parents=True, exist_ok=True)
    recovered = []
    for chunk_index, chunk in enumerate(chunks, start=1):
        assert_source_hashes(base_raw, base_conditions, initial_hashes)
        print(
            f"\n[chunk {chunk_index}/{len(chunks)}] {chunk.entity_id}; "
            f"ratios={','.join(chunk.ratios)}; models={','.join(chunk.model_slots)}"
        )
        accepted = None
        # Discover numbers from the union of raw/condition/meta names.  A hard
        # kill can leave CSVs before the meta file is atomically written; such
        # an orphan is never reusable, but it must still reserve its attempt
        # number so the next invocation advances to a fresh path.
        existing_attempts = set()
        for path in mode_dir.glob(f"{chunk.slug}.attempt*"):
            try:
                existing_attempts.add(
                    int(path.name.split(".attempt", 1)[1].split(".", 1)[0])
                )
            except (IndexError, ValueError):
                continue
        for attempt in sorted(existing_attempts):
            raw_path, condition_path, meta_path = attempt_paths(
                mode_dir, chunk, attempt
            )
            report = validate_chunk_attempt(
                raw_path, condition_path, meta_path, chunk, design,
                mock=mock, initial_source_hashes=initial_hashes,
            )
            if report is not None:
                print(f"  reusing completely validated attempt {attempt:03d}")
                accepted = (chunk, raw_path, condition_path, meta_path, report)
                break
            meta = _read_attempt_meta(meta_path)
            if meta and meta.get("return_code") == 0:
                manual_issues = manual_review_messages(
                    raw_path, condition_path, chunk, design, mock=mock
                )
                if manual_issues:
                    raise SalvageError(
                        "a preserved completed chunk contains an OTHER response; "
                        "refusing outcome-dependent resampling. Review the "
                        f"attempt first: {'; '.join(manual_issues)}"
                    )
        if accepted is not None:
            recovered.append(accepted)
            continue

        attempt = max(existing_attempts, default=0)
        for _ in range(max_new_attempts):
            attempt += 1
            raw_path, condition_path, meta_path = attempt_paths(
                mode_dir, chunk, attempt
            )
            if any(path.exists() for path in (raw_path, condition_path, meta_path)):
                raise SalvageError(
                    f"fresh attempt path unexpectedly exists for {chunk.slug} "
                    f"attempt {attempt:03d}"
                )
            command = chunk_command(
                chunk, raw_path, condition_path, python_executable, mock=mock
            )
            print(f"  starting fresh attempt {attempt:03d}")
            started = datetime.now(timezone.utc).isoformat()
            return_code = None
            exception_text = ""
            caught = None
            try:
                completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
                return_code = completed.returncode
            except BaseException as exc:
                caught = exc
                exception_text = f"{type(exc).__name__}: {exc}"
            meta = {
                "chunk": chunk.slug,
                "keys": [key.label() for key in sorted(chunk.keys)],
                "command": command,
                "design": fixed_design_manifest(mock),
                "source_hashes": initial_hashes,
                "started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "return_code": return_code,
                "exception": exception_text,
            }
            _atomic_json(meta_path, meta)
            assert_source_hashes(base_raw, base_conditions, initial_hashes)
            if isinstance(caught, (KeyboardInterrupt, SystemExit)):
                raise caught
            report = validate_chunk_attempt(
                raw_path, condition_path, meta_path, chunk, design,
                mock=mock, initial_source_hashes=initial_hashes,
            )
            if report is not None:
                print(f"  attempt {attempt:03d} passed exhaustive chunk validation")
                accepted = (chunk, raw_path, condition_path, meta_path, report)
                break
            if return_code == 0:
                manual_issues = manual_review_messages(
                    raw_path, condition_path, chunk, design, mock=mock
                )
                if manual_issues:
                    raise SalvageError(
                        "a completed chunk contains an OTHER response; refusing "
                        "outcome-dependent resampling. Review the preserved "
                        f"attempt first: {'; '.join(manual_issues)}"
                    )
            print(
                f"  attempt {attempt:03d} was not reusable "
                f"(exit={return_code!r}); its files remain untouched"
            )
        if accepted is None:
            raise SalvageError(
                f"chunk {chunk.slug} did not produce a valid complete attempt "
                f"after {max_new_attempts} new attempt(s); rerun the salvage "
                "command to continue with fresh attempt files"
            )
        recovered.append(accepted)
    return recovered


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _display_path(path: Path) -> str:
    """Use a repo-relative manifest path when possible, absolute otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sort_key(design: StudyDesign, key: ConditionKey):
    entity_order = {entity["entity_id"]: i for i, entity in enumerate(design.entities)}
    return (
        entity_order[key.entity_id],
        design.ratios.index(key.ratio),
        MODEL_ORDER.index(key.model_slot),
    )


def validate_full_factorial(report: AuditReport, design: StudyDesign) -> None:
    if report.unexpected_keys or report.pending_keys:
        raise SalvageError(
            f"merged evidence is incomplete: valid={len(report.valid_keys)}, "
            f"pending={len(report.pending_keys)}, unexpected={len(report.unexpected_keys)}"
        )
    conditions = report.valid_condition_rows
    if len(conditions) != 1350:
        raise SalvageError(f"merged condition count is {len(conditions)}, expected 1,350")
    if Counter(row["model_slot"] for row in conditions) != Counter(
        {slot: 450 for slot in MODEL_ORDER}
    ):
        raise SalvageError("merged model factorial is not exactly 450 per model")
    if Counter(row["ratio"] for row in conditions) != Counter(
        {ratio: 225 for ratio in design.ratios}
    ):
        raise SalvageError("merged ratio factorial is not exactly 225 per ratio")
    if Counter(row["entity_id"] for row in conditions) != Counter(
        {entity["entity_id"]: 18 for entity in design.entities}
    ):
        raise SalvageError("merged entity factorial is not exactly 18 per entity")
    assignment_counts = Counter(row["distribution_request_assigned"] for row in conditions)
    if assignment_counts != Counter({"1": 684, "0": 666}):
        raise SalvageError(f"38/37 assignment is wrong: {dict(assignment_counts)}")
    inline_counts = Counter(row["inline_confidence_requested"] for row in conditions)
    if inline_counts != Counter({"1": 570, "0": 780}):
        raise SalvageError(f"inline-confidence counts are wrong: {dict(inline_counts)}")
    primary_terminal = [
        row for row in report.valid_raw_rows
        if row["call_phase"] == "primary" and row["trial_record"] == "1"
    ]
    if len(primary_terminal) != 4050:
        raise SalvageError(
            f"merged terminal primary count is {len(primary_terminal)}, expected 4,050"
        )
    if any(row["attempt_status"] != "accepted" for row in primary_terminal):
        raise SalvageError("merged terminal primary rows are not all accepted")


def publish_merge(
    base_report: AuditReport,
    recovered,
    design: StudyDesign,
    output_raw: Path,
    output_conditions: Path,
    manifest_path: Path,
    source_paths: dict[str, Path],
    initial_hashes: dict[str, str],
) -> dict:
    destinations = (output_raw, output_conditions, manifest_path)
    resolved_sources = {path.resolve() for path in source_paths.values()}
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise SalvageError("output, condition-output, and manifest must be different files")
    if any(path.resolve() in resolved_sources for path in destinations):
        raise SalvageError("salvage outputs must not be either source CSV")
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise SalvageError(
            "refusing to overwrite existing final salvage output(s): "
            + ", ".join(existing)
        )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)

    raw_by_key = defaultdict(list)
    condition_by_key = {}
    for row in base_report.valid_raw_rows:
        raw_by_key[row_key(row)].append(row)
    for row in base_report.valid_condition_rows:
        condition_by_key[row_key(row)] = row
    chunk_manifest = []
    for chunk, raw_path, condition_path, meta_path, report in recovered:
        for row in report.valid_raw_rows:
            raw_by_key[row_key(row)].append(row)
        for row in report.valid_condition_rows:
            key = row_key(row)
            if key in condition_by_key:
                raise SalvageError(f"duplicate merge key: {key.label()}")
            condition_by_key[key] = row
        chunk_manifest.append({
            "chunk": chunk.slug,
            "keys": len(chunk.keys),
            "raw_path": _display_path(raw_path),
            "conditions_path": _display_path(condition_path),
            "meta_path": _display_path(meta_path),
            "raw_sha256": sha256_file(raw_path),
            "conditions_sha256": sha256_file(condition_path),
        })
    if set(condition_by_key) != design.expected_keys or set(raw_by_key) != design.expected_keys:
        raise SalvageError("merge inputs do not cover each expected key exactly once")

    ordered_keys = sorted(design.expected_keys, key=lambda key: _sort_key(design, key))
    merged_conditions = [condition_by_key[key] for key in ordered_keys]
    merged_raw = []
    for key in ordered_keys:
        merged_raw.extend(raw_by_key[key])

    token = uuid.uuid4().hex
    temp_raw = output_raw.with_name(output_raw.name + f".tmp-{token}")
    temp_conditions = output_conditions.with_name(
        output_conditions.name + f".tmp-{token}"
    )
    temp_manifest = manifest_path.with_name(manifest_path.name + f".tmp-{token}")
    temporary_paths = (temp_raw, temp_conditions, temp_manifest)
    published_paths = []
    try:
        _write_csv(temp_raw, experiment.CSV_FIELDS, merged_raw)
        _write_csv(temp_conditions, experiment.CONDITION_FIELDS, merged_conditions)
        final_report = audit_files(
            temp_raw, temp_conditions, design, real_model_specs(), design.expected_keys
        )
        validate_full_factorial(final_report, design)
        manifest = {
            "status": "complete_and_validated",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": experiment.PROTOCOL_VERSION,
            "dataset_sha256": design.dataset_sha256,
            "design": fixed_design_manifest(False),
            "source_raw": _display_path(source_paths["raw"]),
            "source_conditions": _display_path(source_paths["conditions"]),
            "source_hashes": initial_hashes,
            "retained_conditions": len(base_report.valid_keys),
            "replaced_or_recovered_conditions": len(base_report.pending_keys),
            "total_conditions": len(final_report.valid_keys),
            "total_terminal_primary_samples": 4050,
            "run_ids_preserved": sorted(
                {row["run_id"] for row in merged_conditions}
            ),
            "collection_segments": [
                {
                    "run_id": run_id,
                    "condition_count": sum(
                        row["run_id"] == run_id for row in merged_conditions
                    ),
                    "raw_row_count": sum(
                        row["run_id"] == run_id for row in merged_raw
                    ),
                    "terminal_primary_count": sum(
                        row["run_id"] == run_id
                        and row["call_phase"] == "primary"
                        and row["trial_record"] == "1"
                        for row in merged_raw
                    ),
                    "entity_ids": sorted({
                        row["entity_id"] for row in merged_conditions
                        if row["run_id"] == run_id
                    }),
                    "ratios": sorted({
                        row["ratio"] for row in merged_conditions
                        if row["run_id"] == run_id
                    }, key=design.ratios.index),
                    "model_slots": [
                        slot for slot in MODEL_ORDER
                        if any(
                            row["run_id"] == run_id and row["model_slot"] == slot
                            for row in merged_conditions
                        )
                    ],
                }
                for run_id in sorted({row["run_id"] for row in merged_conditions})
            ],
            "derived_rescore_corrections": base_report.derived_corrections,
            "chunks": chunk_manifest,
            "output_raw": _display_path(output_raw),
            "output_conditions": _display_path(output_conditions),
            "output_raw_sha256": sha256_file(temp_raw),
            "output_conditions_sha256": sha256_file(temp_conditions),
        }
        with temp_manifest.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        assert_source_hashes(
            source_paths["raw"], source_paths["conditions"], initial_hashes
        )
        # Each publication is atomic; the manifest is the final completion marker.
        os.replace(temp_raw, output_raw)
        published_paths.append(output_raw)
        os.replace(temp_conditions, output_conditions)
        published_paths.append(output_conditions)
        os.replace(temp_manifest, manifest_path)
        published_paths.append(manifest_path)
        return manifest
    except BaseException:
        # All destinations were proven absent above, so these can only be files
        # published by this invocation.  Roll back a partially completed
        # three-file publication so the next resume is not blocked by an
        # orphaned output without its completion manifest.
        for path in reversed(published_paths):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for path in temporary_paths:
            if path.exists():
                path.unlink()


def print_audit(report: AuditReport, chunks: list[RecoveryChunk]) -> None:
    print("\n=== immutable base audit ===")
    print(f"valid, raw-corroborated conditions: {len(report.valid_keys):,}")
    print(f"whole conditions requiring replacement: {len(report.pending_keys):,}")
    print(f"unexpected keys: {len(report.unexpected_keys):,}")
    print(f"recovery chunks: {len(chunks):,}")
    for key in sorted(report.pending_keys)[:12]:
        reasons = "; ".join(report.issues_by_key.get(key, ["missing"]))
        print(f"  {key.label()}: {reasons}")
    if len(report.pending_keys) > 12:
        print(f"  ... {len(report.pending_keys) - 12} more pending keys")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-raw", type=Path, default=DEFAULT_BASE_RAW)
    parser.add_argument(
        "--base-conditions", type=Path, default=DEFAULT_BASE_CONDITIONS
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_RAW)
    parser.add_argument(
        "--condition-output", type=Path, default=DEFAULT_OUTPUT_CONDITIONS
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--max-new-attempts-per-chunk", type=int, default=3,
        help="fresh files/tries per incomplete chunk in this invocation (default: 3)",
    )
    parser.add_argument(
        "--audit-only", action="store_true",
        help="audit and print the exact recovery plan; make no API calls or files",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="exercise pending chunks offline; validate them but never merge/publish",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_new_attempts_per_chunk < 1:
        raise SalvageError("--max-new-attempts-per-chunk must be at least 1")
    design = load_design()
    base_raw = args.base_raw.resolve()
    base_conditions = args.base_conditions.resolve()
    initial_hashes = source_hashes(base_raw, base_conditions)
    base_report = audit_files(
        base_raw, base_conditions, design, real_model_specs(), design.expected_keys
    )
    if base_report.unexpected_keys:
        raise SalvageError("base CSVs contain keys outside the 75 x 6 x 3 design")
    chunks = plan_chunks(base_report.pending_keys, design)
    print(f"Source raw SHA-256:        {initial_hashes['raw_sha256']}")
    print(f"Source conditions SHA-256: {initial_hashes['conditions_sha256']}")
    print_audit(base_report, chunks)
    if args.audit_only:
        assert_source_hashes(base_raw, base_conditions, initial_hashes)
        return 0

    if not args.mock:
        destinations = (args.output.resolve(), args.condition_output.resolve(), args.manifest.resolve())
        if len(set(destinations)) != 3:
            raise SalvageError("final output paths must be three different files")
        if base_raw in destinations or base_conditions in destinations:
            raise SalvageError("final output paths must not overwrite source CSVs")
        existing = [str(path) for path in destinations if path.exists()]
        if existing:
            raise SalvageError(
                "refusing to overwrite existing final salvage output(s): "
                + ", ".join(existing)
            )

    recovered = recover_chunks(
        chunks,
        design,
        base_raw,
        base_conditions,
        initial_hashes,
        args.work_dir.resolve(),
        args.python,
        mock=args.mock,
        max_new_attempts=args.max_new_attempts_per_chunk,
    )
    assert_source_hashes(base_raw, base_conditions, initial_hashes)
    if args.mock:
        print(
            f"\nMOCK PASS: {len(recovered)} recovery chunks validated offline. "
            "No merged evidence was published."
        )
        return 0

    manifest = publish_merge(
        base_report,
        recovered,
        design,
        args.output.resolve(),
        args.condition_output.resolve(),
        args.manifest.resolve(),
        {"raw": base_raw, "conditions": base_conditions},
        initial_hashes,
    )
    assert_source_hashes(base_raw, base_conditions, initial_hashes)
    print("\nSALVAGE PASS: full Standard factorial validated and published.")
    print(f"  raw:        {args.output.resolve()}")
    print(f"  conditions: {args.condition_output.resolve()}")
    print(f"  manifest:   {args.manifest.resolve()}")
    print(f"  preserved run IDs: {', '.join(manifest['run_ids_preserved'])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SalvageError as exc:
        print(f"SALVAGE REFUSED: {exc}", file=sys.stderr)
        raise SystemExit(1)
