#!/usr/bin/env python3
"""Run the prespecified condition-level Majority Illusion analyses.

This module intentionally consumes *condition* CSVs, not raw per-call logs. The
three repeated primary calls are measurements used to form one modal condition
outcome; they are never treated as three independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import warnings as python_warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "entities.json"
EXPECTED_PROTOCOL_VERSION = "2026-07-17-rich-distribution-v3-balanced"
EXPECTED_MODEL_SLOTS = {"gemini", "deepseek", "claude"}

VALID_CATEGORIES = {"MAJ", "MIN", "COM", "FLAG", "OTHER"}
NONANALYTIC_CATEGORIES = {"TIE", "AMBIGUOUS", "UNSCORED"}
SINGLE_MINORITY_RATIOS = {"2:1", "3:1", "4:1"}
NUMERIC_GENERAL_ATTRIBUTES = {"employee_count", "founding_year"}
CONFIDENCE_COLUMNS = (
    "confidence_best_resolution",
    "posthoc_subjective_confidence",
    "posthoc_probability",
)
ASSIGNMENT_COLUMN = "distribution_request_assigned"
ACTUAL_REQUEST_COLUMN = "inline_confidence_requested"
DISTRIBUTION_METRICS = {
    "p_majority": ("mean_p_majority", "p_majority"),
    "p_minority": ("mean_p_minority", "p_minority"),
    "p_indeterminate": ("mean_p_indeterminate", "p_indeterminate"),
    "p_sources_conflict": ("mean_p_sources_conflict", "p_sources_conflict"),
}

BASE_REQUIRED_COLUMNS = {
    "entity_id",
    "domain",
    "attribute",
    "question",
    "majority_value",
    "minority_value",
    "ratio",
    "n_docs",
    "strategy",
    "doc_positions",
    "model_id",
    "model_slot",
    "model_provider",
    "returned_model_ids",
    "layout_index",
    ASSIGNMENT_COLUMN,
    ACTUAL_REQUEST_COLUMN,
    "modal_category",
    "n_samples",
    "n_scored",
    "n_primary_errors",
    "n_primary_format_errors",
    "n_valid_distributions",
    "self_consistency",
    "self_consistency_all_samples",
    "modal_tie",
    "inline_distributions",
    "posthoc_status",
    "posthoc_skipped",
}
PROVENANCE_COLUMNS = {"protocol_version", "dataset_sha256"}


class AnalysisInputError(ValueError):
    """Raised when condition data cannot safely be pooled or analyzed."""


@dataclass
class AnalysisState:
    warnings: list[str] = field(default_factory=list)
    model_summaries: list[str] = field(default_factory=list)
    coefficients: list[dict] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        print(f"WARNING: {message}", file=sys.stderr)


def _one_of(columns: Iterable[str], candidates: Sequence[str]) -> str | None:
    available = set(columns)
    return next((name for name in candidates if name in available), None)


def _parse_binary(value) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, bool):
        return float(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = math.nan
    if numeric in {0.0, 1.0}:
        return numeric
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "requested", "distribution"}:
        return 1.0
    if text in {"0", "false", "no", "n", "answer_only", "none"}:
        return 0.0
    return math.nan


def _validation_issue(
    state: AnalysisState,
    message: str,
    override: bool,
    override_option: str,
) -> None:
    if not override:
        raise AnalysisInputError(
            f"{message} Use {override_option} only for an explicitly labeled "
            "pilot/audit; overridden output is not final evidence."
        )
    state.warn(f"OVERRIDE ACTIVE: {message}")


def current_dataset_metadata() -> dict:
    raw = DATASET_PATH.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    entities = payload.get("entities", [])
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "entity_ids": [str(entity["entity_id"]) for entity in entities],
        "entity_lookup": {
            str(entity["entity_id"]): {
                "domain": str(entity["domain"]).lower(),
                "attribute": str(entity["attribute"]),
                "question": str(entity["question"]),
                "majority_value": str(entity["majority_value"]),
                "minority_value": str(entity["minority_value"]),
            }
            for entity in entities
        },
        "ratios": [str(ratio) for ratio in payload.get("ratios", [])],
    }


def _count_inline_distributions(value) -> int:
    if pd.isna(value) or str(value).strip() == "":
        return 0
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return -1
    if not isinstance(parsed, list):
        return -1
    return sum(isinstance(item, dict) for item in parsed)


def _parse_json_list(value) -> list | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _model_family_matches(slot: str, model_id: str) -> bool:
    model = str(model_id).strip().lower()
    if not model:
        return False
    if slot == "gemini":
        return "gemini" in model
    if slot == "deepseek":
        return "deepseek" in model
    if slot == "claude":
        return "claude" in model or "anthropic" in model
    return False


def _validate_complete_factorial(
    data: pd.DataFrame,
    metadata: dict,
    state: AnalysisState,
    allow_incomplete: bool,
    audit_override: bool,
) -> None:
    override = allow_incomplete or audit_override
    option = "--allow-incomplete (or --audit-override for non-v3 data)"
    expected_entities = set(metadata["entity_ids"])
    expected_ratios = set(metadata["ratios"])
    observed_entities = set(data["entity_id"])
    observed_ratios = set(data["ratio"])
    observed_slots = set(data["model_slot"])

    if observed_entities != expected_entities:
        _validation_issue(
            state,
            "Entity set is incomplete or unexpected: "
            f"expected {len(expected_entities)}, observed {len(observed_entities)}, "
            f"missing {sorted(expected_entities - observed_entities)[:5]}, "
            f"extra {sorted(observed_entities - expected_entities)[:5]}.",
            override,
            option,
        )
    if observed_ratios != expected_ratios:
        _validation_issue(
            state,
            f"Ratio set mismatch: expected {sorted(expected_ratios)}, "
            f"observed {sorted(observed_ratios)}.",
            override,
            option,
        )
    if observed_slots != EXPECTED_MODEL_SLOTS:
        _validation_issue(
            state,
            f"Three-model slot mismatch: expected {sorted(EXPECTED_MODEL_SLOTS)}, "
            f"observed {sorted(observed_slots)}.",
            override,
            option,
        )

    model_ids_per_slot = data.groupby("model_slot", observed=True)["model_id"].nunique()
    if model_ids_per_slot.gt(1).any():
        _validation_issue(
            state,
            "A model slot maps to multiple requested model IDs: "
            f"{model_ids_per_slot[model_ids_per_slot.gt(1)].to_dict()}.",
            audit_override,
            "--audit-override",
        )

    expected_tuples = set(
        itertools.product(expected_entities, expected_ratios, EXPECTED_MODEL_SLOTS)
    )
    for (strategy, layout), block in data.groupby(
        ["strategy", "layout_index"], dropna=False, observed=True
    ):
        observed_tuples = set(
            block[["entity_id", "ratio", "model_slot"]]
            .itertuples(index=False, name=None)
        )
        missing = expected_tuples - observed_tuples
        extra = observed_tuples - expected_tuples
        if missing or extra or len(block) != len(expected_tuples):
            _validation_issue(
                state,
                f"Incomplete factorial for strategy={strategy}, layout={layout}: "
                f"expected {len(expected_tuples)} rows, observed {len(block)}, "
                f"missing {len(missing)}, extra {len(extra)}.",
                override,
                option,
            )


def load_condition_data(
    paths: Sequence[Path],
    state: AnalysisState,
    allow_incomplete: bool = False,
    audit_override: bool = False,
    expected_control_entities: int = 37,
) -> tuple[pd.DataFrame, dict]:
    """Load and strictly validate final balanced-v3 condition-level CSVs."""
    metadata = current_dataset_metadata()
    expected_entity_count = len(metadata["entity_ids"])
    if not 0 < expected_control_entities < expected_entity_count:
        raise AnalysisInputError(
            f"--expected-control-entities must be between 1 and "
            f"{expected_entity_count - 1}."
        )
    frames: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise AnalysisInputError(f"Input file does not exist: {path}")
        frame = pd.read_csv(path)
        if "call_phase" in frame.columns:
            raise AnalysisInputError(
                f"{path} looks like a raw per-call log (it has call_phase); "
                "provide conditions_*.csv instead."
            )
        frame["source_file"] = str(path.resolve())
        frames.append(frame)

    if not frames:
        raise AnalysisInputError("At least one condition CSV is required.")
    data = pd.concat(frames, ignore_index=True, sort=False)
    if data.empty:
        raise AnalysisInputError("The supplied condition CSVs contain no rows.")

    missing = BASE_REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise AnalysisInputError(
            "Condition CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    missing_provenance = PROVENANCE_COLUMNS - set(data.columns)
    if missing_provenance:
        _validation_issue(
            state,
            "Legacy data lacks provenance columns: "
            + ", ".join(sorted(missing_provenance)),
            audit_override,
            "--audit-override",
        )
        for column in missing_provenance:
            data[column] = "legacy-unknown"

    for column in PROVENANCE_COLUMNS:
        blank = data[column].isna() | data[column].astype(str).str.strip().eq("")
        if blank.any():
            _validation_issue(
                state,
                f"{int(blank.sum())} rows have blank {column} values.",
                audit_override,
                "--audit-override",
            )
            data.loc[blank, column] = "legacy-unknown"

    protocols = sorted(data["protocol_version"].astype(str).unique().tolist())
    datasets = sorted(data["dataset_sha256"].astype(str).unique().tolist())
    if protocols != [EXPECTED_PROTOCOL_VERSION]:
        _validation_issue(
            state,
            f"Expected protocol_version={EXPECTED_PROTOCOL_VERSION}, observed {protocols}.",
            audit_override,
            "--audit-override",
        )
    if datasets != [metadata["sha256"]]:
        _validation_issue(
            state,
            f"Expected current data/entities.json SHA-256 {metadata['sha256']}, "
            f"observed {datasets}.",
            audit_override,
            "--audit-override",
        )

    data["n_samples"] = pd.to_numeric(data["n_samples"], errors="coerce")
    wrong_samples = data["n_samples"].ne(3)
    if wrong_samples.any():
        values = sorted(data.loc[wrong_samples, "n_samples"].dropna().unique().tolist())
        _validation_issue(
            state,
            f"Expected n_samples=3 for every condition, but "
            f"{int(wrong_samples.sum())} rows differ (observed: {values}).",
            audit_override,
            "--audit-override",
        )

    data["modal_category"] = (
        data["modal_category"].fillna("").astype(str).str.strip().str.upper()
    )
    blank_categories = data["modal_category"].eq("")
    if blank_categories.any():
        _validation_issue(
            state,
            f"{int(blank_categories.sum())} rows have blank modal_category values.",
            audit_override,
            "--audit-override",
        )
        data.loc[blank_categories, "modal_category"] = "UNSCORED"
    unknown_categories = sorted(
        set(data["modal_category"]) - VALID_CATEGORIES - NONANALYTIC_CATEGORIES
    )
    if unknown_categories:
        _validation_issue(
            state,
            f"Unknown modal categories found: {unknown_categories}",
            audit_override,
            "--audit-override",
        )
        data.loc[data["modal_category"].isin(unknown_categories), "modal_category"] = (
            "UNSCORED"
        )

    modal_tie = data["modal_tie"].map(_parse_binary)
    invalid_modal_tie = modal_tie.isna()
    if invalid_modal_tie.any():
        _validation_issue(
            state,
            f"{int(invalid_modal_tie.sum())} rows have an invalid modal_tie value.",
            audit_override,
            "--audit-override",
        )
        modal_tie = modal_tie.fillna(0)
    modal_tie = modal_tie.astype(int)
    arbitrary_ties = modal_tie.eq(1) & ~data["modal_category"].isin(
        {"TIE", "AMBIGUOUS"}
    )
    unlabeled_ties = modal_tie.eq(0) & data["modal_category"].isin(
        {"TIE", "AMBIGUOUS"}
    )
    if arbitrary_ties.any() or unlabeled_ties.any():
        _validation_issue(
            state,
            f"Modal tie fields disagree in {int((arbitrary_ties | unlabeled_ties).sum())} rows.",
            audit_override,
            "--audit-override",
        )
        data.loc[arbitrary_ties, "modal_category"] = "AMBIGUOUS"
        modal_tie.loc[unlabeled_ties] = 1
    data["modal_tie"] = modal_tie

    data["n_scored"] = pd.to_numeric(data["n_scored"], errors="coerce")
    invalid_n_scored = (
        data["n_scored"].isna()
        | data["n_scored"].lt(0)
        | data["n_scored"].gt(data["n_samples"])
    )
    if invalid_n_scored.any():
        _validation_issue(
            state,
            f"{int(invalid_n_scored.sum())} rows have invalid n_scored values.",
            audit_override,
            "--audit-override",
        )
    failure_label_mismatch = (
        data["n_scored"].eq(0) & data["modal_category"].ne("UNSCORED")
    ) | (
        data["n_scored"].gt(0) & data["modal_category"].eq("UNSCORED")
    )
    if failure_label_mismatch.any():
        _validation_issue(
            state,
            f"{int(failure_label_mismatch.sum())} rows disagree between n_scored "
            "and the UNSCORED modal label.",
            audit_override,
            "--audit-override",
        )
        data.loc[data["n_scored"].eq(0), "modal_category"] = "UNSCORED"

    for count_column in (
        "n_primary_errors",
        "n_primary_format_errors",
        "n_valid_distributions",
    ):
        data[count_column] = pd.to_numeric(data[count_column], errors="coerce")
        invalid_count = (
            data[count_column].isna()
            | data[count_column].lt(0)
            | data[count_column].gt(data["n_samples"])
        )
        if invalid_count.any():
            _validation_issue(
                state,
                f"{int(invalid_count.sum())} rows have invalid {count_column} values.",
                audit_override,
                "--audit-override",
            )

    data["posthoc_status"] = (
        data["posthoc_status"].fillna("").astype(str).str.strip().str.lower()
    )
    valid_posthoc_statuses = {
        "completed",
        "error",
        "format_error",
        "skipped_modal_tie",
        "skipped_all_unscored",
        "skipped_no_modal_answer",
    }
    invalid_posthoc_status = ~data["posthoc_status"].isin(valid_posthoc_statuses)
    if invalid_posthoc_status.any():
        _validation_issue(
            state,
            f"{int(invalid_posthoc_status.sum())} rows have invalid posthoc_status values.",
            audit_override,
            "--audit-override",
        )
    data["posthoc_skipped"] = data["posthoc_skipped"].map(_parse_binary)
    invalid_posthoc_skipped = data["posthoc_skipped"].isna()
    if invalid_posthoc_skipped.any():
        _validation_issue(
            state,
            f"{int(invalid_posthoc_skipped.sum())} rows have invalid posthoc_skipped values.",
            audit_override,
            "--audit-override",
        )
    status_is_skip = data["posthoc_status"].str.startswith("skipped_")
    skip_mismatch = status_is_skip.ne(data["posthoc_skipped"].eq(1))
    if skip_mismatch.any():
        _validation_issue(
            state,
            f"{int(skip_mismatch.sum())} rows disagree between posthoc_status and posthoc_skipped.",
            audit_override,
            "--audit-override",
        )

    data["ratio"] = data["ratio"].astype(str).str.strip()
    invalid_ratio = ~data["ratio"].str.fullmatch(r"\d+:\d+")
    if invalid_ratio.any():
        _validation_issue(
            state,
            f"{int(invalid_ratio.sum())} rows have malformed ratio labels.",
            audit_override,
            "--audit-override",
        )

    data["strategy"] = data["strategy"].astype(str).str.strip().str.lower()
    data["domain"] = data["domain"].astype(str).str.strip().str.lower()
    data["entity_id"] = data["entity_id"].astype(str).str.strip()
    data["model_id"] = data["model_id"].astype(str).str.strip()
    data["model_slot"] = data["model_slot"].astype(str).str.strip().str.lower()
    data["model_provider"] = (
        data["model_provider"].astype(str).str.strip().str.lower()
    )
    data["layout_index"] = pd.to_numeric(data["layout_index"], errors="coerce")
    invalid_layout = data["layout_index"].isna()
    if invalid_layout.any():
        _validation_issue(
            state,
            f"{int(invalid_layout.sum())} rows have invalid layout_index values.",
            audit_override,
            "--audit-override",
        )

    expected_providers = {
        "gemini": "gemini",
        "deepseek": "openrouter",
        "claude": "openrouter",
    }
    invalid_requested_family = ~data.apply(
        lambda row: _model_family_matches(row["model_slot"], row["model_id"]),
        axis=1,
    )
    invalid_provider = data.apply(
        lambda row: expected_providers.get(row["model_slot"]) != row["model_provider"],
        axis=1,
    )
    parsed_returned_ids = data["returned_model_ids"].map(_parse_json_list)
    invalid_returned_ids = parsed_returned_ids.isna() | pd.Series(
        [
            not values
            or any(
                not _model_family_matches(slot, returned)
                for returned in values
            )
            for values, slot in zip(
                parsed_returned_ids.fillna("").tolist(),
                data["model_slot"].tolist(),
            )
        ],
        index=data.index,
    )
    if invalid_requested_family.any() or invalid_provider.any() or invalid_returned_ids.any():
        _validation_issue(
            state,
            "Model provenance mismatch: "
            f"requested_family={int(invalid_requested_family.sum())}, "
            f"provider={int(invalid_provider.sum())}, "
            f"returned_family_or_missing={int(invalid_returned_ids.sum())}.",
            audit_override,
            "--audit-override",
        )

    metadata_mismatch = pd.Series(False, index=data.index)
    position_mismatch = pd.Series(False, index=data.index)
    n_docs_numeric = pd.to_numeric(data["n_docs"], errors="coerce")
    for index, row in data.iterrows():
        expected = metadata["entity_lookup"].get(row["entity_id"])
        if (
            expected is None
            or row["domain"] != expected["domain"]
            or str(row["attribute"]) != expected["attribute"]
            or str(row["question"]) != expected["question"]
            or str(row["majority_value"]) != expected["majority_value"]
            or str(row["minority_value"]) != expected["minority_value"]
        ):
            metadata_mismatch.loc[index] = True
        try:
            expected_majority, expected_minority = map(int, row["ratio"].split(":"))
        except (AttributeError, ValueError):
            position_mismatch.loc[index] = True
            continue
        expected_docs = expected_majority + expected_minority
        positions = _parse_json_list(row["doc_positions"])
        normalized_positions = (
            [str(value).strip().upper() for value in positions]
            if positions is not None else []
        )
        if (
            pd.isna(n_docs_numeric.loc[index])
            or int(n_docs_numeric.loc[index]) != expected_docs
            or len(normalized_positions) != expected_docs
            or normalized_positions.count("MAJ") != expected_majority
            or normalized_positions.count("MIN") != expected_minority
            or any(value not in {"MAJ", "MIN"} for value in normalized_positions)
        ):
            position_mismatch.loc[index] = True
    data["n_docs"] = n_docs_numeric
    if metadata_mismatch.any() or position_mismatch.any():
        _validation_issue(
            state,
            "Dataset metadata mismatch: "
            f"entity_fields={int(metadata_mismatch.sum())}, "
            f"n_docs_or_doc_positions={int(position_mismatch.sum())}.",
            audit_override,
            "--audit-override",
        )

    data[ASSIGNMENT_COLUMN] = data[ASSIGNMENT_COLUMN].map(_parse_binary)
    data[ACTUAL_REQUEST_COLUMN] = data[ACTUAL_REQUEST_COLUMN].map(_parse_binary)
    invalid_assignment = data[ASSIGNMENT_COLUMN].isna()
    invalid_request = data[ACTUAL_REQUEST_COLUMN].isna()
    if invalid_assignment.any() or invalid_request.any():
        _validation_issue(
            state,
            f"Invalid binary treatment values: assigned={int(invalid_assignment.sum())}, "
            f"actual={int(invalid_request.sum())}.",
            audit_override,
            "--audit-override",
        )

    confidence_cols = [
        column for column in CONFIDENCE_COLUMNS if column in data.columns
    ]
    if not confidence_cols:
        state.warn(
            "No supported post-hoc subjective-confidence column was found; "
            "RQ1 confidence summaries and models will be empty."
        )
        data["subjective_posthoc_confidence"] = math.nan
    else:
        combined_confidence = pd.Series(math.nan, index=data.index, dtype=float)
        for column in confidence_cols:
            combined_confidence = combined_confidence.combine_first(
                pd.to_numeric(data[column], errors="coerce")
            )
        data["subjective_posthoc_confidence"] = combined_confidence
        out_of_range = data["subjective_posthoc_confidence"].notna() & ~data[
            "subjective_posthoc_confidence"
        ].between(0, 100)
        if out_of_range.any():
            _validation_issue(
                state,
                f"{int(out_of_range.sum())} post-hoc confidence values fall outside "
                "0-100.",
                audit_override,
                "--audit-override",
            )
            data.loc[out_of_range, "subjective_posthoc_confidence"] = math.nan

    data["self_consistency_scored_only"] = pd.to_numeric(
        data["self_consistency"], errors="coerce"
    )
    data["self_consistency"] = pd.to_numeric(
        data["self_consistency_all_samples"], errors="coerce"
    )
    invalid_consistency = data["self_consistency"].notna() & ~data[
        "self_consistency"
    ].between(0, 1)
    scored_missing_consistency = data["n_scored"].gt(0) & data[
        "self_consistency"
    ].isna()
    if invalid_consistency.any() or scored_missing_consistency.any():
        _validation_issue(
            state,
            "Invalid self-consistency values: "
            f"out_of_range={int(invalid_consistency.sum())}, "
            f"missing_when_scored={int(scored_missing_consistency.sum())}.",
            audit_override,
            "--audit-override",
        )
        data.loc[invalid_consistency, "self_consistency"] = math.nan

    distribution_sources: dict[str, str | None] = {}
    for canonical, candidates in DISTRIBUTION_METRICS.items():
        source = _one_of(data.columns, candidates)
        distribution_sources[canonical] = source
        if source is None:
            _validation_issue(
                state,
                f"Missing inline distribution metric for {canonical}; expected one "
                f"of {list(candidates)}.",
                audit_override,
                "--audit-override",
            )
            data[canonical] = math.nan
            continue
        data[canonical] = pd.to_numeric(data[source], errors="coerce")
        invalid_metric = data[canonical].notna() & ~data[canonical].between(0, 100)
        if invalid_metric.any():
            _validation_issue(
                state,
                f"{int(invalid_metric.sum())} {canonical} values fall outside 0-100.",
                audit_override,
                "--audit-override",
            )
            data.loc[invalid_metric, canonical] = math.nan

    data["inline_distribution_count"] = data["inline_distributions"].map(
        _count_inline_distributions
    )
    malformed_distributions = data["inline_distribution_count"].lt(0)
    if malformed_distributions.any():
        _validation_issue(
            state,
            f"{int(malformed_distributions.sum())} inline_distributions cells are "
            "not valid JSON lists.",
            audit_override,
            "--audit-override",
        )
        data.loc[malformed_distributions, "inline_distribution_count"] = 0

    metric_columns = list(DISTRIBUTION_METRICS)
    data["distribution_metrics_complete"] = data[metric_columns].notna().all(axis=1)
    data["distribution_format_compliant"] = math.nan
    exposed = data[ACTUAL_REQUEST_COLUMN].eq(1)
    data.loc[exposed, "distribution_format_compliant"] = (
        data.loc[exposed, "inline_distribution_count"].eq(
            data.loc[exposed, "n_samples"]
        )
        & data.loc[exposed, "distribution_metrics_complete"]
    ).astype(float)
    data["distribution_missing"] = (
        exposed & data["inline_distribution_count"].eq(0)
    ).astype(float)
    unexpected_distribution = (
        data[ACTUAL_REQUEST_COLUMN].eq(0)
        & data["inline_distribution_count"].gt(0)
    )
    if unexpected_distribution.any():
        _validation_issue(
            state,
            f"{int(unexpected_distribution.sum())} answer-only rows contain inline "
            "probability distributions.",
            audit_override,
            "--audit-override",
        )

    for optional_rate in ("conflict_mention_rate", "abstention_rate"):
        if optional_rate in data.columns:
            data[optional_rate] = pd.to_numeric(data[optional_rate], errors="coerce")
            invalid_optional = data[optional_rate].notna() & ~data[
                optional_rate
            ].between(0, 1)
            if invalid_optional.any():
                _validation_issue(
                    state,
                    f"{int(invalid_optional.sum())} {optional_rate} values fall "
                    "outside 0-1.",
                    audit_override,
                    "--audit-override",
                )
                data.loc[invalid_optional, optional_rate] = math.nan

    key = [
        "protocol_version",
        "dataset_sha256",
        "entity_id",
        "ratio",
        "strategy",
        "model_slot",
        "layout_index",
    ]
    duplicates = data.duplicated(key, keep=False)
    if duplicates.any():
        example = data.loc[duplicates, key].head(3).to_dict(orient="records")
        _validation_issue(
            state,
            f"Found {int(duplicates.sum())} rows with duplicate condition keys "
            f"{key}; examples: {example}",
            audit_override,
            "--audit-override",
        )
        if audit_override:
            before = len(data)
            data = data.drop_duplicates(key, keep="first").copy()
            state.warn(f"Dropped {before - len(data)} duplicate condition rows.")

    _validate_complete_factorial(
        data,
        metadata,
        state,
        allow_incomplete=allow_incomplete,
        audit_override=audit_override,
    )

    conflict_rows = data[data["ratio"].ne("4:0")]
    assignment_variation = conflict_rows.groupby("entity_id", observed=True)[
        ASSIGNMENT_COLUMN
    ].nunique(dropna=True)
    if assignment_variation.gt(1).any():
        _validation_issue(
            state,
            "Distribution-request assignment changes within entity on conflict ratios.",
            audit_override,
            "--audit-override",
        )
    assignment_actual_mismatch = conflict_rows[ASSIGNMENT_COLUMN].ne(
        conflict_rows[ACTUAL_REQUEST_COLUMN]
    )
    if assignment_actual_mismatch.any():
        _validation_issue(
            state,
            f"{int(assignment_actual_mismatch.sum())} conflict rows differ between "
            "assigned arm and actual distribution exposure.",
            audit_override,
            "--audit-override",
        )
    four_zero_exposed = data["ratio"].eq("4:0") & data[ACTUAL_REQUEST_COLUMN].ne(0)
    if four_zero_exposed.any():
        _validation_issue(
            state,
            f"{int(four_zero_exposed.sum())} 4:0 rows request a distribution; v3 "
            "requires actual exposure=0 for every 4:0 condition.",
            audit_override,
            "--audit-override",
        )

    entity_arms = (
        conflict_rows[["entity_id", ASSIGNMENT_COLUMN]]
        .drop_duplicates("entity_id", keep="first")
    )
    observed_controls = int(entity_arms[ASSIGNMENT_COLUMN].eq(0).sum())
    observed_treatment = int(entity_arms[ASSIGNMENT_COLUMN].eq(1).sum())
    expected_treatment = expected_entity_count - expected_control_entities
    if (
        observed_controls != expected_control_entities
        or observed_treatment != expected_treatment
    ):
        _validation_issue(
            state,
            f"Treatment allocation mismatch: expected {expected_treatment} "
            f"distribution / {expected_control_entities} answer-only entities; "
            f"observed {observed_treatment}/{observed_controls}.",
            allow_incomplete or audit_override,
            "--allow-incomplete (or --audit-override)",
        )
    if expected_control_entities < 20:
        state.warn(
            f"The {expected_control_entities}-entity answer-only arm has fewer "
            "than 20 independent control clusters; treat model/ratio-specific "
            "RQ4 comparisons as exploratory and report their intervals."
        )

    complete_primary = data["n_scored"].eq(data["n_samples"])
    valid = data["modal_category"].isin(VALID_CATEGORIES) & complete_primary
    for category in sorted(VALID_CATEGORIES):
        column = f"is_{category.lower()}"
        data[column] = data["modal_category"].eq(category).astype(float)
        data.loc[~valid, column] = math.nan
    data["is_flag_or_com"] = (
        data["modal_category"].isin({"FLAG", "COM"}).astype(float)
    )
    data.loc[~valid, "is_flag_or_com"] = math.nan
    data["diagnostic_tie"] = data["modal_category"].isin(
        {"TIE", "AMBIGUOUS"}
    ).astype(float)
    data["diagnostic_unscored"] = data["modal_category"].eq("UNSCORED").astype(float)
    data["diagnostic_other"] = data["modal_category"].eq("OTHER").astype(float)
    data["diagnostic_partial_scoring"] = data["n_scored"].between(1, 2).astype(float)
    data["primary_api_error_rate"] = data["n_primary_errors"] / data["n_samples"]
    data["primary_format_error_rate"] = (
        data["n_primary_format_errors"] / data["n_samples"]
    )
    data["diagnostic_posthoc_failed"] = data["posthoc_status"].isin(
        {"error", "format_error"}
    ).astype(float)
    data["diagnostic_posthoc_skipped"] = status_is_skip.astype(float)

    provenance = {
        "protocol_versions": protocols,
        "dataset_sha256_values": datasets,
        "confidence_source_columns": confidence_cols,
        "distribution_metric_sources": distribution_sources,
        "expected_protocol_version": EXPECTED_PROTOCOL_VERSION,
        "expected_dataset_sha256": metadata["sha256"],
        "expected_control_entities": expected_control_entities,
        "observed_control_entities": observed_controls,
        "observed_distribution_entities": observed_treatment,
        "allow_incomplete": allow_incomplete,
        "audit_override": audit_override,
        "condition_key": key,
    }
    return data, provenance


def _wilson_interval(successes: int, total: int, z: float = 1.95996398454):
    if total <= 0:
        return math.nan, math.nan
    p = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    half /= denominator
    return max(0.0, center - half), min(1.0, center + half)


def rate_table(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    outcome: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = data.groupby(list(group_columns), dropna=False, observed=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[outcome], errors="coerce").dropna()
        total = int(len(values))
        successes = int(values.sum()) if total else 0
        lower, upper = _wilson_interval(successes, total)
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "outcome": outcome,
                "n_conditions_total": int(len(group)),
                "n_conditions_analyzed": total,
                "n_event": successes,
                "rate": successes / total if total else math.nan,
                "rate_percent": 100 * successes / total if total else math.nan,
                "ci95_low": lower,
                "ci95_high": upper,
                "ci95_low_percent": 100 * lower if total else math.nan,
                "ci95_high_percent": 100 * upper if total else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def confidence_table(data: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = data.groupby(list(group_columns), dropna=False, observed=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(
            group["subjective_posthoc_confidence"], errors="coerce"
        ).dropna()
        n = int(len(values))
        mean = float(values.mean()) if n else math.nan
        sd = float(values.std(ddof=1)) if n > 1 else math.nan
        se = sd / math.sqrt(n) if n > 1 else math.nan
        half = 1.95996398454 * se if n > 1 else math.nan
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_conditions_total": int(len(group)),
                "n_confidence": n,
                "mean_subjective_confidence": mean,
                "median_subjective_confidence": (
                    float(values.median()) if n else math.nan
                ),
                "sd": sd,
                "se": se,
                "ci95_low": max(0.0, mean - half) if n > 1 else math.nan,
                "ci95_high": min(100.0, mean + half) if n > 1 else math.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def continuous_table(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Summarize continuous condition-level diagnostics with normal CIs."""
    rows: list[dict] = []
    grouped = data.groupby(list(group_columns), dropna=False, observed=True)
    for metric in metrics:
        if metric not in data.columns:
            continue
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            n = int(len(values))
            mean = float(values.mean()) if n else math.nan
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            se = sd / math.sqrt(n) if n > 1 else math.nan
            half = 1.95996398454 * se if n > 1 else math.nan
            low = mean - half if n > 1 else math.nan
            high = mean + half if n > 1 else math.nan
            if bounds and metric in bounds and n > 1:
                low = max(bounds[metric][0], low)
                high = min(bounds[metric][1], high)
            row = dict(zip(group_columns, keys))
            row.update(
                {
                    "metric": metric,
                    "n_conditions_total": int(len(group)),
                    "n_observed": n,
                    "n_missing": int(len(group) - n),
                    "mean": mean,
                    "median": float(values.median()) if n else math.nan,
                    "sd": sd,
                    "se": se,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def distribution_compliance_table(
    data: pd.DataFrame, group_columns: Sequence[str]
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = data.groupby(list(group_columns), dropna=False, observed=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        requested = group[ACTUAL_REQUEST_COLUMN].eq(1)
        exposed = group.loc[requested]
        compliant = pd.to_numeric(
            exposed["distribution_format_compliant"], errors="coerce"
        ).dropna()
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_conditions": int(len(group)),
                "n_distribution_exposed": int(requested.sum()),
                "n_answer_only": int((~requested).sum()),
                "n_full_format_compliant": int(compliant.sum()) if len(compliant) else 0,
                "format_compliance_rate_among_exposed": (
                    float(compliant.mean()) if len(compliant) else math.nan
                ),
                "n_distribution_completely_missing": int(
                    exposed["distribution_missing"].sum()
                ),
                "n_missing_p_majority": int(exposed["p_majority"].isna().sum()),
                "n_missing_p_minority": int(exposed["p_minority"].isna().sum()),
                "n_missing_p_indeterminate": int(
                    exposed["p_indeterminate"].isna().sum()
                ),
                "n_missing_p_sources_conflict": int(
                    exposed["p_sources_conflict"].isna().sum()
                ),
                "n_missing_self_consistency": int(group["self_consistency"].isna().sum()),
                "n_missing_posthoc_confidence": int(
                    group["subjective_posthoc_confidence"].isna().sum()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _parse_minority_position(value) -> tuple[str, int, float] | None:
    try:
        positions = json.loads(value) if isinstance(value, str) else value
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(positions, list):
        return None
    normalized = [str(item).strip().upper() for item in positions]
    indices = [index for index, item in enumerate(normalized) if item == "MIN"]
    if len(indices) != 1:
        return None
    index = indices[0]
    if index == 0:
        label = "first"
    elif index == len(normalized) - 1:
        label = "last"
    else:
        label = "middle"
    normalized_index = index / (len(normalized) - 1) if len(normalized) > 1 else 0.0
    return label, index + 1, normalized_index


def add_rq3_positions(data: pd.DataFrame, state: AnalysisState) -> pd.DataFrame:
    subset = data[data["ratio"].isin(SINGLE_MINORITY_RATIOS)].copy()
    parsed = subset["doc_positions"].map(
        _parse_minority_position
    )
    invalid = parsed.isna()
    if invalid.any():
        state.warn(
            f"RQ3 excluded {int(invalid.sum())} single-minority condition rows whose "
            "doc_positions did not contain exactly one MIN label."
        )
    subset = subset.loc[~invalid].copy()
    parsed = parsed.loc[~invalid]
    subset["minority_position"] = parsed.map(lambda value: value[0])
    subset["minority_position_index"] = parsed.map(lambda value: value[1])
    subset["minority_position_normalized"] = parsed.map(lambda value: value[2])
    return subset


def rq4_effect_table(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    outcomes: Sequence[str],
) -> pd.DataFrame:
    """Absolute assigned-minus-control differences with Newcombe score CIs.

    Each table cell contains one condition per entity and layout. The two arms
    therefore contain independent entities. The primary repeated-measures
    inference remains the entity-clustered model below.
    """
    rows: list[dict] = []
    for outcome in outcomes:
        grouped = data.groupby(list(group_columns), dropna=False, observed=True)
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_columns, keys))
            row["outcome"] = outcome
            arm_stats = {}
            for requested, label in ((0.0, "answer_only"), (1.0, "requested")):
                values = pd.to_numeric(
                    group.loc[
                        group[ASSIGNMENT_COLUMN].eq(requested), outcome
                    ],
                    errors="coerce",
                ).dropna()
                n = int(len(values))
                events = int(values.sum()) if n else 0
                low, high = _wilson_interval(events, n)
                rate = events / n if n else math.nan
                arm_stats[label] = (n, events, rate, low, high)
                row[f"n_{label}"] = n
                row[f"events_{label}"] = events
                row[f"rate_{label}"] = rate
                row[f"rate_{label}_percent"] = 100 * rate if n else math.nan
            control = arm_stats["answer_only"]
            treatment = arm_stats["requested"]
            if control[0] and treatment[0]:
                difference = treatment[2] - control[2]
                # Newcombe's hybrid score interval for p_treatment-p_control.
                lower = difference - math.sqrt(
                    (treatment[2] - treatment[3]) ** 2
                    + (control[4] - control[2]) ** 2
                )
                upper = difference + math.sqrt(
                    (treatment[4] - treatment[2]) ** 2
                    + (control[2] - control[3]) ** 2
                )
                lower = max(-1.0, lower)
                upper = min(1.0, upper)
            else:
                difference = lower = upper = math.nan
            row.update(
                {
                    "absolute_difference": difference,
                    "difference_percentage_points": 100 * difference,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "ci95_low_percentage_points": 100 * lower,
                    "ci95_high_percentage_points": 100 * upper,
                    "ci_method": "Newcombe hybrid score interval for two independent proportions",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def rq4_missing_outcome_bounds(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    outcomes: Sequence[str],
) -> pd.DataFrame:
    """Worst-case bounds for assigned-arm effects with nonanalytic outcomes.

    TIE, UNSCORED, and partially scored conditions have no valid modal binary
    outcome. The lower bound assigns every missing treatment outcome to 0 and
    every missing control outcome to 1; the upper bound does the reverse.
    These transparent bounds show how much complete-case conclusions depend on
    treatment-related missingness.
    """
    rows: list[dict] = []
    for outcome in outcomes:
        for keys, group in data.groupby(
            list(group_columns), dropna=False, observed=True
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_columns, keys))
            row["outcome"] = outcome
            stats = {}
            for arm, label in ((0.0, "answer_only"), (1.0, "requested")):
                arm_values = pd.to_numeric(
                    group.loc[group[ASSIGNMENT_COLUMN].eq(arm), outcome],
                    errors="coerce",
                )
                total = int(len(arm_values))
                observed = arm_values.dropna()
                events = int(observed.sum()) if len(observed) else 0
                missing = total - int(len(observed))
                stats[label] = (total, events, missing)
                row[f"n_assigned_{label}"] = total
                row[f"n_observed_{label}"] = int(len(observed))
                row[f"n_missing_{label}"] = missing
                row[f"events_observed_{label}"] = events
            control = stats["answer_only"]
            treatment = stats["requested"]
            if control[0] and treatment[0]:
                lower = (
                    treatment[1] / treatment[0]
                    - (control[1] + control[2]) / control[0]
                )
                upper = (
                    (treatment[1] + treatment[2]) / treatment[0]
                    - control[1] / control[0]
                )
            else:
                lower = upper = math.nan
            row.update(
                {
                    "worst_case_lower_difference": lower,
                    "worst_case_upper_difference": upper,
                    "worst_case_lower_percentage_points": 100 * lower,
                    "worst_case_upper_percentage_points": 100 * upper,
                    "method": "worst-case bounds over TIE/UNSCORED/partial modal outcomes",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def continuous_effect_table(
    data: pd.DataFrame,
    group_columns: Sequence[str],
    metrics: Sequence[str],
) -> pd.DataFrame:
    """Assigned-minus-control mean differences with independent normal CIs."""
    rows: list[dict] = []
    for metric in metrics:
        if metric not in data.columns:
            continue
        for keys, group in data.groupby(
            list(group_columns), dropna=False, observed=True
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_columns, keys))
            row["metric"] = metric
            stats = {}
            for arm, label in ((0.0, "answer_only"), (1.0, "requested")):
                values = pd.to_numeric(
                    group.loc[group[ASSIGNMENT_COLUMN].eq(arm), metric],
                    errors="coerce",
                ).dropna()
                n = int(len(values))
                mean = float(values.mean()) if n else math.nan
                variance = float(values.var(ddof=1)) if n > 1 else math.nan
                stats[label] = (n, mean, variance)
                row[f"n_{label}"] = n
                row[f"mean_{label}"] = mean
            control = stats["answer_only"]
            treatment = stats["requested"]
            if control[0] > 1 and treatment[0] > 1:
                difference = treatment[1] - control[1]
                se = math.sqrt(
                    treatment[2] / treatment[0] + control[2] / control[0]
                )
                lower = difference - 1.95996398454 * se
                upper = difference + 1.95996398454 * se
            else:
                difference = se = lower = upper = math.nan
            row.update(
                {
                    "mean_difference": difference,
                    "standard_error": se,
                    "ci95_low": lower,
                    "ci95_high": upper,
                    "ci_method": "independent two-sample normal interval",
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _finite_exp(value: float) -> float:
    try:
        return math.exp(float(value))
    except (OverflowError, ValueError, TypeError):
        return math.nan


def fit_clustered_model(
    data: pd.DataFrame,
    formula: str,
    outcome: str,
    label: str,
    family: str,
    state: AnalysisState,
) -> None:
    """Fit entity-clustered GEE, falling back to cluster-robust GLM/OLS."""
    required = {"entity_id", outcome}
    if not required.issubset(data.columns):
        state.warn(f"Skipped {label}: missing columns {sorted(required - set(data.columns))}.")
        return
    work = data.dropna(subset=["entity_id", outcome]).copy()
    if len(work) < 10:
        state.warn(f"Skipped {label}: only {len(work)} analyzable conditions.")
        return
    clusters = int(work["entity_id"].nunique())
    if clusters < 2:
        state.warn(f"Skipped {label}: fewer than two entity clusters.")
        return
    if clusters < 20:
        state.warn(
            f"{label} has only {clusters} entity clusters; asymptotic inference "
            "may be unstable."
        )
    if family == "binomial" and work[outcome].nunique() < 2:
        state.warn(f"Skipped {label}: outcome {outcome} is constant.")
        return

    try:
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
    except ImportError:
        state.warn(
            f"Skipped {label}: statsmodels is not installed. Run "
            "pip install -r requirements.txt and rerun the analysis."
        )
        return

    captured: list[str] = []
    result = None
    method = ""
    primary_error = ""
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            sm_family = (
                sm.families.Binomial()
                if family == "binomial"
                else sm.families.Gaussian()
            )
            model = smf.gee(
                formula,
                groups="entity_id",
                data=work,
                family=sm_family,
                cov_struct=sm.cov_struct.Exchangeable(),
            )
            result = model.fit()
            method = "GEE exchangeable/entity-clustered"
            captured.extend(str(item.message) for item in caught)
    except Exception as exc:  # statsmodels raises several design-specific types
        primary_error = f"{type(exc).__name__}: {exc}"

    if result is None:
        state.warn(f"{label} GEE failed ({primary_error}); trying cluster-robust fallback.")
        try:
            with python_warnings.catch_warnings(record=True) as caught:
                python_warnings.simplefilter("always")
                if family == "binomial":
                    result = smf.glm(
                        formula, data=work, family=sm.families.Binomial()
                    ).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": work["entity_id"]},
                    )
                    method = "binomial GLM/entity-cluster-robust SE"
                else:
                    result = smf.ols(formula, data=work).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": work["entity_id"]},
                    )
                    method = "OLS/entity-cluster-robust SE"
                captured.extend(str(item.message) for item in caught)
        except Exception as exc:
            state.warn(
                f"Skipped {label}: clustered fallback also failed "
                f"({type(exc).__name__}: {exc})."
            )
            return

    for message in captured:
        state.warn(f"{label}: statsmodels warning: {message}")

    with python_warnings.catch_warnings(record=True) as result_warnings:
        python_warnings.simplefilter("always")
        try:
            summary = result.summary().as_text()
        except Exception:
            summary = str(result.summary())
        intervals = result.conf_int()
        params = result.params.copy()
        standard_errors = result.bse.copy()
        statistics = result.tvalues.copy()
        p_values = result.pvalues.copy()
    for warning in result_warnings:
        state.warn(f"{label}: statsmodels result warning: {warning.message}")
    state.model_summaries.append(
        "\n".join(
            [
                "=" * 88,
                label,
                f"Method: {method}",
                f"Formula: {formula}",
                f"Conditions: {len(work)}; entity clusters: {clusters}",
                summary,
            ]
        )
    )

    for index, estimate in params.items():
        if hasattr(intervals, "loc"):
            low, high = intervals.loc[index].iloc[0], intervals.loc[index].iloc[1]
        else:
            position = list(params.index).index(index)
            low, high = intervals[position][0], intervals[position][1]
        row = {
            "analysis": label,
            "outcome": outcome,
            "family": family,
            "fit_method": method,
            "formula": formula,
            "term": index,
            "estimate": float(estimate),
            "std_error": float(standard_errors[index]),
            "statistic": float(statistics[index]),
            "p_value": float(p_values[index]),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "n_conditions": int(len(work)),
            "n_entity_clusters": clusters,
            "effect_scale": "log_odds" if family == "binomial" else "points_0_to_100",
            "odds_ratio": _finite_exp(estimate) if family == "binomial" else math.nan,
            "odds_ratio_ci95_low": _finite_exp(low) if family == "binomial" else math.nan,
            "odds_ratio_ci95_high": _finite_exp(high) if family == "binomial" else math.nan,
        }
        state.coefficients.append(row)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def run_analyses(
    data: pd.DataFrame,
    output_dir: Path,
    provenance: dict,
    input_paths: Sequence[Path],
    state: AnalysisState,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}

    def save(name: str, frame: pd.DataFrame) -> None:
        path = output_dir / f"{name}.csv"
        _write_csv(frame, path)
        outputs[name] = str(path.resolve())

    layout_arm = [
        "ratio",
        "model_id",
        "strategy",
        "layout_index",
        ASSIGNMENT_COLUMN,
    ]

    # RQ1: ratio-driven selection and subjective confidence. Every descriptive
    # table preserves randomized arm instead of silently pooling it.
    rq1_outcomes = [
        "is_maj",
        "is_min",
        "is_com",
        "is_flag",
        "is_other",
        "is_flag_or_com",
    ]
    rq1_outcome_rates = pd.concat(
        [rate_table(data, layout_arm, outcome) for outcome in rq1_outcomes],
        ignore_index=True,
    )
    save("rq1_outcome_rates", rq1_outcome_rates)
    save(
        "rq1_majority_rates",
        rq1_outcome_rates[rq1_outcome_rates["outcome"].eq("is_maj")].copy(),
    )
    save(
        "rq1_posthoc_subjective_confidence",
        confidence_table(data, layout_arm),
    )
    save(
        "rq1_inline_distribution_summaries",
        continuous_table(
            data,
            layout_arm,
            list(DISTRIBUTION_METRICS),
            bounds={metric: (0, 100) for metric in DISTRIBUTION_METRICS},
        ),
    )
    save(
        "rq1_self_consistency",
        continuous_table(
            data,
            layout_arm,
            ["self_consistency"],
            bounds={"self_consistency": (0, 1)},
        ),
    )
    arm = f"C({ASSIGNMENT_COLUMN})"
    fit_clustered_model(
        data,
        f"is_maj ~ C(ratio) * C(model_id) + {arm} * C(ratio) + "
        "C(strategy) + C(layout_index)",
        "is_maj",
        "RQ1 majority selection adjusted for distribution arm",
        "binomial",
        state,
    )
    fit_clustered_model(
        data,
        f"subjective_posthoc_confidence ~ C(ratio) * C(model_id) + "
        f"{arm} * C(ratio) + C(strategy) + C(layout_index)",
        "subjective_posthoc_confidence",
        "RQ1 subjective post-hoc confidence adjusted for distribution arm",
        "gaussian",
        state,
    )

    # RQ2: domain rigidity, preserving arm in tables and models.
    rq2_groups = ["domain", *layout_arm]
    save("rq2_domain_flag_rates", rate_table(data, rq2_groups, "is_flag"))
    fit_clustered_model(
        data,
        f"is_flag ~ C(domain) * C(ratio) + C(domain) * {arm} + "
        "C(domain) * C(strategy) + C(model_id) + C(layout_index)",
        "is_flag",
        "RQ2 banking-versus-general conflict flagging adjusted for arm",
        "binomial",
        state,
    )
    rq2_numeric = data[
        data["domain"].eq("banking")
        | (
            data["domain"].eq("general")
            & data["attribute"].isin(NUMERIC_GENERAL_ATTRIBUTES)
        )
    ].copy()
    save(
        "rq2_numeric_only_sensitivity",
        rate_table(rq2_numeric, rq2_groups, "is_flag"),
    )
    fit_clustered_model(
        rq2_numeric,
        f"is_flag ~ C(domain) * C(ratio) + C(domain) * {arm} + "
        "C(domain) * C(strategy) + C(model_id) + C(layout_index)",
        "is_flag",
        "RQ2 numeric-only domain sensitivity adjusted for arm",
        "binomial",
        state,
    )

    # RQ3: all scored outcomes are descriptive; MAJ and FLAG are prespecified
    # inferential outcomes. Position is unambiguous only for one-minority ratios.
    rq3_data = add_rq3_positions(data, state)
    rq3_groups = [
        "minority_position",
        "ratio",
        "model_id",
        "strategy",
        "layout_index",
        ASSIGNMENT_COLUMN,
    ]
    rq3_outcomes = [
        "is_maj",
        "is_min",
        "is_com",
        "is_flag",
        "is_other",
        "is_flag_or_com",
    ]
    save(
        "rq3_position_outcomes",
        pd.concat(
            [rate_table(rq3_data, rq3_groups, outcome) for outcome in rq3_outcomes],
            ignore_index=True,
        ),
    )
    position_balance = (
        rq3_data.groupby(
            ["ratio", "layout_index", "minority_position"],
            dropna=False,
            observed=True,
        )["entity_id"]
        .nunique()
        .rename("n_entities")
        .reset_index()
    )
    save("rq3_position_balance", position_balance)
    for outcome in ("is_maj", "is_flag"):
        fit_clustered_model(
            rq3_data,
            f"{outcome} ~ minority_position_normalized * C(ratio) + "
            f"minority_position_normalized * {arm} + C(model_id) + C(strategy) + "
            "C(layout_index)",
            outcome,
            f"RQ3 normalized minority-position effect on {outcome} adjusted for arm",
            "binomial",
            state,
        )

    # RQ4 uses entity-level assignment. In v3 assigned equals actual on conflict
    # ratios. Point estimates are complete-case, with worst-case missing-outcome
    # bounds reported separately. 4:0 is excluded because neither arm receives
    # a distribution request there.
    rq4_data = data[data["ratio"].ne("4:0")].copy()
    rq4_groups = [
        ASSIGNMENT_COLUMN,
        "model_id",
        "ratio",
        "strategy",
        "layout_index",
    ]
    rq4_effect_groups = ["model_id", "ratio", "strategy", "layout_index"]
    rq4_roles = {
        "is_flag": "primary",
        "is_maj": "secondary",
        "is_com": "secondary",
        "is_flag_or_com": "secondary",
    }
    rq4_rates = pd.concat(
        [rate_table(rq4_data, rq4_groups, outcome) for outcome in rq4_roles],
        ignore_index=True,
    )
    rq4_rates["estimand_role"] = rq4_rates["outcome"].map(rq4_roles)
    save("rq4_confidence_request_rates", rq4_rates)
    rq4_effects = rq4_effect_table(
        rq4_data, rq4_effect_groups, list(rq4_roles)
    )
    rq4_effects["estimand_role"] = rq4_effects["outcome"].map(rq4_roles)
    save("rq4_confidence_request_effects", rq4_effects)
    rq4_bounds = rq4_missing_outcome_bounds(
        rq4_data, rq4_effect_groups, list(rq4_roles)
    )
    rq4_bounds["estimand_role"] = rq4_bounds["outcome"].map(rq4_roles)
    save("rq4_missing_outcome_bounds", rq4_bounds)

    quality_outcomes = [
        "diagnostic_tie",
        "diagnostic_unscored",
        "diagnostic_other",
        "diagnostic_partial_scoring",
        "diagnostic_posthoc_failed",
        "diagnostic_posthoc_skipped",
    ]
    save(
        "rq4_quality_diagnostics",
        pd.concat(
            [rate_table(rq4_data, rq4_groups, outcome) for outcome in quality_outcomes],
            ignore_index=True,
        ),
    )
    save(
        "rq4_distribution_compliance",
        distribution_compliance_table(rq4_data, rq4_groups),
    )
    collection_error_metrics = [
        "primary_api_error_rate",
        "primary_format_error_rate",
    ]
    save(
        "rq4_collection_error_summaries",
        continuous_table(
            rq4_data,
            rq4_groups,
            collection_error_metrics,
            bounds={metric: (0, 1) for metric in collection_error_metrics},
        ),
    )
    save(
        "rq4_collection_error_effects",
        continuous_effect_table(
            rq4_data, rq4_effect_groups, collection_error_metrics
        ),
    )
    save(
        "rq4_distribution_and_consistency",
        continuous_table(
            rq4_data,
            rq4_groups,
            [*DISTRIBUTION_METRICS, "self_consistency"],
            bounds={
                **{metric: (0, 100) for metric in DISTRIBUTION_METRICS},
                "self_consistency": (0, 1),
            },
        ),
    )

    optional_diagnostics = [
        column
        for column in ("conflict_mention_rate", "abstention_rate")
        if column in rq4_data.columns
    ]
    if optional_diagnostics:
        save(
            "rq4_conflict_abstention_summaries",
            continuous_table(
                rq4_data,
                rq4_groups,
                optional_diagnostics,
                bounds={metric: (0, 1) for metric in optional_diagnostics},
            ),
        )
        save(
            "rq4_conflict_abstention_effects",
            continuous_effect_table(
                rq4_data, rq4_effect_groups, optional_diagnostics
            ),
        )
        for outcome in optional_diagnostics:
            fit_clustered_model(
                rq4_data,
                f"{outcome} ~ {arm} * C(model_id) + {arm} * C(ratio) + "
                "C(strategy) + C(domain) + C(layout_index)",
                outcome,
                f"RQ4 distribution-request effect on {outcome}",
                "gaussian",
                state,
            )
    else:
        state.warn(
            "Condition files do not contain conflict_mention_rate and "
            "abstention_rate; the optional distinction between identifying "
            "conflict and abstaining was not analyzed."
        )
        save(
            "rq4_conflict_abstention_summaries",
            pd.DataFrame(columns=[*rq4_groups, "metric", "n_observed", "mean"]),
        )
        save(
            "rq4_conflict_abstention_effects",
            pd.DataFrame(
                columns=[*rq4_effect_groups, "metric", "mean_difference", "ci95_low", "ci95_high"]
            ),
        )

    for outcome, role in rq4_roles.items():
        fit_clustered_model(
            rq4_data,
            f"{outcome} ~ {arm} * C(model_id) + {arm} * C(ratio) + "
            "C(strategy) + C(domain) + C(layout_index)",
            outcome,
            f"RQ4 {role} assigned-arm complete-case effect on {outcome}",
            "binomial",
            state,
        )

    # Secondary Standard-vs-CoT comparison, always stratified/adjusted by arm.
    strategies = set(data["strategy"].dropna().astype(str).str.lower())
    secondary_tables: list[pd.DataFrame] = []
    if {"standard", "cot"}.issubset(strategies):
        secondary_groups = [
            "strategy",
            "model_id",
            "ratio",
            "layout_index",
            ASSIGNMENT_COLUMN,
        ]
        for outcome in rq4_roles:
            secondary_tables.append(rate_table(data, secondary_groups, outcome))
            fit_clustered_model(
                data,
                f"{outcome} ~ C(strategy) * C(ratio) + C(strategy) * {arm} + "
                "C(model_id) + C(domain) + C(layout_index)",
                outcome,
                f"Secondary Standard-versus-CoT comparison for {outcome}",
                "binomial",
                state,
            )
    else:
        state.warn(
            "Secondary Standard-versus-CoT comparison was not run because both "
            "strategies were not present."
        )
    secondary = (
        pd.concat(secondary_tables, ignore_index=True)
        if secondary_tables
        else pd.DataFrame(
            columns=[
                "strategy",
                "model_id",
                "ratio",
                "layout_index",
                ASSIGNMENT_COLUMN,
                "outcome",
                "n_conditions_analyzed",
                "rate",
            ]
        )
    )
    save("secondary_standard_vs_cot", secondary)

    coefficient_columns = [
        "analysis",
        "outcome",
        "family",
        "fit_method",
        "formula",
        "term",
        "estimate",
        "std_error",
        "statistic",
        "p_value",
        "ci95_low",
        "ci95_high",
        "n_conditions",
        "n_entity_clusters",
        "effect_scale",
        "odds_ratio",
        "odds_ratio_ci95_low",
        "odds_ratio_ci95_high",
    ]
    coefficients = pd.DataFrame(state.coefficients, columns=coefficient_columns)
    path = output_dir / "model_coefficients.csv"
    _write_csv(coefficients, path)
    outputs["model_coefficients"] = str(path.resolve())

    summaries_path = output_dir / "model_summaries.txt"
    summaries_path.write_text(
        "\n\n".join(state.model_summaries)
        if state.model_summaries
        else "No inferential model was successfully fit. See analysis_report.txt.\n",
        encoding="utf-8",
    )
    outputs["model_summaries"] = str(summaries_path.resolve())

    category_counts = {
        str(key): int(value)
        for key, value in data["modal_category"].value_counts(dropna=False).items()
    }
    report = {
        "analysis_unit": (
            "one entity x ratio x model x strategy x layout condition; the three "
            "primary calls are not independent observations"
        ),
        "input_files": [str(path.resolve()) for path in input_paths],
        "provenance": provenance,
        "n_condition_rows": int(len(data)),
        "n_entities": int(data["entity_id"].nunique()),
        "models": sorted(data["model_id"].dropna().astype(str).unique().tolist()),
        "strategies": sorted(data["strategy"].dropna().astype(str).unique().tolist()),
        "ratios": sorted(data["ratio"].dropna().astype(str).unique().tolist()),
        "domains": sorted(data["domain"].dropna().astype(str).unique().tolist()),
        "modal_category_counts": category_counts,
        "n_scored_counts": {
            str(key): int(value)
            for key, value in data["n_scored"].value_counts(dropna=False).items()
        },
        "confidence_request_counts": {
            str(key): int(value)
            for key, value in data["inline_confidence_requested"]
            .value_counts(dropna=False)
            .items()
        },
        "distribution_assignment_counts": {
            str(key): int(value)
            for key, value in data[ASSIGNMENT_COLUMN].value_counts(dropna=False).items()
        },
        "rq3_scope": (
            "single-minority ratios 2:1, 3:1, and 4:1; first/middle/last is "
            "derived from doc_positions"
        ),
        "rq4_primary_outcome": "FLAG",
        "rq4_secondary_outcomes": ["MAJ", "COM", "FLAG|COM"],
        "rq4_estimand": (
            "assigned-distribution minus answer-only complete-case effects on "
            "conflict ratios, with worst-case bounds for TIE/UNSCORED/partial "
            "outcomes; assigned equals actual exposure there; 4:0 is excluded"
        ),
        "warnings": state.warnings,
        "outputs": outputs,
    }
    json_path = output_dir / "analysis_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs["analysis_report_json"] = str(json_path.resolve())

    text_lines = [
        "Majority Illusion condition-level analysis report",
        "=" * 52,
        f"Condition rows: {report['n_condition_rows']}",
        f"Entities: {report['n_entities']}",
        f"Models: {', '.join(report['models'])}",
        f"Strategies: {', '.join(report['strategies'])}",
        f"Ratios: {', '.join(report['ratios'])}",
        "",
        "Analysis unit",
        report["analysis_unit"],
        "",
        "Interpretation safeguards",
        "- Confidence is a subjective best-resolution judgment, not calibrated correctness.",
        "- RQ4 primary outcome is FLAG; MAJ, COM, and FLAG-or-COM are secondary.",
        "- RQ4 uses assigned arm, reports complete-case effects plus worst-case missing-outcome bounds, and excludes 4:0.",
        "- TIE, UNSCORED, OTHER, format compliance, and distribution missingness are reported separately.",
        "- Pilot percentages are not encoded here; all estimates are computed from supplied final condition rows.",
        "- RQ3 is restricted to ratios with exactly one minority document.",
        "",
        f"Warnings ({len(state.warnings)})",
    ]
    text_lines.extend(
        [f"- {message}" for message in state.warnings]
        if state.warnings
        else ["- None"]
    )
    text_lines.extend(["", "Outputs"])
    text_lines.extend(f"- {key}: {value}" for key, value in outputs.items())
    text_path = output_dir / "analysis_report.txt"
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")
    outputs["analysis_report_text"] = str(text_path.resolve())

    # Re-write JSON so its output manifest includes both report paths.
    report["outputs"] = outputs
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run condition-level RQ1-RQ4 analyses with entity-clustered inference."
        )
    )
    parser.add_argument(
        "condition_csvs",
        nargs="+",
        type=Path,
        help="one or more condition-level CSVs (for example Standard and CoT)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "analysis",
        help="directory for tables and model reports (default: results/analysis)",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "allow a partial but otherwise valid v3 factorial for a pilot; "
            "never use incomplete output as final evidence"
        ),
    )
    parser.add_argument(
        "--audit-override",
        action="store_true",
        help=(
            "override provenance/protocol/content guards for an explicitly "
            "labeled audit; never use overridden output as final evidence"
        ),
    )
    parser.add_argument(
        "--allow-legacy-or-mixed",
        dest="audit_override",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--expected-control-entities",
        type=int,
        default=37,
        help=(
            "expected answer-only entity clusters (default: 37, paired with "
            "38 structured-elicitation entities)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = AnalysisState()
    try:
        data, provenance = load_condition_data(
            args.condition_csvs,
            state,
            allow_incomplete=args.allow_incomplete,
            audit_override=args.audit_override,
            expected_control_entities=args.expected_control_entities,
        )
        report = run_analyses(
            data,
            args.output_dir,
            provenance,
            args.condition_csvs,
            state,
        )
    except AnalysisInputError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(
        f"Analyzed {report['n_condition_rows']} condition rows from "
        f"{report['n_entities']} entities. Outputs: {args.output_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
