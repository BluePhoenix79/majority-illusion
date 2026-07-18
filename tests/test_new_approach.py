"""Offline regression tests for the finalized 75-entity protocol."""

import csv
import json
import re
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from harness.run_experiment import (
    CONDITION_FIELDS,
    CSV_FIELDS,
    DATA_PATH,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_FORMAT_RETRIES,
    DEFAULT_GEMINI_MODEL,
    OPENROUTER_ANTHROPIC_BUDGET_TOKENS,
    OPENROUTER_CLAUDE_MAX_TOKENS,
    OPENROUTER_DEFAULT_MAX_TOKENS,
    PROTOCOL_VERSION,
    MockClient,
    build_posthoc_prompt,
    build_prompt,
    call_with_backoff,
    claim_label_mapping,
    dataset_sha256,
    document_side,
    exception_status_code,
    gemini_usage_tokens,
    is_fatal_collection_error,
    is_retryable_generic_gemini_400,
    iter_response_attempts,
    main,
    model_family_matches,
    call_openrouter,
    openrouter_max_tokens,
    openrouter_reasoning_tokens,
    parse_posthoc_response,
    parse_primary_response,
    select_no_inline_confidence_ids,
    summarize_repeats,
    validate_completion_gate,
    validate_protocol_dataset,
)
from visualizations.common import score_response


class FinalizedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        cls.entities = cls.dataset["entities"]

    def test_dataset_is_exactly_75_with_expected_domain_split(self):
        self.assertIsNone(validate_protocol_dataset(self.dataset))
        self.assertEqual(len(self.entities), 75)
        self.assertEqual(len({e["entity_id"] for e in self.entities}), 75)
        self.assertEqual(
            Counter(e["domain"] for e in self.entities),
            Counter({"banking": 30, "general": 45}),
        )

    def test_every_entity_has_all_six_well_formed_ratios(self):
        ratios = ["4:0", "3:1", "2:2", "4:1", "2:1", "3:2"]
        self.assertEqual(self.dataset["ratios"], ratios)
        for entity in self.entities:
            self.assertEqual(set(entity["documents"]), set(ratios))
            for ratio in ratios:
                expected_majority, expected_minority = map(int, ratio.split(":"))
                sides = [
                    document_side(
                        doc["text"], entity["majority_value"],
                        entity["minority_value"],
                    )
                    for doc in entity["documents"][ratio]
                ]
                self.assertEqual(
                    Counter(sides),
                    Counter({"MAJ": expected_majority, "MIN": expected_minority}),
                    (entity["entity_id"], ratio, sides),
                )

    def test_dataset_and_new_csv_schema_have_no_truth_or_calibration_fields(self):
        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        prohibited = {
            "true_value", "true_side", "ground_truth", "calibrated_confidence",
            "platt_confidence", "brier_score", "probability_correct",
        }
        self.assertTrue(prohibited.isdisjoint(set(keys(self.dataset))))
        self.assertTrue(prohibited.isdisjoint(CSV_FIELDS))
        self.assertTrue(prohibited.isdisjoint(CONDITION_FIELDS))
        self.assertIn("confidence_best_resolution", CSV_FIELDS)
        self.assertIn("confidence_best_resolution", CONDITION_FIELDS)

    def test_control_is_exactly_37_and_stratified_by_domain_and_attribute(self):
        selected = select_no_inline_confidence_ids(
            self.entities, 37, self.dataset["seed"]
        )
        self.assertEqual(len(selected), 37)
        selected_entities = [e for e in self.entities if e["entity_id"] in selected]
        self.assertEqual(
            Counter(e["domain"] for e in selected_entities),
            Counter({"banking": 15, "general": 22}),
        )
        by_domain_attribute = defaultdict(Counter)
        all_domain_attributes = defaultdict(set)
        for entity in self.entities:
            all_domain_attributes[entity["domain"]].add(entity["attribute"])
        for entity in selected_entities:
            by_domain_attribute[entity["domain"]][entity["attribute"]] += 1
        for domain in ("banking", "general"):
            self.assertEqual(
                set(by_domain_attribute[domain]), all_domain_attributes[domain]
            )
        self.assertEqual(sum(by_domain_attribute["banking"].values()), 15)
        self.assertEqual(sum(by_domain_attribute["general"].values()), 22)
        self.assertEqual(
            selected,
            select_no_inline_confidence_ids(
                self.entities, 37, self.dataset["seed"]
            ),
        )

    def test_repeated_prompt_is_byte_identical_and_control_is_answer_only(self):
        entity = self.entities[0]
        first = build_prompt(entity, "3:1", trial_idx=1, run_seed=20260714)
        second = build_prompt(entity, "3:1", trial_idx=1, run_seed=20260714)
        self.assertEqual(first, second)
        for field in (
            "p_claim_a", "p_claim_b", "p_indeterminate", "p_sources_conflict"
        ):
            self.assertIn(field, first[0])
        self.assertIn("must sum exactly to 100", first[0])

        control_prompt, _ = build_prompt(
            entity, "3:1", trial_idx=1, run_seed=20260714,
            ask_inline_confidence=False,
        )
        self.assertNotIn("p_claim_a", control_prompt)
        self.assertIn('"answer": "<your answer>"', control_prompt)
        self.assertNotIn("do not report", control_prompt.lower())
        claim_lines = r"Claim A: .*\nClaim B: .*"
        self.assertEqual(
            re.search(claim_lines, first[0]).group(0),
            re.search(claim_lines, control_prompt).group(0),
        )

        cot_prompt, _ = build_prompt(entity, "3:1", strategy="cot")
        self.assertIn("step-by-step reasoning", cot_prompt)

    def test_four_to_zero_is_pure_answer_only_for_every_assignment(self):
        entity = self.entities[0]
        assigned_prompt, _ = build_prompt(
            entity, "4:0", ask_inline_confidence=True
        )
        control_prompt, _ = build_prompt(
            entity, "4:0", ask_inline_confidence=False
        )
        self.assertEqual(assigned_prompt, control_prompt)
        self.assertNotIn("Claim A:", assigned_prompt)
        self.assertNotIn("Claim B:", assigned_prompt)
        self.assertNotIn("p_claim_a", assigned_prompt)
        self.assertIn('"answer": "<your answer>"', assigned_prompt)

    def test_claim_labels_are_exactly_counterbalanced_and_canonicalized(self):
        counts = Counter()
        for entity in self.entities:
            for ratio in self.dataset["ratios"]:
                mapping = claim_label_mapping(
                    entity, ratio, self.dataset["seed"], layout_index=1
                )
                counts[mapping["claim_a_side"]] += 1
                self.assertEqual(
                    {mapping["claim_a_side"], mapping["claim_b_side"]},
                    {"MAJ", "MIN"},
                )
        self.assertEqual(counts, Counter({"MAJ": 225, "MIN": 225}))

        entity = self.entities[0]
        mapping = claim_label_mapping(entity, "3:1", self.dataset["seed"], 1)
        parsed = parse_primary_response(
            '{"answer":"x","p_claim_a":60,"p_claim_b":25,'
            '"p_indeterminate":15,"p_sources_conflict":90}',
            True,
            claim_mapping=mapping,
        )
        self.assertEqual(parsed["format_error"], "")
        self.assertEqual(
            {parsed["p_majority"], parsed["p_minority"]}, {60, 25}
        )
        expected_majority = 60 if mapping["claim_a_side"] == "MAJ" else 25
        self.assertEqual(parsed["p_majority"], expected_majority)

    def test_distribution_parser_enforces_sum_and_independent_conflict_score(self):
        valid = parse_primary_response(
            '{"answer":"x","p_claim_a":40,"p_claim_b":20,'
            '"p_indeterminate":40,"p_sources_conflict":100}', True
        )
        self.assertEqual(valid["format_error"], "")
        invalid = parse_primary_response(
            '{"answer":"x","p_claim_a":40,"p_claim_b":20,'
            '"p_indeterminate":30,"p_sources_conflict":100}', True
        )
        self.assertIn("sum exactly to 100", invalid["format_error"])
        control = parse_primary_response(
            '{"answer":"x","p_claim_a":40}', False
        )
        self.assertIn("control response", control["format_error"])

    def test_numeric_answers_parse_and_malformed_mixed_quotes_stay_detectable(self):
        valid_examples = (
            ('{"answer": 3200}', "3200"),
            ('{"answer": 0}', "0"),
            ('{"answer": 3.5}', "3.5"),
            ('{"answer": "5760"}', "5760"),
        )
        for raw, expected in valid_examples:
            with self.subTest(raw=raw):
                parsed = parse_primary_response(raw, False)
                self.assertEqual(parsed["answer"], expected)
                self.assertEqual(parsed["format_error"], "")

        # This is the exact structure seen in the interrupted CSV: it is not
        # valid numeric JSON because only the closing quote is present. Keep it
        # visible as a format failure so the retry layer can recover it rather
        # than silently altering a model response.
        malformed = parse_primary_response('{"answer": 3200"}', False)
        self.assertEqual(malformed["answer"], "")
        self.assertEqual(malformed["format_error"], "no valid JSON object")

    def test_format_retries_are_identical_logged_attempts_not_extra_samples(self):
        calls = []
        responses = iter((
            '{"answer": 3200"}',
            '{"answer": 3200}',
        ))

        def call(prompt, entity, phase, ask_inline):
            calls.append((prompt, entity, phase, ask_inline))
            return next(responses), 10, 20, 5, DEFAULT_DEEPSEEK_MODEL

        entity = {"entity_id": "E027"}
        attempts = list(iter_response_attempts(
            call,
            "byte-identical prompt",
            entity,
            "primary",
            False,
            lambda raw: parse_primary_response(raw, False),
            max_format_retries=DEFAULT_FORMAT_RETRIES,
        ))
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["attempt_index"], 1)
        self.assertEqual(attempts[0]["parsed"]["format_error"], "no valid JSON object")
        self.assertEqual(attempts[1]["attempt_index"], 2)
        self.assertEqual(attempts[1]["parsed"]["answer"], "3200")
        self.assertEqual(attempts[1]["parsed"]["format_error"], "")
        self.assertEqual(calls, [calls[0], calls[0]])

        exhausted = list(iter_response_attempts(
            lambda *args: ("", 1, 1, 0, DEFAULT_DEEPSEEK_MODEL),
            "same prompt",
            entity,
            "primary",
            True,
            lambda raw: parse_primary_response(raw, True),
            max_format_retries=2,
        ))
        self.assertEqual(len(exhausted), 3)
        self.assertTrue(all(
            attempt["parsed"]["format_error"] for attempt in exhausted
        ))

    def test_permanent_http_errors_are_fatal_but_transient_codes_are_not(self):
        class StatusError(Exception):
            def __init__(self, status_code):
                super().__init__(f"status {status_code}")
                self.status_code = status_code

        for code in (400, 401, 402, 403, 404, 422):
            with self.subTest(code=code):
                error = StatusError(code)
                self.assertEqual(exception_status_code(error), code)
                self.assertTrue(is_fatal_collection_error(error))
        for code in (408, 409, 429, 500, 503):
            with self.subTest(code=code):
                self.assertFalse(is_fatal_collection_error(StatusError(code)))
        self.assertTrue(is_fatal_collection_error(
            RuntimeError("APIStatusError: Error code: 402 - insufficient credits")
        ))

    def test_only_bare_generic_gemini_invalid_argument_is_retryable(self):
        class StatusError(Exception):
            status_code = 400
            status = "INVALID_ARGUMENT"

            def __init__(self, message):
                super().__init__(message)
                self.message = message

        observed = StatusError(
            "Request contains an invalid argument."
        )
        diagnosed = StatusError(
            "400 INVALID_ARGUMENT: temperature must be between 0 and 2"
        )
        extended = StatusError(
            "Request contains an invalid argument. Field max_output_tokens is invalid"
        )
        self.assertTrue(is_retryable_generic_gemini_400(observed))
        self.assertFalse(is_retryable_generic_gemini_400(diagnosed))
        self.assertFalse(is_retryable_generic_gemini_400(extended))
        self.assertFalse(is_retryable_generic_gemini_400(
            RuntimeError("500: Request contains an invalid argument")
        ))

    def test_generic_gemini_400_gets_two_identical_retries(self):
        from google.genai import errors as genai_errors

        calls = []
        error = genai_errors.ClientError(400, {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        })

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise error
            return "accepted"

        self.assertEqual(
            call_with_backoff(flaky, base_delay=0, max_delay=0), "accepted"
        )
        self.assertEqual(len(calls), 3)

    def test_generic_gemini_400_exhausts_after_three_total_calls(self):
        from google.genai import errors as genai_errors

        calls = []
        error = genai_errors.ClientError(400, {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        })

        def always_invalid():
            calls.append(1)
            raise error

        with self.assertRaises(genai_errors.ClientError):
            call_with_backoff(always_invalid, base_delay=0, max_delay=0)
        self.assertEqual(len(calls), 3)

    def test_mixed_gemini_failures_never_exceed_global_attempt_cap(self):
        from google.genai import errors as genai_errors

        generic = genai_errors.ClientError(400, {
            "error": {
                "code": 400,
                "message": "Request contains an invalid argument.",
                "status": "INVALID_ARGUMENT",
            }
        })
        transient = genai_errors.ClientError(429, {
            "error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}
        })
        sequence = [generic, generic, transient, transient]
        calls = []

        def mixed():
            calls.append(1)
            raise sequence[len(calls) - 1]

        with self.assertRaises(genai_errors.ClientError):
            call_with_backoff(mixed, base_delay=0, max_delay=0)
        self.assertEqual(len(calls), 4)

    def test_each_transient_gemini_error_class_is_bounded(self):
        import httpx
        from google.genai import errors as genai_errors

        errors = (
            genai_errors.ClientError(429, {
                "error": {
                    "code": 429, "message": "quota",
                    "status": "RESOURCE_EXHAUSTED",
                }
            }),
            genai_errors.ServerError(503, {
                "error": {
                    "code": 503, "message": "unavailable",
                    "status": "UNAVAILABLE",
                }
            }),
            httpx.ConnectTimeout("temporary connection timeout"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                calls = []

                def unavailable():
                    calls.append(1)
                    raise error

                with self.assertRaises(type(error)):
                    call_with_backoff(unavailable, base_delay=0, max_delay=0)
                self.assertEqual(len(calls), 4)

    def test_diagnosed_gemini_400_is_not_retried(self):
        from google.genai import errors as genai_errors

        calls = []
        error = genai_errors.ClientError(400, {
            "error": {
                "code": 400,
                "message": "temperature must be between 0 and 2",
                "status": "INVALID_ARGUMENT",
            }
        })

        def invalid():
            calls.append(1)
            raise error

        with self.assertRaises(genai_errors.ClientError):
            call_with_backoff(invalid, base_delay=0, max_delay=0)
        self.assertEqual(len(calls), 1)

    def test_mock_collection_logs_retries_without_extra_scientific_samples(self):
        original_call = MockClient.call
        failed_once = set()

        def malformed_once(client, prompt, entity, strategy="standard",
                           phase="primary", ask_inline_confidence=True):
            key = phase
            if key not in failed_once:
                failed_once.add(key)
                return "", 1, 1, 0, f"{client.provider}-MOCK"
            return original_call(
                client, prompt, entity, strategy, phase, ask_inline_confidence
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            condition_path = Path(temp_dir) / "conditions.csv"
            argv = [
                "run_experiment.py", "--mock", "--entity-ids", "E001",
                "--ratios", "4:0", "--models", "gemini", "--trials", "1",
                "--output", str(raw_path),
                "--condition-output", str(condition_path),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                MockClient, "call", new=malformed_once
            ):
                main()

            with raw_path.open(encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with condition_path.open(encoding="utf-8", newline="") as handle:
                conditions = list(csv.DictReader(handle))

        self.assertEqual(len(raw_rows), 4)
        self.assertEqual(
            [row["attempt_status"] for row in raw_rows],
            ["format_retry", "accepted", "format_retry", "accepted"],
        )
        self.assertEqual(
            [row["trial_record"] for row in raw_rows], ["0", "1", "0", "1"]
        )
        self.assertEqual(len(conditions), 1)
        condition = conditions[0]
        self.assertEqual(condition["n_samples"], "1")
        self.assertEqual(condition["n_scored"], "1")
        self.assertEqual(condition["primary_api_attempts"], "2")
        self.assertEqual(condition["primary_format_retries"], "1")
        self.assertEqual(condition["posthoc_api_attempts"], "2")
        self.assertEqual(condition["posthoc_format_retries"], "1")

    def test_mock_collection_flushes_and_aborts_on_first_fatal_error(self):
        class CreditError(Exception):
            status_code = 402

        def no_credit(*args, **kwargs):
            raise CreditError("insufficient credits")

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            condition_path = Path(temp_dir) / "conditions.csv"
            argv = [
                "run_experiment.py", "--mock", "--entity-ids", "E001",
                "--ratios", "4:0", "--models", "gemini", "--trials", "1",
                "--output", str(raw_path),
                "--condition-output", str(condition_path),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                MockClient, "call", new=no_credit
            ):
                with self.assertRaises(SystemExit) as raised:
                    main()
            with raw_path.open(encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with condition_path.open(encoding="utf-8", newline="") as handle:
                conditions = list(csv.DictReader(handle))

        self.assertIn("FATAL collection error", str(raised.exception))
        self.assertEqual(len(raw_rows), 1)
        self.assertEqual(raw_rows[0]["attempt_status"], "fatal_error")
        self.assertEqual(raw_rows[0]["trial_record"], "1")
        self.assertIn("CreditError", raw_rows[0]["error"])
        self.assertEqual(conditions, [])

    def test_collection_aborts_immediately_on_exhausted_primary_format_failure(self):
        def always_blank(client, prompt, entity, strategy="standard",
                         phase="primary", ask_inline_confidence=True):
            return "", 1, 1, 0, f"{client.provider}-MOCK"

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            condition_path = Path(temp_dir) / "conditions.csv"
            argv = [
                "run_experiment.py", "--mock", "--entity-ids", "E001",
                "--ratios", "4:0", "--models", "gemini", "--trials", "1",
                "--output", str(raw_path), "--condition-output",
                str(condition_path),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                MockClient, "call", new=always_blank
            ):
                with self.assertRaises(SystemExit) as raised:
                    main()
            with raw_path.open(encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with condition_path.open(encoding="utf-8", newline="") as handle:
                conditions = list(csv.DictReader(handle))

        self.assertIn("format_exhausted", str(raised.exception))
        self.assertEqual(
            [row["attempt_status"] for row in raw_rows],
            ["format_retry", "format_retry", "format_exhausted"],
        )
        self.assertEqual(conditions, [])

    def test_completion_gate_accepts_only_genuine_modal_tie_skip(self):
        row = {
            "entity_id": "E001", "ratio": "2:2", "model_slot": "gemini",
            "n_samples": 3, "n_scored": 3, "n_primary_errors": 0,
            "n_primary_format_errors": 0, "inline_confidence_requested": 0,
            "n_valid_distributions": 0, "modal_category": "TIE",
            "posthoc_status": "skipped_modal_tie", "posthoc_skipped": 1,
            "posthoc_error": "",
        }
        valid = validate_completion_gate([row], {"gemini": 1}, 3)
        self.assertTrue(valid["passed"])

        failed = dict(row)
        failed["posthoc_status"] = "skipped_all_unscored"
        invalid = validate_completion_gate([failed], {"gemini": 1}, 3)
        self.assertFalse(invalid["passed"])

    def test_collection_aborts_immediately_on_exhausted_posthoc_format(self):
        original_call = MockClient.call

        def fail_posthoc(client, prompt, entity, strategy="standard",
                         phase="primary", ask_inline_confidence=True):
            if phase == "posthoc":
                return "", 1, 1, 0, f"{client.provider}-MOCK"
            return original_call(
                client, prompt, entity, strategy, phase, ask_inline_confidence
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            raw_path = Path(temp_dir) / "raw.csv"
            condition_path = Path(temp_dir) / "conditions.csv"
            argv = [
                "run_experiment.py", "--mock", "--entity-ids", "E001",
                "--ratios", "4:0", "--models", "gemini", "--trials", "1",
                "--format-retries", "0", "--output", str(raw_path),
                "--condition-output", str(condition_path),
            ]
            with patch.object(sys, "argv", argv), patch.object(
                MockClient, "call", new=fail_posthoc
            ):
                with self.assertRaises(SystemExit) as raised:
                    main()
            with raw_path.open(encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
            with condition_path.open(encoding="utf-8", newline="") as handle:
                conditions = list(csv.DictReader(handle))

        self.assertIn("format_exhausted", str(raised.exception))
        self.assertEqual(
            [row["attempt_status"] for row in raw_rows],
            ["accepted", "format_exhausted"],
        )
        self.assertEqual(conditions, [])

    def test_posthoc_is_truth_neutral_with_legacy_read_compatibility(self):
        prompt = build_posthoc_prompt("documents", "answer")
        self.assertIn("best resolution of the supplied documents", prompt)
        self.assertIn("confidence_best_resolution", prompt)
        self.assertNotIn("single true", prompt.lower())
        self.assertEqual(
            parse_posthoc_response('{"confidence_best_resolution":72}'),
            (72, ""),
        )
        self.assertEqual(
            parse_posthoc_response('{"probability_correct":71}'), (71, "")
        )

    def test_exact_position_matching_handles_e003_100_vs_1000(self):
        entity = next(e for e in self.entities if e["entity_id"] == "E003")
        for ratio in self.dataset["ratios"]:
            _, positions = build_prompt(
                entity, ratio, trial_idx=1, run_seed=self.dataset["seed"]
            )
            expected_majority, expected_minority = map(int, ratio.split(":"))
            self.assertEqual(positions.count("MAJ"), expected_majority)
            self.assertEqual(positions.count("MIN"), expected_minority)
            self.assertNotIn("UNK", positions)

    def test_modal_ties_and_all_failures_are_not_given_arbitrary_answers(self):
        tie_rows = [
            {"response_category": category, "parsed_answer": category.lower()}
            for category in ("MAJ", "MIN", "FLAG")
        ]
        tie = summarize_repeats(tie_rows)
        self.assertEqual(tie["modal_category"], "TIE")
        self.assertEqual(tie["modal_tie"], 1)
        self.assertIsNone(tie["representative"])

        failed = summarize_repeats([
            {"response_category": "UNSCORED", "parsed_answer": ""}
            for _ in range(3)
        ])
        self.assertEqual(failed["modal_category"], "UNSCORED")
        self.assertEqual(failed["n_scored"], 0)
        self.assertIsNone(failed["representative"])

    def test_terse_conflict_and_unknown_answers_are_flagged_without_overreach(self):
        base = {
            "majority_value": "$25",
            "minority_value": "$35",
            "claim_a_value": "$25",
            "claim_b_value": "$35",
            "error": "",
        }
        for answer in (
            "unknown",
            "unclear",
            "conflict",
            "conflicting",
            "conflicting sources",
            "conflicting information",
            "conflicting information in documents",
            "the documents are in conflict",
            "indeterminate due to conflicting evidence",
        ):
            with self.subTest(answer=answer):
                scored = score_response({**base, "parsed_answer": answer})
                self.assertEqual(scored["category"], "FLAG")
                self.assertEqual(scored["abstained"], 1)

        for answer in (
            "unclear, but the answer is $25",
            "Unclear Holdings",
            "it is unclear whether $25 is correct",
        ):
            with self.subTest(answer=answer):
                scored = score_response({**base, "parsed_answer": answer})
                self.assertNotEqual(scored["category"], "FLAG")
                self.assertEqual(scored["abstained"], 0)

        # Mentioning a contradiction is not itself abstention when the model
        # still chooses one or both supplied values.
        majority = score_response({
            **base,
            "parsed_answer": "The sources conflict, but the answer is $25.",
        })
        self.assertEqual(majority["category"], "MAJ")
        self.assertEqual(majority["mentions_conflict"], 1)
        self.assertEqual(majority["abstained"], 0)

        compromise = score_response({
            **base,
            "parsed_answer": "The conflicting documents report $25 and $35.",
        })
        self.assertEqual(compromise["category"], "COM")
        self.assertEqual(compromise["mentions_conflict"], 1)
        self.assertEqual(compromise["abstained"], 0)

        unrelated = score_response({**base, "parsed_answer": "Nocturne Holdings"})
        self.assertEqual(unrelated["category"], "OTHER")

    def test_model_slots_and_families_cannot_be_silently_swapped(self):
        self.assertIn("gemini", DEFAULT_GEMINI_MODEL)
        self.assertIn("deepseek", DEFAULT_DEEPSEEK_MODEL)
        self.assertTrue(
            "claude" in DEFAULT_CLAUDE_MODEL or "anthropic" in DEFAULT_CLAUDE_MODEL
        )
        self.assertTrue(model_family_matches("gemini", "gemini-3.5-flash"))
        self.assertTrue(model_family_matches("deepseek", "deepseek/v4-flash"))
        self.assertTrue(model_family_matches("claude", "anthropic/claude-haiku"))
        self.assertFalse(model_family_matches("deepseek", "anthropic/claude-haiku"))
        self.assertFalse(model_family_matches("claude", "deepseek/v4-flash"))

    def test_claude_output_budget_exceeds_reasoning_budget(self):
        self.assertEqual(
            openrouter_max_tokens(DEFAULT_CLAUDE_MODEL),
            OPENROUTER_CLAUDE_MAX_TOKENS,
        )
        self.assertGreater(
            OPENROUTER_CLAUDE_MAX_TOKENS,
            OPENROUTER_ANTHROPIC_BUDGET_TOKENS,
        )
        self.assertEqual(
            openrouter_max_tokens(DEFAULT_DEEPSEEK_MODEL),
            OPENROUTER_DEFAULT_MAX_TOKENS,
        )

    def test_reasoning_token_accounting_does_not_double_count(self):
        gemini_usage = SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            thoughts_token_count=30,
        )
        self.assertEqual(gemini_usage_tokens(gemini_usage), (10, 50, 30))

        usage = SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=100,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=40),
        )
        self.assertEqual(openrouter_reasoning_tokens(usage), 40)

        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                usage=usage,
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content='{"answer":"x"}')
                )],
                model=DEFAULT_CLAUDE_MODEL,
            )

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        result = call_openrouter(
            client, DEFAULT_CLAUDE_MODEL, "prompt", temperature=1.0
        )
        self.assertEqual(captured["max_tokens"], OPENROUTER_CLAUDE_MAX_TOKENS)
        self.assertEqual(result[2], 100)  # already includes the 40 reasoning tokens
        self.assertEqual(result[3], 40)

    def test_protocol_metadata_is_in_both_schemas(self):
        metadata = {
            "protocol_version", "dataset_sha256", "run_seed", "layout_index"
        }
        self.assertTrue(metadata.issubset(CSV_FIELDS))
        self.assertTrue(metadata.issubset(CONDITION_FIELDS))
        self.assertTrue(PROTOCOL_VERSION.endswith("rich-distribution-v3-balanced"))
        for schema in (CSV_FIELDS, CONDITION_FIELDS):
            self.assertIn("reasoning_tokens", schema)
            self.assertIn("distribution_request_assigned", schema)
            self.assertIn("inline_confidence_requested", schema)
        self.assertIn("mentions_conflict", CSV_FIELDS)
        self.assertIn("abstained", CSV_FIELDS)
        for field in ("attempt_index", "attempt_status", "trial_record"):
            self.assertIn(field, CSV_FIELDS)
        for field in (
            "n_primary_errors", "n_primary_format_errors",
            "n_valid_distributions", "posthoc_status", "posthoc_skipped",
            "conflict_mention_count", "abstention_count",
            "primary_api_attempts", "primary_format_retries",
            "posthoc_api_attempts", "posthoc_format_retries",
        ):
            self.assertIn(field, CONDITION_FIELDS)
        digest = dataset_sha256()
        self.assertEqual(len(digest), 64)
        int(digest, 16)


if __name__ == "__main__":
    unittest.main()
