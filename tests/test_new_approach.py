"""Offline regression tests for the finalized 75-entity protocol."""

import json
import re
import unittest
from collections import Counter, defaultdict
from types import SimpleNamespace

from harness.run_experiment import (
    CONDITION_FIELDS,
    CSV_FIELDS,
    DATA_PATH,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GEMINI_MODEL,
    OPENROUTER_ANTHROPIC_BUDGET_TOKENS,
    OPENROUTER_CLAUDE_MAX_TOKENS,
    OPENROUTER_DEFAULT_MAX_TOKENS,
    PROTOCOL_VERSION,
    build_posthoc_prompt,
    build_prompt,
    claim_label_mapping,
    dataset_sha256,
    document_side,
    gemini_usage_tokens,
    model_family_matches,
    call_openrouter,
    openrouter_max_tokens,
    openrouter_reasoning_tokens,
    parse_posthoc_response,
    parse_primary_response,
    select_no_inline_confidence_ids,
    summarize_repeats,
    validate_protocol_dataset,
)


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
        for field in (
            "n_primary_errors", "n_primary_format_errors",
            "n_valid_distributions", "posthoc_status", "posthoc_skipped",
            "conflict_mention_count", "abstention_count",
        ):
            self.assertIn(field, CONDITION_FIELDS)
        digest = dataset_sha256()
        self.assertEqual(len(digest), 64)
        int(digest, 16)


if __name__ == "__main__":
    unittest.main()
