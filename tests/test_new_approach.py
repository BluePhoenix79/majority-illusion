"""Offline regression tests for the revised confidence workflow."""

import copy
import json
import unittest

from harness.calibration import calibrate_condition_rows
from harness.run_experiment import (
    DATA_PATH,
    build_prompt,
    model_family_matches,
    parse_primary_response,
    select_no_inline_confidence_ids,
)


class RevisedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    def test_ground_truth_is_balanced_and_independent_field(self):
        entities = self.dataset["entities"]
        counts = {
            side: sum(entity["true_side"] == side for entity in entities)
            for side in ("MAJ", "MIN")
        }
        self.assertLessEqual(abs(counts["MAJ"] - counts["MIN"]), 1)
        for entity in entities:
            expected = (
                entity["majority_value"]
                if entity["true_side"] == "MAJ"
                else entity["minority_value"]
            )
            self.assertEqual(entity["true_value"], expected)

    def test_no_inline_control_is_exact_and_domain_stratified(self):
        entities = self.dataset["entities"]
        selected = select_no_inline_confidence_ids(
            entities, 10, self.dataset["seed"]
        )
        self.assertEqual(len(selected), 10)
        self.assertEqual(
            sum(e["entity_id"] in selected and e["domain"] == "banking"
                for e in entities),
            4,
        )
        self.assertEqual(
            sum(e["entity_id"] in selected and e["domain"] == "general"
                for e in entities),
            6,
        )
        self.assertEqual(
            sum(e["entity_id"] in selected and e["true_side"] == "MAJ"
                for e in entities),
            5,
        )
        self.assertEqual(
            sum(e["entity_id"] in selected and e["true_side"] == "MIN"
                for e in entities),
            5,
        )

    def test_repeated_prompt_is_byte_identical(self):
        entity = self.dataset["entities"][0]
        first = build_prompt(entity, "3:1", trial_idx=1, run_seed=20260714)
        second = build_prompt(entity, "3:1", trial_idx=1, run_seed=20260714)
        self.assertEqual(first, second)
        self.assertIn("probability_correct", first[0])

        control = build_prompt(
            entity, "3:1", trial_idx=1, run_seed=20260714,
            ask_inline_confidence=False,
        )
        self.assertNotIn('"probability_correct"', control[0])

    def test_probability_parser_and_legacy_separation(self):
        parsed = parse_primary_response(
            '{"answer":"$25","probability_correct":72}', True
        )
        self.assertEqual(parsed["probability_correct"], 72)
        self.assertEqual(parsed["legacy_confidence_1_5"], "")

        legacy = parse_primary_response(
            '{"answer":"$25","confidence":5}', True
        )
        self.assertEqual(legacy["probability_correct"], "")
        self.assertEqual(legacy["legacy_confidence_1_5"], 5)
        self.assertIn("missing probability_correct", legacy["format_error"])

    def test_model_families_cannot_be_silently_swapped(self):
        self.assertTrue(model_family_matches("deepseek", "deepseek/v4-flash"))
        self.assertTrue(model_family_matches("claude", "anthropic/claude-haiku"))
        self.assertFalse(model_family_matches("deepseek", "anthropic/claude-haiku"))
        self.assertFalse(model_family_matches("claude", "deepseek/v4-flash"))

    def test_platt_calibration_does_not_use_self_consistency(self):
        rows = []
        raw = [90, 80, 30, 20]
        labels = [1, 0, 1, 0]
        for index, (probability, label) in enumerate(zip(raw, labels), start=1):
            rows.append({
                "model_id": "test-model",
                "entity_id": f"E{index:03d}",
                "posthoc_probability": probability,
                "modal_correct": label,
                "self_consistency": 1.0,
            })
        changed_stability = copy.deepcopy(rows)
        for index, row in enumerate(changed_stability):
            row["self_consistency"] = (index + 1) / 10

        calibrate_condition_rows(rows)
        calibrate_condition_rows(changed_stability)
        self.assertEqual(
            [row["calibrated_confidence"] for row in rows],
            [row["calibrated_confidence"] for row in changed_stability],
        )


if __name__ == "__main__":
    unittest.main()
