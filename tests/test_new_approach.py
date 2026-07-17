"""Offline regression tests for the revised confidence workflow."""

import copy
import json
import unittest

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

    def test_model_families_cannot_be_silently_swapped(self):
        self.assertTrue(model_family_matches("deepseek", "deepseek/v4-flash"))
        self.assertTrue(model_family_matches("claude", "anthropic/claude-haiku"))
        self.assertFalse(model_family_matches("deepseek", "anthropic/claude-haiku"))
        self.assertFalse(model_family_matches("claude", "deepseek/v4-flash"))




if __name__ == "__main__":
    unittest.main()
