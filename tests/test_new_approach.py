"""Offline regression tests for the revised confidence workflow."""

import hashlib
import json
import unittest

from harness.run_experiment import (
    CONDITION_FIELDS,
    CSV_FIELDS,
    DATA_PATH,
    build_prompt,
    build_posthoc_prompt,
    model_family_matches,
    parse_primary_response,
    select_no_inline_confidence_ids,
)


class RevisedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    def test_fictional_dataset_has_no_researcher_assigned_truth(self):
        for entity in self.dataset["entities"]:
            self.assertNotIn("true_side", entity)
            self.assertNotIn("true_value", entity)

    def test_dataset_has_100_unique_entities_with_40_60_domain_mix(self):
        entities = self.dataset["entities"]
        self.assertEqual(len(entities), 100)
        self.assertEqual(
            [entity["entity_id"] for entity in entities],
            [f"E{index:03d}" for index in range(1, 101)],
        )
        self.assertEqual(len({entity["entity_name"] for entity in entities}), 100)
        self.assertEqual(sum(e["domain"] == "banking" for e in entities), 40)
        self.assertEqual(sum(e["domain"] == "general" for e in entities), 60)
        # Each 25-entity expansion batch is 10 banking + 15 general. Checking the
        # two slices separately guards against a later batch perturbing an
        # earlier one (see BATCHES in generate_dataset.py).
        exp1, exp2 = entities[50:75], entities[75:100]
        self.assertEqual(sum(e["domain"] == "banking" for e in exp1), 10)
        self.assertEqual(sum(e["domain"] == "general" for e in exp1), 15)
        self.assertEqual(sum(e["domain"] == "banking" for e in exp2), 10)
        self.assertEqual(sum(e["domain"] == "general" for e in exp2), 15)

        canonical_first_50 = json.dumps(
            entities[:50], sort_keys=True, separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(canonical_first_50).hexdigest(),
            "2ebcc063baa52d7b60e7a0712b9abaedd9646d824ba806d867ca9cae669e5403",
        )

    def test_every_entity_has_all_six_well_formed_ratio_conditions(self):
        expected_ratios = set(self.dataset["ratios"])
        self.assertEqual(
            expected_ratios,
            {"4:0", "3:1", "2:2", "4:1", "2:1", "3:2"},
        )
        for entity in self.dataset["entities"]:
            self.assertNotEqual(entity["majority_value"], entity["minority_value"])
            self.assertEqual(set(entity["documents"]), expected_ratios)
            for ratio, documents in entity["documents"].items():
                majority_count, minority_count = map(int, ratio.split(":"))
                self.assertEqual(len(documents), majority_count + minority_count)
                self.assertEqual(
                    len({document["style"] for document in documents}),
                    len(documents),
                )

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
        first = build_prompt(
            entity, "3:1", trial_idx=1, run_seed=20260714,
            return_core=True,
        )
        second = build_prompt(
            entity, "3:1", trial_idx=1, run_seed=20260714,
            return_core=True,
        )
        self.assertEqual(first, second)
        self.assertIn("probability_correct", first[0])
        self.assertNotIn("do not equate", first[0].lower())

        control = build_prompt(
            entity, "3:1", trial_idx=1, run_seed=20260714,
            ask_inline_confidence=False,
        )
        self.assertNotIn('"probability_correct"', control[0])

        posthoc = build_posthoc_prompt(first[2], "$25")
        self.assertNotIn("single true factual value", posthoc)
        self.assertIn("best resolution of the supplied documents", posthoc)

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

    def test_output_schemas_do_not_claim_truth_or_calibration(self):
        forbidden = {
            "true_side", "true_value", "modal_correct",
            "calibrated_confidence", "calibration_method",
            "calibration_status", "platt_slope_full",
            "platt_intercept_full",
        }
        self.assertTrue(forbidden.isdisjoint(CSV_FIELDS))
        self.assertTrue(forbidden.isdisjoint(CONDITION_FIELDS))


if __name__ == "__main__":
    unittest.main()
