"""Focused regression tests for visualization scoring and condition schemas."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

VISUALIZATIONS_DIR = Path(__file__).resolve().parent
if str(VISUALIZATIONS_DIR) not in sys.path:
    sys.path.insert(0, str(VISUALIZATIONS_DIR))

from common import (  # noqa: E402
    CATEGORY_ORDER,
    analytic_rate_rows,
    classify,
    load_results,
    newcombe_difference_ci,
    score_response,
)
from analyze_agreement import condition_table  # noqa: E402


def response(answer, majority="Alice", minority="Bob", error=""):
    return {
        "parsed_answer": answer,
        "majority_value": majority,
        "minority_value": minority,
        "error": error,
    }


class ScoringTests(unittest.TestCase):
    def test_conflict_mention_with_both_values_is_compromise_not_flag(self):
        scored = score_response(response(
            "The sources conflict: some say Alice and others Bob; "
            "the best answer is Alice."
        ))
        self.assertEqual(scored["category"], "COM")
        self.assertEqual(scored["mentions_conflict"], 1)
        self.assertEqual(scored["abstained"], 0)

    def test_conflict_mention_with_one_selected_value_remains_selection(self):
        scored = score_response(response(
            "Although the reports conflict, the best-supported answer is Alice."
        ))
        self.assertEqual(scored["category"], "MAJ")
        self.assertEqual(scored["mentions_conflict"], 1)
        self.assertEqual(scored["abstained"], 0)

    def test_text_value_followed_by_punctuation_is_still_matched(self):
        self.assertEqual(classify(response("The best answer is Alice.")), "MAJ")

    def test_true_refusal_is_flag_and_separate_diagnostics_are_set(self):
        scored = score_response(response(
            "The sources conflict between Alice and Bob, so the answer cannot "
            "be determined from these documents."
        ))
        self.assertEqual(scored["category"], "FLAG")
        self.assertEqual(scored["mentions_conflict"], 1)
        self.assertEqual(scored["abstained"], 1)

    def test_numeric_values_do_not_match_as_substrings(self):
        row = response("The stated lending cap is $1000.", "$100", "$1000")
        self.assertEqual(classify(row), "MIN")

    def test_error_is_unscored(self):
        scored = score_response(response("Alice", error="timeout"))
        self.assertEqual(
            scored,
            {"category": "UNSCORED", "mentions_conflict": 0, "abstained": 0},
        )


class ConditionSchemaTests(unittest.TestCase):
    def test_stability_uses_all_planned_samples(self):
        frame = pd.DataFrame([{
            "modal_category": "MAJ",
            "self_consistency": "1.0",
            "self_consistency_all_samples": str(1 / 3),
            "n_samples": "3",
            "response_categories": '["MAJ", "UNSCORED", "UNSCORED"]',
            "ratio": "3:1",
            "model_id": "model",
        }])
        condition = condition_table(frame).iloc[0]
        self.assertAlmostEqual(condition["self_consistency"], 1 / 3)

    def test_ties_are_visible_but_excluded_from_rate_denominators(self):
        frame = pd.DataFrame({
            "category": ["MAJ", "MIN", "COM", "FLAG", "OTHER", "TIE",
                         "AMBIGUOUS", "UNSCORED"]
        })
        analytic = analytic_rate_rows(frame)
        self.assertEqual(
            set(analytic["category"]), {"MAJ", "MIN", "COM", "FLAG", "OTHER"}
        )
        self.assertIn("TIE", CATEGORY_ORDER)
        self.assertIn("AMBIGUOUS", CATEGORY_ORDER)
        self.assertIn("UNSCORED", CATEGORY_ORDER)

    def test_partial_three_call_condition_is_excluded_from_rates(self):
        frame = pd.DataFrame({
            "category": ["MAJ", "MAJ"],
            "n_scored": [3, 1],
            "n_samples": [3, 3],
        })
        analytic = analytic_rate_rows(frame)
        self.assertEqual(len(analytic), 1)
        self.assertEqual(int(analytic.iloc[0]["n_scored"]), 3)

    def test_new_condition_confidence_and_distribution_fields_are_numeric(self):
        condition = {
            "modal_category": "TIE",
            "modal_tie": "1",
            "modal_answer": "",
            "posthoc_error": "posthoc_skipped_modal_tie",
            "ratio": "3:1",
            "model_id": "deepseek/deepseek-v4-flash",
            "distribution_request_assigned": "1",
            "confidence_best_resolution": "72",
            "mean_p_majority": "55.5",
            "mean_p_minority": "20",
            "mean_p_indeterminate": "24.5",
            "mean_p_sources_conflict": "91",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions_test.csv"
            pd.DataFrame([condition]).to_csv(path, index=False)
            loaded = load_results([path])

        row = loaded.iloc[0]
        self.assertEqual(row["category"], "TIE")
        self.assertEqual(row["subjective_best_resolution_confidence"], 72)
        self.assertEqual(row["confidence"], 72)
        self.assertEqual(row["mean_p_majority"], 55.5)
        self.assertEqual(row["mean_p_minority"], 20)
        self.assertEqual(row["mean_p_indeterminate"], 24.5)
        self.assertEqual(row["mean_p_sources_conflict"], 91)

    def test_legacy_arbitrary_modal_choice_is_made_ambiguous(self):
        condition = {
            "modal_category": "MAJ",
            "modal_tie": "1",
            "modal_answer": "Alice",
            "posthoc_error": "",
            "ratio": "3:1",
            "model_id": "model",
            "distribution_request_assigned": "0",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions_legacy.csv"
            pd.DataFrame([condition]).to_csv(path, index=False)
            loaded = load_results([path])
        self.assertEqual(loaded.iloc[0]["category"], "AMBIGUOUS")
        self.assertTrue(analytic_rate_rows(loaded).empty)

    def test_mixed_arms_require_an_explicit_choice(self):
        rows = []
        for assignment in (1, 0):
            rows.append({
                "modal_category": "MAJ",
                "modal_tie": "0",
                "modal_answer": "Alice",
                "posthoc_error": "",
                "ratio": "3:1",
                "model_id": "model",
                "distribution_request_assigned": assignment,
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions_mixed.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaises(SystemExit):
                load_results([path])
            treatment = load_results([path], arm="distribution")
            control = load_results([path], arm="answer-only")
            both = load_results([path], arm="all")

        self.assertEqual(len(treatment), 1)
        self.assertEqual(treatment.iloc[0]["elicitation_arm"],
                         "Distribution requested")
        self.assertEqual(len(control), 1)
        self.assertEqual(control.iloc[0]["elicitation_arm"], "Answer only")
        self.assertEqual(len(both), 2)

    def test_mixed_strategies_require_an_explicit_choice(self):
        rows = [
            {
                "modal_category": "MAJ",
                "modal_tie": "0",
                "modal_answer": "Alice",
                "posthoc_error": "",
                "ratio": "3:1",
                "model_id": "model",
                "distribution_request_assigned": "1",
                "strategy": strategy,
            }
            for strategy in ("standard", "cot")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "conditions_strategies.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            with self.assertRaises(SystemExit):
                load_results([path], arm="distribution")
            standard = load_results(
                [path], strategy="standard", arm="distribution"
            )
        self.assertEqual(len(standard), 1)

    def test_newcombe_interval_reports_treatment_minus_control(self):
        difference, lower, upper = newcombe_difference_ci(5, 10, 2, 10)
        self.assertAlmostEqual(difference, 0.3)
        self.assertLess(lower, difference)
        self.assertGreater(upper, difference)

        missing = newcombe_difference_ci(0, 0, 2, 10)
        self.assertTrue(all(pd.isna(value) for value in missing))


if __name__ == "__main__":
    unittest.main()
