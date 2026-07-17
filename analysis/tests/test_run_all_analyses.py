import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from harness.run_experiment import PROTOCOL_VERSION as HARNESS_PROTOCOL_VERSION

from analysis.run_all_analyses import (
    EXPECTED_PROTOCOL_VERSION,
    AnalysisInputError,
    AnalysisState,
    current_dataset_metadata,
    load_condition_data,
    run_analyses,
)


POSITIONS = {
    "4:0": ["MAJ", "MAJ", "MAJ", "MAJ"],
    "3:1": ["MIN", "MAJ", "MAJ", "MAJ"],
    "2:2": ["MIN", "MAJ", "MIN", "MAJ"],
    "4:1": ["MAJ", "MIN", "MAJ", "MAJ", "MAJ"],
    "2:1": ["MAJ", "MIN", "MAJ"],
    "3:2": ["MAJ", "MIN", "MAJ", "MIN", "MAJ"],
}
MODELS = {
    "gemini": "gemini-3.5-flash",
    "deepseek": "deepseek/deepseek-v4-flash",
    "claude": "anthropic/claude-haiku-4.5",
}


def complete_current_frame() -> pd.DataFrame:
    metadata = current_dataset_metadata()
    categories = ["MAJ", "MIN", "COM", "FLAG", "OTHER"]
    rows = []
    for entity_index, entity_id in enumerate(metadata["entity_ids"]):
        details = metadata["entity_lookup"][entity_id]
        assigned = int(entity_index >= 37)
        for ratio_index, ratio in enumerate(metadata["ratios"]):
            actual = int(assigned and ratio != "4:0")
            for model_index, (slot, model_id) in enumerate(MODELS.items()):
                category = categories[(entity_index + ratio_index + model_index) % 5]
                distributions = (
                    [
                        {
                            "p_majority": 55,
                            "p_minority": 25,
                            "p_indeterminate": 20,
                            "p_sources_conflict": 80,
                        }
                    ]
                    * 3
                    if actual
                    else []
                )
                rows.append(
                    {
                        "protocol_version": EXPECTED_PROTOCOL_VERSION,
                        "dataset_sha256": metadata["sha256"],
                        "entity_id": entity_id,
                        "domain": details["domain"],
                        "attribute": details["attribute"],
                        "question": details["question"],
                        "majority_value": details["majority_value"],
                        "minority_value": details["minority_value"],
                        "ratio": ratio,
                        "n_docs": len(POSITIONS[ratio]),
                        "strategy": "standard",
                        "layout_index": 1,
                        "doc_positions": json.dumps(POSITIONS[ratio]),
                        "distribution_request_assigned": assigned,
                        "inline_confidence_requested": actual,
                        "model_slot": slot,
                        "model_provider": (
                            "gemini" if slot == "gemini" else "openrouter"
                        ),
                        "model_id": model_id,
                        "returned_model_ids": json.dumps([model_id]),
                        "modal_category": category,
                        "n_samples": 3,
                        "n_scored": 3,
                        "n_primary_errors": 0,
                        "n_primary_format_errors": 0,
                        "n_valid_distributions": 3 if actual else 0,
                        "self_consistency": 2 / 3,
                        "self_consistency_all_samples": 2 / 3,
                        "modal_tie": 0,
                        "inline_distributions": json.dumps(distributions),
                        "mean_p_majority": 55 if actual else "",
                        "mean_p_minority": 25 if actual else "",
                        "mean_p_indeterminate": 20 if actual else "",
                        "mean_p_sources_conflict": 80 if actual else "",
                        "confidence_best_resolution": 65,
                        "posthoc_status": "completed",
                        "posthoc_skipped": 0,
                        "conflict_mention_rate": (entity_index + ratio_index) % 4 / 3,
                        "abstention_rate": (entity_index + model_index) % 4 / 3,
                    }
                )
    return pd.DataFrame(rows)


class AnalysisBalancedV3Tests(unittest.TestCase):
    def write_frame(self, frame: pd.DataFrame, directory: Path, name="conditions.csv"):
        path = directory / name
        frame.to_csv(path, index=False)
        return path

    def test_other_tie_and_unscored_are_distinct(self):
        self.assertEqual(EXPECTED_PROTOCOL_VERSION, HARNESS_PROTOCOL_VERSION)
        with tempfile.TemporaryDirectory() as tmp:
            frame = complete_current_frame()
            frame.loc[0, "modal_category"] = "OTHER"
            frame.loc[1, ["modal_category", "modal_tie", "self_consistency"]] = [
                "TIE",
                1,
                1 / 3,
            ]
            frame.loc[2, "modal_category"] = "UNSCORED"
            frame.loc[2, "n_scored"] = 0
            frame.loc[2, "self_consistency"] = float("nan")
            path = self.write_frame(frame, Path(tmp))
            data, _ = load_condition_data([path], AnalysisState())
            self.assertEqual(data.loc[0, "is_other"], 1)
            self.assertTrue(pd.isna(data.loc[1, "is_flag"]))
            self.assertEqual(data.loc[1, "diagnostic_tie"], 1)
            self.assertEqual(data.loc[2, "diagnostic_unscored"], 1)

    def test_strict_provenance_hash_and_completeness(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            frame = complete_current_frame()
            valid = self.write_frame(frame, directory, "valid.csv")
            loaded, provenance = load_condition_data([valid], AnalysisState())
            self.assertEqual(len(loaded), 75 * 6 * 3)
            self.assertEqual(provenance["observed_control_entities"], 37)

            bad_protocol = frame.copy()
            bad_protocol["protocol_version"] = "old"
            path = self.write_frame(bad_protocol, directory, "bad_protocol.csv")
            with self.assertRaises(AnalysisInputError):
                load_condition_data([path], AnalysisState())

            bad_hash = frame.copy()
            bad_hash["dataset_sha256"] = "0" * 64
            path = self.write_frame(bad_hash, directory, "bad_hash.csv")
            with self.assertRaises(AnalysisInputError):
                load_condition_data([path], AnalysisState())

            bad_metadata = frame.copy()
            bad_metadata.loc[0, "domain"] = "general"
            path = self.write_frame(bad_metadata, directory, "bad_metadata.csv")
            with self.assertRaises(AnalysisInputError):
                load_condition_data([path], AnalysisState())

            bad_returned_model = frame.copy()
            bad_returned_model.loc[0, "returned_model_ids"] = json.dumps(
                ["anthropic/claude-haiku-4.5"]
            )
            path = self.write_frame(
                bad_returned_model, directory, "bad_returned_model.csv"
            )
            with self.assertRaises(AnalysisInputError):
                load_condition_data([path], AnalysisState())

            incomplete = self.write_frame(frame.iloc[:-1], directory, "partial.csv")
            with self.assertRaises(AnalysisInputError):
                load_condition_data([incomplete], AnalysisState())
            partial, _ = load_condition_data(
                [incomplete], AnalysisState(), allow_incomplete=True
            )
            self.assertEqual(len(partial), len(frame) - 1)

    def test_full_outputs_and_prespecified_rq4(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self.write_frame(complete_current_frame(), directory)
            state = AnalysisState()
            data, provenance = load_condition_data([path], state)
            report = run_analyses(
                data, directory / "out", provenance, [path], state
            )
            expected = {
                "rq1_outcome_rates",
                "rq1_inline_distribution_summaries",
                "rq2_domain_flag_rates",
                "rq3_position_outcomes",
                "rq3_position_balance",
                "rq4_confidence_request_effects",
                "rq4_missing_outcome_bounds",
                "rq4_quality_diagnostics",
                "rq4_distribution_compliance",
                "rq4_collection_error_effects",
                "rq4_conflict_abstention_effects",
                "model_coefficients",
                "analysis_report_json",
            }
            self.assertTrue(expected.issubset(report["outputs"]))
            effects = pd.read_csv(
                directory / "out" / "rq4_confidence_request_effects.csv"
            )
            self.assertNotIn("4:0", set(effects["ratio"]))
            self.assertEqual(
                set(effects.loc[effects["estimand_role"].eq("primary"), "outcome"]),
                {"is_flag"},
            )
            self.assertTrue(
                effects["ci_method"].str.contains("Newcombe hybrid").all()
            )
            rq3 = pd.read_csv(directory / "out" / "rq3_position_outcomes.csv")
            self.assertIn("is_other", set(rq3["outcome"]))


if __name__ == "__main__":
    unittest.main()
