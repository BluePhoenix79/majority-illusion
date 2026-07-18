"""Offline tests for non-destructive Standard-run recovery."""

import csv
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness import run_experiment as experiment
from harness import salvage_standard_run as salvage


class StandardSalvageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design = salvage.load_design()
        cls.before_hashes = salvage.source_hashes(
            salvage.DEFAULT_BASE_RAW, salvage.DEFAULT_BASE_CONDITIONS
        )
        cls.report = salvage.audit_files(
            salvage.DEFAULT_BASE_RAW,
            salvage.DEFAULT_BASE_CONDITIONS,
            cls.design,
            salvage.real_model_specs(),
            cls.design.expected_keys,
        )

    def test_committed_partial_has_exact_golden_hashes_and_recovery_set(self):
        self.assertEqual(
            self.before_hashes,
            {
                "raw_sha256": (
                    "0190a32c7365068a96d63b4876cdd7a92bfe4bb41f10d1ab47d5cf783d9bd33b"
                ),
                "conditions_sha256": (
                    "ffba9230d7b8f9e0b08556a003417f6c3a93bcb66e6d943c09b46c4f76643906"
                ),
            },
        )
        self.assertEqual(len(self.design.expected_keys), 1350)
        self.assertEqual(len(self.report.valid_keys), 1085)
        self.assertEqual(len(self.report.pending_keys), 265)
        self.assertFalse(self.report.unexpected_keys)
        self.assertIn(
            salvage.ConditionKey("E021", "2:2", "deepseek"),
            self.report.pending_keys,
        )
        for slot in salvage.MODEL_ORDER:
            self.assertIn(
                salvage.ConditionKey("E061", "2:2", slot),
                self.report.pending_keys,
            )
        self.assertEqual(
            salvage.source_hashes(
                salvage.DEFAULT_BASE_RAW, salvage.DEFAULT_BASE_CONDITIONS
            ),
            self.before_hashes,
        )

    def test_classifier_correction_is_derived_without_changing_posthoc_target(self):
        key = salvage.ConditionKey("E059", "2:2", "gemini")
        self.assertIn(key, self.report.valid_keys)
        corrected_raw = [
            row for row in self.report.valid_raw_rows
            if salvage.row_key(row) == key
            and row["call_phase"] == "primary"
            and row["trial_record"] == "1"
        ]
        corrected_raw.sort(key=lambda row: int(row["trial_index"]))
        self.assertEqual(
            [row["response_category"] for row in corrected_raw],
            ["FLAG", "FLAG", "FLAG"],
        )
        self.assertEqual(corrected_raw[0]["parsed_answer"], "unclear")
        self.assertEqual(corrected_raw[0]["abstained"], "1")
        condition = next(
            row for row in self.report.valid_condition_rows
            if salvage.row_key(row) == key
        )
        self.assertEqual(json.loads(condition["response_categories"]), ["FLAG"] * 3)
        self.assertEqual(condition["modal_count"], "3")
        self.assertEqual(condition["self_consistency"], "1.0")
        self.assertEqual(condition["abstention_count"], "3")
        # This is the actual answer the already-billed post-hoc prompt evaluated.
        self.assertEqual(condition["modal_answer"], "indeterminate")
        self.assertEqual(condition["posthoc_status"], "completed")
        self.assertEqual(condition["confidence_best_resolution"], "100")
        corrections = [
            item for item in self.report.derived_corrections
            if item["key"] == key.label()
        ]
        self.assertEqual(len(corrections), 1)
        self.assertTrue(corrections[0]["preserved_posthoc"])

    def test_pending_conditions_are_bounded_entity_ratio_chunks(self):
        chunks = salvage.plan_chunks(self.report.pending_keys, self.design)
        self.assertEqual(len(chunks), 89)
        self.assertEqual(set().union(*(chunk.keys for chunk in chunks)), self.report.pending_keys)
        self.assertTrue(all(len(chunk.ratios) == 1 for chunk in chunks))
        self.assertTrue(all(1 <= len(chunk.keys) <= 3 for chunk in chunks))
        self.assertEqual(chunks[0].entity_id, "E021")
        self.assertEqual(chunks[0].ratios, ("2:2",))
        self.assertEqual(chunks[0].model_slots, ("deepseek",))

    def test_chunk_command_pins_every_production_setting_and_fresh_paths(self):
        chunk = salvage.plan_chunks(self.report.pending_keys, self.design)[0]
        with tempfile.TemporaryDirectory() as tmp:
            raw, conditions, meta = salvage.attempt_paths(Path(tmp), chunk, 1)
            self.assertNotEqual(raw, conditions)
            self.assertNotEqual(raw, meta)
            self.assertIn("attempt001", raw.name)
            command = salvage.chunk_command(
                chunk, raw, conditions, sys.executable, mock=False
            )
        joined = " ".join(command)
        for fragment in (
            "--strategy standard",
            "--trials 3",
            "--temperature 1.0",
            "--layout-index 1",
            "--no-inline-confidence-entities 37",
            "--models deepseek",
            "--gemini-model gemini-3.5-flash",
            "--deepseek-model deepseek/deepseek-v4-flash",
            "--claude-model anthropic/claude-haiku-4.5",
        ):
            self.assertIn(fragment, joined)
        self.assertNotIn(" cot ", f" {joined} ")
        contract = salvage.fixed_design_manifest(False)["generation_contract"]
        self.assertEqual(contract["gemini_max_output_tokens"], 2048)
        self.assertEqual(contract["openrouter_default_max_tokens"], 2048)
        self.assertEqual(contract["route"]["mode"], "vertex_project_auth")
        self.assertEqual(contract["route"]["location"], "global")
        self.assertNotIn("project", contract["route"])
        self.assertEqual(len(contract["run_experiment_sha256"]), 64)

    def test_exact_schema_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.csv"
            path.write_text("entity_id,ratio\nE001,4:0\n", encoding="utf-8")
            with self.assertRaisesRegex(salvage.SalvageError, "schema mismatch"):
                salvage.read_csv_exact(path, experiment.CSV_FIELDS)

    def test_mock_chunk_is_structurally_valid_offline(self):
        chunk = salvage.plan_chunks(self.report.pending_keys, self.design)[0]
        with tempfile.TemporaryDirectory() as tmp:
            raw, conditions, _ = salvage.attempt_paths(Path(tmp), chunk, 1)
            command = salvage.chunk_command(
                chunk, raw, conditions, sys.executable, mock=True
            )
            completed = subprocess.run(
                command,
                cwd=salvage.REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = salvage.audit_files(
                raw,
                conditions,
                self.design,
                salvage.mock_model_specs(),
                set(chunk.keys),
            )
            self.assertEqual(report.valid_keys, set(chunk.keys))
            self.assertFalse(report.pending_keys)
            self.assertFalse(report.unexpected_keys)

    def test_nonzero_attempt_is_never_reused(self):
        chunk = salvage.plan_chunks(self.report.pending_keys, self.design)[0]
        with tempfile.TemporaryDirectory() as tmp:
            raw, conditions, meta = salvage.attempt_paths(Path(tmp), chunk, 1)
            raw.write_text(",".join(experiment.CSV_FIELDS) + "\n", encoding="utf-8")
            conditions.write_text(
                ",".join(experiment.CONDITION_FIELDS) + "\n", encoding="utf-8"
            )
            meta.write_text(
                json.dumps({
                    "return_code": 1,
                    "design": salvage.fixed_design_manifest(False),
                    "source_hashes": self.before_hashes,
                }),
                encoding="utf-8",
            )
            self.assertIsNone(
                salvage.validate_chunk_attempt(
                    raw,
                    conditions,
                    meta,
                    chunk,
                    self.design,
                    mock=False,
                    initial_source_hashes=self.before_hashes,
                )
            )

    def test_valid_mock_attempt_is_resumed_without_another_subprocess(self):
        chunk = salvage.plan_chunks(self.report.pending_keys, self.design)[0]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            first = salvage.recover_chunks(
                [chunk],
                self.design,
                salvage.DEFAULT_BASE_RAW,
                salvage.DEFAULT_BASE_CONDITIONS,
                self.before_hashes,
                work,
                sys.executable,
                mock=True,
                max_new_attempts=1,
            )
            self.assertEqual(len(first), 1)
            with patch.object(salvage.subprocess, "run") as runner:
                second = salvage.recover_chunks(
                    [chunk],
                    self.design,
                    salvage.DEFAULT_BASE_RAW,
                    salvage.DEFAULT_BASE_CONDITIONS,
                    self.before_hashes,
                    work,
                    sys.executable,
                    mock=True,
                    max_new_attempts=1,
                )
                runner.assert_not_called()
            self.assertEqual(second[0][1], first[0][1])

    def test_orphan_attempt_files_advance_to_a_fresh_number(self):
        chunk = salvage.plan_chunks(self.report.pending_keys, self.design)[0]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            mode_dir = work / "mock"
            mode_dir.mkdir(parents=True)
            orphan_raw, _, _ = salvage.attempt_paths(mode_dir, chunk, 1)
            orphan_raw.write_text("interrupted", encoding="utf-8")
            recovered = salvage.recover_chunks(
                [chunk],
                self.design,
                salvage.DEFAULT_BASE_RAW,
                salvage.DEFAULT_BASE_CONDITIONS,
                self.before_hashes,
                work,
                sys.executable,
                mock=True,
                max_new_attempts=1,
            )
            self.assertIn("attempt002", recovered[0][1].name)
            self.assertEqual(orphan_raw.read_text(encoding="utf-8"), "interrupted")

    def test_any_new_other_response_requires_manual_review(self):
        key = salvage.ConditionKey("E059", "2:2", "gemini")
        condition = copy.deepcopy(next(
            row for row in self.report.valid_condition_rows
            if salvage.row_key(row) == key
        ))
        raw_rows = copy.deepcopy([
            row for row in self.report.valid_raw_rows
            if salvage.row_key(row) == key
        ])
        primary = next(
            row for row in raw_rows
            if row["call_phase"] == "primary"
            and row["trial_record"] == "1"
        )
        payload = json.loads(primary["raw_response"])
        payload["answer"] = "Unclear Holdings"
        primary["raw_response"] = json.dumps(payload)
        primary["parsed_answer"] = "Unclear Holdings"
        primary["response_category"] = "OTHER"
        primary["mentions_conflict"] = "0"
        primary["abstained"] = "0"
        issues = salvage.validate_condition(
            condition,
            raw_rows,
            key,
            self.design,
            salvage.real_model_specs(),
        )
        self.assertTrue(any("OTHER and requires manual review" in issue for issue in issues))

    def test_other_response_halts_instead_of_sampling_a_new_chunk(self):
        chunk = salvage.plan_chunks(self.report.pending_keys, self.design)[0]
        review_report = salvage.AuditReport(
            raw_rows=[],
            condition_rows=[],
            valid_keys=set(),
            pending_keys=set(chunk.keys),
            issues_by_key={
                next(iter(chunk.keys)): [
                    "accepted primary response is OTHER and requires manual review"
                ]
            },
            unexpected_keys=set(),
            derived_corrections=[],
        )
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            salvage.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as runner, patch.object(
            salvage, "validate_chunk_attempt", return_value=None
        ), patch.object(
            salvage, "audit_files", return_value=review_report
        ):
            with self.assertRaisesRegex(
                salvage.SalvageError, "refusing outcome-dependent resampling"
            ):
                salvage.recover_chunks(
                    [chunk],
                    self.design,
                    salvage.DEFAULT_BASE_RAW,
                    salvage.DEFAULT_BASE_CONDITIONS,
                    self.before_hashes,
                    Path(tmp) / "work",
                    sys.executable,
                    mock=True,
                    max_new_attempts=3,
                )
            self.assertEqual(runner.call_count, 1)

    def test_preserved_other_attempt_halts_again_on_next_invocation(self):
        chunk = salvage.plan_chunks(self.report.pending_keys, self.design)[0]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            mode_dir = work / "mock"
            mode_dir.mkdir(parents=True)
            raw, conditions, meta = salvage.attempt_paths(mode_dir, chunk, 1)
            raw.write_text("preserved raw", encoding="utf-8")
            conditions.write_text("preserved conditions", encoding="utf-8")
            meta.write_text(json.dumps({"return_code": 0}), encoding="utf-8")
            with patch.object(
                salvage, "validate_chunk_attempt", return_value=None
            ), patch.object(
                salvage,
                "manual_review_messages",
                return_value=["E021/2:2/deepseek: requires manual review"],
            ), patch.object(salvage.subprocess, "run") as runner:
                with self.assertRaisesRegex(
                    salvage.SalvageError, "refusing outcome-dependent resampling"
                ):
                    salvage.recover_chunks(
                        [chunk],
                        self.design,
                        salvage.DEFAULT_BASE_RAW,
                        salvage.DEFAULT_BASE_CONDITIONS,
                        self.before_hashes,
                        work,
                        sys.executable,
                        mock=True,
                        max_new_attempts=3,
                    )
                runner.assert_not_called()

    def test_existing_final_output_refuses_before_any_recovery_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            output = tmp / "already-there.csv"
            output.write_text("do not overwrite", encoding="utf-8")
            conditions = tmp / "conditions.csv"
            manifest = tmp / "manifest.json"
            with patch.object(salvage, "recover_chunks") as recover:
                with self.assertRaisesRegex(salvage.SalvageError, "refusing to overwrite"):
                    salvage.main([
                        "--output", str(output),
                        "--condition-output", str(conditions),
                        "--manifest", str(manifest),
                        "--work-dir", str(tmp / "work"),
                    ])
                recover.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "do not overwrite")

    def test_partial_three_file_publication_rolls_back_new_destinations(self):
        raw_rows = []
        condition_rows = []
        for key in self.design.expected_keys:
            raw_rows.append({
                "entity_id": key.entity_id,
                "ratio": key.ratio,
                "model_slot": key.model_slot,
                "run_id": "fixture",
                "call_phase": "primary",
                "trial_record": "1",
            })
            condition_rows.append({
                "entity_id": key.entity_id,
                "ratio": key.ratio,
                "model_slot": key.model_slot,
                "run_id": "fixture",
            })
        report = salvage.AuditReport(
            raw_rows=raw_rows,
            condition_rows=condition_rows,
            valid_keys=set(self.design.expected_keys),
            pending_keys=set(),
            issues_by_key={},
            unexpected_keys=set(),
            derived_corrections=[],
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source_raw = tmp / "source_raw.csv"
            source_conditions = tmp / "source_conditions.csv"
            source_raw.write_text("source", encoding="utf-8")
            source_conditions.write_text("source", encoding="utf-8")
            output = tmp / "final_raw.csv"
            conditions = tmp / "final_conditions.csv"
            manifest = tmp / "manifest.json"
            real_replace = salvage.os.replace
            replace_calls = []

            def fail_second_replace(source, destination):
                replace_calls.append(Path(destination))
                if len(replace_calls) == 2:
                    raise OSError("simulated disk failure")
                real_replace(source, destination)

            with patch.object(salvage, "audit_files", return_value=report), patch.object(
                salvage, "validate_full_factorial"
            ), patch.object(
                salvage, "sha256_file", return_value="fixture-hash"
            ), patch.object(salvage.os, "replace", side_effect=fail_second_replace):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    salvage.publish_merge(
                        report,
                        [],
                        self.design,
                        output,
                        conditions,
                        manifest,
                        {"raw": source_raw, "conditions": source_conditions},
                        {
                            "raw_sha256": "fixture-hash",
                            "conditions_sha256": "fixture-hash",
                        },
                    )

            self.assertFalse(output.exists())
            self.assertFalse(conditions.exists())
            self.assertFalse(manifest.exists())
            self.assertEqual(source_raw.read_text(encoding="utf-8"), "source")
            self.assertEqual(
                source_conditions.read_text(encoding="utf-8"), "source"
            )


if __name__ == "__main__":
    unittest.main()
