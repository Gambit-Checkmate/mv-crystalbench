"""Synthetic checks for ablation aggregation and rendering; never calls an API."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.image as mpimg
from matplotlib.figure import Figure

from ablation_plots import (build_plot_tables, plot_ablation, _figure, _plot_curves,
                            build_multi_plot_tables, plot_multi_ablation, _plot_multi_summary,
                            STRATEGIES, VIEW_COUNTS)


def synthetic_records() -> list[dict]:
    records = []
    for panel in ("Q1a", "Q2hard", "Q3"):
        for index in range(4):
            qid = f"{panel}-{index}"
            for k in VIEW_COUNTS:
                for strategy in STRATEGIES:
                    correct = index == 3
                    records.append({
                        "question_id": f"{qid}::ablation::k{k}::{strategy}",
                        "base_question_id": qid, "base_structure": "base-b" if index == 3 else "base-a",
                        "qtype": "Q2" if panel == "Q2hard" else panel,
                        "difficulty": "hard" if panel == "Q2hard" else "medium",
                        "k": k, "strategy": strategy,
                        "selected_view_ids": [f"scene-{qid}:v{view:02d}" for view in range(k)],
                        "visibility_score_mean": min(1.0, (index + 1) * k / 32),
                        "model": "synthetic", "provider": "dry_run", "condition": "image", "dry_run": True,
                        "status": "ok", "correct": correct, "invalid": False,
                        "prediction": "A" if correct else "B", "raw_response": "Answer: A" if correct else "Answer: B",
                    })
    return records


def synthetic_real_records(model: str = "gpt", per_type: int = 4) -> list[dict]:
    """An in-memory fixture only; these records never become formal results."""
    records = synthetic_records()
    records = [row for row in records if int(row["base_question_id"].rsplit("-", 1)[1]) < per_type]
    for row in records:
        row.update(model=model, provider="synthetic-test-only", dry_run=False, baseline=None,
                   config_fingerprint=f"synthetic-config-{model}", dataset_fingerprint=f"synthetic-data-{per_type}")
    return records


class AblationPlotTests(unittest.TestCase):
    def test_cluster_ci_and_common_endpoint(self):
        curves, binned, info = build_plot_tables(synthetic_records(), bootstrap=1000, seed=9)
        self.assertEqual(len(curves), 48)
        self.assertEqual(len(binned), 15)
        row = next(row for row in curves if row["panel"] == "Q1a" and row["strategy"] == "random" and row["k"] == 1)
        self.assertEqual((row["n"], row["n_structures"], row["accuracy"]), (4, 2, 0.25))
        # Three incorrect questions share a base; cluster bootstrap must keep
        # them together and can draw either all-correct or all-incorrect bases.
        self.assertEqual((row["ci_low"], row["ci_high"]), (0, 1))
        endpoint = [row for row in curves if row["panel"] == "Q1a" and row["k"] == 8]
        self.assertEqual(len({(row["accuracy"], row["ci_low"], row["ci_high"], row["n"]) for row in endpoint}), 1)
        self.assertEqual(info["visibility_observations_before"], 192)
        self.assertEqual(info["visibility_observations_after"], 156)
        self.assertEqual(info["k8_removed_duplicates"], 36)
        self.assertEqual(sum(row["n"] for row in binned), 156)

    def test_reproducibility_and_input_order(self):
        records = synthetic_records()
        first = build_plot_tables(records, bootstrap=100, seed=7)
        self.assertEqual(first, build_plot_tables(reversed(records), bootstrap=100, seed=7))

    def test_error_invalid_and_skip_denominators(self):
        records = synthetic_records()
        chosen = [row for row in records if row["qtype"] == "Q1a" and row["k"] == 1 and row["strategy"] == "random"]
        chosen[0].update(status="error", correct=False)
        chosen[1].update(status="invalid_response", correct=False, invalid=True)
        chosen[2].update(status="skipped", correct=None)
        curves, _, _ = build_plot_tables(records, bootstrap=100)
        row = next(row for row in curves if row["panel"] == "Q1a" and row["strategy"] == "random" and row["k"] == 1)
        self.assertEqual((row["n"], row["correct"], row["error_count"], row["invalid_count"], row["skipped_count"]), (3, 1, 1, 1, 1))
        self.assertAlmostEqual(row["accuracy"], 1 / 3)

    def test_conflicting_duplicate_and_k8_rejected(self):
        records = synthetic_records()
        copied = deepcopy(records[0])
        copied["correct"] = not copied["correct"]
        with self.assertRaisesRegex(ValueError, "Conflicting repeated"):
            build_plot_tables(records + [copied], bootstrap=100)
        next(row for row in records if row["k"] == 8)["visibility_score_mean"] = 0.111
        with self.assertRaisesRegex(ValueError, "shared k=8"):
            build_plot_tables(records, bootstrap=100)

    def test_identical_repeats_and_end_bins(self):
        records = synthetic_records()
        records[0]["visibility_score_mean"] = 0.0
        curves, bins, info = build_plot_tables(records + [deepcopy(records[0])], bootstrap=100)
        self.assertEqual(info["identical_duplicates_removed"], 1)
        self.assertEqual(info["visibility_observations_before"], 192)
        self.assertGreater(next(row["n"] for row in bins if row["panel"] == "Q1a" and row["bin_index"] == 0), 0)
        final = next(row for row in bins if row["panel"] == "Q1a" and row["bin_index"] == 4)
        self.assertTrue(final["right_inclusive"])
        self.assertEqual(final["n"], 1)
        self.assertEqual(final["mean_visibility"], 1.0)

    def test_mixed_run_and_invalid_score_rejected(self):
        records = synthetic_records()
        records[0]["dry_run"] = False
        with self.assertRaisesRegex(ValueError, "one model"):
            build_plot_tables(records, bootstrap=100)
        records = synthetic_records()
        records[0]["visibility_score_mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            build_plot_tables(records, bootstrap=100)

    def test_mixed_fingerprints_rejected(self):
        for field in ("dataset_fingerprint", "config_fingerprint"):
            with self.subTest(field=field):
                records = synthetic_records()
                for row in records:
                    row[field] = "same-run"
                records[0][field] = "another-run"
                with self.assertRaisesRegex(ValueError, "fingerprints"):
                    build_plot_tables(records, bootstrap=100)

    def test_capped_intervals_and_single_shared_k8_bar(self):
        curves, _, info = build_plot_tables(synthetic_records(), bootstrap=100)
        figure, axes = _figure(info, "test")
        with tempfile.TemporaryDirectory() as temp:
            with patch("ablation_plots._figure", return_value=(figure, axes)):
                _plot_curves(curves, info, Path(temp) / "curves.png")
        for axis in axes:
            # Four strategies at k=1,2,4 and exactly one shared interval at k=8.
            self.assertEqual(len(axis.containers), 13)
            locations = [float(container.lines[2][0].get_segments()[0][0, 0]) for container in axis.containers]
            self.assertEqual(locations.count(8), 1)
            for container in axis.containers:
                self.assertEqual(len(container.lines[1]), 2)
                self.assertTrue(all(line.get_markersize() > 0 for line in container.lines[1]))

    def test_two_png_and_csv_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "synthetic.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in synthetic_records()), encoding="utf-8")
            paths = plot_ablation(source, directory / "figures", bootstrap=100)
            for key in ("ablation_views_png", "visibility_vs_accuracy_png"):
                self.assertEqual(mpimg.imread(paths[key]).shape[:2], (1080, 1920))
            for key, count in (("ablation_views_csv", 48), ("visibility_vs_accuracy_csv", 15)):
                with Path(paths[key]).open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), count)
            info = json.loads(Path(paths["plot_info_json"]).read_text(encoding="utf-8"))
            self.assertTrue(info["dry_run"])
            self.assertIn("not a causal", info["interpretation"])


class MultiAblationPlotTests(unittest.TestCase):
    def test_complete_models_with_different_quotas_and_clustered_pooling(self):
        runs = {"kimi": synthetic_real_records("kimi", 2), "gpt": synthetic_real_records(),
                "claude": synthetic_real_records("claude")}
        curves, bins, summary, info = build_multi_plot_tables(
            runs, bootstrap=1000, seed=9, expected_per_type={"gpt": 4, "claude": 4, "kimi": 2})
        self.assertEqual(info["models"], ["gpt", "claude", "kimi"])
        self.assertEqual((len(curves), len(bins), len(summary)), (144, 45, 24))
        self.assertEqual(info["shared_base_questions_across_models"], 6)
        gpt = next(row for row in summary if row["model"] == "gpt" and row["k"] == 1)
        self.assertEqual((gpt["n"], gpt["n_structures"], gpt["accuracy"]), (12, 2, 0.25))
        # The same two bases span all three task groups, so pooling must not
        # invent six independent clusters or bootstrap each task separately.
        self.assertEqual((gpt["ci_low"], gpt["ci_high"]), (0, 1))
        for model in info["models"]:
            shared = [row for row in summary if row["model"] == model and row["k"] == 8]
            self.assertEqual(len({(row["accuracy"], row["ci_low"], row["ci_high"], row["n"]) for row in shared}), 1)

    def test_partial_model_excluded_without_empty_row(self):
        partial = synthetic_real_records("claude")[:-1]
        curves, _, summary, info = build_multi_plot_tables(
            {"gpt": synthetic_real_records(), "claude": partial}, bootstrap=100)
        self.assertEqual(info["models"], ["gpt"])
        self.assertEqual({row["model"] for row in curves + summary}, {"gpt"})
        self.assertEqual(info["excluded_runs"][0]["model"], "claude")
        self.assertIn("16 strategy/view", str(info["excluded_runs"]))

    def test_skipped_and_all_request_failure_runs_excluded(self):
        failed = synthetic_real_records("claude")
        for row in failed:
            row.update(status="error", correct=False, prediction=None, raw_response="")
        _, _, _, info = build_multi_plot_tables(
            {"gpt": synthetic_real_records(), "claude": failed}, bootstrap=100)
        self.assertEqual(info["models"], ["gpt"])
        with self.assertRaisesRegex(ValueError, "No complete real"):
            build_multi_plot_tables({"claude": failed}, bootstrap=100)
        partial = synthetic_real_records("kimi")
        partial[0].update(status="skipped", correct=None)
        _, _, _, info = build_multi_plot_tables(
            {"gpt": synthetic_real_records(), "kimi": partial}, bootstrap=100)
        self.assertEqual(info["models"], ["gpt"])

    def test_terminal_errors_remain_wrong_without_selective_removal(self):
        rows = synthetic_real_records()
        row = next(row for row in rows if row["correct"] and row["k"] == 1 and row["strategy"] == "max_visibility")
        row.update(status="error", correct=False, prediction=None, raw_response="")
        _, _, summary, _ = build_multi_plot_tables({"gpt": rows}, bootstrap=100)
        result = next(row for row in summary if row["strategy"] == "max_visibility" and row["k"] == 1)
        self.assertEqual((result["n"], result["error_count"], result["correct"]), (12, 1, 2))

    def test_dry_run_baseline_and_changed_shared_geometry_rejected(self):
        with self.assertRaisesRegex(ValueError, "dry_run=false"):
            build_multi_plot_tables({"gpt": synthetic_real_records(), "synthetic": synthetic_records()}, bootstrap=100)
        baseline = synthetic_real_records()
        baseline[0]["baseline"] = "random"
        with self.assertRaisesRegex(ValueError, "Baseline"):
            build_multi_plot_tables({"gpt": baseline}, bootstrap=100)
        changed = synthetic_real_records("claude")
        changed[0]["selected_view_ids"] = ["different-scene:v00"]
        with self.assertRaisesRegex(ValueError, "geometry/answer"):
            build_multi_plot_tables({"gpt": synthetic_real_records(), "claude": changed}, bootstrap=100)

    def test_quota_and_task_balance(self):
        with self.assertRaisesRegex(ValueError, "expected 5"):
            build_multi_plot_tables({"gpt": synthetic_real_records()}, bootstrap=100, expected_per_type=5)
        unbalanced = [row for row in synthetic_real_records() if row["base_question_id"] != "Q3-3"]
        with self.assertRaisesRegex(ValueError, "equal nonzero"):
            build_multi_plot_tables({"gpt": unbalanced}, bootstrap=100)

    def test_file_outputs_and_missing_failed_model(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            inputs = []
            for model, count in (("gpt", 4), ("claude", 4), ("kimi", 2)):
                source = directory / f"ablation_{model}.jsonl"
                source.write_text("".join(json.dumps(row) + "\n" for row in synthetic_real_records(model, count)), encoding="utf-8")
                source.with_suffix(".summary.json").write_text(json.dumps({"full_planned_base_questions": count * 3}), encoding="utf-8")
                inputs.append(source)
            inputs.append(directory / "missing.jsonl")
            paths = plot_multi_ablation(inputs, directory / "figures", bootstrap=100)
            self.assertEqual(len(paths), 7)
            self.assertEqual(mpimg.imread(paths["ablation_summary_png"]).shape[:2], (1080, 1920))
            info = json.loads(Path(paths["plot_info_json"]).read_text(encoding="utf-8"))
            self.assertEqual(info["models"], ["gpt", "claude", "kimi"])
            self.assertEqual(info["runs"]["kimi"]["expected_per_type"], 2)
            self.assertIn("result file not found", str(info["excluded_runs"]))
            for key, expected in (("ablation_views_csv", 144), ("ablation_summary_csv", 24), ("visibility_vs_accuracy_csv", 45)):
                with Path(paths[key]).open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), expected)

    def test_sidecar_detects_balanced_but_truncated_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source = directory / "partial.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in synthetic_real_records("gpt", 2)), encoding="utf-8")
            source.with_suffix(".summary.json").write_text(json.dumps({"full_planned_base_questions": 12}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected 4"):
                plot_multi_ablation([source], directory / "figures", bootstrap=100)

    def test_summary_has_one_capped_k8_interval_per_model(self):
        _, _, summary, info = build_multi_plot_tables(
            {"gpt": synthetic_real_records(), "claude": synthetic_real_records("claude")}, bootstrap=100)
        figure = Figure(figsize=(12.8, 7.2), dpi=150)
        with tempfile.TemporaryDirectory() as temp:
            with patch("ablation_plots.Figure", return_value=figure):
                _plot_multi_summary(summary, info, Path(temp) / "summary.png")
        axis = figure.axes[0]
        # Each model has 2 strategies × 3 non-shared k values + 1 shared k=8.
        self.assertEqual(len(axis.containers), 14)
        positions = [float(container.lines[2][0].get_segments()[0][0, 0]) for container in axis.containers]
        self.assertEqual(positions.count(8), 2)
        curves = [line for line in axis.lines if line.get_marker() in {"o", "s"}]
        self.assertEqual([line.get_linestyle() for line in curves], ["-", "--", "-", "--"])
        self.assertEqual(curves[0].get_color(), curves[1].get_color())
        self.assertNotEqual(curves[0].get_color(), curves[2].get_color())


if __name__ == "__main__":
    unittest.main()
