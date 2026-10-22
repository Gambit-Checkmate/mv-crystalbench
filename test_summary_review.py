"""Independent cluster-weighting and result-grouping regression checks."""

from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from summarize import cluster_interval, load_results, summarize_records


def _row(identifier, base, correct=True, **overrides):
    row = {
        "question_id": identifier, "base_structure": base, "correct": correct,
        "status": "ok", "invalid": False, "model": "m", "provider": "p",
        "condition": "image", "dry_run": False, "qtype": "Q4", "difficulty": "",
        "meta": {}, "dataset_fingerprint": "d", "config_fingerprint": "c",
    }
    row.update(overrides)
    return row


class SummaryReviewTests(unittest.TestCase):
    def test_unequal_cluster_sizes_match_exhaustive_cluster_bootstrap_population(self):
        # These five bases have very different numbers of augmented questions.
        # Enumerating all 5**5 cluster draws gives an independent exact target.
        counts = [(90, 100), (0, 1), (1, 2), (1, 3), (10, 10)]
        rows = [
            _row(f"{base}-{i}", f"base-{base}", i < successes)
            for base, (successes, total) in enumerate(counts) for i in range(total)
        ]
        exhaustive = []
        for drawn in product(range(5), repeat=5):
            successes = sum(counts[i][0] for i in drawn)
            total = sum(counts[i][1] for i in drawn)
            exhaustive.append(successes / total)
        expected = np.quantile(exhaustive, [0.025, 0.975])
        accuracy, low, high, bases, _ = cluster_interval(rows, repeats=40000, seed=83)
        self.assertAlmostEqual(accuracy, 102 / 116)
        self.assertEqual(bases, 5)
        self.assertAlmostEqual(low, expected[0], delta=0.025)
        self.assertAlmostEqual(high, expected[1], delta=0.025)
        self.assertGreater(high - low, 0.25)

    def test_all_skipped_has_zero_denominator_and_no_confidence_interval(self):
        rows = [
            _row("visual-only", None, None, status="skipped", qtype="Q1a", condition="text")
        ]
        overall = next(row for row in summarize_records(rows) if row["scope"] == "overall")
        for key in ("accuracy", "ci_low", "ci_high", "invalid_rate", "error_rate"):
            self.assertIsNone(overall[key])
        self.assertEqual(overall["n"], 0)
        self.assertEqual(overall["n_structures"], 0)
        self.assertEqual(overall["skipped_count"], 1)

    def test_metadata_base_fallback_clusters_augmentations_together(self):
        rows = [
            _row("a1", None, meta={"base_structure": "base-a"}),
            _row("a2", None, False, meta={"base_structure": "base-a"}),
            _row("b1", "base-b", False),
        ]
        accuracy, _, _, bases, _ = cluster_interval(rows)
        self.assertEqual(bases, 2)
        self.assertAlmostEqual(accuracy, 1 / 3)

    def test_conditions_run_fingerprints_and_dry_runs_stay_separate(self):
        rows = [
            _row("q", "base-a", True),
            _row("q", "base-a", False, condition="text"),
            _row("q", "base-a", False, config_fingerprint="another-config"),
            _row("q", "base-a", False, dataset_fingerprint="another-dataset"),
            _row("q", "base-a", False, dry_run=True),
        ]
        overall = [row for row in summarize_records(rows) if row["scope"] == "overall"]
        self.assertEqual(len(overall), 5)
        self.assertTrue(all(row["n"] == 1 for row in overall))

    def test_loader_normalizes_condition_before_deduplicating_copied_results(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "copy.jsonl"
            record = _row("same", "base", condition="image_text")
            first.write_text(json.dumps(record) + "\n", encoding="utf-8")
            record["condition"] = "image+text"
            second.write_text(json.dumps(record) + "\n", encoding="utf-8")
            actual = load_results([first, second])
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0]["condition"], "image+text")

    def test_q2_split_and_q3_q5_subgroup_denominators(self):
        records = [
            _row("q2e", "base-a", qtype="Q2", difficulty="easy"),
            _row("q2h", "base-b", False, qtype="Q2", difficulty="hard"),
            _row("q3t", "base-a", False, qtype="Q3",
                 status="invalid_response", invalid=True, meta={"depth_trap": True}),
            _row("q3f", "base-b", qtype="Q3", meta={"depth_trap": False}),
            _row("q5p", "base-a", qtype="Q5", meta={"negative_type": None}),
            _row("q5n", "base-b", False, qtype="Q5", status="error",
                 meta={"negative_type": "stretch_axis"}),
        ]
        rows = summarize_records(records)
        by_group = {(r["scope"], r["qtype"], r["difficulty"], r["subgroup"]): r for r in rows}
        self.assertEqual(by_group[("qtype", "Q2", "easy", "")]["accuracy"], 1)
        self.assertEqual(by_group[("qtype", "Q2", "hard", "")]["accuracy"], 0)
        trap = by_group[("q3_depth_trap", "Q3", "", "depth_trap=true")]
        self.assertEqual((trap["n"], trap["correct"], trap["invalid_count"]), (1, 0, 1))
        negative = by_group[("q5_negative_type", "Q5", "", "stretch_axis")]
        self.assertEqual((negative["n"], negative["accuracy"], negative["error_count"]), (1, 0, 1))


if __name__ == "__main__":
    unittest.main()
