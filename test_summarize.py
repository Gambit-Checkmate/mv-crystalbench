"""Cluster uncertainty, skipped/invalid denominators and duplicate handling."""

import json
from pathlib import Path
import tempfile
import unittest

from summarize import build_main_tables, cluster_interval, load_results, summarize_records, write_main_tables


def record(identifier, base, correct=True, status="ok", qtype="Q4", **updates):
    value = dict(
        question_id=identifier, base_structure=base, correct=correct, status=status,
        model="test", provider="offline", condition="image", dry_run=True,
        qtype=qtype, difficulty="", invalid=status == "invalid_response",
        meta={}, dataset_fingerprint="dataset", config_fingerprint="config",
    )
    value.update(updates)
    return value


class SummaryTests(unittest.TestCase):
    def test_no_image_and_main_tables_preserve_coverage(self):
        questions = [dict(id="a", qtype="Q4", difficulty="", meta={"text_applicable": True}),
                     dict(id="b", qtype="Q1a", difficulty="", meta={"text_applicable": False})]
        rows = [record("a", "a", model="gpt", condition="no_image", dry_run=False),
                record("b", "b", model="gpt", condition="no_image", dry_run=False, qtype="Q1a",
                       correct=False, status="error"),
                record("a", "a", model="gpt", condition="text", dry_run=False)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            loaded = load_results([path])
            summary = build_main_tables(loaded, questions, repeats=100)
            actual = next(r for r in summary if (r["model"], r["condition"], r["task"]) == ("gpt", "no_image", "overall"))
            self.assertEqual((actual["n"], actual["expected_n"], actual["error_count"], actual["accuracy"]), (2, 2, 1, .5))
            missing = next(r for r in summary if (r["model"], r["condition"], r["task"]) == ("kimi", "image", "overall"))
            self.assertEqual((missing["n"], missing["expected_n"], missing["accuracy"]), (0, 2, None))
            visual_text = next(r for r in summary if (r["model"], r["condition"], r["task"]) == ("gpt", "text", "Q1a"))
            self.assertEqual(visual_text["expected_n"], 0)
            outputs = write_main_tables(loaded, Path(directory) / "main_tables.txt", questions, repeats=100)
            self.assertTrue(all(Path(value).is_file() for value in outputs.values()))

    def test_main_tables_reject_mixed_run_alias(self):
        rows = [record("a", "a", model="gpt", dry_run=False),
                record("b", "b", model="gpt", dry_run=False, config_fingerprint="other")]
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            build_main_tables(rows, repeats=100)

    def test_final_three_conditions_keep_missing_models_visible(self):
        rows = [record("a", "a", model="gpt", dry_run=False)]
        result = build_main_tables(rows, repeats=100, conditions=("image", "text", "no_image"))
        self.assertEqual({row["condition"] for row in result}, {"image", "text", "no_image"})
        self.assertEqual({row["model"] for row in result}, {"gpt", "claude", "kimi", "random"})

    def test_structure_resampling_preserves_correlated_answers(self):
        rows = [record(f"a{i}", "a", True) for i in range(20)]
        rows += [record(f"b{i}", "b", False) for i in range(20)]
        result = cluster_interval(rows, repeats=4000, seed=10)
        self.assertEqual(result[:4], (0.5, 0.0, 1.0, 2))
        self.assertEqual(result, cluster_interval(rows, repeats=4000, seed=10))

    def test_skips_invalid_errors_and_absent_majority_are_distinct(self):
        rows = [
            record("a", "a", True),
            record("b", "b", False, "invalid_response"),
            record("c", "c", False, "error"),
            record("d", "d", False, baseline_detail={"majority_absent": True}),
            record("e", "e", None, "skipped"),
        ]
        overall = next(row for row in summarize_records(rows) if row["scope"] == "overall")
        self.assertEqual(overall["n"], 4)
        self.assertEqual(overall["n_structures"], 4)
        self.assertEqual(overall["accuracy"], 0.25)
        self.assertEqual(overall["invalid_rate"], 0.25)
        self.assertEqual(overall["error_rate"], 0.25)
        self.assertEqual(overall["skipped_count"], 1)

    def test_zero_and_single_cluster_do_not_claim_confidence(self):
        self.assertEqual(cluster_interval([])[:4], (None, None, None, 0))
        self.assertEqual(cluster_interval([record("a", "only")])[:4], (1.0, None, None, 1))
        with self.assertRaises(ValueError):
            cluster_interval([record("a", None)])

    def test_subgroups_and_q2_difficulties(self):
        rows = [
            record("a", "a", qtype="Q3", meta={"depth_trap": True}),
            record("b", "b", qtype="Q3", meta={"depth_trap": False}),
            record("c", "c", qtype="Q5", meta={"negative_type": "polymorph"}),
            record("d", "d", qtype="Q5", meta={"negative_type": None}),
            record("e", "e", qtype="Q2", difficulty="easy"),
            record("f", "f", qtype="Q2", difficulty="hard"),
        ]
        summary = summarize_records(rows)
        self.assertEqual({row["subgroup"] for row in summary if row["scope"] == "q3_depth_trap"},
                         {"depth_trap=true", "depth_trap=false"})
        self.assertEqual({row["subgroup"] for row in summary if row["scope"] == "q5_negative_type"},
                         {"polymorph", "positive"})
        self.assertEqual({row["difficulty"] for row in summary if row["qtype"] == "Q2"},
                         {"easy", "hard"})

    def test_exact_duplicate_dedup_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "a.jsonl", Path(directory) / "b.jsonl"
            value = record("a", "a")
            first.write_text(json.dumps(value) + "\n", encoding="utf-8")
            second.write_text(json.dumps(value) + "\n", encoding="utf-8")
            self.assertEqual(len(load_results([first, second])), 1)
            value["correct"] = False
            second.write_text(json.dumps(value) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                load_results([first, second])

    def test_inconsistent_status_and_q5_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            invalid_records = [
                record("a", "a", True, "error"),
                record("a", "a", False, "invalid_response", invalid=False),
                record("a", "a", False, "skipped"),
                record("a", "a", qtype="Q5", meta={}),
                record("a", "a", qtype="Q5",
                       meta={"negative_type": "polymorph", "is_same_structure": True}),
            ]
            for value in invalid_records:
                with self.subTest(record=value):
                    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_results([path])


if __name__ == "__main__":
    unittest.main()
