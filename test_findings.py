"""Paired contrasts, coverage, clustered uncertainty, and honest missing outputs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from findings import _difference_interval, ablation_counts_from_summaries, build_findings, paired_difference
from summarize import MAIN_TASKS, build_main_tables, summarize_records


def fixture():
    questions, rows = [], []
    for task in MAIN_TASKS:
        for i in range(25):
            qtype, difficulty = ("Q2", task.split("_")[1]) if task.startswith("Q2_") else (task, "")
            q = dict(id=f"{task}-{i}", qtype=qtype, difficulty=difficulty,
                     meta={"base_structure": f"base-{i}", "text_applicable": task not in {"Q1a", "Q6"}})
            if qtype == "Q3":
                q["meta"]["depth_trap"] = i % 2 == 0
            if qtype == "Q5":
                q["meta"]["negative_type"] = None
            questions.append(q)
            for condition in ("image", "text", "no_image"):
                skipped = condition == "text" and not q["meta"]["text_applicable"]
                correct = q["meta"]["text_applicable"] if condition != "no_image" else len(questions) % 4 == 0
                rows.append(dict(question_id=q["id"], qtype=qtype, difficulty=difficulty,
                                 base_structure=q["meta"]["base_structure"], meta=copy.deepcopy(q["meta"]),
                                 model="gpt", provider="mock", condition=condition, dry_run=False,
                                 correct=None if skipped else correct, status="skipped" if skipped else "ok",
                                 dataset_fingerprint="dataset", config_fingerprint="config", invalid=False))
    return questions, rows


def record(qid, base, correct):
    return dict(question_id=qid, base_question_id=qid, base_structure=base,
                qtype="Q3", difficulty="", status="ok", correct=correct)


class FindingsTests(unittest.TestCase):
    def test_accuracy_cis_match_main_table_and_terminal_summary(self):
        questions, rows = fixture()
        conditions = ("image", "text", "no_image")
        main = build_main_tables(rows, questions, repeats=500, models=("gpt",), conditions=conditions)
        lookup = {(row["condition"], row["task"]): row for row in main}
        terminal = summarize_records(rows, repeats=500)
        for row in terminal:
            if row["scope"] not in {"overall", "qtype"}:
                continue
            task = ("overall" if row["scope"] == "overall" else
                    f"Q2_{row['difficulty']}" if row["qtype"] == "Q2" else row["qtype"])
            expected = lookup[row["condition"], task]
            self.assertEqual((row["accuracy"], row["ci_low"], row["ci_high"]),
                             (expected["accuracy"], expected["ci_low"], expected["ci_high"]))
        _, stats = build_findings(rows, [], questions, models=("gpt",), repeats=500, conditions=conditions)
        overall = {"image": stats["image_overall"]["gpt"],
                   "text": stats["image_text_paired"]["gpt"]["text_full"],
                   "no_image": stats["no_image"]["gpt"]["accuracy"]}
        for condition, row in overall.items():
            expected = lookup[condition, "overall"]
            self.assertEqual((row["accuracy"], row["ci_low"], row["ci_high"]),
                             (expected["accuracy"], expected["ci_low"], expected["ci_high"]))
        for row in stats["lowest"]["gpt"]["task"]:
            expected = lookup["image", row["label"]]
            self.assertEqual((row["accuracy"], row["ci_low"], row["ci_high"]),
                             (expected["accuracy"], expected["ci_low"], expected["ci_high"]))
    def test_ablation_counts_use_planned_summary_and_per_model_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ablation_kimi.jsonl"
            path.with_suffix(".summary.json").write_text(json.dumps({
                "model": "kimi", "full_planned_base_questions": 60,
                "base_questions": 3, "by_task": {"Q1a": 1, "Q2hard": 1, "Q3": 1},
            }), encoding="utf-8")
            counts, sources = ablation_counts_from_summaries([path])
            self.assertEqual(counts, {"kimi": 20})
            self.assertEqual(set(sources), {"kimi"})
            questions, _ = fixture()
            _, stats = build_findings([], [], questions, models=("gpt", "kimi"), repeats=100,
                                      ablation_expected_counts=counts)
            self.assertEqual(stats["ablation"]["gpt/image/Q3/k1_max_minus_min"]["expected_n"], 30)
            self.assertEqual(stats["ablation"]["kimi/image/Q3/k1_max_minus_min"]["expected_n"], 20)

    def test_image_text_uses_150_paired_not_200_vs_150(self):
        questions, rows = fixture()
        text, stats = build_findings(rows, [], questions, models=("gpt",), repeats=100)
        comparison = stats["image_text_paired"]["gpt"]
        self.assertEqual(comparison["image_full"]["accuracy"], .75)
        self.assertEqual(comparison["text_full"]["accuracy"], 1.)
        self.assertEqual(comparison["paired"]["paired_n"], 150)
        self.assertEqual(comparison["paired"]["difference"], 0.)
        self.assertEqual(comparison["paired"]["ci_low"], 0.)
        self.assertIn("common text-applicable question IDs only", text)
        self.assertEqual(stats["no_image"]["gpt"]["versus_chance"]["difference"], 0.)

    def test_paired_cluster_bootstrap_preserves_correlated_pairs(self):
        left = [record(f"a{i}", "a", True) for i in range(10)] + [record(f"b{i}", "b", False) for i in range(10)]
        right = [record(f"a{i}", "a", False) for i in range(10)] + [record(f"b{i}", "b", True) for i in range(10)]
        stats = paired_difference(left, right, repeats=1000, seed=12)
        self.assertEqual((stats["difference"], stats["ci_low"], stats["ci_high"]), (0., -1., 1.))
        self.assertEqual(stats["n_structures"], 2)

    def test_partial_missing_and_failures_remain_explicit(self):
        questions, rows = fixture()
        rows = [row for row in rows if not (row["condition"] == "text" and row["question_id"] == "Q3-0")]
        item = next(row for row in rows if row["condition"] == "image" and row["question_id"] == "Q3-1")
        item.update(status="error", correct=False)
        _, stats = build_findings(rows, [], questions, models=("gpt", "kimi"), repeats=100)
        paired = stats["image_text_paired"]["gpt"]["paired"]
        self.assertEqual((paired["paired_n"], paired["expected_n"], paired["right_missing"]), (149, 150, 1))
        self.assertEqual(paired["left_error_count"], 1)
        self.assertIsNone(stats["image_overall"]["kimi"]["accuracy"])
        self.assertEqual(stats["image_overall"]["kimi"]["missing_count"], 200)
        self.assertTrue(all(row["ci_low"] is None for row in stats["lowest"]["gpt"]["base_structure"]))

    def test_dry_run_never_becomes_real_finding(self):
        questions, rows = fixture()
        for row in rows:
            row["dry_run"] = True
        text, stats = build_findings(rows, [], questions, models=("gpt",), repeats=100)
        self.assertEqual(stats["ignored_dry_run_records"], 600)
        self.assertIsNone(stats["image_overall"]["gpt"]["accuracy"])
        self.assertIn("unavailable (no applicable image results)", text)

    def test_depth_trap_joint_cluster_contrast_and_pair_identity(self):
        left = [record("a", "one", False), record("b", "two", False)]
        right = [record("c", "one", True), record("d", "two", True)]
        stats = _difference_interval(left, right, repeats=100, seed=3)
        self.assertEqual((stats["difference"], stats["ci_low"], stats["ci_high"]), (-1., -1., -1.))
        bad = copy.deepcopy(left)
        bad[0]["base_structure"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity"):
            paired_difference(left, bad, repeats=100)

    def test_ablation_pairing_deduplicates_shared_k8(self):
        questions, rows = fixture()
        ablation = []
        for i in range(2):
            for k, strategy, correct in [(1, "max_visibility", True), (1, "min_visibility", False),
                                         (1, "random", False), *[(8, s, True) for s in
                                           ("random", "canonical", "max_visibility", "min_visibility")]]:
                row = dict(question_id=f"q{i}-{k}-{strategy}", base_question_id=f"q{i}",
                           base_structure=f"base-{i}", qtype="Q3", difficulty="", model="gpt",
                           condition="image", provider="mock", status="ok", correct=correct,
                           dry_run=False, invalid=False, meta={"depth_trap": True},
                           k=k, strategy=strategy, selected_view_ids=[f"v{j}" for j in range(k)],
                           visibility_score_mean=1., prediction="A" if correct else "B",
                           raw_response="same" if k == 8 else strategy,
                           dataset_fingerprint="abl", config_fingerprint="abl-config")
                ablation.append(row)
        _, stats = build_findings(rows, ablation, questions, models=("gpt",), repeats=100)
        max_min = stats["ablation"]["gpt/image/Q3/k1_max_minus_min"]
        random_k8 = stats["ablation"]["gpt/image/Q3/k1_random_minus_k8"]
        self.assertEqual((max_min["difference"], max_min["paired_n"], max_min["expected_n"]), (1., 2, 30))
        self.assertEqual((random_k8["difference"], random_k8["paired_n"]), (-1., 2))
        self.assertEqual(random_k8["unseen_cohort_n"], 28)
        ablation[-1]["correct"] = False
        with self.assertRaisesRegex(ValueError, "shared k=8"):
            build_findings(rows, ablation, questions, models=("gpt",), repeats=100)


if __name__ == "__main__":
    unittest.main()
