"""Offline tests of view selection, response reuse, and checkpoint expansion."""

import asyncio
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase.build import bulk
from PIL import Image

import ablation


MATRIX = np.array([
    [1, 0, 0], [0, 1, 0], [.5, .5, .5], [0, 0, 1],
    [.25, .25, 0], [.125, 0, 0], [.75, 0, 0], [0, .125, 0],
])


class AnswersForbidden(dict):
    """Detect accidental answer-dependent use by the selection path."""

    def __getitem__(self, key):
        if key in {"answer", "answer_text", "coordination_number"}:
            raise AssertionError(f"Selector accessed {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in {"answer", "answer_text", "coordination_number"}:
            raise AssertionError(f"Selector accessed {key}")
        return super().get(key, default)


def fixture():
    """Small complete Na first shell, with eight deterministic visible-fraction rows."""
    original = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    positions = [[0., 0., 0.]] + [
        [sign * 2.82 if dimension == axis else 0. for dimension in range(3)]
        for axis in range(3) for sign in (-1, 1)
    ]
    atoms = [{"display_index": i, "element": "Na" if i == 0 else "Cl",
              "position_angstrom": position, "highlight": "ABC"[i] if i < 3 else None}
             for i, position in enumerate(positions)]
    matrix = np.column_stack([MATRIX, MATRIX, MATRIX[:, 0]])
    views = [{"view_id": f"v{i:02d}", "image_path": f"v{i:02d}.png",
              "metadata": {
                  "atoms": [{"visible_fraction": float(x)} for x in matrix[i]],
                  "bonds": [{"display_indices": [0, j]} for j in range(1, 7)],
              }} for i in range(8)]
    scene = {
        "scene_id": "fixture", "views": views, "display_atoms": atoms,
        "atoms": {"atomic_numbers": original.numbers.tolist(),
                  "positions_angstrom": original.positions.tolist(),
                  "lattice_vectors_angstrom": original.cell.tolist(),
                  "periodic": original.pbc.tolist(), "bond_pairs": [["Na", "Cl"]]},
    }
    questions = []
    for task in ablation.TASKS:
        q = {"id": f"q-{task}", "qtype": "Q2" if task == "Q2hard" else task,
             "difficulty": "hard", "structure": "SECRET", "scene_id": "fixture",
             "view_ids": [f"fixture:v{i:02d}" for i in range(3)],
             "image_paths": [f"v{i:02d}.png" for i in range(3)],
             "question": "The three images show the same finite cell. Select a choice.",
             "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
             "answer": "A", "answer_text": "one", "text_repr": "",
             "meta": {"base_structure": "fixture", "element": "Cl",
                      "center_display_index": 0, "neighbor_display_indices": list(range(1, 7)),
                      "coordination_number": 6, "display_indices": {"A": 0, "B": 1, "C": 2},
                      "text_applicable": task != "Q1a"}}
        questions.append(q)
    return questions, {"fixture": scene}


class SelectorTests(unittest.TestCase):
    def test_greedy_uses_updated_union_in_both_directions(self):
        self.assertEqual(ablation.select_view_indices(MATRIX, 4, "max_visibility"), [2, 0, 1, 3])
        self.assertEqual(ablation.select_view_indices(MATRIX, 2, "min_visibility"), [5, 7])
        for strategy in ("max_visibility", "min_visibility"):
            order = ablation.select_view_indices(MATRIX, 8, strategy)
            for step, selected in enumerate(order):
                available = set(range(8)) - set(order[:step])
                scores = {i: np.max(MATRIX[order[:step] + [i]], axis=0).sum() for i in available}
                best = min(available, key=lambda i: (
                    -scores[i] if strategy == "max_visibility" else scores[i], i))
                self.assertEqual(selected, best)

    def test_union_score_and_zero_empty_set(self):
        self.assertEqual(ablation.visibility_score(MATRIX, []), {"sum": 0., "mean": 0., "atom_count": 3})
        score = ablation.visibility_score(MATRIX, [0, 2])
        self.assertEqual(score["sum"], 2.)
        self.assertAlmostEqual(score["mean"], 2 / 3)
        with self.assertRaises(ValueError):
            ablation.visibility_score(MATRIX, [True])

    def test_greedy_and_canonical_nested_common_endpoint(self):
        for strategy in ablation.STRATEGIES:
            full = ablation.select_view_indices(MATRIX, 8, strategy, seed=71)
            self.assertEqual(sorted(full), list(range(8)))
            if strategy != "random":
                for k in (1, 2, 4):
                    self.assertEqual(ablation.select_view_indices(MATRIX, k, strategy, seed=71), full[:k])
            self.assertEqual(ablation.visibility_score(MATRIX, full)["sum"], 3.)
        self.assertEqual(ablation.select_view_indices(MATRIX, 4, "canonical"), [0, 1, 2, 3])

    def test_random_samples_each_k_independently_and_reproducibly(self):
        with patch("ablation.np.random.default_rng", wraps=np.random.default_rng) as rng:
            samples = {k: ablation.select_view_indices(MATRIX, k, "random", seed=71)
                       for k in (1, 2, 4, 8)}
            draw_seeds = [call.args[0] for call in rng.call_args_list]
        self.assertEqual(len(set(draw_seeds)), 4)
        for k, sample in samples.items():
            self.assertEqual(len(set(sample)), k)
            self.assertEqual(sample, ablation.select_view_indices(MATRIX, k, "random", seed=71))
        # This fixed seed deliberately checks the absence of a shared prefix;
        # independent draws may still happen to overlap or nest for other seeds.
        self.assertNotEqual(samples[2], samples[4][:2])
        self.assertEqual(sorted(samples[8]), list(range(8)))

    def test_rejects_bad_visibility(self):
        for matrix in (np.zeros((7, 2)), np.empty((8, 0)), np.full((8, 2), np.nan),
                       np.full((8, 2), -0.1), np.full((8, 2), 1.1)):
            with self.assertRaises(ValueError):
                ablation.select_view_indices(matrix, 1, "random")

    def test_diverse_cohort_ignores_answers(self):
        questions, _ = fixture()
        pool = []
        for q in questions:
            for i in range(6):
                item = copy.deepcopy(q)
                item["id"] += str(i)
                item["meta"]["base_structure"] = "base-" + str(i // 2)
                pool.append(AnswersForbidden(item))
        chosen = ablation.select_questions(pool, per_type=3, seed=17)
        self.assertEqual(len(chosen), 9)
        for task in ablation.TASKS:
            self.assertEqual(len({q["meta"]["base_structure"] for q in chosen
                                  if ablation.task_name(q) == task}), 3)
        self.assertEqual([q["id"] for q in chosen[:6]],
                         [q["id"] for q in ablation.select_questions(pool, 3, 17, limit=6)])

    def test_task_atoms_use_geometry_not_answers_or_coordination_number(self):
        questions, scenes = fixture()
        expected = [list(range(1, 7)), list(range(7)), [0, 1, 2]]
        for question, indices in zip(questions, expected):
            question = AnswersForbidden(question)
            question["meta"] = AnswersForbidden(question["meta"])
            self.assertEqual(ablation.relevant_atom_indices(question, scenes["fixture"]), indices)
        altered = copy.deepcopy(questions[1])
        altered["meta"]["neighbor_display_indices"] = [1]
        with self.assertRaisesRegex(ValueError, "neighbor metadata"):
            ablation.relevant_atom_indices(altered, scenes["fixture"])
        bad_scene = copy.deepcopy(scenes["fixture"])
        bad_scene["views"][0]["metadata"]["bonds"] = []
        with self.assertRaisesRegex(ValueError, "bond graph"):
            ablation.relevant_atom_indices(questions[1], bad_scene)

    def test_plan_deduplicates_and_neutralizes_fixed_image_count(self):
        questions, scenes = fixture()
        plan = ablation.build_ablation_plan(questions, scenes)
        self.assertEqual(len(plan["variants"]), 48)
        self.assertLess(len(plan["requests"]), 48)
        for question in questions:
            final = [v for v in plan["variants"] if v["base_question_id"] == question["id"] and v["k"] == 8]
            self.assertEqual(len(final), 4)
            self.assertEqual(len({v["request_question_id"] for v in final}), 1)
            self.assertEqual(sum(v["api_request_reused"] for v in final), 3)
            self.assertEqual(len({tuple(v["selected_view_ids"]) for v in final}), 1)
        self.assertTrue(all("three images" not in q["question"].lower() for q in plan["requests"]))
        self.assertTrue(all("strategy" not in q["meta"]["ablation"] for q in plan["requests"]))
        changed = copy.deepcopy(questions)
        for q in changed:
            q["answer"], q["answer_text"] = "B", "two"
        self.assertEqual(plan["variants"], ablation.build_ablation_plan(changed, scenes)["variants"])

    def test_expansion_marks_one_charge_per_shared_response(self):
        questions, scenes = fixture()
        plan = ablation.build_ablation_plan(questions, scenes)
        responses = [{"question_id": q["id"], "condition": "image", "meta": {},
                      "attempts": 2, "usage": {"total_tokens": 30},
                      "correct": True, "raw_response": "Answer: A"}
                     for q in plan["requests"]]
        rows = ablation.expand_results(plan, responses)
        self.assertEqual(sum(r["charged_attempts"] for r in rows), 2 * len(responses))
        self.assertEqual(sum((r["charged_usage"] or {}).get("total_tokens", 0) for r in rows),
                         30 * len(responses))


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / "results" / "ablation_test.jsonl"
        self.questions, self.scenes = fixture()
        self.config = {"alias": "test", "model": "offline", "provider": "offline"}
        for i in range(8):
            Image.new("RGB", (16, 16), (i * 30, 20, 40)).save(self.root / f"v{i:02d}.png")

    def tearDown(self):
        self.temp.cleanup()

    def run_offline(self, **kwargs):
        with patch("evaluate._http_post", side_effect=AssertionError("No network allowed")):
            return asyncio.run(ablation.run_ablation(
                self.questions, self.scenes, self.config, self.output, data_dir=self.root,
                per_type=1, dry_run=True, **kwargs))

    def test_resume_then_expand_limit_without_repaying_shared_inputs(self):
        first, summary = self.run_offline(limit=1)
        self.assertEqual(len(first), 16)
        cache = Path(summary["request_cache"])
        self.assertEqual(cache.parent.name, ".ablation_cache")
        before = cache.read_bytes()
        repeated, _ = self.run_offline(limit=1)
        self.assertEqual(first, repeated)
        self.assertEqual(cache.read_bytes(), before)
        rows, summary = self.run_offline(limit=3)
        self.assertEqual(len(rows), 48)
        self.assertEqual(summary["by_task"], {"Q1a": 1, "Q2hard": 1, "Q3": 1})
        self.assertLess(summary["unique_evaluations"], 48)
        self.assertEqual(Counter(row["status"] for row in rows), {"ok": 48})
        self.assertTrue(cache.read_bytes().startswith(before))
        for q in self.questions:
            shared = [r for r in rows if r["base_question_id"] == q["id"] and r["k"] == 8]
            for field in ("prediction", "correct", "raw_response", "visibility_score_mean"):
                self.assertEqual(len({json.dumps(r[field]) for r in shared}), 1)
            self.assertEqual(sum(not r["api_request_reused"] for r in shared), 1)
            self.assertTrue(all(r["charged_attempts"] == 0 for r in shared))
        self.assertEqual(len({r["question_id"] for r in rows}), 48)
        original_hash = self.output.read_bytes()
        self.questions[0]["question"] += " New prompt."
        with self.assertRaisesRegex(ValueError, "different dataset/configuration"):
            self.run_offline()
        self.assertEqual(self.output.read_bytes(), original_hash)


if __name__ == "__main__":
    unittest.main()
