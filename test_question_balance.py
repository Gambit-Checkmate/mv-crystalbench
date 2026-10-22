"""Regression tests for English prompts, source caps, and the 200-item subset."""

from collections import Counter
import unittest

from question_utils import make_draft, model_prompt
from questions import (
    CRYSTAL_SYSTEMS, Q5_OPTIONS, finalize_questions, make_mini,
    preflight_catalog, q4_plan,
)
from structures import get_structures


class QuestionBalanceTests(unittest.TestCase):
    def test_draft_keeps_internal_source_out_of_the_prompt(self):
        scene = {
            "scene_id": "opaque", "text_repr": "Lattice parameters: a=5.0",
            "views": [{"view_id": f"v{i:02d}", "image_path": f"opaque/v{i:02d}.png"}
                      for i in range(8)],
        }
        draft = make_draft(
            "Q1b", "reasoning", "INTERNAL_SECRET", scene, "How many Na atoms?",
            ["1", "2", "3", "4"], "4", {},
        )
        self.assertEqual(draft["meta"]["base_structure"], "INTERNAL_SECRET")
        item = finalize_questions([draft], 7)[0]
        prompt = model_prompt(item, include_text=True)
        self.assertNotIn("INTERNAL_SECRET", prompt)
        self.assertIn("Answer with exactly one option letter", prompt)
        self.assertTrue(prompt.isascii())

    def test_real_catalog_balanced_q4_capacity(self):
        structures = get_structures()
        target = 100 if len(structures) >= 40 else 4
        plan, pools, total = q4_plan(structures, target)
        self.assertEqual(len(set(plan.values())), 1)
        self.assertEqual(sum(plan.values()), total)
        for kind, quota in plan.items():
            self.assertLessEqual(quota, 4 * len(pools[kind]))
        report = preflight_catalog(structures, target)
        self.assertEqual(report["base_structure_count"], len(structures))
        if len(structures) >= 40:
            self.assertGreaterEqual(len(plan), 6)
            self.assertGreaterEqual(total, 80)
            self.assertEqual(total, 96)

    @staticmethod
    def fixture_questions():
        rows = []
        q4_names = [name for name in CRYSTAL_SYSTEMS if name != "triclinic"]
        for qtype in ("Q1a", "Q1b", "Q2", "Q3", "Q4", "Q5", "Q6"):
            count = 120 if qtype == "Q2" else 96 if qtype == "Q4" else 100
            for index in range(count):
                base = f"base-{index % 48:02d}"
                meta = {"base_structure": base, "paired_count_id": f"count-{index}"}
                answer_text = str(index % 4)
                difficulty = "medium"
                if qtype == "Q2":
                    difficulty = "easy" if index < 60 else "hard"
                if qtype == "Q3":
                    meta["depth_trap"] = index % 2 == 0
                    answer_text = "B is closer" if (index // 2) % 2 == 0 else "C is closer"
                if qtype == "Q4":
                    answer_text = q4_names[index % 6]
                if qtype == "Q5":
                    kind = None if index < 50 else (
                        "swap_elements", "delete_atom", "stretch_axis", "polymorph"
                    )[(index - 50) % 4]
                    meta.update(is_same_structure=kind is None, negative_type=kind)
                    answer_text = Q5_OPTIONS[
                        0 if kind is None else 1 if kind == "swap_elements"
                        else 3 if kind == "stretch_axis" else 2
                    ]
                if qtype == "Q6":
                    answer_text = ("<100>", "<110>", "<111>", "<112>")[index % 4]
                rows.append({
                    "id": f"{qtype}-{index:03d}", "qtype": qtype, "difficulty": difficulty,
                    "meta": meta, "answer": "ABCD"[index % 4], "answer_text": answer_text,
                })
        return rows

    def test_mini_quotas_pairing_diversity_and_reproducibility(self):
        full = self.fixture_questions()
        mini = make_mini(full, seed=73)
        self.assertEqual(mini, make_mini(full, seed=73))
        self.assertEqual(len(mini), 200)
        self.assertEqual(len({q["id"] for q in mini}), 200)
        self.assertEqual(Counter(q["qtype"] for q in mini), {
            "Q1a": 25, "Q1b": 25, "Q2": 50, "Q3": 25, "Q4": 25, "Q5": 25, "Q6": 25,
        })
        self.assertEqual(Counter(q["difficulty"] for q in mini if q["qtype"] == "Q2"),
                         {"easy": 25, "hard": 25})
        self.assertEqual(
            {q["meta"]["paired_count_id"] for q in mini if q["qtype"] == "Q1a"},
            {q["meta"]["paired_count_id"] for q in mini if q["qtype"] == "Q1b"},
        )
        self.assertEqual(sum(q["meta"]["is_same_structure"] for q in mini if q["qtype"] == "Q5"), 13)
        q3 = [q for q in mini if q["qtype"] == "Q3"]
        self.assertIn(sum(q["meta"]["depth_trap"] for q in q3), (12, 13))
        self.assertEqual(sorted(Counter(q["answer_text"] for q in q3).values()), [12, 13])
        for qtype in ("Q1a", "Q2", "Q3", "Q5", "Q6"):
            self.assertGreaterEqual(
                len({q["meta"]["base_structure"] for q in mini if q["qtype"] == qtype}), 20,
            )


if __name__ == "__main__":
    unittest.main()
