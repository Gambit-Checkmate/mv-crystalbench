"""Independent mathematical and cross-module ablation checks, entirely offline."""

import copy
import unittest

import numpy as np

from ablation import (STRATEGIES, build_ablation_plan, expand_results,
                      select_view_indices, visibility_score)
from ablation_plots import build_plot_tables
from evaluate import build_prompt
from test_ablation import fixture


class IndependentSelectorReview(unittest.TestCase):
    def test_greedy_choices_against_direct_set_enumeration(self):
        # Quantized matrices deliberately create exact ties. The oracle
        # recomputes each candidate's complete union, without incremental state.
        rng = np.random.default_rng(604)
        for atom_count in (1, 5, 11):
            for _ in range(20):
                matrix = rng.integers(0, 9, size=(8, atom_count)) / 8
                for strategy in ('max_visibility', 'min_visibility'):
                    order = select_view_indices(matrix, 8, strategy)
                    for step, chosen in enumerate(order):
                        available = sorted(set(range(8)) - set(order[:step]))
                        scored = [(float(np.max(matrix[order[:step]+[j]], axis=0).sum()), j)
                                  for j in available]
                        target = (min(scored, key=lambda item: (-item[0], item[1]))
                                  if strategy == 'max_visibility' else min(scored))
                        self.assertEqual(chosen, target[1])
                    full_score = visibility_score(matrix, order)
                    self.assertAlmostEqual(full_score['sum'], float(np.max(matrix, axis=0).sum()))
                    self.assertAlmostEqual(full_score['mean'], full_score['sum']/atom_count)

    def test_random_is_deterministic_per_k_without_forced_nesting(self):
        matrix = np.ones((8, 3))
        nested = 0
        for seed in range(100):
            one = select_view_indices(matrix, 1, 'random', seed)
            two = select_view_indices(matrix, 2, 'random', seed)
            self.assertEqual(two, select_view_indices(matrix*0.1, 2, 'random', seed))
            self.assertEqual(len(set(two)), 2)
            nested += set(one).issubset(two)
            self.assertEqual(sorted(select_view_indices(matrix, 8, 'random', seed)), list(range(8)))
        self.assertGreater(nested, 0)
        self.assertLess(nested, 80)

    def test_answer_mutations_do_not_change_plan_or_model_prompt(self):
        questions, scenes = fixture()
        original = build_ablation_plan(questions, scenes)
        mutated = copy.deepcopy(questions)
        for q in mutated:
            q['answer'], q['answer_text'] = 'D', 'deliberately invalid oracle value'
            q['meta']['coordination_number'] = 999
            q['meta']['depth_trap'] = 'not a geometric selection input'
        modified = build_ablation_plan(mutated, scenes)
        self.assertEqual(original['variants'], modified['variants'])
        for before, after in zip(original['requests'], modified['requests']):
            self.assertEqual(build_prompt(before, 'image'), build_prompt(after, 'image'))
            prompt = build_prompt(before, 'image')
            self.assertNotIn('SECRET', prompt)
            self.assertNotIn('visibility_score', prompt)
            self.assertNotIn('max_visibility', prompt)
            self.assertNotIn('three images', prompt.lower())
            self.assertEqual(len(before['image_paths']), before['meta']['ablation']['k'])

    def test_selection_response_reuse_and_bootstrap_integrate(self):
        templates, scenes = fixture()
        questions = []
        for base in range(3):
            for original in templates:
                q = copy.deepcopy(original)
                q['id'] += f'-base{base}'
                q['meta']['base_structure'] = f'base-{base}'
                questions.append(q)
        plan = build_ablation_plan(questions, scenes)
        responses = []
        for q in plan['requests']:
            correct = q['meta']['base_structure'] != 'base-0'
            responses.append(dict(question_id=q['id'], condition='image', meta=q['meta'],
                qtype=q['qtype'], difficulty=q['difficulty'], model='offline-review',
                provider='offline', dry_run=True, status='ok', correct=correct, invalid=False,
                prediction='A' if correct else 'B', raw_response='Answer: A' if correct else 'Answer: B',
                attempts=1, usage=dict(prompt_tokens=10, completion_tokens=1, total_tokens=11)))
        expanded = expand_results(plan, responses)
        curves, bins, info = build_plot_tables(expanded, bootstrap=1000, seed=333)
        self.assertEqual(len(expanded), 9*16)
        self.assertEqual(sum(r['charged_attempts'] for r in expanded), len(responses))
        self.assertEqual(sum((r['charged_usage'] or {}).get('total_tokens', 0) for r in expanded), 11*len(responses))
        self.assertEqual(info['visibility_observations_after'], 9*13)
        self.assertEqual(info['k8_removed_duplicates'], 9*3)
        self.assertEqual(sum(row['n'] for row in bins), 9*13)
        for curve in curves:
            self.assertEqual((curve['n'], curve['n_structures']), (3, 3))
            self.assertAlmostEqual(curve['accuracy'], 2/3)
            # Finite bootstrap draws can put the lower 2.5% quantile on either
            # side of the 1/27 all-incorrect cluster draw; interpolation is valid.
            self.assertGreaterEqual(curve['ci_low'], 0)
            self.assertLessEqual(curve['ci_low'], 1/3)
            self.assertEqual(curve['ci_high'], 1)
        for panel in ('Q1a', 'Q2hard', 'Q3'):
            common = [row for row in curves if row['panel'] == panel and row['k'] == 8]
            self.assertEqual(len({(row['accuracy'], row['ci_low'], row['ci_high']) for row in common}), 1)
        for q in questions:
            endpoint = [r for r in expanded if r['base_question_id'] == q['id'] and r['k'] == 8]
            self.assertEqual(len(endpoint), len(STRATEGIES))
            for field in ('request_question_id', 'prediction', 'raw_response', 'correct', 'visibility_score_mean'):
                self.assertEqual(len({row[field] for row in endpoint}), 1)


if __name__ == '__main__':
    unittest.main()
