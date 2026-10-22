"""Offline independent checks of pilot balance and empirical cost calibration."""

from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest

from evaluate import build_prompt
from pilot import STRATA, calibrate_overhead, report, select_pilot, stratum


def fixture_questions():
    rows = []
    for group in STRATA:
        qtype, _, difficulty = group.partition('/')
        for index in range(25):
            images = 6 if qtype == 'Q5' else 1 if qtype == 'Q6' else 3
            text_applicable = qtype not in {'Q1a', 'Q6'}
            meta = dict(base_structure=f'base-{index}', text_applicable=text_applicable,
                        paired_count_id=f'paired-{index}', depth_trap=bool(index % 2))
            if qtype == 'Q5':
                meta['negative_type'] = (None, 'swap_elements', 'delete_atom', 'stretch_axis', 'polymorph')[index % 5]
                meta['view_groups'] = [dict(image_indices=[0, 1, 2]), dict(image_indices=[3, 4, 5])]
            answer_text = f'system-{index % 6}' if qtype == 'Q4' else 'correct choice'
            rows.append(dict(id=f'{group}-{index}', qtype=qtype, difficulty=difficulty or 'standard',
                scene_id=f'scene-{group}-{index}', structure='INTERNAL_ONLY',
                image_paths=[f'image-{i}.png' for i in range(images)],
                view_ids=[f'v{i:02d}' for i in range(images)],
                question='Choose the correct alternative from the supplied evidence.',
                options=dict(A=answer_text, B='other choice B', C='other choice C', D='other choice D'),
                answer='A', answer_text=answer_text, meta=meta,
                text_repr=('Lattice 5 5 5 90 90 90\nNa 0 0 0\nCl 0.5 0.5 0.5' if text_applicable else '')))
    return rows


def fixture_results(questions):
    rows = []
    for q in select_pilot(questions):
        # A known gateway-like constant plus per-image cost and estimated text.
        tokens = 4000 + 300 * len(q['image_paths']) + round(len(build_prompt(q, 'image')) / 3)
        output = 20 + STRATA.index(stratum(q))
        rows.append(dict(question_id=q['id'], condition='image', model='test-alias', model_name='test-model',
            response_model='test-model', provider='test-provider', config_fingerprint='one-config',
            dataset_fingerprint='one-dataset', dry_run=False, baseline=None,
            qtype=q['qtype'], difficulty=q['difficulty'], image_paths=q['image_paths'],
            status='ok', correct=True, invalid=False, parse_method='final_answer_line',
            latency_seconds=2.5, attempts=1, raw_response='Answer: A',
            usage=dict(prompt_tokens=tokens, completion_tokens=output, total_tokens=tokens+output,
                       prompt_tokens_details=dict(cached_tokens=1000),
                       completion_tokens_details=dict(reasoning_tokens=output-2))))
    return rows


class CalibrationTests(unittest.TestCase):
    def test_recovers_known_affine_image_cost(self):
        observations = [dict(prompt_tokens=4000+300*n+200, prompt_characters=600, image_count=n)
                        for n in (1, 1, 3, 3, 6, 6)]
        fit = calibrate_overhead(observations)
        self.assertTrue(fit['identifiable'])
        self.assertAlmostEqual(fit['overhead_tokens'], 4000)
        self.assertAlmostEqual(fit['per_image_tokens'], 300)
        self.assertLess(fit['rmse_tokens'], 1e-8)

    def test_nonnegative_boundary_is_refit_not_clipped(self):
        # OLS is -10+10*x; constrained optimum is 0+4*x, not 0+10*x.
        observations = [dict(prompt_tokens=t, prompt_characters=0, image_count=x)
                        for x, t in ((1, 0), (2, 10))]
        fit = calibrate_overhead(observations)
        self.assertAlmostEqual(fit['overhead_tokens'], 0)
        self.assertAlmostEqual(fit['per_image_tokens'], 4)
        # Negative per-image OLS instead requires refitting the intercept.
        observations[0]['prompt_tokens'], observations[1]['prompt_tokens'] = 10, 0
        fit = calibrate_overhead(observations)
        self.assertAlmostEqual(fit['overhead_tokens'], 5)
        self.assertAlmostEqual(fit['per_image_tokens'], 0)

    def test_unidentified_design_uses_explicit_conservative_fallback(self):
        fit = calibrate_overhead([dict(prompt_tokens=5000, prompt_characters=600, image_count=3)])
        self.assertFalse(fit['identifiable'])
        self.assertEqual(fit['overhead_tokens'], 5000)
        self.assertIn('fallback', fit['method'])

    def test_rejects_invalid_calibration_numbers(self):
        observation = dict(prompt_tokens=100, prompt_characters=20, image_count=1)
        for value in (-1, float('nan'), float('inf'), True, 0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibrate_overhead([{**observation, 'prompt_tokens': value}])
        for ratio in (0, -1, float('nan'), True):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                calibrate_overhead([observation], ratio)


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.questions = fixture_questions()
        self.results = fixture_results(self.questions)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.mini = self.root / 'mini.jsonl'
        self.mini.write_text(''.join(json.dumps(q)+'\n' for q in self.questions), 'utf-8')

    def run_report(self, rows=None, **kwargs):
        source = self.root / 'results.jsonl'
        source.write_text(''.join(json.dumps(r)+'\n' for r in (self.results if rows is None else rows)), 'utf-8')
        return report(source, self.mini, output_dir=self.root / 'report', **kwargs)

    def test_selection_is_balanced_paired_and_reproducible(self):
        first = select_pilot(self.questions, seed=77)
        self.assertEqual(first, select_pilot(list(reversed(self.questions)), seed=77))
        self.assertEqual(Counter(stratum(q) for q in first), {s: 3 for s in STRATA})
        self.assertEqual(len({q['id'] for q in first}), 24)
        paired = [{q['meta']['paired_count_id'] for q in first if q['qtype'] == kind} for kind in ('Q1a', 'Q1b')]
        self.assertEqual(*paired)
        self.assertEqual({q['meta']['depth_trap'] for q in first if q['qtype'] == 'Q3'}, {False, True})
        q5 = [q['meta']['negative_type'] for q in first if q['qtype'] == 'Q5']
        self.assertIn(None, q5)
        self.assertEqual(len(set(q5)), 3)
        self.assertEqual(sum(len(q['image_paths']) for q in first), 75)

    def test_stratified_costs_call_counts_and_shared_overhead(self):
        result = self.run_report()
        self.assertEqual(result['records'], 24)
        self.assertEqual(result['usage_records'], 24)
        self.assertEqual(result['observed_cached_input_tokens'], 24_000)
        expected_prompt = sum(r['usage']['prompt_tokens'] for r in self.results)
        expected_output = sum(r['usage']['completion_tokens'] for r in self.results)
        self.assertEqual(result['observed_prompt_tokens'], expected_prompt)
        self.assertEqual(result['observed_completion_tokens'], expected_output)
        self.assertAlmostEqual(result['observed_usage_reference_cost_usd'], (expected_prompt*2+expected_output*10)/1e6)
        estimates = {r['condition']: r for r in result['mini_estimates']}
        self.assertEqual({k: r['requests'] for k,r in estimates.items()}, {'image': 200, 'text': 150, 'image+text': 200})
        self.assertAlmostEqual(estimates['image']['estimated_usd'], (expected_prompt*2+expected_output*10)/1e6*25/3)
        self.assertAlmostEqual(result['overhead_calibration']['3']['overhead_tokens'], 4000, delta=1)
        self.assertGreater(estimates['text']['estimated_input_tokens'], 4000*150)
        self.assertGreater(estimates['image+text']['estimated_input_tokens'], estimates['image']['estimated_input_tokens'])
        for estimate in estimates.values():
            low, high = estimate['text_tokenization_sensitivity_usd']
            self.assertLessEqual(low, estimate['estimated_usd'])
            self.assertGreaterEqual(high, estimate['estimated_usd'])
            self.assertGreater(estimate['conservative_planning_usd'], estimate['estimated_usd'])
        self.assertEqual(result['mean_latency_seconds'], 2.5)
        self.assertEqual(result['price']['account_group_multiplier'], None)
        self.assertFalse(result['price']['cache_discount_applied'])

    def test_missing_usage_is_counted_without_zero_imputation(self):
        self.results[0]['usage'] = None
        result = self.run_report()
        self.assertEqual(result['missing_usage_records'], 1)
        self.assertEqual(result['usage_records'], 23)
        self.assertEqual(result['usage_coverage_by_stratum']['Q1a']['usage_records'], 2)
        for row in self.results[:3]:
            row['usage'] = None
        with self.assertRaisesRegex(ValueError, 'every pilot stratum'):
            self.run_report()

    def test_rejects_invalid_prices_usage_and_latency(self):
        for key in ('input_rate', 'output_rate', 'max_output_tokens'):
            for value in (-1, float('nan'), float('inf'), True):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.run_report(**{key: value})
        for value in (-1, float('nan'), float('inf'), True, 1.5):
            rows = copy.deepcopy(self.results)
            rows[0]['usage']['prompt_tokens'] = value
            with self.subTest(tokens=value), self.assertRaises(ValueError):
                self.run_report(rows)
        rows = copy.deepcopy(self.results)
        rows[0]['latency_seconds'] = float('nan')
        with self.assertRaises(ValueError):
            self.run_report(rows)

    def test_rejects_mixed_or_duplicated_runs(self):
        for key in ('model', 'model_name', 'provider', 'config_fingerprint', 'dataset_fingerprint'):
            rows = copy.deepcopy(self.results)
            rows[0][key] += '-different'
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'one model'):
                self.run_report(rows)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_report(self.results + [self.results[0]])
        rows = copy.deepcopy(self.results)
        rows[0]['dry_run'] = True
        with self.assertRaisesRegex(ValueError, 'real model'):
            self.run_report(rows)

    def test_planning_output_never_below_observed_output(self):
        row = self.results[0]
        row['usage']['completion_tokens'] = 2048
        row['usage']['total_tokens'] = row['usage']['prompt_tokens'] + 2048
        result = self.run_report()
        self.assertEqual(result['planning_output_tokens_per_request'], 2048)
        for estimate in result['mini_estimates']:
            self.assertEqual(estimate['planning_output_tokens'], estimate['requests']*2048)


if __name__ == '__main__':
    unittest.main()
