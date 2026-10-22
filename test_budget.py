"""Concurrent budget, retry and crash-recovery tests; all model calls are mocked."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from budget import BudgetExceeded, BudgetTracker, estimate_reservation
import evaluate
from test_evaluate import question


def estimate(amount=1.):
    return dict(reserved_usd=amount, input_tokens=1000, output_tokens=100,
                input_per_million=1000., output_per_million=0.)


def response(prompt=100, completion=5):
    return dict(model='offline', choices=[dict(message=dict(content='Answer: B'))],
                usage=dict(prompt_tokens=prompt, completion_tokens=completion,
                           total_tokens=prompt+completion))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_atomic_shared_reservation_under_eight_concurrent_workers(self):
        ledger = self.root/'shared.json'
        trackers = [BudgetTracker(ledger, 8) for _ in range(2)]
        def reserve(index):
            try:
                return trackers[index % 2].reserve(f'request-{index}', estimate())
            except BudgetExceeded:
                return None
        with ThreadPoolExecutor(max_workers=8) as pool:
            receipts = list(pool.map(reserve, range(32)))
        self.assertEqual(sum(receipt is not None for receipt in receipts), 8)
        self.assertEqual(trackers[0].snapshot()['committed_usd'], 8)
        self.assertTrue(trackers[1].snapshot()['stopped'])
        self.assertEqual(len(json.loads(ledger.read_text())['entries']), 8)

    def test_settlement_retention_and_reopening_are_idempotent(self):
        tracker = BudgetTracker(self.root/'ledger.json', 10)
        first = tracker.reserve('r1', estimate())
        tracker.settle(first, response=response(prompt=400))
        tracker.settle(first, response=response(prompt=900))
        self.assertAlmostEqual(tracker.snapshot()['committed_usd'], .4)
        second = tracker.reserve('r2', estimate())
        tracker.settle(second, failed=True)
        third = tracker.reserve('r3', estimate())
        tracker.settle(third, response={'choices':[]})
        fourth = tracker.reserve('interrupted', estimate())
        reopened = BudgetTracker(self.root/'ledger.json', 10)
        self.assertAlmostEqual(reopened.snapshot()['committed_usd'], 3.4)
        self.assertEqual(reopened.request_entries('interrupted')[0]['status'], 'reserved')
        self.assertEqual(reopened.request_entries('r3')[0]['status'], 'missing_usage_retained')
        with self.assertRaises(ValueError):
            BudgetTracker(self.root/'ledger.json', 11)

    def test_unexpected_usage_overrun_is_recorded_and_stops_gate(self):
        tracker = BudgetTracker(self.root/'ledger.json', 2)
        entry = tracker.reserve('r', estimate())
        record = tracker.settle(entry, response=response(prompt=3000))
        self.assertEqual(record['reservation_overrun_usd'], 2)
        self.assertEqual(tracker.snapshot()['committed_usd'], 3)
        self.assertTrue(tracker.snapshot()['stopped'])
        with self.assertRaises(BudgetExceeded):
            tracker.reserve('next', estimate())

    def test_estimate_retains_all_original_montage_images_and_validates_rates(self):
        config = dict(price_input_per_million=2, price_output_per_million=10, max_tokens=100)
        messages = [dict(content=[dict(type='text', text='abc'), dict(type='image_url', image_url={'url':'unused'})])]
        result = estimate_reservation(messages, config, image_count=8)
        self.assertEqual(result['input_tokens'], 3+8192+8*4096)
        self.assertAlmostEqual(result['reserved_usd'], (result['input_tokens']*2+100*10)/1e6)
        for invalid in (-1, float('nan'), float('inf'), True, None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                estimate_reservation(messages, {**config, 'price_input_per_million':invalid})

    def test_overrun_stop_reason_is_preserved_and_audited_ack_keeps_charges(self):
        tracker = BudgetTracker(self.root/'ledger.json', 10)
        entry = tracker.reserve('overrun', estimate())
        tracker.settle(entry, response=response(prompt=1500))
        reason = tracker.snapshot()['stop_reason']
        with self.assertRaises(BudgetExceeded):
            tracker.reserve('queued', estimate())
        self.assertEqual(tracker.snapshot()['stop_reason'],reason)
        before = tracker.snapshot()['committed_usd']
        audit = tracker.acknowledge_reservation_overrun('Raised input allowance to 200000 and output allowance to 16384.')
        self.assertTrue(audit['reopened'])
        self.assertEqual(tracker.snapshot()['committed_usd'],before)
        self.assertEqual(tracker.snapshot()['limit_usd'],10)
        self.assertFalse(tracker.snapshot()['stopped'])
        self.assertEqual(len(json.loads(tracker.path.read_text())['audit_events']),1)

    def test_ack_refuses_unsettled_budget_limit_and_requires_explicit_legacy_opt_in(self):
        tracker = BudgetTracker(self.root/'ledger.json', 10)
        pending = tracker.reserve('pending', estimate())
        entry = tracker.reserve('overrun', estimate())
        tracker.settle(entry, response=response(prompt=1500))
        with self.assertRaisesRegex(ValueError,'unsettled'):
            tracker.acknowledge_reservation_overrun('Corrected bounds.')
        tracker.settle(pending, failed=True)
        data = json.loads(tracker.path.read_text())
        data.pop('stop_category')
        data['stop_reason'] = 'Next attempt reservation would exceed the shared budget.'
        tracker.path.write_text(json.dumps(data), 'utf-8')
        with self.assertRaisesRegex(ValueError,'Only a reservation'):
            tracker.acknowledge_reservation_overrun('Corrected bounds.')
        audit = tracker.acknowledge_reservation_overrun('Corrected bounds; verified legacy overwrite.',
                                                       allow_legacy_reason_recovery=True)
        self.assertTrue(audit['legacy_reason_recovered'])
        with self.assertRaises(BudgetExceeded):
            tracker.reserve('too-expensive', estimate(20))
        with self.assertRaises(ValueError):
            tracker.acknowledge_reservation_overrun('Cannot bypass budget.', allow_legacy_reason_recovery=True)

    def test_output_reservation_allowance_does_not_change_requested_cap(self):
        config = dict(price_input_per_million=2,price_output_per_million=10,
                      max_tokens=1024,budget_output_tokens_allowance=16384)
        reserved = estimate_reservation([dict(content='A')],config)
        self.assertEqual(reserved['output_tokens'],16384)
        self.assertEqual(config['max_tokens'],1024)

    def test_disabled_enforcement_ignores_old_stops_and_limits_but_records_cost(self):
        path = self.root/'ledger.json'
        old = BudgetTracker(path, 1)
        entry = old.reserve('old',estimate())
        old.settle(entry,response=response(prompt=3000))
        self.assertTrue(old.snapshot()['stopped'])
        unlimited = BudgetTracker(path,999,enforce=False)
        snapshot = unlimited.snapshot()
        self.assertFalse(snapshot['enforced'])
        self.assertFalse(snapshot['stopped'])
        self.assertIsNone(snapshot['limit_usd'])
        self.assertIsNone(snapshot['remaining_usd'])
        self.assertEqual(snapshot['committed_usd'],3)
        entry = unlimited.reserve('new-overrun',estimate())
        settled = unlimited.settle(entry,response=response(prompt=5000))
        self.assertEqual(settled['reservation_overrun_usd'],4)
        huge = unlimited.reserve('above-all-old-limits',estimate(2000))
        unlimited.settle(huge,failed=True)
        self.assertEqual(unlimited.snapshot()['committed_usd'],2008)
        self.assertFalse(unlimited.snapshot()['stopped'])
        events = json.loads(path.read_text())['audit_events']
        self.assertEqual(events[-1]['event'],'disable_budget_enforcement')
        self.assertEqual(events[-1]['committed_usd'],3)

    def test_disabled_enforcement_audits_and_clears_old_operator_pause(self):
        path = self.root/'ledger.json'
        tracker = BudgetTracker(path,40)
        ledger = json.loads(path.read_text())
        ledger.update(stopped=True,stop_category='operator_reconfigure',stop_reason='Drain before config change.')
        path.write_text(json.dumps(ledger),'utf-8')
        resumed = BudgetTracker(path,40,enforce=False)
        self.assertFalse(resumed.snapshot()['stopped'])
        audit = json.loads(path.read_text())['audit_events'][-1]
        self.assertEqual(audit['previous_stop_category'],'operator_reconfigure')
        self.assertEqual(audit['previous_stop_reason'],'Drain before config change.')


class EvaluationBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root/'result.jsonl'
        self.config = dict(alias='offline', model='offline', provider='mock',
                           api_key_env='NO_REAL_API_KEY_NEEDED', requires_api_key=False,
                           price_input_per_million=1., price_output_per_million=2.,
                           max_tokens=10, budget_input_overhead_tokens=0)

    def run_evaluate(self, tracker, questions=None, **kwargs):
        return asyncio.run(evaluate.evaluate_questions(
            questions or [question()], self.config, self.output, data_dir=self.root,
            conditions=('no_image',), budget=tracker, **kwargs))

    def reserve_amount(self):
        return estimate_reservation(evaluate.build_messages(question(), 'no_image', self.root, self.config),
                                    self.config)['reserved_usd']

    def test_retry_cannot_bypass_gate_and_attempts_keep_unknown_cost(self):
        amount = self.reserve_amount()
        tracker = BudgetTracker(self.root/'ledger.json', amount*2.1)
        with patch.object(evaluate, '_http_post', side_effect=evaluate.APIError('retry', retryable=True)) as post:
            rows = self.run_evaluate(tracker, [question('first'),question('second')],
                                     concurrency=1, max_retries=5, backoff_base=0)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['attempts'], 2)
        self.assertTrue(rows[0]['budget_exhausted'])
        self.assertAlmostEqual(rows[0]['cost_usd'], amount*2)
        self.assertAlmostEqual(tracker.snapshot()['committed_usd'], amount*2)
        self.assertEqual(len(self.output.read_text().splitlines()), 1)

    def test_concurrency_stops_dispatch_and_saves_completed_responses(self):
        amount = self.reserve_amount()
        tracker = BudgetTracker(self.root/'ledger.json', amount*2.1)
        def slow_response(*args):
            time.sleep(.04)
            return response()
        questions = [question(str(index)) for index in range(16)]
        with patch.object(evaluate, '_http_post', side_effect=slow_response) as post:
            rows = self.run_evaluate(tracker, questions, concurrency=8)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['status'] == 'ok' for row in rows))
        self.assertAlmostEqual(sum(row['cost_usd'] for row in rows), tracker.snapshot()['committed_usd'])
        with patch.object(evaluate, '_http_post', side_effect=AssertionError('Resume must not call again')):
            resumed = self.run_evaluate(BudgetTracker(tracker.path, amount*2.1), questions, concurrency=8)
        self.assertEqual(rows, resumed)

    def test_crash_after_settlement_recovers_response_without_another_charge(self):
        tracker = BudgetTracker(self.root/'ledger.json', 10)
        with patch.object(evaluate, '_http_post', return_value=response()) as post:
            first = self.run_evaluate(tracker)
        self.assertEqual(post.call_count, 1)
        before = tracker.snapshot()
        # Simulate loss of the final result write after the durable settlement.
        self.output.unlink()
        with patch.object(evaluate, '_http_post', side_effect=AssertionError('Recovered response must be reused')):
            recovered = self.run_evaluate(BudgetTracker(tracker.path, 10))
        self.assertEqual(tracker.snapshot(), before)
        self.assertTrue(recovered[0]['budget_response_recovered'])
        self.assertEqual(recovered[0]['raw_response'], first[0]['raw_response'])
        self.assertEqual(recovered[0]['cost_usd'], first[0]['cost_usd'])

    def test_success_without_usage_keeps_reservation(self):
        tracker = BudgetTracker(self.root/'ledger.json', 10)
        missing = response()
        missing.pop('usage')
        with patch.object(evaluate, '_http_post', return_value=missing):
            rows = self.run_evaluate(tracker)
        self.assertAlmostEqual(rows[0]['cost_usd'], self.reserve_amount())
        self.assertEqual(tracker.snapshot()['attempts'], 1)


if __name__ == '__main__':
    unittest.main()
