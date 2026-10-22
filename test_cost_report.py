"""Cost reports use read-only ledger snapshots and never call provider APIs."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cost_report import build_cost_report, format_cost_report, read_ledger_report


def entry(identifier, status="settled", charged=.11, usage=None, request="request", model="gpt"):
    return dict(id=identifier, request_key=request, model=model, status=status,
                reserved_usd=1., charged_usd=charged,
                usage={"prompt_tokens": 100, "completion_tokens": 5} if usage is None else usage,
                input_per_million=1000., output_per_million=2000.,
                response={"choices": [{"message": {"content": "PRIVATE_RESPONSE_NOT_FOR_REPORT"}}]})


def write_ledger(path, entries):
    value = dict(schema_version=1, limit_usd=40., enforced=False, stopped=False,
                 entries={row["id"]: row for row in entries},
                 audit_events=[{"committed_usd": 999999., "event": "historical_snapshot"}])
    path.write_text(json.dumps(value), encoding="utf-8")


class CostReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main = self.root / "main_budget.json"
        self.ablation = self.root / "ablation_budget.json"

    def test_usage_and_three_retained_categories_never_equal_bill(self):
        rows = [entry("first"), entry("retry", "failed_retained", 1., usage={}),
                entry("missing", "missing_usage_retained", 2., usage={}, request="missing"),
                entry("pending", "reserved", 3., usage={}, request="pending")]
        write_ledger(self.main, rows)
        write_ledger(self.ablation, [])
        before = self.main.read_bytes(), self.ablation.read_bytes()
        report = build_cost_report(self.main, self.ablation)
        totals = report["combined"]
        self.assertAlmostEqual(totals["usage_estimated_usd"], .11)
        self.assertEqual(totals["retained_allowances_usd"], 6.)
        self.assertAlmostEqual(totals["ledger_committed_usd"], 6.11)
        self.assertEqual((totals["usage_priced_attempts"], totals["unpriced_attempts"]), (1, 3))
        self.assertEqual(totals["additional_attempts"], 1)
        self.assertEqual(totals["prompt_tokens"], 100)
        self.assertEqual(before, (self.main.read_bytes(), self.ablation.read_bytes()))
        self.assertFalse(self.main.with_suffix(".json.lock").exists())
        self.assertNotIn("PRIVATE_RESPONSE", json.dumps(report))
        text = format_cost_report(report)
        self.assertIn("NOT actual billed spend", text)
        self.assertIn("Budget enforcement: disabled", text)

    def test_valid_usage_with_retained_status_has_separate_adjustment(self):
        write_ledger(self.main, [entry("failed-with-usage", "failed_retained", 1.)])
        stats = read_ledger_report(self.main)["totals"]
        self.assertAlmostEqual(stats["usage_estimated_usd"], .11)
        self.assertAlmostEqual(stats["usage_ledger_adjustment_usd"], .89)
        self.assertEqual(stats["retained_allowances_usd"], 0.)
        self.assertTrue(stats["ledger_reconciled"])
        self.assertTrue(stats["warnings"])

    def test_invalid_usage_and_unknown_status_are_unverified(self):
        write_ledger(self.main, [entry("bad-usage", charged=4., usage={"prompt_tokens": True, "completion_tokens": 1}),
                                 entry("future", "future-status", 5., usage={})])
        stats = read_ledger_report(self.main)["totals"]
        self.assertEqual(stats["unverifiable_settled_usd"], 4.)
        self.assertEqual(stats["unknown_status_retained_usd"], 5.)
        self.assertEqual(stats["usage_priced_attempts"], 0)
        self.assertEqual(stats["ledger_committed_usd"], 9.)
        self.assertEqual(len(stats["warnings"]), 2)

    def test_missing_ledger_is_not_reported_as_known_zero(self):
        write_ledger(self.main, [])
        report = build_cost_report(self.main, self.ablation)
        self.assertFalse(report["ledgers"]["ablation"]["available"])
        self.assertFalse(report["combined"]["coverage_complete"])
        self.assertIn("absent-ledger costs are unknown", format_cost_report(report))
        with self.assertRaisesRegex(ValueError, "counted twice"):
            build_cost_report(self.main, self.main)

    def test_malformed_ledger_amount_is_rejected(self):
        bad = entry("bad")
        bad["charged_usd"] = float("nan")
        write_ledger(self.main, [bad])
        with self.assertRaisesRegex(ValueError, "finite"):
            read_ledger_report(self.main)

    def test_cli_writes_report_without_changing_ledger_or_lock(self):
        write_ledger(self.main, [entry("one")])
        write_ledger(self.ablation, [])
        before = self.main.read_bytes(), self.ablation.read_bytes()
        script = Path(__file__).with_name("cost_report.py")
        command = [sys.executable, str(script), "--main-ledger", str(self.main),
                   "--ablation-ledger", str(self.ablation), "--output", str(self.root / "report.txt")]
        subprocess.run(command, check=True, capture_output=True)
        self.assertTrue((self.root / "report.json").exists())
        self.assertEqual(before, (self.main.read_bytes(), self.ablation.read_bytes()))
        self.assertFalse(list(self.root.glob("*.lock")))
        guarded = subprocess.run([*command[:-1], str(self.main)], capture_output=True)
        self.assertNotEqual(guarded.returncode, 0)
        self.assertEqual(before, (self.main.read_bytes(), self.ablation.read_bytes()))


if __name__ == "__main__":
    unittest.main()
