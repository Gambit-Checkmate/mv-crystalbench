"""Read API ledgers without mutating them; separate usage estimates and reserves.

Provider invoices are not read. A ledger's committed/charged amount combines
usage-priced attempts and retained planning allowances, so it is not a bill.
Every recorded attempt remains included, including archived smoke attempts and
retries. Evaluation JSONLs and audit-event snapshots are never added again.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


RETAINED_FIELDS = (
    "failed_retained_usd", "missing_usage_retained_usd", "outstanding_reserved_usd",
    "unverifiable_settled_usd", "unknown_status_retained_usd",
)
MONEY_FIELDS = ("usage_estimated_usd", "usage_linked_ledger_usd", "usage_ledger_adjustment_usd",
                "ledger_committed_usd", "retained_allowances_usd", *RETAINED_FIELDS)
COUNT_FIELDS = ("attempts", "usage_priced_attempts", "unpriced_attempts", "prompt_tokens",
                "completion_tokens", "unique_requests", "additional_attempts")


def _number(value, label, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < 0 or integer and int(value) != value):
        raise ValueError(f"{label} must be a finite nonnegative {'integer' if integer else 'number'}.")
    return int(value) if integer else float(value)


def _usage_price(entry):
    usage = entry.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        prompt = _number(usage.get("prompt_tokens"), "prompt_tokens", integer=True)
        completion = _number(usage.get("completion_tokens"), "completion_tokens", integer=True)
        input_rate = _number(entry.get("input_per_million"), "input_per_million")
        output_rate = _number(entry.get("output_per_million"), "output_per_million")
    except ValueError:
        return None
    return dict(prompt_tokens=prompt, completion_tokens=completion,
                estimated_usd=(prompt * input_rate + completion * output_rate) / 1_000_000)


def _aggregate_entries(entries):
    amounts = defaultdict(list)
    counts = Counter()
    statuses = Counter()
    requests = Counter()
    warnings = []
    for identifier, entry in entries:
        charged = _number(entry.get("charged_usd"), f"{identifier}: charged_usd")
        status = entry.get("status", "unknown")
        if not isinstance(status, str):
            raise ValueError(f"{identifier}: ledger status must be a string.")
        counts["attempts"] += 1
        statuses[status] += 1
        requests[str(entry.get("request_key") or identifier)] += 1
        amounts["ledger_committed_usd"].append(charged)
        priced = _usage_price(entry)
        if priced is not None:
            counts["usage_priced_attempts"] += 1
            counts["prompt_tokens"] += priced["prompt_tokens"]
            counts["completion_tokens"] += priced["completion_tokens"]
            amounts["usage_estimated_usd"].append(priced["estimated_usd"])
            amounts["usage_linked_ledger_usd"].append(charged)
            adjustment = charged - priced["estimated_usd"]
            amounts["usage_ledger_adjustment_usd"].append(adjustment)
            if not math.isclose(adjustment, 0, abs_tol=1e-9):
                warnings.append(f"{identifier}: ledger charge differs from usage-priced estimate by ${adjustment:.8f}.")
            if status != "settled":
                warnings.append(f"{identifier}: usage is priceable but ledger status is {status}; estimate and ledger adjustment remain separate.")
        else:
            counts["unpriced_attempts"] += 1
            category = {"failed_retained": "failed_retained_usd", "missing_usage_retained": "missing_usage_retained_usd",
                        "reserved": "outstanding_reserved_usd", "settled": "unverifiable_settled_usd"}.get(
                            status, "unknown_status_retained_usd")
            amounts[category].append(charged)
            amounts["retained_allowances_usd"].append(charged)
            if status == "settled":
                warnings.append(f"{identifier}: settled entry lacks complete valid usage/rates; its ledger amount is unverifiable.")
            elif category == "unknown_status_retained_usd":
                warnings.append(f"{identifier}: unknown status {status}; its ledger amount is retained as unverified.")
    result = {field: math.fsum(amounts[field]) for field in MONEY_FIELDS}
    result.update({field: counts[field] for field in COUNT_FIELDS})
    result.update(unique_requests=len(requests), additional_attempts=sum(n - 1 for n in requests.values()),
                  statuses=dict(sorted(statuses.items())), warnings=warnings)
    reconciled = result["usage_estimated_usd"] + result["usage_ledger_adjustment_usd"] + result["retained_allowances_usd"]
    result["ledger_reconciled"] = math.isclose(result["ledger_committed_usd"], reconciled, rel_tol=1e-12, abs_tol=1e-9)
    return result


def read_ledger_report(path):
    """Read one atomic file snapshot, without BudgetTracker, locks, or writes."""
    path = Path(path).expanduser().resolve()
    observed = datetime.now(timezone.utc).isoformat()
    if not path.exists():
        return dict(path=str(path), available=False, observed_at_utc=observed,
                    note="Ledger file is absent; cost is unknown, not zero.")
    raw = path.read_bytes()
    ledger = json.loads(raw)
    if ledger.get("schema_version") != 1 or not isinstance(ledger.get("entries"), dict):
        raise ValueError(f"Unsupported budget ledger schema: {path}")
    entries = ledger["entries"]
    if any(not isinstance(entry, dict) for entry in entries.values()):
        raise ValueError(f"Invalid budget entries: {path}")
    for identifier, entry in entries.items():
        if "id" in entry and entry["id"] != identifier:
            raise ValueError(f"Entry ID differs from ledger dictionary key: {identifier}")
    by_model = defaultdict(list)
    for identifier, entry in entries.items():
        by_model[str(entry.get("model") or "unknown")].append((identifier, entry))
    return dict(path=str(path), available=True, observed_at_utc=observed,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                enforced=bool(ledger.get("enforced", True)),
                historical_limit_usd=ledger.get("limit_usd"), stopped=bool(ledger.get("stopped", False)),
                stop_category=ledger.get("stop_category"), audit_event_count=len(ledger.get("audit_events", [])),
                totals=_aggregate_entries(list(entries.items())),
                by_model={model: _aggregate_entries(items) for model, items in sorted(by_model.items())})


def build_cost_report(main_ledger, ablation_ledger):
    paths = [Path(main_ledger).expanduser().resolve(), Path(ablation_ledger).expanduser().resolve()]
    if paths[0] == paths[1]:
        raise ValueError("Main and ablation ledgers must be different files; otherwise costs would be counted twice.")
    ledgers = {name: read_ledger_report(path) for name, path in zip(("main", "ablation"), paths)}
    present = [report for report in ledgers.values() if report["available"]]
    combined = {field: math.fsum(report["totals"][field] for report in present) for field in MONEY_FIELDS}
    combined.update({field: sum(report["totals"][field] for report in present) for field in COUNT_FIELDS})
    combined["available_ledgers"] = len(present)
    combined["expected_ledgers"] = 2
    combined["coverage_complete"] = len(present) == 2
    report = dict(schema_version=1, currency="USD", ledgers=ledgers, combined=combined,
                  estimate_basis="returned prompt/completion usage multiplied by per-attempt stored model token rates",
                  billing_note="These are token-usage cost estimates and planning allowances, not confirmed provider invoices.",
                  history_policy="All ledger attempts are retained, including archived smoke failures and retries. Audit snapshots and result JSONLs are not added again.",
                  snapshot_note="Each ledger is a read-only atomic-file snapshot; concurrent requests may settle after it is read.")
    return report


def format_cost_report(report):
    lines = ["MV-CrystalBench API cost report (USD)", "",
             "Usage-based estimate = returned token counts x stored model rates; no invoice was queried.",
             "Retained allowances cover failed/no-usage/pending attempts. They are not confirmed spending.",
             "Ledger commitment = usage estimate + ledger adjustment + retained allowances; it is NOT a bill.",
             "All historical ledger attempts remain included, even when their result files were archived.",
             "Audit-event snapshots and expanded/reused evaluation rows are not additional charges.", ""]
    for name, ledger in report["ledgers"].items():
        lines.append(f"{name}: {ledger['path']}")
        lines.append(f"Snapshot read: {ledger['observed_at_utc']}")
        if not ledger["available"]:
            lines.extend([ledger["note"], ""])
            continue
        lines.append(f"Budget enforcement: {'enabled' if ledger['enforced'] else 'disabled'}; ledger attempts={ledger['totals']['attempts']}.")
        for label, stats in [*ledger["by_model"].items(), ("TOTAL", ledger["totals"])]:
            lines.append(f"  {label}: usage estimate=${stats['usage_estimated_usd']:.6f} "
                         f"({stats['usage_priced_attempts']} attempts); retained=${stats['retained_allowances_usd']:.6f}; "
                         f"ledger adjustment=${stats['usage_ledger_adjustment_usd']:.6f}; "
                         f"ledger commitment=${stats['ledger_committed_usd']:.6f} (not a bill)")
            lines.append(f"    retained breakdown: failed=${stats['failed_retained_usd']:.6f}, "
                         f"missing/invalid usage=${stats['missing_usage_retained_usd']:.6f}, "
                         f"pending/interrupted=${stats['outstanding_reserved_usd']:.6f}, "
                         f"unverifiable settled=${stats['unverifiable_settled_usd']:.6f}, "
                         f"unknown status=${stats['unknown_status_retained_usd']:.6f}; "
                         f"attempts={stats['attempts']}, unique requests={stats['unique_requests']}, "
                         f"additional attempts={stats['additional_attempts']}")
        for warning in ledger["totals"]["warnings"]:
            lines.append("  Data note: " + warning)
        lines.append("")
    total = report["combined"]
    lines.extend([f"Combined coverage: {total['available_ledgers']}/{total['expected_ledgers']} ledgers.",
                  f"Returned-usage estimate: ${total['usage_estimated_usd']:.6f} across {total['usage_priced_attempts']} attempts.",
                  f"Unverified retained allowances: ${total['retained_allowances_usd']:.6f} across {total['unpriced_attempts']} attempts.",
                  f"Ledger adjustment relative to usage estimate: ${total['usage_ledger_adjustment_usd']:.6f}.",
                  f"Ledger commitment (NOT actual billed spend): ${total['ledger_committed_usd']:.6f}.",
                  "Live snapshots may change as pending requests return; provider billing can differ from these token-based estimates."])
    if not total["coverage_complete"]:
        lines.append("Combined numbers cover available ledgers only; absent-ledger costs are unknown.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-ledger", type=Path, default=Path("results/final_run/main_budget.json"))
    parser.add_argument("--ablation-ledger", type=Path, default=Path("results/final_run/ablation_budget.json"))
    parser.add_argument("--output", type=Path, help="Optionally write text and adjacent JSON; otherwise print only.")
    args = parser.parse_args()
    report = build_cost_report(args.main_ledger, args.ablation_ledger)
    text = format_cost_report(report)
    if args.output:
        protected = {Path(args.main_ledger).resolve(), Path(args.ablation_ledger).resolve()}
        if args.output.resolve() in protected or args.output.with_suffix(".json").resolve() in protected:
            parser.error("Report output must not overwrite either source ledger.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
