"""Summarize evaluation JSONL with bootstrap confidence intervals by structure.

Clusters are base_structure, never individual augmented questions. Each draw
resamples whole structures, and the accuracy is the question-weighted mean
over the selected clusters. Skipped text-inapplicable items are excluded;
invalid responses and request errors count as incorrect and are reported.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


FIELDS = [
    "model", "provider", "condition", "dry_run", "response_models",
    "dataset_fingerprint", "config_fingerprint", "scope", "qtype", "difficulty",
    "subgroup", "n", "n_structures", "correct", "accuracy", "ci_low", "ci_high",
    "invalid_count", "invalid_rate", "error_count", "error_rate", "skipped_count",
    "bootstrap_replicates", "ci_method", "ci_note",
]


def _seed(seed: int, identity) -> int:
    value = json.dumps([seed, identity], sort_keys=True, ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


def accuracy_seed(seed: int, model: str, condition: str, task: str = "overall") -> int:
    """One bootstrap identity for the same accuracy cohort in every report.

    Task is overall, Q1a, Q1b, Q2_easy, Q2_hard, Q3, Q4, Q5, or Q6.
    Paired contrasts and other subgroup statistics retain separate identities.
    The identity preserves the existing main-table confidence intervals.
    """
    return _seed(seed, ["main-table", model, condition, task])


def cluster_interval(records: Iterable[dict], repeats=2000, seed=20261021):
    """Return micro accuracy and percentile 95% CI by resampling base structures."""
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    clusters = defaultdict(lambda: [0, 0])
    for record in records:
        if record.get("status") == "skipped":
            continue
        base = record.get("base_structure") or record.get("meta", {}).get("base_structure")
        if not base:
            raise ValueError(f"Missing base_structure for {record.get('question_id')}.")
        clusters[str(base)][0] += int(record.get("status") == "ok" and record.get("correct") is True)
        clusters[str(base)][1] += 1
    if not clusters:
        return None, None, None, 0, "no applicable questions"
    counts = np.array([clusters[key] for key in sorted(clusters)], dtype=float)
    accuracy = float(counts[:, 0].sum() / counts[:, 1].sum())
    if len(counts) == 1:
        return accuracy, None, None, 1, "fewer than two independent structures"
    rng = np.random.default_rng(seed)
    # Multinomial counts are equivalent to drawing n_clusters cluster IDs
    # with replacement, but avoid duplicating the individual question records.
    weights = rng.multinomial(len(counts), np.full(len(counts), 1 / len(counts)), size=repeats)
    samples = (weights @ counts[:, 0]) / (weights @ counts[:, 1])
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return accuracy, float(lower), float(upper), len(counts), ""


def load_results(paths: Iterable[Path]) -> list[dict]:
    """Read result files, rejecting ambiguous duplicate observations."""
    unique = {}
    for path in sorted(paths):
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                required = {"question_id", "model", "condition", "qtype", "status"}
                if not required <= record.keys():
                    raise ValueError(f"{path}:{line_number}: not an evaluation result record.")
                record = dict(record)
                record["condition"] = {"image_text": "image+text", "no-image": "no_image"}.get(
                    record["condition"], record["condition"])
                if record["condition"] not in {"image", "text", "image+text", "no_image"}:
                    raise ValueError(f"Unknown input condition: {record['condition']}.")
                if record["status"] not in {"ok", "invalid_response", "error", "skipped"}:
                    raise ValueError(f"Unknown result status: {record['status']}.")
                if record["status"] != "skipped" and not isinstance(record.get("correct"), bool):
                    raise ValueError("Scored results must have a Boolean correct field.")
                if record["status"] in {"error", "invalid_response"} and record["correct"]:
                    raise ValueError("Error/invalid results cannot be correct.")
                if record["status"] == "skipped" and record.get("correct") is not None:
                    raise ValueError("Skipped results must have correct=null.")
                if bool(record.get("invalid", record["status"] == "invalid_response")) != (
                        record["status"] == "invalid_response"):
                    raise ValueError("The invalid flag disagrees with result status.")
                if record["qtype"] == "Q5":
                    meta = record.get("meta", {})
                    if "negative_type" not in meta or meta["negative_type"] not in {
                            None, "swap_elements", "delete_atom", "stretch_axis", "polymorph"}:
                        raise ValueError("Q5 requires a known negative_type (or null for positive).")
                    if "is_same_structure" in meta and (
                            meta["is_same_structure"] is not (meta["negative_type"] is None)):
                        raise ValueError("Q5 positive/negative metadata disagrees.")
                key = (
                    record["model"], record.get("provider"), record["condition"],
                    record.get("dry_run", False), record.get("dataset_fingerprint", ""),
                    record.get("config_fingerprint", ""), record["question_id"],
                )
                if key in unique and record != unique[key]:
                    raise ValueError(
                        f"Conflicting repeated result for {record['question_id']} in {path}. "
                        "Select a single coherent run with --input.")
                unique[key] = record
    return list(unique.values())


def summarize_records(records: list[dict], repeats=2000, seed=20261021) -> list[dict]:
    runs = defaultdict(list)
    for record in records:
        key = (
            record["model"], record.get("provider", ""), record["condition"],
            bool(record.get("dry_run", False)), record.get("dataset_fingerprint", ""),
            record.get("config_fingerprint", ""),
        )
        runs[key].append(record)
    rows = []
    for run_key, run in sorted(runs.items()):
        groups = defaultdict(list)
        for record in run:
            groups[("overall", "ALL", "", "")].append(record)
            qtype = record["qtype"]
            difficulty = record.get("difficulty", "") if qtype == "Q2" else ""
            groups[("qtype", qtype, difficulty, "")].append(record)
            meta = record.get("meta", {})
            if qtype == "Q3":
                if not isinstance(meta.get("depth_trap"), bool):
                    raise ValueError("Q3 results require a Boolean meta.depth_trap.")
                subgroup = "depth_trap=" + str(meta["depth_trap"]).lower()
                groups[("q3_depth_trap", qtype, "", subgroup)].append(record)
            if qtype == "Q5":
                kind = meta.get("negative_type") or "positive"
                groups[("q5_negative_type", qtype, "", str(kind))].append(record)
        for group_key, members in sorted(groups.items()):
            applicable = [row for row in members if row["status"] != "skipped"]
            if group_key[0] == "overall":
                bootstrap_seed = accuracy_seed(seed, run_key[0], run_key[2])
            elif group_key[0] == "qtype":
                task = (f"Q2_{group_key[2]}" if group_key[1] == "Q2" else group_key[1])
                bootstrap_seed = accuracy_seed(seed, run_key[0], run_key[2], task)
            else:
                bootstrap_seed = _seed(seed, [run_key, group_key])
            acc, lower, upper, n_clusters, note = cluster_interval(
                applicable, repeats, bootstrap_seed)
            invalid = sum(bool(row.get("invalid", row["status"] == "invalid_response"))
                          for row in applicable)
            errors = sum(row["status"] == "error" for row in applicable)
            n = len(applicable)
            rows.append(dict(
                model=run_key[0], provider=run_key[1], condition=run_key[2],
                dry_run=run_key[3],
                response_models=";".join(sorted({str(row["response_model"])
                                                for row in applicable if row.get("response_model")})),
                dataset_fingerprint=run_key[4], config_fingerprint=run_key[5],
                scope=group_key[0], qtype=group_key[1], difficulty=group_key[2],
                subgroup=group_key[3], n=n, n_structures=n_clusters,
                correct=sum(row["status"] == "ok" and row["correct"] is True for row in applicable),
                accuracy=acc, ci_low=lower, ci_high=upper,
                invalid_count=invalid, invalid_rate=invalid / n if n else None,
                error_count=errors, error_rate=errors / n if n else None,
                skipped_count=len(members) - n, bootstrap_replicates=repeats,
                ci_method="percentile 95%, resample whole base_structure clusters",
                ci_note=note,
            ))
    return rows


def _percent(value):
    return "-" if value is None else f"{value * 100:.1f}%"


MAIN_MODELS = ("gpt", "claude", "kimi", "random")
MAIN_CONDITIONS = ("image", "text", "image+text", "no_image")
MAIN_TASKS = ("Q1a", "Q1b", "Q2_easy", "Q2_hard", "Q3", "Q4", "Q5", "Q6")


def task_label(record):
    return f"Q2_{record.get('difficulty', '')}" if record["qtype"] == "Q2" else record["qtype"]


def coherent_main_runs(records, models=MAIN_MODELS):
    """Select genuine main runs, rejecting mixtures hidden behind a model alias."""
    runs = defaultdict(list)
    for record in records:
        if (record["model"] not in models or record.get("dry_run", False)
                or "k" in record or "strategy" in record):
            continue
        runs[(record["model"], record["condition"])].append(record)
    for key, members in runs.items():
        signatures = {(row.get("dataset_fingerprint", ""), row.get("config_fingerprint", ""),
                       row.get("provider", "")) for row in members}
        if len(signatures) > 1:
            raise ValueError(f"Ambiguous main run {key}: select one dataset/configuration with --input.")
        ids = [row["question_id"] for row in members]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Repeated question IDs in main run {key}.")
    return dict(runs)


def build_main_tables(records, questions=None, repeats=2000, seed=20261021,
                      models=MAIN_MODELS, conditions=MAIN_CONDITIONS):
    """Return display cells and machine-readable CIs with explicit coverage.

    The question manifest defines expected denominators, including the two
    text-inapplicable visual tasks. Missing results are reported, never invented
    or added to an accuracy denominator. Recorded errors count as incorrect.
    """
    runs = coherent_main_runs(records, models)
    questions = list(questions or [])
    expected_lookup = {q["id"]: q for q in questions}
    if len(expected_lookup) != len(questions):
        raise ValueError("The expected question manifest contains duplicate IDs.")
    rows = []
    if not conditions or any(condition not in MAIN_CONDITIONS for condition in conditions):
        raise ValueError("Select at least one recognized main-table condition.")
    for condition in dict.fromkeys(conditions):
        expected = [q for q in questions if condition != "text"
                    or q.get("meta", {}).get("text_applicable", True)]
        for model in models:
            run = runs.get((model, condition), [])
            if expected_lookup:
                unknown = {r["question_id"] for r in run} - set(expected_lookup)
                if unknown:
                    raise ValueError(f"{model}/{condition}: {len(unknown)} results outside --questions manifest.")
            for task in (*MAIN_TASKS, "overall"):
                members = [r for r in run if task == "overall" or task_label(r) == task]
                expected_ids = {q["id"] for q in expected if task == "overall" or task_label(q) == task}
                applicable = [r for r in members if r["status"] != "skipped"]
                if expected_lookup and any(r["question_id"] not in expected_ids for r in applicable):
                    raise ValueError(f"{model}/{condition}/{task}: scored a text-inapplicable question.")
                acc, low, high, bases, note = cluster_interval(
                    applicable, repeats, accuracy_seed(seed, model, condition, task))
                n = len(applicable)
                expected_n = len(expected_ids) if questions else None
                row = dict(model=model, condition=condition, task=task, n=n,
                           expected_n=expected_n, missing_count=(expected_n - n if expected_n is not None else None),
                           recorded_count=len(members), skipped_count=len(members) - n,
                           n_structures=bases,
                           correct=sum(r["status"] == "ok" and r["correct"] for r in applicable),
                           accuracy=acc, ci_low=low, ci_high=high,
                           error_count=sum(r["status"] == "error" for r in applicable),
                           invalid_count=sum(r["status"] == "invalid_response" for r in applicable),
                           bootstrap_replicates=repeats,
                           ci_method="percentile 95%, resample whole base_structure clusters", ci_note=note)
                rows.append(row)
    return rows


def write_main_tables(records, output, questions=None, repeats=2000, seed=20261021,
                      models=MAIN_MODELS, conditions=MAIN_CONDITIONS):
    """Write one model-by-task table per condition plus long and wide CSVs."""
    rows = build_main_tables(records, questions, repeats, seed, models, conditions)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lookup = {(r["model"], r["condition"], r["task"]): r for r in rows}
    lines = ["Main evaluation: accuracy (%) [95% cluster-bootstrap CI]", 
             "Clusters: base_structure; errors/invalid responses are incorrect; skipped items are excluded.",
             "Cells show n/expected coverage. Missing results are not scored; dry runs are excluded.", ""]
    wide_rows = []
    for condition in dict.fromkeys(conditions):
        lines.extend([f"Condition: {condition}", " | ".join(("Model", *MAIN_TASKS, "overall"))])
        for model in models:
            cells = []
            wide = {"condition": condition, "model": model}
            for task in (*MAIN_TASKS, "overall"):
                row = lookup[(model, condition, task)]
                if row["expected_n"] == 0:
                    cell = "N/A (not applicable)"
                elif not row["n"]:
                    cell = f"N/A (0/{row['expected_n'] if row['expected_n'] is not None else '?'})"
                else:
                    ci = (f"[{100*row['ci_low']:.1f}, {100*row['ci_high']:.1f}]"
                          if row["ci_low"] is not None else "[CI unavailable]")
                    cell = f"{100*row['accuracy']:.1f} {ci} (n={row['n']}/{row['expected_n'] if row['expected_n'] is not None else '?'})"
                cells.append(cell)
                wide[task] = cell
            lines.append(" | ".join((model, *cells)))
            wide_rows.append(wide)
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ci_path = output.with_name(output.stem + "_ci.csv")
    with ci_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    wide_path = output.with_suffix(".csv")
    with wide_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "model", *MAIN_TASKS, "overall"])
        writer.writeheader()
        writer.writerows(wide_rows)
    return {"text": str(output.resolve()), "csv": str(wide_path.resolve()), "ci_csv": str(ci_path.resolve())}


def print_table(rows, scopes=("overall", "qtype", "q3_depth_trap", "q5_negative_type")):
    print("Accuracy and 95% cluster-bootstrap CI (grouped by base_structure)")
    print("Skipped questions are excluded; invalid responses and API errors count as incorrect.")
    print(f"{'Model':<23} {'Condition':<11} {'Group':<34} {'N':>4} {'Bases':>5} "
          f"{'Acc':>7} {'95% CI':>16} {'Invalid':>8} {'Errors':>7}")
    print("-" * 130)
    for row in rows:
        if row["scope"] not in scopes:
            continue
        model = row["model"] + (" [dry-run]" if row["dry_run"] else "")
        label = row["qtype"]
        if row["difficulty"]:
            label += "/" + row["difficulty"]
        if row["subgroup"]:
            label += "/" + row["subgroup"]
        ci = (f"[{100 * row['ci_low']:.1f}, {100 * row['ci_high']:.1f}]%"
              if row["ci_low"] is not None else "N/A")
        print(f"{model:<23} {row['condition']:<11} {label:<34} {row['n']:>4} "
              f"{row['n_structures']:>5} {_percent(row['accuracy']):>7} {ci:>16} "
              f"{_percent(row['invalid_rate']):>8} {_percent(row['error_rate']):>7}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--input", type=Path, nargs="+", help="Use only these JSONL result files.")
    parser.add_argument("--output", type=Path, help="Default: results-dir/summary.csv")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--overview-only", action="store_true",
                        help="Print overall rows only; CSV still contains all breakdowns.")
    parser.add_argument("--questions", type=Path, default=Path("data/questions_mini.jsonl"),
                        help="Expected main-evaluation cohort, used to report coverage.")
    parser.add_argument("--main-tables", type=Path, help="Default: output directory/main_tables.txt")
    parser.add_argument("--models", nargs="+", default=list(MAIN_MODELS))
    parser.add_argument("--conditions", nargs="+", choices=MAIN_CONDITIONS,
                        default=list(MAIN_CONDITIONS), help="Conditions to display in main tables.")
    args = parser.parse_args()
    # Ablations have k/strategy groups and their own plots/CSV. Pooling those
    # observations here would silently inflate standard-evaluation sample sizes.
    paths = args.input if args.input else sorted(
        path for path in args.results_dir.glob("*.jsonl")
        if not path.name.startswith("ablation_")
    )
    if not paths:
        parser.error("No result JSONL files found.")
    records = load_results(paths)
    rows = summarize_records(records, args.bootstrap, args.seed)
    output = args.output or args.results_dir / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_table(rows, scopes=("overall",) if args.overview_only else
                ("overall", "qtype", "q3_depth_trap", "q5_negative_type"))
    print(f"\nCSV: {output.resolve()}")
    questions = ([json.loads(line) for line in args.questions.read_text("utf-8").splitlines() if line.strip()]
                 if args.questions.exists() else [])
    main_paths = write_main_tables(records, args.main_tables or output.parent / "main_tables.txt",
                                  questions, args.bootstrap, args.seed, args.models, args.conditions)
    print("Main tables: " + json.dumps(main_paths, ensure_ascii=False))


if __name__ == "__main__":
    main()
