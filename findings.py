"""Produce coverage-aware findings from explicitly selected real evaluation runs.

All intervals resample whole base_structure clusters. Image/text comparisons
are paired on question IDs in the text-applicable cohort. Missing observations
are not fabricated; recorded errors and invalid responses count as incorrect.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from summarize import (
    MAIN_CONDITIONS, MAIN_TASKS, _seed, accuracy_seed, build_main_tables, cluster_interval,
    coherent_main_runs, load_results, task_label,
)


MODELS = ("gpt", "claude", "kimi")


def _base(row):
    base = row.get("base_structure") or row.get("meta", {}).get("base_structure")
    if not base:
        raise ValueError(f"Missing base_structure: {row.get('question_id')}")
    return str(base)


def _correct(row):
    return int(row["status"] == "ok" and row["correct"] is True)


def _applicable(rows):
    return [row for row in rows if row["status"] != "skipped"]


def accuracy_statistics(rows, repeats=2000, seed=20261021, expected=None):
    applicable = _applicable(rows)
    accuracy, low, high, bases, note = cluster_interval(applicable, repeats, seed)
    return dict(accuracy=accuracy, ci_low=low, ci_high=high, n=len(applicable),
                correct=sum(_correct(row) for row in applicable), n_structures=bases,
                expected_n=expected, missing_count=expected - len(applicable) if expected is not None else None,
                recorded_count=len(rows), skipped_count=len(rows) - len(applicable),
                error_count=sum(row["status"] == "error" for row in applicable),
                invalid_count=sum(row["status"] == "invalid_response" for row in applicable), ci_note=note)


def _difference_interval(left, right, repeats, seed):
    """Joint cluster bootstrap; both groups use the same sampled base weights."""
    if repeats < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    left, right = _applicable(left), _applicable(right)
    left_acc = sum(map(_correct, left)) / len(left) if left else None
    right_acc = sum(map(_correct, right)) / len(right) if right else None
    result = dict(difference=None, ci_low=None, ci_high=None, left_accuracy=left_acc,
                  right_accuracy=right_acc, left_n=len(left), right_n=len(right),
                  left_error_count=sum(r["status"] == "error" for r in left),
                  right_error_count=sum(r["status"] == "error" for r in right),
                  left_invalid_count=sum(r["status"] == "invalid_response" for r in left),
                  right_invalid_count=sum(r["status"] == "invalid_response" for r in right),
                  n_structures=0, valid_bootstrap_replicates=0, ci_note="")
    if not left or not right:
        result["ci_note"] = "one or both comparison groups have no applicable results"
        return result
    result["difference"] = left_acc - right_acc
    clusters = defaultdict(lambda: [0, 0, 0, 0])
    for rows, offset in ((left, 0), (right, 2)):
        for row in rows:
            clusters[_base(row)][offset] += _correct(row)
            clusters[_base(row)][offset + 1] += 1
    counts = np.array([clusters[key] for key in sorted(clusters)], dtype=float)
    result["n_structures"] = len(counts)
    if len(counts) < 2:
        result["ci_note"] = "fewer than two independent structures"
        return result
    weights = np.random.default_rng(seed).multinomial(
        len(counts), np.full(len(counts), 1 / len(counts)), size=repeats)
    totals = weights @ counts
    valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
    result["valid_bootstrap_replicates"] = int(valid.sum())
    if valid.sum() < 100:
        result["ci_note"] = "fewer than 100 bootstrap draws contain both groups"
        return result
    samples = totals[valid, 0] / totals[valid, 1] - totals[valid, 2] / totals[valid, 3]
    result["ci_low"], result["ci_high"] = map(float, np.quantile(samples, [.025, .975]))
    if not valid.all():
        result["ci_note"] = f"{int((~valid).sum())} resamples lacked one group and were excluded"
    return result


def paired_difference(left, right, *, expected_ids=None, id_field="question_id",
                      repeats=2000, seed=20261021):
    """Compare only common applicable observations, retaining failure coverage."""
    def index(rows):
        result = {}
        for row in rows:
            key = row[id_field]
            if key in result:
                raise ValueError(f"Repeated pairing ID: {key}")
            if expected_ids is None or key in expected_ids:
                result[key] = row
        return result

    lmap, rmap = index(left), index(right)
    expected = set(expected_ids) if expected_ids is not None else set(lmap) | set(rmap)
    lvalid = {key for key, row in lmap.items() if row["status"] != "skipped"}
    rvalid = {key for key, row in rmap.items() if row["status"] != "skipped"}
    common = sorted(lvalid & rvalid)
    for key in common:
        if (_base(lmap[key]) != _base(rmap[key]) or task_label(lmap[key]) != task_label(rmap[key])):
            raise ValueError(f"Paired question identity disagrees for {key}.")
    result = _difference_interval([lmap[key] for key in common], [rmap[key] for key in common], repeats, seed)
    result.update(paired_n=len(common), expected_n=len(expected),
                  left_available=len(lvalid), right_available=len(rvalid),
                  left_missing=len(expected - lvalid), right_missing=len(expected - rvalid),
                  unpaired_left=len(lvalid - rvalid), unpaired_right=len(rvalid - lvalid),
                  paired_question_ids=common)
    return result


def _percent(value):
    return "N/A" if value is None else f"{100 * value:.1f}%"


def _ci(stats, difference=False):
    if stats["ci_low"] is None:
        return "CI unavailable" + (f" ({stats['ci_note']})" if stats.get("ci_note") else "")
    suffix = " pp" if difference else "%"
    return f"95% CI [{100 * stats['ci_low']:.1f}, {100 * stats['ci_high']:.1f}]{suffix}"


def _accuracy_text(stats):
    expected = "?" if stats["expected_n"] is None else str(stats["expected_n"])
    return (f"{_percent(stats['accuracy'])}; {_ci(stats)}; correct={stats['correct']}/{stats['n']}; "
            f"coverage={stats['n']}/{expected}; bases={stats['n_structures']}; "
            f"errors={stats['error_count']}, invalid={stats['invalid_count']}, skipped={stats['skipped_count']}")


def _difference_text(stats, paired=False):
    value = "N/A" if stats["difference"] is None else f"{100 * stats['difference']:+.1f} pp"
    text = (f"{value}; {_ci(stats, difference=True)}; "
            f"left={_percent(stats['left_accuracy'])} (n={stats['left_n']}), "
            f"right={_percent(stats['right_accuracy'])} (n={stats['right_n']}); "
            f"bases={stats['n_structures']}; errors L/R={stats['left_error_count']}/{stats['right_error_count']}, "
            f"invalid L/R={stats['left_invalid_count']}/{stats['right_invalid_count']}")
    if paired:
        text += (f"; paired coverage={stats['paired_n']}/{stats['expected_n']}; "
                 f"available L/R={stats['left_available']}/{stats['right_available']}; "
                 f"missing L/R={stats['left_missing']}/{stats['right_missing']}")
    if stats.get("ci_note") and stats["ci_low"] is not None:
        text += "; " + stats["ci_note"]
    return text


def _ablation_runs(records, models):
    from ablation_plots import _normalized_records, _visibility_observations

    grouped = defaultdict(list)
    for row in records:
        if row.get("dry_run", False) or row["model"] not in models:
            continue
        grouped[(row["model"], row["condition"])].append(row)
    result = {}
    for key, run in grouped.items():
        run, _ = _normalized_records(run)
        _visibility_observations(run)  # Reject inconsistent shared endpoints.
        result[key] = run
    return result


def build_findings(records, ablation_records, questions, *, models=MODELS,
                   repeats=2000, seed=20261021, ablation_per_type=30,
                   ablation_expected_counts=None, conditions=MAIN_CONDITIONS):
    """Return human-readable findings and all numerical statistics for auditing."""
    if ablation_per_type < 1:
        raise ValueError("ablation_per_type must be positive.")
    ablation_expected_counts = dict(ablation_expected_counts or {})
    if any(type(value) is not int or value < 1 for value in ablation_expected_counts.values()):
        raise ValueError("Expected ablation counts must be positive integers per model and task.")
    questions = list(questions)
    expected = {q["id"]: q for q in questions}
    if not expected or len(expected) != len(questions):
        raise ValueError("Supply a nonempty question manifest with unique IDs.")
    runs = coherent_main_runs(records, models)
    # Reuse main-table manifest/denominator checks before making comparisons.
    build_main_tables(records, questions, repeats, seed, models, conditions)
    for (model, condition), run in runs.items():
        for row in run:
            question = expected[row["question_id"]]
            if (_base(row) != str(question["meta"]["base_structure"])
                    or task_label(row) != task_label(question)):
                raise ValueError(f"{model}/{condition}: question identity differs from the manifest.")
            if row["qtype"] == "Q3" and row.get("meta", {}).get("depth_trap") is not question["meta"]["depth_trap"]:
                raise ValueError("Q3 depth_trap differs from the expected manifest.")
    ablation_runs = _ablation_runs(ablation_records, models)
    text_ids = {qid for qid, q in expected.items() if q.get("meta", {}).get("text_applicable", True)}
    statistics = dict(image_overall={}, image_text_paired={}, no_image={}, q3_depth_trap={},
                      ablation={}, lowest={}, bootstrap_replicates=repeats, seed=seed,
                      expected_main_n=len(expected), expected_text_n=len(text_ids),
                      conditions=list(dict.fromkeys(conditions)),
                      ignored_dry_run_records=sum(bool(r.get("dry_run", False)) for r in records),
                      ignored_ablation_dry_run_records=sum(bool(r.get("dry_run", False)) for r in ablation_records),
                      expected_ablation_per_type={model: ablation_expected_counts.get(model, ablation_per_type)
                                                  for model in models})
    lines = ["MV-CrystalBench: key findings", "",
             f"Expected main cohort: {len(expected)} questions; text-applicable paired cohort: {len(text_ids)}.",
             f"Intervals: percentile 95% cluster bootstrap over base_structure ({repeats} draws; seed={seed}).",
             "Recorded API errors and invalid outputs count as incorrect; skipped items are excluded.",
             "Missing results are reported and excluded, so partial coverage is not a full-cohort estimate.",
             "All real records are used regardless of success; dry runs never support substantive conclusions.",
             "Differences are left minus right in percentage points (pp); no significance claim is inferred from CI overlap.",
             "", "1. Image accuracy and paired image minus text"]
    for model in models:
        image = runs.get((model, "image"), [])
        text = runs.get((model, "text"), [])
        image_stats = accuracy_statistics(image, repeats, accuracy_seed(seed, model, "image"), len(expected))
        text_stats = accuracy_statistics(text, repeats, accuracy_seed(seed, model, "text"), len(text_ids))
        paired = paired_difference(image, text, expected_ids=text_ids, repeats=repeats,
                                   seed=_seed(seed, [model, "image-text-paired"]))
        statistics["image_overall"][model] = image_stats
        statistics["image_text_paired"][model] = {"image_full": image_stats, "text_full": text_stats, "paired": paired}
        lines.extend([f"{model} image: {_accuracy_text(image_stats)}",
                      f"{model} text: {_accuracy_text(text_stats)}",
                      f"{model} image - text, common text-applicable question IDs only: {_difference_text(paired, True)}"])
    lines.extend(["", "2. No-image control versus the 25% random-choice reference"])
    for model in models:
        stats = accuracy_statistics(runs.get((model, "no_image"), []), repeats,
                                    accuracy_seed(seed, model, "no_image"), len(expected))
        contrast = {**stats, "difference": None if stats["accuracy"] is None else stats["accuracy"] - .25,
                    "ci_low": None if stats["ci_low"] is None else stats["ci_low"] - .25,
                    "ci_high": None if stats["ci_high"] is None else stats["ci_high"] - .25}
        statistics["no_image"][model] = {"accuracy": stats, "versus_chance": contrast}
        delta = "N/A" if contrast["difference"] is None else f"{100*contrast['difference']:+.1f} pp"
        lines.append(f"{model}: {_accuracy_text(stats)}; minus 25%={delta}; {_ci(contrast, True)}")
    lines.extend(["", "3. Q3 depth trap: true minus false, separately by input condition"])
    for model in models:
        for condition in dict.fromkeys(conditions):
            members = [r for r in runs.get((model, condition), []) if r["qtype"] == "Q3"]
            left = [r for r in members if r["meta"]["depth_trap"] is True]
            right = [r for r in members if r["meta"]["depth_trap"] is False]
            stats = _difference_interval(left, right, repeats, _seed(seed, [model, condition, "depth-trap"]))
            expected_left = sum(q["qtype"] == "Q3" and q["meta"]["depth_trap"] is True for q in questions)
            expected_right = sum(q["qtype"] == "Q3" and q["meta"]["depth_trap"] is False for q in questions)
            stats.update(expected_left_n=expected_left, expected_right_n=expected_right)
            statistics["q3_depth_trap"][f"{model}/{condition}"] = stats
            lines.append(f"{model}/{condition}: {_difference_text(stats)}; "
                         f"coverage true={stats['left_n']}/{expected_left}, false={stats['right_n']}/{expected_right}")
    lines.extend(["", "4-5. Paired view ablations: k=1 max minus min; k=1 random minus shared k=8"])
    for model in models:
        available_conditions = sorted({condition for candidate, condition in ablation_runs if candidate == model}) or ["image"]
        for condition in available_conditions:
            run = ablation_runs.get((model, condition), [])
            for task in ("Q1a", "Q2_hard", "Q3", "overall"):
                group = [r for r in run if task == "overall" or task_label(r) == task]
                cohort_ids = {r["base_question_id"] for r in group}
                expected_n = ablation_expected_counts.get(model, ablation_per_type) * (3 if task == "overall" else 1)
                if len(cohort_ids) > expected_n:
                    raise ValueError("Ablation has more base questions than --ablation-per-type allows.")
                k8 = {}
                for row in group:
                    if row["k"] == 8:
                        k8.setdefault(row["base_question_id"], row)
                contrasts = {
                    "k1_max_minus_min": ([r for r in group if r["k"] == 1 and r["strategy"] == "max_visibility"],
                                           [r for r in group if r["k"] == 1 and r["strategy"] == "min_visibility"]),
                    "k1_random_minus_k8": ([r for r in group if r["k"] == 1 and r["strategy"] == "random"],
                                             list(k8.values())),
                }
                for contrast, (left, right) in contrasts.items():
                    stats = paired_difference(left, right, expected_ids=cohort_ids, id_field="base_question_id",
                                              repeats=repeats, seed=_seed(seed, [model, condition, task, contrast]))
                    # The configured study denominator can exceed the observed
                    # union when a run ended before reaching later questions.
                    stats.update(expected_n=expected_n, observed_cohort_n=len(cohort_ids),
                                 unseen_cohort_n=expected_n - len(cohort_ids))
                    stats["left_missing"] = expected_n - stats["left_available"]
                    stats["right_missing"] = expected_n - stats["right_available"]
                    identity = f"{model}/{condition}/{task}/{contrast}"
                    statistics["ablation"][identity] = stats
                    lines.append(f"{identity}: {_difference_text(stats, True)}; "
                                 f"observed base questions={len(cohort_ids)}/{expected_n}")
    lines.extend(["", "6. Three lowest observed image task accuracies and base-structure accuracies per model",
                  "These are descriptive rankings over available results; errors may dominate incomplete runs.",
                  "A single base structure has only one cluster: a cluster-bootstrap CI is unavailable, not zero-width."])
    for model in models:
        image = _applicable(runs.get((model, "image"), []))
        by_task, by_base = defaultdict(list), defaultdict(list)
        for row in image:
            by_task[task_label(row)].append(row)
            by_base[_base(row)].append(row)
        statistics["lowest"][model] = {}
        for kind, groups in (("task", by_task), ("base_structure", by_base)):
            ranked = []
            for label, rows in groups.items():
                expected_n = sum((task_label(q) == label if kind == "task" else q["meta"]["base_structure"] == label)
                                 for q in questions)
                group_seed = (accuracy_seed(seed, model, "image", label) if kind == "task"
                              else _seed(seed, [model, kind, label]))
                stats = accuracy_statistics(rows, repeats, group_seed, expected_n)
                ranked.append({"label": label, **stats})
            ranked.sort(key=lambda stats: (stats["accuracy"], -stats["n"], stats["label"]))
            statistics["lowest"][model][kind] = ranked[:3]
            if not ranked:
                lines.append(f"{model} lowest {kind}: unavailable (no applicable image results).")
            for rank, stats in enumerate(ranked[:3], 1):
                lines.append(f"{model} lowest {kind} #{rank} {stats['label']}: {_accuracy_text(stats)}")
    lines.extend(["", f"Excluded dry-run observations: main={statistics['ignored_dry_run_records']}; "
                  f"ablation={statistics['ignored_ablation_dry_run_records']}.",
                  "All reported comparisons are descriptive; no multiple-comparison adjustment is applied."])
    return "\n".join(lines) + "\n", statistics


def ablation_counts_from_summaries(paths):
    """Read planned cohort sizes, never infer targets from incomplete row counts."""
    counts, sources = {}, {}
    for path in paths:
        summary_path = Path(path).with_suffix(".summary.json")
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text("utf-8"))
        model = summary.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"Missing model in ablation summary: {summary_path}")
        per_type = summary.get("per_type", summary.get("ablation_per_type"))
        if per_type is None:
            planned = summary.get("full_planned_base_questions")
            if type(planned) is not int or planned < 3 or planned % 3:
                raise ValueError(f"Missing balanced planned cohort size in {summary_path}")
            per_type = planned // 3
        if type(per_type) is not int or per_type < 1:
            raise ValueError(f"Invalid per-type planned cohort size in {summary_path}")
        if model in counts and counts[model] != per_type:
            raise ValueError(f"Conflicting planned ablation sizes for model {model}.")
        counts[model], sources[model] = per_type, str(summary_path.resolve())
    return counts, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Explicit real main-evaluation JSONLs; random baselines may also be supplied.")
    parser.add_argument("--ablation", type=Path, nargs="*", default=[])
    parser.add_argument("--questions", type=Path, default=Path("data/questions_mini.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/key_findings.txt"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--conditions", nargs="+", choices=MAIN_CONDITIONS,
                        default=list(MAIN_CONDITIONS), help="Conditions to include in the Q3 comparison.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--ablation-per-type", type=int, default=30,
                        help="Fallback planned questions per task; adjacent .summary.json takes precedence.")
    parser.add_argument("--ablation-counts", type=json.loads, default={},
                        help='Explicit per-model overrides, e.g. {"gpt":30,"claude":30,"kimi":20}.')
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.questions.read_text("utf-8").splitlines() if line.strip()]
    counts, count_sources = ablation_counts_from_summaries(args.ablation)
    if not isinstance(args.ablation_counts, dict):
        parser.error("--ablation-counts must be a JSON object mapping model aliases to counts.")
    counts.update(args.ablation_counts)
    text, statistics = build_findings(load_results(args.input), load_results(args.ablation), questions,
                                      models=args.models, repeats=args.bootstrap, seed=args.seed,
                                      ablation_per_type=args.ablation_per_type, ablation_expected_counts=counts,
                                      conditions=args.conditions)
    statistics["source_files"] = [str(path.resolve()) for path in args.input]
    statistics["ablation_source_files"] = [str(path.resolve()) for path in args.ablation]
    statistics["question_manifest"] = str(args.questions.resolve())
    statistics["ablation_expected_count_sources"] = count_sources
    statistics["ablation_expected_count_overrides"] = args.ablation_counts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Findings: {args.output.resolve()}")


if __name__ == "__main__":
    main()
