"""Plot view ablations with confidence intervals clustered by base structure.

The curves retain each (question, k, strategy) observation. The visibility
association retains these observations at k < 8, but the shared eight-view
endpoint contributes only once per question. This is a descriptive association,
not an estimate of a causal effect. Invalid outputs and request errors count
as incorrect; skipped questions do not enter accuracy denominators.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
import numpy as np

from summarize import cluster_interval, load_results


STRATEGIES = ("random", "canonical", "max_visibility", "min_visibility")
VIEW_COUNTS = (1, 2, 4, 8)
PANELS = ("Q1a", "Q2hard", "Q3")
PANEL_LABELS = {"Q1a": "Q1a · Visible atom count", "Q2hard": "Q2 hard · Coordination", "Q3": "Q3 · Relative distance"}
STRATEGY_LABELS = {"random": "Random", "canonical": "Canonical", "max_visibility": "Max visibility", "min_visibility": "Min visibility"}
COLORS = {"random": "#627084", "canonical": "#3868C8", "max_visibility": "#087F70", "min_visibility": "#C24D45"}
MARKERS = {"random": "o", "canonical": "s", "max_visibility": "^", "min_visibility": "D"}
DEDUP_POLICY = (
    "For k < 8, retain one observation per (base_question_id, k, strategy), "
    "including strategies selecting the same views. For k = 8, validate the "
    "shared response and retain one observation per base_question_id. "
    "Accuracy excludes skipped records; errors and invalid responses count as wrong."
)
CI_METHOD = "percentile 95%; resample whole base_structure clusters"
MODEL_COLORS = {"gpt": "#3868C8", "claude": "#C56A21", "kimi": "#087F70"}
MODEL_ORDER = {"gpt": 0, "claude": 1, "kimi": 2}


def _seed(seed: int, identity) -> int:
    payload = json.dumps([seed, identity], sort_keys=True, ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _panel(record: dict) -> str:
    if record.get("qtype") == "Q2" and record.get("difficulty") == "hard":
        return "Q2hard"
    if record.get("qtype") in {"Q1a", "Q3"}:
        return record["qtype"]
    raise ValueError("Ablation plots require Q1a, Q2 with difficulty=hard, or Q3.")


def _normalized_records(records: Iterable[dict]) -> tuple[list[dict], int]:
    """Validate a coherent run and discard identical repeated JSONL rows."""
    unique = {}
    count = 0
    run_keys = set()
    question_identity = {}
    for original in records:
        count += 1
        row = dict(original)
        panel = _panel(row)
        for field in ("base_question_id", "base_structure", "model"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Ablation result requires nonempty {field}.")
        if type(row.get("k")) is not int or row["k"] not in VIEW_COUNTS:
            raise ValueError("Ablation k must be 1, 2, 4, or 8.")
        if row.get("strategy") not in STRATEGIES:
            raise ValueError(f"Unknown ablation strategy: {row.get('strategy')}.")
        views = row.get("selected_view_ids")
        if not isinstance(views, list) or len(views) != row["k"] or any(
                not isinstance(view, str) or not view for view in views) or len(set(views)) != len(views):
            raise ValueError("selected_view_ids must contain exactly k distinct view IDs.")
        score = row.get("visibility_score_mean", row.get("visibility_score"))
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("visibility_score_mean must be a finite value in [0, 1].")
        if "visibility_score" in row and not np.isclose(row["visibility_score"], score, atol=1e-12, rtol=0):
            raise ValueError("visibility_score and visibility_score_mean disagree.")
        row["visibility_score_mean"] = float(score)
        status = row.get("status")
        if status not in {"ok", "invalid_response", "error", "skipped"}:
            raise ValueError(f"Unknown result status: {status}.")
        if status == "skipped":
            if row.get("correct") is not None:
                raise ValueError("Skipped results must have correct=null.")
        elif not isinstance(row.get("correct"), bool):
            raise ValueError("Scored results must have a Boolean correct field.")
        if status in {"error", "invalid_response"} and row["correct"]:
            raise ValueError("Error/invalid results cannot be correct.")
        if bool(row.get("invalid", status == "invalid_response")) != (status == "invalid_response"):
            raise ValueError("The invalid flag disagrees with result status.")
        if not isinstance(row.get("dry_run", False), bool):
            raise ValueError("dry_run must be Boolean.")
        row["condition"] = {"image_text": "image+text"}.get(row.get("condition", "image"), row.get("condition", "image"))
        run_keys.add((row["model"], row.get("provider", ""), row["condition"], row.get("dry_run", False),
                      row.get("dataset_fingerprint", ""), row.get("config_fingerprint", "")))
        identity = (panel, row["base_structure"])
        qid = row["base_question_id"]
        if qid in question_identity and question_identity[qid] != identity:
            raise ValueError(f"Inconsistent question identity for {qid}.")
        question_identity[qid] = identity
        key = (qid, row["k"], row["strategy"])
        if key in unique and row != unique[key]:
            raise ValueError(f"Conflicting repeated ablation result for {key}.")
        unique[key] = row
    if not unique:
        raise ValueError("No ablation result records were supplied.")
    if len(run_keys) != 1:
        raise ValueError("Select one model/provider/condition/dry-run combination and consistent dataset/config fingerprints per plot.")
    rows = [unique[key] for key in sorted(unique)]
    return rows, count - len(rows)


def _visibility_observations(records: list[dict]) -> tuple[list[dict], int]:
    """Deduplicate the shared k=8 endpoint after checking paired agreement."""
    groups = defaultdict(list)
    result = []
    for row in records:
        if row["k"] == 8:
            groups[row["base_question_id"]].append(row)
        else:
            result.append(row)
    removed = 0
    for qid, group in sorted(groups.items()):
        group.sort(key=lambda row: (row["strategy"] != "canonical", row["strategy"]))
        reference = group[0]
        reference_views = set(reference["selected_view_ids"])
        for row in group[1:]:
            equal_response = all(row.get(field) == reference.get(field) for field in (
                "status", "correct", "prediction", "raw_response", "base_structure", "qtype", "difficulty"))
            if (set(row["selected_view_ids"]) != reference_views or not equal_response or
                    not np.isclose(row["visibility_score_mean"], reference["visibility_score_mean"], rtol=0, atol=1e-12)):
                raise ValueError(f"The shared k=8 endpoint disagrees across strategies for {qid}.")
        result.append(reference)
        removed += len(group) - 1
    return result, removed


def _statistics(records: list[dict], bootstrap: int, seed: int) -> dict:
    applicable = [row for row in records if row["status"] != "skipped"]
    accuracy, low, high, clusters, note = cluster_interval(applicable, bootstrap, seed)
    return {
        "n": len(applicable),
        "n_structures": clusters,
        "correct": sum(row["status"] == "ok" and row["correct"] is True for row in applicable),
        "accuracy": accuracy,
        "ci_low": low,
        "ci_high": high,
        "invalid_count": sum(row["status"] == "invalid_response" for row in applicable),
        "error_count": sum(row["status"] == "error" for row in applicable),
        "skipped_count": len(records) - len(applicable),
        "mean_visibility": float(np.mean([row["visibility_score_mean"] for row in applicable])) if applicable else None,
        "bootstrap_replicates": bootstrap,
        "ci_method": CI_METHOD,
        "ci_note": note,
    }


def build_plot_tables(records: Iterable[dict], *, bootstrap: int = 2000, seed: int = 20261021,
                      bins: int = 5) -> tuple[list[dict], list[dict], dict]:
    """Return curve rows, visibility-bin rows, and the explicit counting policy.

    Every curve has a row for each k, including empty cells, so incomplete runs
    remain visible as missing points. Bin intervals are left-closed/right-open,
    except the final interval includes 1.0. Confidence intervals resample all
    observations of each base structure together, never individual questions.
    """
    if bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates.")
    if type(bins) is not int or bins < 2 or bins > 20:
        raise ValueError("bins must be an integer between 2 and 20.")
    rows, identical_duplicates = _normalized_records(records)
    visibility_rows, removed = _visibility_observations(rows)
    reference = rows[0]
    common = {"model": reference["model"], "provider": reference.get("provider", ""),
              "condition": reference["condition"], "dry_run": reference.get("dry_run", False),
              "dataset_fingerprint": reference.get("dataset_fingerprint", ""),
              "config_fingerprint": reference.get("config_fingerprint", "")}
    curves = []
    for panel in PANELS:
        for strategy in STRATEGIES:
            for k in VIEW_COUNTS:
                members = [row for row in rows if _panel(row) == panel and row["strategy"] == strategy and row["k"] == k]
                # Shared k=8 observations also share bootstrap draws, so their
                # confidence intervals converge exactly, not only their means.
                identity = [panel, "shared" if k == 8 else strategy, k]
                curves.append({**common, "panel": panel, "strategy": strategy, "k": k,
                               "k8_shared": k == 8, **_statistics(members, bootstrap, _seed(seed, identity))})
    edges = np.linspace(0.0, 1.0, bins + 1)
    binned = []
    for panel in PANELS:
        panel_rows = [row for row in visibility_rows if _panel(row) == panel]
        memberships = defaultdict(list)
        for row in panel_rows:
            index = min(int(np.searchsorted(edges, row["visibility_score_mean"], side="right") - 1), bins - 1)
            memberships[index].append(row)
        for index in range(bins):
            binned.append({**common, "panel": panel, "bin_index": index,
                           "bin_left": float(edges[index]), "bin_right": float(edges[index + 1]),
                           "bin_center": float((edges[index] + edges[index + 1]) / 2),
                           "right_inclusive": index == bins - 1,
                           **_statistics(memberships[index], bootstrap, _seed(seed, [panel, "visibility", index])),
                           "dedup_policy": DEDUP_POLICY})
    info = {**common, "bootstrap_replicates": bootstrap, "seed": seed, "bins": bins,
            "ci_method": CI_METHOD, "identical_duplicates_removed": identical_duplicates,
            "visibility_observations_before": len(rows), "visibility_observations_after": len(visibility_rows),
            "visibility_scored_before": sum(row["status"] != "skipped" for row in rows),
            "visibility_scored_after": sum(row["status"] != "skipped" for row in visibility_rows),
            "k8_removed_duplicates": removed, "dedup_policy": DEDUP_POLICY,
            "interpretation": "Descriptive association between visibility and accuracy, not a causal estimate.",
            "missing_curve_cells": sum(row["n"] == 0 for row in curves)}
    return curves, binned, info


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figure(info: dict, title: str) -> tuple[Figure, np.ndarray]:
    figure = Figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, sharey=True)
    figure.subplots_adjust(left=0.075, right=0.98, bottom=0.245, top=0.69, wspace=0.16)
    if info["dry_run"]:
        figure.text(0.5, 0.955, "DRY RUN — RANDOM ANSWERS", ha="center", va="center",
                    fontsize=23, fontweight="bold", color="#AD342E",
                    bbox={"facecolor": "#FFF0ED", "edgecolor": "none", "pad": 9})
    else:
        figure.text(0.5, 0.955, str(info["model"]), ha="center", va="center", fontsize=15, color="#586373")
    figure.text(0.5, 0.865, title, ha="center", va="center", fontsize=25, fontweight="bold", color="#172535")
    for panel, axis in zip(PANELS, axes):
        axis.set_title(PANEL_LABELS[panel], fontsize=15, fontweight="bold", pad=16)
        axis.set_ylim(-2, 108)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.tick_params(axis="both", labelsize=13, length=0, pad=7)
        axis.grid(axis="y", color="#E5E9EF", linewidth=0.8)
        axis.axhline(25, color="#8691A0", linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
        axis.spines[["top", "right"]].set_visible(False)
        for name in ("left", "bottom"):
            axis.spines[name].set_color("#CBD2DA")
    axes[0].set_ylabel("Accuracy (%)", fontsize=15, labelpad=12)
    return figure, axes


def _draw_ci(axis, x: float, low: float | None, high: float | None, color: str,
             *, shared: bool = False) -> None:
    """Draw the exact percentile range with caps, independent of point location."""
    if low is None or high is None:
        return
    # A percentile interval need not contain its point estimate. Centering the
    # invisible errorbar marker at the interval midpoint avoids clipping or
    # changing the reported range to satisfy nonnegative yerr requirements.
    axis.errorbar([x], [(low + high) * 50], yerr=[(high - low) * 50], fmt="none",
                  ecolor=color, elinewidth=2.0 if shared else 1.5,
                  capsize=6 if shared else 5, capthick=1.6 if shared else 1.2,
                  alpha=0.9 if shared else 0.65, zorder=4 if shared else 2)


def _plot_curves(rows: list[dict], info: dict, path: Path) -> None:
    figure, axes = _figure(info, "View selection: accuracy with 95% CI")
    for panel, axis in zip(PANELS, axes):
        for strategy in STRATEGIES:
            data = [row for row in rows if row["panel"] == panel and row["strategy"] == strategy]
            x = np.array([row["k"] for row in data], dtype=float)
            y = np.array([100 * row["accuracy"] if row["accuracy"] is not None else np.nan for row in data])
            for row in data:
                if row["k"] != 8:
                    _draw_ci(axis, row["k"], row["ci_low"], row["ci_high"], COLORS[strategy])
            axis.plot(x, y, color=COLORS[strategy], marker=MARKERS[strategy], markersize=7,
                      markevery=[index for index, row in enumerate(data) if row["k"] != 8],
                      linewidth=2.5, label=STRATEGY_LABELS[strategy], zorder=3)
        shared = next((row for row in rows if row["panel"] == panel and row["k"] == 8 and row["accuracy"] is not None), None)
        if shared is not None:
            _draw_ci(axis, 8, shared["ci_low"], shared["ci_high"], "#172535", shared=True)
            axis.scatter([8], [shared["accuracy"] * 100], s=88, color="#172535", edgecolors="white", linewidth=1.3, zorder=5)
        axis.set_xscale("log", base=2)
        axis.set_xlim(0.85, 9.5)
        axis.set_xticks(VIEW_COUNTS, [str(value) for value in VIEW_COUNTS])
        axis.set_xlabel("Number of views", fontsize=15, labelpad=10)
        ns = sorted({row["n"] for row in rows if row["panel"] == panel and row["n"]})
        note = "No scored results" if not ns else (f"n = {ns[0]} per point" if len(ns) == 1 else f"n = {ns[0]}–{ns[-1]} per point")
        axis.text(0.5, -0.22, note, transform=axis.transAxes, ha="center", fontsize=12, color="#586373")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.805), ncol=4,
                  frameon=False, fontsize=13, columnspacing=1.7, handlelength=2.0)
    figure.text(0.5, 0.105, "Capped bars: 95% bootstrap CI by base structure  ·  Dashed line: 25% chance", ha="center", fontsize=12, color="#586373")
    figure.text(0.5, 0.065, "Eight views: one shared black point and CI; all strategies use the same answer.", ha="center", fontsize=12, color="#586373")
    figure.savefig(path, dpi=150, facecolor="white", metadata={"Description": json.dumps(info)})


def _plot_visibility(rows: list[dict], info: dict, path: Path) -> None:
    figure, axes = _figure(info, "Visibility and accuracy")
    figure.text(0.5, 0.795, "Binned association · not a causal estimate", ha="center", fontsize=15, color="#586373")
    for panel, axis in zip(PANELS, axes):
        data = [row for row in rows if row["panel"] == panel]
        x = np.array([row["bin_center"] for row in data])
        y = np.array([100 * row["accuracy"] if row["accuracy"] is not None else np.nan for row in data])
        axis.plot(x, y, color="#3868C8", linewidth=2.5, marker="o", markersize=8, zorder=3)
        for row in data:
            if row["accuracy"] is None:
                continue
            center = row["bin_center"]
            accuracy = row["accuracy"] * 100
            if row["ci_low"] is not None:
                low, high = row["ci_low"] * 100, row["ci_high"] * 100
                axis.vlines(center, low, high, colors="#3868C8", linewidth=1.9, zorder=2)
                axis.hlines([low, high], center - 0.018, center + 0.018, colors="#3868C8", linewidth=1.9, zorder=2)
            axis.annotate(f"n={row['n']}", (center, accuracy), xytext=(0, 12), textcoords="offset points",
                          ha="center", va="bottom", fontsize=11, fontweight="bold", color="#172535",
                          bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
        axis.set_xlim(-0.03, 1.03)
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".50", ".75", "1"])
        axis.set_xlabel("Normalized visibility score", fontsize=14, labelpad=10)
        axis.text(0.5, -0.22, f"n = {sum(row['n'] for row in data)} observations", transform=axis.transAxes,
                  ha="center", fontsize=12, color="#586373")
    figure.text(0.5, 0.105, "Bars: 95% bootstrap CI by base structure  ·  Dashed line: 25% chance", ha="center", fontsize=12, color="#586373")
    figure.text(0.5, 0.065, "k < 8: one observation per question × strategy × k.  k = 8: once per question.", ha="center", fontsize=12, color="#586373")
    figure.savefig(path, dpi=150, facecolor="white", metadata={"Description": json.dumps(info)})


def plot_ablation(result_path: str | Path, out_dir: str | Path = "figures", *, bootstrap: int = 2000,
                  seed: int = 20261021, bins: int = 5) -> dict[str, str]:
    """Read one result JSONL and write two 1920×1080 PNGs and their CSV tables."""
    records = load_results([Path(result_path)])
    curves, binned, info = build_plot_tables(records, bootstrap=bootstrap, seed=seed, bins=bins)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"ablation_views_png": directory / "ablation_views.png",
             "visibility_vs_accuracy_png": directory / "visibility_vs_accuracy.png",
             "ablation_views_csv": directory / "ablation_views.csv",
             "visibility_vs_accuracy_csv": directory / "visibility_vs_accuracy.csv",
             "plot_info_json": directory / "ablation_plot_info.json"}
    _write_csv(paths["ablation_views_csv"], curves)
    _write_csv(paths["visibility_vs_accuracy_csv"], binned)
    paths["plot_info_json"].write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_curves(curves, info, paths["ablation_views_png"])
    _plot_visibility(binned, info, paths["visibility_vs_accuracy_png"])
    return {key: str(path.resolve()) for key, path in paths.items()}


def _model_sort(model: str) -> tuple[int, str]:
    return MODEL_ORDER.get(model.lower(), 100), model


def build_multi_plot_tables(records_by_model: dict[str, Iterable[dict]], *, bootstrap: int = 2000,
                            seed: int = 20261021, bins: int = 5,
                            expected_per_type: int | dict[str, int] | None = None
                            ) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Aggregate complete real model runs, excluding incomplete model cohorts.

    A complete cohort has all three task groups, equally many base questions
    in each, and all 16 (strategy, k) combinations for every question. Optional
    expected_per_type may be a shared quota or a model-to-quota mapping. Errors
    and invalid responses remain incorrect; a run containing only request
    errors, or any skipped question, is excluded. Dry-run and baseline inputs
    are rejected, not silently mixed into real-model figures.
    """
    if not records_by_model:
        raise ValueError("No model runs were supplied.")
    if isinstance(expected_per_type, dict):
        quotas = dict(expected_per_type)
    elif expected_per_type is None:
        quotas = {}
    elif type(expected_per_type) is int:
        quotas = {model: expected_per_type for model in records_by_model}
    else:
        raise ValueError("expected_per_type must be a positive integer, mapping, or None.")
    if any(type(value) is not int or value < 1 for value in quotas.values()):
        raise ValueError("Every expected question quota must be a positive integer.")
    curves, binned, summary = [], [], []
    included, excluded, cohort_ids = {}, [], {}
    common_cells = {}
    required_cells = {(k, strategy) for k in VIEW_COUNTS for strategy in STRATEGIES}
    for model in sorted(records_by_model, key=_model_sort):
        original = list(records_by_model[model])
        if not original:
            excluded.append({"model": model, "reasons": ["empty result file"]})
            continue
        if any(row.get("dry_run") is not False for row in original):
            raise ValueError("Multi-model formal plots require explicit dry_run=false in every row.")
        if any(row.get("baseline") is not None for row in original):
            raise ValueError("Baseline answers cannot be included in formal multi-model plots.")
        rows, duplicate_count = _normalized_records(original)
        if any(row["model"] != model for row in rows):
            raise ValueError(f"The model key {model} disagrees with its result records.")
        if any(row["condition"] != "image" for row in rows):
            raise ValueError("Formal view ablations require condition=image.")
        # Invalid shared endpoints are data inconsistencies, not incomplete
        # jobs. Reject these even when another model can still be plotted.
        _visibility_observations(rows)
        question_rows = defaultdict(list)
        task_ids = {panel: set() for panel in PANELS}
        for row in rows:
            question_rows[row["base_question_id"]].append(row)
            task_ids[_panel(row)].add(row["base_question_id"])
        counts = {panel: len(ids) for panel, ids in task_ids.items()}
        reasons = []
        if not all(counts.values()) or len(set(counts.values())) != 1:
            reasons.append(f"three task groups must have equal nonzero question counts; found {counts}")
        quota = quotas.get(model)
        if quota is not None and any(count != quota for count in counts.values()):
            reasons.append(f"expected {quota} questions per task; found {counts}")
        incomplete = [qid for qid, members in question_rows.items()
                      if {(row["k"], row["strategy"]) for row in members} != required_cells]
        if incomplete:
            reasons.append(f"{len(incomplete)} questions are missing one or more of the 16 strategy/view cells")
        if any(row["status"] == "skipped" for row in rows):
            reasons.append("image ablation contains skipped questions")
        if not any(row["status"] in {"ok", "invalid_response"} for row in rows):
            reasons.append("no completed model responses; all requests failed")
        if reasons:
            excluded.append({"model": model, "reasons": reasons, "questions_by_task": counts,
                             "records": len(rows), "expected_per_type": quota})
            continue
        # Different quotas are allowed, but a question shared by two models
        # must still refer to the same structures, views and visibility score.
        for row in rows:
            key = (row["base_question_id"], row["k"], row["strategy"])
            signature = (_panel(row), row["base_structure"], tuple(row["selected_view_ids"]), row.get("answer"))
            previous = common_cells.get(key)
            if previous is not None and (previous[0] != signature or not np.isclose(
                    previous[1], row["visibility_score_mean"], atol=1e-12, rtol=0)):
                raise ValueError(f"Shared question geometry/answer differs across models for {key}.")
            common_cells[key] = (signature, row["visibility_score_mean"])
        model_curves, model_bins, model_info = build_plot_tables(
            rows, bootstrap=bootstrap, seed=seed, bins=bins)
        model_info.update(questions_by_task=counts, base_questions=len(question_rows),
                          expected_per_type=quota, identical_duplicates_removed=duplicate_count)
        included[model] = model_info
        cohort_ids[model] = set(question_rows)
        curves.extend(model_curves)
        binned.extend(model_bins)
        common = {key: model_info[key] for key in ("model", "provider", "condition", "dry_run",
                                                  "dataset_fingerprint", "config_fingerprint")}
        for strategy in ("max_visibility", "min_visibility"):
            for k in VIEW_COUNTS:
                members = [row for row in rows if row["strategy"] == strategy and row["k"] == k]
                identity = ["all_tasks", "shared" if k == 8 else strategy, k]
                summary.append({**common, "panel": "ALL", "strategy": strategy, "k": k,
                                "k8_shared": k == 8,
                                **_statistics(members, bootstrap, _seed(seed, identity))})
    if not included:
        raise ValueError("No complete real model runs are available: " + json.dumps(excluded, ensure_ascii=False))
    info = {"models": list(included), "runs": included, "excluded_runs": excluded,
            "dry_run": False, "bootstrap_replicates": bootstrap, "seed": seed, "bins": bins,
            "ci_method": CI_METHOD, "dedup_policy": DEDUP_POLICY,
            "shared_base_questions_across_models": len(set.intersection(*cohort_ids.values())),
            "cohort_policy": "Each model uses its complete, task-balanced cohort; model quotas may differ. Shared questions use identical views.",
            "summary_accuracy": "Question-weighted accuracy pooling Q1a, Q2 hard and Q3; resample base_structure clusters across all three tasks together.",
            "summary_k8_policy": "Each strategy has one record per question; the common endpoint and confidence interval are drawn only once per model.",
            "interpretation": "Descriptive results; differing model cohorts are reported with their sample sizes and are not paired model comparisons."}
    return curves, binned, summary, info


def _multi_figure(models: list[str], title: str, *, subtitle: str = "") -> tuple[Figure, np.ndarray]:
    count = len(models)
    height = 3.25 * count + 2.9
    figure = Figure(figsize=(15.2, height), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(count, 3, squeeze=False, sharey=True)
    figure.subplots_adjust(left=0.095, right=0.985, bottom=1.35 / height,
                           top=1 - 1.45 / height, hspace=0.52, wspace=0.13)
    figure.text(0.5, 1 - 0.28 / height, title, ha="center", va="center", fontsize=24,
                fontweight="bold", color="#172535")
    if subtitle:
        figure.text(0.5, 1 - 0.67 / height, subtitle, ha="center", fontsize=13, color="#586373")
    for row_index, model in enumerate(models):
        for column, (panel, axis) in enumerate(zip(PANELS, axes[row_index])):
            if row_index == 0:
                axis.set_title(PANEL_LABELS[panel], fontsize=15, fontweight="bold", pad=12)
            if column == 0:
                axis.set_ylabel(f"{model.upper()}\nAccuracy (%)", fontsize=15, fontweight="bold", labelpad=10)
            axis.set_ylim(-2, 108)
            axis.set_yticks([0, 25, 50, 75, 100])
            axis.tick_params(axis="both", labelsize=12, length=0, pad=7)
            axis.grid(axis="y", color="#E5E9EF", linewidth=0.8)
            axis.axhline(25, color="#8691A0", linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
            axis.spines[["top", "right"]].set_visible(False)
            for side in ("left", "bottom"):
                axis.spines[side].set_color("#CBD2DA")
    return figure, axes


def _plot_multi_curves(rows: list[dict], info: dict, path: Path) -> None:
    figure, axes = _multi_figure(info["models"], "View selection: real-model accuracy with 95% CI")
    for model_index, model in enumerate(info["models"]):
        for panel, axis in zip(PANELS, axes[model_index]):
            for strategy in STRATEGIES:
                data = [row for row in rows if row["model"] == model and row["panel"] == panel and row["strategy"] == strategy]
                for row in data:
                    if row["k"] != 8:
                        _draw_ci(axis, row["k"], row["ci_low"], row["ci_high"], COLORS[strategy])
                axis.plot([row["k"] for row in data], [row["accuracy"] * 100 for row in data],
                          color=COLORS[strategy], marker=MARKERS[strategy], markevery=[0, 1, 2],
                          markersize=6, linewidth=2.4, label=STRATEGY_LABELS[strategy], zorder=3)
            shared = next(row for row in rows if row["model"] == model and row["panel"] == panel and row["k"] == 8)
            _draw_ci(axis, 8, shared["ci_low"], shared["ci_high"], "#172535", shared=True)
            axis.scatter([8], [shared["accuracy"] * 100], s=65, color="#172535", edgecolors="white", linewidth=1, zorder=5)
            axis.set_xscale("log", base=2)
            axis.set_xlim(0.85, 9.5)
            axis.set_xticks(VIEW_COUNTS, [str(value) for value in VIEW_COUNTS])
            axis.set_xlabel(f"Number of views  ·  n = {shared['n']} per point", fontsize=12, labelpad=8)
    height = figure.get_figheight()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.54 / height),
                  ncol=4, frameon=False, fontsize=13, columnspacing=1.8)
    figure.text(0.5, 0.42 / height, "Capped bars: 95% CI by base structure  ·  Dashed horizontal line: 25% chance", ha="center", fontsize=12, color="#586373")
    figure.text(0.5, 0.15 / height, "Eight views: one shared black point and CI per model/task. Only complete real-model runs are shown.", ha="center", fontsize=11, color="#586373")
    figure.savefig(path, dpi=150, facecolor="white", metadata={"Description": json.dumps(info)})


def _plot_multi_visibility(rows: list[dict], info: dict, path: Path) -> None:
    figure, axes = _multi_figure(info["models"], "Visibility and real-model accuracy",
                               subtitle="Binned association; not a causal estimate")
    for model_index, model in enumerate(info["models"]):
        for panel, axis in zip(PANELS, axes[model_index]):
            data = [row for row in rows if row["model"] == model and row["panel"] == panel]
            axis.plot([row["bin_center"] for row in data],
                      [np.nan if row["accuracy"] is None else row["accuracy"] * 100 for row in data],
                      color="#3868C8", marker="o", markersize=6, linewidth=2.2, zorder=3)
            for row in data:
                if row["accuracy"] is None:
                    continue
                _draw_ci(axis, row["bin_center"], row["ci_low"], row["ci_high"], "#3868C8")
                axis.annotate(f"n={row['n']}", (row["bin_center"], row["accuracy"] * 100),
                              xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
                              bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
            axis.set_xlim(-0.03, 1.03)
            axis.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", ".25", ".50", ".75", "1"])
            axis.set_xlabel("Normalized visibility score", fontsize=12, labelpad=8)
    height = figure.get_figheight()
    figure.text(0.5, 0.42 / height, "Capped bars: 95% CI by base structure  ·  Dashed horizontal line: 25% chance", ha="center", fontsize=12, color="#586373")
    figure.text(0.5, 0.15 / height, "k < 8: one observation per question × strategy × k.  k = 8: once per question.", ha="center", fontsize=11, color="#586373")
    figure.savefig(path, dpi=150, facecolor="white", metadata={"Description": json.dumps(info)})


def _plot_multi_summary(rows: list[dict], info: dict, path: Path) -> None:
    figure = Figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    figure.subplots_adjust(left=0.11, right=0.96, bottom=0.23, top=0.71)
    figure.text(0.5, 0.93, "Visibility selection across models", ha="center", fontsize=25,
                fontweight="bold", color="#172535")
    figure.text(0.5, 0.875, "Pooled Q1a, Q2 hard and Q3 · complete real-model runs", ha="center", fontsize=15, color="#586373")
    handles = []
    fallback_colors = ["#3868C8", "#C56A21", "#087F70", "#9054A7"]
    for index, model in enumerate(info["models"]):
        color = MODEL_COLORS.get(model.lower(), fallback_colors[index % len(fallback_colors)])
        for strategy, style in (("max_visibility", "-"), ("min_visibility", "--")):
            data = [row for row in rows if row["model"] == model and row["strategy"] == strategy]
            for row in data:
                if row["k"] != 8:
                    _draw_ci(axis, row["k"], row["ci_low"], row["ci_high"], color)
            axis.plot([row["k"] for row in data], [row["accuracy"] * 100 for row in data],
                      color=color, linestyle=style, marker="o" if strategy == "max_visibility" else "s",
                      markevery=[0, 1, 2], markersize=7, linewidth=2.7, zorder=3)
        shared = next(row for row in rows if row["model"] == model and row["k"] == 8)
        _draw_ci(axis, 8, shared["ci_low"], shared["ci_high"], color, shared=True)
        axis.scatter([8], [shared["accuracy"] * 100], s=88, color=color, edgecolors="white", linewidth=1, zorder=5)
        handles.append(Line2D([0], [0], color=color, linewidth=3, label=f"{model.upper()} (n={shared['n']})"))
    handles += [Line2D([0], [0], color="#172535", linewidth=2.5, linestyle="-", label="Max visibility"),
                Line2D([0], [0], color="#172535", linewidth=2.5, linestyle="--", label="Min visibility")]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.83), ncol=len(handles),
                  frameon=False, fontsize=12, columnspacing=1.4, handlelength=2.6)
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.85, 9.5)
    axis.set_xticks(VIEW_COUNTS, [str(value) for value in VIEW_COUNTS])
    axis.set_ylim(-2, 105)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.tick_params(labelsize=14, length=0, pad=8)
    axis.set_xlabel("Number of views", fontsize=16, labelpad=12)
    axis.set_ylabel("Accuracy (%)", fontsize=16, labelpad=12)
    axis.grid(axis="y", color="#E5E9EF", linewidth=0.8)
    axis.axhline(25, color="#8691A0", linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#CBD2DA")
    figure.text(0.5, 0.10, "Capped bars: 95% bootstrap CI by base structure across all three tasks", ha="center", fontsize=12, color="#586373")
    figure.text(0.5, 0.055, "Eight-view answers and CIs are shared within each model. Model cohorts may differ; n is shown above.", ha="center", fontsize=11, color="#586373")
    figure.savefig(path, dpi=150, facecolor="white", metadata={"Description": json.dumps(info)})


def plot_multi_ablation(result_paths: Iterable[str | Path], out_dir: str | Path = "figures", *,
                         bootstrap: int = 2000, seed: int = 20261021, bins: int = 5,
                         expected_per_type: int | dict[str, int] | None = None) -> dict[str, str]:
    """Plot one or more complete real runs; never invent rows for missing models.

    With no explicit quota, a sibling .summary.json can supply the expected
    full_planned_base_questions / 3. If it is absent, balanced actual cohorts
    with complete 16-cell question grids are accepted. Different model quotas
    are permitted and sample sizes remain visible in every output.
    """
    if isinstance(result_paths, (str, Path)):
        result_paths = [result_paths]
    paths = [Path(path) for path in result_paths]
    records_by_model = defaultdict(list)
    inferred_quotas = {}
    source_paths = defaultdict(list)
    unavailable = []
    for path in paths:
        if not path.exists():
            unavailable.append({"source_path": str(path.resolve()), "reasons": ["result file not found"]})
            continue
        records = load_results([path])
        if not records:
            unavailable.append({"source_path": str(path.resolve()), "reasons": ["empty result file"]})
            continue
        models = {row["model"] for row in records}
        if len(models) != 1:
            raise ValueError(f"Each input must contain exactly one model run: {path}.")
        model = next(iter(models))
        records_by_model[model].extend(records)
        source_paths[model].append(str(path.resolve()))
        sidecar = path.with_suffix(".summary.json")
        if expected_per_type is None and sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            planned = metadata.get("full_planned_base_questions")
            if type(planned) is int and planned > 0 and planned % 3 == 0:
                quota = planned // 3
                if model in inferred_quotas and inferred_quotas[model] != quota:
                    raise ValueError(f"Conflicting planned question counts for {model}.")
                inferred_quotas[model] = quota
    if not records_by_model:
        raise ValueError("No complete real model runs are available: " + json.dumps(unavailable, ensure_ascii=False))
    quota_argument = inferred_quotas if expected_per_type is None else expected_per_type
    curves, binned, summary, info = build_multi_plot_tables(
        dict(records_by_model), bootstrap=bootstrap, seed=seed, bins=bins, expected_per_type=quota_argument)
    info["source_paths"] = dict(source_paths)
    info["excluded_runs"].extend(unavailable)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_paths = {"ablation_views_png": directory / "ablation_views.png",
                    "ablation_summary_png": directory / "ablation_summary.png",
                    "visibility_vs_accuracy_png": directory / "visibility_vs_accuracy.png",
                    "ablation_views_csv": directory / "ablation_views.csv",
                    "ablation_summary_csv": directory / "ablation_summary.csv",
                    "visibility_vs_accuracy_csv": directory / "visibility_vs_accuracy.csv",
                    "plot_info_json": directory / "ablation_plot_info.json"}
    _write_csv(output_paths["ablation_views_csv"], curves)
    _write_csv(output_paths["ablation_summary_csv"], summary)
    _write_csv(output_paths["visibility_vs_accuracy_csv"], binned)
    output_paths["plot_info_json"].write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _plot_multi_curves(curves, info, output_paths["ablation_views_png"])
    _plot_multi_summary(summary, info, output_paths["ablation_summary_png"])
    _plot_multi_visibility(binned, info, output_paths["visibility_vs_accuracy_png"])
    return {key: str(path.resolve()) for key, path in output_paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, nargs="+", help="One or more results/ablation_MODEL.jsonl files")
    parser.add_argument("--multi", action="store_true", help="Use formal multi-model output even for a single successful model")
    parser.add_argument("--expected-per-type", type=int, help="Optional shared per-task quota for formal multi-model validation")
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()
    if args.multi or len(args.input) > 1:
        output = plot_multi_ablation(args.input, args.output_dir, bootstrap=args.bootstrap,
                                    seed=args.seed, bins=args.bins, expected_per_type=args.expected_per_type)
    else:
        output = plot_ablation(args.input[0], args.output_dir, bootstrap=args.bootstrap,
                               seed=args.seed, bins=args.bins)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
