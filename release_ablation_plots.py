"""Refresh two release figures from existing CSV statistics, without evaluation.

Only ablation_summary.png, ablation_q3.png and ablation_release_audit.json are
written. All point estimates and confidence intervals come directly from the
accepted CSVs; no bootstrap, model request, or dataset mutation is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np


COUNTS = (1, 2, 4, 8)
STRATEGIES = ("random", "canonical", "max_visibility", "min_visibility")
COLORS = {"random": "#172535", "canonical": "#3868C8", "max_visibility": "#087F70", "min_visibility": "#C24D45"}
MARKERS = {"random": "o", "canonical": "s", "max_visibility": "^", "min_visibility": "D"}
LABELS = {"random": "Random (emphasized)", "canonical": "Canonical", "max_visibility": "Max visibility", "min_visibility": "Min visibility"}
MODEL_COLORS = {"gpt": "#3868C8", "claude": "#C56A21"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_csv(path: Path, panel: str, strategies: tuple[str, ...]) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw = [row for row in csv.DictReader(handle) if row["panel"] == panel]
    if not raw:
        raise ValueError(f"No {panel} rows in {path}.")
    rows = []
    identities = set()
    for source in raw:
        if source["dry_run"].lower() != "false" or source["condition"] != "image":
            raise ValueError("Release figures require real image-condition results.")
        row = dict(source)
        row["k"], row["n"] = int(row["k"]), int(row["n"])
        for key in ("accuracy", "ci_low", "ci_high"):
            row[key] = float(row[key]) if row[key] else None
        if row["accuracy"] is None or not np.isfinite(row["accuracy"]) or not 0 <= row["accuracy"] <= 1:
            raise ValueError("Every plotted cell requires an accuracy in [0, 1].")
        if row["ci_low"] is None or row["ci_high"] is None or not 0 <= row["ci_low"] <= row["ci_high"] <= 1:
            raise ValueError("Release figures require complete existing confidence intervals.")
        key = (row["model"], row["strategy"], row["k"])
        if key in identities or row["n"] <= 0:
            raise ValueError("Duplicate or empty CSV cell.")
        identities.add(key)
        rows.append(row)
    for model in {row["model"] for row in rows}:
        selected = [row for row in rows if row["model"] == model]
        cells = {(row["strategy"], row["k"]) for row in selected}
        if cells != {(strategy, k) for strategy in strategies for k in COUNTS}:
            raise ValueError(f"Incomplete or unexpected strategy grid for {model}/{panel}.")
        if len({row["n"] for row in selected}) != 1:
            raise ValueError(f"Question counts vary across {model}/{panel} cells.")
        shared = [row for row in selected if row["k"] == 8]
        reference = shared[0]
        for row in shared[1:]:
            if any(not np.isclose(row[key], reference[key], atol=1e-12, rtol=0)
                   for key in ("accuracy", "ci_low", "ci_high")):
                raise ValueError(f"Eight-view endpoints differ across strategies for {model}.")
    return rows


def _interval(axis, row: dict, color: str, *, emphasis=False) -> None:
    low, high = row["ci_low"] * 100, row["ci_high"] * 100
    axis.errorbar([row["k"]], [(low + high) / 2], yerr=[(high - low) / 2], fmt="none",
                  ecolor=color, elinewidth=2.1 if emphasis else 1.4, capsize=6 if emphasis else 4,
                  capthick=1.6 if emphasis else 1.2, alpha=0.86 if emphasis else 0.48,
                  zorder=4 if emphasis else 2)


def _axes_style(axis, minimum: float) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.85, 9.5)
    axis.set_xticks(COUNTS, [str(k) for k in COUNTS])
    axis.set_ylim(minimum, 100)
    axis.set_yticks(np.arange(minimum, 101, 10))
    axis.tick_params(labelsize=15, length=0, pad=8)
    axis.set_xlabel("Number of views", fontsize=17, labelpad=12)
    axis.grid(axis="y", color="#E5E9EF", linewidth=0.9)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#CBD2DA")


def _summary(rows: list[dict], models: list[str], path: Path) -> None:
    figure = Figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    figure.subplots_adjust(left=0.11, right=0.96, bottom=0.23, top=0.71)
    figure.text(0.5, 0.93, "Visibility selection across models", ha="center", fontsize=25,
                fontweight="bold", color="#172535")
    figure.text(0.5, 0.875, "Pooled Q1a, Q2 hard and Q3 · complete real-model runs", ha="center", fontsize=15, color="#586373")
    handles = []
    for model in models:
        color = MODEL_COLORS[model]
        for strategy, style in (("max_visibility", "-"), ("min_visibility", "--")):
            data = sorted([row for row in rows if row["model"] == model and row["strategy"] == strategy], key=lambda row: row["k"])
            for row in data:
                if row["k"] != 8:
                    _interval(axis, row, color)
            axis.plot([row["k"] for row in data], [row["accuracy"] * 100 for row in data],
                      color=color, linestyle=style, marker="o" if strategy == "max_visibility" else "s",
                      markevery=[0, 1, 2], markersize=8, linewidth=3, zorder=3)
        shared = next(row for row in rows if row["model"] == model and row["k"] == 8)
        _interval(axis, shared, color, emphasis=True)
        axis.scatter([8], [shared["accuracy"] * 100], s=100, color=color, edgecolors="white", linewidth=1.2, zorder=5)
        handles.append(Line2D([0], [0], color=color, linewidth=3, label=f"{model.upper()} (n={shared['n']})"))
    handles.extend([Line2D([0], [0], color="#172535", linewidth=2.8, linestyle="-", label="Max visibility"),
                    Line2D([0], [0], color="#172535", linewidth=2.8, linestyle="--", label="Min visibility")])
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.83), ncol=4,
                  frameon=False, fontsize=13, columnspacing=1.5, handlelength=2.6)
    _axes_style(axis, 60)
    axis.set_ylabel("Accuracy (%)", fontsize=17, labelpad=12)
    figure.text(0.5, 0.10, "Y-axis: 60–100% · Capped bars: existing 95% bootstrap CI by base structure", ha="center", fontsize=12, color="#586373")
    figure.text(0.5, 0.055, "Eight-view answers and CIs are shared within each model; each endpoint is drawn once.", ha="center", fontsize=12, color="#586373")
    figure.savefig(path, dpi=150, facecolor="white")


def _q3(rows: list[dict], models: list[str], path: Path) -> dict:
    if len(models) != 2:
        raise ValueError("This release layout requires the two completed GPT and Claude runs.")
    figure = Figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2, sharey=True)
    figure.subplots_adjust(left=0.085, right=0.975, bottom=0.30, top=0.70, wspace=0.16)
    figure.text(0.5, 0.94, "Q3: relative distance across views", ha="center", fontsize=27,
                fontweight="bold", color="#172535")
    figure.text(0.5, 0.885, "Random-view curves emphasized · common eight-view endpoints", ha="center", fontsize=15, color="#586373")
    handles = [Line2D([0], [0], color=COLORS[strategy], marker=MARKERS[strategy],
                      linewidth=3.6 if strategy == "random" else 2, markersize=7,
                      label=LABELS[strategy]) for strategy in STRATEGIES]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.84), ncol=4,
                  frameon=False, fontsize=12, columnspacing=1.6, handlelength=2.2)
    metrics = {}
    for model, axis in zip(models, axes):
        for strategy in ("canonical", "max_visibility", "min_visibility", "random"):
            emphasized = strategy == "random"
            data = sorted([row for row in rows if row["model"] == model and row["strategy"] == strategy], key=lambda row: row["k"])
            for row in data:
                if row["k"] != 8:
                    _interval(axis, row, COLORS[strategy], emphasis=emphasized)
            axis.plot([row["k"] for row in data], [row["accuracy"] * 100 for row in data],
                      color=COLORS[strategy], marker=MARKERS[strategy], markevery=[0, 1, 2],
                      linewidth=3.6 if emphasized else 1.8, markersize=8 if emphasized else 6,
                      alpha=1 if emphasized else 0.65, zorder=6 if emphasized else 3)
        shared = next(row for row in rows if row["model"] == model and row["k"] == 8)
        first = next(row for row in rows if row["model"] == model and row["k"] == 1 and row["strategy"] == "random")
        _interval(axis, shared, COLORS["random"], emphasis=True)
        axis.scatter([8], [shared["accuracy"] * 100], s=110, color=COLORS["random"], edgecolors="white", linewidth=1.3, zorder=7)
        axis.set_title(f"{model.upper()} · n = {shared['n']}", fontsize=19, fontweight="bold", pad=15)
        _axes_style(axis, 50)
        center = (axis.get_position().x0 + axis.get_position().x1) / 2
        gain = 100 * (shared["accuracy"] - first["accuracy"])
        figure.text(center, 0.155, f"{gain:+.1f} pp", ha="center", fontsize=21, fontweight="bold", color="#172535")
        figure.text(center, 0.115, f"1 random view: {first['accuracy'] * 100:.1f}%  →  8 views: {shared['accuracy'] * 100:.1f}%",
                    ha="center", fontsize=12, color="#586373")
        metrics[model] = {"n": shared["n"], "k1_random_accuracy": first["accuracy"],
                          "k8_accuracy": shared["accuracy"], "gain_pp": gain,
                          "gain_interpretation": "Observed point-estimate difference; no difference CI is inferred from marginal error bars."}
    axes[0].set_ylabel("Accuracy (%)", fontsize=17, labelpad=12)
    figure.text(0.5, 0.055, "Bars: existing 95% cluster-bootstrap CI by base structure · Gains are point estimates", ha="center", fontsize=12, color="#586373")
    figure.savefig(path, dpi=150, facecolor="white")
    return metrics


def plot_release_figures(figures_dir: str | Path = "figures") -> dict:
    """Write only the two requested PNGs and a source/output checksum audit."""
    directory = Path(figures_dir)
    summary_source = directory / "ablation_summary.csv"
    views_source = directory / "ablation_views.csv"
    protected_names = ("ablation_summary.csv", "ablation_views.csv", "visibility_vs_accuracy.csv",
                       "ablation_views.png", "visibility_vs_accuracy.png", "ablation_plot_info.json")
    protected = {name: _sha256(directory / name) for name in protected_names if (directory / name).exists()}
    summary_rows = _load_csv(summary_source, "ALL", ("max_visibility", "min_visibility"))
    q3_rows = _load_csv(views_source, "Q3", STRATEGIES)
    models = [model for model in ("gpt", "claude") if model in {row["model"] for row in summary_rows}]
    if set(models) != {"gpt", "claude"} or {row["model"] for row in q3_rows} != set(models):
        raise ValueError("Release sources must contain the completed GPT and Claude runs.")
    summary_path = directory / "ablation_summary.png"
    q3_path = directory / "ablation_q3.png"
    previous_summary_sha = _sha256(summary_path) if summary_path.exists() else None
    _summary(summary_rows, models, summary_path)
    metrics = _q3(q3_rows, models, q3_path)
    after = {name: _sha256(directory / name) for name in protected}
    if after != protected:
        raise RuntimeError("A protected source/figure changed during release plotting.")
    audit = {"models": models, "source_csv_sha256": {name: protected[name] for name in ("ablation_summary.csv", "ablation_views.csv")},
             "protected_files_sha256": protected, "protected_files_unchanged": True,
             "statistics_recomputed": False, "api_calls": False, "data_results_written": False,
             "resolution_pixels": [1920, 1080], "summary_ylim_percent": [60, 100],
             "q3_ylim_percent": [50, 100], "q3_gain_metrics": metrics,
             "previous_summary_png_sha256": previous_summary_sha,
             "output_png_sha256": {summary_path.name: _sha256(summary_path), q3_path.name: _sha256(q3_path)}}
    audit_path = directory / "ablation_release_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    return {"ablation_summary_png": str(summary_path.resolve()), "ablation_q3_png": str(q3_path.resolve()),
            "audit_json": str(audit_path.resolve()), "q3_gain_metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    print(json.dumps(plot_release_figures(args.figures_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
