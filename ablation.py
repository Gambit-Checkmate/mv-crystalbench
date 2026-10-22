"""Geometry-only view ablations with deduplicated, resumable evaluation.

The selectors never receive answers. Every original question gets 4 strategies
at k=1,2,4,8; identical view subsets share one evaluate_questions response.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
from ase import Atoms

from bonding import display_bonds
from evaluate import evaluate_questions, load_model_config, load_questions, normalize_condition


STRATEGIES = ("random", "canonical", "max_visibility", "min_visibility")
VIEW_COUNTS = (1, 2, 4, 8)
TASKS = ("Q1a", "Q2hard", "Q3")
SELECTOR_VERSION = "display-union-visible-greedy-v2-independent-random-k"
DEFAULT_SEED = 20261021


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _seed(*values) -> int:
    return int(_digest(values)[:16], 16)


def task_name(question: dict) -> str | None:
    if question["qtype"] == "Q2":
        return "Q2hard" if question["difficulty"] == "hard" else None
    return question["qtype"] if question["qtype"] in {"Q1a", "Q3"} else None


def select_questions(questions: list[dict], per_type: int = 40, seed: int = DEFAULT_SEED,
                     limit: int | None = None) -> list[dict]:
    """Pick diverse bases without inspecting answer text or answer letters.

    Each task is shuffled reproducibly, then bases with fewer selected questions
    are preferred. Tasks are interleaved, so --limit 24 gives 8 questions each.
    """
    if isinstance(per_type, bool) or not isinstance(per_type, int) or per_type < 1:
        raise ValueError("per_type must be positive.")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be positive.")
    chosen = {}
    for task in TASKS:
        pool = [question for question in questions if task_name(question) == task]
        if len(pool) < per_type:
            raise ValueError(f"{task} has {len(pool)} questions; need {per_type}.")
        random.Random(_seed(seed, task, "question-subset")).shuffle(pool)
        coverage = Counter()
        chosen[task] = []
        for _ in range(per_type):
            index = min(range(len(pool)), key=lambda i: (
                coverage[pool[i]["meta"]["base_structure"]], i,
            ))
            question = pool.pop(index)
            chosen[task].append(question)
            coverage[question["meta"]["base_structure"]] += 1
    interleaved = [chosen[task][index] for index in range(per_type) for task in TASKS]
    return interleaved if limit is None else interleaved[:limit]


def load_scenes(path: str | Path) -> dict[str, dict]:
    scenes = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        scene = json.loads(line)
        if scene["scene_id"] in scenes:
            raise ValueError("Duplicate scene ID in manifest.")
        scenes[scene["scene_id"]] = scene
    return scenes


def _scene_views(scene: dict) -> list[dict]:
    lookup = {view["view_id"]: view for view in scene["views"]}
    if len(scene["views"]) != 8 or len(lookup) != 8 or set(lookup) != {f"v{i:02d}" for i in range(8)}:
        raise ValueError("Each scene needs exactly the candidate views v00-v07.")
    return [lookup[f"v{i:02d}"] for i in range(8)]


def relevant_atom_indices(question: dict, scene: dict) -> list[int]:
    """Locate task atoms from element identities, highlights, and geometry.

    Q2 neighbors are recomputed from original periodic pair cutoffs and displayed
    Cartesian positions, then checked against the scene bond list and meta IDs.
    The stored coordination_number and all answer fields are never accessed.
    """
    display = scene["display_atoms"]
    if [atom["display_index"] for atom in display] != list(range(len(display))):
        raise ValueError("Display atom indices must be contiguous.")
    meta = question["meta"]
    labels = {atom["highlight"]: atom["display_index"]
              for atom in display if atom.get("highlight")}
    task = task_name(question)
    if task == "Q1a":
        indices = [atom["display_index"] for atom in display if atom["element"] == meta["element"]]
    elif task == "Q2hard":
        if "A" not in labels:
            raise ValueError("Q2 requires a displayed atom labeled A.")
        center = labels["A"]
        if meta.get("center_display_index", center) != center:
            raise ValueError("Q2 center metadata disagrees with the displayed label.")
        source = scene["atoms"]
        original = Atoms(numbers=source["atomic_numbers"], positions=source["positions_angstrom"],
                         cell=source["lattice_vectors_angstrom"], pbc=source["periodic"])
        original.info["bond_pairs"] = source["bond_pairs"]
        shown = Atoms(symbols=[atom["element"] for atom in display],
                      positions=[atom["position_angstrom"] for atom in display], pbc=False)
        pairs = {(i, j) for i, j, _ in display_bonds(original, shown)}
        recorded = {tuple(sorted(bond["display_indices"]))
                    for bond in _scene_views(scene)[0]["metadata"]["bonds"]}
        if pairs != recorded:
            raise ValueError("The scene bond graph disagrees with recomputed geometry.")
        neighbors = {j if i == center else i for i, j in pairs if center in (i, j)}
        if neighbors != set(meta["neighbor_display_indices"]):
            raise ValueError("Q2 neighbor metadata disagrees with recomputed displayed bonds.")
        indices = [center, *sorted(neighbors)]
    elif task == "Q3":
        if any(label not in labels for label in "ABC"):
            raise ValueError("Q3 requires displayed labels A, B, and C.")
        indices = [labels[label] for label in "ABC"]
        recorded = meta.get("display_indices")
        if recorded is not None and any(recorded[label] != labels[label] for label in "ABC"):
            raise ValueError("Q3 highlight metadata disagrees with the scene.")
    else:
        raise ValueError("Only Q1a, Q2 hard, and Q3 belong to this ablation.")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Relevant atom indices must be nonempty and unique.")
    return list(map(int, indices))


def visibility_matrix(scene: dict, relevant: list[int]) -> np.ndarray:
    """Return 8 x N atom-visible fractions for a fixed, task-relevant atom set."""
    matrix = np.array([
        [view["metadata"]["atoms"][index]["visible_fraction"] for index in relevant]
        for view in _scene_views(scene)
    ], dtype=float)
    _validate_visibility(matrix)
    return matrix


def _validate_visibility(visibility) -> np.ndarray:
    matrix = np.asarray(visibility, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != 8 or matrix.shape[1] < 1:
        raise ValueError("Visibility must be an 8 x N matrix with N >= 1.")
    if not np.isfinite(matrix).all() or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("Visible fractions must be finite and within [0, 1].")
    return matrix


def visibility_score(visibility, selected: list[int]) -> dict[str, float | int]:
    """Per-atom union proxy: sum_i max(visible_i in selected views)."""
    matrix = _validate_visibility(visibility)
    if (len(set(selected)) != len(selected) or
            any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
                or index < 0 or index >= 8 for index in selected)):
        raise ValueError("Selected view indices must be distinct indices in [0, 8).")
    union = np.max(matrix[selected], axis=0) if selected else np.zeros(matrix.shape[1])
    return {"sum": float(union.sum()), "mean": float(union.mean()), "atom_count": matrix.shape[1]}


def select_view_indices(visibility, k: int, strategy: str,
                        seed: int = DEFAULT_SEED) -> list[int]:
    """Pure selector; no question, structure identity, or answer is available.

    Max/min visibility greedily optimize the updated union score at every step.
    Ties choose the smallest candidate index. Random draws a separate deterministic
    sample for each k from the question-specific seed; cross-k samples are not
    constrained to nest. Canonical prioritizes v00/v01/v02, then v03-07.
    """
    matrix = _validate_visibility(visibility)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k not in VIEW_COUNTS:
        raise ValueError("k must be one of 1, 2, 4, 8.")
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy == "canonical":
        return list(range(k))
    if strategy == "random":
        draw_seed = _seed(int(seed), int(k), "independent-random-view-subset")
        return list(map(int, np.random.default_rng(draw_seed).choice(8, size=k, replace=False)))
    selected = []
    covered = np.zeros(matrix.shape[1])
    for _ in range(k):
        remaining = [index for index in range(8) if index not in selected]
        scores = {index: float(np.maximum(covered, matrix[index]).sum()) for index in remaining}
        choice = min(remaining, key=lambda index: (
            -scores[index] if strategy == "max_visibility" else scores[index], index,
        ))
        selected.append(choice)
        covered = np.maximum(covered, matrix[choice])
    return selected


def _adaptive_question_text(text: str) -> str:
    text = re.sub(r"\bThe three images\b", "The supplied images", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthree views\b", "supplied views", text, flags=re.IGNORECASE)
    return text


def build_ablation_plan(questions: list[dict], scenes: dict[str, dict], seed: int = DEFAULT_SEED,
                        strategies=STRATEGIES, view_counts=VIEW_COUNTS) -> dict[str, Any]:
    """Build all variants and one evaluation input per distinct question/subset."""
    strategies, view_counts = tuple(strategies), tuple(view_counts)
    if not strategies or len(set(strategies)) != len(strategies) or any(s not in STRATEGIES for s in strategies):
        raise ValueError("Strategies must be a nonempty unique subset of the supported strategies.")
    if not view_counts or len(set(view_counts)) != len(view_counts) or any(k not in VIEW_COUNTS for k in view_counts):
        raise ValueError("View counts must be a nonempty unique subset of 1, 2, 4, 8.")
    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Duplicate base question IDs.")
    requests, variants, seen = [], [], {}
    for source in questions:
        scene = scenes[source["scene_id"]]
        views = _scene_views(scene)
        atoms = relevant_atom_indices(source, scene)
        matrix = visibility_matrix(scene, atoms)
        qualified = [f"{scene['scene_id']}:{view['view_id']}" for view in views]
        question_seed = _seed(seed, source["id"], "view-sampling")
        for k in view_counts:
            for strategy in strategies:
                order = select_view_indices(matrix, k, strategy, question_seed)
                # Every shared subset is sent in the same order, so reuse never
                # assumes invariance to image order. The greedy order is retained.
                indices = sorted(order)
                score = visibility_score(matrix, indices)
                selected_ids = [qualified[index] for index in indices]
                key = (source["id"], tuple(indices))
                variant_id = f"{source['id']}::ablation::k{k}::{strategy}"
                descriptor = {
                    "base_question_id": source["id"], "question_id": variant_id,
                    "base_structure": source["meta"]["base_structure"], "k": k,
                    "strategy": strategy, "selected_view_ids": selected_ids,
                    "selection_order_view_ids": [qualified[index] for index in order],
                    "candidate_view_ids": qualified, "relevant_atom_indices": atoms,
                    "relevant_atom_count": len(atoms),
                    "visibility_score": score["mean"], "visibility_score_mean": score["mean"],
                    "visibility_score_sum": score["sum"], "selector_version": SELECTOR_VERSION,
                    "selection_input_fingerprint": _digest({"visibility": matrix.tolist(), "atoms": atoms}),
                }
                if key not in seen:
                    request_id = f"{source['id']}::views::{_digest(indices)[:12]}"
                    question = copy.deepcopy(source)
                    question.update(id=request_id, view_ids=selected_ids,
                                    image_paths=[views[index]["image_path"] for index in indices],
                                    question=_adaptive_question_text(source["question"]))
                    question["meta"]["ablation"] = {
                        "base_question_id": source["id"], "k": k,
                        "selected_view_ids": selected_ids, "relevant_atom_indices": atoms,
                        "visibility_score_sum": score["sum"], "visibility_score_mean": score["mean"],
                        "selector_version": SELECTOR_VERSION,
                    }
                    requests.append(question)
                    seen[key] = (request_id, variant_id)
                request_id, representative = seen[key]
                descriptor.update(request_question_id=request_id,
                                  reused_from=None if representative == variant_id else representative,
                                  api_request_reused=representative != variant_id)
                variants.append(descriptor)
    return {"requests": requests, "variants": variants,
            "base_question_ids": [question["id"] for question in questions],
            "strategies": list(strategies), "view_counts": list(view_counts), "seed": seed}


def expand_results(plan: dict, responses: list[dict], condition: str = "image",
                   selected_base_ids: set[str] | None = None) -> list[dict]:
    """Copy shared responses into variant rows, explicitly marking reused cost."""
    lookup = {(row["question_id"], row["condition"]): row for row in responses}
    expanded = []
    for descriptor in plan["variants"]:
        if selected_base_ids is not None and descriptor["base_question_id"] not in selected_base_ids:
            continue
        response = lookup.get((descriptor["request_question_id"], condition))
        if response is None:
            continue
        row = copy.deepcopy(response)
        row.update(descriptor)
        row["meta"]["ablation"] = copy.deepcopy(descriptor)
        row["charged_usage"] = None if descriptor["api_request_reused"] else response.get("usage")
        row["charged_attempts"] = 0 if descriptor["api_request_reused"] else response.get("attempts", 0)
        row["request_input_fingerprint"] = response.get("input_fingerprint")
        expanded.append(row)
    return expanded


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


async def run_ablation(questions: list[dict], scenes: dict[str, dict], config: dict,
                       output: str | Path, *, data_dir: str | Path, per_type: int = 40,
                       seed: int = DEFAULT_SEED, limit: int | None = None, dry_run: bool = False,
                       concurrency: int = 4, timeout: float = 120, max_retries: int = 5,
                       resume: bool = True, condition: str = "image", cache_path=None,
                       strategies=STRATEGIES, view_counts=VIEW_COUNTS, budget=None) -> tuple[list[dict], dict]:
    """Evaluate once per unique input. The same full plan supports limit expansion."""
    output = Path(output).expanduser().resolve()
    condition = normalize_condition(condition)
    selected = select_questions(questions, per_type=per_type, seed=seed)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be positive.")
    active = selected if limit is None else selected[:limit]
    active_ids = {question["id"] for question in active}
    plan = build_ablation_plan(selected, scenes, seed, strategies, view_counts)
    unique_limit = sum(q["meta"]["ablation"]["base_question_id"] in active_ids for q in plan["requests"])
    alias = str(config.get("alias", "dry-run"))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", alias):
        raise ValueError("Model aliases used in filenames may contain letters, digits, _, ., and - only.")
    cache = (Path(cache_path).expanduser().resolve() if cache_path else
             output.parent / ".ablation_cache" / f"{alias}.requests.jsonl")
    if cache == output:
        raise ValueError("Request cache and expanded result paths must be different.")
    if output.exists() and not cache.exists():
        raise ValueError("Expanded results exist without their request cache; choose a new output/cache path.")
    responses = await evaluate_questions(
        plan["requests"], config, cache, data_dir=data_dir, conditions=(condition,),
        dry_run=dry_run, concurrency=concurrency, timeout=timeout, max_retries=max_retries,
        resume=resume, seed=seed, limit=unique_limit, budget=budget,
    )
    expanded = expand_results(plan, responses, condition, active_ids)
    _write_jsonl(output, expanded)
    summary = {
        "output": str(output), "request_cache": str(cache), "condition": condition,
        "model": alias, "dry_run": bool(dry_run), "seed": seed,
        "base_questions": len(active), "full_planned_base_questions": len(selected),
        "by_task": dict(Counter(task_name(q) for q in active)),
        "unique_bases_by_task": {task: len({q["meta"]["base_structure"] for q in active
                                            if task_name(q) == task}) for task in TASKS},
        "variant_records": len(expanded), "unique_evaluations": len(responses),
        "reused_records": sum(row["api_request_reused"] for row in expanded),
        "statuses": dict(Counter(row["status"] for row in expanded)),
        "strategies": list(strategies), "view_counts": list(view_counts),
        "selector_version": SELECTOR_VERSION,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return expanded, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/questions.jsonl"))
    parser.add_argument("--scenes", type=Path, default=Path("data/scenes.jsonl"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("models.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--per-type", type=int, default=40)
    parser.add_argument("--limit", type=int, help="Limit original questions; 24 gives 8 per task.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--condition", default="image", choices=("image", "text", "image+text"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-output", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--allow-reasoning", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if not args.dry_run and not args.model:
        parser.error("Provide --model for a real evaluation, or use --dry-run.")
    config = (load_model_config(args.config, args.model) if args.model else
              {"alias": "dry-run", "model": "dry-run", "provider": "offline"})
    config.update(temperature=args.temperature, allow_reasoning=args.allow_reasoning)
    output = args.output or Path("results") / f"ablation_{config['alias']}.jsonl"
    rows, summary = asyncio.run(run_ablation(
        load_questions(args.questions), load_scenes(args.scenes), config, output,
        data_dir=args.data_dir or args.questions.parent, per_type=args.per_type,
        seed=args.seed, limit=args.limit, dry_run=args.dry_run, concurrency=args.concurrency,
        timeout=args.timeout, max_retries=args.max_retries, resume=not args.no_resume,
        condition=args.condition, cache_path=args.cache_output,
    ))
    if not args.no_plots:
        from ablation_plots import plot_ablation
        summary["figures"] = plot_ablation(output, args.figures_dir,
                                           bootstrap=args.bootstrap, seed=args.seed)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
