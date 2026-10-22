"""Run authorized final experiments with shared, non-enforcing cost ledgers.

Credentials come exclusively from the environment. Each stage is independently
resumable. A failed vision smoke test excludes that model from subsequent stages.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import statistics
import time

from PIL import Image, ImageDraw, ImageFont

from evaluate import evaluate_questions, load_model_config, load_questions
from budget import BudgetTracker
from ablation import run_ablation, load_scenes, select_questions, build_ablation_plan

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "final_run"
CONDITIONS = ("image", "text", "no_image")
SEED = 20261022


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def emit(event, **data):
    print(json.dumps({"event": event, **data}, ensure_ascii=False), flush=True)


def ordered_mini():
    groups = defaultdict(list)
    for q in load_questions(ROOT / "data/questions_mini.jsonl"):
        group = q["qtype"] + ("_" + q["difficulty"] if q["qtype"] == "Q2" else "")
        groups[group].append(q)
    names = ("Q1a", "Q1b", "Q2_easy", "Q2_hard", "Q3", "Q4", "Q5", "Q6")
    if any(len(groups[name]) != 25 for name in names):
        raise ValueError("Expected the accepted 200-question mini, eight groups of 25.")
    return [groups[name][i] for i in range(25) for name in names]


def vision_controls():
    """Two same-prompt nonce pictures: the correct option is only in the pixels."""
    rng = random.Random(SEED)
    options = {letter: "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(7))
               for letter in "ABCD"}
    folder = OUT / "vision_control"
    folder.mkdir(parents=True, exist_ok=True)
    questions = []
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 70)
    for i, letter in enumerate(("B", "D")):
        path = folder / f"control_{i}.png"
        image = Image.new("RGB", (512, 512), "white")
        draw = ImageDraw.Draw(image)
        draw.text((256, 256), options[letter], font=font, fill="black", anchor="mm")
        image.save(path)
        questions.append(dict(
            id=f"vision-control-{i}", qtype="VISION_CONTROL", difficulty="control",
            structure="", scene_id=f"vision-control-{i}", view_ids=[f"control-{i}"],
            image_paths=[str(path)], question="Which exact code is printed in the supplied image?",
            options=options, answer=letter, answer_text=options[letter], text_repr="",
            meta={"base_structure": "vision_control", "text_applicable": False},
        ))
    return questions


def status_path():
    return OUT / "status.json"


def read_status():
    return json.loads(status_path().read_text(encoding="utf-8")) if status_path().exists() else {}


def model_config(alias):
    return load_model_config(ROOT / "models.yaml", alias)


def stats(rows):
    scored = [r for r in rows if r["status"] != "skipped"]
    return dict(records=len(rows), scored=len(scored), statuses=dict(Counter(r["status"] for r in rows)),
                correct=sum(bool(r.get("correct")) for r in scored),
                response_models=dict(Counter(r.get("response_model") or "missing" for r in scored)),
                invalid=sum(bool(r.get("invalid")) for r in scored),
                final_answer_lines=sum(r.get("parse_method") == "final_answer_line" for r in scored),
                prompt_tokens=sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in scored),
                completion_tokens=sum((r.get("usage") or {}).get("completion_tokens", 0) for r in scored),
                mean_latency=statistics.mean(r["latency_seconds"] for r in scored) if scored else None)


async def dispatch_models(models, worker, *, parallel=False):
    """Let every in-flight model finish before surfacing a parallel failure."""
    if len(set(models)) != len(models):
        raise ValueError('Specify each model only once.')
    if not parallel:
        for alias in models:
            await worker(alias)
        return
    outcomes = await asyncio.gather(*(worker(alias) for alias in models), return_exceptions=True)
    failures = [(alias, outcome) for alias, outcome in zip(models, outcomes) if isinstance(outcome, BaseException)]
    if failures:
        raise RuntimeError('Model stage failures after all model tasks settled: ' + '; '.join(
            f'{alias}: {type(error).__name__}: {error}' for alias, error in failures))


def shared_run_state():
    estimates_path = OUT / 'ablation_budget_estimates.json'
    return dict(statuses=read_status(), lock=asyncio.Lock(),
                estimates=json.loads(estimates_path.read_text('utf-8')) if estimates_path.exists() else {})


async def smoke(models, tracker, *, parallel_models=False, _state=None):
    questions = ordered_mini()
    state = _state if _state is not None else shared_run_state()
    statuses, status_lock = state['statuses'], state['lock']
    async def one_model(alias):
        config = model_config(alias)
        rows = await evaluate_questions(questions, config, OUT / f"{alias}_image.jsonl",
            data_dir=ROOT / "data", limit=8, concurrency=8, timeout=300, max_retries=1,
            seed=SEED, budget=tracker)
        passed = (len(rows) == 8 and all(r["status"] == "ok" and
                  r.get("parse_method") == "final_answer_line" and r.get("response_model") for r in rows))
        controls = []
        if passed:
            controls = await evaluate_questions(vision_controls(), config, OUT / f"{alias}_vision_control.jsonl",
                data_dir=ROOT, concurrency=2, timeout=300, max_retries=1, seed=SEED, budget=tracker)
            passed = len(controls) == 2 and all(r["status"] == "ok" and r["correct"] for r in controls)
        reason = None if passed else [r.get("error") or r.get("raw_response") for r in rows + controls
                                     if r["status"] != "ok" or r.get("qtype") == "VISION_CONTROL" and not r["correct"]]
        async with status_lock:
            statuses[alias] = {**statuses.get(alias, {}), **dict(smoke_passed=bool(passed), smoke=stats(rows),
                                   vision_controls=stats(controls), failure_reason=reason,
                                   requested_model=config["model"], provider=config["provider"])}
            save(status_path(), statuses)
        emit("smoke_complete", model=alias, passed=bool(passed), **statuses[alias], budget=tracker.snapshot())
    await dispatch_models(models, one_model, parallel=parallel_models)


async def main_table(models, tracker, *, parallel_models=False, _state=None, run_baselines=True):
    questions = ordered_mini()
    state = _state if _state is not None else shared_run_state()
    statuses, status_lock = state['statuses'], state['lock']
    async def one_model(alias):
        if not statuses.get(alias, {}).get("smoke_passed"):
            emit("skip_model", model=alias, reason="Vision smoke test has not passed.")
            return
        config = model_config(alias)
        for condition in CONDITIONS:
            path = OUT / f"{alias}_{condition}.jsonl"
            rows = await evaluate_questions(questions, config, path, data_dir=ROOT / "data",
                conditions=(condition,), concurrency=8, timeout=300, max_retries=1,
                seed=SEED, budget=tracker)
            summary = stats(rows)
            async with status_lock:
                statuses[alias].setdefault("main", {})[condition] = summary
                save(status_path(), statuses)
            emit("main_condition_complete", model=alias, condition=condition, **summary, budget=tracker.snapshot())
            if len(rows) != 200:
                emit("stopped", stage="main", reason="Incomplete results or explicit operator pause.")
                return
    await dispatch_models(models, one_model, parallel=parallel_models)
    if run_baselines:
        await random_baselines(questions)


async def random_baselines(questions=None):
    questions = ordered_mini() if questions is None else questions
    for condition in CONDITIONS:
        rows = await evaluate_questions(questions, {"alias": "random", "model": "random", "provider": "baseline"},
            OUT / f"random_{condition}.jsonl", data_dir=ROOT / "data", conditions=(condition,),
            baseline="random", concurrency=8, seed=SEED)
        emit("baseline_complete", condition=condition, **stats(rows))


def ablation_cost_projection(plan, observed, config, cache_path):
    """Price only HTTP requests missing from the resumable unique-request cache."""
    import math
    from evaluate import COMPLETED_STATUSES, _read_results

    with_usage = [r for r in observed if isinstance(r.get('usage'), dict) and all(
        isinstance(r['usage'].get(key), (int, float)) and not isinstance(r['usage'][key], bool)
        and math.isfinite(r['usage'][key]) and r['usage'][key] >= 0
        for key in ('prompt_tokens', 'completion_tokens'))]
    if not with_usage:
        raise ValueError('No observed usage for ablation cost planning.')
    cached, _ = _read_results(Path(cache_path))
    latest = {(r['question_id'], r['condition']): r for r in cached}
    pending = [q for q in plan['requests'] if latest.get((q['id'], 'image'), {}).get('status') not in COMPLETED_STATUSES]
    mean_images = statistics.mean(len(r['image_paths']) for r in with_usage)
    if mean_images <= 0:
        raise ValueError('Ablation planning requires observed image requests.')
    mean_input = statistics.mean(r['usage']['prompt_tokens'] for r in with_usage)
    mean_output = statistics.mean(r['usage']['completion_tokens'] for r in with_usage)
    projected = sum((mean_input * max(1, len(q['image_paths'])/mean_images) * config['price_input_per_million']
                     + mean_output * config['price_output_per_million'])/1e6 for q in pending) * 1.2
    return dict(estimated_usd_with_20pct_margin=projected,
                planned_unique_requests=len(plan['requests']), pending_unique_requests=len(pending),
                completed_unique_requests=len(plan['requests'])-len(pending),
                image_input_mean=mean_input, image_output_mean=mean_output)


def ablation_plan_identity(plan):
    import hashlib
    payload = json.dumps(plan['requests'], sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


async def ablations(models, tracker, *, parallel_models=False, _state=None):
    state = _state if _state is not None else shared_run_state()
    statuses, status_lock = state['statuses'], state['lock']
    all_questions = load_questions(ROOT / "data/questions.jsonl")
    scenes = load_scenes(ROOT / "data/scenes.jsonl")
    estimates_path = OUT / 'ablation_budget_estimates.json'
    estimates = state['estimates']
    async def one_model(alias):
        if not statuses.get(alias, {}).get("smoke_passed"):
            return
        config = model_config(alias)
        plan_path = OUT / f'ablation_{alias}.plan.json'
        saved_plan = json.loads(plan_path.read_text('utf-8')) if plan_path.exists() else None
        per_type = 30
        if saved_plan and saved_plan.get('per_type') != 30:
            raise ValueError('The authorized cohort is now 30 questions per task. Archive an older smaller plan/cache before starting this cohort.')
        chosen = select_questions(all_questions, per_type=per_type, seed=20261021)
        plan = build_ablation_plan(chosen, scenes, seed=20261021)
        if saved_plan and (saved_plan.get('seed') != 20261021 or
                           saved_plan.get('request_fingerprint') != ablation_plan_identity(plan)):
            raise ValueError('Saved ablation plan differs from the current questions/scenes; refusing to change a running cohort.')
        image_path = OUT / f"{alias}_image.jsonl"
        observed = [json.loads(line) for line in image_path.read_text(encoding="utf-8").splitlines()]
        cache = ROOT / 'results' / '.ablation_cache' / f'{alias}.requests.jsonl'
        # A pre-existing cache without a frozen plan cannot safely switch its
        # cohort size. New plans are persisted before the first network request.
        if not saved_plan and cache.exists() and cache.stat().st_size:
            raise ValueError(f'{alias}: existing request cache has no saved cohort plan; restore its original plan before resuming.')
        try:
            projection = ablation_cost_projection(plan, observed, config, cache)
        except ValueError as error:
            if str(error) != 'No observed usage for ablation cost planning.':
                raise
            projection = dict(estimated_usd_with_20pct_margin=None,
                              planned_unique_requests=len(plan['requests']),
                              projection_note='No observed usage; cost projection unavailable. Evaluation remains authorized.')
        snapshot = tracker.snapshot()
        remaining = snapshot["remaining_usd"]
        async with status_lock:
            estimates[alias] = dict(per_type=per_type, remaining_usd=remaining, budget_enforced=False, **projection)
            save(estimates_path, estimates)
        emit("ablation_plan", model=alias, **estimates[alias])
        if not saved_plan:
            save(plan_path, dict(per_type=per_type, seed=20261021,
                 request_fingerprint=ablation_plan_identity(plan),
                 base_question_ids=plan.get('base_question_ids', [q['id'] for q in chosen]),
                 planned_unique_requests=len(plan['requests'])))
        path = ROOT / "results" / f"ablation_{alias}.jsonl"
        rows, summary = await run_ablation(all_questions, scenes, config, path,
            data_dir=ROOT / "data", per_type=per_type, seed=20261021, concurrency=8,
            timeout=300, max_retries=1, budget=tracker)
        async with status_lock:
            statuses[alias]["ablation"] = summary
            save(status_path(), statuses)
        emit("ablation_complete", **summary, budget=tracker.snapshot())
        if len(rows) != per_type*3*16:
            emit("stopped", stage="ablation", reason="Incomplete results or explicit operator pause.")
            return
    await dispatch_models(models, one_model, parallel=parallel_models)


async def pipeline(models, main_tracker, ablation_tracker, *, parallel_models=False):
    """Advance each model independently, sharing status safely across stages."""
    state = shared_run_state()
    async def one_model(alias):
        await smoke([alias], main_tracker, _state=state)
        if not state['statuses'].get(alias, {}).get('smoke_passed'):
            return
        await main_table([alias], main_tracker, _state=state, run_baselines=False)
        progress = state['statuses'][alias].get('main', {})
        if any(progress.get(condition, {}).get('records') != 200 for condition in CONDITIONS):
            emit('skip_ablation', model=alias, reason='Main evaluation is incomplete or explicitly paused.')
            return
        await ablations([alias], ablation_tracker, _state=state)
    try:
        await dispatch_models(models, one_model, parallel=parallel_models)
    finally:
        # These use deterministic local answers and carry no API charges.
        await random_baselines()


async def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'pipeline':
        main_tracker = BudgetTracker(OUT / 'main_budget.json', 40, enforce=False)
        ablation_tracker = BudgetTracker(OUT / 'ablation_budget.json', 45, enforce=False)
        await pipeline(args.models, main_tracker, ablation_tracker, parallel_models=args.parallel_models)
    elif args.stage in {"smoke", "main"}:
        tracker = BudgetTracker(OUT / "main_budget.json", 40, enforce=False)
        await (smoke(args.models, tracker, parallel_models=args.parallel_models) if args.stage == "smoke"
               else main_table(args.models, tracker, parallel_models=args.parallel_models))
    else:
        tracker = BudgetTracker(OUT / "ablation_budget.json", 45, enforce=False)
        await ablations(args.models, tracker, parallel_models=args.parallel_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "main", "ablation", "pipeline"))
    parser.add_argument("--models", nargs="+", default=["gpt", "claude", "kimi"], choices=("gpt", "claude", "kimi"))
    parser.add_argument('--parallel-models', action='store_true',
                        help='Run models concurrently within one process, each with 8 request workers; safely merge shared status.')
    asyncio.run(run(parser.parse_args()))
