"""Resumable OpenAI-compatible evaluation and deterministic offline baselines.

No network access occurs in --dry-run or --baseline mode. API credentials are
read from environment variables named by models.yaml, never stored in results.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import random
import re
import time
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from budget import BudgetExceeded, estimate_reservation, pricing


CONDITIONS = ("image", "text", "image+text", "no_image")
COMPLETED_STATUSES = {"ok", "invalid_response", "error", "skipped"}
PROMPT_VERSION = "english-images-first-numbered-groups-v2"


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def load_questions(path: str | Path) -> list[dict]:
    questions = []
    ids = set()
    for number, line in enumerate(Path(path).read_text("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item["id"] in ids:
            raise ValueError(f"Duplicate question ID at line {number}: {item['id']}")
        ids.add(item["id"])
        if set(item["options"]) != set("ABCD") or item["options"].get(item["answer"]) != item["answer_text"]:
            raise ValueError(f"Invalid choices/answer in {item['id']}")
        questions.append(item)
    return questions


def load_model_config(path: str | Path, alias: str) -> dict:
    import yaml

    contents = yaml.safe_load(Path(path).read_text("utf-8"))
    if not isinstance(contents, dict) or not isinstance(contents.get("models"), dict):
        raise ValueError("models.yaml must contain a models mapping.")
    if alias not in contents["models"]:
        raise ValueError(f"Unknown model alias: {alias}")
    config = {**contents.get("defaults", {}), **contents["models"][alias], "alias": alias}
    if "api_key" in config:
        raise ValueError("Use api_key_env, never a literal api_key in configuration.")
    for field in ("base_url", "model", "api_key_env"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise ValueError(f"Model {alias} requires {field}.")
    config.setdefault("provider", "openai-compatible")
    config.setdefault("supports_multi_image", True)
    config.setdefault("temperature", 0.0)
    config.setdefault("max_tokens", 1024)
    config.setdefault("token_limit_field", "max_tokens")
    if config["token_limit_field"] not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError("token_limit_field must be max_tokens or max_completion_tokens.")
    if not isinstance(config["supports_multi_image"], bool):
        raise ValueError("supports_multi_image must be a boolean.")
    if not isinstance(config.get('omit_temperature', False), bool):
        raise ValueError('omit_temperature must be a boolean.')
    return config


def normalize_condition(value: str) -> str:
    value = {"image_text": "image+text", "no-image": "no_image"}.get(value, value)
    if value not in CONDITIONS:
        raise ValueError(f"Unknown input condition: {value}")
    return value


def _parse_answer(text: str) -> tuple[str | None, str]:
    """Prefer an exact final Answer line; otherwise the last standalone A-D.

    The fallback is uppercase-only: English articles such as "a" are not votes.
    A bare letter embedded in a word, number, or identifier is not accepted.
    """
    if not isinstance(text, str) or not text.strip():
        return None, "invalid"
    last_line = text.strip().splitlines()[-1].strip()
    match = re.fullmatch(r"Answer:\s*([ABCD])", last_line)
    if match:
        return match.group(1), "final_answer_line"
    matches = re.findall(r"(?<![A-Za-z0-9_])([ABCD])(?![A-Za-z0-9_])", text)
    return (matches[-1], "last_standalone_letter") if matches else (None, "invalid")


def parse_answer(text: str) -> str | None:
    return _parse_answer(text)[0]


def _image_labels(question: dict) -> list[str]:
    labels = [f"Image {index + 1}" for index in range(len(question["image_paths"]))]
    groups = question.get("meta", {}).get("view_groups")
    if groups:
        for group_number, group in enumerate(groups, 1):
            for index in group["image_indices"]:
                if not isinstance(index, int) or not 0 <= index < len(labels):
                    raise ValueError("Invalid Q5 image group index.")
                labels[index] = f"Group {group_number}, Image {index + 1}"
    elif question["qtype"] == "Q5":
        raise ValueError("Q5 requires explicit image groups.")
    return labels


def build_prompt(question: dict, condition: str, *, montage=False, allow_reasoning=False) -> str:
    """Use English dataset fields verbatim; no structure names or meta dump."""
    condition = normalize_condition(condition)
    parts = [
        "Answer the following multiple-choice question about a crystal structure.",
    ]
    if condition != "no_image":
        parts.append("Use only the supplied visual and/or coordinate evidence.")
    if condition in {"image", "image+text"}:
        labels = _image_labels(question)
        parts.append("Images are supplied in this order: " + "; ".join(labels) + ".")
        if montage:
            parts.append("They are joined horizontally in one image; each panel is labeled.")
        if question["qtype"] == "Q5":
            parts.append("Compare Group 1 with Group 2; views within a group describe the same structure.")
    parts.append(question["question"])
    parts.extend(f"{letter}. {question['options'][letter]}" for letter in "ABCD")
    if condition in {"text", "image+text"} and question.get("meta", {}).get("text_applicable", True):
        if not question.get("text_repr"):
            raise ValueError(f"Text-applicable question {question['id']} has no text_repr.")
        parts.extend(["Coordinate description:", question["text_repr"]])
    parts.append("You may include a brief explanation before your final answer."
                 if allow_reasoning else "Do not include reasoning or explanations.")
    parts.append("Answer with a single letter (A, B, C, or D) on the last line "
                 "in the format: Answer: X")
    return "\n\n".join(parts)


def _image_paths(question: dict, data_dir: Path) -> list[Path]:
    root = data_dir.resolve()
    paths = []
    for value in question["image_paths"]:
        path = (root / value).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Question image paths must stay inside the data directory.")
        if not path.is_file():
            raise FileNotFoundError(f"Missing question image: {value}")
        paths.append(path)
    return paths


def _data_url(data: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def _horizontal_image(paths: list[Path], labels: list[str]) -> bytes:
    from PIL import Image, ImageDraw

    images = []
    try:
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if not images:
            raise ValueError("At least one image is required.")
        strip = 32
        canvas = Image.new("RGB", (sum(image.width for image in images),
                                  max(image.height for image in images) + strip), "white")
        draw = ImageDraw.Draw(canvas)
        x = 0
        for image, label in zip(images, labels):
            canvas.paste(image, (x, strip))
            draw.text((x + 10, 9), label, fill="black")
            x += image.width
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        canvas.close()
        return buffer.getvalue()
    finally:
        for image in images:
            image.close()


def build_messages(question: dict, condition: str, data_dir: Path, config: dict) -> list[dict]:
    condition = normalize_condition(condition)
    montage = (condition in {"image", "image+text"} and not config.get("supports_multi_image", True)
               and len(question["image_paths"]) > 1)
    prompt = build_prompt(
        question, condition, montage=montage,
        allow_reasoning=config.get("allow_reasoning", False),
    )
    if condition in {"text", "no_image"}:
        return [{"role": "user", "content": prompt}]
    paths = _image_paths(question, Path(data_dir))
    labels = _image_labels(question)
    content = []
    if montage:
        content.append({"type": "text", "text": "Combined image panels: " + "; ".join(labels)})
        content.append({"type": "image_url", "image_url": {
            "url": _data_url(_horizontal_image(paths, labels)),
        }})
    else:
        for path, label in zip(paths, labels):
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            content.extend([
                {"type": "text", "text": label},
                {"type": "image_url", "image_url": {"url": _data_url(path.read_bytes(), mime)}},
            ])
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class APIError(RuntimeError):
    def __init__(self, message, *, retryable=False, status_code=None):
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status_code = status_code


def _http_post(config: dict, payload: dict, timeout: float) -> dict:
    """One nonstreaming HTTP request. Never log the Authorization header."""
    key = os.environ.get(config["api_key_env"], "")
    if not key and config.get("requires_api_key", True):
        raise APIError(f"Missing environment variable {config['api_key_env']}.")
    endpoint = config["base_url"].rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    extra_headers = config.get('headers', {})
    if not isinstance(extra_headers, dict) or any(
            not isinstance(name, str) or not isinstance(value, str)
            or name.lower() in {'authorization', 'proxy-authorization', 'cookie'}
            or '\n' in name+value or '\r' in name+value
            for name, value in extra_headers.items()):
        raise APIError('headers must be plain string headers without credentials or line breaks.')
    headers.update(extra_headers)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urlrequest.Request(
        endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urlerror.HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        if key:
            detail = detail.replace(key, "[REDACTED]")
        raise APIError(
            f"HTTP {error.code}: {detail}", status_code=error.code,
            retryable=error.code in {408, 409, 429} or 500 <= error.code < 600,
        ) from None
    except (urlerror.URLError, TimeoutError, OSError) as error:
        raise APIError(f"{type(error).__name__}: {error}", retryable=True) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise APIError("The endpoint returned non-JSON content.", retryable=True) from None


def _response_text(response: dict) -> str:
    try:
        message = response["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        raise APIError("Malformed chat completion: missing choices[0].message.", retryable=True)
    if not isinstance(message, dict):
        raise APIError("Malformed chat completion: message must be an object.", retryable=True)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"] for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    if content is None:
        return str(message.get("refusal") or "")
    raise APIError("Malformed message content.", retryable=True)


def majority_answers(questions: list[dict]) -> dict[str, str]:
    """Global answer TEXT majority, not option-letter frequency; lexicographic ties."""
    counts = defaultdict(Counter)
    for question in questions:
        counts[_majority_group(question)][question["answer_text"]] += 1
    return {
        group: min(values, key=lambda answer: (-values[answer], answer))
        for group, values in counts.items()
    }


def _majority_group(question: dict) -> str:
    return (f"Q2:{question['difficulty']}" if question["qtype"] == "Q2"
            else question["qtype"])


def _result_base(question, config, condition, baseline, dry_run, run_id,
                 config_fingerprint, dataset_fingerprint) -> dict:
    return {
        "schema_version": 1, "run_id": run_id, "question_id": question["id"],
        "model": config["alias"], "provider": config.get("provider", "openai-compatible"),
        "model_name": config.get("model", config["alias"]), "response_model": None,
        "condition": condition, "temperature": None if config.get('omit_temperature', False) else config.get("temperature", 0.0),
        "baseline": baseline, "dry_run": bool(dry_run),
        "status": "ok", "invalid": False, "prediction": None, "prediction_text": None,
        "answer": question["answer"], "answer_text": question["answer_text"],
        "correct": False, "raw_response": "", "parse_method": None, "error": None,
        "finish_reason": None,
        "attempts": 0, "latency_seconds": 0.0, "usage": None,
        "cost_usd": 0.0, "cost_basis": "not_billed", "budget_attempt_ids": [],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "qtype": question["qtype"], "difficulty": question["difficulty"],
        "structure": question.get("structure"),
        "base_structure": question.get("meta", {}).get(
            "base_structure", question.get("base_structure", question.get("structure"))
        ),
        "scene_id": question.get("scene_id"), "view_ids": question.get("view_ids", []),
        "image_paths": question.get("image_paths", []), "meta": question.get("meta", {}),
        "input_fingerprint": _fingerprint(question),
        "dataset_fingerprint": dataset_fingerprint, "config_fingerprint": config_fingerprint,
        "baseline_detail": None,
    }


async def _evaluate_one(question, config, condition, data_dir, baseline, dry_run,
                        majorities, seed, timeout, max_retries, backoff_base, base, budget=None):
    start = time.perf_counter()
    if condition == "text" and not question.get("meta", {}).get("text_applicable", True):
        base.update(status="skipped", correct=None,
                    skip_reason="text_applicable=false", latency_seconds=0.0)
        return base
    if dry_run:
        try:
            # Dry runs exercise exactly the same input/template/image path as
            # paid requests, including base64 encoding and optional montages.
            build_messages(question, condition, data_dir, config)
        except (ValueError, OSError) as error:
            base.update(
                status="error", correct=False,
                error={"type": type(error).__name__, "message": str(error),
                       "retryable": False, "status_code": None},
                latency_seconds=time.perf_counter() - start,
            )
            return base
    if baseline or dry_run:
        if baseline == "majority":
            group = _majority_group(question)
            if group not in majorities:
                raise ValueError(f"No full-dataset majority for {group}.")
            chosen = majorities[group]
            prediction = next((key for key, value in question["options"].items()
                               if value == chosen), None)
            base["baseline_detail"] = {
                "group": group, "majority_answer_text": chosen,
                "majority_absent": prediction is None,
                "tie_break": "lexicographically smallest answer_text",
                "absent_policy": "predict no listed option and count incorrect; do not substitute",
            }
            base["prediction_text"] = chosen
            base["parse_method"] = "answer_text_majority"
        else:
            deterministic_seed = int(_fingerprint([seed, question["id"],
                                                  baseline or "dry-run"])[:16], 16)
            prediction = random.Random(deterministic_seed).choice("ABCD")
            base["prediction_text"] = question["options"][prediction]
            base["parse_method"] = "seeded_uniform_random"
        base.update(prediction=prediction, correct=prediction == question["answer"],
                    raw_response=f"Answer: {prediction}" if prediction else "",
                    latency_seconds=time.perf_counter() - start)
        if dry_run:
            base["prediction"], base["parse_method"] = _parse_answer(base["raw_response"])
        return base
    request_key = _fingerprint([base['run_id'], question['id'], condition])
    prior = await asyncio.to_thread(budget.request_entries, request_key) if budget else []
    try:
        messages = build_messages(question, condition, data_dir, config)
        payload = {
            "model": config["model"], "messages": messages,
            "stream": False,
            config.get("token_limit_field", "max_tokens"): config.get("max_tokens", 1024),
        }
        if not config.get('omit_temperature', False):
            payload['temperature'] = config.get('temperature', 0.0)
        extras = config.get("extra_body", {})
        if any(key in payload for key in extras) or (config.get('omit_temperature') and 'temperature' in extras):
            raise ValueError("extra_body cannot override core request fields.")
        payload.update(extras)
        estimate = estimate_reservation(messages, config, image_count=(
            len(question['image_paths']) if condition in {'image', 'image+text'} else 0
        )) if budget else None
        recovered = next((entry['response'] for entry in reversed(prior) if 'response' in entry), None)
        attempts = [None] if recovered is not None else range(len(prior), max_retries + 1)
        base['attempts'] = len(prior)
        for attempt in attempts:
            entry_id = None
            try:
                if recovered is not None:
                    response = recovered
                    base['budget_response_recovered'] = True
                else:
                    if budget:
                        entry_id = await asyncio.to_thread(budget.reserve, request_key, estimate, model=config['alias'])
                    base["attempts"] = attempt + 1
                    response = await asyncio.to_thread(_http_post, config, payload, timeout)
                    if budget:
                        await asyncio.to_thread(budget.settle, entry_id, response=response)
                raw = _response_text(response)
                prediction, method = _parse_answer(raw)
                usage = response.get("usage")
                base.update(
                    response_model=response.get("model"), raw_response=raw,
                    finish_reason=response['choices'][0].get('finish_reason'),
                    prediction=prediction, parse_method=method,
                    prediction_text=question["options"].get(prediction),
                    correct=prediction == question["answer"],
                    invalid=prediction is None,
                    status="invalid_response" if prediction is None else "ok",
                    usage=usage if isinstance(usage, dict) else None,
                )
                break
            except (APIError, ValueError, OSError) as error:
                if budget and entry_id is not None:
                    await asyncio.to_thread(budget.settle, entry_id, failed=True)
                if isinstance(error, APIError) and error.retryable and attempt is not None and attempt < max_retries:
                    await asyncio.sleep(min(32.0, backoff_base * 2**attempt))
                    continue
                raise
        else:
            raise APIError('No HTTP attempts remain after previously reserved attempts.')
    except BudgetExceeded as error:
        if base['attempts'] == 0:
            raise
        base.update(status='error', correct=False, budget_exhausted=True,
                    error={'type': 'BudgetExceeded', 'message': str(error),
                           'retryable': False, 'status_code': None})
    except (APIError, ValueError, OSError) as error:
        base.update(
            status="error", correct=False,
            error={"type": type(error).__name__, "message": str(error),
                   "retryable": bool(getattr(error, "retryable", False)),
                   "status_code": getattr(error, "status_code", None)},
        )
    if budget:
        charged = await asyncio.to_thread(budget.request_entries, request_key)
        base.update(cost_usd=sum(entry['charged_usd'] for entry in charged),
                    cost_basis='public_base_rates_with_retained_uncertain_attempts',
                    budget_attempt_ids=[entry['id'] for entry in charged],
                    budget_ledger=str(budget.path))
    elif not baseline and not dry_run:
        usage = base.get('usage') or {}
        try:
            in_rate, out_rate = pricing(config)
            if all(isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0
                   for key in ('prompt_tokens', 'completion_tokens')):
                base.update(cost_usd=(usage['prompt_tokens']*in_rate+usage['completion_tokens']*out_rate)/1e6,
                            cost_basis='reported_usage_public_base_rates')
            else:
                base.update(cost_usd=None, cost_basis='unknown_missing_usage')
        except ValueError:
            base.update(cost_usd=None, cost_basis='unknown_missing_public_rates')
    base["latency_seconds"] = time.perf_counter() - start
    return base


def _read_results(path: Path) -> tuple[list[dict], bytes | None]:
    if not path.exists():
        return [], None
    results = []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    interrupted_tail = None
    for index, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            results.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines) and not line.endswith((b"\n", b"\r")):
                interrupted_tail = line
            else:
                raise ValueError(f"Invalid JSONL at {path.name}:{index}; refusing to alter a malformed completed line.")
    return results, interrupted_tail


async def evaluate_questions(
    questions: list[dict], config: dict, output: str | Path, *, data_dir: str | Path,
    conditions=("image",), baseline=None, dry_run=False, full_questions=None,
    concurrency=4, timeout=120.0, max_retries=5, limit=None, resume=True,
    seed=20261021, backoff_base=1.0, budget=None,
) -> list[dict]:
    """Append new results; return all matching current rows with latest retry wins.

    All recorded rows (including errors, invalid and skipped) are resumed;
    per-call HTTP retries already handle transient errors. Different fingerprints are
    rejected before writing, so changed experiments cannot silently mix.
    ``max_retries=5`` means up to six HTTP attempts including the initial call.
    """
    if baseline not in {None, "random", "majority"}:
        raise ValueError("baseline must be random or majority.")
    if baseline and dry_run:
        raise ValueError("Choose either a baseline or dry_run.")
    if concurrency < 1 or timeout <= 0 or not 0 <= max_retries <= 5 or backoff_base < 0:
        raise ValueError("Require concurrency>=1, timeout>0, 0<=max_retries<=5, backoff_base>=0.")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive.")
    normalized = tuple(dict.fromkeys(normalize_condition(c) for c in conditions))
    if not normalized:
        raise ValueError("At least one condition is required.")
    if baseline == "majority" and not full_questions:
        raise ValueError("The majority baseline requires the full question dataset.")
    majorities = majority_answers(full_questions) if baseline == "majority" else {}
    config = dict(config)
    config.setdefault("alias", "dry-run" if dry_run else baseline)
    if not config["alias"]:
        raise ValueError("A model alias is required.")
    config.setdefault("model", config["alias"])
    config.setdefault("temperature", 0.0)
    safe_config = {key: value for key, value in config.items() if key != "api_key"}
    if "api_key" in config:
        raise ValueError("Literal API keys are prohibited in configuration.")
    config_fingerprint = _fingerprint({
        "model": safe_config, "baseline": baseline, "dry_run": dry_run, "seed": seed,
        "prompt_version": PROMPT_VERSION, "majorities": majorities,
    })
    dataset_fingerprint = _fingerprint(questions)
    run_id = _fingerprint([config_fingerprint, dataset_fingerprint])[:20]
    output = Path(output).expanduser().resolve()
    existing, interrupted_tail = _read_results(output)
    if output.exists() and output.stat().st_size and not resume:
        raise ValueError("Output already contains results; choose a new output path.")
    latest = {}
    for record in existing:
        if (record.get("config_fingerprint") != config_fingerprint
                or record.get("dataset_fingerprint") != dataset_fingerprint):
            raise ValueError("Existing results use a different dataset/configuration; choose a new output path.")
        latest[(record["question_id"], record["condition"])] = record
    selected = questions if limit is None else questions[:limit]
    jobs = [
        (question, condition) for question in selected for condition in normalized
        if latest.get((question["id"], condition), {}).get("status") not in COMPLETED_STATUSES
    ]
    # Credentials are needed only for actual pending API calls.
    actual_jobs = [(q, c) for q, c in jobs if not (
        c == "text" and not q.get("meta", {}).get("text_applicable", True)
    )]
    if actual_jobs and not baseline and not dry_run:
        if budget is not None:
            pricing(config)
        if "api_key_env" not in config:
            raise ValueError("API model configuration requires api_key_env.")
        if config.get("requires_api_key", True) and not os.environ.get(config["api_key_env"]):
            raise ValueError(f"Set the environment variable {config['api_key_env']} before API evaluation.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if interrupted_tail is not None:
        # Preserve an interrupted final write before trimming it. Never discard
        # a malformed newline-terminated record or alter a mismatched run.
        backup = output.with_name(output.name + ".interrupted-" +
                                  hashlib.sha256(interrupted_tail).hexdigest()[:12] + ".txt")
        backup.write_bytes(interrupted_tail)
        with output.open("r+b") as handle:
            handle.truncate(output.stat().st_size - len(interrupted_tail))
    if output.exists() and output.stat().st_size:
        with output.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) not in (b"\n", b"\r")
        if needs_newline:
            with output.open("ab") as handle:
                handle.write(b"\n")
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    budget_stop = asyncio.Event()
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        async def worker(question, condition):
            async with semaphore:
                if budget_stop.is_set():
                    return
                base = _result_base(question, config, condition, baseline, dry_run,
                                    run_id, config_fingerprint, dataset_fingerprint)
                try:
                    result = await _evaluate_one(
                        question, config, condition, Path(data_dir), baseline, dry_run,
                        majorities, seed, timeout, max_retries, backoff_base, base,
                        budget=budget if not baseline and not dry_run else None,
                    )
                except BudgetExceeded:
                    budget_stop.set()
                    return
                if result.get('budget_exhausted'):
                    budget_stop.set()
                async with lock:
                    handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    latest[(question["id"], condition)] = result
        await asyncio.gather(*(worker(question, condition) for question, condition in jobs))
    keys = {(question["id"], condition) for question in selected for condition in normalized}
    return [latest[key] for key in sorted(keys) if key in latest]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=Path("data/questions_mini.jsonl"))
    parser.add_argument("--full-questions", type=Path, default=Path("data/questions.jsonl"))
    parser.add_argument("--data-dir", type=Path, help="Image root; defaults to questions parent.")
    parser.add_argument("--config", type=Path, default=Path("models.yaml"))
    parser.add_argument("--model", help="Alias under models: in the YAML configuration.")
    parser.add_argument("--condition", default="image",
                        choices=[*CONDITIONS, "image_text", "all"])
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--baseline", choices=["random", "majority"])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--allow-reasoning", action="store_true",
                        help="Allow a brief explanation before the final answer line.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--budget-ledger", type=Path, help="Shared persistent reservation ledger across models.")
    parser.add_argument("--budget-limit-usd", type=float, help="Hard reservation ceiling at public base rates.")
    args = parser.parse_args()
    if (args.budget_ledger is None) != (args.budget_limit_usd is None):
        parser.error('--budget-ledger and --budget-limit-usd must be supplied together.')
    if args.budget_ledger is not None:
        from budget import BudgetTracker
        budget = BudgetTracker(args.budget_ledger, args.budget_limit_usd)
    else:
        budget = None
    if not args.model and not args.dry_run and not args.baseline:
        parser.error("--model is required unless using --dry-run or --baseline.")
    if args.baseline:
        config = {"alias": args.baseline, "model": args.baseline, "provider": "baseline"}
    elif args.model:
        config = load_model_config(args.config, args.model)
    else:
        config = {"alias": "dry-run", "model": "dry-run", "provider": "offline"}
    config["temperature"] = args.temperature
    config["allow_reasoning"] = args.allow_reasoning
    conditions = CONDITIONS if args.condition == "all" else (normalize_condition(args.condition),)
    full = load_questions(args.full_questions) if args.baseline == "majority" else None
    questions = load_questions(args.questions)
    if args.output:
        runs = [(args.output, conditions)]
    else:
        runs = [
            (Path("results") / f"{config['alias']}_{condition.replace('+', '_')}.jsonl",
             (condition,))
            for condition in conditions
        ]
    for output, run_conditions in runs:
        results = asyncio.run(evaluate_questions(
            questions, config, output,
            data_dir=args.data_dir or args.questions.parent, conditions=run_conditions,
            baseline=args.baseline, dry_run=args.dry_run, full_questions=full,
            concurrency=args.concurrency, timeout=args.timeout, max_retries=args.max_retries,
            limit=args.limit, resume=not args.no_resume, seed=args.seed,
            budget=budget,
        ))
        scored = [record for record in results if record["status"] != "skipped"]
        print(json.dumps({
            "output": str(output.resolve()), "records": len(results),
            "statuses": dict(Counter(record["status"] for record in results)),
            "accuracy": sum(bool(record["correct"]) for record in scored) / len(scored) if scored else None,
            "scored": len(scored), "dry_run": args.dry_run, "baseline": args.baseline,
            "budget": budget.snapshot() if budget else None,
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
