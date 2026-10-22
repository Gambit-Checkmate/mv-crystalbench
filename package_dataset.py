"""Build the compact, byte-preserving Hugging Face dataset package.

Only question-referenced PNGs are copied. This never calls a model or changes
the source dataset. Existing user files are neither removed nor overwritten:
on reruns, only unchanged files owned by this script's manifest may be updated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


PACKAGE_ID = "mv-crystalbench-hf-package-v1"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_relative(value: str) -> Path:
    """Accept portable, normalized relative paths without traversal or drives."""
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or ":" in value
        or path.is_absolute()
        or any(part in {".", ".."} for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError(f"Unsafe relative package path: {value!r}")
    return Path(*path.parts)


def checked_path(root: Path, relative: str) -> Path:
    path = root / safe_relative(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its designated directory: {relative}")
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink() or getattr(current, "is_junction", lambda: False)():
            raise ValueError(f"Symlink/junction is not supported: {relative}")
    return path


def read_questions(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    content = path.read_bytes()
    questions = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate question IDs in {path.name}")
    return content, questions


def task_counts(questions: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(
        f"Q2_{question['difficulty']}" if question["qtype"] == "Q2" else question["qtype"]
        for question in questions
    ).items()))


def validate_package(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Check every file hash, image reference and mini/full membership."""
    expected = set(manifest["files"]) | {"manifest.json"}
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(f"Package file-set mismatch: extra={sorted(actual-expected)}, missing={sorted(expected-actual)}")
    for name, item in manifest["files"].items():
        data = checked_path(output, name).read_bytes()
        if len(data) != item["size_bytes"] or sha256_bytes(data) != item["sha256"]:
            raise ValueError(f"Manifest mismatch: {name}")
    full_bytes, full = read_questions(output / "questions.jsonl")
    mini_bytes, mini = read_questions(output / "questions_mini.jsonl")
    full_by_id = {question["id"]: question for question in full}
    if any(full_by_id.get(question["id"]) != question for question in mini):
        raise ValueError("Mini must contain exact full-set question objects")
    referenced = {image for question in full + mini for image in question["image_paths"]}
    actual_images = {
        path.relative_to(output / "images").as_posix()
        for path in (output / "images").rglob("*") if path.is_file()
    }
    if referenced != actual_images:
        raise ValueError("Images directory must contain exactly the referenced PNG set")
    for name in sorted(referenced):
        with Image.open(checked_path(output / "images", name)) as image:
            if image.format != "PNG" or image.size != (512, 512):
                raise ValueError(f"Expected a 512x512 PNG: {name}")
            image.load()
    for name, value in [("questions.jsonl", full_bytes), ("questions_mini.jsonl", mini_bytes)]:
        if sha256_bytes(value) != manifest["source_question_sha256"][name]:
            raise ValueError(f"Question bytes differ from source: {name}")
    return {
        "full_rows": len(full), "mini_rows": len(mini),
        "unique_referenced_pngs": len(referenced), "images_decoded": len(referenced),
        "mini_exact_subset": True, "manifest_verified": True,
        "file_count_including_manifest": len(actual),
        "total_bytes_including_manifest": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()),
    }


def build_package(project: Path, output: Path) -> dict[str, Any]:
    project = project.resolve()
    # Refuse destructive or ambiguous target locations; this tool never deletes.
    original_output = output if output.is_absolute() else project / output
    if original_output.is_symlink() or getattr(original_output, "is_junction", lambda: False)():
        raise ValueError("Output directory cannot be a symlink/junction")
    output = original_output.resolve()
    if output == project or not output.is_relative_to(project):
        raise ValueError("Output must be a dedicated subdirectory inside this project")
    if output.relative_to(project).parts[0] in {"data", "results", "figures", ".venv", ".git"}:
        raise ValueError("Output must not overlap source data, results, figures or runtime directories")

    source = project / "data"
    source_bytes: dict[str, bytes] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for name in ["questions.jsonl", "questions_mini.jsonl"]:
        source_bytes[name], rows[name] = read_questions(source / name)
    full, mini = rows["questions.jsonl"], rows["questions_mini.jsonl"]
    if len(full) != 716 or len(mini) != 200:
        raise ValueError("This release expects the frozen 716-question full and 200-question mini sets")
    full_by_id = {question["id"]: question for question in full}
    if any(full_by_id.get(question["id"]) != question for question in mini):
        raise ValueError("Mini contains an absent or modified full-set question")
    references = sorted({path for question in full + mini for path in question["image_paths"]})
    if len(references) != 1507:
        raise ValueError(f"This release expects 1507 referenced PNGs; found {len(references)}")

    # Pre-read payload and its sources so all safety checks happen before writes.
    payload = dict(source_bytes)
    source_hashes = {f"data/{name}": sha256_bytes(value) for name, value in source_bytes.items()}
    for relative in references:
        safe = safe_relative(relative)
        if safe.parts[0] != "scenes" or safe.suffix.lower() != ".png":
            raise ValueError(f"Unexpected image reference: {relative}")
        image_source = checked_path(source, relative)
        data = image_source.read_bytes()
        payload[f"images/{relative}"] = data
        source_hashes[f"data/{relative}"] = sha256_bytes(data)

    additions = {
        "LICENSE": "LICENSE-DATA",
        "provenance/aflow/LICENSE": "data/structure_sources/aflow/LICENSE",
        "provenance/aflow/NOTICE": "data/structure_sources/aflow/NOTICE",
        "provenance/aflow/selection.json": "data/structure_sources/aflow/selection.json",
    }
    selection = json.loads((project / additions["provenance/aflow/selection.json"]).read_text(encoding="utf-8"))
    for prototype in sorted(set(selection["selected"].values())):
        additions[f"provenance/aflow/{prototype}/base.bib"] = f"data/structure_sources/aflow/{prototype}/base.bib"
    for relative, original in additions.items():
        data = checked_path(project, original).read_bytes()
        payload[relative] = data
        source_hashes[original] = sha256_bytes(data)

    substitutions = {
        "FULL_COUNT": str(len(full)), "MINI_COUNT": str(len(mini)),
        "IMAGE_COUNT": f"{len(references):,}",
        "BASE_COUNT": str(len({question["meta"]["base_structure"] for question in full})),
    }
    card = (project / "docs/hf_dataset_card.md").read_text(encoding="utf-8")
    for key, value in substitutions.items():
        card = card.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", card):
        raise ValueError("Unresolved dataset-card template fields")
    payload["README.md"] = card.encode("utf-8")
    payload["provenance/README.md"] = (
        "# Structure-source attribution\n\n"
        "The generated questions and PNGs are licensed under CC BY 4.0; see ../LICENSE. "
        "Upstream material retains its own license.\n\n"
        "The AFLOW prototype encyclopedia supplied 28 of the 48 structure prototypes. "
        "Its pinned revision is `fb7421bb2522888b6a0a4ff91e124c6c65191096`: "
        "https://github.com/aflow-org/aflow_prototype_encyclopedia .\n\n"
        "Copyright 2026 M.J. Mehl, H. Eckert, S. Curtarolo; Apache License 2.0. "
        "The original LICENSE, retained NOTICE, source-selection hashes, and prototype "
        "bibliographies are included under aflow/. The NOTICE describes the full code "
        "repository's source collection; this compact dataset package includes the "
        "bibliographies and attribution, not the CIF/info.json files referenced by "
        "selection.json. Those source files remain in the companion code repository.\n\n"
        "ASE expands asymmetric sites and spglib standardizes the conventional cells. "
        "Benchmark bond rules, Q2 eligibility, and question generation are project "
        "additions, not AFLOW annotations. The other 20 catalogue entries use locally "
        "constructed classic prototypes and ASE builders.\n"
    ).encode("utf-8")

    old_manifest_path = checked_path(output, "manifest.json")
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8")) if old_manifest_path.exists() else None
    if old_manifest is not None and old_manifest.get("package_id") != PACKAGE_ID:
        raise ValueError("Existing manifest does not belong to this packaging tool")
    old_files = old_manifest.get("files", {}) if old_manifest else {}
    existing = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()} if output.exists() else set()
    unexpected = existing - (set(payload) | {"manifest.json"})
    if unexpected:
        raise ValueError(f"Unexpected existing files; preserving them without changes: {sorted(unexpected)}")
    for relative, desired in payload.items():
        target = checked_path(output, relative)
        if target.exists():
            current = target.read_bytes()
            if current != desired and sha256_bytes(current) != old_files.get(relative, {}).get("sha256"):
                raise ValueError(f"Refusing to overwrite an unmanaged or user-modified file: {relative}")

    manifest = {
        "package_id": PACKAGE_ID,
        "schema_version": 1,
        "seed": 20261021,
        "image_paths_base": "images",
        "full_rows": len(full), "mini_rows": len(mini),
        "full_task_counts": task_counts(full), "mini_task_counts": task_counts(mini),
        "unique_referenced_pngs": len(references),
        "mini_is_exact_subset": True,
        "source_question_sha256": {name: sha256_bytes(value) for name, value in source_bytes.items()},
        "manifest_hash_policy": "Every other package file is listed; manifest.json intentionally has no self-hash.",
        "payload_file_count": len(payload),
        "payload_total_bytes": sum(len(value) for value in payload.values()),
        "files": {
            relative: {"sha256": sha256_bytes(value), "size_bytes": len(value)}
            for relative, value in sorted(payload.items())
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for relative, value in sorted(payload.items()):
        target = checked_path(output, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != value:
            target.write_bytes(value)
    old_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    result = validate_package(output, manifest)
    for relative, original_sha256 in source_hashes.items():
        if sha256_bytes(checked_path(project, relative).read_bytes()) != original_sha256:
            raise ValueError(f"Source changed while packaging: {relative}")
    result["sources_unchanged"] = True
    result["source_question_sha256"] = manifest["source_question_sha256"]
    result["total_mib_including_manifest"] = round(result["total_bytes_including_manifest"] / 1024 ** 2, 6)
    result["output"] = output.relative_to(project).as_posix()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=Path("hf_dataset"))
    args = parser.parse_args()
    print(json.dumps(build_package(args.project_root, args.out_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
