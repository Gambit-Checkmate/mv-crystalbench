"""Shared deterministic augmentation and question records (no API access)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
from ase import Atoms


def stable_seed(*values) -> int:
    blob = json.dumps(values, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def geometry_key(atoms: Atoms) -> str:
    """Canonical periodic geometry identity, independent of atom ordering."""
    rows = sorted(
        (int(number), *(round(float(v % 1), 8) for v in position))
        for number, position in zip(atoms.numbers, atoms.get_scaled_positions())
    )
    data = [np.round(atoms.cell.array, 8).tolist(), rows]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def variant_atoms(base: Atoms, index: int, seed: int, strain: bool = False):
    """Return a copy with a documented origin shift and optional <=2% strain.

    Rational shifts retain useful boundary sites. General-position shifts are
    included as controls with fewer boundary images. Geometry is never altered
    in the source catalog. Anisotropic variants are labeled as synthetic.
    """
    atoms = base.copy()
    rng = np.random.default_rng(stable_seed(seed, index, base.info.get("name", ""), strain))
    shifts = [
        [0, 0, 0], [0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
        [0.25, 0.25, 0], [0, 0.25, 0.25], [0.25, 0, 0.25],
        [1 / 3, 0, 0], [0, 1 / 3, 0], [0.125, 0.125, 0],
    ]
    shift = np.array(shifts[index] if index < len(shifts) else rng.uniform(0, 1, 3))
    scale = rng.uniform(0.98, 1.02, 3) if strain and index else np.ones(3)
    fractional = atoms.get_scaled_positions(wrap=True)
    atoms.set_cell(atoms.cell.array * scale[:, None], scale_atoms=False)
    atoms.set_scaled_positions((fractional + shift) % 1)
    atoms.wrap()
    augmentation = {
        "variant_index": index, "origin_shift_fractional": shift.tolist(),
        "axis_scale": scale.tolist(),
        "synthetic_strain": bool(strain and index),
    }
    return atoms, augmentation


def make_draft(
    qtype, difficulty, structure, scene, question, choices, answer_text, meta,
    text_repr=None, views=None,
):
    """Produce a question before globally counterbalancing the answer letters.

    view_ids are scene-qualified strings (scene_id:view_id), so Q5 can refer
    unambiguously to two scenes. Paths are relative to the dataset directory.
    """
    choices = [str(value) for value in choices]
    answer_text = str(answer_text)
    if len(choices) != 4 or len(set(choices)) != 4 or choices.count(answer_text) != 1:
        raise ValueError("Questions require exactly four unique choices and one answer.")
    selected = scene["views"][:3] if views is None else views
    meta = dict(meta)
    meta.setdefault("base_structure", structure)
    meta.setdefault("text_applicable", qtype not in {"Q1a", "Q6"})
    text = scene["text_repr"] if text_repr is None else text_repr
    if not meta["text_applicable"]:
        text = ""
    return {
        "qtype": qtype, "difficulty": difficulty, "structure": structure,
        "scene_id": scene["scene_id"],
        "view_ids": [f"{view.get('scene_id', scene['scene_id'])}:{view['view_id']}"
                     for view in selected],
        "image_paths": [view["image_path"] for view in selected],
        "question": question, "_choices": choices, "answer_text": answer_text,
        "meta": meta, "text_repr": text,
    }


def numeric_choices(answer: int, rng) -> list[str]:
    """Nearby positive integer distractors, with no duplicate options."""
    candidates = set()
    for offset in (1, 2, 3, 4, max(1, answer // 2)):
        candidates.update((answer - offset, answer + offset))
    candidates.discard(answer)
    candidates = sorted(value for value in candidates if value >= 0)
    nearby = sorted(candidates, key=lambda value: (abs(value - answer), value))[:7]
    wrong = rng.sample(nearby, 3)
    return [str(answer), *(str(value) for value in wrong)]


def model_prompt(question: dict, include_text: bool = False) -> str:
    """The ONLY model-facing text: never serialize internal metadata wholesale."""
    parts = [question["question"]]
    parts.extend(f"{letter}. {value}" for letter, value in question["options"].items())
    if include_text and question["meta"].get("text_applicable", True):
        parts.append(question["text_repr"])
    parts.append("Answer with exactly one option letter: A, B, C, or D.")
    return "\n".join(parts)
