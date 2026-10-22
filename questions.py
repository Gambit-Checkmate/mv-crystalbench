"""Generate reproducible, automatically checked multiview crystal questions.

No model API is called. Run: python questions.py --out-dir data --target 100
The 48-base catalog produces 716 questions: Q4 has 96 balanced questions,
Q2 has 60 easy + 60 hard, and the other five types have 100 each.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import re

import numpy as np
import spglib
from ase import Atoms

from question_utils import (
    geometry_key, make_draft, model_prompt, numeric_choices, stable_seed, variant_atoms,
)
from render import BOUNDARY_TOLERANCE, _prepare_views
from scenes import DEFAULT_SEED, SceneStore, write_review
from structures import SYMPREC, get_structures


CRYSTAL_SYSTEMS = ("triclinic", "monoclinic", "orthorhombic", "tetragonal",
                   "trigonal", "hexagonal", "cubic")
Q5_OPTIONS = ("Same structure", "Different element identities",
              "Different atomic arrangement", "Different lattice dimensions")
Q5_QUESTION = (
    "The first three images form Group 1; the last three form Group 2. "
    "Do they represent the same periodic structure, allowing a global translation "
    "and different viewing directions? If different, use these categories: "
    "'Different element identities' means element substitutions at unchanged sites; "
    "'Different atomic arrangement' includes missing atoms or a different polymorph; "
    "'Different lattice dimensions' means only the cell dimensions changed while "
    "fractional atomic coordinates stayed unchanged."
)
POLYMORPHS = {
    "TiO2_rutile": "TiO2_anatase", "TiO2_anatase": "TiO2_rutile",
    "ZnS_zincblende": "ZnS_wurtzite", "ZnS_wurtzite": "ZnS_zincblende",
    "C_diamond": "C_graphite", "C_graphite": "C_diamond",
}


def symmetry_dataset(atoms):
    result = spglib.get_symmetry_dataset(
        (atoms.cell.array, atoms.get_scaled_positions(), atoms.numbers), symprec=SYMPREC
    )
    if result is None:
        raise ValueError("Cannot determine structure symmetry.")
    return result


def crystal_system(number: int) -> str:
    for upper, name in zip((2, 15, 74, 142, 167, 194, 230), CRYSTAL_SYSTEMS):
        if number <= upper:
            return name
    raise ValueError("Invalid space-group number.")


def directions_equivalent(first, second, rotations, cell) -> bool:
    """Actual spglib fractional-coordinate rotations, not Cartesian guesses."""
    target = np.asarray(second, dtype=float) @ cell
    target /= np.linalg.norm(target)
    for rotation in rotations:
        transformed = (rotation @ np.asarray(first, dtype=float)) @ cell
        transformed /= np.linalg.norm(transformed)
        if abs(float(np.dot(transformed, target))) >= 1 - 1e-8:
            return True
    return False


def q6_directions(atoms):
    dataset = symmetry_dataset(atoms)
    system = crystal_system(int(dataset.number))
    if system == "cubic":
        choices = ["<100>", "<110>", "<111>", "<112>"]
        directions = [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 2]]
    else:
        choices = ["[001]", "[100]", "[110]", "[111]"]
        directions = [[0, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]]
    for first, second in combinations(directions, 2):
        if directions_equivalent(first, second, dataset.rotations, atoms.cell.array):
            return None
    return choices, directions, int(dataset.number), system


def _weighted_count(shown, element, cell):
    fractional = shown.positions @ np.linalg.inv(cell)
    near_boundary = (
        (np.abs(fractional) <= BOUNDARY_TOLERANCE)
        | (np.abs(fractional - 1) <= BOUNDARY_TOLERANCE)
    )
    weights = 0.5 ** np.sum(near_boundary, axis=1)
    return float(weights[np.array(shown.get_chemical_symbols()) == element].sum())


def q4_plan(structures, target):
    """Balance real base-catalog crystal systems under the four-per-base cap."""
    pools = defaultdict(list)
    for name, atoms in structures.items():
        pools[crystal_system(int(symmetry_dataset(atoms).number))].append(name)
    systems = sorted(pools)
    minimum = min(6, len(systems), target)
    if target >= 80 and len(systems) < 6:
        raise ValueError("Q4 requires at least six real catalog crystal systems.")
    candidates = []
    for size in range(minimum, len(systems) + 1):
        if size > target:
            continue
        for selected in combinations(systems, size):
            each = min(target // size, min(4 * len(pools[kind]) for kind in selected))
            candidates.append((size * each, size, tuple(selected), each))
    total, _, selected, each = max(candidates)
    return {kind: each for kind in selected}, dict(pools), total


def preflight_catalog(structures, target=100):
    plan, pools, q4_total = q4_plan(structures, target)
    reliable = [name for name, atoms in structures.items() if atoms.info.get("q2_eligible", True)]
    direction_bases = [name for name, atoms in structures.items() if q6_directions(atoms) is not None]
    return {
        "base_structure_count": len(structures),
        "catalog_crystal_systems": {kind: len(names) for kind, names in sorted(pools.items())},
        "per_qtype_base_cap": 4, "q2_per_difficulty_base_cap": 2,
        "q4_plan": plan, "q4_target": q4_total,
        "q2_eligible_base_count": len(reliable),
        "q2_maximum_per_difficulty": 2 * len(reliable),
        "q2_excluded_bases": [name for name in structures if name not in reliable],
        "q6_eligible_base_count": len(direction_bases),
        "q6_maximum_questions": 4 * len(direction_bases),
        "q6_symmetry_excluded_bases": [name for name in structures if name not in direction_bases],
        "general_maximum_per_qtype": 4 * len(structures),
    }


def generate_count_system_direction(structures, store, target, seed):
    """Round-robin base coverage, matched counts, and balanced real symmetry."""
    if target > 4 * len(structures):
        raise ValueError("Target exceeds the four-per-base question cap.")
    rng = random.Random(stable_seed(seed, "count-system-direction"))
    names = list(structures)
    rng.shuffle(names)
    counts = {"Q1a": [], "Q1b": [], "Q4": [], "Q6": []}
    records = []
    record_cache = {}
    rejected = Counter()

    def scene_for(name, variant):
        key = (name, variant)
        if key not in record_cache:
            atoms, augmentation = variant_atoms(structures[name], variant, seed, strain=False)
            scene = store.get(atoms, name, variant_key=f"origin-{variant}")
            record_cache[key] = (name, atoms, scene, augmentation)
            records.append(record_cache[key])
        return record_cache[key]

    coverage = Counter()
    cursors = Counter()
    pending = defaultdict(list)
    seen = set()
    while len(counts["Q1a"]) < target:
        progress = False
        for name in names:
            if coverage[name] >= 4 or len(counts["Q1a"]) >= target:
                continue
            while not pending[name] and cursors[name] < 20:
                variant = cursors[name]
                cursors[name] += 1
                atoms, augmentation = variant_atoms(structures[name], variant, seed, strain=False)
                geometry = geometry_key(atoms)
                shown, _, _, views = _prepare_views(
                    atoms, store.directions, "cell", (2, 2, 2), None, None, store.tilts
                )
                symbols = np.array(shown.get_chemical_symbols())
                union = np.max([view.visible for view in views[:3]], axis=0)
                for element in sorted(set(symbols)):
                    signature = (name, geometry, element)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    mask = symbols == element
                    if not np.all(union[mask] >= 0.5):
                        rejected["count_element_not_visible"] += 1
                        continue
                    pending[name].append((variant, element, int(mask.sum()),
                                          atoms.get_chemical_symbols().count(element),
                                          _weighted_count(shown, element, atoms.cell.array),
                                          float(union[mask].min())))
            if not pending[name]:
                continue
            variant, element, displayed, cell_count, weighted, visible = pending[name].pop(0)
            _, atoms, scene, augmentation = scene_for(name, variant)
            if not np.isclose(weighted, cell_count, atol=1e-7):
                raise RuntimeError(f"Boundary sharing mismatch: {name}")
            common = {
                "element": element, "paired_count_id": f"{scene['scene_id']}:{element}",
                "display_count": displayed, "cell_count": cell_count,
                "boundary_weighted_count": weighted, "minimum_union_visibility": visible,
                "augmentation": augmentation,
            }
            counts["Q1a"].append(make_draft(
                "Q1a", "perception", name, scene,
                f"The three images show the same finite cell. Including the displayed "
                f"periodic copies on its boundary, how many {element} atoms are drawn? "
                "Do not count the same atom more than once across views.",
                numeric_choices(displayed, rng), str(displayed),
                {**common, "text_applicable": False},
            ))
            choices = numeric_choices(cell_count, rng)
            if displayed != cell_count and str(displayed) not in choices:
                choices[-1] = str(displayed)
            counts["Q1b"].append(make_draft(
                "Q1b", "chemical_reasoning", name, scene,
                "Account for boundary sharing: a corner atom contributes 1/8, "
                "an edge atom 1/4, a face atom 1/2, and an interior atom 1. "
                f"How many {element} atoms belong to one outlined unit cell?",
                choices, str(cell_count), {**common, "text_applicable": True},
            ))
            coverage[name] += 1
            progress = True
        if not progress:
            raise RuntimeError(f"Q1 reaches only {len(counts['Q1a'])}/{target} with base cap 4.")

    plan, pools, _ = q4_plan(structures, target)
    for kind in sorted(plan):
        pool = list(pools[kind])
        rng.shuffle(pool)
        for ordinal in range(plan[kind]):
            name = pool[ordinal % len(pool)]
            variant = ordinal // len(pool)
            _, atoms, scene, augmentation = scene_for(name, variant)
            choices = [kind, *rng.sample([item for item in CRYSTAL_SYSTEMS if item != kind], 3)]
            counts["Q4"].append(make_draft(
                "Q4", "symmetry", name, scene,
                "Which crystal system describes the symmetry of this periodic structure?",
                choices, kind,
                {"spacegroup_number": int(symmetry_dataset(atoms).number),
                 "augmentation": augmentation, "balanced_crystal_system": kind},
            ))

    eligible = [(name, q6_directions(structures[name])) for name in names]
    eligible = [(name, info) for name, info in eligible if info is not None]
    rejected["direction_options_symmetry_equivalent"] = len(names) - len(eligible)
    if 4 * len(eligible) < target:
        raise RuntimeError(f"Only {len(eligible)} bases have four non-equivalent Q6 options.")
    strict_cache = {}
    # One direction per base in each round covers the catalog before repeating it.
    for round_index in range(4):
        for name, info in eligible:
            if len(counts["Q6"]) >= target:
                break
            _, atoms, scene, augmentation = scene_for(name, 0)
            choices, directions, sg, system = info
            if name not in strict_cache:
                strict_cache[name] = store.ensure_strict(scene, atoms, directions)
            direction_index = (round_index + stable_seed(seed, name, "q6-offset")) % 4
            answer = choices[direction_index]
            counts["Q6"].append(make_draft(
                "Q6", "perception", name, scene,
                "No extra tilt is applied. Which crystallographic direction family "
                "contains the viewing direction?" if system == "cubic" else
                "No extra tilt is applied. Which crystallographic direction is the camera viewing along?",
                choices, answer,
                {"text_applicable": False, "direction_uvw": directions[direction_index],
                 "options_directions": dict(zip(choices, directions)),
                 "spacegroup_number": sg, "symmetry_inequivalent_verified": True,
                 "tilt": 0, "augmentation": augmentation},
                views=[strict_cache[name][direction_index]],
            ))
    print("Base scenes:", {key: len(value) for key, value in counts.items()}, flush=True)
    return sum(counts.values(), []), records, dict(rejected)

def species_assignment_equivalent(first, second) -> bool:
    """Reject global species swaps equivalent under undecorated-site symmetry.

    For this Q5 mutation the cell and all sites are unchanged. The undecorated
    site's spglib group enumerates all candidate lattice rotations/translations.
    This catches the otherwise mislabeled NaCl/CsCl global sublattice swaps.
    """
    if Counter(first.numbers) != Counter(second.numbers):
        return False
    if not np.allclose(first.cell, second.cell, atol=1e-8):
        return False
    a, b = first.get_scaled_positions(), second.get_scaled_positions()
    operations = spglib.get_symmetry(
        (first.cell.array, a, np.ones(len(first), dtype=int)), symprec=SYMPREC
    )
    if operations is None:
        raise ValueError("Cannot check species-swap equivalence.")
    for rotation, translation in zip(operations["rotations"], operations["translations"]):
        transformed = a @ rotation.T + translation
        matched = True
        for number, point in zip(first.numbers, transformed):
            candidates = b[second.numbers == number]
            delta = candidates - point
            delta -= np.rint(delta)
            if not np.any(np.linalg.norm(delta @ first.cell.array, axis=1) < 1e-5):
                matched = False
                break
        if matched:
            return True
    return False


def _q5_mutation(kind, name, atoms, structures, rng, variant, seed):
    changed = atoms.copy()
    details = {}
    if kind == "swap_elements":
        pairs = list(combinations(sorted(set(atoms.get_chemical_symbols())), 2))
        rng.shuffle(pairs)
        for first, second in pairs:
            symbols = atoms.get_chemical_symbols()
            changed.set_chemical_symbols([
                second if symbol == first else first if symbol == second else symbol
                for symbol in symbols
            ])
            if not species_assignment_equivalent(atoms, changed):
                details = {"swapped_elements": [first, second],
                           "symmetry_equivalent_swap_rejected": True}
                return changed, details
        return None
    if kind == "delete_atom":
        populations = Counter(atoms.get_chemical_symbols())
        eligible = [i for i, atom in enumerate(atoms) if populations[atom.symbol] >= 2]
        if not eligible:
            return None
        index = rng.choice(eligible)
        details = {"deleted_original_index": index, "deleted_element": atoms[index].symbol,
                   "deleted_fractional_position": atoms.get_scaled_positions()[index].tolist()}
        del changed[index]
    elif kind == "stretch_axis":
        axis = rng.randrange(3)
        cell = changed.cell.array.copy()
        cell[axis] *= 1.08
        changed.set_cell(cell, scale_atoms=True)
        details = {"stretched_axis": axis, "stretch_factor": 1.08,
                   "fractional_positions_unchanged": True}
    elif kind == "polymorph":
        alternate = POLYMORPHS.get(name)
        if alternate is None:
            return None
        changed, augmentation = variant_atoms(structures[alternate], variant, seed)
        details = {"other_structure": alternate, "other_augmentation": augmentation}
    return changed, details


def generate_q5(structures, store, records, target, seed):
    if target % 2:
        raise ValueError("Q5 target must be even to ensure an exact 1:1 ratio.")
    rng = random.Random(stable_seed(seed, "same-structure"))
    tasks = ["positive"] * (target // 2)
    kinds = ["swap_elements", "delete_atom", "stretch_axis", "polymorph"]
    tasks += [kinds[index % len(kinds)] for index in range(target // 2)]
    rng.shuffle(tasks)
    # Reserve the limited polymorph capacity before allocating general samples.
    tasks.sort(key=lambda kind: 0 if kind == "polymorph" else 1)
    records_by_base = defaultdict(list)
    for record in records:
        records_by_base[record[0]].append(record)
    for pool in records_by_base.values():
        rng.shuffle(pool)
    coverage = Counter()
    used, result = set(), []
    for ordinal, kind in enumerate(tasks):
        accepted = None
        available = list(POLYMORPHS) if kind == "polymorph" else list(structures)
        rng.shuffle(available)
        available.sort(key=lambda name: coverage[name])
        for attempt in range(600):
            name = available[attempt % len(available)]
            if coverage[name] >= 4:
                continue
            if kind == "positive" and records_by_base[name]:
                name, atoms, first_scene, augmentation = records_by_base[name].pop()
            else:
                variant = coverage[name] + attempt // len(available)
                atoms, augmentation = variant_atoms(structures[name], variant, seed)
                first_scene = None
            identity = (kind, name, geometry_key(atoms))
            if identity in used:
                continue
            if kind == "positive":
                changed, mutation = atoms, {}
            else:
                candidate = _q5_mutation(
                    kind, name, atoms, structures, rng, augmentation["variant_index"], seed
                )
                if candidate is None:
                    continue
                changed, mutation = candidate
            accepted = name, atoms, changed, first_scene, augmentation, mutation, identity
            break
        if accepted is None:
            raise RuntimeError(f"No valid Q5 candidate for {kind}.")
        name, atoms, changed, first_scene, augmentation, mutation, identity = accepted
        used.add(identity)
        coverage[name] += 1
        if first_scene is None:
            first_scene = store.get(atoms, name, variant_key=f"q5-source-{augmentation['variant_index']}")
        second_scene = first_scene if kind == "positive" else store.get(
            changed, name, variant_key=f"q5-{kind}-{ordinal}"
        )
        answer = Q5_OPTIONS[0 if kind == "positive" else
                            1 if kind == "swap_elements" else
                            3 if kind == "stretch_axis" else 2]
        selected = [
            {**view, "scene_id": first_scene["scene_id"]} for view in first_scene["views"][:3]
        ] + [
            {**view, "scene_id": second_scene["scene_id"]} for view in second_scene["views"][3:6]
        ]
        result.append(make_draft(
            "Q5", "comparison", name, first_scene, Q5_QUESTION, Q5_OPTIONS, answer,
            {"negative_type": None if kind == "positive" else kind,
             "is_same_structure": kind == "positive",
             "secondary_scene_id": second_scene["scene_id"],
             "view_groups": [{"label": "Group 1", "image_indices": [0, 1, 2]},
                             {"label": "Group 2", "image_indices": [3, 4, 5]}],
             "augmentation": augmentation, **mutation},
            text_repr="Group 1:\n" + first_scene["text_repr"] + "\n\nGroup 2:\n" + second_scene["text_repr"],
            views=selected,
        ))
        if (ordinal + 1) % 20 == 0:
            print(f"Q5: {ordinal + 1}/{target}", flush=True)
    return result


def finalize_questions(drafts, seed):
    rng = random.Random(stable_seed(seed, "answer-order"))
    groups = defaultdict(list)
    for draft in drafts:
        groups[draft["qtype"]].append(draft)
    finalized = []
    for qtype, group in sorted(groups.items()):
        positions = [index % 4 for index in range(len(group))]
        rng.shuffle(positions)
        for index, (draft, correct_position) in enumerate(zip(group, positions)):
            item = dict(draft)
            choices = list(item.pop("_choices"))
            answer = item["answer_text"]
            incorrect = [choice for choice in choices if choice != answer]
            rng.shuffle(incorrect)
            incorrect.insert(correct_position, answer)
            item.update(
                id=f"mvcb-{qtype.lower()}-{index:04d}",
                options=dict(zip("ABCD", incorrect)), answer="ABCD"[correct_position],
            )
            finalized.append(item)
    rng.shuffle(finalized)
    return finalized


def validate_questions(questions, store):
    """Fail before delivery if references, labels, pairing, or prompt safety fail."""
    ids, signatures = set(), set()
    paired = defaultdict(dict)
    base_counts = Counter()
    q2_base_counts = Counter()
    forbidden = list(get_structures()) + [
        "岩盐", "氯化铯型", "闪锌矿", "纤锌矿", "金红石", "锐钛矿",
        "钙钛矿", "萤石", "尖晶石", "刚玉", "黄铁矿", "金刚石", "石墨",
        "Fm-3m", "Fd-3m", "P4_2/mnm", "space group", "空间群",
        "rutile", "anatase", "perovskite", "fluorite", "spinel", "sphalerite",
        "wurtzite", "pyrite", "corundum", "diamond", "graphite",
    ]
    for item in questions:
        if item["id"] in ids:
            raise ValueError("Duplicate question ID.")
        ids.add(item["id"])
        base = item["meta"].get("base_structure")
        if not base or base != item["structure"]:
            raise ValueError("Every question must identify its internal base_structure.")
        base_counts[item["qtype"], base] += 1
        if base_counts[item["qtype"], base] > 4:
            raise ValueError(f"More than four {item['qtype']} questions from {base}.")
        if item["qtype"] == "Q2":
            q2_base_counts[item["difficulty"], base] += 1
            if q2_base_counts[item["difficulty"], base] > 2:
                raise ValueError("Q2 allows at most two questions per base per difficulty.")
        if set(item["options"]) != set("ABCD") or len(set(item["options"].values())) != 4:
            raise ValueError("Invalid four-choice options.")
        if item["options"][item["answer"]] != item["answer_text"]:
            raise ValueError("Answer text and letter disagree.")
        if len(item["view_ids"]) != len(item["image_paths"]):
            raise ValueError("Image references do not match.")
        for reference, path in zip(item["view_ids"], item["image_paths"]):
            sid, vid = reference.split(":", 1)
            scene = store.scenes[sid]
            lookup = {v["view_id"]: v for v in scene["views"] + scene["strict_views"]}
            if lookup[vid]["image_path"] != path or not (store.root / path).is_file():
                raise ValueError("Missing or inconsistent view reference.")
        if item["qtype"] in {"Q1a", "Q6"}:
            if item["text_repr"] or item["meta"]["text_applicable"]:
                raise ValueError("Visual-only task has a text shortcut.")
        prompt = model_prompt(item, include_text=True)
        if re.search(r"[\u4e00-\u9fff]", prompt):
            raise ValueError(f"Model-facing text must be English: {item['id']}.")
        if any(word.lower() in prompt.lower() for word in forbidden):
            raise ValueError(f"Internal structure identity leaked in {item['id']}.")
        # Q4's competing category options are necessary; no correct category
        # is stated in its question/background or in any other task's prompt.
        background = item["question"] + item["text_repr"]
        if any(system in background for system in CRYSTAL_SYSTEMS):
            raise ValueError("Crystal-system answer leaked in background text.")
        signature = (item["qtype"], tuple(item["image_paths"]), item["question"])
        if signature in signatures:
            raise ValueError("Duplicate question content.")
        signatures.add(signature)
        if item["qtype"] in {"Q1a", "Q1b"}:
            paired[item["meta"]["paired_count_id"]][item["qtype"]] = item
    for pair in paired.values():
        if set(pair) != {"Q1a", "Q1b"}:
            raise ValueError("Unpaired perceptual/stoichiometric count question.")
        if pair["Q1a"]["image_paths"] != pair["Q1b"]["image_paths"]:
            raise ValueError("Matched count tasks must use the same images.")


def statistics(questions, store):
    by_type = Counter(q["qtype"] for q in questions)
    q3 = [q for q in questions if q["qtype"] == "Q3"]
    q5 = [q for q in questions if q["qtype"] == "Q5"]
    structures = get_structures()
    systems = Counter(crystal_system(int(symmetry_dataset(atoms).number))
                      for atoms in structures.values())
    per_type = {}
    for kind in sorted(by_type):
        subset = [q for q in questions if q["qtype"] == kind]
        bases = Counter(q["meta"]["base_structure"] for q in subset)
        per_type[kind] = {
            "count": len(subset), "unique_base_structures": len(bases),
            "max_questions_per_base": max(bases.values()),
            "base_structure_counts": dict(sorted(bases.items())),
            "answer_text_distribution": dict(sorted(Counter(q["answer_text"] for q in subset).items())),
            "answer_letter_distribution": dict(sorted(Counter(q["answer"] for q in subset).items())),
        }
    return {
        "total_questions": len(questions), "by_qtype": dict(sorted(by_type.items())),
        "q2_difficulties": dict(Counter(q["difficulty"] for q in questions if q["qtype"] == "Q2")),
        "answer_letters": dict(sorted(Counter(q["answer"] for q in questions).items())),
        "answer_letters_by_qtype": {kind: values["answer_letter_distribution"]
                                     for kind, values in per_type.items()},
        "per_qtype": per_type,
        "q3_answer_text_distribution": dict(Counter(q["answer_text"] for q in q3)),
        "q3_depth_trap_count": sum(q["meta"]["depth_trap"] for q in q3),
        "q3_depth_trap_ratio": sum(q["meta"]["depth_trap"] for q in q3) / len(q3) if q3 else 0,
        "q3_depth_trap_definition": "2D order in default first view v00 is opposite the 3D order",
        "q4_crystal_system_distribution": dict(Counter(q["answer_text"] for q in questions
                                                       if q["qtype"] == "Q4")),
        "q5_positive": sum(q["meta"]["is_same_structure"] for q in q5),
        "q5_negative_types": dict(sorted(Counter(q["meta"]["negative_type"] for q in q5
                                                   if q["meta"]["negative_type"]).items())),
        "scene_count": len(store.scenes), "default_scene_images": 8 * len(store.scenes),
        "strict_axis_images": sum(len(s["strict_views"]) for s in store.scenes.values()),
        "source_structure_count": len({q["meta"]["base_structure"] for q in questions}),
        "catalog_structure_count": len(structures),
        "catalog_crystal_system_distribution": dict(sorted(systems.items())),
    }


def _diverse_sample(pool, count, rng, usage=None):
    """Greedily spread a fixed-seed sample across bases before repeating them."""
    if len(pool) < count:
        raise ValueError(f"Only {len(pool)} candidates for a {count}-question sample.")
    remaining = list(pool)
    rng.shuffle(remaining)
    bases = Counter() if usage is None else usage
    answers, letters, chosen = Counter(), Counter(), []
    while len(chosen) < count:
        index = min(range(len(remaining)), key=lambda i: (
            bases[remaining[i]["meta"]["base_structure"]],
            answers[remaining[i]["answer_text"]], letters[remaining[i]["answer"]], i,
        ))
        item = remaining.pop(index)
        chosen.append(item)
        bases[item["meta"]["base_structure"]] += 1
        answers[item["answer_text"]] += 1
        letters[item["answer"]] += 1
    return chosen


def _stratified_sample(pool, count, key, rng):
    strata = defaultdict(list)
    for item in pool:
        strata[key(item)].append(item)
    keys = list(strata)
    rng.shuffle(keys)
    usage = Counter()
    chosen = []
    for ordinal, bucket in enumerate(keys):
        quota = count // len(keys) + (ordinal < count % len(keys))
        chosen.extend(_diverse_sample(strata[bucket], quota, rng, usage))
    return chosen


def make_mini(questions, seed=DEFAULT_SEED):
    """200 questions: 25 per task, with Q2 easy and hard as separate tasks."""
    rng = random.Random(stable_seed(seed, "mini-200"))
    groups = defaultdict(list)
    for item in questions:
        groups[item["qtype"]].append(item)
    if any(len(groups[kind]) < 25 for kind in ("Q1a", "Q1b", "Q3", "Q4", "Q5", "Q6")):
        return []  # Small smoke datasets intentionally do not pretend to be 200 rows.
    if any(sum(q["difficulty"] == difficulty for q in groups["Q2"]) < 25
           for difficulty in ("easy", "hard")):
        return []
    first = _diverse_sample(groups["Q1a"], 25, rng)
    paired = {q["meta"]["paired_count_id"]: q for q in groups["Q1b"]}
    mini = first + [paired[q["meta"]["paired_count_id"]] for q in first]
    for difficulty in ("easy", "hard"):
        mini.extend(_diverse_sample([q for q in groups["Q2"] if q["difficulty"] == difficulty], 25, rng))
    mini.extend(_stratified_sample(groups["Q3"], 25,
                                   lambda q: (q["meta"]["depth_trap"], q["answer_text"]), rng))
    mini.extend(_stratified_sample(groups["Q4"], 25, lambda q: q["answer_text"], rng))
    mini.extend(_diverse_sample([q for q in groups["Q5"] if q["meta"]["is_same_structure"]], 13, rng))
    usage = Counter(q["meta"]["base_structure"] for q in mini if q["qtype"] == "Q5")
    for kind in ("swap_elements", "delete_atom", "stretch_axis", "polymorph"):
        mini.extend(_diverse_sample([q for q in groups["Q5"] if q["meta"]["negative_type"] == kind],
                                    3, rng, usage))
    mini.extend(_diverse_sample(groups["Q6"], 25, rng))
    rng.shuffle(mini)
    if len(mini) != 200 or len({q["id"] for q in mini}) != 200:
        raise RuntimeError("The mini dataset must have exactly 200 distinct questions.")
    return mini

def generate_dataset(out_dir="data", target=100, seed=DEFAULT_SEED):
    if target < 4 or target % 2:
        raise ValueError("--target must be an even integer >= 4.")
    from question_coordination import generate_q2
    from question_distance import generate_q3

    structures = get_structures()
    preflight = preflight_catalog(structures, target)
    print("Catalog preflight:", json.dumps(preflight, ensure_ascii=False), flush=True)
    store = SceneStore(out_dir, seed)
    drafts, records, rejections = generate_count_system_direction(structures, store, target, seed)
    store.save_manifest()
    print("Generating Q2...", flush=True)
    per_difficulty = min(max(2, round(target * 0.6)),
                         preflight["q2_maximum_per_difficulty"])
    drafts += generate_q2(structures, store, per_difficulty, per_difficulty, seed)
    store.save_manifest()
    print("Generating Q3...", flush=True)
    drafts += generate_q3(structures, store, target, seed)
    store.save_manifest()
    print("Generating Q5...", flush=True)
    drafts += generate_q5(structures, store, records, target, seed)
    questions = finalize_questions(drafts, seed)
    validate_questions(questions, store)
    store.save_manifest()
    question_path = store.root / "questions.jsonl"
    with question_path.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False, allow_nan=False) + "\n")
    stats = statistics(questions, store)
    stats.update(seed=seed, selection_rejections=rejections, catalog_preflight=preflight)
    mini = make_mini(questions, seed)
    if mini:
        validate_questions(mini, store)
        with (store.root / "questions_mini.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for question in mini:
                handle.write(json.dumps(question, ensure_ascii=False, allow_nan=False) + "\n")
        mini_stats = statistics(mini, store)
        (store.root / "stats_mini.json").write_text(
            json.dumps(mini_stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stats["mini_total_questions"] = len(mini)
        stats["mini_unique_bases_by_qtype"] = {
            kind: values["unique_base_structures"] for kind, values in mini_stats["per_qtype"].items()
        }
    (store.root / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_path = write_review(questions, store.root, seed)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(f"Questions: {question_path}\nReview: {review_path}", flush=True)
    return questions, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--target", type=int, default=100, help="Target per type; Q2 uses 1.2x.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--preflight", action="store_true",
                        help="Inspect catalog capacity and balanced Q4 targets without rendering.")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight_catalog(get_structures(), args.target), indent=2))
    else:
        generate_dataset(args.out_dir, args.target, args.seed)


if __name__ == "__main__":
    main()
