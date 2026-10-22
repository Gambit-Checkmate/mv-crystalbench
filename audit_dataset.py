"""Independent on-disk audit of generated scenes and multiple-choice questions.

This module deliberately does not import questions.py or its validation code.
Periodic neighbor counts and symmetry answers are recomputed from scene geometry.
Run: python audit_dataset.py --data-dir data
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
import spglib


REQUIRED = {
    "id", "qtype", "difficulty", "structure", "scene_id", "view_ids",
    "image_paths", "question", "options", "answer", "answer_text", "meta", "text_repr",
}
MINERAL_WORDS = (
    "金红石", "锐钛矿", "钙钛矿", "萤石", "尖晶石", "闪锌矿", "纤锌矿",
    "黄铁矿", "刚玉", "金刚石", "石墨", "rutile", "anatase", "perovskite",
    "fluorite", "spinel", "sphalerite", "wurtzite", "pyrite", "corundum", "diamond",
)
SYSTEM_NAMES = {
    "triclinic": "三斜", "monoclinic": "单斜", "orthorhombic": "正交",
    "tetragonal": "四方", "trigonal": "三方", "hexagonal": "六方", "cubic": "立方",
}


def _load_lines(path: Path) -> list[dict]:
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return [
        json.loads(line, parse_constant=reject_constant)
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _atoms(scene: dict) -> Atoms:
    source = scene["atoms"]
    result = Atoms(
        numbers=source["atomic_numbers"], positions=source["positions_angstrom"],
        cell=source["lattice_vectors_angstrom"], pbc=source["periodic"],
    )
    result.info["bond_pairs"] = source["bond_pairs"]
    return result


def _system(atoms: Atoms) -> str:
    dataset = spglib.get_symmetry_dataset(
        (atoms.cell.array, atoms.get_scaled_positions(wrap=True), atoms.numbers),
        symprec=1e-5,
    )
    if dataset is None:
        raise ValueError("spglib could not classify a scene.")
    number = int(dataset.number)
    for maximum, name in (
        (2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
        (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic"),
    ):
        if number <= maximum:
            return name
    raise ValueError(f"Invalid space group number {number}")


def _periodic_ground_truth(atoms: Atoms) -> tuple[dict, np.ndarray]:
    """Recompute graph without using the shared bonding module.

    Every wrapped pair has a periodic image with all fractional components
    within +/-0.5, hence distance <= 1.5*max(cell-vector-length). The slightly
    larger search radius also includes 1.10 times that guaranteed upper bound.
    """
    allowed = {tuple(sorted(pair)) for pair in atoms.info["bond_pairs"]}
    radius = 2.0 * max(atoms.cell.lengths()) + 1e-8
    ii, jj, dd = neighbor_list("ijd", atoms, radius, self_interaction=False)
    symbols = atoms.get_chemical_symbols()
    minima = {pair: np.inf for pair in allowed}
    edges = []
    for i, j, distance in zip(ii, jj, dd):
        pair = tuple(sorted((symbols[i], symbols[j])))
        if pair in allowed and distance > 1e-9:
            minima[pair] = min(minima[pair], float(distance))
            edges.append((int(i), pair, float(distance)))
    minima = {pair: distance for pair, distance in minima.items() if np.isfinite(distance)}
    degrees = np.zeros(len(atoms), dtype=int)
    for index, pair, distance in edges:
        if distance <= 1.10 * minima[pair] + 1e-9:
            degrees[index] += 1
    return minima, degrees


def _number(text: Any) -> int:
    if isinstance(text, (int, np.integer)):
        return int(text)
    match = re.fullmatch(r"\s*(\d+)\s*(?:个|个原子)?\s*", str(text))
    if not match:
        raise ValueError(f"Expected an integer answer, got {text!r}")
    return int(match.group(1))


def _direction(text: str) -> np.ndarray:
    match = re.search(r"[\[<〈⟨]\s*([-\d\s,]+)\s*[\]>〉⟩]", text)
    if not match:
        raise ValueError(f"No crystal direction in {text!r}")
    content = match.group(1).strip()
    if re.fullmatch(r"\d{3}", content):
        values = [int(number) for number in content]
    else:
        values = [int(number) for number in re.findall(r"-?\d+", content)]
    if len(values) != 3 or not any(values):
        raise ValueError(f"Invalid direction {text!r}")
    return np.asarray(values, dtype=float)


def _equivalent(first: np.ndarray, second: np.ndarray, atoms: Atoms) -> bool:
    symmetry = spglib.get_symmetry(
        (atoms.cell.array, atoms.get_scaled_positions(wrap=True), atoms.numbers),
        symprec=1e-5,
    )
    if symmetry is None:
        raise ValueError("spglib could not find direction equivalence operations.")
    target = second @ atoms.cell.array
    target /= np.linalg.norm(target)
    for rotation in symmetry["rotations"]:
        candidate = (rotation @ first) @ atoms.cell.array
        candidate /= np.linalg.norm(candidate)
        # Looking along +/- the same line cannot make two distinct options.
        if abs(float(np.dot(candidate, target))) >= 1 - 1e-8:
            return True
    return False


def _normalize_negative(value: str) -> str:
    value = str(value).lower()
    if any(token in value for token in ("swap", "exchange", "交换")):
        return "swap_elements"
    if any(token in value for token in ("delet", "remove", "missing", "删除")):
        return "delete_atom"
    if any(token in value for token in ("stretch", "strain", "lattice", "拉伸")):
        return "stretch_axis"
    if any(token in value for token in ("poly", "多型")):
        return "polymorph"
    if value in ("none", "positive", "same", ""):
        return "positive"
    return value


def _same_decoration(first: Atoms, second: Atoms) -> bool:
    """Detect element swaps that are merely symmetries of the same geometry."""
    if len(first) != len(second) or not np.allclose(first.cell.array, second.cell.array):
        return False
    fractional = first.get_scaled_positions(wrap=True)
    target = second.get_scaled_positions(wrap=True)
    symmetry = spglib.get_symmetry(
        (first.cell.array, fractional, np.ones(len(first), dtype=int)), symprec=1e-5,
    )
    if symmetry is None:
        raise ValueError("Could not obtain undecorated-site symmetry.")
    for rotation, translation in zip(symmetry["rotations"], symmetry["translations"]):
        transformed = fractional @ rotation.T + translation
        delta = transformed[:, None, :] - target[None, :, :]
        delta -= np.rint(delta)
        distances = np.linalg.norm(delta @ first.cell.array, axis=2)
        indices = np.argmin(distances, axis=1)
        if (len(set(indices)) == len(first)
                and np.all(distances[np.arange(len(first)), indices] < 1e-5)
                and np.array_equal(first.numbers, second.numbers[indices])):
            return True
    return False


def audit_questions(root: str | Path, questions_name: str = "questions.jsonl") -> dict[str, Any]:
    """Read generated files, return a report with errors instead of failing fast."""
    root = Path(root).expanduser().resolve()
    questions_sha256 = hashlib.sha256((root / questions_name).read_bytes()).hexdigest()
    questions = _load_lines(root / questions_name)
    is_mini = Path(questions_name).name == "questions_mini.jsonl"
    scene_list = _load_lines(root / "scenes.jsonl")
    scenes = {scene["scene_id"]: scene for scene in scene_list}
    from structures import get_structures
    catalog = get_structures()
    errors: list[str] = []
    warnings: list[str] = []
    checks = 0
    views: dict[str, tuple[dict, dict]] = {}
    geometry_cache = {}

    def check(condition: bool, message: str) -> bool:
        nonlocal checks
        checks += 1
        if not condition:
            errors.append(message)
        return bool(condition)

    def ground(scene):
        key = hashlib.sha256(
            json.dumps(scene["atoms"], sort_keys=True).encode()
        ).hexdigest()
        if key not in geometry_cache:
            atoms = _atoms(scene)
            geometry_cache[key] = (atoms, *_periodic_ground_truth(atoms))
        return geometry_cache[key]

    check(bool(questions), "questions.jsonl is empty")
    check(len(scenes) == len(scene_list), "Duplicate scene IDs")
    for scene_id, scene in scenes.items():
        prefix = f"scene {scene_id}"
        check(len(scene.get("views", [])) == 8, f"{prefix}: expected 8 default views")
        check(
            [view["view_id"] for view in scene["views"]] == [f"v{i:02d}" for i in range(8)],
            f"{prefix}: default view IDs must be v00-v07",
        )
        check(
            np.allclose([view["tilt"] for view in scene["views"]], [8, 8, 8, 0, 0, 0, 0, 0]),
            f"{prefix}: wrong default tilts",
        )
        try:
            atoms, minima, _ = ground(scene)
            display = scene["display_atoms"]
            positions = np.asarray([item["position_angstrom"] for item in display])
            symbols = [item["element"] for item in display]
            original_symbols = atoms.get_chemical_symbols()
            celldisp = np.asarray(scene["atoms"].get(
                "cell_display_displacement_angstrom", [0, 0, 0]
            ))
            for index, item in enumerate(display):
                original = item["original_index"]
                check(item["display_index"] == index, f"{prefix}: non-contiguous display mapping")
                check(item["element"] == original_symbols[original],
                      f"{prefix}: display/original element mismatch")
                expected_position = (
                    atoms.positions[original] + np.asarray(item["translation"]) @ atoms.cell.array
                    - celldisp
                )
                check(np.allclose(positions[index], expected_position, atol=1e-8),
                      f"{prefix}: periodic-image position mismatch")
            expected_bonds = set()
            for first in range(len(display)):
                for second in range(first + 1, len(display)):
                    pair = tuple(sorted((symbols[first], symbols[second])))
                    distance = np.linalg.norm(positions[first] - positions[second])
                    if pair in minima and 1e-9 < distance <= 1.10 * minima[pair] + 1e-9:
                        expected_bonds.add((first, second))
            for view in scene["views"] + scene.get("strict_views", []):
                ref = f"{scene_id}:{view['view_id']}"
                check(ref not in views, f"{prefix}: duplicate view reference {ref}")
                views[ref] = (scene, view)
                metadata = view["metadata"]
                check(len(metadata["atoms"]) == len(display), f"{ref}: wrong atom count")
                rotation = np.asarray(metadata["camera"]["rotation_columns"], dtype=float)
                projected = positions @ rotation
                screen = np.asarray([item["screen_xy_px"] for item in metadata["atoms"]])
                pixel_scale = float(metadata["camera"]["pixels_per_angstrom"])
                offsets = screen - projected[:, :2] * [pixel_scale, -pixel_scale]
                check(np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
                      and np.isclose(np.linalg.det(rotation), 1),
                      f"{ref}: camera rotation is not orthonormal")
                check(pixel_scale > 0 and np.allclose(offsets, offsets[0], atol=1e-7),
                      f"{ref}: pixel coordinates disagree with camera projection")
                depths = np.asarray([item["depth"] for item in metadata["atoms"]])
                check(np.allclose(depths, projected[:, 2] - projected[:, 2].mean(), atol=1e-8),
                      f"{ref}: recorded depths disagree with 3D projection")
                if view["view_id"].startswith("vS"):
                    check(view["tilt"] == 0, f"{ref}: strict view is tilted")
                actual_bonds = {
                    tuple(sorted(bond["display_indices"])) for bond in metadata["bonds"]
                }
                check(actual_bonds == expected_bonds, f"{ref}: rendered bond graph mismatch")
                for key in ("image_path", "metadata_path"):
                    path = (root / view[key]).resolve()
                    check(path.is_relative_to(root), f"{ref}: path escapes data directory")
                    check(path.is_file(), f"{ref}: missing {key}")
                    check(path.stem == view["view_id"], f"{ref}: filename is not anonymous")
                image_path = root / view["image_path"]
                if image_path.is_file():
                    with image_path.open("rb") as image_handle:
                        header = image_handle.read(24)
                    check(header[:8] == b"\x89PNG\r\n\x1a\n", f"{ref}: invalid PNG signature")
                    if len(header) >= 24:
                        check(struct.unpack(">II", header[16:24]) == (512, 512),
                              f"{ref}: expected 512x512 PNG")
                metadata_path = root / view["metadata_path"]
                if metadata_path.is_file():
                    check(json.loads(metadata_path.read_text(encoding="utf-8")) == metadata,
                          f"{ref}: sidecar disagrees with scene manifest")
                for item in metadata["atoms"]:
                    check(0 <= item["visible_fraction"] <= 1, f"{ref}: invalid visibility")
                    check(len(item["screen_xy_px"]) == 2, f"{ref}: invalid screen coordinates")
                    check(item["radius_px"] * 2 <= 512 / 10 + 1e-8,
                          f"{ref}: atom diameter exceeds 1/10 frame")
            for word in MINERAL_WORDS:
                check(word not in scene["text_repr"].lower(),
                      f"{prefix}: mineral name leaked in text_repr: {word}")
            check("spacegroup" not in scene["text_repr"].lower(),
                  f"{prefix}: space group leaked in text_repr")
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            errors.append(f"{prefix}: malformed scene: {exc}")

    counts = Counter()
    letters = Counter()
    type_letters = {}
    q3_traps = []
    q5_negative_types = Counter()
    count_pairs = {"Q1a": set(), "Q1b": set()}
    hard_q2_visibility = []
    base_counts = Counter()
    q2_base_counts = Counter()
    answer_texts = {}
    seen_ids = set()
    for question in questions:
        prefix = f"question {question.get('id', '<missing>')}"
        if not check(REQUIRED <= set(question), f"{prefix}: missing required fields"):
            continue
        qtype = str(question["qtype"])
        counts[qtype] += 1
        answer_texts.setdefault(qtype, Counter())[question["answer_text"]] += 1
        base = question["meta"].get("base_structure")
        check(base == question["structure"], f"{prefix}: missing/inconsistent meta.base_structure")
        base_counts[qtype, base] += 1
        check(base_counts[qtype, base] <= 4, f"{prefix}: base contributes over four {qtype} questions")
        if qtype == "Q2":
            check(base in catalog and catalog[base].info.get("q2_eligible", True),
                  f"{prefix}: Q2 uses an excluded or unknown base structure")
            q2_base_counts[question["difficulty"], base] += 1
            check(q2_base_counts[question["difficulty"], base] <= 2,
                  f"{prefix}: base contributes over two Q2 questions of this difficulty")
        letters[question["answer"]] += 1
        type_letters.setdefault(qtype, Counter())[question["answer"]] += 1
        check(question["id"] not in seen_ids, f"{prefix}: duplicate ID")
        seen_ids.add(question["id"])
        check(question["answer"] in "ABCD", f"{prefix}: invalid answer letter")
        options = question["options"]
        if not check(isinstance(options, Mapping) and set(options) == set("ABCD"),
                     f"{prefix}: options must map A-D to four choices"):
            continue
        check(len(set(str(value).strip() for value in options.values())) == 4,
              f"{prefix}: duplicate option text")
        check(options.get(question["answer"]) == question["answer_text"],
              f"{prefix}: answer text disagrees with option")
        check(not re.search(r"[\u4e00-\u9fff]", " ".join([
            str(question["question"]), str(question["answer_text"]), *map(str, options.values()),
            str(question["text_repr"]),
        ])), f"{prefix}: model-facing fields are not English")
        check(len(question["view_ids"]) == len(question["image_paths"]),
              f"{prefix}: image and view lists differ in length")
        resolved = []
        for ref, image_path in zip(question["view_ids"], question["image_paths"]):
            if ":" not in ref:
                ref = f"{question['scene_id']}:{ref}"
            if check(ref in views, f"{prefix}: unknown view {ref}"):
                scene, view = views[ref]
                resolved.append((scene, view))
                check(view["image_path"] == image_path, f"{prefix}: view/image mismatch")
        if not resolved:
            continue
        main_scene = resolved[0][0]
        check(question["scene_id"] == main_scene["scene_id"],
              f"{prefix}: scene_id differs from first view")
        # Q4 necessarily offers crystal-system names, so do not forbid its options.
        for word in MINERAL_WORDS:
            check(word not in str(question["question"]).lower(),
                  f"{prefix}: mineral name leaked in question: {word}")
        for internal in {scene["structure"] for scene in scenes.values()}:
            if len(internal) >= 4:
                check(internal not in question["question"],
                      f"{prefix}: internal structure name leaked in question")
        meta = question["meta"]
        try:
            atoms, _, degrees = ground(main_scene)
            display = main_scene["display_atoms"]
            labels = {item["highlight"]: item["display_index"]
                      for item in display if item.get("highlight")}
            if qtype in {"Q1a", "Q1b"}:
                element = meta.get("element", meta.get("target_element"))
                if element is None:
                    raise ValueError("Q1 meta must identify the queried element")
                shown_indices = [i for i, item in enumerate(display) if item["element"] == element]
                expected = (len(shown_indices) if qtype == "Q1a"
                            else atoms.get_chemical_symbols().count(element))
                count_pairs[qtype].add((
                    main_scene["scene_id"], element, tuple(question["image_paths"]),
                ))
                check(_number(question["answer_text"]) == expected,
                      f"{prefix}: wrong {'display' if qtype == 'Q1a' else 'cell'} count")
                check(meta.get("display_count") == len(shown_indices),
                      f"{prefix}: meta display_count mismatch")
                check(meta.get("cell_count") == atoms.get_chemical_symbols().count(element),
                      f"{prefix}: meta cell_count mismatch")
                if main_scene["display"] == "cell":
                    fractional = (
                        np.asarray([display[i]["position_angstrom"] for i in shown_indices])
                        @ np.linalg.inv(atoms.cell.array)
                    )
                    on_boundary = np.isclose(fractional, 0, atol=1e-3) | np.isclose(
                        fractional, 1, atol=1e-3
                    )
                    weighted = float(np.sum(0.5 ** np.count_nonzero(on_boundary, axis=1)))
                    check(abs(weighted - atoms.get_chemical_symbols().count(element)) < 1e-6,
                          f"{prefix}: fractional boundary sharing does not recover cell count")
                if qtype == "Q1a":
                    check(all(max(view["metadata"]["atoms"][index]["visible_fraction"]
                                  for _, view in resolved) >= 0.5
                              for index in shown_indices),
                          f"{prefix}: at least one queried atom lacks visibility >=0.5")
                    check(question["text_repr"] == "" and meta.get("text_applicable") is False,
                          f"{prefix}: Q1a must disable text")
                else:
                    check(main_scene["display"] == "cell", f"{prefix}: Q1b requires cell display")
            elif qtype.startswith("Q2"):
                center = int(meta.get("center_display_index", labels.get("A", -1)))
                if center < 0:
                    raise ValueError("No highlighted Q2 center A")
                original = display[center]["original_index"]
                expected = int(degrees[original])
                check(_number(question["answer_text"]) == expected,
                      f"{prefix}: periodic coordination count mismatch")
                shown_degree = sum(
                    center in bond["display_indices"]
                    for bond in resolved[0][1]["metadata"]["bonds"]
                )
                check(shown_degree == expected, f"{prefix}: displayed coordination shell incomplete")
                check(meta.get("coordination_number") == expected,
                      f"{prefix}: meta coordination_number mismatch")
                check(all(view["metadata"]["atoms"][center]["visible_fraction"] >= 0.7
                          for view in main_scene["views"]),
                      f"{prefix}: Q2 center not >=0.7 visible in all eight views")
                neighbor_indices = {
                    index for bond in resolved[0][1]["metadata"]["bonds"]
                    if center in bond["display_indices"]
                    for index in bond["display_indices"] if index != center
                }
                if question["difficulty"] == "easy" or qtype.endswith("easy"):
                    check(main_scene["display"] == "environment",
                          f"{prefix}: easy Q2 must use environment")
                    check(len(display) == expected + 1,
                          f"{prefix}: easy environment has non-shell atoms")
                    check(all(max(view["metadata"]["atoms"][index]["visible_fraction"]
                                  for view in main_scene["views"][:3]) >= 0.5
                              for index in neighbor_indices),
                          f"{prefix}: easy Q2 neighbor is hidden in every default view")
                else:
                    neighbor_best = {
                        index: max(view["metadata"]["atoms"][index]["visible_fraction"]
                                   for view in main_scene["views"][:3])
                        for index in sorted(neighbor_indices)
                    }
                    hard_q2_visibility.append({
                        "id": question["id"], "neighbor_count": len(neighbor_best),
                        "neighbors_below_half_in_all_default_views": [
                            index for index, fraction in neighbor_best.items() if fraction < 0.5
                        ],
                        "minimum_neighbor_best_visibility": min(neighbor_best.values()),
                    })
            elif qtype == "Q3":
                check(set(options.values()) == {
                    "B is closer", "C is closer", "Equally close", "Cannot be determined",
                }, f"{prefix}: Q3 must use the four requested English option texts")
                first, second, third = [labels[label] for label in ("A", "B", "C")]
                positions = np.asarray([item["position_angstrom"] for item in display])
                db = float(np.linalg.norm(positions[first] - positions[second]))
                dc = float(np.linalg.norm(positions[first] - positions[third]))
                gap = abs(db - dc) / max(db, dc)
                check(gap >= 0.15 - 1e-8,
                      f"{prefix}: Q3 distance gap below 15%")
                check(np.isclose(meta["distance_AB_angstrom"], db)
                      and np.isclose(meta["distance_AC_angstrom"], dc)
                      and np.isclose(meta["relative_distance_gap"], gap),
                      f"{prefix}: Q3 recorded distances/gap mismatch")
                correct = "B" if db < dc else "C"
                check(correct in str(question["answer_text"]),
                      f"{prefix}: Q3 distance answer mismatch")
                default_view = main_scene["views"][0]
                xy = np.asarray([item["screen_xy_px"] for item in default_view["metadata"]["atoms"]])
                pb = float(np.linalg.norm(xy[first] - xy[second]))
                pc = float(np.linalg.norm(xy[first] - xy[third]))
                trap = (db - dc) * (pb - pc) < 0
                check(meta.get("depth_trap") is bool(trap), f"{prefix}: depth_trap mismatch in v00")
                for index in (first, second, third):
                    check(all(view["metadata"]["atoms"][index]["visible_fraction"] >= 0.7
                              for view in main_scene["views"]),
                          f"{prefix}: Q3 highlight not >=0.7 visible in all eight views")
                for projection in meta["projected_distances"]:
                    view = next(view for view in main_scene["views"]
                                if view["view_id"] == projection["view_id"])
                    coords = np.asarray([item["screen_xy_px"] for item in view["metadata"]["atoms"]])
                    distance_b = float(np.linalg.norm(coords[first] - coords[second]))
                    distance_c = float(np.linalg.norm(coords[first] - coords[third]))
                    check(np.isclose(distance_b, projection["distance_AB_px"])
                          and np.isclose(distance_c, projection["distance_AC_px"]),
                          f"{prefix}: recorded projected distances mismatch")
                    check(projection["opposite_to_3d"] is bool((db - dc) * (distance_b - distance_c) < 0),
                          f"{prefix}: recorded per-view trap mismatch")
                q3_traps.append(bool(trap))
            elif qtype == "Q4":
                system = _system(atoms)
                answer = str(question["answer_text"]).lower()
                check(system in answer or SYSTEM_NAMES[system] in answer,
                      f"{prefix}: wrong spglib crystal-system answer")
            elif qtype == "Q5":
                negative_type = _normalize_negative(meta.get(
                    "negative_type", meta.get("negative_sample_type", "")
                ))
                q5_negative_types[negative_type] += 1
                image_groups = meta.get("view_groups", [])
                check(len(image_groups) == 2, f"{prefix}: Q5 requires two image groups")
                check(len(resolved) == 6, f"{prefix}: Q5 requires 3+3 default images")
                if len(resolved) >= 6:
                    check([view["view_id"] for _, view in resolved[:3]] == ["v00", "v01", "v02"],
                          f"{prefix}: group1 must use the three tilted axes")
                    check(all(view["view_id"] in {"v03", "v04", "v05", "v06", "v07"}
                              for _, view in resolved[3:]),
                          f"{prefix}: group2 must use random directions")
                    first_scene, second_scene = resolved[0][0], resolved[3][0]
                    first_atoms = _atoms(first_scene)
                    second_atoms = _atoms(second_scene)
                    if negative_type == "positive":
                        check(first_scene["atoms"] == second_scene["atoms"],
                              f"{prefix}: positive Q5 source structures differ")
                        check("same structure" == str(question["answer_text"]).lower(),
                              f"{prefix}: positive Q5 wrong answer")
                    elif negative_type == "swap_elements":
                        check("element" in question["answer_text"].lower(), f"{prefix}: swap wrong answer category")
                        check(not _same_decoration(first_atoms, second_atoms),
                              f"{prefix}: swapped elements are symmetry-equivalent")
                    elif negative_type in {"delete_atom", "polymorph"}:
                        check("arrangement" in question["answer_text"].lower(),
                              f"{prefix}: deletion/polytype wrong answer category")
                        if negative_type == "delete_atom":
                            check(len(second_atoms) == len(first_atoms) - 1,
                                  f"{prefix}: deletion must remove exactly one original atom")
                        else:
                            check(
                                first_atoms.get_chemical_formula(empirical=True)
                                == second_atoms.get_chemical_formula(empirical=True),
                                f"{prefix}: polytypes do not have the same composition",
                            )
                    elif negative_type == "stretch_axis":
                        check("dimensions" in question["answer_text"].lower(),
                              f"{prefix}: stretch wrong answer category")
                        check(np.array_equal(first_atoms.numbers, second_atoms.numbers),
                              f"{prefix}: stretch changed atom identities")
                        check(np.allclose(first_atoms.get_scaled_positions(),
                                          second_atoms.get_scaled_positions(), atol=1e-8),
                              f"{prefix}: stretch changed fractional coordinates")
                        ratios = second_atoms.cell.lengths() / first_atoms.cell.lengths()
                        check(np.allclose(sorted(ratios), [1, 1, 1.08], atol=1e-8),
                              f"{prefix}: strain must stretch exactly one axis by 8%")
                    else:
                        check(False, f"{prefix}: unknown Q5 negative type {negative_type}")
            elif qtype == "Q6":
                check(len(resolved) == 1 and resolved[0][1]["tilt"] == 0,
                      f"{prefix}: Q6 requires one strict view")
                check(question["text_repr"] == "" and meta.get("text_applicable") is False,
                      f"{prefix}: Q6 must disable text")
                directions = {letter: _direction(str(text)) for letter, text in options.items()}
                for first in "ABCD":
                    for second in "ABCD":
                        if first < second:
                            check(not _equivalent(directions[first], directions[second], atoms),
                                  f"{prefix}: Q6 options {first}/{second} symmetry-equivalent")
                actual = np.asarray(resolved[0][1]["direction"], dtype=float)
                check(_equivalent(actual, directions[question["answer"]], atoms),
                      f"{prefix}: Q6 answer does not match camera direction")
            else:
                check(False, f"{prefix}: unknown question type {qtype}")
            if qtype not in {"Q1a", "Q6"}:
                check(bool(question["text_repr"]), f"{prefix}: missing applicable text_repr")
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
            errors.append(f"{prefix}: semantic audit failed: {exc}")
    if q3_traps:
        check(abs(np.mean(q3_traps) - 0.5) <= 0.05, "Q3 trap ratio differs from 50% by >5 points")
    if q5_negative_types:
        positives = q5_negative_types["positive"]
        negatives = sum(q5_negative_types.values()) - positives
        check(abs(positives - negatives) <= (1 if is_mini else 0),
              "Q5 positive/negative ratio is not balanced")
        negative_counts = [
            q5_negative_types[kind] for kind in (
                "swap_elements", "delete_atom", "stretch_axis", "polymorph"
            )
        ]
        check(max(negative_counts) - min(negative_counts) <= 1, "Q5 negatives are not balanced")
    if questions:
        check(max(letters.values()) - min(letters.get(letter, 0) for letter in "ABCD")
              <= max(4, int(0.05 * len(questions))),
              "Correct answer letters are not approximately balanced")
    check(count_pairs["Q1a"] == count_pairs["Q1b"],
          "Q1a/Q1b do not use matching element/scene/image sets")
    if answer_texts.get("Q3"):
        semantic = answer_texts["Q3"]
        check(set(semantic) <= {"B is closer", "C is closer"},
              "Q3 has an invalid semantic answer")
        check(abs(semantic["B is closer"] - semantic["C is closer"]) <= 1,
              "Q3 B/C semantic answers are not balanced")
    if counts["Q4"] >= 80 or is_mini:
        systems = answer_texts["Q4"]
        check(len(systems) >= 6, "Q4 must cover at least six crystal systems")
        check(max(systems.values()) - min(systems.values()) <= 1,
              "Q4 crystal systems are not balanced")
    if is_mini:
        expected = {kind: 25 for kind in ("Q1a", "Q1b", "Q3", "Q4", "Q5", "Q6")}
        expected["Q2"] = 50
        check(dict(counts) == expected and len(questions) == 200,
              "Mini must contain 25 per task plus 25 each for Q2 easy/hard")
        check(Counter(q["difficulty"] for q in questions if q["qtype"] == "Q2")
              == {"easy": 25, "hard": 25}, "Mini Q2 difficulty quotas differ")
    elif not 600 <= len(questions) <= 900:
        warnings.append(f"Question count {len(questions)} is outside target 600-900")
    return {
        "passed": not errors, "checks": checks, "question_count": len(questions),
        "questions_file": questions_name, "questions_sha256": questions_sha256,
        "scene_count": len(scenes), "view_count": len(views),
        "qtype_counts": dict(sorted(counts.items())),
        "unique_base_structures_by_qtype": {
            kind: len({base for (qtype, base) in base_counts if qtype == kind})
            for kind in sorted(counts)
        },
        "answer_text_distributions": {
            kind: dict(sorted(values.items())) for kind, values in sorted(answer_texts.items())
        },
        "answer_letters": {letter: letters.get(letter, 0) for letter in "ABCD"},
        "answer_letters_by_type": {
            kind: {letter: counter.get(letter, 0) for letter in "ABCD"}
            for kind, counter in sorted(type_letters.items())
        },
        "q3_depth_trap_ratio": float(np.mean(q3_traps)) if q3_traps else None,
        "q5_sample_types": dict(sorted(q5_negative_types.items())),
        "q2_hard_visibility": {
            "question_count": len(hard_q2_visibility),
            "questions_with_neighbor_below_half_in_all_default_views": sum(
                bool(item["neighbors_below_half_in_all_default_views"])
                for item in hard_q2_visibility
            ),
            "neighbor_count": sum(item["neighbor_count"] for item in hard_q2_visibility),
            "neighbors_below_half_in_all_default_views": sum(
                len(item["neighbors_below_half_in_all_default_views"])
                for item in hard_q2_visibility
            ),
            "affected_questions": [
                item for item in hard_q2_visibility
                if item["neighbors_below_half_in_all_default_views"]
            ],
        },
        "errors": errors, "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--questions", "--questions-file", dest="questions", default="questions.jsonl",
                        help="Question JSONL filename relative to data-dir.")
    args = parser.parse_args()
    report = audit_questions(args.data_dir, args.questions)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
