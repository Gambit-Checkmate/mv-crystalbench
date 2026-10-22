"""Q2 coordination questions using the renderer's exact first-shell rule.

Easy scenes contain only the center and its permitted periodic neighbors.
Hard scenes keep the complete cell/supercell and accept only centers whose
entire periodic shell is present. Both enforce >=70% center visibility in
every one of the eight stored views; no fallback silently weakens this rule.
"""

from __future__ import annotations

from collections import Counter
import random
from typing import Any

import numpy as np
from ase import Atoms

from bonding import (
    bond_cutoffs, bond_minima, coordination_neighbors, display_bonds,
)
from question_utils import geometry_key, make_draft, variant_atoms
from render import _prepare_views, build_display_atoms


MIN_CENTER_VISIBLE = 0.7
MIN_NEIGHBOR_VISIBLE = 0.5
COORDINATION_DISTRACTORS = (2, 3, 4, 6, 8, 12)


def _mapping_key(index, translation) -> tuple[int, tuple[int, int, int]]:
    return int(index), tuple(int(value) for value in translation)


def _choices(answer: int) -> list[str]:
    wrong = sorted(
        (number for number in COORDINATION_DISTRACTORS if number != answer),
        key=lambda number: (abs(number - answer), number),
    )[:3]
    return [str(answer), *(str(number) for number in wrong)]


def _center_metadata(
    atoms: Atoms, center: int, center_display: int, center_translation,
    neighbors: list[dict], neighbor_indices: list[int], views, augmentation,
    mode: str,
) -> dict[str, Any]:
    minima = bond_minima(atoms)
    cutoffs = bond_cutoffs(atoms)
    return {
        "coordination_number": len(neighbors),
        "center_original_index": int(center),
        "center_display_index": int(center_display),
        "center_translation": list(map(int, center_translation)),
        "center_element": atoms[center].symbol,
        "neighbor_display_indices": list(map(int, neighbor_indices)),
        "periodic_neighbors": neighbors,
        "bond_rule": "allowed element pair and distance <= 1.10 * global periodic pair d_min",
        "pair_cutoffs": [
            {
                "elements": list(pair), "d_min_angstrom": minima[pair],
                "cutoff_angstrom": cutoff,
            }
            for pair, cutoff in sorted(cutoffs.items())
        ],
        "complete_shell": True,
        "displayed_center_degree": len(neighbors),
        "display": mode,
        "center_visibility_all_views": [
            float(view.visible[center_display]) for view in views
        ],
        "min_center_visible_required": MIN_CENTER_VISIBLE,
        "neighbor_best_visibility_default_views": [
            float(max(view.visible[index] for view in views[:3]))
            for index in neighbor_indices
        ],
        "augmentation": augmentation,
        "text_applicable": True,
    }


def _easy_candidate(atoms: Atoms, store, rng, augmentation):
    original_indices = list(range(len(atoms)))
    rng.shuffle(original_indices)
    # Inequivalent environments are tried first. Equivalent sites may produce
    # identical projected neighborhoods, so do not prefer many duplicate sites.
    ordered = sorted(original_indices, key=lambda index: atoms[index].symbol)
    for center in ordered:
        neighbors = coordination_neighbors(atoms, center)
        if not neighbors:
            continue
        cutoff = max(float(item["cutoff"]) for item in neighbors) + 1e-7
        _, full_mapping = build_display_atoms(
            atoms, display="environment", center=center, cutoff=cutoff,
        )
        wanted = {
            _mapping_key(item["original_index"], item["translation"])
            for item in neighbors
        }
        wanted.add(_mapping_key(center, (0, 0, 0)))
        selected = [
            index for index, (original, translation) in enumerate(full_mapping)
            if _mapping_key(original, translation) in wanted
        ]
        if len(selected) != len(neighbors) + 1:
            raise RuntimeError("The environment expansion omitted a periodic neighbor")
        shown, mapping, _, views = _prepare_views(
            atoms, store.directions, "environment", (2, 2, 2), center, cutoff,
            store.tilts, display_indices=selected,
        )
        mapping_lookup = {
            _mapping_key(original, translation): index
            for index, (original, translation) in enumerate(mapping)
        }
        center_display = mapping_lookup[_mapping_key(center, (0, 0, 0))]
        neighbor_indices = [
            mapping_lookup[_mapping_key(item["original_index"], item["translation"])]
            for item in neighbors
        ]
        if any(view.visible[center_display] < MIN_CENTER_VISIBLE for view in views):
            continue
        if any(
            max(view.visible[index] for view in views[:3]) < MIN_NEIGHBOR_VISIBLE
            for index in neighbor_indices
        ):
            continue
        center_degree = sum(
            center_display in (i, j) for i, j, _ in display_bonds(atoms, shown)
        )
        if center_degree != len(neighbors):
            raise RuntimeError("Displayed and periodic coordination disagree")
        meta = _center_metadata(
            atoms, center, center_display, (0, 0, 0), neighbors,
            neighbor_indices, views, augmentation, "environment",
        )
        meta.update({
            "min_neighbor_visible_in_any_default_view": MIN_NEIGHBOR_VISIBLE,
            "environment_only_center_and_shell": True,
            "environment_cutoff_angstrom": cutoff,
        })
        parameters = {
            "display": "environment", "center": center, "cutoff": cutoff,
            "display_indices": selected, "highlight": {center_display: "A"},
        }
        return parameters, meta
    return None


def _hard_candidate(atoms: Atoms, store, rng, augmentation):
    neighbors_by_center = {
        center: coordination_neighbors(atoms, center) for center in range(len(atoms))
    }
    for mode in ("cell", "supercell"):
        shown, mapping, _, views = _prepare_views(
            atoms, store.directions, mode, (2, 2, 2), None, None, store.tilts,
        )
        lookup = {
            _mapping_key(original, translation): index
            for index, (original, translation) in enumerate(mapping)
        }
        candidates = list(range(len(shown)))
        rng.shuffle(candidates)
        for center_display in candidates:
            center, center_translation = mapping[center_display]
            neighbors = neighbors_by_center[center]
            if not neighbors:
                continue
            if any(
                view.visible[center_display] < MIN_CENTER_VISIBLE for view in views
            ):
                continue
            translated_neighbor_keys = [
                _mapping_key(
                    item["original_index"],
                    np.asarray(item["translation"]) + center_translation,
                )
                for item in neighbors
            ]
            if not all(key in lookup for key in translated_neighbor_keys):
                continue
            neighbor_indices = [lookup[key] for key in translated_neighbor_keys]
            center_degree = sum(
                center_display in (i, j) for i, j, _ in display_bonds(atoms, shown)
            )
            if center_degree != len(neighbors):
                raise RuntimeError("A complete displayed shell has inconsistent degree")
            meta = _center_metadata(
                atoms, center, center_display, center_translation, neighbors,
                neighbor_indices, views, augmentation, mode,
            )
            meta["repeat"] = [2, 2, 2] if mode == "supercell" else [1, 1, 1]
            parameters = {
                "display": mode, "repeat": (2, 2, 2),
                "highlight": {center_display: "A"},
            }
            return parameters, meta
    return None


def generate_q2(
    structures: dict[str, Atoms], store, n_easy: int = 60, n_hard: int = 60,
    seed: int = 20261021,
) -> list[dict[str, Any]]:
    """Render qualified Q2 scenes and return unlettered question drafts.

    Structures are visited round-robin, with seeded axis strains of at most
    +/-2% plus documented origin shifts. Ineligible structures are skipped;
    after a bounded search a shortfall raises rather than fabricating data.
    """
    if not structures:
        raise ValueError("At least one source structure is required")
    if any(
        not isinstance(number, int) or isinstance(number, bool) or number < 0
        for number in (n_easy, n_hard)
    ):
        raise ValueError("Question targets must be nonnegative integers")
    names = [name for name, atoms in structures.items() if atoms.info.get("q2_eligible", True)]
    if not names:
        raise ValueError("No structures are eligible for reliable coordination questions.")
    if max(n_easy, n_hard) > 2 * len(names):
        raise ValueError("Q2 target exceeds the two-per-base, per-difficulty capacity.")
    rng = random.Random(int(seed))
    rng.shuffle(names)
    drafts = []
    for difficulty, target, picker in (
        ("easy", n_easy, _easy_candidate), ("hard", n_hard, _hard_candidate),
    ):
        accepted = 0
        attempted = 0
        seen = set()
        coverage: Counter[str] = Counter()
        max_attempts = max(100, target * 30)
        while accepted < target and attempted < max_attempts:
            name = names[attempted % len(names)]
            variant_index = 1 + attempted // len(names)
            atoms, augmentation = variant_atoms(
                structures[name], variant_index, int(seed), strain=True,
            )
            attempted += 1
            if coverage[name] >= 2:
                continue
            geometry = geometry_key(atoms)
            if geometry in seen:
                continue
            candidate = picker(atoms, store, rng, augmentation)
            if candidate is None:
                continue
            parameters, meta = candidate
            scene = store.get(
                atoms, name, **parameters,
                variant_key=f"Q2-{difficulty}-{variant_index}",
            )
            # Verify persisted image metadata as well as the preflight geometry.
            center_display = meta["center_display_index"]
            if any(
                view["metadata"]["atoms"][center_display]["visible_fraction"]
                < MIN_CENTER_VISIBLE
                for view in scene["views"]
            ):
                raise RuntimeError("Saved center visibility differs from the preflight")
            answer = meta["coordination_number"]
            question = (
                "Using the gray bonds, what is the first-neighbor coordination number "
                "of the atom labeled A? Count only atoms directly bonded to A, and "
                "count the same atom only once across the views. "
            )
            if difficulty == "easy":
                question += "Only A and its complete first coordination shell are displayed."
            else:
                question += "A's complete first coordination shell is contained in the displayed region."
            drafts.append(make_draft(
                "Q2", difficulty, name, scene, question, _choices(answer),
                str(answer), meta,
            ))
            seen.add(geometry)
            coverage[name] += 1
            accepted += 1
            if accepted % 20 == 0:
                print(f"Q2 {difficulty}: {accepted}/{target} qualified scenes", flush=True)
        if accepted != target:
            raise RuntimeError(
                f"Q2 {difficulty}: only {accepted}/{target} scenes passed complete-shell "
                f"and all-eight-view visibility checks in {attempted} attempts; "
                "the visibility threshold has not been relaxed."
            )
        print(
            f"Q2 {difficulty}: {accepted} questions, {len(coverage)} structures, "
            f"{attempted} attempts; all-view center visibility >= {MIN_CENTER_VISIBLE}",
            flush=True,
        )
    return drafts
