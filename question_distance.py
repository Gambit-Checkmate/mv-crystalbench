"""Distance questions with visible labels and balanced projection reversals."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral
from typing import Any

import numpy as np
from ase import Atoms

from render import _prepare_views


@dataclass
class _Geometry:
    atoms: Atoms
    display: str
    augmentation: dict[str, Any]
    variant_index: int
    identity: str
    shown: Atoms
    views: list
    eligible: np.ndarray
    pools: dict[bool, list[tuple[int, int, int]]]


def _candidate_geometry(atoms, store, display, augmentation, variant_index, rng):
    from question_utils import geometry_key

    shown, _, _, views = _prepare_views(
        atoms, store.directions, display, (2, 2, 2), None, None, store.tilts
    )
    worst = np.min([view.visible for view in views], axis=0)
    eligible = np.flatnonzero(worst >= 0.7)
    pools = {False: [], True: []}
    if len(eligible) >= 3:
        xyz = shown.positions
        for a in eligible:
            others = eligible[eligible != a]
            for b, c in combinations(others, 2):
                ab, ac = np.linalg.norm(xyz[[b, c]] - xyz[a], axis=1)
                if min(ab, ac) <= 1e-9 or abs(ab - ac) / max(ab, ac) < 0.15:
                    continue
                projected = np.array([
                    np.linalg.norm(view.screen_xy[[b, c]] - view.screen_xy[a], axis=1)
                    for view in views[:3]
                ])
                products = (projected[:, 0] - projected[:, 1]) * (ab - ac)
                trap = products < -1e-8
                # Keep perceptually substantive reversals. A tiny subpixel
                # ordering flip should not define a depth trap.
                if trap[0] and abs(projected[0, 0] - projected[0, 1]) < 5:
                    continue
                pools[bool(trap[0])].append((int(a), int(b), int(c)))
        for trap, candidates in pools.items():
            rng.shuffle(candidates)
            if not trap:
                # Prefer nontraps in v00 that still have a reversal in another
                # default view, without changing the reproducible v00 label.
                def has_other_trap(triple):
                    a, b, c = triple
                    ab, ac = np.linalg.norm(xyz[[b, c]] - xyz[a], axis=1)
                    return any(
                        (np.linalg.norm(view.screen_xy[b] - view.screen_xy[a])
                         - np.linalg.norm(view.screen_xy[c] - view.screen_xy[a]))
                        * (ab - ac) < -1e-8 for view in views[1:3]
                    )
                candidates.sort(key=has_other_trap)
                # Lists are popped from the end: preferred candidates last.
    return _Geometry(
        atoms, display, augmentation, variant_index, geometry_key(atoms),
        shown, views, eligible, pools
    )


def _distance_meta(geometry, triple, augmentation):
    a, b, c = triple
    ab, ac = np.linalg.norm(geometry.shown.positions[[b, c]]
                           - geometry.shown.positions[a], axis=1)
    projections = []
    trap_view_ids = []
    for index, view in enumerate(geometry.views[:3]):
        pab, pac = np.linalg.norm(view.screen_xy[[b, c]] - view.screen_xy[a], axis=1)
        opposite = bool((pab - pac) * (ab - ac) < -1e-8)
        view_id = f"v{index:02d}"
        if opposite:
            trap_view_ids.append(view_id)
        projections.append({
            "view_id": view_id,
            "distance_AB_px": float(pab),
            "distance_AC_px": float(pac),
            "opposite_to_3d": opposite,
        })
    visibility = {
        label: [
            float(view.visible[index]) for view in geometry.views
        ] for label, index in zip(("A", "B", "C"), triple)
    }
    return {
        "depth_trap": "v00" in trap_view_ids,
        "depth_trap_definition": (
            "The ordering of A-B and A-C screen distances in the first default "
            "view (v00) is opposite to their displayed Cartesian 3D distances."
        ),
        "depth_trap_reference_view": "v00",
        "trap_view_ids": trap_view_ids,
        "projected_distances": projections,
        "display_indices": {"A": int(a), "B": int(b), "C": int(c)},
        "distance_AB_angstrom": float(ab),
        "distance_AC_angstrom": float(ac),
        "relative_distance_gap": float(abs(ab - ac) / max(ab, ac)),
        "distance_convention": "Cartesian Euclidean distance between displayed atoms; no MIC",
        "minimum_required_visible_fraction": 0.7,
        "highlight_visibility_by_view": visibility,
        "visibility_view_ids": [f"v{i:02d}" for i in range(8)],
        "augmentation": augmentation,
        "text_applicable": True,
    }


def generate_q3(structures, store, target=100, seed=20261021) -> list[dict[str, Any]]:
    """Render Q3 drafts with >=15% 3D gaps and exactly balanced v00 traps.

    Every highlighted atom has at least 70% raster visibility in every one of
    the scene's eight views. Triples are distinct beyond B/C label swapping.
    Selection rotates through structures; origin shifts and, if needed, a
    supercell provide alternatives when a prototype lacks visible candidates.
    """
    from question_utils import make_draft, variant_atoms

    if isinstance(target, bool) or not isinstance(target, Integral) or target < 0:
        raise ValueError("target must be a nonnegative integer.")
    if target == 0:
        return []
    if not structures:
        raise ValueError("At least one structure is required.")
    rng = np.random.default_rng(int(seed))
    names = list(structures)
    geometries: dict[str, list[_Geometry]] = defaultdict(list)
    next_variant = defaultdict(int)
    used = set()
    unavailable = []
    output = []
    coverage = Counter()
    if target > 4 * len(names):
        raise ValueError("Q3 target exceeds the four-per-base capacity.")

    def find_candidate(name, trap):
        # First use computed geometry. New augmentation is only generated when
        # its existing candidates for the required trap class are exhausted.
        for geometry in geometries[name]:
            while geometry.pools[trap]:
                candidate = geometry.pools[trap].pop()
                signature = (name, geometry.identity, geometry.display,
                             candidate[0], *sorted(candidate[1:]))
                if signature not in used:
                    used.add(signature)
                    return geometry, candidate
        # Up to sixteen translated origins per structure; start with cell
        # views, then explore a 2x2x2 display when smaller displays are sparse.
        while next_variant[name] < 20:
            variant = next_variant[name]
            next_variant[name] += 1
            atoms, augmentation = variant_atoms(
                structures[name], variant, seed=int(seed), strain=False
            )
            display = "cell" if variant < 16 else "supercell"
            geometry = _candidate_geometry(
                atoms, store, display, augmentation, variant, rng
            )
            geometries[name].append(geometry)
            while geometry.pools[trap]:
                candidate = geometry.pools[trap].pop()
                signature = (name, geometry.identity, display, candidate[0], *sorted(candidate[1:]))
                if signature not in used:
                    used.add(signature)
                    return geometry, candidate
        return None

    for number in range(int(target)):
        need_trap = number % 2 == 0
        # Rotate starting prototype each cycle, so even-sized libraries do not
        # confound prototype identity with the alternating trap class.
        start = (number + number // len(names)) % len(names)
        found = False
        for offset in range(len(names)):
            name = names[(start + offset) % len(names)]
            if coverage[name] >= 4:
                continue
            # A label placement can fail in dense geometry. Reject that triple
            # and select another rather than relaxing visibility constraints.
            for _ in range(20):
                result = find_candidate(name, need_trap)
                if result is None:
                    if name not in unavailable:
                        unavailable.append(name)
                    break
                geometry, triple = result
                a, b, c = triple
                # Counterbalance the semantic answer independently of trap class.
                want_b_closer = (number // 2) % 2 == 0
                current_b_closer = (
                    np.linalg.norm(geometry.shown.positions[b] - geometry.shown.positions[a])
                    < np.linalg.norm(geometry.shown.positions[c] - geometry.shown.positions[a])
                )
                if current_b_closer != want_b_closer:
                    b, c = c, b
                triple = (a, b, c)
                meta = _distance_meta(geometry, triple, geometry.augmentation)
                try:
                    scene = store.get(
                        geometry.atoms, name, display=geometry.display,
                        highlight={a: "A", b: "B", c: "C"}, repeat=(2, 2, 2),
                        variant_key=f"q3-origin-{geometry.variant_index}",
                    )
                except ValueError as error:
                    if "No unobstructed location for label" not in str(error):
                        raise
                    continue
                answer = ("B is closer" if meta["distance_AB_angstrom"] < meta["distance_AC_angstrom"]
                          else "C is closer")
                output.append(make_draft(
                    "Q3", "hard" if meta["depth_trap"] else "medium",
                    name, scene,
                    "Consider the labeled atoms A, B, and C. Which of B and C is closer "
                    "to A in three-dimensional Euclidean distance between the displayed "
                    "atoms? Use all views.",
                    ["B is closer", "C is closer", "Equally close", "Cannot be determined"],
                    answer, meta,
                ))
                coverage[name] += 1
                found = True
                break
            if found:
                break
        if not found:
            visibility_counts = {
                name: max((len(g.eligible) for g in values), default=0)
                for name, values in geometries.items()
            }
            raise RuntimeError(
                f"Only {len(output)}/{target} valid Q3 questions; cannot supply "
                f"depth_trap={need_trap} while retaining 0.7 visibility in all "
                f"8 views. Best eligible-atom counts: {visibility_counts}."
            )
    if unavailable:
        print("Q3 coverage note: some requested trap classes exhausted for "
              + ", ".join(unavailable) + "; other structures filled those slots.")
    return output
