"""Q3 geometry and quota checks without writing or rendering images."""

import json
from collections import Counter
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms

import question_distance as q3
from question_utils import geometry_key
from render import _prepare_views
from structures import get_structures


class _NoRenderStore:
    """Keep only the geometry of accepted candidates; never create artifacts."""

    def __init__(self, seed=20261021):
        random_directions = np.random.default_rng(seed).normal(size=(5, 3))
        random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
        self.directions = [[1, 0, 0], [1, 1, 0], [1, 1, 1], *random_directions.tolist()]
        self.tilts = [8, 8, 8, 0, 0, 0, 0, 0]
        self.accepted = {}

    def get(self, atoms, name, **parameters):
        sid = f"test-scene-{len(self.accepted)}"
        self.accepted[sid] = atoms, name, parameters
        return {
            "scene_id": sid, "text_repr": "",
            "views": [{"view_id": f"v{i:02d}", "image_path": f"{sid}/v{i:02d}.png"}
                      for i in range(8)],
        }


class DistanceQuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = get_structures()
        cls.store = _NoRenderStore()
        cls.drafts = q3.generate_q3(cls.structures, cls.store, target=40)

    def test_exact_quota_unique_triples_and_all_eight_view_visibility(self):
        self.assertEqual(len(self.drafts), 40)
        self.assertEqual(sum(q["meta"]["depth_trap"] for q in self.drafts), 20)
        self.assertEqual(Counter(q["answer_text"] for q in self.drafts),
                         {"B is closer": 20, "C is closer": 20})
        self.assertLessEqual(max(Counter(q["meta"]["base_structure"] for q in self.drafts).values()), 4)
        seen = set()
        for draft in self.drafts:
            with self.subTest(scene=draft["scene_id"]):
                atoms, name, parameters = self.store.accepted[draft["scene_id"]]
                highlight = parameters["highlight"]
                indices = {label: index for index, label in highlight.items()}
                a, b, c = (indices[label] for label in "ABC")
                signature = (name, geometry_key(atoms), parameters["display"],
                             a, *sorted((b, c)))
                self.assertNotIn(signature, seen)
                seen.add(signature)
                shown, _, _, views = _prepare_views(
                    atoms, self.store.directions, parameters["display"], (2, 2, 2),
                    None, None, self.store.tilts,
                )
                distances = np.linalg.norm(shown.positions[[b, c]] - shown.positions[a], axis=1)
                gap = abs(distances[0] - distances[1]) / max(distances)
                self.assertGreaterEqual(gap, 0.15)
                for label, index in indices.items():
                    saved = draft["meta"]["highlight_visibility_by_view"][label]
                    actual = [view.visible[index] for view in views]
                    np.testing.assert_allclose(saved, actual, atol=1e-12)
                    self.assertGreaterEqual(min(actual), 0.7)
                meta = draft["meta"]
                self.assertEqual(meta["display_indices"], indices)
                self.assertAlmostEqual(meta["distance_AB_angstrom"], distances[0])
                self.assertAlmostEqual(meta["distance_AC_angstrom"], distances[1])
                self.assertAlmostEqual(meta["relative_distance_gap"], gap)
                expected_answer = "B is closer" if distances[0] < distances[1] else "C is closer"
                self.assertEqual(draft["answer_text"], expected_answer)
                actual_traps = []
                for i, view in enumerate(views[:3]):
                    projection = np.linalg.norm(view.screen_xy[[b, c]] - view.screen_xy[a], axis=1)
                    reverse = bool((projection[0] - projection[1])
                                   * (distances[0] - distances[1]) < -1e-8)
                    item = meta["projected_distances"][i]
                    self.assertAlmostEqual(item["distance_AB_px"], projection[0])
                    self.assertAlmostEqual(item["distance_AC_px"], projection[1])
                    self.assertEqual(item["opposite_to_3d"], reverse)
                    if reverse:
                        actual_traps.append(f"v{i:02d}")
                self.assertEqual(meta["trap_view_ids"], actual_traps)
                self.assertIs(type(meta["depth_trap"]), bool)
                self.assertEqual(meta["depth_trap"], "v00" in actual_traps)
                json.dumps(meta, ensure_ascii=False, allow_nan=False)

    def test_last_view_visibility_cannot_be_silently_ignored(self):
        atoms = Atoms("CCC", positions=[[0, 0, 0], [1, 0, 5], [3, 0, 0]],
                      cell=[10, 10, 10], pbc=True)
        xy = np.array([[100, 100], [120, 100], [160, 100]])
        for last_visible, expected_present in ((0.7, True), (0.6999, False)):
            with self.subTest(last_visible=last_visible):
                views = [
                    SimpleNamespace(screen_xy=xy, visible=np.array([1., 1., 1.]))
                    for _ in range(8)
                ]
                views[-1].visible[1] = last_visible
                with patch.object(q3, "_prepare_views", return_value=(atoms, [], [], views)):
                    geometry = q3._candidate_geometry(
                        atoms, self.store, "cell", {}, 0, np.random.default_rng(42)
                    )
                self.assertEqual((0, 1, 2) in geometry.pools[True], expected_present)

    def test_distance_gap_boundary_and_subpixel_trap_rejection(self):
        for gap, projection_difference, expected in (
            (0.1499, 20, False), (0.1501, 20, True), (0.1501, 4.9, False),
            (0.1501, 5.0, True),
        ):
            with self.subTest(gap=gap, pixels=projection_difference):
                atoms = Atoms(
                    "CCC", positions=[[0, 0, 0], [10 * (1 - gap), 0, 0], [0, 10, 0]],
                    cell=[20, 20, 20], pbc=True,
                )
                # B is closer in 3D but farther in projection.
                xy = np.array([[100, 100], [160 + projection_difference, 100], [160, 100]])
                views = [
                    SimpleNamespace(screen_xy=xy, visible=np.ones(3)) for _ in range(8)
                ]
                with patch.object(q3, "_prepare_views", return_value=(atoms, [], [], views)):
                    geometry = q3._candidate_geometry(
                        atoms, self.store, "cell", {}, 0, np.random.default_rng(42)
                    )
                self.assertEqual((0, 1, 2) in geometry.pools[True], expected)

    def test_swapping_b_c_preserves_trap_and_swaps_distances(self):
        atoms = self.structures["NaCl"]
        geometry = q3._candidate_geometry(
            atoms, self.store, "cell", {}, 0, np.random.default_rng(42)
        )
        a, b, c = geometry.pools[True][0]
        original = q3._distance_meta(geometry, (a, b, c), {})
        swapped = q3._distance_meta(geometry, (a, c, b), {})
        self.assertTrue(original["depth_trap"])
        self.assertEqual(original["depth_trap"], swapped["depth_trap"])
        self.assertEqual(original["trap_view_ids"], swapped["trap_view_ids"])
        self.assertEqual(original["distance_AB_angstrom"], swapped["distance_AC_angstrom"])
        self.assertEqual(original["distance_AC_angstrom"], swapped["distance_AB_angstrom"])


if __name__ == "__main__":
    unittest.main()
