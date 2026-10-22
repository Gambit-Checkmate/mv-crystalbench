"""Scientific geometry and rendering-contract checks; no external API calls."""

import copy
from itertools import product
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from ase import Atoms
from ase.data.colors import jmol_colors
from matplotlib import rc_context
from matplotlib.image import imread

import render
from structures import get_structures


class TaskOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = get_structures()

    def assert_mapping_positions(self, original, displayed, mapping):
        self.assertEqual(len(displayed), len(mapping))
        self.assertEqual(len(mapping), len(set(
            (int(index), tuple(shift)) for index, shift in mapping
        )))
        for position, number, (original_index, shift) in zip(
            displayed.positions, displayed.numbers, mapping
        ):
            shift = np.asarray(shift)
            np.testing.assert_array_equal(shift, shift.astype(int))
            self.assertEqual(number, original.numbers[original_index])
            np.testing.assert_allclose(
                position,
                original.positions[original_index] + shift @ original.cell
                - original.get_celldisp().reshape(3),
                atol=1e-12,
            )

    def test_rutile_direction_uses_lattice_lengths(self):
        atoms = self.structures["TiO2_rutile"]
        rotation = render.direction_to_rotation(atoms, (1, 1, 1), tilt=0)
        expected = np.array([4.594, 4.594, 2.959])
        np.testing.assert_allclose(
            rotation[:, 2], expected / np.linalg.norm(expected), atol=1e-12
        )
        np.testing.assert_allclose(
            expected @ rotation, [0, 0, np.linalg.norm(expected)], atol=1e-12
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_hexagonal_direction_and_parallel_up_fallback(self):
        atoms = self.structures["Mg_hcp"]
        rotation = render.direction_to_rotation(atoms, (1, 1, 0), tilt=0)
        np.testing.assert_allclose(
            rotation[:, 2], [0.5, np.sqrt(3) / 2, 0], atol=1e-12
        )
        for direction in ((0, 0, 1), (0, 0, -1), (-1, 2, 3)):
            for tilt in (0, 8):
                rotation = render.direction_to_rotation(atoms, direction, tilt=tilt)
                np.testing.assert_allclose(
                    rotation.T @ rotation, np.eye(3), atol=1e-12
                )
                self.assertAlmostEqual(np.linalg.det(rotation), 1.0)

    def test_cell_adds_all_boundary_images_and_preserves_mapping(self):
        original = self.structures["NaCl"]
        displayed, mapping = render.build_display_atoms(original)
        self.assertEqual(len(displayed), 27)
        self.assert_mapping_positions(original, displayed, mapping)
        scaled = displayed.positions @ np.linalg.inv(original.cell)
        expected = np.asarray(list(product((0, 0.5, 1), repeat=3)))
        self.assertEqual(
            {tuple(np.round(row, 10)) for row in scaled},
            {tuple(row) for row in expected},
        )

    def test_boundary_tolerance_and_no_duplicate_periodic_sites(self):
        original = Atoms(
            "C", scaled_positions=[[-0.0005, 0.5, 0.5]],
            cell=[3, 4, 5], pbc=True,
        )
        displayed, mapping = render.build_display_atoms(original)
        self.assertEqual(len(displayed), 2)
        self.assertEqual({tuple(shift) for _, shift in mapping}, {(0, 0, 0), (1, 0, 0)})
        self.assert_mapping_positions(original, displayed, mapping)

    def test_supercell_preserves_original_indices_and_extended_cell(self):
        original = self.structures["NaCl"]
        displayed, mapping = render.build_display_atoms(
            original, display="supercell", repeat=(2, 2, 2)
        )
        self.assertEqual(len(displayed), 125)
        self.assert_mapping_positions(original, displayed, mapping)
        np.testing.assert_allclose(displayed.cell, original.cell.array * 2)
        scaled = displayed.positions @ np.linalg.inv(original.cell)
        self.assertGreaterEqual(scaled.min(), -1e-3)
        self.assertLessEqual(scaled.max(), 2 + 1e-3)

    def test_environment_contains_complete_nacl_and_rutile_first_shells(self):
        for name, neighbor, cutoff in (
            ("NaCl", "Cl", 3.11),
            ("TiO2_rutile", "O", 2.2),
        ):
            with self.subTest(name=name):
                original = self.structures[name]
                displayed, mapping = render.build_display_atoms(
                    original, display="environment", center=0, cutoff=cutoff
                )
                self.assertEqual(len(displayed), 7)
                self.assertEqual(displayed.get_chemical_symbols().count(neighbor), 6)
                self.assertIn((0, (0, 0, 0)), [
                    (int(index), tuple(shift)) for index, shift in mapping
                ])
                self.assert_mapping_positions(original, displayed, mapping)
                distances = np.linalg.norm(
                    displayed.positions - original.positions[0], axis=1
                )
                self.assertEqual(np.count_nonzero(distances < 1e-10), 1)
                self.assertTrue(np.all(distances <= cutoff + 1e-10))

    def test_environment_search_handles_skew_cell_images_beyond_adjacent_cells(self):
        original = Atoms(
            "C", scaled_positions=[[0.5, 0.5, 0.5]],
            cell=[[3, 0, 0], [2.8, 0.4, 0], [0, 0, 3]], pbc=True,
        )
        cutoff = 0.9
        displayed, mapping = render.build_display_atoms(
            original, display="environment", center=0, cutoff=cutoff
        )
        expected = {
            shift for shift in product(range(-5, 6), repeat=3)
            if np.linalg.norm(np.asarray(shift) @ original.cell) <= cutoff + 1e-10
        }
        actual = {tuple(shift) for _, shift in mapping}
        self.assertEqual(actual, expected)
        self.assertTrue(any(max(map(abs, shift)) > 1 for shift in actual))
        self.assert_mapping_positions(original, displayed, mapping)

    def test_environment_center_is_original_index_and_positions_remain_physical(self):
        original = self.structures["NaCl"].copy()
        displacement = np.array([1.1, -0.4, 0.8])
        original.positions += displacement
        original.set_celldisp(displacement)
        displayed, mapping = render.build_display_atoms(
            original, display="environment", center=1, cutoff=3.11
        )
        self.assertEqual(len(displayed), 7)
        self.assertEqual(displayed.get_chemical_symbols().count("Na"), 6)
        self.assert_mapping_positions(original, displayed, mapping)
        center_index = next(
            i for i, (index, shift) in enumerate(mapping)
            if index == 1 and tuple(shift) == (0, 0, 0)
        )
        np.testing.assert_allclose(
            displayed.positions[center_index],
            original.positions[1] - displacement,
            atol=1e-12,
        )

    def test_png_size_white_background_and_input_not_modified(self):
        atoms = self.structures["TiO2_rutile"]
        positions = atoms.positions.copy()
        cell = atoms.cell.array.copy()
        info = copy.deepcopy(atoms.info)
        celldisp = atoms.get_celldisp().copy()
        with tempfile.TemporaryDirectory() as directory:
            # Ambient plotting preferences must not change output dimensions
            # or add an external LaTeX dependency to a headless export.
            with rc_context({"savefig.bbox": "tight", "text.usetex": True,
                             "figure.autolayout": True}):
                paths = render.render_views(
                    atoms, render.DEMO_DIRECTIONS, directory, highlight={0: "A"}
                )
            self.assertEqual(len(paths), 3)
            self.assertEqual(len(set(paths)), 3)
            for path in paths:
                self.assertIsInstance(path, Path)
                pixels = imread(path)
                self.assertEqual(pixels.shape[:2], (512, 512))
                rgb = pixels[:, :, :3]
                np.testing.assert_allclose(rgb[0, 0], [1, 1, 1])
                self.assertGreater(np.sum(np.any(rgb < 0.95, axis=2)), 500)
                self.assertTrue(path.with_suffix(".json").is_file())
        np.testing.assert_array_equal(atoms.positions, positions)
        np.testing.assert_array_equal(atoms.cell.array, cell)
        np.testing.assert_array_equal(atoms.get_celldisp(), celldisp)
        self.assertEqual(atoms.info, info)

    def test_translated_display_cell_has_identical_rendering(self):
        atoms = self.structures["NaCl"]
        shifted = atoms.copy()
        shift = np.array([2.5, -1.25, 0.75])
        shifted.positions += shift
        shifted.set_celldisp(shift)
        with tempfile.TemporaryDirectory() as directory:
            original = render.render_views(
                atoms, [(1, 1, 1)], Path(directory) / "original", highlight=0
            )[0]
            translated = render.render_views(
                shifted, [(1, 1, 1)], Path(directory) / "translated", highlight=0
            )[0]
            np.testing.assert_array_equal(imread(original), imread(translated))
        np.testing.assert_array_equal(shifted.get_celldisp(), shift)

    def test_invalid_inputs_create_no_images(self):
        atoms = self.structures["NaCl"]
        displayed, _ = render.build_display_atoms(atoms)
        with tempfile.TemporaryDirectory() as directory:
            for directions, highlight, error in (
                ([(1, 0, 0), (0, 0, 0)], None, ValueError),
                ([(np.nan, 0, 1)], None, ValueError),
                ([], None, ValueError),
                ([(1, 0, 0)], len(displayed), IndexError),
                ([(1, 0, 0)], {0: "A", 1: "A"}, ValueError),
            ):
                with self.subTest(directions=directions, highlight=highlight):
                    with self.assertRaises(error):
                        render.render_views(atoms, directions, directory, highlight)
            self.assertFalse(list(Path(directory).iterdir()))

    def test_environment_requires_explicit_finite_positive_cutoff(self):
        atoms = self.structures["NaCl"]
        for cutoff in (None, 0, -1, np.nan, np.inf):
            with self.subTest(cutoff=cutoff):
                with self.assertRaises((TypeError, ValueError)):
                    render.build_display_atoms(
                        atoms, display="environment", center=0, cutoff=cutoff
                    )

    def test_raster_visibility_exact_overlap_and_front_order(self):
        coordinates = np.array([[256, 256], [256, 256]], dtype=float)
        radii = np.array([24, 24], dtype=float)
        for depths, expected in (
            ([1, 2], [0, 1]),
            ([2, 1], [1, 0]),
            ([1, 1], [0, 1]),
        ):
            with self.subTest(depths=depths):
                actual = render._visible_fractions(
                    coordinates, radii, np.asarray(depths, dtype=float)
                )
                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_raster_partial_occlusion_matches_circle_intersection_area(self):
        radius, distance = 25.0, 20.0
        coordinates = np.array([[240, 256], [240 + distance, 256]])
        actual = render._visible_fractions(
            coordinates, np.array([radius, radius]), np.array([1, 2])
        )
        overlap = (
            2 * radius**2 * np.arccos(distance / (2 * radius))
            - distance / 2 * np.sqrt(4 * radius**2 - distance**2)
        )
        expected_rear = 1 - overlap / (np.pi * radius**2)
        self.assertAlmostEqual(actual[0], expected_rear, delta=0.01)
        self.assertAlmostEqual(actual[1], 1, delta=0.001)

    def test_raster_visibility_uses_union_of_all_foreground_disks(self):
        coordinates = np.array([[250, 250], [238, 250], [262, 250]], dtype=float)
        radii = np.array([20, 20, 20], dtype=float)
        all_three = render._visible_fractions(
            coordinates, radii, np.array([0, 1, 2], dtype=float)
        )
        each_left = render._visible_fractions(
            coordinates[:2], radii[:2], np.array([0, 1], dtype=float)
        )
        each_right = render._visible_fractions(
            coordinates[[0, 2]], radii[[0, 2]], np.array([0, 1], dtype=float)
        )
        self.assertLess(all_three[0], each_left[0] - 0.1)
        self.assertLess(all_three[0], each_right[0] - 0.1)
        self.assertGreaterEqual(all_three.min(), 0)
        self.assertLessEqual(all_three.max(), 1)

    def test_json_tracks_every_display_image_atom_and_radius_limit(self):
        original = self.structures["NaCl"]
        displayed, mapping = render.build_display_atoms(original)
        with tempfile.TemporaryDirectory() as directory:
            paths = render.render_views(original, render.DEMO_DIRECTIONS, directory)
            for path in paths:
                metadata = json.loads(path.with_suffix(".json").read_text("utf-8"))
                self.assertEqual(metadata["display_mode"], "cell")
                self.assertEqual(metadata["image_size_px"], [512, 512])
                self.assertEqual(metadata["camera"]["tilt_degrees"], 8)
                self.assertEqual(len(metadata["atoms"]), len(displayed))
                for index, item in enumerate(metadata["atoms"]):
                    self.assertEqual(item["display_index"], index)
                    self.assertEqual(item["original_index"], mapping[index][0])
                    self.assertEqual(tuple(item["translation"]), tuple(mapping[index][1]))
                    self.assertEqual(item["element"], displayed[index].symbol)
                    self.assertEqual(len(item["screen_xy_px"]), 2)
                    self.assertTrue(np.isfinite(item["screen_xy_px"]).all())
                    self.assertTrue(np.isfinite(item["depth"]))
                    self.assertGreater(item["radius_px"], 0)
                    self.assertLessEqual(2 * item["radius_px"], 512 / 10 + 1e-8)
                    self.assertGreaterEqual(item["visible_fraction"], 0)
                    self.assertLessEqual(item["visible_fraction"], 1)
                self.assertFalse(metadata["highlight_labels"])

    def test_tilt_filenames_are_distinct_and_strict_axis_reverses_front_atom(self):
        atoms = Atoms(
            "OO", positions=[[1, 1, 1], [1, 1, 2]],
            cell=[3, 3, 3], pbc=True,
        )
        directions = [(0, 0, 1), (0, 0, -1)]
        with tempfile.TemporaryDirectory() as directory:
            strict = render.render_views(atoms, directions, directory, tilt=0)
            tilted = render.render_views(atoms, directions, directory, tilt=8)
            self.assertEqual(len(set(strict + tilted)), 4)
            for path, expected in zip(strict, ([0, 1], [1, 0])):
                metadata = json.loads(path.with_suffix(".json").read_text("utf-8"))
                actual = [a["visible_fraction"] for a in metadata["atoms"]]
                np.testing.assert_allclose(actual, expected, atol=1e-12)
            self.assertFalse(np.array_equal(imread(strict[0]), imread(tilted[0])))

    def test_pick_visible_atoms_agrees_with_saved_metadata_and_sort_order(self):
        atoms = self.structures["NaCl"]
        directions = render.DEMO_DIRECTIONS
        displayed, _ = render.build_display_atoms(atoms)
        with tempfile.TemporaryDirectory() as directory:
            paths = render.render_views(atoms, directions, directory)
            fractions = np.asarray([
                [item["visible_fraction"] for item in json.loads(
                    path.with_suffix(".json").read_text("utf-8")
                )["atoms"]]
                for path in paths
            ])
        worst = fractions.min(axis=0)
        eligible = {i for i, value in enumerate(worst) if value >= 0.7}
        picked = render.pick_visible_atoms(atoms, directions, len(displayed))
        self.assertEqual(set(picked), eligible)
        self.assertEqual(len(picked), len(set(picked)))
        self.assertTrue(np.all(np.diff(worst[picked]) <= 1e-12))
        self.assertEqual(render.pick_visible_atoms(atoms, directions, 2), picked[:2])
        self.assertEqual(render.pick_visible_atoms(atoms, directions, 0), [])
        # Each oxygen is hidden in one of these exact opposite views.
        aligned = Atoms(
            "OO", positions=[[1, 1, 1], [1, 1, 2]],
            cell=[3, 3, 3], pbc=True,
        )
        self.assertEqual(render.pick_visible_atoms(
            aligned, [(0, 0, 1), (0, 0, -1)], 2, tilt=0
        ), [])

    def test_environment_highlight_is_display_index_and_label_avoids_atoms(self):
        original = self.structures["NaCl"]
        displayed, mapping = render.build_display_atoms(
            original, display="environment", center=0, cutoff=3.11
        )
        central = next(
            i for i, (index, shift) in enumerate(mapping)
            if index == 0 and tuple(shift) == (0, 0, 0)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = render.render_views(
                original, [(1, 1, 1)], directory, highlight={central: "A"},
                display="environment", center=0, cutoff=3.11,
            )[0]
            metadata = json.loads(path.with_suffix(".json").read_text("utf-8"))
        labels = metadata["highlight_labels"]
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels[0]["display_index"], central)
        self.assertEqual(labels[0]["label"], "A")
        self.assertEqual(metadata["atoms"][central]["original_index"], 0)
        x0, y0, x1, y1 = labels[0]["bbox_px"]
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)
        self.assertTrue(0 <= x0 <= x1 <= 512)
        self.assertTrue(0 <= y0 <= y1 <= 512)
        for atom in metadata["atoms"]:
            x, y = atom["screen_xy_px"]
            dx = max(x0 - x, 0, x - x1)
            dy = max(y0 - y, 0, y - y1)
            self.assertGreaterEqual(
                np.hypot(dx, dy), atom["radius_px"] - 1e-6
            )

    def test_environment_bonds_match_six_nearest_neighbors_and_can_be_disabled(self):
        for name, cutoff in (("NaCl", 3.11), ("TiO2_rutile", 2.2)):
            original = self.structures[name]
            _, mapping = render.build_display_atoms(
                original, display="environment", center=0, cutoff=cutoff
            )
            central = next(
                i for i, (index, shift) in enumerate(mapping)
                if index == 0 and tuple(shift) == (0, 0, 0)
            )
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                for enabled in (True, False):
                    path = render.render_views(
                        original, [(1, 1, 1)], Path(directory) / str(enabled),
                        display="environment", center=0, cutoff=cutoff, bonds=enabled,
                    )[0]
                    metadata = json.loads(path.with_suffix(".json").read_text("utf-8"))
                    bonds = metadata["bonds"]
                    self.assertEqual(len(bonds), 6 if enabled else 0)
                    if enabled:
                        neighbors = []
                        for bond in bonds:
                            self.assertIn(central, bond["display_indices"])
                            self.assertLess(bond["distance_angstrom"], cutoff)
                            neighbors.append(next(
                                i for i in bond["display_indices"] if i != central
                            ))
                        self.assertEqual(len(set(neighbors)), 6)

    def test_fog_and_foreground_atom_occluding_a_background_bond(self):
        # One background O--O bond passes behind a disconnected foreground O.
        # This checks rendered pixels, not just the saved bond declarations.
        atoms = Atoms(
            "OOO", positions=[[1, 1, 1], [2.4, 1, 1], [1.7, 1, 3]],
            cell=[4, 4, 4], pbc=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            bonded = render.render_views(
                atoms, [(0, 0, 1)], Path(directory) / "bonds", tilt=0, bonds=True
            )[0]
            plain = render.render_views(
                atoms, [(0, 0, 1)], Path(directory) / "plain", tilt=0, bonds=False
            )[0]
            metadata = json.loads(bonded.with_suffix(".json").read_text("utf-8"))
            image = imread(bonded)[:, :, :3]
            no_bonds = imread(plain)[:, :, :3]
        self.assertEqual(len(metadata["bonds"]), 1)
        base = jmol_colors[8]
        for index, expected in ((0, base * 0.6 + 0.4), (2, base)):
            x, y = np.rint(metadata["atoms"][index]["screen_xy_px"]).astype(int)
            np.testing.assert_allclose(image[y, x], expected, atol=2 / 255)
        first = np.asarray(metadata["atoms"][0]["screen_xy_px"])
        second = np.asarray(metadata["atoms"][1]["screen_xy_px"])
        x, y = np.rint(0.75 * first + 0.25 * second).astype(int)
        # A visible part of the bond is gray, and disappears with bonds=False.
        neighborhood = image[y - 1:y + 2, x - 1:x + 2]
        gray = neighborhood.reshape(-1, 3)[
            np.argmin(neighborhood.reshape(-1, 3).sum(axis=1))
        ]
        self.assertLess(gray.max(), 0.9)
        self.assertLess(gray.max() - gray.min(), 0.03)
        np.testing.assert_allclose(no_bonds[y, x], [1, 1, 1], atol=1 / 255)


if __name__ == "__main__":
    unittest.main()
