"""Color science, mixed-camera and selected-environment regression checks."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from matplotlib.colors import to_rgb
from matplotlib.image import imread

import render
from structures import get_structures


class ColorAndSceneTests(unittest.TestCase):
    def test_srgb_d65_reference_values(self):
        # Standard D65 sRGB reference points: white, black, red.
        expected = [[100, 0, 0], [0, 0, 0], [53.2408, 80.0925, 67.2032]]
        actual = render.rgb_to_lab([[1, 1, 1], [0, 0, 0], [1, 0, 0]])
        np.testing.assert_allclose(actual, expected, atol=5e-4)
        for invalid in ([1, 2, 3], [0, 0], [np.nan, 0, 0]):
            with self.assertRaises(ValueError):
                render.rgb_to_lab(invalid)

    def test_default_jmol_retained_for_distinct_elements_and_single_element(self):
        for atoms in (Atoms("NaCl"), Atoms("Cu")):
            colors = render.element_colors(atoms)
            for symbol, color in colors.items():
                np.testing.assert_allclose(color, jmol_colors[atomic_numbers[symbol]])

    def test_close_element_colors_trigger_sorted_fixed_palette_for_entire_structure(self):
        colors = jmol_colors.copy()
        # Force two elements to be exactly identical; a third remains distinct.
        # The fallback must apply to ALL elements, independent of atom ordering.
        colors[atomic_numbers["Mg"]] = colors[atomic_numbers["Al"]]
        with patch.object(render, "jmol_colors", colors):
            first = render.element_colors(Atoms("OMgAl"))
            second = render.element_colors(Atoms("AlMgO"))
        for index, symbol in enumerate(("Al", "Mg", "O")):
            expected = to_rgb(render.HIGH_CONTRAST_PALETTE[index])
            np.testing.assert_allclose(first[symbol], expected)
            np.testing.assert_allclose(first[symbol], second[symbol])

    def test_every_structure_matches_the_delta_e_switch_rule(self):
        for name, atoms in get_structures().items():
            with self.subTest(structure=name):
                symbols = sorted(set(atoms.get_chemical_symbols()))
                expected_jmol = np.array([jmol_colors[atomic_numbers[s]] for s in symbols])
                lab = render.rgb_to_lab(expected_jmol)
                close = any(
                    np.linalg.norm(lab[i] - lab[j]) < 40
                    for i in range(len(symbols)) for j in range(i + 1, len(symbols))
                ) or np.any(np.linalg.norm(
                    lab - render.rgb_to_lab(to_rgb(render.HIGHLIGHT_COLOR)), axis=-1) < 40)
                fog_lab = render.rgb_to_lab(expected_jmol * 0.6 + 0.4)
                close = close or any(np.linalg.norm(fog_lab[i] - fog_lab[j]) < 40
                                     for i in range(len(symbols)) for j in range(i))
                actual = render.element_colors(atoms)
                for index, symbol in enumerate(symbols):
                    expected = (to_rgb(render.HIGH_CONTRAST_PALETTE[index])
                                if close else expected_jmol[index])
                    np.testing.assert_allclose(actual[symbol], expected)
                actual_lab = render.rgb_to_lab(np.array(list(actual.values())))
                highlight_lab = render.rgb_to_lab(to_rgb(render.HIGHLIGHT_COLOR))
                for i in range(len(symbols)):
                    self.assertGreaterEqual(float(np.linalg.norm(actual_lab[i] - highlight_lab)), 40)
                    for j in range(i):
                        self.assertGreaterEqual(float(np.linalg.norm(actual_lab[i] - actual_lab[j])), 40)

    def test_caf2_and_highlight_conflicts_trigger_fallback(self):
        colors, mode, original_minimum = render._element_palette(Atoms("CaF2"))
        self.assertEqual(mode, "fixed_high_contrast")
        self.assertGreater(original_minimum, 40)  # The threshold alone is insufficient.
        original = np.array([jmol_colors[atomic_numbers[s]] for s in ("Ca", "F")])
        fog_minimum, _ = render._color_separation(original * 0.6 + 0.4)
        self.assertLess(fog_minimum, 40)
        changed = jmol_colors.copy()
        changed[atomic_numbers["Cu"]] = to_rgb(render.HIGHLIGHT_COLOR)
        with patch.object(render, "jmol_colors", changed):
            colors, mode, _ = render._element_palette(Atoms("Cu"))
            self.assertEqual(mode, "fixed_high_contrast")
            pair, highlight = render._color_separation(np.array(list(colors.values())))
            self.assertIsNone(pair)
            self.assertGreaterEqual(highlight, 40)

    def test_entire_fallback_palette_is_separated(self):
        pair, highlight = render._color_separation(np.array(
            [to_rgb(color) for color in render.HIGH_CONTRAST_PALETTE]))
        self.assertGreaterEqual(pair, 40)
        self.assertGreaterEqual(highlight, 40)

    def test_palette_metadata_and_atom_pixels_use_same_base_colors(self):
        atoms = Atoms("AlMgO", positions=[[1, 1, 1], [3, 1, 1], [5, 1, 1]],
                      cell=[6, 3, 3], pbc=True)
        atoms.info["bond_pairs"] = [("Al", "O"), ("Mg", "O")]
        colors = jmol_colors.copy()
        colors[atomic_numbers["Mg"]] = colors[atomic_numbers["Al"]]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(render, "jmol_colors", colors):
                path = render.render_views(atoms, [(0, 0, 1)], directory, tilt=0)[0]
            metadata = json.loads(path.with_suffix(".json").read_text("utf-8"))
            pixels = imread(path)[:, :, :3]
        self.assertEqual(metadata["color_policy"]["palette"], "fixed_high_contrast")
        for item in metadata["atoms"]:
            x, y = np.rint(item["screen_xy_px"]).astype(int)
            base = np.array(metadata["color_policy"]["element_rgb"][item["element"]])
            fog = item["fog_white_fraction"]
            np.testing.assert_allclose(pixels[y, x], base * (1 - fog) + fog, atol=2 / 255)

    def test_mixed_tilts_share_scale_and_match_visible_picker(self):
        atoms = get_structures()["NaCl"]
        directions = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (0.1, 0.3, 0.7)]
        tilts = [8, 8, 8, 0]
        with tempfile.TemporaryDirectory() as directory:
            paths = render.render_views(atoms, directions, directory, tilt=tilts)
            data = [json.loads(path.with_suffix(".json").read_text("utf-8")) for path in paths]
            self.assertTrue(all("tilt_8deg" in path.name for path in paths[:3]))
            self.assertIn("strict", paths[3].name)
        self.assertEqual([item["camera"]["tilt_degrees"] for item in data], tilts)
        self.assertEqual(len({item["camera"]["pixels_per_angstrom"] for item in data}), 1)
        worst = np.min([[atom["visible_fraction"] for atom in item["atoms"]] for item in data], axis=0)
        picked = render.pick_visible_atoms(atoms, directions, len(worst), tilt=tilts)
        self.assertEqual(set(picked), set(np.flatnonzero(worst >= 0.7)))
        with self.assertRaises(ValueError):
            render._prepare_views(atoms, directions, "cell", (2, 2, 2), None, None, [8])

    def test_display_selection_preserves_mapping_and_renumbers_highlights(self):
        atoms = get_structures()["NaCl"]
        shown, mapping = render.build_display_atoms(atoms)
        selection = [4, 2, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = render.render_views(
                atoms, [(1, 1, 1)], directory, display_indices=selection, highlight={1: "A"}
            )[0]
            data = json.loads(path.with_suffix(".json").read_text("utf-8"))
        self.assertEqual(data["display_indices"], selection)
        self.assertEqual(len(data["atoms"]), 3)
        for index, atom in enumerate(data["atoms"]):
            original_index, translation = mapping[selection[index]]
            self.assertEqual(atom["display_index"], index)
            self.assertEqual(atom["original_index"], original_index)
            self.assertEqual(tuple(atom["translation"]), tuple(translation))
            np.testing.assert_allclose(atom["position_angstrom"], shown.positions[selection[index]])
        self.assertEqual(data["atoms"][1]["highlight"], "A")
        for invalid in ([], [0, 0], [-1], [len(shown)], [True]):
            with self.subTest(selection=invalid), self.assertRaises(ValueError):
                render._prepare_views(atoms, [(1, 1, 1)], "cell", (2, 2, 2),
                                      None, None, 8, invalid)


if __name__ == "__main__":
    unittest.main()
