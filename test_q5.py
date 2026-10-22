"""Scientific negative-sample and symmetry-equivalence regression tests."""

from collections import Counter
from itertools import combinations
import random
import unittest

import numpy as np

from question_utils import variant_atoms
from questions import (
    POLYMORPHS, Q5_OPTIONS, _q5_mutation, directions_equivalent, generate_q5,
    q6_directions, species_assignment_equivalent, symmetry_dataset,
)
from structures import get_structures


class _MemorySceneStore:
    """Avoid PNG generation while checking Q5 scene grouping and mutations."""

    def __init__(self):
        self.scenes = {}
        self.atoms = {}

    def get(self, atoms, structure_key, **kwargs):
        scene_id = f"s{len(self.scenes):04d}"
        scene = {
            "scene_id": scene_id, "text_repr": "Geometry-only text",
            "views": [
                {"view_id": f"v{index:02d}", "image_path": f"{scene_id}/v{index:02d}.png"}
                for index in range(8)
            ],
        }
        self.scenes[scene_id] = scene
        self.atoms[scene_id] = atoms.copy()
        return scene


class Q5NegativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = get_structures()

    @staticmethod
    def global_swap(atoms):
        first, second = sorted(set(atoms.get_chemical_symbols()))
        changed = atoms.copy()
        changed.set_chemical_symbols([
            second if symbol == first else first
            for symbol in atoms.get_chemical_symbols()
        ])
        return changed

    def test_global_sublattice_swaps_are_not_false_negatives(self):
        # A different color at every drawn site does not guarantee a distinct
        # infinite crystal: lattice symmetries can exchange whole sublattices.
        for name in ("NaCl", "CsCl", "ZnS_zincblende", "ZnS_wurtzite", "SiC_3C"):
            for variant in (0, 1, 7):
                atoms, _ = variant_atoms(self.structures[name], variant, 37)
                changed = self.global_swap(atoms)
                with self.subTest(structure=name, origin_variant=variant):
                    self.assertTrue(species_assignment_equivalent(atoms, changed))
                    self.assertIsNone(_q5_mutation(
                        "swap_elements", name, atoms, self.structures,
                        random.Random(17), variant, 37,
                    ))

    def test_same_composition_does_not_imply_equivalent_species_assignment(self):
        # Ni and As occupy inequivalent environments even though their counts
        # are equal. A comparison of formulas alone would lose this negative.
        atoms = self.structures["NiAs"]
        changed = self.global_swap(atoms)
        self.assertEqual(Counter(atoms.numbers), Counter(changed.numbers))
        self.assertFalse(species_assignment_equivalent(atoms, changed))
        self.assertIsNotNone(_q5_mutation(
            "swap_elements", "NiAs", atoms, self.structures, random.Random(3), 0, 7,
        ))

    def test_equivalence_is_invariant_to_atom_order_and_cell_translations(self):
        atoms = self.structures["NaCl"]
        changed = self.global_swap(atoms)[::-1]
        changed.positions += np.array([2, -1, 3]) @ atoms.cell.array
        self.assertTrue(species_assignment_equivalent(atoms, changed))
        self.assertTrue(species_assignment_equivalent(changed, atoms))
        self.assertTrue(species_assignment_equivalent(atoms, atoms[::-1]))

    def test_all_retained_swaps_are_inequivalent_and_preserve_geometry(self):
        for name, atoms in self.structures.items():
            positions = atoms.positions.copy()
            numbers = atoms.numbers.copy()
            cell = atoms.cell.array.copy()
            candidate = _q5_mutation(
                "swap_elements", name, atoms, self.structures,
                random.Random(51), 0, 42,
            )
            with self.subTest(structure=name):
                np.testing.assert_array_equal(atoms.positions, positions)
                np.testing.assert_array_equal(atoms.numbers, numbers)
                np.testing.assert_array_equal(atoms.cell.array, cell)
                if candidate is not None:
                    changed, details = candidate
                    self.assertFalse(species_assignment_equivalent(atoms, changed))
                    np.testing.assert_array_equal(changed.positions, positions)
                    np.testing.assert_array_equal(changed.cell.array, cell)
                    self.assertEqual(len(details["swapped_elements"]), 2)

    def test_delete_atom_retains_species_and_changes_exactly_one_population(self):
        atoms = self.structures["TiO2_rutile"]
        changed, details = _q5_mutation(
            "delete_atom", "TiO2_rutile", atoms, self.structures,
            random.Random(83), 0, 42,
        )
        self.assertEqual(len(changed), len(atoms) - 1)
        self.assertEqual(set(changed.get_chemical_symbols()), set(atoms.get_chemical_symbols()))
        difference = Counter(atoms.get_chemical_symbols()) - Counter(changed.get_chemical_symbols())
        self.assertEqual(difference, {details["deleted_element"]: 1})
        np.testing.assert_array_equal(changed.cell.array, atoms.cell.array)
        expected = atoms.copy()
        del expected[details["deleted_original_index"]]
        np.testing.assert_array_equal(changed.positions, expected.positions)
        # Removing either site from this cell would remove an element entirely.
        self.assertIsNone(_q5_mutation(
            "delete_atom", "CsCl", self.structures["CsCl"], self.structures,
            random.Random(83), 0, 42,
        ))

    def test_axis_stretch_is_exactly_eight_percent_at_fixed_fractional_sites(self):
        atoms = self.structures["Mg_hcp"]
        changed, details = _q5_mutation(
            "stretch_axis", "Mg_hcp", atoms, self.structures,
            random.Random(31), 0, 42,
        )
        expected_cell = atoms.cell.array.copy()
        expected_cell[details["stretched_axis"]] *= 1.08
        np.testing.assert_allclose(changed.cell.array, expected_cell, atol=1e-12)
        np.testing.assert_allclose(
            changed.get_scaled_positions(), atoms.get_scaled_positions(), atol=1e-12,
        )
        self.assertAlmostEqual(changed.get_volume() / atoms.get_volume(), 1.08)
        np.testing.assert_array_equal(changed.numbers, atoms.numbers)

    def test_polymorph_pairs_have_same_composition_and_different_symmetry(self):
        for name, alternate in POLYMORPHS.items():
            atoms = self.structures[name]
            candidate = _q5_mutation(
                "polymorph", name, atoms, self.structures, random.Random(4), 3, 42,
            )
            with self.subTest(structure=name):
                self.assertIsNotNone(candidate)
                changed, details = candidate
                self.assertEqual(details["other_structure"], alternate)
                counts_a = Counter(atoms.get_chemical_symbols())
                counts_b = Counter(changed.get_chemical_symbols())
                self.assertEqual(set(counts_a), set(counts_b))
                for symbol in counts_a:
                    self.assertAlmostEqual(
                        counts_a[symbol] / len(atoms), counts_b[symbol] / len(changed),
                    )
                self.assertNotEqual(
                    int(symmetry_dataset(atoms).number),
                    int(symmetry_dataset(changed).number),
                )

    def test_generated_q5_balances_valid_negatives_and_uses_different_view_groups(self):
        store = _MemorySceneStore()
        questions = generate_q5(
            self.structures, store, records=[], target=16, seed=20261021,
        )
        self.assertEqual(len(questions), 16)
        positives = [q for q in questions if q["meta"]["is_same_structure"]]
        negatives = [q for q in questions if not q["meta"]["is_same_structure"]]
        self.assertEqual(len(positives), len(negatives))
        self.assertEqual(Counter(q["meta"]["negative_type"] for q in negatives), {
            "swap_elements": 2, "delete_atom": 2, "stretch_axis": 2, "polymorph": 2,
        })
        answers = {
            None: Q5_OPTIONS[0], "swap_elements": Q5_OPTIONS[1],
            "delete_atom": Q5_OPTIONS[2], "polymorph": Q5_OPTIONS[2],
            "stretch_axis": Q5_OPTIONS[3],
        }
        for question in questions:
            kind = question["meta"]["negative_type"]
            self.assertEqual(question["answer_text"], answers[kind])
            self.assertEqual(
                [reference.rsplit(":", 1)[1] for reference in question["view_ids"]],
                ["v00", "v01", "v02", "v03", "v04", "v05"],
            )
            first_id = question["scene_id"]
            second_id = question["meta"]["secondary_scene_id"]
            if kind is None:
                self.assertEqual(first_id, second_id)
            else:
                self.assertNotEqual(first_id, second_id)
            if kind == "swap_elements":
                self.assertFalse(species_assignment_equivalent(
                    store.atoms[first_id], store.atoms[second_id],
                ))


class Q6SymmetryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = get_structures()

    def test_hexagonal_axes_are_checked_in_the_fractional_basis(self):
        atoms = self.structures["Mg_hcp"]
        rotations = symmetry_dataset(atoms).rotations
        self.assertTrue(directions_equivalent(
            [1, 0, 0], [1, 1, 0], rotations, atoms.cell.array,
        ))
        self.assertFalse(directions_equivalent(
            [0, 0, 1], [1, 0, 0], rotations, atoms.cell.array,
        ))
        self.assertIsNone(q6_directions(atoms))

    def test_cubic_axis_families_are_distinct_but_permuted_axes_are_equivalent(self):
        atoms = self.structures["NaCl"]
        rotations = symmetry_dataset(atoms).rotations
        self.assertTrue(directions_equivalent(
            [1, 0, 0], [0, 1, 0], rotations, atoms.cell.array,
        ))
        result = q6_directions(atoms)
        self.assertIsNotNone(result)
        choices, directions, _, _ = result
        self.assertEqual(choices, ["<100>", "<110>", "<111>", "<112>"])
        for first, second in combinations(directions, 2):
            self.assertFalse(directions_equivalent(first, second, rotations, atoms.cell.array))

    def test_every_retained_question_has_four_pairwise_inequivalent_directions(self):
        retained, rejected = [], []
        for name, atoms in self.structures.items():
            result = q6_directions(atoms)
            if result is None:
                rejected.append(name)
                continue
            retained.append(name)
            _, directions, _, _ = result
            rotations = symmetry_dataset(atoms).rotations
            for first, second in combinations(directions, 2):
                with self.subTest(structure=name, first=first, second=second):
                    self.assertFalse(directions_equivalent(
                        first, second, rotations, atoms.cell.array,
                    ))
        self.assertIn("TiO2_rutile", retained)
        self.assertIn("ZnS_wurtzite", rejected)
        self.assertIn("C_graphite", rejected)
        self.assertIn("Al2O3_corundum", rejected)


if __name__ == "__main__":
    unittest.main()
