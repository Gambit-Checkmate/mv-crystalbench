"""Ground-truth coordination and shared displayed-bond regression checks."""

from itertools import product
import json
import unittest

import numpy as np
from ase import Atoms

from bonding import (
    bond_cutoffs, bond_minima, coordination_neighbors, coordination_number,
    display_bonds,
)
from structures import get_structures


class BondingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = get_structures()

    def test_every_requested_coordination_at_every_matching_site(self):
        expected = {
            "NaCl": {"Na": 6},
            "CsCl": {"Cs": 8},
            "ZnS_zincblende": {"Zn": 4},
            "ZnS_wurtzite": {"Zn": 4},
            "TiO2_rutile": {"Ti": 6},
            "CaF2_fluorite": {"Ca": 8, "F": 4},
            "SrTiO3_perovskite": {"Ti": 6},
            "Cu_fcc": {"Cu": 12},
            "Fe_bcc": {"Fe": 8},
            "MgAl2O4_spinel": {"Mg": 4, "Al": 6},
        }
        for name, counts in expected.items():
            atoms = self.structures[name]
            for index, symbol in enumerate(atoms.get_chemical_symbols()):
                if symbol in counts:
                    with self.subTest(structure=name, index=index, element=symbol):
                        self.assertEqual(coordination_number(atoms, index), counts[symbol])

    def test_metadata_and_pair_minima_for_entire_library(self):
        for name, atoms in self.structures.items():
            with self.subTest(structure=name):
                self.assertIsInstance(atoms.info["bond_pairs"], list)
                json.dumps(atoms.info)
                minima = bond_minima(atoms)
                cutoffs = bond_cutoffs(atoms)
                self.assertEqual(
                    set(minima),
                    {tuple(sorted(pair)) for pair in atoms.info["bond_pairs"]},
                )
                for pair, distance in minima.items():
                    self.assertGreater(distance, 0)
                    self.assertAlmostEqual(cutoffs[pair], 1.10 * distance)
        self.assertAlmostEqual(bond_minima(self.structures["NaCl"])[("Cl", "Na")], 2.82)

    def test_image_translations_are_physical_and_match_displayed_bonds(self):
        for name in ("NaCl", "CsCl", "TiO2_rutile", "CaF2_fluorite", "MgAl2O4_spinel"):
            atoms = self.structures[name]
            center = 0
            neighbors = coordination_neighbors(atoms, center)
            points = [atoms.positions[center]]
            numbers = [atoms.numbers[center]]
            for neighbor in neighbors:
                point = (
                    atoms.positions[neighbor["original_index"]]
                    + np.asarray(neighbor["translation"]) @ atoms.cell
                )
                self.assertAlmostEqual(
                    np.linalg.norm(point - atoms.positions[center]),
                    neighbor["distance"],
                )
                points.append(point)
                numbers.append(atoms.numbers[neighbor["original_index"]])
            displayed = Atoms(numbers=numbers, positions=points)
            displayed_edges = display_bonds(atoms, displayed)
            from_center = [edge for edge in displayed_edges if edge[0] == 0]
            self.assertEqual(len(from_center), len(neighbors))
            self.assertEqual({edge[1] for edge in from_center}, set(range(1, len(points))))

    def test_no_spurious_cation_cation_or_strontium_bonds(self):
        for name in ("CsCl", "SrTiO3_perovskite", "TiO2_rutile", "CaF2_fluorite"):
            atoms = self.structures[name]
            displayed = atoms.repeat((2, 2, 2))
            edges = display_bonds(atoms, displayed)
            allowed = {tuple(sorted(pair)) for pair in atoms.info["bond_pairs"]}
            self.assertTrue(edges)
            for i, j, distance in edges:
                pair = tuple(sorted((displayed[i].symbol, displayed[j].symbol)))
                self.assertIn(pair, allowed)
                self.assertLessEqual(distance, bond_cutoffs(atoms)[pair] + 1e-9)
        perovskite = self.structures["SrTiO3_perovskite"]
        for atom in perovskite:
            if atom.symbol == "Sr":
                self.assertEqual(coordination_number(perovskite, atom.index), 0)

    def test_single_site_metal_and_skew_cell_are_not_limited_to_adjacent_images(self):
        primitive_cu = Atoms(
            "Cu", scaled_positions=[[0, 0, 0]],
            cell=[[0, 1.8, 1.8], [1.8, 0, 1.8], [1.8, 1.8, 0]], pbc=True,
        )
        self.assertEqual(coordination_number(primitive_cu, 0), 12)
        skew = Atoms(
            "C", scaled_positions=[[0.3, 0.4, 0.5]],
            cell=[[3, 0, 0], [1.4, 0.1, 0], [0, 0, 3]], pbc=True,
        )
        candidates = [
            (shift, np.linalg.norm(np.asarray(shift) @ skew.cell))
            for shift in product(range(-12, 13), repeat=3)
            if shift != (0, 0, 0)
        ]
        d_min = min(distance for _, distance in candidates)
        expected = {
            shift for shift, distance in candidates
            if distance <= 1.1 * d_min + 1e-9
        }
        actual = {tuple(n["translation"]) for n in coordination_neighbors(skew, 0)}
        self.assertEqual(actual, expected)
        self.assertTrue(any(max(map(abs, shift)) > 1 for shift in actual))
        self.assertAlmostEqual(bond_minima(skew)[("C", "C")], d_min)

    def test_changed_geometry_and_symbols_do_not_use_stale_cache(self):
        atoms = self.structures["NaCl"].copy()
        original = bond_minima(atoms)
        atoms.positions += [9.7, -4.2, 3.1]
        self.assertAlmostEqual(bond_minima(atoms)[("Cl", "Na")], original[("Cl", "Na")])
        self.assertEqual(coordination_number(atoms, 0), 6)
        atoms.set_cell(atoms.cell.array * 1.08, scale_atoms=True)
        self.assertAlmostEqual(bond_minima(atoms)[("Cl", "Na")], original[("Cl", "Na")] * 1.08)
        self.assertEqual(coordination_number(atoms, 0), 6)
        symbols = atoms.get_chemical_symbols()
        atoms.set_chemical_symbols(["Cl" if s == "Na" else "Na" for s in symbols])
        self.assertEqual(coordination_number(atoms, 0), 6)

    def test_displayed_distance_is_not_replaced_by_minimum_image(self):
        atoms = self.structures["NaCl"]
        displayed = Atoms("NaCl", positions=[[0, 0, 0], [8.46, 0, 0]])
        self.assertEqual(display_bonds(atoms, displayed), [])
        displayed.positions[1] = [2.82, 0, 0]
        self.assertEqual(len(display_bonds(atoms, displayed)), 1)

    def test_missing_species_and_undeclared_multielement_policy(self):
        atoms = self.structures["CsCl"].copy()
        del atoms[1]
        self.assertEqual(bond_minima(atoms), {})
        self.assertEqual(coordination_number(atoms, 0), 0)
        atoms = self.structures["CsCl"].copy()
        del atoms.info["bond_pairs"]
        with self.assertRaisesRegex(ValueError, "bond_pairs"):
            bond_minima(atoms)


if __name__ == "__main__":
    unittest.main()
