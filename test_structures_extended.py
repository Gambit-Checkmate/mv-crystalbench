"""Source provenance, diversity and coordination checks for the 48-cell catalog."""

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
from ase.neighborlist import neighbor_list

from bonding import coordination_number
from structure_catalog import (
    EXPECTED_COORDINATION, Q2_EXCLUSIONS, SOURCE_COMMIT, SOURCE_ROOT,
    crystal_system_name,
)
from structures import analyze_structure, get_structures


class ExtendedStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structures = get_structures()

    def test_catalog_meets_size_and_actual_symmetry_diversity_requirements(self):
        self.assertEqual(len(self.structures), 48)
        systems = Counter()
        for name, atoms in self.structures.items():
            with self.subTest(structure=name):
                measured = analyze_structure(atoms)
                self.assertEqual(
                    measured["spacegroup_number"], atoms.info["expected_spacegroup_number"]
                )
                system = crystal_system_name(measured["spacegroup_number"])
                self.assertEqual(system, atoms.info["crystal_system"])
                self.assertEqual(atoms.info["base_structure"], name)
                systems[system] += 1
        self.assertEqual(systems, {
            "cubic": 12, "tetragonal": 6, "trigonal": 6,
            "hexagonal": 8, "orthorhombic": 8, "monoclinic": 8,
        })
        self.assertLessEqual(systems["cubic"] / len(self.structures), 0.35)
        self.assertGreaterEqual(min(systems.values()), 6)

    def test_original_twenty_structures_retain_exact_geometry(self):
        reference_path = SOURCE_ROOT.parent / "original_geometry_sha256.json"
        reference = json.loads(reference_path.read_text("utf-8"))
        self.assertEqual(len(reference), 20)
        for name, expected_hash in reference.items():
            atoms = self.structures[name]
            geometry = {
                "numbers": atoms.numbers.tolist(), "cell": atoms.cell.array.tolist(),
                "positions": atoms.positions.tolist(),
            }
            actual_hash = hashlib.sha256(json.dumps(
                geometry, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()
            with self.subTest(structure=name):
                self.assertEqual(actual_hash, expected_hash)

    def test_every_source_is_pinned_and_has_original_paper_citation(self):
        selection = json.loads((SOURCE_ROOT / "selection.json").read_text("utf-8"))
        self.assertEqual(selection["commit"], SOURCE_COMMIT)
        self.assertEqual(len(selection["selected"]), 28)
        self.assertTrue((SOURCE_ROOT / "LICENSE").is_file())
        self.assertTrue((SOURCE_ROOT / "NOTICE").is_file())
        self.assertNotIn("TiO2_brookite", self.structures)
        for name, label in selection["selected"].items():
            atoms = self.structures[name]
            with self.subTest(structure=name):
                self.assertEqual(atoms.info["source_commit"], SOURCE_COMMIT)
                self.assertEqual(atoms.info["source_prototype_label"], label)
                self.assertTrue(atoms.info["source_citation_bibtex"].strip())
                for filename, expected in selection["source_sha256"][label].items():
                    actual = hashlib.sha256((SOURCE_ROOT / label / filename).read_bytes()).hexdigest()
                    self.assertEqual(actual, expected)
                self.assertEqual(
                    atoms.info["source_cif_sha256"],
                    selection["source_sha256"][label]["aflow.cif"],
                )

    def test_q2_eligibility_is_explicit_and_exclusions_explain_the_short_shell(self):
        self.assertEqual(sum(a.info["q2_eligible"] for a in self.structures.values()), 39)
        for name, atoms in self.structures.items():
            with self.subTest(structure=name):
                self.assertIsInstance(atoms.info["q2_eligible"], bool)
                if not atoms.info["q2_eligible"]:
                    self.assertIn(name, Q2_EXCLUSIONS)
                    self.assertGreater(len(atoms.info["q2_ineligible_reason"]), 40)
                self.assertTrue(atoms.info["bond_pairs"])
                symbols = set(atoms.get_chemical_symbols())
                for pair in atoms.info["bond_pairs"]:
                    self.assertEqual(len(pair), 2)
                    self.assertTrue(set(pair) <= symbols)
                json.dumps(atoms.info, allow_nan=False)

    def test_extended_eligible_coordination_matches_known_local_geometry(self):
        for name, expected in EXPECTED_COORDINATION.items():
            atoms = self.structures[name]
            measured = defaultdict(set)
            for index, symbol in enumerate(atoms.get_chemical_symbols()):
                measured[symbol].add(coordination_number(atoms, index))
            with self.subTest(structure=name):
                self.assertTrue(atoms.info["q2_eligible"])
                self.assertEqual(
                    {symbol: sorted(values) for symbol, values in measured.items()},
                    expected,
                )

    def test_distorted_shells_are_excluded_without_overriding_the_shared_rule(self):
        # These examples deliberately preserve the globally defined cutoff
        # rather than adding crystal-specific exceptions to recover textbook CN.
        examples = {
            "ZrO2_baddeleyite": ("Zr", 6),
            "Ga2O3_beta": ("Ga", 4),
            "PbCl2_cotunnite": ("Pb", 7),
            "BaTiO3_tetragonal": ("Ti", 5),
            "LiNbO3_ferroelectric": ("Li", 3),
        }
        for name, (element, expected) in examples.items():
            atoms = self.structures[name]
            self.assertFalse(atoms.info["q2_eligible"])
            for index, symbol in enumerate(atoms.get_chemical_symbols()):
                if symbol == element:
                    with self.subTest(structure=name, index=index):
                        self.assertEqual(coordination_number(atoms, index), expected)

    def test_expanded_cells_have_no_duplicate_sites_or_unphysical_tiny_contacts(self):
        for name, atoms in self.structures.items():
            with self.subTest(structure=name):
                self.assertTrue(np.all(atoms.pbc))
                self.assertGreater(atoms.get_volume(), 0)
                self.assertTrue(np.isfinite(atoms.positions).all())
                # All included models are heavy-atom solids; any contact below
                # 0.5 A would expose duplicated symmetry images or a bad basis.
                distances = neighbor_list("d", atoms, 0.5, self_interaction=False)
                self.assertEqual(len(distances), 0)

    def test_repeated_calls_return_independent_atoms_and_metadata(self):
        first = get_structures()
        second = get_structures()
        name = "CuO_tenorite"
        first[name].positions[0] += [0.1, 0.2, 0.3]
        first[name].info["bond_pairs"][0][0] = "H"
        self.assertFalse(np.array_equal(first[name].positions, second[name].positions))
        self.assertNotEqual(first[name].info["bond_pairs"], second[name].info["bond_pairs"])


if __name__ == "__main__":
    unittest.main()
