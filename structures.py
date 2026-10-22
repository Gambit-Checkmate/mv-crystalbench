"""Deterministic 48-structure crystal library for MV-CrystalBench.

All lengths are in angstroms.  These are ideal, fully occupied, periodic
prototype models with rounded, representative lattice parameters, not measured
samples or relaxed ground-state structures.  There is no magnetic symmetry,
thermal motion, disorder, partial occupancy, or cation inversion.

The initial 20 cells retain their original geometry. An additional 28
non-cubic prototypes are reconstructed from bundled, pinned AFLOW sources by
structure_catalog.py, with source citations and Q2 eligibility in atoms.info.

Cubic structures use conventional cubic cells; hexagonal structures use the
standard three-vector hexagonal cell.  Corundum uses its 30-atom hexagonal
setting.  No duplicate atoms are added at the opposite faces of a cell.

Useful references for the constructors and symmetry conventions:
https://docs.ase-lib.org/ase/spacegroup/spacegroup.html
https://docs.ase-lib.org/_modules/ase/build/bulk.html
https://spglib.readthedocs.io/en/v2.7.0/api/python-api.html
https://www.aflowlib.org/prototype-encyclopedia/A2B_tI12_141_e_a-001/
https://aflow.org/prototype-encyclopedia/A2BC4_cF56_227_c_b_e-001/
https://www.aflowlib.org/prototype-encyclopedia/prototype_index.html

Run ``python structures.py`` to print and check all structures.  Every call to
``get_structures()`` builds fresh Atoms objects and checks their space groups,
cell populations, and compositions.  It raises on failure, including under
``python -O``.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import spglib
from ase import Atoms
from ase.build import bulk
from ase.formula import Formula
from ase.spacegroup import crystal


SYMPREC = 1.0e-5  # Cartesian distance tolerance in angstroms for ideal models.

# Element pairs allowed by the benchmark's first-neighbor bond definition.
# Values remain plain lists so all atoms.info metadata is JSON serializable.
# A pair is unordered; bonding.py canonicalizes the order of its symbols.
BOND_PAIRS = {
    "NaCl": [["Na", "Cl"]],
    "CsCl": [["Cs", "Cl"]],
    "ZnS_zincblende": [["Zn", "S"]],
    "ZnS_wurtzite": [["Zn", "S"]],
    "Si_diamond": [["Si", "Si"]],
    "C_diamond": [["C", "C"]],
    "Cu_fcc": [["Cu", "Cu"]],
    "Fe_bcc": [["Fe", "Fe"]],
    "Mg_hcp": [["Mg", "Mg"]],
    "SrTiO3_perovskite": [["Ti", "O"]],
    "TiO2_rutile": [["Ti", "O"]],
    "CaF2_fluorite": [["Ca", "F"]],
    "MgAl2O4_spinel": [["Mg", "O"], ["Al", "O"]],
    "Al2O3_corundum": [["Al", "O"]],
    "NiAs": [["Ni", "As"]],
    "FeS2_pyrite": [["Fe", "S"], ["S", "S"]],
    "TiO2_anatase": [["Ti", "O"]],
    "Sn_beta": [["Sn", "Sn"]],
    "C_graphite": [["C", "C"]],
    "SiC_3C": [["Si", "C"]],
}


def analyze_structure(atoms: Atoms, symprec: float = SYMPREC) -> dict[str, Any]:
    """Compute JSON-compatible structural metadata without changing ``atoms``.

    Space-group analysis includes element identities and periodic geometry.
    Initial magnetic moments, if supplied by ASE, are deliberately ignored.
    ``formula`` is the empirical Hill formula (e.g. ClNa); ``cell_formula``
    preserves the population of the supplied cell (e.g. Cl4Na4).
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ase.Atoms object")
    if len(atoms) == 0:
        raise ValueError("Cannot analyze an empty structure")
    if not np.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be a finite positive distance")
    if not np.all(atoms.pbc):
        raise ValueError("Crystal symmetry requires periodicity in all 3 axes")
    if not np.all(np.isfinite(atoms.positions)):
        raise ValueError("Atomic positions must be finite")
    cell = np.asarray(atoms.cell, dtype=float)
    if not np.all(np.isfinite(cell)) or abs(np.linalg.det(cell)) < 1.0e-10:
        raise ValueError("The crystal cell must be finite and nonsingular")
    if np.any(atoms.numbers <= 0):
        raise ValueError("Every atom must have a real chemical element")

    dataset = spglib.get_symmetry_dataset(
        (cell, atoms.get_scaled_positions(wrap=True), atoms.numbers),
        symprec=float(symprec),
    )
    if dataset is None:
        raise ValueError("spglib could not determine the crystal symmetry")

    return {
        "formula": atoms.get_chemical_formula(mode="hill", empirical=True),
        "cell_formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "spacegroup_number": int(dataset.number),
        "spacegroup_symbol": str(dataset.international),
        "symprec_angstrom": float(symprec),
        "cell_parameters": atoms.cell.cellpar().tolist(),
    }


def _cubic(
    symbols: str | list[str],
    basis: list[tuple[float, float, float]],
    spacegroup: int,
    a: float,
    setting: int = 1,
) -> Atoms:
    return crystal(
        symbols,
        basis=basis,
        spacegroup=spacegroup,
        setting=setting,
        cellpar=[a, a, a, 90, 90, 90],
        pbc=True,
        primitive_cell=False,
    )


def _check_prototype(
    name: str,
    atoms: Atoms,
    expected_spacegroup: int,
    expected_formula: str,
    formula_units: int,
) -> None:
    """Fail explicitly if a constructor creates the wrong prototype cell."""
    expected_counts = {
        symbol: count * formula_units
        for symbol, count in Formula(expected_formula).count().items()
    }
    actual_counts = dict(Counter(atoms.get_chemical_symbols()))
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"{name}: wrong cell composition: "
            f"expected {expected_counts}, got {actual_counts}"
        )
    detected = atoms.info["spacegroup_number"]
    if detected != expected_spacegroup:
        raise RuntimeError(
            f"{name}: expected space group {expected_spacegroup}, "
            f"got {detected} ({atoms.info['spacegroup_symbol']})"
        )


def get_structures() -> dict[str, Atoms]:
    """Return 48 named, independently constructed and symmetry-checked cells.

    The keys ``NaCl`` and ``TiO2_rutile`` are used by the rendering demo.
    ``atoms.info`` contains JSON-compatible names, formulas, cell parameters,
    measured space-group metadata, and the model assumption.

    Coordinates and populations refer to one cell with unique atoms, not the
    duplicated corner atoms often used in textbook drawings.  For a supercell,
    use ``atoms.repeat((2, 2, 2))`` and refresh size-dependent metadata with
    ``supercell.info.update(analyze_structure(supercell))``.
    """
    # Each entry: key, cell, expected SG, conventional formula, formula units,
    #            prototype, parameter/model note.
    entries = [
        (
            "NaCl",
            bulk("NaCl", "rocksalt", a=5.64, cubic=True),
            225, "NaCl", 4, "rocksalt",
            "a=5.64",
        ),
        (
            "CsCl",
            bulk("CsCl", "cesiumchloride", a=4.12, cubic=True),
            221, "CsCl", 1, "cesium chloride",
            "a=4.12",
        ),
        (
            "ZnS_zincblende",
            bulk("ZnS", "zincblende", a=5.41, cubic=True),
            216, "ZnS", 4, "zincblende",
            "a=5.41",
        ),
        (
            "ZnS_wurtzite",
            bulk("ZnS", "wurtzite", a=3.82, c=6.26, u=0.375),
            186, "ZnS", 2, "wurtzite",
            "a=3.82, c=6.26; idealized internal coordinate u=3/8",
        ),
        (
            "Si_diamond",
            bulk("Si", "diamond", a=5.43, cubic=True),
            227, "Si", 8, "diamond",
            "a=5.43",
        ),
        (
            "C_diamond",
            bulk("C", "diamond", a=3.567, cubic=True),
            227, "C", 8, "diamond",
            "a=3.567",
        ),
        (
            "Cu_fcc",
            bulk("Cu", "fcc", a=3.615, cubic=True),
            225, "Cu", 4, "face-centered cubic",
            "a=3.615",
        ),
        (
            "Fe_bcc",
            bulk("Fe", "bcc", a=2.866, cubic=True),
            229, "Fe", 2, "body-centered cubic",
            "a=2.866; structural symmetry only, no magnetic space group",
        ),
        (
            "Mg_hcp",
            bulk("Mg", "hcp", a=3.21, c=5.21),
            194, "Mg", 2, "hexagonal close-packed",
            "a=3.21, c=5.21",
        ),
        (
            "SrTiO3_perovskite",
            _cubic(
                ["Sr", "Ti", "O"],
                [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0)],
                spacegroup=221,
                a=3.905,
            ),
            221, "SrTiO3", 1, "cubic perovskite",
            "a=3.905; ideal cubic phase, no octahedral tilts",
        ),
        (
            "TiO2_rutile",
            crystal(
                ["Ti", "O"],
                basis=[(0, 0, 0), (0.305, 0.305, 0)],
                spacegroup=136,
                cellpar=[4.594, 4.594, 2.959, 90, 90, 90],
                pbc=True,
            ),
            136, "TiO2", 2, "rutile",
            "a=4.594, c=2.959, oxygen u=0.305",
        ),
        (
            "CaF2_fluorite",
            bulk("CaF2", "fluorite", a=5.462, cubic=True),
            225, "CaF2", 4, "fluorite",
            "a=5.462",
        ),
        (
            "MgAl2O4_spinel",
            # Origin choice 2: Mg 8a, Al 16d, O 32e.  Normal spinel
            # means Mg occupies tetrahedral sites and Al octahedral sites.
            _cubic(
                ["Mg", "Al", "O"],
                [(0.125, 0.125, 0.125), (0.5, 0.5, 0.5),
                 (0.261, 0.261, 0.261)],
                spacegroup=227,
                a=8.083,
                setting=2,
            ),
            227, "MgAl2O4", 8, "normal spinel",
            "a=8.083, oxygen u=0.261; origin choice 2, no Mg/Al inversion",
        ),
        (
            "Al2O3_corundum",
            crystal(
                ["Al", "O"],
                basis=[(0, 0, 0.3522), (0.306, 0, 0.25)],
                spacegroup=167,
                setting=1,
                cellpar=[4.759, 4.759, 12.991, 90, 90, 120],
                pbc=True,
            ),
            167, "Al2O3", 6, "corundum",
            "a=4.759, c=12.991; hexagonal axes, Al z=0.3522, O x=0.306",
        ),
        (
            "NiAs",
            crystal(
                ["Ni", "As"],
                basis=[(0, 0, 0), (1 / 3, 2 / 3, 0.25)],
                spacegroup=194,
                cellpar=[3.62, 3.62, 5.03, 90, 90, 120],
                pbc=True,
            ),
            194, "NiAs", 2, "nickel arsenide",
            "a=3.62, c=5.03",
        ),
        (
            "FeS2_pyrite",
            _cubic(
                ["Fe", "S"],
                [(0, 0, 0), (0.385, 0.385, 0.385)],
                spacegroup=205,
                a=5.417,
            ),
            205, "FeS2", 4, "pyrite",
            "a=5.417, sulfur u=0.385",
        ),
        (
            "TiO2_anatase",
            # Origin choice 1: Ti at (0,0,0); O z=0.208.  Do not combine
            # this z coordinate with the origin-choice-2 Wyckoff positions.
            crystal(
                ["Ti", "O"],
                basis=[(0, 0, 0), (0, 0, 0.208)],
                spacegroup=141,
                setting=1,
                cellpar=[3.784, 3.784, 9.515, 90, 90, 90],
                pbc=True,
            ),
            141, "TiO2", 4, "anatase",
            "a=3.784, c=9.515, O z=0.208 in origin choice 1",
        ),
        (
            "Sn_beta",
            crystal(
                "Sn",
                basis=[(0, 0, 0)],
                spacegroup=141,
                setting=1,
                cellpar=[5.832, 5.832, 3.182, 90, 90, 90],
                pbc=True,
            ),
            141, "Sn", 4, "beta tin",
            "a=5.832, c=3.182; origin choice 1",
        ),
        (
            "C_graphite",
            crystal(
                ["C", "C"],
                basis=[(0, 0, 0.25), (1 / 3, 2 / 3, 0.25)],
                spacegroup=194,
                cellpar=[2.46, 2.46, 6.708, 90, 90, 120],
                pbc=True,
            ),
            194, "C", 4, "hexagonal graphite",
            "a=2.46, c=6.708; Bernal (AB) stacking",
        ),
        (
            "SiC_3C",
            bulk("SiC", "zincblende", a=4.36, cubic=True),
            216, "SiC", 4, "3C silicon carbide",
            "a=4.36; cubic 3C polytype",
        ),
    ]

    structures: dict[str, Atoms] = {}
    for name, atoms, expected_sg, formula, units, prototype, note in entries:
        atoms.wrap()
        # ASE's crystal() inserts a Spacegroup Python object into info.
        # Replace it with plain computed metadata for future JSONL export.
        atoms.info.clear()
        atoms.info.update(analyze_structure(atoms))
        atoms.info.update(
            {
                "name": name,
                "base_structure": name,
                "display_formula": formula,
                "prototype": prototype,
                "expected_spacegroup_number": expected_sg,
                "model": "idealized fully occupied periodic prototype",
                "parameter_note": note,
                "bond_pairs": [pair.copy() for pair in BOND_PAIRS[name]],
                "q2_eligible": True,
                "q2_ineligible_reason": "",
            }
        )
        _check_prototype(name, atoms, expected_sg, formula, units)
        if name in structures:
            raise RuntimeError(f"Duplicate structure name: {name}")
        structures[name] = atoms
    from structure_catalog import build_extended_structures, crystal_system_name

    for atoms in structures.values():
        atoms.info["crystal_system"] = crystal_system_name(atoms.info["spacegroup_number"])
    extended = build_extended_structures(analyze_structure)
    if set(extended) & set(structures):
        raise RuntimeError("Extended prototype names collide with the original catalog")
    structures.update(extended)
    return structures


def main() -> None:
    structures = get_structures()
    print("MV-CrystalBench: idealized prototype cells; lengths in angstroms")
    print(f"spglib symmetry tolerance: {SYMPREC:g} angstrom")
    print(f"{'Name':<23} {'Formula':<10} {'Atoms':>5}  {'Space group':<17} Check")
    print("-" * 67)
    for name, atoms in structures.items():
        info = atoms.info
        group = f"{info['spacegroup_symbol']} ({info['spacegroup_number']})"
        print(
            f"{name:<23} {info['display_formula']:<10} {len(atoms):>5}  "
            f"{group:<17} OK"
        )
    print(
        f"PASS: all {len(structures)} structures have their expected space "
        "group, cell population, and composition."
    )
    print("Crystal systems:", dict(sorted(Counter(
        atoms.info["crystal_system"] for atoms in structures.values()
    ).items())))
    print("Q2 eligible:", sum(a.info["q2_eligible"] for a in structures.values()))


if __name__ == "__main__":
    main()
