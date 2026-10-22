"""Non-cubic prototypes reconstructed from pinned, bundled AFLOW source CIFs.

Source: https://github.com/aflow-org/aflow_prototype_encyclopedia
Commit: fb7421bb2522888b6a0a4ff91e124c6c65191096
The original CIF, source-paper BibTeX, upstream metadata, and Apache-2.0 license
are bundled in data/structure_sources/aflow. Nothing is downloaded at runtime.
No pymatgen code or dependency is used.

ASE ``spacegroup.crystal`` expands the published asymmetric sites using the
actual CIF symmetry operations (including its origin choice and centering).
spglib then chooses a conventional cell and idealizes only within 1e-5 A.
All models are fully occupied, static, nonmagnetic prototypes: source
temperature/pressure and displacement parameters are not simulation conditions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import hashlib
import json
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.io.cif import parse_cif
from ase.spacegroup import crystal


SOURCE_ROOT = Path(__file__).resolve().parent / "data" / "structure_sources" / "aflow"
SOURCE_COMMIT = "fb7421bb2522888b6a0a4ff91e124c6c65191096"
SOURCE_REPOSITORY = "https://github.com/aflow-org/aflow_prototype_encyclopedia"

# These are explicitly checked short-shell coordination numbers, under the
# benchmark's allowed-pair and 1.10*d_min definition, at every matching site.
# They describe the chosen bond graph; they are not oxidation states.
EXPECTED_COORDINATION = {
    "CuO_tenorite": {"Cu": [4], "O": [4]},
    "SiO2_coesite": {"Si": [4], "O": [2]},
    "As2S3_orpiment": {"As": [3], "S": [2]},
    "CuCl2_tolbachite": {"Cu": [4], "Cl": [2]},
    "P_black": {"P": [3]},
    "Fe3C_cementite": {"C": [6], "Fe": [2]},
    "CaCO3_aragonite": {"Ca": [9], "C": [3], "O": [4]},
    "Mg2SiO4_forsterite": {"Mg": [6], "Si": [4], "O": [4]},
    "FeS2_marcasite": {"Fe": [6], "S": [4]},
    "SnS_alpha": {"Sn": [3], "S": [3]},
    "SiO2_alpha_quartz": {"Si": [4], "O": [2]},
    "CaCO3_calcite": {"Ca": [6], "C": [3], "O": [3]},
    "As_A7": {"As": [3]},
    "Se_trigonal": {"Se": [2]},
    "PbO_litharge": {"Pb": [4], "O": [4]},
    "CuFeS2_chalcopyrite": {"Cu": [4], "Fe": [4], "S": [4]},
    "WC_hexagonal": {"W": [6], "C": [6]},
    "MoS2_2H": {"Mo": [6], "S": [3]},
    "BN_hexagonal": {"B": [3], "N": [3]},
}

Q2_EXCLUSIONS = {
    "ZrO2_baddeleyite": (
        "The global 1.10*d_min Zr-O cutoff gives Zr=6 and truncates the "
        "distorted sevenfold Zr coordination; excluded from Q2."
    ),
    "Ga2O3_beta": (
        "The global Ga-O minimum belongs to the tetrahedral site. Its cutoff "
        "truncates the octahedral Ga site to four neighbors; excluded from Q2."
    ),
    "Ag2S_acanthite": (
        "Inequivalent Ag sites have irregular coordination and different "
        "short-distance shells; a global pair cutoff is not a reliable "
        "conventional coordination label."
    ),
    "NiTi_B19prime": (
        "Distorted metallic coordination spans a broad Ni-Ti distance range; "
        "the pair cutoff returns seven neighbors and is not used as a "
        "conventional coordination label."
    ),
    "PbCl2_cotunnite": (
        "The global 1.10*d_min Pb-Cl cutoff gives Pb=7 and truncates the "
        "broad ninefold Pb coordination shell; excluded from Q2."
    ),
    "U_alpha": (
        "Strongly anisotropic metallic shells make a unique conventional "
        "coordination number inappropriate for this short-distance rule."
    ),
    "LiNbO3_ferroelectric": (
        "The LiO6 environment is split into short and long triplets. "
        "The pair cutoff keeps only three Li-O neighbors; excluded from Q2."
    ),
    "BaTiO3_tetragonal": (
        "Polar Ti displacement splits the TiO6 shell; the 1.10*d_min cutoff "
        "keeps five Ti-O neighbors. The rule is preserved and Q2 is excluded."
    ),
    "AlB2": (
        "B-B and Al-B graphs combine covalent sheets with metallic interlayer "
        "neighbors. No single conventional coordination label is assigned."
    ),
}

COORDINATION_NOTES = {
    "CuCl2_tolbachite": (
        "Count the four short Cu-Cl neighbors; the two long Jahn-Teller "
        "contacts are outside the first short-distance shell."
    ),
    "SnS_alpha": "Count the three short Sn-S neighbors; longer secondary contacts are excluded.",
    "As_A7": "Count the three short As-As neighbors; the longer second triplet is excluded.",
    "Fe3C_cementite": (
        "Only Fe-C pairs are drawn: C has six Fe neighbors and Fe has two C "
        "neighbors. The metallic Fe-Fe network is outside this bond definition."
    ),
    "CaCO3_aragonite": "The Ca-O shell is ninefold, while each carbonate C has three O neighbors.",
    "CaCO3_calcite": "The Ca-O shell is sixfold, while each carbonate C has three O neighbors.",
}


def crystal_system_name(number: int) -> str:
    for upper, name in (
        (2, "triclinic"), (15, "monoclinic"), (74, "orthorhombic"),
        (142, "tetragonal"), (167, "trigonal"), (194, "hexagonal"), (230, "cubic"),
    ):
        if 1 <= number <= upper:
            return name
    raise ValueError("Space-group number must be in [1, 230]")


def build_extended_structures(analyze: Callable) -> dict[str, Atoms]:
    """Return 28 fresh structures, checking source composition and space group."""
    selection = json.loads((SOURCE_ROOT / "selection.json").read_text("utf-8"))
    if selection["commit"] != SOURCE_COMMIT:
        raise RuntimeError("The structural source manifest is not the pinned revision")
    bond_pairs = json.loads((SOURCE_ROOT / "bond_pairs.json").read_text("utf-8"))
    result = {}
    for name, label in selection["selected"].items():
        source_dir = SOURCE_ROOT / label
        for filename, expected_hash in selection["source_sha256"][label].items():
            actual_hash = hashlib.sha256((source_dir / filename).read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(f"{name}: bundled {filename} differs from its pinned hash")
        source = json.loads((source_dir / "info.json").read_text("utf-8"))
        cif_path = source_dir / "aflow.cif"
        block = next(parse_cif(str(cif_path)))
        symbols = block.get_symbols()
        occupancies = np.asarray(block["_atom_site_occupancy"], dtype=float)
        if not np.all(occupancies == 1):
            raise RuntimeError(f"{name}: partial occupancy is not supported")
        basis = np.column_stack([block["_atom_site_fract_" + axis] for axis in "xyz"])
        raw = crystal(
            symbols, basis=basis, cell=block.get_cell(),
            spacegroup=block.get_spacegroup(subtrans_included=True),
            pbc=True, primitive_cell=False, onduplicates="error",
        )
        standardized = spglib.standardize_cell(
            (raw.cell.array, raw.get_scaled_positions(), raw.numbers),
            to_primitive=False, no_idealize=False, symprec=1e-5,
        )
        if standardized is None:
            raise RuntimeError(f"{name}: cannot standardize source prototype")
        lattice, fractional, numbers = standardized
        atoms = Atoms(numbers=numbers, scaled_positions=fractional, cell=lattice, pbc=True)
        atoms.wrap()
        source_counts = Counter()
        for symbol, multiplicity in zip(symbols, block["_atom_site_symmetry_multiplicity"]):
            source_counts[symbol] += int(multiplicity)
        if Counter(atoms.get_chemical_symbols()) != source_counts:
            raise RuntimeError(f"{name}: expanded composition disagrees with source multiplicities")
        atoms.info.update(analyze(atoms))
        expected_sg = int(source["space_group_number"])
        if atoms.info["spacegroup_number"] != expected_sg:
            raise RuntimeError(
                f"{name}: source SG {expected_sg}, detected {atoms.info['spacegroup_number']}"
            )
        eligible = name in EXPECTED_COORDINATION
        if not eligible and name not in Q2_EXCLUSIONS:
            raise RuntimeError(f"{name}: coordination policy was not reviewed")
        atoms.info.update({
            "name": name, "base_structure": name,
            "display_formula": atoms.get_chemical_formula(empirical=True),
            "prototype": source["title"], "expected_spacegroup_number": expected_sg,
            "crystal_system": crystal_system_name(expected_sg),
            "bond_pairs": bond_pairs[name],
            "q2_eligible": eligible,
            "q2_ineligible_reason": Q2_EXCLUSIONS.get(name, ""),
            "coordination_expected": EXPECTED_COORDINATION.get(name, {}),
            "coordination_note": COORDINATION_NOTES.get(name, ""),
            "model": "fully occupied static AFLOW prototype; conventional nonmagnetic idealization",
            "parameter_note": (
                "Published source coordinates expanded with ASE crystal; "
                "spglib conventional standardization/idealization at 1e-5 angstrom. "
                "Source cell basis may differ; displacement parameters are omitted."
            ),
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "source_prototype_label": label,
            "source_url": "https://aflow.org/p/" + label,
            "source_cif_url": (
                "https://raw.githubusercontent.com/aflow-org/aflow_prototype_encyclopedia/"
                + SOURCE_COMMIT + "/data/" + label + "/aflow.cif"
            ),
            "source_cif_sha256": hashlib.sha256(cif_path.read_bytes()).hexdigest(),
            "source_citation_bibtex": (source_dir / "base.bib").read_text("utf-8"),
            "source_notes": source.get("comments", ""),
            "source_license": "Apache-2.0; bundled original LICENSE",
        })
        result[name] = atoms
    return result
