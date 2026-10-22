"""One periodic first-neighbor definition for rendering and ground truth.

An unordered element pair is allowed only if it appears in atoms.info's
``bond_pairs``. For each allowed pair its shortest nonzero distance anywhere in
the infinite periodic structure is d_min. All distances <= 1.10*d_min are bonds.
This is a geometric benchmark convention, not a bond-order or valence model.

ASE's periodic neighbor list supplies image translations even for skew cells;
there is no fixed [-1, 1]^3 image search and no minimum-image collapsing of
multiple physical neighbors. Displayed bonds use actual displayed coordinates.
"""

from __future__ import annotations

from functools import lru_cache
from numbers import Integral
from typing import Any

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.neighborlist import neighbor_list

SHELL_FACTOR = 1.10
DISTANCE_TOLERANCE = 1.0e-9
Pair = tuple[str, str]


def _allowed_pairs(atoms: Atoms) -> tuple[Pair, ...]:
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ase.Atoms object")
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("A nonempty structure with finite positions is required")
    if not np.all(atoms.pbc):
        raise ValueError("Periodic bonding requires periodicity along all axes")
    if (not np.isfinite(atoms.cell).all()
            or np.linalg.matrix_rank(atoms.cell.array) != 3):
        raise ValueError("A finite nonsingular three-dimensional cell is required")
    symbols = set(atoms.get_chemical_symbols())
    if any(symbol not in atomic_numbers or symbol == "X" for symbol in symbols):
        raise ValueError("Every atom must have a real chemical element")
    raw_pairs = atoms.info.get("bond_pairs")
    if raw_pairs is None:
        if len(symbols) != 1:
            raise ValueError("Multielement structures require explicit info['bond_pairs']")
        symbol = next(iter(symbols))
        raw_pairs = [[symbol, symbol]]
    if not isinstance(raw_pairs, (list, tuple, set)):
        raise ValueError("bond_pairs must be a collection of two-element symbol pairs")
    pairs: set[Pair] = set()
    for raw in raw_pairs:
        if (not isinstance(raw, (list, tuple)) or len(raw) != 2
                or any(not isinstance(s, str) or s not in atomic_numbers
                       or s == "X" for s in raw)):
            raise ValueError("Each bond_pairs item must contain two element symbols")
        # Deleting an atom may remove a species entirely. Such a pair has no
        # instances in the modified crystal and should not create phantom bonds.
        if all(symbol in symbols for symbol in raw):
            pairs.add(tuple(sorted(raw)))
    return tuple(sorted(pairs))


def _geometry_key(atoms: Atoms, pairs: tuple[Pair, ...]) -> tuple:
    return (
        tuple(int(v) for v in atoms.numbers),
        tuple(float(v) for v in atoms.positions.ravel()),
        tuple(float(v) for v in atoms.cell.array.ravel()),
        pairs,
    )


@lru_cache(maxsize=512)
def _periodic_graph(key: tuple) -> tuple[tuple, tuple]:
    """Cache immutable minima and directed image edges by actual geometry."""
    numbers, position_values, cell_values, pairs = key
    if not pairs:
        return (), ()
    atoms = Atoms(
        numbers=numbers, positions=np.asarray(position_values).reshape(-1, 3),
        cell=np.asarray(cell_values).reshape(3, 3), pbc=True,
    )
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_scaled_positions(wrap=True) @ atoms.cell.array
    cell_vector_bound = float(np.min(atoms.cell.lengths()))

    # Each upper bound is the length of a known real pair of periodic sites.
    # Thus a periodic search up to their maximum necessarily contains the
    # global minimum for every allowed pair, including single-site metals.
    upper_bounds = []
    for first, second in pairs:
        a = np.array([i for i, s in enumerate(symbols) if s == first])
        b = np.array([i for i, s in enumerate(symbols) if s == second])
        distances = np.linalg.norm(
            positions[a, None, :] - positions[b][None, :, :], axis=2
        )
        nonzero = distances[distances > DISTANCE_TOLERANCE]
        candidates = []
        if nonzero.size:
            candidates.append(float(nonzero.min()))
        if first == second or np.any(distances <= DISTANCE_TOLERANCE):
            candidates.append(cell_vector_bound)
        if not candidates:
            raise ValueError(f"No finite periodic distance for {first}-{second}")
        upper_bounds.append(min(candidates))
    search_radius = max(upper_bounds) * SHELL_FACTOR + 10 * DISTANCE_TOLERANCE
    indices_i, indices_j, shifts, distances = neighbor_list(
        "ijSd", atoms, search_radius, self_interaction=False,
    )
    minima = {pair: float("inf") for pair in pairs}
    pair_set = set(pairs)
    allowed_edges = []
    for i, j, shift, distance in zip(indices_i, indices_j, shifts, distances):
        pair = tuple(sorted((symbols[int(i)], symbols[int(j)])))
        if pair not in pair_set or distance <= DISTANCE_TOLERANCE:
            continue
        value = float(distance)
        minima[pair] = min(minima[pair], value)
        allowed_edges.append(
            (int(i), int(j), tuple(int(s) for s in shift), value, pair)
        )
    if any(not np.isfinite(value) for value in minima.values()):
        raise RuntimeError("Periodic search did not find every allowed element pair")
    edges = tuple(
        edge for edge in allowed_edges
        if edge[3] <= SHELL_FACTOR * minima[edge[4]] + DISTANCE_TOLERANCE
    )
    return tuple(sorted(minima.items())), edges


def bond_minima(atoms: Atoms) -> dict[Pair, float]:
    """Return global periodic d_min in angstroms for each present allowed pair."""
    pairs = _allowed_pairs(atoms)
    minima, _ = _periodic_graph(_geometry_key(atoms, pairs))
    return dict(minima)


def bond_cutoffs(atoms: Atoms) -> dict[Pair, float]:
    """Return 1.10*d_min; unordered keys have alphabetically sorted symbols."""
    return {pair: SHELL_FACTOR * value for pair, value in bond_minima(atoms).items()}


def coordination_neighbors(atoms: Atoms, center: int) -> list[dict[str, Any]]:
    """Return every bonded periodic image around an original atom index.

    Each neighbor position is atoms.positions[original_index] +
    translation @ atoms.cell. Distinct translations of the same original atom
    are distinct physical neighbors. The center's own zero image is excluded.
    """
    pairs = _allowed_pairs(atoms)
    if (isinstance(center, (bool, np.bool_)) or not isinstance(center, Integral)
            or not 0 <= center < len(atoms)):
        raise ValueError("center must be an original atom index")
    minima_items, edges = _periodic_graph(_geometry_key(atoms, pairs))
    minima = dict(minima_items)
    symbols = atoms.get_chemical_symbols()
    neighbors = [
        {
            "original_index": j,
            "translation": list(shift),
            "distance": distance,
            "element": symbols[j],
            "pair": list(pair),
            "cutoff": SHELL_FACTOR * minima[pair],
        }
        for i, j, shift, distance, pair in edges
        if i == center
    ]
    neighbors.sort(key=lambda item: (
        item["distance"], item["original_index"], item["translation"]
    ))
    return neighbors


def coordination_number(atoms: Atoms, center: int) -> int:
    """Count periodic first-shell neighbors under the shared bond definition."""
    return len(coordination_neighbors(atoms, center))


def display_bonds(
    original_atoms: Atoms, display_atoms: Atoms,
) -> list[tuple[int, int, float]]:
    """Return unique visible i<j bonds without wrapping displayed distances."""
    cutoffs = bond_cutoffs(original_atoms)
    if not isinstance(display_atoms, Atoms):
        raise TypeError("display_atoms must be an ase.Atoms object")
    if not np.isfinite(display_atoms.positions).all():
        raise ValueError("Displayed atom positions must be finite")
    symbols = display_atoms.get_chemical_symbols()
    result = []
    for i in range(len(display_atoms)):
        distances = np.linalg.norm(
            display_atoms.positions[i + 1:] - display_atoms.positions[i], axis=1
        )
        for j, distance in enumerate(distances, start=i + 1):
            pair = tuple(sorted((symbols[i], symbols[j])))
            cutoff = cutoffs.get(pair)
            if (cutoff is not None and distance > DISTANCE_TOLERANCE
                    and distance <= cutoff + DISTANCE_TOLERANCE):
                result.append((i, j, float(distance)))
    return result
