"""Periodic crystal drawings with ASE, depth cues, and visibility metadata.

Directions use the original cell's direct basis: [uvw] = u*a + v*b + w*c.
The camera sits on the positive side. PNGs contain no structure/direction names.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
import json
from numbers import Integral
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from ase.visualize.plot import plot_atoms
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.text import Text

from bonding import bond_cutoffs, display_bonds

IMAGE_SIZE = 512
DPI = 100
RADIUS_SCALE = 0.45
MAX_ATOM_DIAMETER_PX = IMAGE_SIZE / 10
HIGHLIGHT_COLOR = "#ff0099"
BOUNDARY_TOLERANCE = 1e-3
PLOT_SIZE = 472.0
PLOT_LEFT = 20.0
PLOT_BOTTOM = 32.0
STANDARD_DIRECTIONS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1))
DEMO_DIRECTIONS = ((1, 0, 0), (1, 1, 0), (1, 1, 1))
HIGH_CONTRAST_PALETTE = (
    # Every pair, including the reserved highlight, is >=40 CIE76 Delta E.
    # Purple/blue and pink/highlight from tab10 do not meet this constraint.
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#8c564b", "#bcbd22", "#17becf",
)
MIN_ELEMENT_DELTA_E = 40.0


def rgb_to_lab(rgb: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 1] to CIELAB using the D65 white point.

    Accepts one RGB triple or an (..., 3) array. Euclidean differences in the
    returned coordinates are CIE76 Delta E; no optional color package needed.
    """
    values = np.asarray(rgb, dtype=float)
    if (values.shape[-1:] != (3,) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)):
        raise ValueError("sRGB colors must be finite RGB triples in [0, 1].")
    linear = np.where(values <= 0.04045, values / 12.92,
                      ((values + 0.055) / 1.055) ** 2.4)
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = (linear @ matrix.T) / np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def _color_separation(colors: np.ndarray) -> tuple[float | None, float]:
    lab = rgb_to_lab(colors)
    pair_minimum = None
    if len(colors) > 1:
        distances = np.linalg.norm(lab[:, None, :] - lab[None, :, :], axis=-1)
        pair_minimum = float(distances[np.triu_indices(len(colors), 1)].min())
    highlight_lab = rgb_to_lab(to_rgb(HIGHLIGHT_COLOR))
    highlight_minimum = float(np.linalg.norm(lab - highlight_lab, axis=-1).min())
    return pair_minimum, highlight_minimum


def _element_palette(atoms: Atoms) -> tuple[dict[str, np.ndarray], str, float | None]:
    by_symbol = {atom.symbol: atom.number for atom in atoms}
    symbols = sorted(by_symbol)
    default = np.array([jmol_colors[by_symbol[symbol]] for symbol in symbols])
    minimum, highlight_minimum = _color_separation(default)
    # Ca/F's base CIE76 distance is ~41.72: raising the threshold alone
    # would retain the nearly identical greens. Include the renderer's
    # maximum common fog when deciding whether to abandon Jmol.
    fog_minimum, _ = _color_separation(default * 0.6 + 0.4)
    mode = "jmol"
    if ((minimum is not None and minimum < MIN_ELEMENT_DELTA_E)
            or (fog_minimum is not None and fog_minimum < MIN_ELEMENT_DELTA_E)
            or highlight_minimum < MIN_ELEMENT_DELTA_E):
        if len(symbols) > len(HIGH_CONTRAST_PALETTE):
            raise ValueError("The validated fixed palette supports at most 7 elements.")
        default = np.array([to_rgb(color) for color in HIGH_CONTRAST_PALETTE[:len(symbols)]])
        mode = "fixed_high_contrast"
    final_minimum, final_highlight = _color_separation(default)
    if ((final_minimum is not None and final_minimum < MIN_ELEMENT_DELTA_E)
            or final_highlight < MIN_ELEMENT_DELTA_E):
        raise ValueError("Element/highlight colors do not satisfy the Delta E constraint.")
    return dict(zip(symbols, default)), mode, minimum


def element_colors(atoms: Atoms) -> dict[str, np.ndarray]:
    """Jmol or a fixed palette, separated by >=40 Delta E, including highlight.

    Colors are assigned by sorted element symbols and evaluated on the original
    structure, so environments and supercells use the same element identity.
    Weak Jmol separation at 40% common depth fog also triggers fallback.
    Returned base colors are before depth fog and match the legend.
    """
    return _element_palette(atoms)[0]


def _validate_atoms(atoms: Atoms) -> None:
    if not isinstance(atoms, Atoms):
        raise TypeError("atoms must be an ase.Atoms object.")
    if len(atoms) == 0 or not np.isfinite(atoms.positions).all():
        raise ValueError("A nonempty structure with finite positions is required.")
    if not np.isfinite(atoms.cell).all() or np.linalg.matrix_rank(atoms.cell) != 3:
        raise ValueError("A finite, full-rank 3D cell is required.")
    if not np.isfinite(atoms.get_celldisp()).all():
        raise ValueError("The cell display displacement must be finite.")
    if not np.all(atoms.pbc):
        raise ValueError("Periodic crystal display requires pbc=True on all axes.")


def build_display_atoms(
    atoms: Atoms, display: str = "cell", repeat: Sequence[int] = (2, 2, 2),
    center: int | None = None, cutoff: float | None = None,
) -> tuple[Atoms, list[tuple[int, tuple[int, int, int]]]]:
    """Expand periodic images and return (display_atoms, index_mapping).

    Mapping entries are (original_index, integer_translation); display position
    = original_position + translation @ original_cell - original_celldisp.
    Ordering is original index, then lexicographic translation.

    cell: fractional coordinates in [-1e-3, 1+1e-3], including both faces.
    supercell: the same convention over [0, repeat] in the original basis.
    environment: all periodic images within cutoff angstroms of original atom
    ``center`` at translation (0,0,0); requires an explicit cutoff.
    """
    _validate_atoms(atoms)
    if display not in {"cell", "supercell", "environment"}:
        raise ValueError("display must be 'cell', 'supercell', or 'environment'.")
    cell = atoms.cell.array.copy()
    inverse = np.linalg.inv(cell)
    positions = atoms.positions - atoms.get_celldisp().reshape(3)
    fractional = positions @ inverse
    extent = np.ones(3, dtype=int)
    if display == "supercell":
        values = tuple(repeat)
        if len(values) != 3 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 1
            for v in values
        ):
            raise ValueError("repeat must contain three positive integers.")
        extent = np.asarray(values, dtype=int)
    if display == "environment":
        if (isinstance(center, (bool, np.bool_)) or not isinstance(center, Integral)
                or not 0 <= center < len(atoms)):
            raise ValueError("environment requires a valid original atom center.")
        if cutoff is None or not np.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("environment requires a finite positive cutoff.")
        # Inverse-cell column norms bound each fractional component of a ball,
        # including large shifts that cancel each other in very skew cells.
        fractional_radius = float(cutoff) * np.linalg.norm(inverse, axis=0)
    display_positions, numbers, mapping = [], [], []
    for index, frac in enumerate(fractional):
        if display == "environment":
            offset = fractional[center] - frac
            lo = np.ceil(offset - fractional_radius - 1e-10).astype(int)
            hi = np.floor(offset + fractional_radius + 1e-10).astype(int)
        else:
            lo = np.ceil(-BOUNDARY_TOLERANCE - frac).astype(int)
            hi = np.floor(extent + BOUNDARY_TOLERANCE - frac).astype(int)
        for shift in product(*(range(int(a), int(b) + 1) for a, b in zip(lo, hi))):
            position = positions[index] + np.asarray(shift) @ cell
            if display == "environment":
                if np.linalg.norm(position - positions[center]) > cutoff + 1e-10:
                    continue
            display_positions.append(position)
            numbers.append(int(atoms.numbers[index]))
            mapping.append((index, shift))
    if not mapping:
        raise ValueError("The selected display contains no atoms.")
    shown = Atoms(numbers=numbers, positions=display_positions,
                  cell=extent[:, None] * cell, pbc=False)
    shown.new_array("original_index", np.array([m[0] for m in mapping], dtype=int))
    shown.new_array("image_translation", np.array([m[1] for m in mapping], dtype=int))
    return shown, mapping


def direction_to_rotation(
    atoms: Atoms, direction: Sequence[float], tilt: float = 0.0
) -> np.ndarray:
    """ASE row-vector rotation; rotate the camera about screen x then y.

    At tilt=0 the camera +z axis is exactly [uvw]. Projected c is up when
    possible, otherwise b then a. Hexagonal directions have three indices,
    not four-index Miller-Bravais notation. Both tilt angles are in degrees.
    """
    uvw = np.asarray(direction, dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    if not np.isfinite(cell).all() or np.linalg.matrix_rank(cell) != 3:
        raise ValueError("A finite, full-rank 3D cell is required.")
    if uvw.shape != (3,) or not np.isfinite(uvw).all():
        raise ValueError("Each direction must contain three finite numbers.")
    if not np.isfinite(tilt):
        raise ValueError("tilt must be a finite angle in degrees.")
    magnitude = np.max(np.abs(uvw))
    if magnitude == 0:
        raise ValueError("The viewing direction must not be [0 0 0].")
    toward_camera = (uvw / magnitude) @ cell
    toward_camera /= np.linalg.norm(toward_camera)
    for candidate in cell[::-1]:
        candidate = candidate / np.linalg.norm(candidate)
        up = candidate - np.dot(candidate, toward_camera) * toward_camera
        if np.linalg.norm(up) > 1e-8:
            up /= np.linalg.norm(up)
            right = np.cross(up, toward_camera)
            right /= np.linalg.norm(right)
            up = np.cross(toward_camera, right)
            rotation = np.column_stack((right, up, toward_camera))
            angle = np.deg2rad(float(tilt))
            c, s = np.cos(angle), np.sin(angle)
            rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
            ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            return rotation @ rx @ ry
    raise ValueError("Could not construct a camera basis.")


def _parse_highlights(highlight, atom_count: int) -> dict[int, str]:
    if highlight is None:
        return {}
    if isinstance(highlight, Integral) and not isinstance(highlight, bool):
        highlight = {int(highlight): "A"}
    if not isinstance(highlight, Mapping):
        raise TypeError("highlight must be None, a display index, or {index: label}.")
    labels: dict[int, str] = {}
    for index, label in highlight.items():
        if isinstance(index, bool) or not isinstance(index, Integral):
            raise TypeError("Highlight display indices must be integers.")
        if not 0 <= index < atom_count:
            raise IndexError(f"Display index {index} is outside [0, {atom_count}).")
        if not isinstance(label, str) or len(label) != 1 or not "A" <= label <= "Z":
            raise ValueError("Use a single uppercase label such as A, B, or C.")
        if label in labels.values():
            raise ValueError("Highlight labels must be unique.")
        labels[int(index)] = label
    return dict(sorted(labels.items(), key=lambda item: item[1]))


def _visible_fractions(
    screen_xy_px: np.ndarray, radii_px: np.ndarray, depths: np.ndarray,
    image_size: int = IMAGE_SIZE, supersample: int = 2,
) -> np.ndarray:
    """Opaque-disc visible fractions on a 2x raster, with top-left coordinates.

    Larger depth is nearer. Equal-depth ties draw in display-index order.
    This metric excludes bonds, frames, outlines, and annotations.
    """
    xy = np.asarray(screen_xy_px, dtype=float) * supersample
    radii = np.asarray(radii_px, dtype=float) * supersample
    size = image_size * supersample
    owner = np.full((size, size), -1, dtype=np.int32)
    areas = np.zeros(len(xy), dtype=int)
    for index in np.argsort(depths, kind="stable"):
        x, y = xy[index]
        radius = radii[index]
        x0, x1 = max(0, int(np.floor(x - radius))), min(size, int(np.ceil(x + radius)))
        y0, y1 = max(0, int(np.floor(y - radius))), min(size, int(np.ceil(y + radius)))
        if x1 <= x0 or y1 <= y0:
            continue
        gx = np.arange(x0, x1) + 0.5 - x
        gy = np.arange(y0, y1) + 0.5 - y
        mask = gx[None, :] ** 2 + gy[:, None] ** 2 <= radius ** 2
        areas[index] = np.count_nonzero(mask)
        owner[y0:y1, x0:x1][mask] = index
    visible = np.bincount(owner[owner >= 0], minlength=len(xy))
    return np.divide(visible, areas, out=np.zeros(len(xy), dtype=float), where=areas > 0)


@dataclass
class _View:
    direction: np.ndarray
    tilt: float
    rotation: np.ndarray
    projected: np.ndarray
    lower: np.ndarray
    side: float
    screen_xy: np.ndarray
    radii_px: np.ndarray
    visible: np.ndarray
    depths: np.ndarray


def _prepare_views(atoms, directions, display, repeat, center, cutoff, tilt,
                   display_indices=None):
    shown, mapping = build_display_atoms(atoms, display, repeat, center, cutoff)
    if display_indices is not None:
        indices = list(display_indices)
        if not indices or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
            or not 0 <= i < len(shown) for i in indices
        ):
            raise ValueError("display_indices must contain valid display atom indices.")
        if len(indices) != len(set(indices)):
            raise ValueError("display_indices must not contain duplicate indices.")
        shown = shown[indices]
        mapping = [mapping[i] for i in indices]
    directions = [np.asarray(d, dtype=float) for d in directions]
    if not directions:
        raise ValueError("Provide at least one viewing direction.")
    if np.ndim(tilt) == 0:
        tilts = [float(tilt)] * len(directions)
    else:
        tilts = list(tilt)
        if len(tilts) != len(directions):
            raise ValueError("A tilt sequence must have one angle per direction.")
    rotations = [direction_to_rotation(atoms, d, angle) for d, angle in zip(directions, tilts)]
    vertices = np.asarray(list(product((0, 1), repeat=3))) @ shown.cell
    projected = [shown.positions @ rotation for rotation in rotations]
    vertex_views = [vertices @ rotation for rotation in rotations]
    bounds = []
    for points, corners in zip(projected, vertex_views):
        cloud = points if display == "environment" else np.vstack((points, corners))
        bounds.append((cloud[:, :2].min(0), cloud[:, :2].max(0)))
    span = max(float(np.max(hi - lo)) for lo, hi in bounds)
    nominal = RADIUS_SCALE * covalent_radii[shown.numbers]
    if span < 1e-10:
        span = float(max(nominal.max() * 8, 1.0))
    # Solve radius cap jointly with fitting; 5px free around every sphere.
    padding_px = 5.0
    cap_px = (MAX_ATOM_DIAMETER_PX - 1.2) / 2
    max_radius = min(float(nominal.max()),
                     cap_px * span / (PLOT_SIZE - 2 * padding_px - 2 * cap_px))
    radii = nominal * max_radius / nominal.max()
    side = (span + 2 * max_radius) / (1 - 2 * padding_px / PLOT_SIZE)
    pixels_per_angstrom = PLOT_SIZE / side
    views = []
    for direction, angle, rotation, points, corners in zip(
        directions, tilts, rotations, projected, vertex_views
    ):
        lo = (points[:, :2] - radii[:, None]).min(0)
        hi = (points[:, :2] + radii[:, None]).max(0)
        if display != "environment":
            lo = np.minimum(lo, corners[:, :2].min(0))
            hi = np.maximum(hi, corners[:, :2].max(0))
        lower = (lo + hi - side) / 2
        screen_xy = (points[:, :2] - lower) * pixels_per_angstrom
        screen_xy[:, 0] += PLOT_LEFT
        screen_xy[:, 1] = IMAGE_SIZE - PLOT_BOTTOM - screen_xy[:, 1]
        radii_px = radii * pixels_per_angstrom
        depths = points[:, 2] - points[:, 2].mean()
        visible = _visible_fractions(screen_xy, radii_px, depths)
        views.append(_View(direction, float(angle), rotation, points, lower, side,
                           screen_xy, radii_px, visible, depths))
    return shown, mapping, radii, views


def pick_visible_atoms(
    atoms: Atoms, directions: Iterable[Sequence[float]], k: int,
    min_visible: float = 0.7, *, display: str = "cell",
    repeat: Sequence[int] = (2, 2, 2), center: int | None = None,
    cutoff: float | None = None, tilt: float | Sequence[float] = 8.0,
    display_indices: Sequence[int] | None = None,
) -> list[int]:
    """Return up to k display indices sufficiently visible in EVERY view.

    Uses the renderer's exact expansion, fitting, radii and visibility raster.
    Rank by worst-view visibility, then index. Return fewer if too few qualify.
    Supply identical directions and display/tilt options when rendering.
    """
    if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
        raise ValueError("k must be a nonnegative integer.")
    if not np.isfinite(min_visible) or not 0 <= min_visible <= 1:
        raise ValueError("min_visible must be between 0 and 1.")
    _, _, _, views = _prepare_views(
        atoms, directions, display, repeat, center, cutoff, tilt, display_indices
    )
    worst = np.min([view.visible for view in views], axis=0)
    return [
        int(index) for index in np.argsort(-worst, kind="stable")
        if worst[index] >= min_visible
    ][:k]


def _depth_lines(ax, view: _View, lines, colors, widths, *, dashed=False, background=False):
    """Split at <=1 screen pixel; batch segments between adjacent atom depths."""
    sorted_depth = np.sort(view.projected[:, 2], kind="stable")
    groups = defaultdict(lambda: ([], [], []))
    px_scale = PLOT_SIZE / view.side
    for line, color, width in zip(lines, colors, widths):
        length_px = np.linalg.norm(line[1, :2] - line[0, :2]) * px_scale
        count = max(1, int(np.ceil(length_px)))
        points = line[0] + np.linspace(0, 1, count + 1)[:, None] * (line[1] - line[0])
        for number in range(count):
            if dashed and (number // 4) % 2:
                continue
            segment = points[number:number + 2]
            band = 0 if background else int(
                np.searchsorted(sorted_depth, segment[:, 2].mean(), side="right")
            )
            segments, band_colors, band_widths = groups[band]
            segments.append(segment[:, :2] - view.lower)
            band_colors.append(color)
            band_widths.append(width)
    for band, (segments, band_colors, band_widths) in groups.items():
        ax.add_collection(LineCollection(
            segments, colors=band_colors, linewidths=band_widths,
            capstyle="butt", antialiased=True, zorder=0 if background else 2 * band + 1,
        ))


def _draw_bonds_and_frame(ax, shown, view, radii, pairs, draw_cell):
    lines, colors, widths = [], [], []
    zmin, zmax = view.projected[:, 2].min(), view.projected[:, 2].max()
    for first, second, _ in pairs:
        p, q = view.projected[[first, second]]
        distance_xy = np.linalg.norm(q[:2] - p[:2])
        # Clip endpoint silhouettes: otherwise the bond can overwrite its own
        # farther endpoint disc. Other occluders are handled by segment depth.
        pad = 0.6 * view.side / PLOT_SIZE
        if distance_xy <= radii[first] + radii[second] + 2 * pad:
            continue
        lo = (radii[first] + pad) / distance_xy
        hi = 1 - (radii[second] + pad) / distance_xy
        line = np.array([p + lo * (q - p), p + hi * (q - p)])
        fog = 0.4 * (zmax - line[:, 2].mean()) / max(zmax - zmin, 1e-12)
        lines.append(line)
        colors.append((1 - fog) * np.array([0.48, 0.50, 0.53]) + fog)
        widths.append(1.15)
    _depth_lines(ax, view, lines, colors, widths)
    if draw_cell:
        vertices = np.asarray(list(product((0, 1), repeat=3)))
        lines = []
        for first, second in product(range(8), repeat=2):
            if first < second and np.abs(vertices[first] - vertices[second]).sum() == 1:
                lines.append((vertices[[first, second]] @ shown.cell) @ view.rotation)
        # The cell outline is a background guide, not a physical occluder.
        # In particular, it must not make corner atoms appear transparent.
        _depth_lines(ax, view, lines, [(0.62, 0.65, 0.69)] * 12, [0.6] * 12,
                     dashed=True, background=True)


def _label_positions(view: _View, labels: Mapping[int, str]) -> list[dict]:
    """Find rectangles disjoint from all atom discs and previous labels."""
    boxes, placements = [], []
    half = np.array([11.0, 12.0])
    for index, label in labels.items():
        target = view.screen_xy[index]
        candidates = []
        for distance in (view.radii_px[index] + 22, 48, 70, 96):
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                candidates.append(target + distance * np.array([np.cos(angle), np.sin(angle)]))
        candidates.extend(
            np.array([x, y]) for x in np.arange(34, 480, 20)
            for y in np.arange(22, 468, 20)
        )
        best, best_score = None, np.inf
        for point in candidates:
            low, high = point - half, point + half
            if np.any(low < [23, 10]) or np.any(high > [489, 477]):
                continue
            nearest = np.clip(view.screen_xy, low, high)
            if np.any(np.linalg.norm(view.screen_xy - nearest, axis=1) < view.radii_px + 3):
                continue
            if any(np.all(low < box[2:] + 3) and np.all(high > box[:2] - 3) for box in boxes):
                continue
            vector = point - target
            t = np.clip(((view.screen_xy - target) @ vector) / np.dot(vector, vector), 0, 1)
            distances = np.linalg.norm(view.screen_xy - (target + t[:, None] * vector), axis=1)
            crossings = np.count_nonzero((distances < view.radii_px + 2) & (np.arange(len(t)) != index))
            score = np.linalg.norm(vector) + 60 * crossings
            if score < best_score:
                best, best_score = (point, np.r_[low, high]), score
        if best is None:
            raise ValueError(f"No unobstructed location for label {label}; choose fewer highlights.")
        point, box = best
        boxes.append(box)
        placements.append({"display_index": index, "label": label,
                           "center_px": point.tolist(), "bbox_px": box.tolist()})
    return placements


def _screen_to_data(view, xy):
    xy = np.asarray(xy, dtype=float).copy()
    xy[0] -= PLOT_LEFT
    xy[1] = IMAGE_SIZE - PLOT_BOTTOM - xy[1]
    return xy * view.side / PLOT_SIZE


def _draw_view(shown, radii, view, pairs, labels, placements, path, draw_cell, palette):
    fig = Figure(figsize=(IMAGE_SIZE / DPI, IMAGE_SIZE / DPI), dpi=DPI, layout="none")
    canvas = FigureCanvasAgg(fig)
    fig.set_facecolor("white")
    ax = fig.add_axes((PLOT_LEFT / IMAGE_SIZE, PLOT_BOTTOM / IMAGE_SIZE,
                       PLOT_SIZE / IMAGE_SIZE, PLOT_SIZE / IMAGE_SIZE))
    base_colors = np.array([palette[symbol] for symbol in shown.get_chemical_symbols()])
    colors = base_colors.copy()
    for index in labels:
        colors[index] = to_rgb(HIGHLIGHT_COLOR)
    z = view.projected[:, 2]
    fog = 0.4 * (z.max() - z) / max(float(np.ptp(z)), 1e-12)
    colors = colors * (1 - fog[:, None]) + fog[:, None]
    try:
        # ASE creates every sphere; cell/bond segments use the same disc depths.
        plot_atoms(
            shown, ax=ax, rotation=view.rotation, radii=radii, colors=colors,
            show_unit_cell=0, scale=1.0,
            bbox=(*view.lower, *(view.lower + view.side)), auto_image_plane_z="legacy",
        )
        circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
        if len(circles) != len(shown):
            raise RuntimeError("ASE did not produce one sphere per display atom.")
        ranks = np.empty(len(shown), dtype=int)
        ranks[np.argsort(z, kind="stable")] = np.arange(len(shown))
        # ASE uses default argsort; match those patches, then impose stable ties.
        for circle, index in zip(circles, np.argsort(z)):
            circle.set_zorder(2 * (int(ranks[index]) + 1))
            circle.set_linewidth(0.65)
            circle.set_edgecolor(np.full(3, 0.18 * (1 - fog[index]) + fog[index]))
        _draw_bonds_and_frame(ax, shown, view, radii, pairs, draw_cell)
        ax.set_xlim(0, view.side)
        ax.set_ylim(0, view.side)
        ax.set_aspect("equal")
        ax.set_axis_off()
        for placement in placements:
            index = placement["display_index"]
            ax.annotate(
                placement["label"], xy=_screen_to_data(view, view.screen_xy[index]),
                xytext=_screen_to_data(view, placement["center_px"]), textcoords="data",
                ha="center", va="center", fontsize=10, fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec=HIGHLIGHT_COLOR),
                arrowprops=dict(arrowstyle="-", color=HIGHLIGHT_COLOR, lw=0.9,
                                shrinkA=4, shrinkB=float(view.radii_px[index] * 72 / DPI)),
                zorder=2 * len(shown) + 10, annotation_clip=False,
            )
        symbols = shown.get_chemical_symbols()
        handles = [
            Line2D([], [], marker="o", linestyle="none",
                   markerfacecolor=base_colors[symbols.index(symbol)],
                   markeredgecolor="#555555", markersize=6, label=symbol)
            for symbol in sorted(set(symbols))
        ]
        if labels:
            handles.append(Line2D([], [], marker="o", linestyle="none",
                                  markerfacecolor=HIGHLIGHT_COLOR, markeredgecolor="#555555",
                                  markersize=6, label="Highlighted"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.002),
                   ncol=len(handles), frameon=False, fontsize=8,
                   handletextpad=0.4, columnspacing=1.2)
        for text in fig.findobj(match=Text):
            text.set_usetex(False)
            text.set_fontfamily("DejaVu Sans")
        canvas.print_png(path)
    finally:
        fig.clear()


def _direction_slug(direction):
    return "_".join(f"{float(value):.6g}" for value in direction)


def render_views(
    atoms: Atoms, directions: Iterable[Sequence[float]], out_dir: str | Path,
    highlight: int | Mapping[int, str] | None = None, *, display: str = "cell",
    repeat: Sequence[int] = (2, 2, 2), center: int | None = None,
    cutoff: float | None = None, bonds: bool = True,
    tilt: float | Sequence[float] = 8.0,
    display_indices: Sequence[int] | None = None,
) -> list[Path]:
    """Save PNGs and same-stem JSONs; return ordered absolute PNG paths.

    highlight uses DISPLAY indices. build_display_atoms exposes their mapping.
    pick_visible_atoms uses identical geometry to select reliable highlights.
    The input is unchanged. All views in a call share one physical scale.
    Radius ratios follow covalent radii; a common reduction caps every outlined
    diameter at 1/10 of the image. Bonds use allowed element pairs and a cutoff
    of 1.10 times each pair's periodic minimum distance, matching coordination.
    ``tilt`` may be one angle or one angle per direction. ``display_indices``
    filters the expanded display before numbering/highlighting (for pure shells).
    """
    shown, mapping, radii, views = _prepare_views(
        atoms, directions, display, repeat, center, cutoff, tilt, display_indices
    )
    labels = _parse_highlights(highlight, len(shown))
    placements = [_label_positions(view, labels) for view in views]
    pairs = display_bonds(atoms, shown) if bonds else []
    cutoffs = bond_cutoffs(atoms) if bonds else {}
    palette, color_mode, minimum_delta_e = _element_palette(atoms)
    output = Path(out_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    if display == "supercell":
        mode_tag = "supercell_" + "x".join(map(str, repeat))
    elif display == "environment":
        mode_tag = f"environment_c{center}_r{float(cutoff):g}"
    else:
        mode_tag = "cell"
    for number, (view, annotations) in enumerate(zip(views, placements)):
        tilt_tag = "strict" if view.tilt == 0 else f"tilt_{view.tilt:g}deg"
        stem = (f"view_{number:02d}_{_direction_slug(view.direction)}_"
                f"{tilt_tag}_{mode_tag}_{'bonds' if bonds else 'no_bonds'}")
        path = output / f"{stem}.png"
        _draw_view(shown, radii, view, pairs, labels, annotations, path,
                   display != "environment", palette)
        metadata = {
            "schema_version": 3, "image_size_px": [IMAGE_SIZE, IMAGE_SIZE],
            "display_mode": display,
            "display_indices": None if display_indices is None else list(map(int, display_indices)),
            "repeat": list(map(int, repeat)) if display == "supercell" else [1, 1, 1],
            "environment_center_original_index": int(center) if display == "environment" else None,
            "cutoff_angstrom": float(cutoff) if display == "environment" else None,
            "camera": {
                "direction_uvw": view.direction.tolist(), "tilt_degrees": view.tilt,
                "tilt_order": "screen horizontal x, then updated screen vertical y",
                "rotation_columns": view.rotation.tolist(),
                "toward_camera_cartesian": view.rotation[:, 2].tolist(),
                "pixels_per_angstrom": PLOT_SIZE / view.side,
                "screen_origin": "top-left; x right, y down; pixel coordinates",
                "depth_convention": "angstrom; larger nearer; centered at displayed atom mean",
            },
            "visibility_method": {
                "method": "opaque circular discs, 2x supersampled raster",
                "occluders": "nearer atom discs only; excludes bonds, cell lines, labels, outlines",
                "equal_depth_order": "ascending display_index, larger drawn last",
            },
            "bond_rule": "allowed element pair AND 0 < distance <= 1.10 * periodic_pair_d_min",
            "bond_pair_cutoffs_angstrom": [
                {"elements": list(pair), "cutoff": value}
                for pair, value in sorted(cutoffs.items())
            ],
            "color_policy": {
                "palette": color_mode, "comparison": "CIELAB D65, CIE76",
                "threshold_delta_e": MIN_ELEMENT_DELTA_E,
                "jmol_minimum_pair_delta_e": minimum_delta_e,
                "minimum_pair_delta_e": _color_separation(np.array(list(palette.values())))[0],
                "minimum_highlight_delta_e": _color_separation(np.array(list(palette.values())))[1],
                "highlight_rgb": list(to_rgb(HIGHLIGHT_COLOR)),
                "comparison_stage": "base colors before depth-dependent white fog",
                "jmol_fallback_checks": "base element pairs, fixed highlight, and 40% common white fog",
                "element_rgb": {symbol: color.tolist() for symbol, color in palette.items()},
            },
            "bonds": [{"display_indices": [i, j], "distance_angstrom": d} for i, j, d in pairs],
            "highlight_labels": annotations,
            "atoms": [
                {
                    "display_index": index, "original_index": int(mapping[index][0]),
                    "translation": list(map(int, mapping[index][1])),
                    "element": shown[index].symbol,
                    "position_angstrom": shown.positions[index].tolist(),
                    "screen_xy_px": view.screen_xy[index].tolist(),
                    "depth": float(view.depths[index]), "radius_px": float(view.radii_px[index]),
                    "visible_fraction": float(view.visible[index]),
                    "fog_white_fraction": float(
                        0.4 * (view.depths.max() - view.depths[index])
                        / max(float(np.ptp(view.depths)), 1e-12)
                    ),
                    "highlight": labels.get(index),
                }
                for index in range(len(shown))
            ],
        }
        path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def make_contact_sheet(named_paths: Sequence[tuple[str, Path]], out_path: str | Path) -> Path:
    """Catalog-sized grid; structure names appear only on this inspection sheet."""
    if not named_paths:
        raise ValueError("At least one structure is required for the inspection sheet.")
    columns = 4 if len(named_paths) <= 20 else 6
    rows = (len(named_paths) + columns - 1) // columns
    tile, title_height = 400, 30
    fig = Figure(figsize=(columns * tile / DPI, rows * (tile + title_height) / DPI),
                 dpi=DPI, layout="none")
    canvas = FigureCanvasAgg(fig)
    fig.set_facecolor("white")
    for index, (name, path) in enumerate(named_paths):
        row, col = divmod(index, columns)
        ax = fig.add_axes((col / columns, (rows - 1 - row) / rows,
                           1 / columns, tile / (rows * (tile + title_height))))
        ax.imshow(imread(path))
        ax.set_axis_off()
        fig.text((col + 0.5) / columns, (rows - row) / rows - 10 / (rows * (tile + title_height)), name,
                 ha="center", va="top", fontsize=11, fontweight="bold",
                 fontfamily="DejaVu Sans", usetex=False, color="#263448")
    output = Path(out_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.print_png(output)
    fig.clear()
    return output


def generate_inspection(out_dir: str | Path) -> list[Path]:
    """Generate the complete catalog sheet and two complete local shells."""
    from structures import get_structures
    root = Path(out_dir).expanduser().resolve()
    structures = get_structures()
    named = []
    for name, atoms in structures.items():
        image = render_views(atoms, [(1, 1, 1)], root / "cell" / name,
                             display="cell", bonds=True, tilt=8)[0]
        named.append((name, image))
        print(f"Rendered {name}")
    output = [make_contact_sheet(named, root / "contact_sheet.png")]
    for name, cutoff, neighbor in (("NaCl", 3.11, "Cl"), ("TiO2_rutile", 2.2, "O")):
        atoms = structures[name]
        shown, mapping = build_display_atoms(atoms, "environment", center=0, cutoff=cutoff)
        center_display = mapping.index((0, (0, 0, 0)))
        direction = None
        for candidate in ((1, 1, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1), (2, 3, 4)):
            qualifying = pick_visible_atoms(
                atoms, [candidate], len(shown), min_visible=0.95,
                display="environment", center=0, cutoff=cutoff, tilt=8,
            )
            if len(qualifying) == len(shown):
                direction = candidate
                break
        if direction is None:
            raise RuntimeError(f"No inspection angle exposes all neighbors for {name}.")
        if len(shown) != 7 or shown.get_chemical_symbols().count(neighbor) != 6:
            raise RuntimeError(f"Incomplete coordination environment for {name}.")
        image = render_views(
            atoms, [direction], root / "environment" / name,
            highlight={center_display: "A"}, display="environment",
            center=0, cutoff=cutoff, bonds=True, tilt=8,
        )[0]
        destination = root / f"{name}_environment.png"
        destination.write_bytes(image.read_bytes())
        destination.with_suffix(".json").write_bytes(image.with_suffix(".json").read_bytes())
        output.append(destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).resolve().parent / "data" / "render_review")
    parser.add_argument("--inspection", action="store_true",
                        help="Generate the complete catalog sheet and two environments.")
    parser.add_argument("--highlight", action="store_true",
                        help="Demo: label an atom visible in all three views.")
    parser.add_argument("--tilt", type=float, default=8.0)
    args = parser.parse_args()
    if args.inspection:
        for path in generate_inspection(args.out_dir):
            print(path)
        return
    from structures import get_structures
    structures = get_structures()
    for name in ("NaCl", "TiO2_rutile"):
        atoms = structures[name]
        highlights = None
        if args.highlight:
            candidates = pick_visible_atoms(atoms, DEMO_DIRECTIONS, 1, tilt=args.tilt)
            if not candidates:
                raise RuntimeError(f"No atom visible in all demo views for {name}.")
            highlights = {candidates[0]: "A"}
        for path in render_views(
            atoms, DEMO_DIRECTIONS, args.out_dir / (name + ("_highlight" if args.highlight else "")),
            highlight=highlights, tilt=args.tilt,
        ):
            print(path)


if __name__ == "__main__":
    main()
