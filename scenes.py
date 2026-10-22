"""Reusable anonymous, eight-view scenes and a local question review page.

Only scene manifests contain internal structure identifiers. PNG paths contain
opaque scene hashes and view ordinals, including for strict-axis questions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import html
import json
from numbers import Integral
from pathlib import Path
import random
from typing import Any

import numpy as np
from ase import Atoms

from render import render_views


SCENE_SCHEMA_VERSION = 1
RENDER_CACHE_VERSION = "bond-shell-palette-8view-1"
DEFAULT_SEED = 20261021
DEFAULT_VIEW_IDS = ("v00", "v01", "v02")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(item) for item in value]
    return value


def _dump_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _highlight_dict(highlight: int | Mapping[int, str] | None) -> dict[int, str]:
    if highlight is None:
        return {}
    if isinstance(highlight, Integral) and not isinstance(highlight, bool):
        return {int(highlight): "A"}
    if not isinstance(highlight, Mapping):
        raise TypeError("highlight must be None, a display index, or {index: label}.")
    return {int(index): label for index, label in highlight.items()}


def _atoms_record(atoms: Atoms) -> dict[str, Any]:
    """Persist enough geometry to reconstruct scenes without private names."""
    bond_pairs = sorted(
        sorted(str(symbol) for symbol in pair)
        for pair in atoms.info.get("bond_pairs", [])
    )
    return {
        "lattice_vectors_angstrom": atoms.cell.array.tolist(),
        "positions_angstrom": atoms.positions.tolist(),
        "atomic_numbers": atoms.numbers.tolist(),
        "periodic": atoms.pbc.tolist(),
        "cell_display_displacement_angstrom": atoms.get_celldisp().reshape(3).tolist(),
        "bond_pairs": bond_pairs,
    }


def structure_text(atoms: Atoms, display_atoms: Sequence[Mapping[str, Any]]) -> str:
    """Geometry-only text using displayed positions in the ORIGINAL cell basis.

    Unwrapped coordinates preserve displayed boundary images and supercells.
    Unlike a CIF, this list intentionally includes all displayed periodic copies.
    """
    a, b, c, alpha, beta, gamma = atoms.cell.cellpar()
    lines = [
        "Lattice parameters (lengths in angstrom, angles in degrees):",
        f"a={a:.9f}, b={b:.9f}, c={c:.9f}, "
        f"alpha={alpha:.9f}, beta={beta:.9f}, gamma={gamma:.9f}",
        "Lattice vectors (rows a, b, c; angstrom):",
    ]
    lines.extend(" ".join(f"{number:.9f}" for number in row) for row in atoms.cell.array)
    lines.extend([
        "Displayed atoms: index element fractional_x fractional_y fractional_z label",
        "Fractional coordinates use the lattice vectors above; periodic copies "
        "are listed separately and coordinates are not wrapped.",
    ])
    inverse_cell = np.linalg.inv(atoms.cell.array)
    for entry in display_atoms:
        fractional = np.asarray(entry["position_angstrom"], dtype=float) @ inverse_cell
        label = f" [{entry['highlight']}]" if entry.get("highlight") else ""
        lines.append(
            f"{int(entry['display_index'])} {entry['element']} "
            + " ".join(f"{number:.9f}" for number in fractional)
            + label
        )
    return "\n".join(lines)


class SceneStore:
    """Cache all eight images of each geometry/display/highlight scene.

    ``root`` is the generated data directory. View paths are POSIX-style paths
    relative to this directory, so review.html and questions.jsonl are portable.
    The random directions are deterministic three-index crystal directions.
    """

    def __init__(self, root: str | Path, seed: int = DEFAULT_SEED):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        random_directions = rng.normal(size=(5, 3))
        random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
        self.directions = [
            [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0],
            *random_directions.tolist(),
        ]
        self.tilts = [8.0, 8.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.scenes: dict[str, dict[str, Any]] = {}
        # Changes to rendering or chemistry must never silently reuse old PNGs.
        source_hash = hashlib.sha256(RENDER_CACHE_VERSION.encode())
        for filename in ("render.py", "bonding.py"):
            source = Path(__file__).with_name(filename)
            if source.exists():
                source_hash.update(source.read_bytes())
        self.renderer_revision = source_hash.hexdigest()

    def _view(self, path: Path, scene_dir: Path, view_id: str) -> dict[str, Any]:
        """Consume renderer outputs and replace filenames with anonymous IDs."""
        metadata_path = path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        final_path = scene_dir / f"{view_id}.png"
        final_metadata_path = scene_dir / f"{view_id}.json"
        if path != final_path:
            path.replace(final_path)
        if metadata_path != final_metadata_path:
            metadata_path.replace(final_metadata_path)
        return {
            "view_id": view_id,
            "image_path": final_path.relative_to(self.root).as_posix(),
            "metadata_path": final_metadata_path.relative_to(self.root).as_posix(),
            "direction": metadata["camera"]["direction_uvw"],
            "tilt": metadata["camera"]["tilt_degrees"],
            "metadata": metadata,
        }

    def _valid_cached_scene(self, scene: dict[str, Any], cache_key: str) -> bool:
        if (scene.get("cache_key") != cache_key
                or len(scene.get("views", [])) != 8
                or scene.get("schema_version") != SCENE_SCHEMA_VERSION):
            return False
        return all(
            (self.root / view["image_path"]).is_file()
            and (self.root / view["metadata_path"]).is_file()
            for view in scene["views"] + scene.get("strict_views", [])
        )

    def _save_scene(self, scene: dict[str, Any]) -> None:
        path = self.root / "scenes" / scene["scene_id"] / "scene.json"
        _dump_json(path, scene)

    def get(
        self, atoms: Atoms, structure_key: str, display: str = "cell",
        highlight: int | Mapping[int, str] | None = None,
        repeat: Sequence[int] = (2, 2, 2), center: int | None = None,
        cutoff: float | None = None, display_indices: Sequence[int] | None = None,
        variant_key: str = "",
    ) -> dict[str, Any]:
        """Return a complete scene; new scenes render eight PNG/JSON pairs."""
        geometry = _atoms_record(atoms)
        highlights = _highlight_dict(highlight)
        parameters = {
            "display": display, "repeat": [int(number) for number in repeat],
            "center": None if center is None else int(center),
            "cutoff": None if cutoff is None else float(cutoff),
            "display_indices": (
                None if display_indices is None else [int(index) for index in display_indices]
            ),
            "highlight": highlights,
        }
        identity = {
            "schema_version": SCENE_SCHEMA_VERSION,
            "renderer_revision": self.renderer_revision,
            "geometry": geometry, "parameters": parameters,
            "directions": self.directions, "tilts": self.tilts,
            "seed": self.seed, "variant_key": str(variant_key),
        }
        cache_key = hashlib.sha256(json.dumps(
            _jsonable(identity), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()).hexdigest()
        scene_id = "s" + cache_key[:20]
        if scene_id in self.scenes:
            return self.scenes[scene_id]
        scene_dir = self.root / "scenes" / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        cache_path = scene_dir / "scene.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cached = {}
            if self._valid_cached_scene(cached, cache_key):
                self.scenes[scene_id] = cached
                return cached
        paths = render_views(
            atoms, self.directions, scene_dir, highlight=highlights,
            display=display, repeat=repeat, center=center, cutoff=cutoff,
            bonds=True, tilt=self.tilts, display_indices=display_indices,
        )
        views = [
            self._view(path, scene_dir, f"v{index:02d}")
            for index, path in enumerate(paths)
        ]
        if len(views) != 8:
            raise RuntimeError("A scene must have exactly eight default views.")
        # Spatial properties are view independent; projections are held in views.
        display_atoms = [{
            key: atom[key] for key in (
                "display_index", "original_index", "translation",
                "element", "position_angstrom", "highlight",
            )
        } for atom in views[0]["metadata"]["atoms"]]
        scene = {
            "schema_version": SCENE_SCHEMA_VERSION, "scene_id": scene_id,
            "cache_key": cache_key, "renderer_revision": self.renderer_revision,
            "seed": self.seed,
            "random_direction_method": "five normalized Gaussian three-index directions",
            "structure": str(structure_key), "variant_key": str(variant_key),
            **parameters,
            "atoms": geometry, "display_atoms": display_atoms,
            "views": views, "strict_views": [],
            "default_view_ids": list(DEFAULT_VIEW_IDS),
            "text_repr": structure_text(atoms, display_atoms),
        }
        self.scenes[scene_id] = scene
        self._save_scene(scene)
        return scene

    def ensure_strict(
        self, scene: dict[str, Any], atoms: Atoms,
        directions: Sequence[Sequence[float]],
    ) -> list[dict[str, Any]]:
        """Return separately cached, anonymous, tilt=0 views of this scene.

        Strict views live in scene['strict_views']; the eight ablation views stay
        unchanged. All directions requested in the same call share a scale.
        """
        if scene["scene_id"] not in self.scenes:
            raise ValueError("The scene must be obtained from this SceneStore.")
        if _atoms_record(atoms) != scene["atoms"]:
            raise ValueError("Strict views must use the same source structure as the scene.")
        requested = [np.asarray(direction, dtype=float).tolist() for direction in directions]
        strict = scene.setdefault("strict_views", [])
        existing = {tuple(view["direction"]): view for view in strict}
        missing = list(dict.fromkeys(
            tuple(direction) for direction in requested if tuple(direction) not in existing
        ))
        if missing:
            scene_dir = self.root / "scenes" / scene["scene_id"]
            paths = render_views(
                atoms, missing, scene_dir,
                highlight={int(index): label for index, label in scene["highlight"].items()},
                display=scene["display"], repeat=scene["repeat"],
                center=scene["center"], cutoff=scene["cutoff"], bonds=True, tilt=0.0,
                display_indices=scene["display_indices"],
            )
            for path in paths:
                view = self._view(path, scene_dir, f"vS{len(strict):02d}")
                strict.append(view)
                existing[tuple(view["direction"])] = view
            self._save_scene(scene)
        return [existing[tuple(direction)] for direction in requested]

    def save_manifest(self) -> Path:
        """Persist only scenes used by this run, ordered by scene ID."""
        path = self.root / "scenes.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for scene_id in sorted(self.scenes):
                handle.write(json.dumps(
                    _jsonable(self.scenes[scene_id]), ensure_ascii=False,
                    separators=(",", ":"), allow_nan=False,
                ) + "\n")
        return path


def write_review(
    questions: Sequence[Mapping[str, Any]], root: str | Path,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Sample three questions per type (and each Q2 difficulty) into local HTML."""
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for question in questions:
        qtype = str(question["qtype"])
        if qtype == "Q2":
            qtype += " / " + str(question.get("difficulty", "unspecified"))
        groups[qtype].append(question)
    rng = random.Random(int(seed))
    cards = []
    links = []
    for number, (kind, members) in enumerate(sorted(groups.items())):
        anchor = f"section-{number}"
        links.append(f'<a href="#{anchor}">{html.escape(kind)}</a>')
        cards.append(f'<h2 id="{anchor}">{html.escape(kind)} '
                     f'<small>（共 {len(members)} 题，抽查 {min(3, len(members))} 题）</small></h2>')
        for question in rng.sample(members, min(3, len(members))):
            figures = {}
            view_ids = question.get("view_ids", [])
            for index, image_path in enumerate(question.get("image_paths", [])):
                # Parent output uses POSIX paths relative to data/, same as HTML.
                caption = str(view_ids[index]) if index < len(view_ids) else f"image {index + 1}"
                figures[index] = (
                    '<figure><a href="' + html.escape(str(image_path), quote=True)
                    + '"><img loading="lazy" width="512" height="512" src="'
                    + html.escape(str(image_path), quote=True)
                    + '" alt="' + html.escape(caption, quote=True)
                    + '"></a><figcaption>' + html.escape(caption)
                    + "</figcaption></figure>"
                )
            image_groups = question.get("meta", {}).get("view_groups", [])
            if image_groups:
                used = set()
                image_html = []
                for group in image_groups:
                    indices = [int(index) for index in group.get("image_indices", [])]
                    if len(set(indices)) != len(indices) or any(
                        index not in figures or index in used for index in indices
                    ):
                        raise ValueError(f"Invalid review view_groups for {question['id']}.")
                    used.update(indices)
                    image_html.append(
                        '<section class="view-group"><h3>'
                        + html.escape(str(group.get("label", "")))
                        + '</h3><div class="images">'
                        + "".join(figures[index] for index in indices)
                        + "</div></section>"
                    )
                if used != set(figures):
                    raise ValueError(f"Incomplete review view_groups for {question['id']}.")
                image_html = '<div class="view-groups">' + "".join(image_html) + "</div>"
            else:
                image_html = '<div class="images">' + "".join(figures.values()) + "</div>"
            options = question["options"]
            if isinstance(options, Mapping):
                option_items = options.items()
            else:
                option_items = zip("ABCD", options)
            option_html = "".join(
                '<li class="' + ("correct" if letter == question["answer"] else "")
                + '"><b>' + html.escape(str(letter)) + ".</b> "
                + html.escape(str(text)) + "</li>"
                for letter, text in option_items
            )
            meta = html.escape(json.dumps(
                _jsonable(question.get("meta", {})),
                ensure_ascii=False, indent=2, allow_nan=False,
            ))
            cards.append(
                '<article><header><code>' + html.escape(str(question["id"]))
                + "</code><span>" + html.escape(str(question.get("structure", "")))
                + " · " + html.escape(str(question.get("difficulty", "")))
                + '</span></header>' + image_html
                + '<p class="question">' + html.escape(str(question["question"]))
                + '</p><ol class="options">' + option_html
                + '</ol><p class="answer">正确答案：'
                + html.escape(str(question["answer"])) + " — "
                + html.escape(str(question["answer_text"]))
                + '</p><details open><summary>meta（内部检查字段）</summary><pre>'
                + meta + "</pre></details>"
                + "<details><summary>纯文本表示</summary><pre>"
                + html.escape(str(question.get("text_repr", "")))
                + "</pre></details></article>"
            )
    document = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MV-CrystalBench 人工检查</title>
<style>
*{box-sizing:border-box}body{font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif;
background:#f3f5f8;color:#172232;margin:0}main{max-width:1500px;margin:auto;padding:24px}
h1{margin-bottom:4px}h2{margin-top:38px}h2 small{font-size:14px;font-weight:normal}
nav{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}nav a{background:white;padding:6px 14px;
border:1px solid #c7d0dd;border-radius:6px;color:#245aa1;text-decoration:none}
article{background:white;border:1px solid #d8dfe8;border-radius:9px;padding:20px;margin:18px 0}
header{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#576578}
header code{color:#183c6e;font-weight:bold}.images{display:flex;flex-wrap:wrap;gap:8px;margin-top:14px}
.view-groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:18px}
.view-group{border:1px solid #d8dfe8;border-radius:6px;padding:12px;margin-top:12px}
.view-group h3{margin:0}.view-group .images{margin-top:8px}
.view-group figure{flex-basis:130px}
figure{margin:0;flex:1 1 230px;max-width:420px}figure img{display:block;width:100%;height:auto;
border:1px solid #e4e8ed}figcaption{font-size:12px;text-align:center;color:#637183;overflow-wrap:anywhere}
.question{font-size:18px;font-weight:600;white-space:pre-wrap}.options{padding:0;list-style:none;
display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:8px}
.options li{border:1px solid #d9dee7;border-radius:5px;padding:8px 12px}
.options .correct{border-color:#438d69;background:#eff9f3}.answer{font-weight:600;color:#216c45}
details{margin-top:12px}summary{cursor:pointer}pre{background:#f6f8fa;padding:12px;overflow:auto;
max-height:480px;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.5 ui-monospace,monospace}
.note{color:#536174}
</style></head><body><main>
<h1>MV-CrystalBench 人工检查</h1>
<p class="note">此页包含内部结构名、标准答案和元数据，仅供人工审核，不作为模型输入。
每题型固定随机抽取 3 题，配位数 easy / hard 分别抽取；点击图片可查看原图。</p>
<nav>""" + "".join(links) + "</nav>" + "\n".join(cards) + """
</main></body></html>
"""
    path = root / "review.html"
    path.write_text(document, encoding="utf-8")
    return path
