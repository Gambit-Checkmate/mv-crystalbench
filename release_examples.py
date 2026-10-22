"""Compose two documented failure examples from frozen mini/image results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from PIL import Image

ROOT = Path(__file__).resolve().parent
INK, MUTED = "#142438", "#526174"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def inputs(qid):
    manifest = ROOT / "data/questions_mini.jsonl"
    q = next(q for q in read_jsonl(manifest) if q["id"] == qid)
    paths = {m: ROOT / f"results/final_run/{m}_image.jsonl" for m in ("gpt", "claude")}
    responses = {m: next(r for r in read_jsonl(p) if r["question_id"] == qid) for m, p in paths.items()}
    assert all(r["status"] == "ok" and not r["correct"] and not r["dry_run"] for r in responses.values())
    audit = dict(question_id=qid, question=q, condition="image",
                 selection="A mini question both models answered incorrectly; illustrative, not representative.",
                 responses={m: {k: r.get(k) for k in ("prediction", "raw_response", "response_model", "correct", "status")} for m, r in responses.items()},
                 source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest, *paths.values())})
    return q, responses, audit


def canvas(title, subtitle):
    fig = Figure(figsize=(12.8, 7.2), dpi=150, facecolor="white")
    FigureCanvasAgg(fig)
    fig.text(.035, .953, title, fontsize=27, weight="bold", color=INK, va="top")
    fig.text(.035, .887, subtitle, fontsize=13, color=MUTED, va="top")
    return fig


def panel(fig, path, rect, label):
    ax = fig.add_axes(rect)
    # The source workspace has complete scenes; the release has referenced PNGs.
    source_path = ROOT / "data" / path
    if not source_path.is_file():
        source_path = ROOT / "hf_dataset" / "images" / path
    with Image.open(source_path) as source:
        ax.imshow(source.convert("RGB"))
    ax.set_axis_off()
    ax.text(.5, 1.015, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=12, color=MUTED)


def answer_box(fig, xy, width, height, heading, answer, color, fill, detail=None):
    x, y = xy
    fig.add_artist(FancyBboxPatch((x, y), width, height,
        boxstyle="round,pad=0.008,rounding_size=0.009", transform=fig.transFigure,
        linewidth=1.2, edgecolor=color, facecolor=fill))
    fig.text(x+.015, y+height-.022, heading, color=color, fontsize=12, weight="bold", va="top")
    fig.text(x+.015, y+height-.062, textwrap.fill(answer, width=31), color=INK,
             fontsize=14, weight="bold", va="top", linespacing=1.1)
    if detail:
        fig.text(x+.015, y+.022, detail, color=MUTED, fontsize=11, va="bottom")


def save(fig, name, audit):
    target = ROOT / "figures" / name
    fig.savefig(target, dpi=150, facecolor="white", metadata={"Description": json.dumps(audit)})
    with Image.open(target) as im:
        assert im.size == (1920, 1080)
    target.with_suffix(".data.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(target)


def q5():
    q, responses, audit = inputs("mvcb-q5-0046")
    assert q["qtype"] == "Q5" and q["meta"]["is_same_structure"] and len(q["image_paths"]) == 6
    fig = canvas("Same crystal, different viewpoints", "Q5 positive example · both models answered incorrectly in the image condition")
    for row, group in enumerate(("GROUP 1 · images 1–3", "GROUP 2 · images 4–6")):
        y = .495 if row == 0 else .105
        for col in range(3):
            i = row*3+col
            panel(fig, q["image_paths"][i], [.035+col*.202, y, .192, .3413], f"Group {row+1} · Image {i+1}")
    x = .665
    fig.text(x, .838, "QUESTION", fontsize=12, weight="bold", color=MUTED)
    fig.text(x, .803, textwrap.fill(q["question"], width=49), fontsize=11.5, linespacing=1.38, va="top", color=INK)
    answer_box(fig, (x, .333), .292, .12, "CORRECT ANSWER", f"{q['answer']} · {q['answer_text']}", "#087F70", "#EDF8F5")
    for y, model in ((.183, "gpt"), (.033, "claude")):
        letter = responses[model]["prediction"]
        answer_box(fig, (x, y), .292, .12, f"{model.upper()} · INCORRECT",
                   f"{letter} · {q['options'][letter]}", "#B1443A", "#FFF3F0")
    fig.text(.035, .035, f"Mini question {q['id']} · original images and complete question; model choices are unchanged.", fontsize=10.5, color=MUTED)
    save(fig, "example_q5.png", audit)


def q3():
    q, responses, audit = inputs("mvcb-q3-0030")
    assert q["qtype"] == "Q3" and q["meta"]["depth_trap"] and len(q["image_paths"]) == 3
    fig = canvas("A depth trap both models missed", "Q3 · projected distance can reverse the true three-dimensional ordering")
    fig.text(.035, .825, textwrap.fill(q["question"], width=124), color=INK, fontsize=14, va="top", linespacing=1.35)
    for i, path in enumerate(q["image_paths"]):
        panel(fig, path, [.05+i*.322, .25, .2667, .4741], f"Image {i+1} · {q['view_ids'][i].split(':')[-1]}")
    distance = f"A–C = {q['meta']['distance_AC_angstrom']:.3f} Å; A–B = {q['meta']['distance_AB_angstrom']:.3f} Å"
    answer_box(fig, (.04, .063), .285, .145, "CORRECT ANSWER", f"{q['answer']} · {q['answer_text']}", "#087F70", "#EDF8F5", distance)
    for x, model in ((.362, "gpt"), (.684, "claude")):
        letter = responses[model]["prediction"]
        answer_box(fig, (x, .063), .285, .145, f"{model.upper()} · INCORRECT", f"{letter} · {q['options'][letter]}", "#B1443A", "#FFF3F0", "Recorded image-condition answer")
    fig.text(.035, .022, f"Mini question {q['id']} · depth_trap=true in v00 (also v01) · illustrative failure; all three images were supplied.", fontsize=10.5, color=MUTED)
    save(fig, "example_q3.png", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("q5", "q3", "all"), default="all")
    args = parser.parse_args()
    if args.figure in ("q5", "all"):
        q5()
    if args.figure in ("q3", "all"):
        q3()
