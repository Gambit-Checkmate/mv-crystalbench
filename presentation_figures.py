"""Presentation figures from frozen evaluation outputs; never alters source data.

Run --figure hero first, then --figure q5 (or --figure all for that same order).
All auxiliary audit data are written beside the two presentation PNGs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
TASKS = ('Q1b', 'Q2_easy', 'Q2_hard', 'Q3', 'Q4', 'Q5')
SERIES = (('gpt', 'image'), ('gpt', 'text'), ('claude', 'image'), ('claude', 'text'))
COLORS = {'gpt': '#2563EB', 'claude': '#C56B27'}
LIGHT = {'gpt': '#DCE7FF', 'claude': '#F8E5D6'}
INK, MUTED, GRID = '#142438', '#526174', '#E4E9F0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canvas(title, subtitle, *, title_size=25):
    fig = Figure(figsize=(12.8, 7.2), dpi=150, facecolor='white')
    FigureCanvasAgg(fig)
    fig.text(.067, .965, 'MV-CRYSTALBENCH', fontsize=12, weight='bold', color=MUTED, va='top')
    fig.text(.067, .915, title, fontsize=title_size, weight='bold', color=INK, va='top')
    fig.text(.067, .850, subtitle, fontsize=14, color=MUTED, va='top')
    return fig


def style_axis(ax, *, upper=124):
    ax.set_ylim(0, upper)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='both', length=0, labelsize=15, pad=10, colors=INK)
    ax.set_ylabel('Accuracy (%)', fontsize=17, color=INK, labelpad=14)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#B7C3D1')
    ax.grid(axis='y', color=GRID, linewidth=.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(25, color='#64748B', linewidth=1.6, linestyle=(0, (5, 4)), zorder=1)


def legend(fig, *, y=.795):
    patches = [Patch(facecolor=COLORS[model] if condition == 'image' else LIGHT[model],
                     edgecolor=COLORS[model], hatch=None if condition == 'image' else '///',
                     label=f'{"GPT" if model == "gpt" else "Claude"} {condition}')
               for model, condition in SERIES]
    fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(.52, y),
               ncol=4, frameon=False, fontsize=15, handlelength=1.7, columnspacing=2.2)


def save_figure(fig, path, audit):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor='white', metadata={'Description': json.dumps(audit)})
    with Image.open(path) as image:
        if image.size != (1920, 1080):
            raise ValueError(f'Unexpected image dimensions: {image.size}')
    path.with_suffix('.data.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n', 'utf-8')
    return str(path.resolve())


def hero(out_dir=ROOT/'figures'):
    source = ROOT/'results/main_tables_ci.csv'
    with source.open(encoding='utf-8', newline='') as handle:
        all_rows = list(csv.DictReader(handle))
    rows = {}
    for raw in all_rows:
        key = (raw['model'], raw['condition'], raw['task'])
        if key[:2] not in SERIES or key[2] not in TASKS:
            continue
        if key in rows:
            raise ValueError(f'Duplicate summary identity: {key}')
        row = {**raw, **{field: float(raw[field]) for field in ('accuracy','ci_low','ci_high')},
               **{field: int(raw[field]) for field in ('n','correct','missing_count','expected_n')}}
        if row['n'] != 25 or row['missing_count'] != 0 or row['expected_n'] != 25:
            raise ValueError(f'Expected a complete paired 25-question task: {key}')
        if not 0 <= row['ci_low'] <= row['accuracy'] <= row['ci_high'] <= 1:
            raise ValueError(f'Invalid accuracy interval: {key}')
        if not np.isclose(row['accuracy'], row['correct']/row['n']):
            raise ValueError(f'Accuracy/count mismatch: {key}')
        rows[key] = row
    if len(rows) != len(SERIES)*len(TASKS):
        raise ValueError('The presentation needs all 24 image/text task summaries.')
    gaps = {model: 100*(rows[model,'image','Q5']['accuracy']-rows[model,'text','Q5']['accuracy'])
            for model in ('gpt','claude')}
    if not np.allclose([gaps['gpt'],gaps['claude']],[-36,-68]):
        raise ValueError(f'Frozen Q5 gaps changed: {gaps}')
    fig = canvas('Same question, same information: pictures vs. coordinates',
                 'Identical questions; structures supplied as rendered views or atomic coordinates.')
    legend(fig)
    ax = fig.add_axes([.085,.190,.885,.555])
    style_axis(ax)
    ax.set_xlim(-.60,5.60)
    ax.axvspan(4.49,5.51,facecolor='#F2F5FA',zorder=0)
    positions = np.arange(len(TASKS))
    offsets = [-.33,-.11,.11,.33]
    for offset, (model,condition) in zip(offsets,SERIES):
        selected = [rows[model,condition,task] for task in TASKS]
        heights = np.array([row['accuracy']*100 for row in selected])
        low = np.array([row['ci_low']*100 for row in selected])
        high = np.array([row['ci_high']*100 for row in selected])
        color = COLORS[model] if condition == 'image' else LIGHT[model]
        ax.bar(positions+offset,heights,width=.195,color=color,edgecolor=COLORS[model],
               linewidth=1.1,hatch=None if condition=='image' else '///',zorder=3)
        ax.errorbar(positions+offset,heights,yerr=[heights-low,high-heights],fmt='none',
                    ecolor='#38465A',elinewidth=1.2,capsize=3,capthick=1.2,zorder=4)
        for x,height,upper in zip(positions+offset,heights,high):
            ax.text(x,upper+2.1,f'{height:.0f}',ha='center',va='bottom',fontsize=12,
                    weight='bold',color=INK,zorder=5)
    ax.set_xticks(positions,['Q1b\nCell stoichiometry','Q2 easy\nCoordination',
                            'Q2 hard\nCoordination','Q3\nRelative distance',
                            'Q4\nCrystal system','Q5\nSame structure'])
    ax.text(5,119,'GPT  −36 pp',ha='center',va='center',color=COLORS['gpt'],fontsize=14,weight='bold')
    ax.text(5,111,'Claude  −68 pp',ha='center',va='center',color=COLORS['claude'],fontsize=14,weight='bold')
    fig.text(.085,.079,'25 paired questions per task  ·  Error bars: 95% cluster bootstrap CI  ·  Dashed line: random (25%)',
             fontsize=12.5,color=MUTED)
    fig.text(.085,.045,'Q1a and Q6 are excluded because their text-only condition is not applicable.  Q5 gaps = image − text.',
             fontsize=11.5,color=MUTED)
    audit = dict(figure='hero_image_vs_text',source=str(source.relative_to(ROOT)),source_sha256=digest(source),
                 title='Same question, same information: pictures vs. coordinates',
                 image_size=[1920,1080],q5_image_minus_text_pp=gaps,
                 rows=[rows[model,condition,task] for task in TASKS for model,condition in SERIES])
    return save_figure(fig,Path(out_dir)/'hero_image_vs_text.png',audit)


def q5_breakdown(out_dir=ROOT/'figures'):
    verified = {}
    sources = []
    cohort = None
    expected = {('gpt','image'):7,('gpt','text'):13,('claude','image'):4,('claude','text'):13}
    for model,condition in SERIES:
        source = ROOT/'results/final_run'/f'{model}_{condition}.jsonl'
        records = [json.loads(line) for line in source.read_text('utf-8').splitlines() if line.strip()]
        positive = [row for row in records if row['qtype']=='Q5'
                    and row['meta'].get('is_same_structure') is True]
        ids = {row['question_id'] for row in positive}
        if len(positive) != 13 or len(ids) != 13:
            raise ValueError(f'{model}/{condition}: need exactly 13 unique positive Q5 records.')
        if cohort is not None and ids != cohort:
            raise ValueError('Positive Q5 image/text cohorts differ.')
        cohort = ids
        if any(row['model']!=model or row['condition']!=condition or row.get('dry_run')
               or row.get('baseline') or row['meta'].get('negative_type') is not None
               or row['answer_text']!='Same structure' or row['status']!='ok' for row in positive):
            raise ValueError(f'{model}/{condition}: unexpected or non-final Q5 positive records.')
        if len({row['config_fingerprint'] for row in records}) != 1:
            raise ValueError('Mixed evaluation configurations in source JSONL.')
        correct = sum(row['correct'] is True for row in positive)
        if correct != expected[model,condition]:
            raise ValueError(f'{model}/{condition}: observed {correct}/13, expected {expected[model,condition]}/13.')
        verified[model,condition] = dict(model=model,condition=condition,n=13,correct=correct,
            accuracy=correct/13,question_ids=sorted(ids),
            correct_question_ids=sorted(row['question_id'] for row in positive if row['correct'] is True))
        sources.append(dict(path=str(source.relative_to(ROOT)),sha256=digest(source)))
    fig = canvas('Same crystal, different viewing angle: is it the same structure?',
                 'Positive Q5 pairs only: both image groups show the same structure from different viewpoints.',
                 title_size=22)
    # Conditions are explained in the footer; keep the subtitle region clear.
    ax = fig.add_axes([.085,.185,.885,.605])
    style_axis(ax,upper=110)
    ax.set_xlim(-.65,1.65)
    ax.set_xticks([0,1],['GPT','Claude'])
    ax.tick_params(axis='x',labelsize=21,pad=14)
    for index,model in enumerate(('gpt','claude')):
        for offset,condition in ((-.18,'image'),(.18,'text')):
            row = verified[model,condition]
            value = row['accuracy']*100
            x = index+offset
            ax.bar(x,value,width=.31,color=COLORS[model] if condition=='image' else LIGHT[model],
                   edgecolor=COLORS[model],linewidth=1.3,hatch=None if condition=='image' else '///',zorder=3)
            text_color = 'white' if condition=='image' else INK
            ax.text(x,value/2+4,f"{row['correct']} / 13",fontsize=29,weight='bold',color=text_color,
                    ha='center',va='center',zorder=5,
                    bbox=dict(facecolor=COLORS[model] if condition=='image' else LIGHT[model],edgecolor='none',pad=3))
            ax.text(x,value/2-6,f'{value:.1f}%',fontsize=20,color=text_color,ha='center',va='center',zorder=5,
                    bbox=dict(facecolor=COLORS[model] if condition=='image' else LIGHT[model],edgecolor='none',pad=2))
    fig.text(.085,.080,'The same 13 positive question IDs are compared in every model/input condition.',fontsize=14,color=MUTED)
    fig.text(.085,.043,'Counts verified directly from final evaluation JSONL.  Solid = image; hatched = coordinates.  Dashed line = random (25%).',
             fontsize=11.5,color=MUTED)
    audit = dict(figure='q5_breakdown',sources=sources,image_size=[1920,1080],
                 title='Same crystal, different viewing angle: is it the same structure?',
                 cohort='Q5 positives only; is_same_structure=true and negative_type=null',
                 verified_rows=[verified[key] for key in SERIES],confidence_intervals='Not plotted; raw counts shown.')
    return save_figure(fig,Path(out_dir)/'q5_breakdown.png',audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure',choices=('hero','q5','all'),default='all')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'figures')
    args = parser.parse_args()
    with matplotlib.rc_context({'font.family':'DejaVu Sans','hatch.linewidth':.75}):
        if args.figure in {'hero','all'}:
            print(hero(args.output_dir),flush=True)
        if args.figure in {'q5','all'}:
            print(q5_breakdown(args.output_dir),flush=True)


if __name__=='__main__':
    main()
