"""Prepare a balanced 24-question pilot; report usage and transparent cost estimates.

This script never calls an API. Run evaluate.py on its prepared question file.
Only image usage is observed; other conditions use explicit local text estimates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics

import numpy as np

from evaluate import build_prompt, load_questions

STRATA = ('Q1a', 'Q1b', 'Q2/easy', 'Q2/hard', 'Q3', 'Q4', 'Q5', 'Q6')


def _nonnegative_number(value, name, *, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
            or (integer and int(value) != value)):
        raise ValueError(f'{name} must be a finite nonnegative ' + ('integer.' if integer else 'number.'))
    return int(value) if integer else float(value)


def calibrate_overhead(observations, chars_per_token=3.0):
    """Fit nonnegative overhead + per-image cost to image-only usage.

    Each observation has prompt_tokens, prompt_characters and image_count.
    This is an empirical planning model, not evidence of a gateway prefix or
    the tariff of a text-only request. Image count is confounded with task type.
    """
    chars_per_token = _nonnegative_number(chars_per_token, 'chars_per_token')
    if chars_per_token == 0 or not observations:
        raise ValueError('Calibration needs observations and a positive characters/token ratio.')
    tokens, characters, images = [], [], []
    for row in observations:
        tokens.append(_nonnegative_number(row['prompt_tokens'], 'prompt_tokens', integer=True))
        characters.append(_nonnegative_number(row['prompt_characters'], 'prompt_characters', integer=True))
        images.append(_nonnegative_number(row['image_count'], 'image_count', integer=True))
        if images[-1] == 0:
            raise ValueError('Image calibration requires at least one image per observation.')
    x = np.asarray(images, dtype=float)
    y = np.asarray(tokens, dtype=float) - np.asarray(characters) / chars_per_token
    identifiable = len(observations) >= 2 and len(set(images)) >= 2
    if identifiable:
        # A convex two-parameter NNLS optimum lies in the nonnegative interior,
        # on one of its axes, or at the origin. Clipping unconstrained OLS is
        # generally NOT the constrained optimum, so compare all four cases.
        intercept, per_image = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        candidates = [(0., 0.), (max(0., float(y.mean())), 0.),
                      (0., max(0., float(np.dot(x, y) / np.dot(x, x))))]
        if intercept >= 0 and per_image >= 0:
            candidates.append((float(intercept), float(per_image)))
        intercept, per_image = min(candidates, key=lambda ab: float(np.sum((y - ab[0] - ab[1] * x) ** 2)))
        method = 'nonnegative_least_squares'
    else:
        # Image cost and overhead cannot be separated with one image count.
        # Retain all observed input as a deliberately conservative proxy.
        intercept, per_image = float(max(tokens)), 0.
        method = 'unidentified_max_observed_input_fallback'
    return dict(chars_per_token=chars_per_token, observations=len(observations),
                image_counts=sorted(set(images)), identifiable=identifiable, method=method,
                overhead_tokens=intercept, per_image_tokens=per_image,
                rmse_tokens=float(np.sqrt(np.mean((y - intercept - per_image * x) ** 2))))


def stratum(question):
    return 'Q2/' + question['difficulty'] if question['qtype'] == 'Q2' else question['qtype']


def select_pilot(questions, seed=20261021):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for q in questions:
        groups[stratum(q)].append(q)
    chosen = []
    paired = None
    for group in STRATA:
        pool = sorted(groups[group], key=lambda q: q['id'])
        rng.shuffle(pool)
        if group == 'Q1b':
            rows = [q for q in pool if q['meta']['paired_count_id'] in paired]
        else:
            rows, seen_bases, seen_classes = [], set(), set()
            while len(rows) < 3:
                available = [q for q in pool if q not in rows]
                if not available:
                    raise ValueError('Insufficient pilot candidates: ' + group)
                def category(q):
                    if group == 'Q3':
                        return q['meta']['depth_trap']
                    if group == 'Q4':
                        return q['answer_text']
                    if group == 'Q5':
                        return q['meta'].get('negative_type') or 'positive'
                    return ''
                if group == 'Q5' and not rows:
                    available = [q for q in available if q['meta'].get('negative_type') is None]
                q = min(available, key=lambda q: (
                    q['meta']['base_structure'] in seen_bases,
                    category(q) in seen_classes,
                ))
                rows.append(q)
                seen_bases.add(q['meta']['base_structure'])
                seen_classes.add(category(q))
        if len(rows) != 3:
            raise ValueError('Pilot requires exactly three questions per stratum.')
        if group == 'Q1a':
            paired = {q['meta']['paired_count_id'] for q in rows}
        chosen.extend(rows)
    return chosen


def prepare(mini_path, output, seed=20261021):
    questions = select_pilot(load_questions(mini_path), seed)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(q, ensure_ascii=False) + '\n' for q in questions), 'utf-8')
    manifest = dict(seed=seed, source=str(mini_path), count=len(questions),
                    strata=dict(Counter(stratum(q) for q in questions)),
                    question_ids=[q['id'] for q in questions],
                    image_count=sum(len(q['image_paths']) for q in questions))
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), 'utf-8')
    return manifest


def report(result_path, mini_path, *, input_rate=2.0, output_rate=10.0,
           price_source='https://nexfaro.com/api/v1/model-plaza',
           price_basis='Nexfaro public base tariff matches OpenAI Standard; account group multiplier unverified',
           output_dir='results/pilot', max_output_tokens=1024):
    input_rate = _nonnegative_number(input_rate, 'input_rate')
    output_rate = _nonnegative_number(output_rate, 'output_rate')
    max_output_tokens = _nonnegative_number(max_output_tokens, 'max_output_tokens', integer=True)
    if max_output_tokens == 0:
        raise ValueError('max_output_tokens must be positive.')
    results = [json.loads(line) for line in Path(result_path).read_text('utf-8').splitlines() if line.strip()]
    mini = load_questions(mini_path)
    by_id = {q['id']: q for q in mini}
    if not results:
        raise ValueError('Pilot results are empty.')
    if any(r['question_id'] not in by_id or r['condition'] != 'image' for r in results):
        raise ValueError('Report requires an image-only pilot drawn from the current mini.')
    if len({r['question_id'] for r in results}) != len(results):
        raise ValueError('Duplicate pilot question IDs; use a single deduplicated result file.')
    identity_fields = ('model', 'model_name', 'provider', 'config_fingerprint', 'dataset_fingerprint')
    if any(not isinstance(row.get(field), str) or not row[field]
           for row in results for field in identity_fields):
        raise ValueError('Pilot records require model/provider/configuration/dataset identities.')
    if len({tuple(row[field] for field in identity_fields) for row in results}) != 1:
        raise ValueError('Pilot report requires one model/provider/configuration/dataset combination.')
    if any(row.get('dry_run') is not False or row.get('baseline') for row in results):
        raise ValueError('Cost calibration requires real model responses, not dry runs or baselines.')
    strata = Counter(stratum(by_id[r['question_id']]) for r in results)
    if strata != Counter({key: 3 for key in STRATA}):
        raise ValueError('Pilot requires exactly three records per each of the eight strata.')
    for row in results:
        question = by_id[row['question_id']]
        if row.get('qtype') != question['qtype'] or row.get('difficulty') != question['difficulty']:
            raise ValueError('Pilot result task does not match the current mini question.')
        if row.get('image_paths') != question['image_paths']:
            raise ValueError('Pilot result images do not match the current mini question.')
        if row.get('status') not in {'ok', 'invalid_response', 'error'}:
            raise ValueError('Unexpected pilot result status.')
        if not isinstance(row.get('correct'), bool):
            raise ValueError('Scored pilot records require Boolean correctness.')
        if row['status'] != 'ok' and row['correct']:
            raise ValueError('Error/invalid pilot results cannot be correct.')
        _nonnegative_number(row.get('latency_seconds'), 'latency_seconds')
    groups = defaultdict(list)
    valid_usage = []
    for row in results:
        usage = row.get('usage') or {}
        if not isinstance(usage, dict):
            raise ValueError('usage must be a mapping or null.')
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if key in usage and usage[key] is not None:
                _nonnegative_number(usage[key], key, integer=True)
        for detail, key, total in (('prompt_tokens_details', 'cached_tokens', 'prompt_tokens'),
                                   ('completion_tokens_details', 'reasoning_tokens', 'completion_tokens')):
            details = usage.get(detail) or {}
            if not isinstance(details, dict):
                raise ValueError(f'{detail} must be a mapping or null.')
            if key in details and details[key] is not None:
                number = _nonnegative_number(details[key], key, integer=True)
                if usage.get(total) is not None and number > usage[total]:
                    raise ValueError(f'{key} cannot exceed {total}.')
        if all(usage.get(k) is not None for k in ('prompt_tokens', 'completion_tokens')):
            if usage.get('total_tokens') is not None and usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
                raise ValueError('Token usage total disagrees with prompt + completion counts.')
            valid_usage.append(row)
            groups[stratum(by_id[row['question_id']])].append(row)
    if set(groups) != set(STRATA):
        raise ValueError('Cost extrapolation needs observed token usage in every pilot stratum.')
    def mean(group, field):
        return statistics.mean(row['usage'][field] for row in groups[group])
    observations = [dict(prompt_tokens=row['usage']['prompt_tokens'],
                         prompt_characters=len(build_prompt(by_id[row['question_id']], 'image')),
                         image_count=len(by_id[row['question_id']]['image_paths'])) for row in valid_usage]
    calibrations = {str(ratio): calibrate_overhead(observations, ratio) for ratio in (2, 3, 4)}
    max_input = max(row['usage']['prompt_tokens'] for row in valid_usage)
    observed_max_output = max(row['usage']['completion_tokens'] for row in valid_usage)
    planning_output = max(max_output_tokens, observed_max_output)
    estimates = []
    # Local estimate: 3 characters/token, with 2-4 sensitivity interval.
    # UTF-8 byte length + 64 is a deliberately conservative planning proxy,
    # not a tokenizer guarantee for a third-party gateway.
    for condition in ('image', 'text', 'image+text'):
        scenario_inputs = {ratio: 0.0 for ratio in (2, 3, 4)}
        byte_proxy = total_out = 0.0
        count = 0
        for q in mini:
            applicable = q.get('meta', {}).get('text_applicable', True)
            if condition == 'text' and not applicable:
                continue
            group = stratum(q)
            measured_in = mean(group, 'prompt_tokens')
            measured_out = mean(group, 'completion_tokens')
            if condition == 'image':
                scenarios = {ratio: measured_in for ratio in scenario_inputs}
                upper = measured_in
            elif condition == 'text':
                prompt = build_prompt(q, 'text')
                scenarios = {ratio: calibrations[str(ratio)]['overhead_tokens'] + len(prompt) / ratio
                             for ratio in scenario_inputs}
                upper = max_input + len(prompt.encode('utf-8')) + 64
            else:
                extra = '\n\nCoordinate description:\n\n' + q['text_repr'] if applicable else ''
                scenarios = {ratio: measured_in + len(extra) / ratio for ratio in scenario_inputs}
                upper = measured_in + len(extra.encode('utf-8')) + (64 if extra else 0)
            for ratio in scenario_inputs:
                scenario_inputs[ratio] += scenarios[ratio]
            byte_proxy += upper
            total_out += measured_out
            count += 1
        cost = lambda inp, out: (inp * input_rate + out * output_rate) / 1e6
        estimates.append(dict(condition=condition, requests=count,
            estimated_input_tokens=round(scenario_inputs[3]), estimated_output_tokens=round(total_out),
            estimated_usd=cost(scenario_inputs[3], total_out),
            text_tokenization_sensitivity_usd=[cost(min(scenario_inputs.values()),total_out),
                                               cost(max(scenario_inputs.values()),total_out)],
            input_scenarios_by_chars_per_token={str(ratio): round(tokens) for ratio, tokens in scenario_inputs.items()},
            planning_input_token_proxy=round(byte_proxy), planning_output_tokens=count*planning_output,
            conservative_planning_usd=1.2*cost(byte_proxy,count*planning_output)))
    prompt_sum = sum(r['usage']['prompt_tokens'] for r in valid_usage)
    completion_sum = sum(r['usage']['completion_tokens'] for r in valid_usage)
    cached_sum = sum((r['usage'].get('prompt_tokens_details') or {}).get('cached_tokens') or 0
                     for r in valid_usage)
    reasoning_sum = sum((r['usage'].get('completion_tokens_details') or {}).get('reasoning_tokens') or 0
                        for r in valid_usage)
    latencies = [r['latency_seconds'] for r in results]
    summary = dict(
        model=sorted({r['model_name'] for r in results}),
        response_models=dict(Counter(r.get('response_model') for r in results)),
        records=len(results), statuses=dict(Counter(r['status'] for r in results)),
        strata=dict(Counter(stratum(by_id[r['question_id']]) for r in results)),
        invalid_rate=sum(bool(r.get('invalid')) for r in results)/len(results),
        final_answer_line_count=sum(r['parse_method']=='final_answer_line' for r in results),
        exact_model_match_count=sum(r.get('response_model')==r['model_name'] for r in results),
        accuracy=sum(r['correct'] is True for r in results)/len(results),
        usage_records=len(valid_usage), missing_usage_records=len(results)-len(valid_usage),
        usage_coverage_by_stratum={group: dict(records=strata[group], usage_records=len(groups[group]),
             mean_prompt_tokens=mean(group,'prompt_tokens'), mean_completion_tokens=mean(group,'completion_tokens'))
             for group in STRATA},
        observed_prompt_tokens=prompt_sum, observed_completion_tokens=completion_sum,
        observed_total_tokens=prompt_sum+completion_sum,
        observed_cached_input_tokens=cached_sum, observed_reasoning_tokens=reasoning_sum,
        mean_prompt_tokens=prompt_sum/len(valid_usage), mean_completion_tokens=completion_sum/len(valid_usage),
        mean_latency_seconds=statistics.mean(latencies), median_latency_seconds=statistics.median(latencies),
        min_latency_seconds=min(latencies), max_latency_seconds=max(latencies),
        overhead_calibration=calibrations,
        planning_output_cap_configured=max_output_tokens,
        observed_max_completion_tokens=observed_max_output,
        planning_output_tokens_per_request=planning_output,
        price=dict(currency='USD',input_per_million=input_rate,output_per_million=output_rate,
                   basis=price_basis,source=price_source,checked_date='2026-10-21',
                   gateway_source='https://nexfaro.com/api/v1/model-plaza',
                   gateway_public_snapshot='results/pilot/nexfaro_public_pricing.json',
                   account_group_multiplier=None, cache_discount_applied=False),
        observed_usage_reference_cost_usd=(prompt_sum*input_rate+completion_sum*output_rate)/1e6,
        mini_estimates=estimates,
        full_mini_estimated_usd=sum(x['estimated_usd'] for x in estimates),
        full_mini_conservative_planning_usd=sum(x['conservative_planning_usd'] for x in estimates),
        assumptions=[
            'Only image requests are observed. Other conditions have not been called.',
            'Image usage extrapolated separately for eight task strata; missing usage is not treated as zero.',
            'Text input uses 3 characters/token; 2-4 sensitivity is not a confidence interval.',
            'Text-only fixed overhead is inferred by nonnegative regression on image usage; it has not been measured for text requests.',
            'Each characters/token scenario is recalibrated independently; sensitivity endpoints are the minimum and maximum scenario costs.',
            'Image count is confounded with task type. A fitted intercept or repeated cached count does not prove a fixed gateway prefix.',
            'Image+text conservatively adds the coordinate text to observed image input.',
            'Output point estimate uses image-pilot mean per stratum in all conditions.',
            'Observed completion tokens include the reasoning tokens reported by the gateway.',
            'Planning proxy uses UTF-8 byte counts for new text, the larger of configured cap and observed maximum output, and 20% reserve.',
            'Text-only planning retains the maximum observed image input as an overhead proxy; it is not a token or cost guarantee.',
            'No cache discounts applied. Hidden gateway overhead or failed-attempt charges are unobserved.',
            f"{sum(x['requests'] for x in estimates)} planned requests: " + ', '.join(f"{x['requests']} {x['condition']}" for x in estimates) + '; pure-visual text-only questions are skipped.',
            'Third-party actual billing can differ from the reference tariff.'
        ])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'pilot_report.json').write_text(json.dumps(summary,indent=2,allow_nan=False), 'utf-8')
    columns = ['question_id','qtype','difficulty','response_model','status','parse_method','correct',
               'prompt_tokens','completion_tokens','total_tokens','latency_seconds','attempts','raw_response']
    with (output_dir/'pilot_per_question.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns)
        writer.writeheader()
        for r in results:
            record={k:r.get(k) for k in columns}
            record.update({k:(r.get('usage') or {}).get(k) for k in ('prompt_tokens','completion_tokens','total_tokens')})
            writer.writerow(record)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mini',type=Path,default=Path('data/questions_mini.jsonl'))
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--output',type=Path,default=Path('data/questions_pilot24.jsonl'))
    parser.add_argument('--report',type=Path)
    parser.add_argument('--report-dir',type=Path,default=Path('results/pilot'))
    parser.add_argument('--input-rate',type=float,default=2.0)
    parser.add_argument('--output-rate',type=float,default=10.0)
    parser.add_argument('--seed',type=int,default=20261021)
    args=parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.mini,args.output,args.seed),indent=2))
    if args.report:
        print(json.dumps(report(args.report,args.mini,input_rate=args.input_rate,
             output_rate=args.output_rate,output_dir=args.report_dir),indent=2))
    if not args.prepare and not args.report:
        parser.error('Choose --prepare or --report.')


if __name__=='__main__':
    main()
