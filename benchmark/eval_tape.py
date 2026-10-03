#!/usr/bin/env python
"""Score a plan against tape / laser ground truth (benchmark/gt/<name>.json, key "rooms").

The measured room is located in the plan by the user's own labels: the photo folder name (photo
tier) or the video seconds when the camera was in that room (video tier). Reports the error and
whether the truth lies inside the plan's 95 % interval (calibration), per dimension.
usage: python benchmark/eval_tape.py <plan.json> <gt.json>"""
import json
import sys
from collections import Counter

import numpy as np
from shapely.geometry import Point, Polygon


def locate(plan, spec):
    rooms = plan['rooms']
    if spec.get('photo_folder'):
        m = [r for r in rooms if r.get('folder_label') == spec['photo_folder']]
        if m:
            return max(m, key=lambda r: r['folder_votes'].get(spec['photo_folder'], 0)), 'photo folder'
    dbg = plan.get('debug', {})
    if spec.get('video_seconds') and dbg.get('camera_time_s'):
        a, b = spec['video_seconds']
        uv = [xy for xy, t in zip(dbg['camera_uv'], dbg['camera_time_s']) if a <= t <= b]
        hits = Counter()
        for xy in uv:
            for r in rooms:
                if Polygon(r['polygon']).buffer(0.1).contains(Point(*xy)):
                    hits[r['id']] += 1
        if hits:
            rid = hits.most_common(1)[0][0]
            return next(r for r in rooms if r['id'] == rid), f'video {a}-{b}s ({hits[rid]}/{len(uv)} cameras)'
    return None, 'not found'


def score(plan, gt):
    rows = []
    for name, spec in gt['rooms'].items():
        r, how = locate(plan, spec)
        if r is None:
            rows.append({'room': name, 'found': False}); continue
        d = r['dimensions']
        pred = sorted([(d['length_u']['value_m'], d['length_u']['ci95_m']), (d['length_v']['value_m'], d['length_v']['ci95_m'])], reverse=True)
        truth = sorted(spec['lengths_m'], reverse=True)
        for (v, ci), t, lab in zip(pred, truth, ('length', 'width')):
            rows.append({'room': name, 'plan_space': f"{r['id']} {r['name']}", 'matched_by': how, 'dim': lab,
                         'truth_m': t, 'pred_m': v, 'ci95_m': ci, 'error_m': round(v - t, 4),
                         'error_pct': round((v - t) / t * 100, 2), 'inside_ci': abs(v - t) <= ci})
        c = r.get('ceiling_height')
        if spec.get('ceiling_height_m'):
            t = spec['ceiling_height_m']
            if c:
                rows.append({'room': name, 'plan_space': f"{r['id']} {r['name']}", 'matched_by': how, 'dim': 'ceiling',
                             'truth_m': t, 'pred_m': c['value_m'], 'ci95_m': c['ci95_m'], 'error_m': round(c['value_m'] - t, 4),
                             'error_pct': round((c['value_m'] - t) / t * 100, 2), 'inside_ci': abs(c['value_m'] - t) <= c['ci95_m']})
            else:
                rows.append({'room': name, 'dim': 'ceiling', 'truth_m': t, 'pred_m': None, 'note': 'ceiling not measured (not seen)'})
    return rows


if __name__ == '__main__':
    plan, gt = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
    for row in score(plan, gt):
        if row.get('pred_m') is None:
            print(row); continue
        print(f"{row['room']:8s} {row['dim']:8s} truth {row['truth_m']:.3f}  plan {row['pred_m']:.3f} +- {row['ci95_m']:.3f}  "
              f"error {row['error_m'] * 100:+.1f} cm ({row['error_pct']:+.1f} %)  {'inside CI' if row['inside_ci'] else 'OUTSIDE CI'}  "
              f"[{row['plan_space']}, {row['matched_by']}]")
