"""Space-segmentation + typing evaluation against frame-level ground truth.

GT: for each true space, frames at which the camera stands inside it (hand-labelled from video).
Metrics per capture:
  split  - true spaces whose frames fall into more than one predicted space
  merge  - predicted spaces containing frames of more than one true space
  missed - GT frames whose camera position lies in no predicted space
  type   - true spaces whose (majority) predicted space has an acceptable type
Usage: eval_spaces.py <capture> [<capture> ...]   (uses outputs/<capture>/plan.json)
"""
import sys, json
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brynz.capture import StrayCapture
from brynz.geometry import plan_coords


def evaluate(cap_name):
    gt = json.load(open(ROOT / 'benchmark/gt' / f'{cap_name}.json'))
    plan = json.load(open(ROOT / 'outputs' / cap_name / 'plan.json'))
    cap = StrayCapture(ROOT / cap_name)
    cam, _ = plan_coords(cap.positions, plan['frame']['yaw_rad'], plan['frame']['floor_y_world'])
    polys = [(r['id'], r['name'], r['type'], Polygon(r['polygon'])) for r in plan['rooms']]
    assign = {}
    for sname, s in gt['spaces'].items():
        assign[sname] = []
        for f in s['frames']:
            p = Point(*cam[f])
            hit = next((r for r in polys if r[3].buffer(0.1).contains(p)), None)
            assign[sname].append(hit[0] if hit else None)
    split = [s for s, a in assign.items() if len({x for x in a if x}) > 1]
    owners = {}
    for s, a in assign.items():
        for x in a:
            if x:
                owners.setdefault(x, set()).add(s)
    merge = {k: sorted(v) for k, v in owners.items() if len(v) > 1}
    missed = sum(x is None for a in assign.values() for x in a)
    nframes = sum(len(a) for a in assign.values())
    typ_ok = []
    for s, a in assign.items():
        ids = [x for x in a if x]
        if not ids:
            typ_ok.append((s, False, None)); continue
        maj = max(set(ids), key=ids.count)
        t = next(r[2] for r in polys if r[0] == maj)
        typ_ok.append((s, t in gt['spaces'][s]['type'], f'{maj}:{t}'))
    n = len(gt['spaces'])
    res = {'capture': cap_name, 'true_spaces': n, 'pred_spaces': len(polys), 'split': split, 'merge': merge,
           'missed_frames': f'{missed}/{nframes}', 'type_correct': f"{sum(t[1] for t in typ_ok)}/{n}",
           'detail': {s: {'pred': a, 'type': t} for (s, ok, t), a in zip(typ_ok, assign.values())},
           'score': round((n - len(split) - len(merge) + sum(t[1] for t in typ_ok)) / (2 * n), 3)}
    return res


if __name__ == '__main__':
    for c in sys.argv[1:]:
        r = evaluate(c)
        print(f"{c}: true {r['true_spaces']} pred {r['pred_spaces']} | split {len(r['split'])} merge {len(r['merge'])} "
              f"missed {r['missed_frames']} type {r['type_correct']} | score {r['score']}")
        for s, d in r['detail'].items():
            print(f"   {s:15s} -> {d['type']}   frames->{d['pred']}")
        if r['merge']:
            print('   merges:', r['merge'])
