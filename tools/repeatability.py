#!/usr/bin/env python
"""Cross-capture repeatability of wall positions (independent of how spaces are segmented).

Two captures of the same home are registered by their measured wall faces (best of 4 x 90 deg
rotations, then trimmed 2-D ICP). Each wall face of A is matched to the parallel, same-facing,
overlapping face of B; the residual offset is the wall-position disagreement. Room widths
(distance between two facing walls) are compared the same way.
usage: python tools/repeatability.py <captureA> <captureB> [--rooms]   (--rooms: per-room dimension table)"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynx.drift import icp2d


def faces(name):
    return json.load(open(f'outputs/{name}/plan.json' if Path(f'outputs/{name}/plan.json').exists() else name))['wall_faces']


def samples(fs, R=np.eye(2), t=np.zeros(2), step=0.05):
    pts = []
    for f in fs:
        for a, b in f['intervals']:
            s = np.arange(a, b, step)
            p = np.stack([s, np.full_like(s, f['coord'])], 1) if f['axis'] == 'u' else np.stack([np.full_like(s, f['coord']), s], 1)
            pts.append(p)
    return np.concatenate(pts) @ R.T + t


def transform_face(f, R, t):
    """Faces stay axis-aligned under a k*90 deg rotation (+ small ICP yaw ignored after snapping)."""
    d = np.array([1.0, 0.0]) if f['axis'] == 'u' else np.array([0.0, 1.0])     # along-wall direction
    nrm = np.array([0.0, f['normal_sign']]) if f['axis'] == 'u' else np.array([f['normal_sign'], 0.0])
    out = []
    for a, b in f['intervals']:
        p0 = a * d + f['coord'] * (1 - d); p1 = b * d + f['coord'] * (1 - d)
        out.append((R @ p0 + t, R @ p1 + t))
    d2, n2 = R @ d, R @ nrm
    axis = 'u' if abs(d2[0]) > 0.9 else 'v'
    k, o = (0, 1) if axis == 'u' else (1, 0)
    ivs = sorted([sorted([p[k], q[k]]) for p, q in out])
    coord = float(np.mean([p[o] for p, _ in out]))
    return {'axis': axis, 'coord': coord, 'normal_sign': int(np.sign(n2[o])), 'intervals': ivs}


def overlap(f, g):
    return sum(max(0, min(b, d) - max(a, c)) for a, b in f['intervals'] for c, d in g['intervals'])


def register(A, B):
    pa = samples(A); tree = cKDTree(pa); best = None
    for k in range(4):
        th = k * np.pi / 2; R0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pb = samples(B, R0)
        for dx in np.arange(-6, 6.01, 0.25):
            for dy in np.arange(-6, 6.01, 0.25):
                sc = (tree.query(pb + [dx, dy], distance_upper_bound=0.15)[0] < 0.15).mean()
                if best is None or sc > best[0]:
                    best = (sc, R0, np.array([dx, dy]))
    _, R0, t0 = best
    r = icp2d(samples(B, R0, t0), tree, pa, max_d=0.3)
    R1, t1, inl = r
    yaw = np.degrees(np.arctan2(R1[1, 0], R1[0, 0]))
    t = R1 @ t0 + t1
    return R0, t, yaw, inl


def main(a, b):
    A, B = faces(a), faces(b)
    R0, t, yaw, inl = register(A, B)
    Bt = [transform_face(f, R0, t) for f in B]
    res = []
    for f in A:
        cand = [g for g in Bt if g['axis'] == f['axis'] and g['normal_sign'] == f['normal_sign']
                and abs(g['coord'] - f['coord']) < 0.25 and overlap(f, g) > 0.5]
        if cand:
            g = max(cand, key=lambda g: overlap(f, g))
            res.append((f, g, g['coord'] - f['coord']))
    off = np.array([r[2] for r in res])
    print(f'{a} vs {b}: registration residual yaw {yaw:.2f} deg, inlier {inl:.0%}')
    print(f'  matched wall faces {len(res)}/{len(A)}; wall-position disagreement: median {np.median(np.abs(off))*100:.1f} cm, '
          f'p90 {np.percentile(np.abs(off), 90)*100:.1f} cm')
    # room widths: pairs of facing walls in A (normals pointing at each other) both matched in B
    widths = []
    for i, (f1, g1, _) in enumerate(res):
        for f2, g2, _ in res[i + 1:]:
            if f1['axis'] != f2['axis'] or f1['normal_sign'] == f2['normal_sign']:
                continue
            lo, hi = (f1, f2) if f1['coord'] < f2['coord'] else (f2, f1)
            if lo['normal_sign'] < 0 or overlap(f1, f2) < 0.5:   # must face each other across the room
                continue
            wa = abs(f2['coord'] - f1['coord']); wb = abs(g2['coord'] - g1['coord'])
            if 0.8 < wa < 8:
                widths.append((wa, wb))
    lg = np.array([abs(r[2]) for r in res if r[0]['solid_length_m'] >= 2.5])
    if len(lg):
        print(f'  structural walls (>= 2.5 m solid): {len(lg)} matched, |offset| median {np.median(lg)*100:.1f} cm, max {lg.max()*100:.1f} cm')
    if widths:
        d = np.array([wb - wa for wa, wb in widths])
        print(f'  {len(widths)} wall-to-wall spans: |difference| median {np.median(np.abs(d))*100:.1f} cm, p90 {np.percentile(np.abs(d), 90)*100:.1f} cm')
        for wa, wb in sorted(widths)[:: max(1, len(widths) // 8)]:
            print(f'     {wa:.3f} m  vs  {wb:.3f} m')


if __name__ == '__main__' and '--rooms' not in sys.argv:
    main(sys.argv[1], sys.argv[2])


def rooms_table(a, b):
    """Per-room dimension repeatability: rooms matched through the frame-level GT labels
    (benchmark/gt/<capture>.json), dimensions compared after sorting (axis order may swap).
    A space that is split or merged in either capture is not the same room in both plans and is
    listed separately (that is a segmentation error, scored by eval_spaces.py)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmark'))
    import eval_spaces as E
    rows, maps, bad = [], {}, set()
    for c in (a, b):
        r = E.evaluate(c)
        bad |= set(r['split']) | {s for v in r['merge'].values() for s in v}
        plan = json.load(open(f'outputs/{c}/plan.json'))
        R = {x['id']: x for x in plan['rooms']}
        maps[c] = {s: R[d['type'].split(':')[0]] for s, d in r['detail'].items()}
    for s in maps[a]:
        if s not in maps[b] or s in bad:
            continue
        da = sorted([maps[a][s]['dimensions'][k]['value_m'] for k in ('length_u', 'length_v')])
        db = sorted([maps[b][s]['dimensions'][k]['value_m'] for k in ('length_u', 'length_v')])
        for x, y, lab in zip(da, db, ('short', 'long')):
            d = abs(x - y)
            rows.append({'space': s, 'wall_pair': lab, a: x, b: y, 'diff_cm': round(d * 100, 1),
                         'pass_1cm_or_0.5pct': d <= max(0.01, 0.005 * max(x, y))})
    return rows, sorted(bad & set(maps[a]) & set(maps[b]))


if __name__ == '__main__' and len(sys.argv) > 3 and sys.argv[3] == '--rooms':
    rows, excluded = rooms_table(sys.argv[1], sys.argv[2])
    for r in rows:
        print(r)
    print('excluded (split/merged in a capture):', excluded)
    d = np.array([r['diff_cm'] for r in rows])
    print(f"{len(rows)} wall-to-wall dimensions in {len({r['space'] for r in rows})} rooms: median diff {np.median(d):.1f} cm, "
          f"pass (<=1 cm or 0.5 %) {sum(r['pass_1cm_or_0.5pct'] for r in rows)}/{len(rows)}")
