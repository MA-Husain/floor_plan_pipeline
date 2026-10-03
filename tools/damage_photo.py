#!/usr/bin/env python
"""Damage check on a single photo (no depth): sliding tiles -> surface gate -> defect classifier.
Writes an overlay with the defect class per tile. Extent here is in image tiles, not metres: metric
extent needs the LiDAR/video tiers (or a reference length in the photo).
usage: python tools/damage_photo.py <image> <out.png>"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynx.damage import DefectClassifier, CLASS_MAP, P_DEFECT, SURFACE_MIN

COLORS = {'crack': (0, 0, 230), 'water_stain': (0, 140, 255), 'mould': (40, 160, 40), 'peeling_paint': (200, 0, 200),
          'spalling': (200, 120, 0)}


def analyse(img, clf, scales=(1 / 4, 1 / 6)):
    H, W = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tiles, boxes = [], []
    for s in scales:
        t = int(min(H, W) * s)
        for y in range(0, H - t + 1, t // 2):
            for x in range(0, W - t + 1, t // 2):
                tiles.append(cv2.resize(rgb[y:y + t, x:x + t], (224, 224), interpolation=cv2.INTER_AREA))
                boxes.append((x, y, t))
    f = clf.embed(tiles)
    P = clf.probs(None, feats=f)
    surf = clf.surface_prob(f)
    plain = clf.classes.index('plain')
    out = []
    for (x, y, t), p, sp in zip(boxes, P, surf):
        pd = 1 - p[plain]
        q = p.copy(); q[plain] = -1
        c = clf.classes[int(q.argmax())]
        out.append({'box': (x, y, t), 'surface': float(sp), 'p_defect': float(pd), 'class': CLASS_MAP[c], 'raw': c,
                    'positive': bool(sp >= SURFACE_MIN and pd >= P_DEFECT)})
    return out


def draw(img, res):
    H, W = img.shape[:2]
    over = img.copy()
    votes = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    cls_map = {}
    for r in res:
        x, y, t = r['box']
        cnt[y:y + t, x:x + t] += 1
        if r['positive']:
            votes[y:y + t, x:x + t] += 1
            cls_map.setdefault(r['class'], np.zeros((H, W), np.float32))[y:y + t, x:x + t] += 1
    hit = votes / np.maximum(cnt, 1) >= 0.5
    if cls_map:
        stack = np.stack([cls_map[k] for k in cls_map]); keys = list(cls_map)
        lab = stack.argmax(0)
        for i, k in enumerate(keys):
            m = hit & (lab == i)
            over[m] = (0.55 * over[m] + 0.45 * np.array(COLORS[k])).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cs, -1, COLORS[k], 3)
    y0 = 30
    for k, c in COLORS.items():
        cv2.rectangle(over, (10, y0 - 18), (32, y0 + 2), c, -1)
        cv2.putText(over, k, (40, y0), 0, 0.7, (0, 0, 0), 4); cv2.putText(over, k, (40, y0), 0, 0.7, (255, 255, 255), 2)
        y0 += 28
    return over, hit


if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    if img is None:
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(sys.argv[1]).convert('RGB')), cv2.COLOR_RGB2BGR)
    clf = DefectClassifier()
    res = analyse(img, clf)
    over, hit = draw(img, res)
    cv2.imwrite(sys.argv[2], over)
    pos = [r for r in res if r['positive']]
    from collections import Counter
    print(f'{len(res)} tiles, {len(pos)} positive; classes {dict(Counter(r["raw"] for r in pos))}; '
          f'{hit.mean():.0%} of image flagged; non-surface tiles gated out: {sum(r["surface"] < SURFACE_MIN for r in res)}')
