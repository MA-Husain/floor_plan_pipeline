"""Space analysis: decide which reconstructed regions are rooms, and what kind.

Design first, objects second. Every region gets structural features (area, inscribed width,
elongation, number of openings, connectivity, whether the operator walked through it, floor
observed, floor below the main level = stairwell). Regions that are not spaces at all are
removed (slivers -> merged into a neighbour; unobserved enclosed blocks -> voids such as built-in
wardrobes or shafts). Remaining spaces are typed by rules on those features, refined by fixtures
seen *from inside the same space*.
"""
import numpy as np
import cv2
from shapely.geometry import Point
from shapely.ops import unary_union

from . import priors


def raster_mask(poly, grid):
    m = np.zeros(grid.shape, np.uint8)
    for g in (poly.geoms if hasattr(poly, 'geoms') else [poly]):
        pts = ((np.array(g.exterior.coords) - grid.origin) / grid.res).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def inscribed_width(poly, grid):
    m = raster_mask(poly, grid).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    return float(cv2.distanceTransform(m, cv2.DIST_L2, 5).max() * 2 * grid.res)


def clean_regions(polys, grid, floor_mask, walk_mask, min_width=priors.SPACE_MIN_WIDTH_M, min_area=priors.SPACE_MIN_AREA_M2, log=print):
    """Merge slivers into the neighbour sharing the longest boundary; drop unobserved voids.
    Returns (new polygons, mapping old index -> new index or -1)."""
    polys = [p for p in polys]
    alive = list(range(len(polys)))
    owner = list(range(len(polys)))

    def root(i):
        while owner[i] != i:
            i = owner[i]
        return i

    changed = True
    while changed:
        changed = False
        for i in list(alive):
            P = polys[i]
            m = raster_mask(P, grid)
            seen = (floor_mask & m).sum() / max(m.sum(), 1)
            walked = (walk_mask & m).sum() / max(m.sum(), 1)
            if seen < 0.12 and walked < 0.02 and P.area < 6:
                log(f'[spaces] void (never observed inside, {P.area:.1f} m2) -> removed')
                alive.remove(i); polys[i] = None
                changed = True
                break
            w = inscribed_width(P, grid)
            if w < min_width or P.area < min_area:
                best, L = None, 0.0
                for j in alive:
                    if j == i:
                        continue
                    s = P.buffer(0.03).intersection(polys[j].buffer(0.03)).area / 0.06
                    if s > L:
                        best, L = j, s
                if best is not None and L > 0.3:
                    polys[best] = unary_union([polys[best], P]).buffer(0.01, join_style='mitre').buffer(-0.01, join_style='mitre')
                    alive.remove(i); owner[i] = best; polys[i] = None
                    changed = True
                    break
    new_index = {}
    out = []
    for i in alive:
        new_index[i] = len(out)
        P = polys[i]
        if P.geom_type != 'Polygon':
            P = max(P.geoms, key=lambda g: g.area)
        out.append(P)
    mapping = []
    for i in range(len(owner)):
        if polys[i] is None and owner[i] == i:
            mapping.append(-1)  # void
            continue
        r = i
        while owner[r] != r:
            r = owner[r]
        mapping.append(new_index.get(r, -1))
    return out, mapping


def features(P, grid, floor_mask, walk_mask, uv, h, ny, openings_for_room, neighbours):
    m = raster_mask(P, grid)
    b = P.minimum_rotated_rectangle.exterior.coords
    e = [np.hypot(b[k + 1][0] - b[k][0], b[k + 1][1] - b[k][1]) for k in range(2)]
    w_in = inscribed_width(P, grid)
    sel = (ny > 0.9) & (h < -0.12) & (h > -3.5)
    r, c = grid.ij(uv[sel])
    below = int(m[r, c].sum())
    return {'area': P.area, 'inscribed_width': w_in, 'long': max(e), 'short': min(e),
            'elongation': max(e) / max(min(e), 0.1),
            'floor_seen': float((floor_mask & m).sum() / max(m.sum(), 1)),
            'walked': float((walk_mask & m).sum() / max(m.sum(), 1)),
            'n_openings': len(openings_for_room), 'degree': len(neighbours),
            'floor_below_level_pts': below}


# fixture instance -> {space type: weight}
WEIGHTS = {
    'toilet': {'Bathroom': 3}, 'shower': {'Bathroom': 3}, 'bathtub': {'Bathroom': 2}, 'sink': {'Bathroom': 1, 'Kitchen': 1},
    'stove': {'Kitchen': 3}, 'refrigerator': {'Kitchen': 2}, 'kitchen cabinet': {'Kitchen': 2}, 'microwave': {'Kitchen': 1},
    'sofa': {'Living': 3}, 'television': {'Living': 1.5, 'Bedroom': 0.5},
    'bed': {'Bedroom': 3}, 'wardrobe': {'Bedroom': 1},
    'dining table': {'Dining': 2}, 'desk': {'Study': 2}, 'office chair': {'Study': 1},
    'staircase': {'Stairs': 3}, 'stair railing': {'Stairs': 2}, 'washing machine': {'Utility': 3},
}
# size priors: (min area, max area) where the type is plausible; outside -> score x0.3
SIZE = {'Bathroom': (1.5, 8.0), 'Kitchen': (3.0, 25), 'Living': (7.0, 60), 'Bedroom': (6.0, 30),
        'Dining': (5.0, 40), 'Study': (4.0, 20), 'Stairs': (2.0, 30), 'Utility': (1.5, 8)}


# Fixtures that by themselves identify a room's use. Desks, TVs, chairs and tables turn up in
# any room, so they never name a space alone: when in doubt the answer is the generic "Room".
DECISIVE = {'toilet': 'Bathroom', 'shower': 'Bathroom', 'bathtub': 'Bathroom',
            'stove': 'Kitchen', 'kitchen cabinet': 'Kitchen', 'refrigerator': 'Kitchen',
            'bed': 'Bedroom', 'sofa': 'Living', 'staircase': 'Stairs', 'stair railing': 'Stairs',
            'washing machine': 'Utility'}
MIN_SCORE = 3.0      # evidence needed to name a type
MIN_MARGIN = 1.5     # top score must beat the runner-up by this factor


def classify(f, objects):
    """Design first: structure decides corridors, closets and stairs; fixture instances (3-D
    clustered) score the rest with size priors. A specific type is named only when a decisive
    fixture supports it and it clearly wins; otherwise the space is a plain "Room".
    Returns (type, reason)."""
    score = {}
    for cls, n in objects.items():
        for t, w in WEIGHTS.get(cls, {}).items():
            score[t] = score.get(t, 0) + w * min(n, 3)
    decisive = {DECISIVE[c] for c in objects if c in DECISIVE}
    stairs_geom = f['floor_below_level_pts'] > 400
    if stairs_geom:
        score['Stairs'] = score.get('Stairs', 0) + 4
        if not decisive - {'Stairs'}:   # geometry counts only when no other use is evident
            decisive.add('Stairs')
    for t in score:
        lo, hi = SIZE.get(t, (0, 1e9))
        if not lo <= f['area'] <= hi:
            score[t] *= 0.3
    best = sorted(((t, v) for t, v in score.items() if t in decisive), key=lambda kv: -kv[1])
    top = best[0] if best else ('Room', 0)
    if f['inscribed_width'] < priors.CORRIDOR_MAX_WIDTH_M and f['elongation'] >= priors.CORRIDOR_MIN_ELONGATION and top[1] < 4:
        return 'Corridor', f"narrow ({f['inscribed_width']:.2f} m) and elongated ({f['elongation']:.1f}:1)"
    if f['area'] < priors.CLOSET_MAX_AREA_M2 and f['n_openings'] <= 1 and top[1] < 3:
        return 'Closet', 'small single-entry space'
    if top[1] >= MIN_SCORE:
        if top[0] == 'Bathroom' and f['area'] < 2.5 and 'shower' not in objects and 'bathtub' not in objects:
            return 'WC', 'toilet in a small space'
        if len(best) > 1 and best[1][1] >= MIN_SCORE and {top[0], best[1][0]} == {'Living', 'Kitchen'}:
            return 'Living / Kitchen', f'open plan: {top[0]} {top[1]:.1f}, {best[1][0]} {best[1][1]:.1f}'
        if len(best) == 1 or top[1] >= MIN_MARGIN * best[1][1]:
            return top[0], ', '.join(f'{k} {v:.1f}' for k, v in best[:3])
    if f['degree'] >= 3 and f['area'] < 15 and not objects:
        return 'Hall', f"connects {f['degree']} spaces, no fixtures"
    return 'Room', 'use not certain from fixtures or structure'


def name_spaces(rooms):
    count = {}
    for r in rooms:
        count[r['type']] = count.get(r['type'], 0) + 1
    seen = {}
    for r in rooms:
        t = r['type']
        seen[t] = seen.get(t, 0) + 1
        r['name'] = t if count[t] == 1 else f'{t} {seen[t]}'


def room_of(rooms_poly, xy, pad=0.05):
    p = Point(*xy)
    for k, P in enumerate(rooms_poly):
        if P.buffer(pad).contains(p):
            return k
    return -1
