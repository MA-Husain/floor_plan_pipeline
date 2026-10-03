"""Photo tier plan: one room per folder, measured on its own, then placed into one plan.

A folder's photos also see neighbouring rooms through doorways, so the whole-capture
segmentation cannot tell whose space is whose on sparse stills. Each folder is therefore planned
separately with the same plan code (lidar.run on that folder's views only); its room is the
segmented space that contains the folder's camera positions. The rooms are then expressed in one
frame using the stitched poses (rgb.photo_stitch), each room's own Manhattan frame snapped to the
building's (multiples of 90 degrees). Rooms with no doorway link are kept, laid out beside the
plan and flagged position_known = False.
"""
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from . import lidar
from .rgb import ReconCapture


def _sub(cap, ids):
    sub = ReconCapture([cap.views[i] for i in ids], lambda m: cap._src(ids[m]), cap.rgb_size, fps=1.0,
                       tier='photo', error_model=cap.error_model, names=[cap.names[i] for i in ids])
    return sub


def _uv_to_world(P, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    u, v = P[:, 0], P[:, 1]
    return np.stack([c * u - s * v, s * u + c * v], 1)          # world (x, z)


def _world_to_uv(X, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * X[:, 0] + s * X[:, 1], -s * X[:, 0] + c * X[:, 1]], 1)


def run(cap, detect=True, log=print, cache_dir=None):
    labels = cap.room_of
    unplaced = set((cap.alignment or {}).get('unplaced', []))
    folders = sorted(set(labels))
    per = {}
    for F in folders:
        ids = [i for i, l in enumerate(labels) if l == F]
        if len(ids) < 2:
            log(f'[photo] {F}: only {len(ids)} photo - not enough to measure a room'); continue
        sub = _sub(cap, ids)
        try:
            plan, ctx = lidar.run(sub, detect=detect, log=lambda *a: None, drift_mode='off',
                                  cache_dir=None if cache_dir is None else f'{cache_dir}/{F}')
        except Exception as e:                       # too little geometry for a plan
            log(f'[photo] {F}: no plan ({type(e).__name__}: {e})'); continue
        cam_uv = np.array(plan['debug']['camera_uv'])
        best, hits = None, 0
        for r in plan['rooms']:
            P = Polygon(r['polygon']).buffer(0.15)
            n = sum(P.contains(Point(*xy)) for xy in cam_uv)
            if n > hits:
                best, hits = r, n
        if best is None:                             # cameras outside every space: take the largest
            best = max(plan['rooms'], key=lambda r: r['area_m2']['value']) if plan['rooms'] else None
        if best is None:
            log(f'[photo] {F}: no room found'); continue
        per[F] = {'plan': plan, 'room': best, 'ctx': ctx, 'ids': ids}
        d = best['dimensions']
        log(f"[photo] {F}: {d['length_u']['value_m']:.2f} x {d['length_v']['value_m']:.2f} m "
            f"(+-{d['length_u']['ci95_m']:.2f}), ceiling {(best['ceiling_height'] or {}).get('value_m')}")
    if not per:
        raise RuntimeError('no room could be measured from the photos')
    # common frame: the largest placed room's plan frame
    ref = max((F for F in per if F not in unplaced), key=lambda F: per[F]['room']['area_m2']['value'], default=None) \
        or max(per, key=lambda F: per[F]['room']['area_m2']['value'])
    yaw0 = per[ref]['plan']['frame']['yaw_rad']
    rooms, polys, openings = [], [], []
    for k, (F, d) in enumerate(per.items()):
        r = dict(d['room'])
        yaw = d['plan']['frame']['yaw_rad']
        P = np.array(r['polygon'])
        Xw = _uv_to_world(P, yaw)                    # this room in world x,z
        uv = _world_to_uv(Xw, yaw0)                  # ... in the common plan frame
        poly = Polygon(uv)
        # snap the room's walls to the building axes (rooms share the Manhattan frame up to k*90 deg)
        dyaw = np.degrees(yaw - yaw0)
        snap = (dyaw + 45) % 90 - 45
        origin = poly.centroid
        poly = affinity.rotate(poly, snap, origin=origin)
        r['polygon'] = np.round(np.array(poly.exterior.coords[:-1]), 4).tolist()
        # this room's openings, carried into the common frame the same way
        for o in d['plan']['openings']:
            if d['room']['id'] not in o['rooms']:
                continue
            a0, a1 = o['span']; c = o['wall_coord']
            ends = np.array([[a0, c], [a1, c]]) if o['axis'] == 'u' else np.array([[c, a0], [c, a1]])
            seg = affinity.rotate(LineString(_world_to_uv(_uv_to_world(ends, yaw), yaw0)), snap, origin=origin)
            (x0, y0), (x1, y1) = seg.coords
            horiz = abs(x1 - x0) >= abs(y1 - y0)
            oo = dict(o)
            oo.update(axis='u' if horiz else 'v', wall_coord=round(float((y0 + y1) / 2 if horiz else (x0 + x1) / 2), 4),
                      span=sorted([round(float(x0 if horiz else y0), 4), round(float(x1 if horiz else y1), 4)]),
                      rooms=[f'R{k + 1}'], room_folder=F)
            openings.append(oo)
        r['id'] = f'R{k + 1}'; r['name'] = F; r['folder_label'] = F
        r['position_known'] = F not in unplaced
        r['snap_rotation_deg'] = round(float(snap), 2)
        rooms.append(r); polys.append(poly)
    # unplaced rooms: a row beside the plan
    placed = [p for r, p in zip(rooms, polys) if r['position_known']]
    x0 = (unary_union(placed).bounds[2] + 1.5) if placed else 0.0
    for r, i in ((r, i) for i, r in enumerate(rooms) if not r['position_known']):
        P = polys[i]; b = P.bounds
        dx, dy = x0 - b[0], -b[1]
        P = affinity.translate(P, dx, dy)
        for o in openings:
            if o['rooms'] == [r['id']]:
                o['span'] = [v + (dx if o['axis'] == 'u' else dy) for v in o['span']]
                o['wall_coord'] += dy if o['axis'] == 'u' else dx
        x0 = P.bounds[2] + 1.0
        r['polygon'] = np.round(np.array(P.exterior.coords[:-1]), 4).tolist(); polys[i] = P
    # overlaps between placed rooms = stitching error, reported
    overlaps = []
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if rooms[i]['position_known'] and rooms[j]['position_known']:
                a = polys[i].intersection(polys[j]).area
                if a > 0.25:
                    overlaps.append({'rooms': [rooms[i]['name'], rooms[j]['name']], 'area_m2': round(a, 2)})
    links = (cap.alignment or {}).get('links', [])
    adjacency = sorted({tuple(sorted((next(r['id'] for r in rooms if r['name'] == l['from']),
                                      next(r['id'] for r in rooms if r['name'] == l['to']))))
                        for l in links if l['from'] in per and l['to'] in per})
    ceilings = [r['ceiling_height']['value_m'] for r in rooms if r.get('ceiling_height')]
    plan = {
        'schema': 'brynx.plan/1.0', 'tier': 'photo', 'capture': str(getattr(cap, 'root', '')),
        'error_model': cap.error_model,
        'frame': {'yaw_rad': yaw0, 'units': 'm', 'axes': 'u,v horizontal (reference room frame); h up from floor'},
        'global_ceiling': {'value_m': round(float(np.median(ceilings)), 4), 'support': len(ceilings)} if ceilings else None,
        'rooms': rooms, 'openings': [dict(o, id=f'O{i + 1}') for i, o in enumerate(openings)],
        'adjacency': adjacency,
        'footprint_m2': round(unary_union([p for r, p in zip(rooms, polys) if r['position_known']] or polys).area, 3),
        'stitch': {'links': links, 'unplaced': sorted(unplaced), 'overlaps': overlaps,
                   'note': 'rooms are joined only through doorway photos that see the next room'},
        'wall_faces': [], 'objects': [],
        'drift': {'mode': 'n/a (photos: no trajectory)'},
        'debug': {},
    }
    log(f"[photo] plan: {len(rooms)} rooms, {len(rooms) - len(unplaced & set(per))} placed, overlaps {len(overlaps)}")
    return plan, per
