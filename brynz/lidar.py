"""Capture -> stitched multi-room plan JSON (output contract).

Runs on a Stray Scanner LiDAR capture, or on any object with the same interface (brynz.rgb's
ReconCapture for the video and photo tiers). The tier's error model widens every interval."""
import time
import numpy as np
import cv2
from shapely.geometry import Point
from shapely.ops import unary_union

from .capture import StrayCapture
from .cloud import build_cloud, voxel_reduce
from .geometry import floor_ceiling, manhattan_yaw, plan_coords, rotate_dirs, Grid
from .plan import oriented_walls, segment, disk
from . import cells, spaces, semantics, drift, priors

# Error budget (1-sigma). iPhone LiDAR range bias at 0.5-3 m is ~0.5-1 cm per surface; residual
# ARKit drift between two faces of one room grows with their separation.
SIGMA_FACE_SYS = 0.006
SIGMA_DRIFT_PER_M = 0.002
Z95 = 1.96
DOOR_MAX = priors.DOOR_MAX_M


LIDAR_ERRORS = {'face_sys': SIGMA_FACE_SYS, 'drift_per_m': SIGMA_DRIFT_PER_M, 'scale_rel': 0.0}
ERR = dict(LIDAR_ERRORS)   # active tier's error model (set by run)


def face_sigma(f):
    if not f['measured']:
        return 0.05 + ERR['face_sys']  # edge inferred from free space only
    return float(np.hypot(f['std'] / np.sqrt(max(f['n'], 1)) * 3, ERR['face_sys']))  # x3: correlated points


def edge_faces(verts, faces, tol=0.03):
    """The measured face under each polygon edge (cell-complex edges lie on cut lines)."""
    out = []
    n = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        horiz = abs(b[0] - a[0]) > abs(b[1] - a[1])
        c = a[1] if horiz else a[0]
        cand = [f for f in faces if f.axis == ('u' if horiz else 'v') and abs(f.coord - c) < tol]
        L = float(np.hypot(*(b - a)))
        mid = float((a[0] + b[0]) / 2 if horiz else (a[1] + b[1]) / 2)
        if cand:
            f = max(cand, key=lambda f: f.n)
            out.append({'horizontal': bool(horiz), 'coord': float(c), 'std': f.std, 'n': f.n, 'measured': True,
                        'length': L, 'mid': mid})
        else:
            out.append({'horizontal': bool(horiz), 'coord': float(c), 'std': None, 'n': 0, 'measured': False,
                        'length': L, 'mid': mid})
    return out


def _principal(edges, lo_side):
    """Wall line carrying the most measured length on one side of the room (edges within 3 cm
    are one line). Free-space edges count 30 %: they bound the room but were not seen as wall."""
    if any(e['measured'] for e in edges):     # a measured wall on this side always beats free space
        edges = [e for e in edges if e['measured']]
    lines = []
    for e in sorted(edges, key=lambda e: e['coord']):
        w = e['length'] * (1.0 if e['measured'] else 0.3)
        if lines and abs(e['coord'] - lines[-1]['coord']) < 0.03:
            lines[-1]['w'] += w; lines[-1]['edges'].append(e)
        else:
            lines.append({'coord': e['coord'], 'w': w, 'edges': [e]})
    best = max(lines, key=lambda l: l['w'])
    e = max(best['edges'], key=lambda e: (e['measured'], e['n']))
    return best['coord'], face_sigma(e), e['measured']


def room_dimensions(faces_info):
    """Room length / width as a tape would take them: between the principal opposing walls of
    each axis (the wall line with the most measured length on each side of the room's centre), not
    between the extreme corners of the outline, which alcoves, reveals and segmentation slivers
    move. The extreme extents are kept as extent_u / extent_v (95 % CIs throughout)."""
    out = {}
    for name, horiz in (('u', False), ('v', True)):
        edges = [f for f in faces_info if f['horizontal'] == horiz]
        cs = np.array([f['coord'] for f in edges]); ls = np.array([f['length'] for f in edges])
        centre = float((cs * ls).sum() / ls.sum())
        lo_e = [f for f in edges if f['coord'] < centre]
        hi_e = [f for f in edges if f['coord'] > centre]
        arr = [(f['coord'], face_sigma(f), f['measured']) for f in edges]
        a, b = min(arr), max(arr)
        if lo_e and hi_e:
            a2, b2 = _principal(lo_e, True), _principal(hi_e, False)
        else:
            a2, b2 = a, b
        for key, (lo, hi) in ((f'length_{name}', (a2, b2)), (f'extent_{name}', (a, b))):
            L = hi[0] - lo[0]
            s = np.sqrt(lo[1] ** 2 + hi[1] ** 2 + (ERR['drift_per_m'] * L) ** 2 + (ERR['scale_rel'] * L) ** 2)
            out[key] = {'value_m': round(L, 4), 'ci95_m': round(Z95 * s, 4),
                        # False: one end is where the capture's view ran out, not a seen wall -
                        # the true wall is at least this far (CI widened by the free-space term)
                        'bounded_by_measured_walls': bool(lo[2] and hi[2])}
    return out


def room_ceiling(uv, h, ny, poly, floor_sigma, band=(1.9, 4.5), cell=0.25):
    """Ceiling = a downward-facing plane spanning the room. Undersides of shelves, wardrobes,
    door heads and stair flights also face down, so a candidate height is only accepted when its
    points cover a real share of the room's plan area; otherwise the ceiling is not measured."""
    m = (ny < -0.95) & (h > band[0]) & (h < band[1])
    if m.sum() < 200:
        return None
    idx = np.nonzero(m)[0]
    idx = idx[:: max(1, len(idx) // 40000)]
    inside = np.array([poly.contains(Point(*p)) for p in uv[idx]])
    hh, pp = h[idx][inside], uv[idx][inside]
    if len(hh) < 150:
        return None
    n_cells = max(1.0, poly.area / cell ** 2)
    hist, e = np.histogram(hh, bins=np.arange(band[0], band[1], 0.01))
    best = None
    for b in np.argsort(hist)[::-1][:40]:            # strongest planes first; the highest well-covered one wins
        pk = e[b] + 0.005
        sel = np.abs(hh - pk) < 0.03
        cells_hit = np.unique(np.floor(pp[sel] / cell).astype(np.int64), axis=0)
        cov = sum(poly.contains(Point(*((c + 0.5) * cell))) for c in cells_hit) / n_cells
        if sel.sum() >= 150 and cov >= priors.CEILING_MIN_COVERAGE and (best is None or pk > best[0] + 0.05):
            best = (pk, cov)
    if best is None:
        return None
    inl = hh[np.abs(hh - best[0]) < 0.03]
    hc = float(np.mean(inl))
    s = np.sqrt((np.std(inl) / np.sqrt(len(inl)) * 3) ** 2 + ERR['face_sys'] ** 2 + floor_sigma ** 2 + (ERR['scale_rel'] * hc) ** 2)
    return {'value_m': round(float(np.mean(inl)), 4), 'ci95_m': round(Z95 * float(s), 4),
            'n_points': int(len(inl)), 'coverage': round(float(best[1]), 3)}


def find_windows(polys, faces, uv, h, fam_u, fam_v, min_w=0.5, max_w=3.0):
    """Exterior wall gaps with wall surface below sill height but none above (glass)."""
    union = unary_union(polys)
    out = []
    for f in faces:
        ivs = sorted(f.intervals)
        for (_, g0), (g1, _) in zip(ivs[:-1], ivs[1:]):
            if not (min_w <= g1 - g0 <= max_w):
                continue
            mid = (g0 + g1) / 2
            p_in = (mid, f.coord + 0.15 * f.sign) if f.axis == 'u' else (f.coord + 0.15 * f.sign, mid)
            p_out = (mid, f.coord - 0.25 * f.sign) if f.axis == 'u' else (f.coord - 0.25 * f.sign, mid)
            if not union.contains(Point(*p_in)) or union.contains(Point(*p_out)):
                continue
            fam = fam_u if f.axis == 'u' else fam_v
            pp, al = (1, 0) if f.axis == 'u' else (0, 1)
            m = fam & (np.abs(uv[:, pp] - f.coord) < 0.06) & (uv[:, al] > g0 + 0.05) & (uv[:, al] < g1 - 0.05)
            hh = h[m]
            low = (hh > 0.25) & (hh < 0.85)
            high = (hh > 1.1) & (hh < 1.8)
            if low.sum() > 30 and high.sum() < 0.3 * low.sum():
                out.append({'type': 'window', 'room_index': spaces.room_of(polys, p_in), 'axis': f.axis,
                            'wall_coord': round(f.coord, 4), 'span': [round(g0, 4), round(g1, 4)],
                            'width_m': {'value': round(g1 - g0, 4), 'ci95': round(Z95 * float(np.hypot(0.02, ERR['scale_rel'] * (g1 - g0))), 4)},
                            'sill_height_m': round(float(np.percentile(hh[low], 98)), 3)})
    return out


def _crisp(cap, poses):
    cl = voxel_reduce(build_cloud(cap, step=max(4, cap.n // 900), poses=poses), 0.02)
    fl, _, _ = floor_ceiling(cl)
    yaw = manhattan_yaw(cl)
    uv, h = plan_coords(cl['xyz'], yaw, fl)
    nuv = rotate_dirs(cl['normal'][:, [0, 2]], yaw)
    return drift.crispness(uv, nuv, h, np.abs(cl['normal'][:, 1]) < 0.3), fl


def drift_stage(cap, poses, mode, log):
    """Plane-anchored pose-graph correction. 'auto' applies it only if walls get crisper
    (ablation numbers are always reported); 'off' = raw ARKit poses; 'on' = always apply.
    'off' still measures the raw ghosting so the ablation can quote it."""
    if mode == 'off':
        return poses, {'mode': 'off', 'applied': False}
    c0, fl = _crisp(cap, poses)
    new, info = drift.correct(cap, poses, fl, log=log)
    if new is poses:
        info.update(mode=mode, crispness_raw_mm=round(c0 * 1000, 2))
        return poses, info
    c1, _ = _crisp(cap, new)
    # accept when ghosting between revisits drops >= 5 % and walls do not get blurrier (> 5 %)
    better = info['revisit_error_corrected_mm'] < 0.95 * info['revisit_error_raw_mm'] and c1 < c0 * 1.05
    apply = mode == 'on' or better
    info.update(mode=mode, applied=bool(apply), crispness_raw_mm=round(c0 * 1000, 2), crispness_corrected_mm=round(c1 * 1000, 2))
    log(f"[drift] wall crispness {c0 * 1000:.1f} -> {c1 * 1000:.1f} mm, revisit ghosting {info['revisit_error_raw_mm']} -> "
        f"{info['revisit_error_corrected_mm']} mm : {'applied' if apply else 'rejected (no improvement)'}")
    return (new if apply else poses), info


def run(capture_dir, step=None, detect=True, poses=None, log=print, cache_dir=None, drift_mode='auto'):
    t0 = time.time()
    cap = StrayCapture(capture_dir) if isinstance(capture_dir, (str, bytes)) or hasattr(capture_dir, 'is_dir') else capture_dir
    tier = getattr(cap, 'tier', 'lidar')
    ERR.clear(); ERR.update(getattr(cap, 'error_model', LIDAR_ERRORS))
    step = step or (max(2, cap.n // 1600) if tier == 'lidar' else 1)
    log(f'[lidar] {cap.n} frames, step {step}')
    poses = cap.poses() if poses is None else poses
    poses, drift_info = drift_stage(cap, poses, drift_mode, log)
    cloud = voxel_reduce(build_cloud(cap, step=step, poses=poses), 0.02)
    log(f'[lidar] cloud {len(cloud["xyz"]):,} pts  ({time.time() - t0:.0f}s)')

    floor, ceil, ceil_support = floor_ceiling(cloud)
    yaw = manhattan_yaw(cloud)
    uv, h = plan_coords(cloud['xyz'], yaw, floor)
    nuv = rotate_dirs(cloud['normal'][:, [0, 2]], yaw)
    ny = cloud['normal'][:, 1]
    vertical = np.abs(ny) < 0.3
    grid = Grid(uv, res=0.02)
    top = np.percentile(h[vertical & (h > 0.1)], 97)
    band = (0.25, min(top, (ceil - floor - 0.15) if ceil else 9))
    cam_uv, _ = plan_coords(poses[:, :3, 3], yaw, floor)

    # evidence rasters
    Wu, Wv = oriented_walls(uv, h, nuv, vertical, grid, band)
    fm = (ny > 0.9) & (np.abs(h) < 0.04)
    r, c = grid.ij(uv[fm]); F = np.zeros(grid.shape, bool); F[r, c] = True
    # barrier: measured walls with door-sized gaps bridged (doors separate spaces, open plan doesn't)
    _, W, barrier, F2, walk = segment(grid, Wu, Wv, F, np.stack(grid.ij(cam_uv), 1), window_max=DOOR_MAX)
    floor_sigma = float(np.hypot(np.std(h[fm]) / np.sqrt(fm.sum()) * 3, 0.004))

    # cell complex on measured faces
    faces = cells.extract_faces(uv, h, nuv, vertical, band)
    carved = cells.carve_free(cloud, uv, h, cam_uv, grid)
    near = cv2.dilate((F2 | walk).astype(np.uint8), disk(int(0.6 / grid.res))).astype(bool)
    carved &= near   # rays through windows / open doors must not create space outside
    tops = (ny > 0.9) & (h > 0.08) & (h < 1.3)
    r2, c2 = grid.ij(uv[tops]); T = np.zeros(grid.shape, bool); T[r2, c2] = True
    E = cv2.morphologyEx((F2 | carved | (T & near)).astype(np.uint8), cv2.MORPH_CLOSE, disk(4)).astype(bool)
    crooms, cops, _ = cells.reconstruct(faces, grid, E, walk, barrier=barrier, inside_thr=0.5, door_max=DOOR_MAX)

    # which regions are spaces at all
    polys, mapping = spaces.clean_regions([cr['poly'] for cr in crooms], grid, F2, walk, log=log)
    polys = [P.simplify(0.005) for P in polys]

    wall_m = vertical & (h > 0.1)
    fam_u = wall_m & (np.abs(nuv[:, 1]) > 0.85)
    fam_v = wall_m & (np.abs(nuv[:, 0]) > 0.85)

    # openings between spaces (remapped after merges)
    openings = []
    for o in cops:
        rs = sorted({mapping[x] for x in o['rooms'] if mapping[x] >= 0})
        a0, a1 = o['span']
        if len(rs) != 2 or a1 - a0 < 0.45:
            continue
        mid = (a0 + a1) / 2
        pt = (mid, o['wall_coord']) if o['axis'] == 'u' else (o['wall_coord'], mid)
        rr, cc = grid.ij(np.array([pt]))
        kind = 'opening' if o['open_plan'] else 'door'
        sig = 0.008 if kind == 'door' else 0.03
        openings.append({'type': kind, 'room_index': rs, 'axis': o['axis'], 'wall_coord': round(float(o['wall_coord']), 4),
                         'span': [round(float(a0), 4), round(float(a1), 4)],
                         'width_m': {'value': round(float(a1 - a0), 4), 'ci95': round(Z95 * float(np.sqrt(2 * max(sig, ERR['face_sys']) ** 2 + (ERR['scale_rel'] * (a1 - a0)) ** 2)), 4)},
                         'walked_through': bool(walk[rr[0], cc[0]])})
    windows = find_windows(polys, faces, uv, h, fam_u, fam_v)

    # per-space measurements + structural features
    rooms = []
    for k, P in enumerate(polys):
        verts = np.array(P.exterior.coords[:-1])
        finfo = edge_faces(verts, faces)
        dims = room_dimensions(finfo)
        a_sig = P.area * np.hypot(dims['extent_u']['ci95_m'] / max(dims['extent_u']['value_m'], 0.1),
                                  dims['extent_v']['ci95_m'] / max(dims['extent_v']['value_m'], 0.1))
        mine = [o for o in openings if k in o['room_index']]
        nbrs = {x for o in mine for x in o['room_index'] if x != k}
        rooms.append({'id': f'R{k + 1}', 'polygon': np.round(verts, 4).tolist(),
                      'area_m2': {'value': round(P.area, 3), 'ci95': round(float(a_sig), 3)},
                      'dimensions': dims, 'ceiling_height': room_ceiling(uv, h, ny, P, floor_sigma),
                      'wall_edges': finfo,
                      'features': spaces.features(P, grid, F2, walk, uv, h, ny, mine, nbrs)})

    # fixtures: per-frame detections -> 3-D object instances -> the space containing each instance
    objects = []
    sightings = [dict() for _ in rooms]
    if detect:
        log('[lidar] fixtures (YOLO-World)...')
        dets = []
        for d in semantics.detect_objects(cap, poses, cache=f'{cache_dir}/detections.json' if cache_dir else None):
            duv, dh = plan_coords(np.array([d['xyz']]), yaw, floor)
            d['uv'] = duv[0].round(3).tolist(); d['height_m'] = round(float(dh[0]), 3)
            if d['conf'] > 0.3:
                dets.append(d)
        objects = semantics.object_instances(dets)
        for o in objects:
            k = spaces.room_of(polys, o['uv'])
            o['room'] = rooms[k]['id'] if k >= 0 else None
            if k >= 0:
                sightings[k][o['class']] = sightings[k].get(o['class'], 0) + 1
    for r, s in zip(rooms, sightings):
        r['type'], r['type_reason'] = spaces.classify(r['features'], s)
        r['fixtures'] = s
    spaces.name_spaces(rooms)
    labels = getattr(cap, 'room_of', None)
    if labels:   # photo tier: the user's room folders name the spaces their photos were taken in
        for k, r in enumerate(rooms):
            inside = [labels[i] for i, xy in enumerate(cam_uv) if polys[k].buffer(0.1).contains(Point(*xy))]
            if inside:
                best = max(set(inside), key=inside.count)
                r['folder_label'] = best
                r['folder_votes'] = {l: inside.count(l) for l in set(inside)}
                r['name'] = best
                unplaced = (getattr(cap, 'alignment', None) or {}).get('unplaced', []) if isinstance(getattr(cap, 'alignment', None), dict) else []
                r['position_known'] = best not in unplaced   # False: no doorway photo tied it to the other rooms

    for o in openings + windows:
        idx = o.pop('room_index')
        o['rooms'] = [rooms[x]['id'] for x in (idx if isinstance(idx, list) else [idx]) if x >= 0]
    openings = openings + windows
    for i, o in enumerate(openings):
        o['id'] = f'O{i + 1}'

    total = unary_union(polys)
    result = {
        'schema': 'brynz.plan/1.0', 'tier': tier, 'capture': str(getattr(cap, 'root', capture_dir)),
        'error_model': dict(ERR),
        'frame': {'yaw_rad': yaw, 'floor_y_world': floor, 'units': 'm',
                  'axes': 'u,v horizontal, aligned to dominant walls; h up from floor'},
        'global_ceiling': None if ceil is None else {'value_m': round(ceil - floor, 4), 'support': ceil_support},
        'drift': drift_info,
        'rooms': rooms, 'openings': openings,
        'adjacency': sorted({tuple(sorted(o['rooms'])) for o in openings if len(o['rooms']) == 2}),
        'footprint_m2': round(total.area, 3),
        'wall_faces': [f.to_dict() for f in faces],
        'objects': objects,
        'timing_s': round(time.time() - t0, 1),
        'debug': {'trajectory_uv': cam_uv[::10].round(3).tolist(),
                  'camera_uv': cam_uv.round(3).tolist() if tier != 'lidar' else None,
                  'camera_time_s': getattr(cap, 'times_s', None),
                  'camera_label': getattr(cap, 'room_of', None),
                  'rgb_alignment': getattr(cap, 'alignment', None)},
    }
    log(f'[lidar] {len(rooms)} spaces, {len(openings)} openings, {time.time() - t0:.0f}s')
    return result, {'grid': grid, 'uv': uv, 'h': h, 'nuv': nuv, 'ny': ny, 'W': W, 'barrier': barrier, 'F': F2,
                    'E': E, 'cloud': cloud, 'capture': cap, 'poses': poses, 'yaw': yaw, 'floor': floor, 'polys': polys,
                    'cam_uv': cam_uv, 'faces': faces}
