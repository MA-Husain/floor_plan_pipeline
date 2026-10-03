"""Per-surface damage: metric surface tiles -> defect classifier -> fused regions -> rules -> scope.

Why tiles and not a detector: water stains, mould, peeling and cracks are textures without object
boundaries, which open-vocabulary box detectors handle poorly (Grounding DINO missed most of our
sample defects). Instead every measured surface of the plan (wall faces, measured ceilings, floor)
is cut into fixed TILE_M x TILE_M cells in its own metric coordinates. Each cell is cropped from
the frames that see it best (close, frontal, fully in view) and classified by a linear probe on
CLIP ViT-L/14 image features, trained on the public BD3 building-defect dataset (CC-BY-4.0;
scripts/train_defect_probe.py; held-out accuracy reported in models/defect_probe.json).
Because a cell is defined on the surface, a positive cell has a known area and position: damage
extent is metric by construction, and a region is the union of adjacent positive cells.
"""
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point

from .cloud import backproject, to_world
from .geometry import plan_coords, rotate_dirs

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = str(ROOT / 'weights' / 'hf')
CLIP_ID = 'openai/clip-vit-large-patch14'
PROBE = ROOT / 'models' / 'defect_probe.npz'   # 21 KB, trained by scripts/train_defect_probe.py

TILE_M = 0.4            # surface cell size (m)
VIEWS_PER_TILE = 3      # best views classified per cell
MIN_TILE_PX = 64        # smallest crop side (rgb px) worth classifying
MAX_RANGE_M = 3.0
MIN_VIEWS = 2           # a cell is positive only if >= 2 views agree
P_DEFECT = 0.6          # mean defect probability needed
SURFACE_MIN = 0.5       # zero-shot 'bare wall / ceiling' probability a crop needs to be judged

CACHE_VERSION = 2

# BD3 class -> output contract class
CLASS_MAP = {'algae': 'mould', 'stain': 'water_stain', 'peeling': 'peeling_paint',
             'spalling': 'spalling', 'minor_crack': 'crack', 'major_crack': 'crack'}


class DefectClassifier:
    def __init__(self, device=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(CLIP_ID, cache_dir=HF_CACHE).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID, cache_dir=HF_CACHE)
        p = np.load(PROBE, allow_pickle=True)
        self.W, self.b, self.classes = p['coef'], p['intercept'], [str(c) for c in p['classes']]

    def embed(self, images, batch=32):
        out = []
        for i in range(0, len(images), batch):
            x = self.proc(images=images[i:i + batch], return_tensors='pt')['pixel_values'].to(self.device)
            with self.torch.no_grad():
                f = self.model.visual_projection(self.model.vision_model(pixel_values=x).pooler_output)
            out.append(self.torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 768))

    SURFACE_PROMPTS = ['a close-up photo of a wall', 'a close-up photo of a plastered wall with cracks',
                       'a close-up photo of a stained wall', 'a close-up photo of a ceiling']
    OTHER_PROMPTS = ['a photo of furniture', 'a photo of clothes', 'a photo of a bed', 'a photo of a window',
                     'a photo of a door', 'a photo of a lamp', 'a photo of a floor', 'a photo of household objects',
                     'a photo of a curtain or blind', 'a photo of a shelf']

    def surface_prob(self, feats):
        """Zero-shot gate: probability that a crop shows a bare wall / ceiling surface (used when
        there is no depth to tell us so, i.e. photo and video tiers)."""
        if not hasattr(self, '_txt'):
            tok = self.proc.tokenizer(self.SURFACE_PROMPTS + self.OTHER_PROMPTS, padding=True, return_tensors='pt').to(self.device)
            with self.torch.no_grad():
                t = self.model.text_projection(self.model.text_model(**tok).pooler_output)
            self._txt = self.torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
        z = 100 * feats @ self._txt.T
        z = np.exp(z - z.max(1, keepdims=True)); z /= z.sum(1, keepdims=True)
        return z[:, :len(self.SURFACE_PROMPTS)].sum(1)

    def probs(self, images, feats=None):
        if feats is not None:
            z = feats @ self.W.T + self.b
            z = np.exp(z - z.max(1, keepdims=True))
            return z / z.sum(1, keepdims=True)
        z = self.embed(images) @ self.W.T + self.b
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)


# ---------------------------------------------------------------- surface cells
def _surfaces(plan):
    """Surfaces to scan: wall faces and measured ceilings (per room)."""
    out = []
    for k, f in enumerate(plan['wall_faces']):
        out.append({'kind': 'wall', 'face': k, 'axis': f['axis'], 'coord': f['coord'], 'sign': f['normal_sign'],
                    'intervals': f['intervals']})
    for r in plan['rooms']:
        c = r.get('ceiling_height')
        if c:   # only ceilings that were actually measured
            out.append({'kind': 'ceiling', 'room': r['id'], 'h': c['value_m']})
    # floors are not scanned: the defect model is trained on wall / render surfaces only, and tile,
    # wood-grain and rug textures read as stains (measured: 10 of 12 false regions on a clean flat).
    return out


def _key(si, a, b):
    return si * 1000000 + (a + 500) * 1000 + (b + 500)


def _cell_keys(uv, h, nuv, ny, surfaces, rooms_poly, room_ids):
    """Per back-projected pixel: surface cell key (-1 = none). Vectorised per surface."""
    n = len(uv)
    key = np.full(n, -1, np.int64)
    meta = {}
    for si, s in enumerate(surfaces):
        if s['kind'] == 'wall':
            pp, al = (1, 0) if s['axis'] == 'u' else (0, 1)
            m = (np.abs(uv[:, pp] - s['coord']) < 0.05) & (np.sign(nuv[:, pp]) == s['sign']) & (np.abs(nuv[:, pp]) > 0.8) & (h > 0.02)
            within = np.zeros(n, bool)
            for a, b in s['intervals']:
                within |= (uv[:, al] > a - 0.1) & (uv[:, al] < b + 0.1)
            m &= within & (key < 0)
            i0, i1 = np.floor(uv[m, al] / TILE_M).astype(int), np.floor(h[m] / TILE_M).astype(int)
        elif s['kind'] == 'ceiling':
            m = (ny < -0.9) & (np.abs(h - s['h']) < 0.05) & (key < 0)
            if m.any():
                P = rooms_poly[room_ids.index(s['room'])].buffer(0.05)
                idx = np.nonzero(m)[0]
                inside = np.array([P.contains(Point(*p)) for p in uv[idx]])
                m[idx[~inside]] = False
            i0, i1 = np.floor(uv[m, 0] / TILE_M).astype(int), np.floor(uv[m, 1] / TILE_M).astype(int)
        else:
            m = (ny > 0.9) & (np.abs(h) < 0.03) & (key < 0)
            i0, i1 = np.floor(uv[m, 0] / TILE_M).astype(int), np.floor(uv[m, 1] / TILE_M).astype(int)
        if not m.any():
            continue
        k = _key(si, i0, i1).astype(np.int64)
        key[m] = k
        for kk, a, b in np.unique(np.c_[k, i0, i1], axis=0):
            meta[int(kk)] = (si, int(a), int(b))
    return key, meta


def plan_views(capture, poses, plan, ctx, every_s=0.5, log=print):
    """Geometry pass (depth only): for every surface cell, the best frames and the crop box."""
    yaw, floor, polys = ctx['yaw'], ctx['floor'], ctx['polys']
    room_ids = [r['id'] for r in plan['rooms']]
    surfaces = _surfaces(plan)
    W, H = capture.rgb_size
    sx, sy = W / capture.depth_size[0], H / capture.depth_size[1]
    step = max(1, int(round(capture.fps * every_s)))
    cand, meta_all = {}, {}
    for i in range(0, capture.n, step):
        f = capture.frame(i, poses[i])
        P, n, pix, dep = backproject(f, max_depth=MAX_RANGE_M, min_conf=2)
        if len(P) < 500:
            continue
        Pw, nw = to_world(P, n, f.T_wc)
        uv, h = plan_coords(Pw, yaw, floor)
        nuv = rotate_dirs(nw[:, [0, 2]], yaw)
        key, meta = _cell_keys(uv, h, nuv, nw[:, 1], surfaces, polys, room_ids)
        meta_all.update(meta)
        ok = key >= 0
        if not ok.any():
            continue
        cosv = np.abs((P / np.linalg.norm(P, axis=1, keepdims=True) * n).sum(1))
        order = np.argsort(key[ok], kind='stable')
        ks, px, dd, cv = key[ok][order], pix[ok][order], dep[ok][order], cosv[ok][order]
        uniq, start = np.unique(ks, return_index=True)
        ends = np.r_[start[1:], len(ks)]
        for k, a, b in zip(uniq, start, ends):
            q = px[a:b]
            x0, y0 = q.min(0); x1, y1 = q.max(0)
            if x0 <= 1 or y0 <= 1 or x1 >= capture.depth_size[0] - 2 or y1 >= capture.depth_size[1] - 2:
                continue   # cell cut by the image border: not fully in view
            box = [int(x0 * sx), int(y0 * sy), int((x1 + 1) * sx), int((y1 + 1) * sy)]
            if min(box[2] - box[0], box[3] - box[1]) < MIN_TILE_PX:
                continue
            d, c = float(np.median(dd[a:b])), float(np.median(cv[a:b]))
            full = TILE_M ** 2 * c / (d ** 2 / (f.K[0, 0] * f.K[1, 1]))   # expected px for an unoccluded cell
            if (b - a) < 0.5 * full:
                continue   # mostly occluded (furniture in front) or a partial edge cell
            cand.setdefault(int(k), []).append((c / max(d, 0.3), i, box, d))   # frontal and close first
    views = {k: sorted(v, reverse=True)[:VIEWS_PER_TILE] for k, v in cand.items()}
    log(f'[damage] {len(views)} surface cells seen fully; {sum(len(v) for v in views.values())} crops to classify')
    return views, meta_all, surfaces


def classify_cells(capture, views, clf, log=print):
    by_frame = {}
    for k, vs in views.items():
        for q, i, box, rng in vs:
            by_frame.setdefault(i, []).append((k, box))
    crops, keys = [], []
    for i, img in capture.rgb_frames(sorted(by_frame)):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for k, (x0, y0, x1, y1) in by_frame[i]:
            crops.append(cv2.resize(rgb[y0:y1, x0:x1], (224, 224), interpolation=cv2.INTER_AREA))
            keys.append(k)
    log(f'[damage] classifying {len(crops)} crops from {len(by_frame)} frames...')
    f = clf.embed(crops)
    pr = clf.probs(None, feats=f) if crops else np.zeros((0, len(clf.classes)))
    sp = clf.surface_prob(f) if crops else np.zeros(0)
    out = {}
    for k, p, g in zip(keys, pr, sp):
        out.setdefault(k, []).append(np.r_[p, g])   # last column: bare-surface probability
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- cells -> regions
def positive_cells(cell_probs, classes):
    plain = classes.index('plain')
    pos = {}
    for k, P in cell_probs.items():
        # views where the crop is mostly an object (socket, switch, cable, picture) are dropped:
        # depth says the cell lies on the wall plane, but flat fixtures lie on it too
        P = P[P[:, -1] >= SURFACE_MIN][:, :-1]
        if len(P) < MIN_VIEWS:
            continue
        m = P.mean(0)
        if 1 - m[plain] < P_DEFECT or (P.argmax(1) != plain).sum() < MIN_VIEWS:
            continue
        cls = classes[int(np.argmax(np.where(np.arange(len(m)) == plain, -1, m)))]
        pos[k] = (CLASS_MAP[cls], cls, float(1 - m[plain]), len(P))
    return pos


def regions_from_cells(cell_probs, meta, surfaces, classes, plan, ctx):
    pos = positive_cells(cell_probs, classes)
    seen, regions = set(), []
    for k in pos:
        if k in seen:
            continue
        si = meta[k][0]
        cls = pos[k][0]
        comp, stack = [], [k]
        seen.add(k)
        while stack:     # 8-connected component of same-class cells on one surface
            c = stack.pop(); comp.append(c)
            _, ca, cb = meta[c]
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    nk = _key(si, ca + da, cb + db)
                    if nk in pos and nk not in seen and pos[nk][0] == cls:
                        seen.add(nk); stack.append(nk)
        regions.append(_region(comp, pos, meta, surfaces[si], plan, ctx))
    regions.sort(key=lambda r: -r['area_m2']['value'])
    for j, r in enumerate(regions):
        r['id'] = f'D{j + 1}'
    return regions


def _region(comp, pos, meta, s, plan, ctx):
    ij = np.array([meta[c][1:] for c in comp])
    si = meta[comp[0]][0]
    n = len(comp)
    edge = sum(1 for c in comp for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1))
               if _key(si, meta[c][1] + da, meta[c][2] + db) not in pos)
    area = n * TILE_M ** 2
    # a positive cell may be only partly damaged: each exposed cell side carries a quarter of
    # +-50% of the cell area; plus 10% for classifier boundary uncertainty
    ci = 0.5 * edge / 4 * TILE_M ** 2 + 0.1 * area
    lo, hi = ij.min(0) * TILE_M, (ij.max(0) + 1) * TILE_M
    sev = [pos[c][1] for c in comp]
    r = {'class': pos[comp[0]][0], 'confidence': round(float(np.mean([pos[c][2] for c in comp])), 3),
         'views': int(np.sum([pos[c][3] for c in comp])), 'cells': n,
         'area_m2': {'value': round(area, 3), 'ci95': round(ci, 3)},
         'extent_m': [round(float(hi[0] - lo[0]), 2), round(float(hi[1] - lo[1]), 2)]}
    rooms, polys = plan['rooms'], ctx['polys']
    if s['kind'] == 'wall':
        r['surface'] = {'kind': 'wall', 'face': s['face'], 'axis': s['axis'], 'coord': s['coord'], 'normal_sign': s['sign']}
        r['span_along_m'] = [round(float(lo[0]), 2), round(float(hi[0]), 2)]
        r['height_m'] = [round(float(lo[1]), 2), round(float(hi[1]), 2)]
        mid = (lo[0] + hi[0]) / 2
        side = (mid, s['coord'] + 0.15 * s['sign']) if s['axis'] == 'u' else (s['coord'] + 0.15 * s['sign'], mid)
        r['surface']['room'] = _room_at(polys, rooms, side)
        r['height_above_floor_m'] = r['height_m'][0]
    else:
        c = ((lo + hi) / 2).tolist()
        r['surface'] = {'kind': s['kind'], 'room': s.get('room') or _room_at(polys, rooms, c)}
        r['center_uv'] = [round(c[0], 2), round(c[1], 2)]
        r['height_above_floor_m'] = s.get('h', 0.0)
    if r['class'] == 'crack':
        r['severity'] = 'major' if sev.count('major_crack') > sev.count('minor_crack') else 'minor'
    return r


def _room_at(polys, rooms, xy):
    p = Point(*xy)
    for P, r in zip(polys, rooms):
        if P.buffer(0.1).contains(p):
            return r['id']
    return None


def scan(capture, poses, plan, ctx, log=print, cache=None, clf=None):
    """Full damage pass. Cell probabilities are cached so re-runs replay the model output."""
    views, meta, surfaces = plan_views(capture, poses, plan, ctx, log=log)
    cache = Path(cache) if cache else None
    probs = None
    if cache and cache.exists():
        c = json.loads(cache.read_text())
        if c.get('tile_m') == TILE_M and c.get('version') == CACHE_VERSION:
            probs = {int(k): np.array(v) for k, v in c['probs'].items()}
            classes = c['classes']
            log(f'[damage] replaying {len(probs)} cached cell classifications')
    if probs is None:
        clf = clf or DefectClassifier()
        probs = classify_cells(capture, views, clf, log=log)
        classes = clf.classes
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({'tile_m': TILE_M, 'version': CACHE_VERSION, 'classes': classes,
                                         'probs': {str(k): v.round(4).tolist() for k, v in probs.items()}}))
    probs = {k: v for k, v in probs.items() if k in meta and k in views}   # cache may hold cells of an older surface set
    regions = regions_from_cells(probs, meta, surfaces, classes, plan, ctx)
    log(f'[damage] {len(probs)} cells classified -> {len(regions)} damage regions')
    inspected = {'cells_classified': len(probs), 'tile_m': TILE_M,
                 'surface_area_inspected_m2': round(len(probs) * TILE_M ** 2, 1)}
    return regions, inspected, (probs, meta, surfaces, classes)


# ---------------------------------------------------------------- concealed-damage rules
WET = {'Bathroom', 'WC', 'Kitchen', 'Utility', 'Living / Kitchen'}
MOIST = {'water_stain', 'mould'}


def concealed_flags(regions, plan):
    rooms = {r['id']: r for r in plan['rooms']}
    adj = {}
    for a, b in plan.get('adjacency', []):
        adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    windows = [o for o in plan['openings'] if o['type'] == 'window']
    flags = []
    for r in regions:
        s, cls = r['surface'], r['class']
        moist = cls in MOIST
        room = rooms.get(s.get('room'))
        if s['kind'] == 'ceiling' and moist:
            flags.append(_flag(r, 'R1-ceiling-moisture', 'Moisture on ceiling: probable leak from plumbing, roof or the unit above; inspect the void above.'))
        if s['kind'] == 'wall' and moist and r['height_above_floor_m'] < 0.4:
            flags.append(_flag(r, 'R2-rising-damp', 'Moisture starting within 0.4 m of the floor: rising damp or leaking floor-level pipework; moisture-meter the wall base.'))
        if s['kind'] == 'wall' and moist and room is not None:
            wet_nbr = [x for x in adj.get(room['id'], ()) if rooms[x]['type'] in WET]
            if room['type'] in WET or wet_nbr:
                flags.append(_flag(r, 'R3-wet-room-wall', f"Moisture on a wall of / next to a wet room ({room['type']}{', ' + ', '.join(wet_nbr) if wet_nbr else ''}): concealed supply or waste leak likely behind the finish."))
        if cls == 'mould' and s['kind'] == 'wall' and _near_window(r, windows):
            flags.append(_flag(r, 'R4-condensation', 'Mould beside a window: condensation or failed window seal; check ventilation and seal.'))
        if cls == 'crack' and (max(r['extent_m']) > 0.5 or r.get('severity') == 'major'):
            flags.append(_flag(r, 'R5-structural-crack', f"Crack {max(r['extent_m']):.1f} m long ({r.get('severity')}): possible movement; monitor width, inspect lintel / framing."))
        if cls == 'spalling':
            flags.append(_flag(r, 'R6-spalling', 'Spalling: moisture or corroding reinforcement behind the surface; sound the area and check for exposed steel.'))
    return flags


def _near_window(r, windows, d=0.6):
    s = r['surface']
    for w in windows:
        if w['axis'] != s['axis'] or abs(w['wall_coord'] - s['coord']) > 0.1:
            continue
        a0, a1 = w['span']; b0, b1 = r['span_along_m']
        if b0 < a1 + d and b1 > a0 - d:
            return True
    return False


def _flag(r, rule, text):
    return {'damage_id': r['id'], 'rule': rule, 'finding': text, 'surface': r['surface']}


# ---------------------------------------------------------------- scope line items
def scope_items(regions, flags, plan):
    items = []
    for r in regions:
        s = r['surface']
        tgt = f"{s.get('room') or '?'} {s['kind']}"
        a = r['area_m2']['value']
        if r['class'] == 'water_stain':
            items.append(_item(r, tgt, 'Stain-block primer and repaint', round(max(a * 1.5, 1.0), 2), 'm2'))
        elif r['class'] == 'mould':
            items.append(_item(r, tgt, 'Fungicidal wash, stain-block, anti-mould paint', round(max(a * 2.0, 1.0), 2), 'm2'))
        elif r['class'] == 'peeling_paint':
            items.append(_item(r, tgt, 'Scrape, fill, prime and repaint', round(max(a * 1.3, 0.5), 2), 'm2'))
        elif r['class'] == 'spalling':
            items.append(_item(r, tgt, 'Break out loose render, patch repair and repaint', round(max(a * 1.3, 0.5), 2), 'm2'))
        elif r['class'] == 'crack':
            items.append(_item(r, tgt, 'Rake out and fill crack, repaint surface', round(max(r['extent_m']), 2), 'm'))
    for f in flags:
        items.append({'damage_id': f['damage_id'], 'target': f"{f['surface'].get('room') or '?'} {f['surface']['kind']}",
                      'action': f"Investigate ({f['rule']})", 'qty': 1, 'unit': 'ea'})
    for i, it in enumerate(items):
        it['id'] = f'S{i + 1}'
    return items


def _item(r, tgt, action, qty, unit):
    return {'damage_id': r['id'], 'target': tgt, 'action': action, 'qty': qty, 'unit': unit}
