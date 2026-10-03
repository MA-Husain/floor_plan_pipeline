"""Manhattan cell-complex floor-plan reconstruction.

1. Wall faces: 1-D peaks of the perpendicular coordinate of wall-family points, split by normal
   sign (a wall seen from both sides gives two faces -> thickness). Each face carries its solid
   intervals along the wall (columns whose height coverage is high enough).
2. Face coordinates become cut lines; they partition the plan into rectangular cells.
3. Cells are labelled inside from observed floor + walked trajectory (summed-area tables).
4. Neighbouring inside cells are merged into rooms unless the shared edge is wall, or a door gap
   (uncovered run <= door_max bounded by wall on both ends along the same line).
Room outlines are unions of cells, so every boundary lies exactly on a measured face.
"""
import numpy as np
from scipy import ndimage
from shapely.geometry import box
from shapely.ops import unary_union

from . import priors


class Face:
    __slots__ = ('axis', 'coord', 'sign', 'intervals', 'std', 'n', 'length')

    def __init__(self, axis, coord, sign, intervals, std, n):
        self.axis, self.coord, self.sign, self.intervals, self.std, self.n = axis, coord, sign, intervals, std, n
        self.length = sum(b - a for a, b in intervals)

    def to_dict(self):
        return {'axis': self.axis, 'coord': round(self.coord, 4), 'normal_sign': int(self.sign),
                'intervals': [[round(a, 3), round(b, 3)] for a, b in self.intervals],
                'std_m': round(self.std, 4), 'n': int(self.n), 'solid_length_m': round(self.length, 3)}


def _intervals(along, hh, band, col=0.05, hbin=0.1, min_cov=priors.WALL_MIN_HEIGHT_COVERAGE, close=0.12, min_len=0.25):
    lo, hi = band
    nb = max(1, int(np.ceil((hi - lo) / hbin)))
    if len(along) == 0:
        return []
    a0 = along.min()
    ci = ((along - a0) / col).astype(int)
    hb = ((hh - lo) / hbin).astype(int).clip(0, nb - 1)
    key = np.unique(ci * nb + hb)
    cov = np.bincount(key // nb, minlength=ci.max() + 1) / nb
    solid = cov >= min_cov
    solid = ndimage.binary_closing(solid, np.ones(int(close / col) * 2 + 1)) | solid
    lab, n = ndimage.label(solid)
    out = []
    for s in ndimage.find_objects(lab):
        a, b = a0 + s[0].start * col, a0 + s[0].stop * col
        if b - a >= min_len:
            out.append((a, b))
    return out


def extract_faces(uv, h, nuv, mask, band, min_solid=priors.WALL_MIN_SOLID_M):
    """Returns list[Face]. axis 'u' = wall runs along u (coord is v); 'v' = runs along v (coord is u)."""
    faces = []
    for axis, pp, al in (('u', 1, 0), ('v', 0, 1)):
        for sign in (1, -1):
            m = mask & (nuv[:, pp] * sign > 0.85) & (h > band[0]) & (h < band[1])
            if m.sum() < 100:
                continue
            c = uv[m, pp]; a = uv[m, al]; hh = h[m]
            bins = np.arange(c.min() - 0.02, c.max() + 0.03, 0.01)
            hist, e = np.histogram(c, bins=bins)
            hs = ndimage.gaussian_filter1d(hist.astype(float), 1.2)
            peaks = np.nonzero((hs > ndimage.maximum_filter1d(hs, 9) - 1e-9) & (hs > 30))[0]
            for p in peaks:
                pc = e[p] + 0.005
                sel = np.abs(c - pc) < 0.03
                if sel.sum() < 60:
                    continue
                ivs = _intervals(a[sel], hh[sel], band)
                if sum(b - a_ for a_, b in ivs) < min_solid:
                    continue
                cc = c[sel]
                med = np.median(cc)
                inl = np.abs(cc - med) < 0.02
                faces.append(Face(axis, float(np.mean(cc[inl])), sign, ivs, float(np.std(cc[inl])), int(inl.sum())))
    return faces


def _cuts(faces, axis, merge=0.035):
    cs = sorted((f.coord, f.length) for f in faces if f.axis == axis)
    out = []
    for c, L in cs:
        if out and c - out[-1][0] < merge:
            pc, pL = out[-1]
            out[-1] = ((pc * pL + c * L) / (pL + L), pL + L)
        else:
            out.append((c, L))
    return np.array([c for c, _ in out])


def _covered(faces_on_line, a0, a1):
    """Fraction of [a0,a1] covered by solid intervals of the given faces."""
    if a1 <= a0:
        return 0.0
    ivs = sorted(iv for f in faces_on_line for iv in f.intervals)
    tot, cur = 0.0, a0
    for a, b in ivs:
        a, b = max(a, cur), min(b, a1)
        if b > a:
            tot += b - a
            cur = b
    return tot / (a1 - a0)


def _merged_ivs(faces_on_line):
    ivs = sorted(iv for f in faces_on_line for iv in f.intervals)
    out = []
    for a, b in ivs:
        if out and a <= out[-1][1] + 0.02:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


class SAT:
    """Summed-area table over a boolean raster in plan coordinates."""
    def __init__(self, mask, grid):
        self.g = grid
        self.S = np.pad(mask.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def frac(self, u0, u1, v0, v1):
        g = self.g
        c0 = int(np.clip(np.floor((u0 - g.origin[0]) / g.res), 0, g.shape[1]))
        c1 = int(np.clip(np.ceil((u1 - g.origin[0]) / g.res), 0, g.shape[1]))
        r0 = int(np.clip(np.floor((v0 - g.origin[1]) / g.res), 0, g.shape[0]))
        r1 = int(np.clip(np.ceil((v1 - g.origin[1]) / g.res), 0, g.shape[0]))
        if c1 <= c0 or r1 <= r0:
            return 0.0
        s = self.S[r1, c1] - self.S[r0, c1] - self.S[r1, c0] + self.S[r0, c0]
        return s / ((r1 - r0) * (c1 - c0))


def reconstruct(faces, grid, floor_mask, walk_mask, door_max=1.25, inside_thr=0.25, min_room=1.0, door_half=0.45, core_min_m2=0.6, barrier=None,
                neck_max=priors.NECK_MAX_M, neck_min_side=priors.NECK_MIN_SIDE_M2):
    us = _cuts(faces, 'v')   # vertical lines at u = c
    vs = _cuts(faces, 'u')   # horizontal lines at v = c
    # bound the complex by the observed extent
    ys, xs = np.nonzero(floor_mask | walk_mask)
    uv_lo = grid.to_uv(ys.min(), xs.min()); uv_hi = grid.to_uv(ys.max(), xs.max())
    us = np.unique(np.concatenate([[uv_lo[0] - 0.05], us[(us > uv_lo[0]) & (us < uv_hi[0])], [uv_hi[0] + 0.05]]))
    vs = np.unique(np.concatenate([[uv_lo[1] - 0.05], vs[(vs > uv_lo[1]) & (vs < uv_hi[1])], [uv_hi[1] + 0.05]]))
    nu, nv = len(us) - 1, len(vs) - 1

    satF = SAT(floor_mask | walk_mask, grid)
    satW = SAT(walk_mask, grid)
    fr = np.zeros((nv, nu)); wk = np.zeros((nv, nu))
    for j in range(nv):
        for i in range(nu):
            fr[j, i] = satF.frac(us[i], us[i + 1], vs[j], vs[j + 1])
            wk[j, i] = satW.frac(us[i], us[i + 1], vs[j], vs[j + 1])
    inside = fr > inside_thr

    # faces grouped by cut line
    def faces_at(axis, c, tol=0.04):
        return [f for f in faces if f.axis == axis and abs(f.coord - c) < tol]

    line_ivs_v = {i: _merged_ivs(faces_at('v', us[i])) for i in range(1, nu)}
    line_ivs_u = {j: _merged_ivs(faces_at('u', vs[j])) for j in range(1, nv)}
    # perpendicular faces crossing/touching a line act as door jambs (doors beside corners)
    def jambs(axis_perp, c):
        out = []
        for f in faces:
            if f.axis != axis_perp:
                continue
            if f.length >= 0.8 and any(a - 0.1 <= c <= b + 0.1 for a, b in f.intervals):
                out.append((f.coord - 0.005, f.coord + 0.005))
        return out
    jamb_v = {i: sorted(line_ivs_v[i] + jambs('u', us[i])) for i in range(1, nu)}
    jamb_u = {j: sorted(line_ivs_u[j] + jambs('v', vs[j])) for j in range(1, nv)}

    def edge_kind(ivs, a0, a1, jb):
        """'wall', 'door' (gap bounded by wall/jamb both sides, short), or 'open'."""
        cov = _covered_ivs(ivs, a0, a1)
        wall_ivs = ivs
        ivs = jb
        if cov > 0.5:
            return 'wall', None
        # find the maximal uncovered run containing the midpoint
        mid = (a0 + a1) / 2
        left = max([b for a, b in ivs if b <= mid + 1e-6] + [-np.inf])
        right = min([a for a, b in ivs if a >= mid - 1e-6] + [np.inf])
        if not (np.isfinite(left) and np.isfinite(right) and priors.DOOR_MIN_M <= right - left <= door_max):
            return 'open', None
        # at least one jamb must be a real collinear wall run, and the line must carry wall
        coll = [iv for iv in wall_ivs if iv[1] - iv[0] > 0.02]
        near = any(abs(b - left) < 0.03 or abs(a - right) < 0.03 for a, b in coll)
        if near and sum(b - a for a, b in coll) >= 0.5:
            return 'door', (left, right)
        return 'open', None

    # vertical edges: between cell (j,i-1) and (j,i) at u=us[i]
    E_v = {}
    for i in range(1, nu):
        for j in range(nv):
            E_v[(j, i)] = edge_kind(line_ivs_v[i], vs[j], vs[j + 1], jamb_v[i])
    E_u = {}
    for j in range(1, nv):
        for i in range(nu):
            E_u[(j, i)] = edge_kind(line_ivs_u[j], us[i], us[i + 1], jamb_u[j])

    # fill unobserved cells enclosed in a room (furniture footprints): a non-inside cell joins if
    # >=3 of its open-edge neighbours are inside
    for _ in range(4):
        add = []
        for j in range(nv):
            for i in range(nu):
                if inside[j, i]:
                    continue
                nb = 0; tot = 0
                for dj, di, ek in ((0, -1, E_v.get((j, i))), (0, 1, E_v.get((j, i + 1))), (-1, 0, E_u.get((j, i))), (1, 0, E_u.get((j + 1, i)))):
                    jj, ii = j + dj, i + di
                    if ek is None or not (0 <= jj < nv and 0 <= ii < nu):
                        continue
                    if ek[0] == 'open':
                        tot += 1; nb += inside[jj, ii]
                if nb >= 3 or (nb >= 2 and fr[j, i] > 0.08):
                    add.append((j, i))
        if not add:
            break
        for c in add:
            inside[c] = True

    # ---- room grouping: morphological segmentation on the inside raster with walls burned in.
    # Eroding by ~door half-width removes door passages; surviving cores are rooms; watershed
    # grows them back; each cell takes the majority label.
    import cv2
    res = grid.res
    def rc(u, v):
        return (int(round((u - grid.origin[0]) / res)), int(round((v - grid.origin[1]) / res)))
    free = np.zeros(grid.shape, np.uint8)
    for j in range(nv):
        for i in range(nu):
            if inside[j, i]:
                x0, y0 = rc(us[i], vs[j]); x1, y1 = rc(us[i + 1], vs[j + 1])
                free[y0:y1 + 1, x0:x1 + 1] = 1
    for (j, i), (k, _) in E_v.items():
        if k == 'wall':
            x, y0 = rc(us[i], vs[j]); _, y1 = rc(us[i], vs[j + 1])
            free[y0:y1 + 1, max(x - 2, 0):x + 3] = 0
    for (j, i), (k, _) in E_u.items():
        if k == 'wall':
            x0, y = rc(us[i], vs[j]); x1, _ = rc(us[i + 1], vs[j])
            free[max(y - 2, 0):y + 3, x0:x1 + 1] = 0
    if barrier is not None:
        # external barrier (bridged walls with doors closed) defines room connectivity directly
        free = free & (~barrier).astype(np.uint8)
        door_half = 0.0
    dist = cv2.distanceTransform(free, cv2.DIST_L2, 5) * res
    cores = (dist > door_half).astype(np.uint8)
    ncore, core_lab = cv2.connectedComponents(cores, connectivity=4)
    areas = np.bincount(core_lab.ravel(), minlength=ncore) * res * res
    keep = np.zeros(ncore, bool); keep[1:] = areas[1:] >= core_min_m2
    core_lab = np.where(keep[core_lab], core_lab, 0)
    markers = core_lab.astype(np.int32).copy()
    markers[free == 0] = -1
    img = np.dstack([(255 - np.clip(dist / max(dist.max(), 1e-6) * 255, 0, 255)).astype(np.uint8)] * 3)
    markers[markers == -1] = 0
    bg = (free == 0)
    markers2 = markers.copy(); markers2[bg] = ncore + 1
    cv2.watershed(img, markers2)
    lab = np.where(bg, 0, markers2).clip(0)
    lab[lab == ncore + 1] = 0

    cell_lab = np.zeros((nv, nu), int)
    for j in range(nv):
        for i in range(nu):
            if not inside[j, i]:
                continue
            x0, y0 = rc(us[i], vs[j]); x1, y1 = rc(us[i + 1], vs[j + 1])
            patch = lab[y0:y1 + 1, x0:x1 + 1].ravel()
            patch = patch[patch > 0]
            cell_lab[j, i] = np.bincount(patch).argmax() if patch.size else 0
    # unlabeled inside cells (thin passages) join their open-edge neighbour with most shared length
    for _ in range(3):
        for j in range(nv):
            for i in range(nu):
                if inside[j, i] and cell_lab[j, i] == 0:
                    share = {}
                    for jj, ii, key, E, L in ((j, i - 1, (j, i), E_v, vs[j + 1] - vs[j]), (j, i + 1, (j, i + 1), E_v, vs[j + 1] - vs[j]),
                                              (j - 1, i, (j, i), E_u, us[i + 1] - us[i]), (j + 1, i, (j + 1, i), E_u, us[i + 1] - us[i])):
                        if 0 <= jj < nv and 0 <= ii < nu and cell_lab[jj, ii] > 0 and E.get(key, ('wall',))[0] != 'wall':
                            share[cell_lab[jj, ii]] = share.get(cell_lab[jj, ii], 0) + L
                    if share:
                        cell_lab[j, i] = max(share, key=share.get)
    groups = {}
    for j in range(nv):
        for i in range(nu):
            if inside[j, i] and cell_lab[j, i] > 0:
                groups.setdefault(cell_lab[j, i], []).append((j, i))
    groups = split_necks(list(groups.values()), us, vs, E_v, E_u, neck_max=neck_max, min_side=neck_min_side)
    groups = [g2 for g in groups for g2 in split_by_cores(g, us, vs, E_v, E_u, max_boundary=2.2, min_side=neck_min_side)]

    rooms = []
    cell_room = -np.ones((nv, nu), int)
    for cells in groups:
        P = unary_union([box(us[i], vs[j], us[i + 1], vs[j + 1]) for j, i in cells]).buffer(1e-6, join_style='mitre').buffer(-1e-6, join_style='mitre')
        if P.area < min_room:
            continue
        for c in cells:
            cell_room[c] = len(rooms)
        rooms.append({'poly': P, 'cells': cells, 'walk': float(np.mean([wk[c] for c in cells]))})

    # openings: maximal runs of non-wall edges between two different rooms (or room/outside
    # for door-kind edges) along one cut line
    runs = []
    for i in range(1, nu):
        cur = None
        for j in range(nv):
            k = E_v[(j, i)][0]
            ra, rb = cell_room[j, i - 1], cell_room[j, i]
            ok = k != 'wall' and ra != rb and ra >= 0 and rb >= 0
            key = tuple(sorted((ra, rb)))
            if ok and cur and cur['key'] == key and abs(cur['span'][1] - vs[j]) < 1e-6:
                cur['span'][1] = vs[j + 1]
            elif ok:
                cur = {'axis': 'v', 'wall_coord': us[i], 'span': [vs[j], vs[j + 1]], 'key': key}; runs.append(cur)
            else:
                cur = None
    for j in range(1, nv):
        cur = None
        for i in range(nu):
            k = E_u[(j, i)][0]
            ra, rb = cell_room[j - 1, i], cell_room[j, i]
            ok = k != 'wall' and ra != rb and ra >= 0 and rb >= 0
            key = tuple(sorted((ra, rb)))
            if ok and cur and cur['key'] == key and abs(cur['span'][1] - us[i]) < 1e-6:
                cur['span'][1] = us[i + 1]
            elif ok:
                cur = {'axis': 'u', 'wall_coord': vs[j], 'span': [us[i], us[i + 1]], 'key': key}; runs.append(cur)
            else:
                cur = None
    openings = {}
    for r in runs:
        r['rooms'] = set(r.pop('key'))
        # tighten span to the gap between solid wall on this line
        ivs = (line_ivs_v if r['axis'] == 'v' else line_ivs_u)
        idx = (np.argmin(np.abs(us - r['wall_coord'])) if r['axis'] == 'v' else np.argmin(np.abs(vs - r['wall_coord'])))
        line = ivs.get(idx, [])
        a0, a1 = r['span']
        for a, b in line:
            if a <= a0 + 0.02 < b:
                a0 = b
            if a < a1 - 0.02 <= b:
                a1 = a
        r['span'] = [a0, a1]
        r['open_plan'] = (a1 - a0) > door_max
        openings[(r['axis'], round(r['wall_coord'], 2), round(a0, 2))] = r
    return rooms, list(openings.values()), {'us': us, 'vs': vs, 'inside': inside, 'fr': fr, 'cell_room': cell_room, 'E_v': E_v, 'E_u': E_u}


def _covered_ivs(ivs, a0, a1):
    if a1 <= a0:
        return 0.0
    tot = 0.0
    for a, b in ivs:
        tot += max(0.0, min(b, a1) - max(a, a0))
    return tot / (a1 - a0)


def carve_free(cloud, uv, h, cam_uv, grid, band=(0.4, 1.6), max_rays=250000, seed=0):
    """2-D free-space carving: every cell on the segment camera->hit (hits in a mid-height band)
    was seen through, so it is empty interior space (or seen through glass - limited by range)."""
    import cv2
    sel = np.nonzero((h > band[0]) & (h < band[1]))[0]
    rng = np.random.default_rng(seed)
    if len(sel) > max_rays:
        sel = rng.choice(sel, max_rays, replace=False)
    fr = cloud['frame'][sel]
    a = ((cam_uv[fr] - grid.origin) / grid.res).astype(np.int32)
    b = ((uv[sel] - grid.origin) / grid.res).astype(np.int32)
    # stop 6 cm short of the hit so the surface itself is not carved
    d = (b - a).astype(float); L = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    b = (b - d / L * 3).astype(np.int32)
    img = np.zeros(grid.shape, np.uint8)
    for (x0, y0), (x1, y1) in zip(a, b):
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), 1, 1)
    return img.astype(bool)


def split_necks(groups, us, vs, E_v, E_u, neck_max=1.6, min_side=2.5):
    """Split open-plan regions at narrow passages (necks) with no door leaf.

    For each grid line crossing a region, the cross-section is the total length of non-wall
    edges joining region cells across that line. A line whose cross-section is <= neck_max and
    whose removal leaves >= min_side m2 on both sides is a neck; the narrowest neck is cut first,
    then both halves are re-examined."""
    def area(c):
        j, i = c
        return (us[i + 1] - us[i]) * (vs[j + 1] - vs[j])

    def components(cells, cut):
        cs = set(cells); seen = set(); comps = []
        for c in cells:
            if c in seen:
                continue
            stack, comp = [c], []
            seen.add(c)
            while stack:
                j, i = stack.pop(); comp.append((j, i))
                for jj, ii, key, E in ((j, i - 1, (j, i), E_v), (j, i + 1, (j, i + 1), E_v),
                                       (j - 1, i, (j, i), E_u), (j + 1, i, (j + 1, i), E_u)):
                    n = (jj, ii)
                    if n in cs and n not in seen and E.get(key, ('wall',))[0] != 'wall' and (E is E_v, key) not in cut:
                        seen.add(n); stack.append(n)
            comps.append(comp)
        return comps

    out, todo = [], [list(g) for g in groups]
    while todo:
        cells = todo.pop()
        cs = set(cells)
        best = None
        for vert, E, lines in ((True, E_v, range(1, len(us) - 1)), (False, E_u, range(1, len(vs) - 1))):
            for L in lines:
                cut = set(); length = 0.0
                for (j, i) in cells:
                    if vert and i == L and (j, i - 1) in cs and E[(j, i)][0] != 'wall':
                        cut.add((True, (j, i))); length += vs[j + 1] - vs[j]
                    if not vert and j == L and (j - 1, i) in cs and E[(j, i)][0] != 'wall':
                        cut.add((False, (j, i))); length += us[i + 1] - us[i]
                if not cut or length > neck_max or (best and length >= best[0]):
                    continue
                comps = components(cells, cut)
                if len(comps) < 2:
                    continue
                areas = sorted(sum(area(c) for c in comp) for comp in comps)
                if areas[-2] >= min_side:
                    best = (length, comps)
        if best is None:
            out.append(cells)
        else:
            todo.extend(best[1])
    return out


def split_by_cores(cells, us, vs, E_v, E_u, max_boundary=2.2, min_side=3.0, big=15.0, res=0.05):
    """Split a large region whose parts meet through several short passages: cores are the parts
    wider than 2r (distance transform), grown back by watershed; accepted only if every new
    boundary is short and every part is big enough."""
    import cv2
    def area(c):
        j, i = c
        return (us[i + 1] - us[i]) * (vs[j + 1] - vs[j])
    if sum(area(c) for c in cells) < big:
        return [cells]
    u0, v0 = us[min(i for _, i in cells)], vs[min(j for j, _ in cells)]
    u1, v1 = us[max(i for _, i in cells) + 1], vs[max(j for j, _ in cells) + 1]
    Wd, Hd = int(np.ceil((u1 - u0) / res)) + 2, int(np.ceil((v1 - v0) / res)) + 2
    img = np.zeros((Hd, Wd), np.uint8)
    def px(u, v):
        return int(round((u - u0) / res)) + 1, int(round((v - v0) / res)) + 1
    for (j, i) in cells:
        x0, y0 = px(us[i], vs[j]); x1, y1 = px(us[i + 1], vs[j + 1])
        img[y0:y1, x0:x1] = 1
    cs = set(cells)
    for (j, i) in cells:   # burn walls between cells of the region
        if (j, i + 1) in cs and E_v.get((j, i + 1), ('open',))[0] == 'wall':
            x, y0 = px(us[i + 1], vs[j]); _, y1 = px(us[i + 1], vs[j + 1]); img[y0:y1, max(x - 1, 0):x + 1] = 0
        if (j + 1, i) in cs and E_u.get((j + 1, i), ('open',))[0] == 'wall':
            x0, y = px(us[i], vs[j + 1]); x1, _ = px(us[i + 1], vs[j + 1]); img[max(y - 1, 0):y + 1, x0:x1] = 0
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 5) * res
    for r in (0.65, 0.75, 0.85):
        n, lab = cv2.connectedComponents((dist > r).astype(np.uint8), connectivity=4)
        sizes = np.bincount(lab.ravel(), minlength=n) * res * res
        keep = [k for k in range(1, n) if sizes[k] >= 1.0]
        if len(keep) < 2:
            continue
        markers = np.zeros(lab.shape, np.int32)
        for t, k in enumerate(keep):
            markers[lab == k] = t + 1
        markers[img == 0] = len(keep) + 1
        cv2.watershed(cv2.cvtColor((255 - np.clip(dist / dist.max() * 255, 0, 255)).astype(np.uint8), cv2.COLOR_GRAY2BGR), markers)
        parts = {}
        for (j, i) in cells:
            x0, y0 = px(us[i], vs[j]); x1, y1 = px(us[i + 1], vs[j + 1])
            patch = markers[y0:y1, x0:x1].ravel()
            patch = patch[(patch > 0) & (patch <= len(keep))]
            t = np.bincount(patch).argmax() if patch.size else 1
            parts.setdefault(t, []).append((j, i))
        if len(parts) < 2 or min(sum(area(c) for c in p) for p in parts.values()) < min_side:
            continue
        owner = {c: t for t, p in parts.items() for c in p}
        bnd = {}
        for (j, i) in cells:
            for n2, key, E, L in (((j, i + 1), (j, i + 1), E_v, vs[j + 1] - vs[j]), ((j + 1, i), (j + 1, i), E_u, us[i + 1] - us[i])):
                if n2 in owner and owner[n2] != owner[(j, i)] and E.get(key, ('wall',))[0] != 'wall':
                    k2 = tuple(sorted((owner[n2], owner[(j, i)])))
                    bnd[k2] = bnd.get(k2, 0) + L
        if bnd and max(bnd.values()) <= max_boundary:
            return [c for p in parts.values() for c in split_by_cores(p, us, vs, E_v, E_u, max_boundary, min_side, big, res)]
    return [cells]
