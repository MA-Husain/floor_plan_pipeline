"""Rooms, walls and openings from wall-evidence + observed-floor grids (Manhattan frame).

Pipeline:
 1. wall cells split by orientation (normal along u -> wall runs along v, and vice versa)
 2. door-sized gaps bridged *along each wall's own axis* (so corridors are never filled)
 3. interior = observed floor + walked trajectory, holes (furniture footprints) filled
 4. rooms = connected components of interior minus closed walls
 Used as the connectivity barrier for the cell complex (brynx.cells).
"""
import numpy as np
import cv2
from scipy import ndimage


def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def oriented_walls(uv, h, nuv, vertical, grid, band, cov_thr=0.4, hbin=0.1):
    """Coverage maps for the two wall families. Returns (Wu, Wv): Wu = walls running along u
    (normal ~ +-v), Wv = walls running along v (normal ~ +-u)."""
    lo, hi = band
    nb = int(np.ceil((hi - lo) / hbin))
    out = []
    for axis in (1, 0):  # normal component along v -> wall along u
        m = vertical & (h > lo) & (h < hi) & (np.abs(nuv[:, axis]) > 0.85)
        r, c = grid.ij(uv[m])
        b = ((h[m] - lo) / hbin).astype(int).clip(0, nb - 1)
        key = np.unique((r * grid.shape[1] + c) * nb + b)
        cov = np.bincount(key // nb, minlength=grid.shape[0] * grid.shape[1]).reshape(grid.shape) / nb
        out.append(cov)
    covU, covV = out
    # a wall seen from one side is ~1 cell thick but coverage gets split across adjacent cells
    # by noise: accumulate across +-2 cells perpendicular to the wall
    covU = ndimage.maximum_filter1d(ndimage.uniform_filter1d(covU, 3, axis=0) * 3, 1, axis=0)
    covV = ndimage.uniform_filter1d(covV, 3, axis=1) * 3
    return covU.clip(0, 1) > cov_thr, covV.clip(0, 1) > cov_thr


def clean(mask, min_len_cells, axis):
    """Drop wall fragments shorter than min_len along their running axis."""
    lab, n = ndimage.label(mask, np.ones((3, 3)))
    if n == 0:
        return mask
    sl = ndimage.find_objects(lab)
    keep = np.zeros(n + 1, bool)
    for i, s in enumerate(sl):
        ext = s[axis].stop - s[axis].start
        keep[i + 1] = ext >= min_len_cells
    return keep[lab]


def bridge(mask, gap_cells, axis):
    """1-D closing along the wall's own axis: bridges door gaps in straight walls only."""
    k = np.ones((1, gap_cells), np.uint8) if axis == 1 else np.ones((gap_cells, 1), np.uint8)
    m = mask.astype(np.uint8)
    # thicken perpendicular slightly so slightly misaligned stubs on either side of a door still meet
    perp = np.ones((5, 1), np.uint8) if axis == 1 else np.ones((1, 5), np.uint8)
    m = cv2.dilate(m, perp)
    p = gap_cells + 2  # pad with empty border: OpenCV erosion treats outside as foreground
    m = cv2.copyMakeBorder(m, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    m = cv2.erode(cv2.dilate(m, k), k, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return m[p:-p, p:-p].astype(bool)


def segment(grid, Wu, Wv, floor_mask, traj_rc, door_max=1.3, window_max=3.0, min_room_m2=1.2):
    res = grid.res
    Wu = clean(Wu, int(0.25 / res), axis=1)
    Wv = clean(Wv, int(0.25 / res), axis=0)
    W = Wu | Wv
    walk = np.zeros(grid.shape, bool)
    walk[traj_rc[:, 0], traj_rc[:, 1]] = True
    walk = cv2.dilate(walk.astype(np.uint8), disk(int(0.25 / res))).astype(bool)
    F = floor_mask | walk
    F = cv2.morphologyEx(F.astype(np.uint8), cv2.MORPH_CLOSE, disk(int(0.15 / res))).astype(bool)

    # Bridge collinear gaps up to window_max (windows: LiDAR passes through glass so the wall
    # vanishes above the sill; unentered doorways). Then re-open every bridge the operator
    # physically walked through - those are doors by definition.
    gap = int(window_max / res)
    Wd = cv2.dilate(W.astype(np.uint8), disk(1)).astype(bool)
    Wc = bridge(Wu, gap, axis=1) | bridge(Wv, gap, axis=0) | Wd
    Wc = cv2.morphologyEx(Wc.astype(np.uint8), cv2.MORPH_CLOSE, disk(4)).astype(bool)
    walked_through = Wc & ~Wd & cv2.dilate(walk.astype(np.uint8), disk(int(0.2 / res))).astype(bool)
    # widen the re-opened part to the full bridged run so door jambs are clean
    lab_b, _ = ndimage.label(Wc & ~Wd, np.ones((3, 3)))
    ids = np.unique(lab_b[walked_through]); ids = ids[ids > 0]
    door_cells = np.isin(lab_b, ids) & (Wc & ~Wd)
    Wc_closed = Wc.copy()          # keeps doors closed -> room separation
    Wc = Wc & ~door_cells          # door-open version (not used for rooms)
    Wc, Wc_open = Wc_closed, Wc

    # Rooms extend to the walls, not just to where floor was observed: label the space not
    # occupied by (closed) walls, keep regions seeded by observed floor / walked path.
    reach = cv2.dilate(F.astype(np.uint8), disk(int(0.3 / res))).astype(bool)
    lab, n = ndimage.label(~Wc)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    rooms = []
    for i in range(1, n + 1):
        m = lab == i
        if (m & F).sum() * res * res < min_room_m2 * 0.5:
            continue
        if i in border:            # wall not closed -> leaks to exterior; bound by observation
            m = m & reach
            m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, disk(int(0.2 / res))).astype(bool)
        if m.sum() * res * res < min_room_m2:
            continue
        rooms.append(ndimage.binary_fill_holes(m))
    return rooms, W, Wc, F, walk
