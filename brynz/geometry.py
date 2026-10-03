"""Gravity / Manhattan frame, floor & ceiling estimation, wall evidence grids."""
import numpy as np
from scipy import ndimage


def floor_ceiling(cloud, bin_m=0.01):
    """Floor = dominant upward-facing horizontal plane; ceiling = dominant downward-facing one
    at least 1.8 m above the floor. Returns (floor_y, ceiling_y or None, ceiling_support)."""
    y, ny = cloud['xyz'][:, 1], cloud['normal'][:, 1]
    up = y[ny > 0.95]
    h, e = np.histogram(up, bins=np.arange(up.min(), up.max() + bin_m, bin_m))
    floor = refine_peak(up, e[np.argmax(h)], bin_m)
    down = y[(ny < -0.95) & (y > floor + 1.8)]
    if len(down) < 2000:
        return floor, None, len(down)
    h, e = np.histogram(down, bins=np.arange(down.min(), down.max() + bin_m, bin_m))
    ceil = refine_peak(down, e[np.argmax(h)], bin_m)
    return floor, ceil, int(h.max())


def refine_peak(vals, guess, bin_m, win=0.03):
    """Sub-bin plane offset: trimmed mean of samples within +-win of the histogram peak."""
    v = vals[np.abs(vals - (guess + bin_m / 2)) < win]
    return float(np.mean(v)) if len(v) else float(guess)


def manhattan_yaw(cloud):
    """Dominant wall orientation from horizontal components of vertical-surface normals.
    Uses the 4-fold symmetric angle histogram; returns yaw (rad) that rotates walls to x/z axes."""
    n = cloud['normal']
    m = np.abs(n[:, 1]) < 0.15
    a = np.arctan2(n[m, 2], n[m, 0])
    a4 = np.mod(a * 4, 2 * np.pi)
    h, e = np.histogram(a4, bins=720, range=(0, 2 * np.pi))
    h = ndimage.gaussian_filter1d(h.astype(float), 3, mode='wrap')
    peak = (e[np.argmax(h)] + e[1] - e[0]) / 4
    # refine with circular mean near peak
    d = np.angle(np.exp(1j * (a4 - peak * 4)))
    sel = np.abs(d) < 0.15
    peak = peak + np.angle(np.mean(np.exp(1j * d[sel]))) / 4
    return float(peak)


def plan_coords(xyz, yaw, floor):
    """World -> plan frame: (u, v) horizontal axes aligned to walls, h = height above floor."""
    c, s = np.cos(yaw), np.sin(yaw)
    x, z = xyz[:, 0], xyz[:, 2]
    u = c * x + s * z
    v = -s * x + c * z
    return np.stack([u, v], 1), xyz[:, 1] - floor


def rotate_dirs(nxz, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([c * nxz[:, 0] + s * nxz[:, 1], -s * nxz[:, 0] + c * nxz[:, 1]], 1)


class Grid:
    def __init__(self, uv, res=0.02, pad=0.5):
        self.res = res
        self.origin = uv.min(0) - pad
        self.shape = tuple((np.ceil((uv.max(0) + pad - self.origin) / res)).astype(int)[::-1] + 1)  # (rows=v, cols=u)

    def ij(self, uv):
        q = np.floor((uv - self.origin) / self.res).astype(int)
        return q[:, 1].clip(0, self.shape[0] - 1), q[:, 0].clip(0, self.shape[1] - 1)

    def to_uv(self, rows, cols):
        return np.stack([cols * self.res + self.origin[0] + self.res / 2, rows * self.res + self.origin[1] + self.res / 2], -1)


def wall_evidence(uv, h, nrm_uv, vertical, grid, wall_height, hbin=0.1):
    """Per-cell vertical coverage: fraction of 10cm height bins (0.2m .. wall_height-0.15m)
    containing vertical-surface points. Furniture stops short; walls span floor to ceiling."""
    lo, hi = 0.2, wall_height - 0.15
    nb = int(np.ceil((hi - lo) / hbin))
    m = vertical & (h > lo) & (h < hi)
    r, c = grid.ij(uv[m])
    b = ((h[m] - lo) / hbin).astype(int).clip(0, nb - 1)
    key = np.unique((r * grid.shape[1] + c) * nb + b)
    cell = key // nb
    cov = np.bincount(cell, minlength=grid.shape[0] * grid.shape[1]).reshape(grid.shape) / nb
    # top coverage: does the column reach the upper 40% of the room (furniture rarely does)
    topkey = key[(key % nb) >= int(nb * 0.6)] // nb
    top = np.bincount(topkey, minlength=grid.shape[0] * grid.shape[1]).reshape(grid.shape) / (nb - int(nb * 0.6))
    return cov, top
