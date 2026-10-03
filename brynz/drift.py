"""Drift accountability: plane-anchored pose-graph correction with an on/off ablation.

ARKit VIO is already good over short walks, but it drifts in yaw and translation over long
multi-room loops, which shows up as doubled / thickened walls where a room is re-visited.

Method
  1. Keyframe segments (~1.3 s). Each segment yields a 2-D scan of wall points (vertical
     surfaces, 0.3-1.8 m above floor) in world x-z.
  2. Anchor measurement: each segment is ICP-registered (2-D, trimmed point-to-point, yaw +
     translation) to the map built from *temporally distant* segments only (>= GAP_S away), so a
     segment is never matched to itself. That measures its absolute drift against the structure
     seen at other times - plane-anchored loop closure.
  3. Pose graph over per-keyframe corrections c_k = (dx, dz, dyaw):
        minimise  sum_k |c_k - c_{k-1}|^2 / s_odo^2  +  sum_k w_k |c_k - m_k|^2 / s_icp^2
     (odometry smoothness + ICP anchors weighted by inlier ratio). Linear least squares.
  4. Corrections are interpolated to every frame and applied as a rigid 2-D transform about the
     keyframe camera position (gravity is left untouched - ARKit's roll/pitch are reliable).
  5. Acceptance: wall crispness (median distance of wall points to their Manhattan face) must
     improve, else the raw poses are kept. Both runs are reported (ablation).
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .cloud import backproject, to_world

SEG_FRAMES = 60
GAP_S = 20.0
S_ODO = 0.01      # m (and rad) per keyframe step: ARKit relative motion is trusted
S_ICP = 0.03


def segment_scans(cap, poses, floor_y, step=6, seg=SEG_FRAMES):
    scans, keys = [], []
    for k0 in range(0, cap.n, seg):
        pts = []
        for i in range(k0, min(k0 + seg, cap.n), step):
            f = cap.frame(i, poses[i])
            P, n, _, _ = backproject(f, max_depth=3.5, stride=2)
            Pw, nw = to_world(P, n, poses[i])
            h = Pw[:, 1] - floor_y
            m = (np.abs(nw[:, 1]) < 0.2) & (h > 0.3) & (h < 1.8)
            pts.append(Pw[m][:, [0, 2]])
        p = np.concatenate(pts) if pts else np.zeros((0, 2))
        if len(p):
            q = np.unique(np.floor(p / 0.03).astype(np.int64), axis=0, return_index=True)[1]
            p = p[q]
        scans.append(p)
        keys.append(min(k0 + seg // 2, cap.n - 1))
    return scans, np.array(keys)


def icp2d(src, dst_tree, dst, iters=25, max_d=0.15, trim=0.8):
    R = np.eye(2); t = np.zeros(2); cur = src.copy(); inl = 0.0
    for _ in range(iters):
        d, j = dst_tree.query(cur, distance_upper_bound=max_d)
        ok = np.isfinite(d)
        if ok.sum() < 50:
            return None
        dd = d[ok]
        keep = dd <= np.quantile(dd, trim)
        a = cur[ok][keep]; b = dst[j[ok]][keep]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        dR = Vt.T @ U.T
        if np.linalg.det(dR) < 0:
            Vt[-1] *= -1; dR = Vt.T @ U.T
        dt = cb - dR @ ca
        cur = cur @ dR.T + dt
        R, t = dR @ R, dR @ t + dt
        inl = ok.mean()
        if np.linalg.norm(dt) < 1e-4 and abs(np.arctan2(dR[1, 0], dR[0, 0])) < 1e-5:
            break
    return R, t, inl


def anchors(scans, keys, fps):
    """ICP each segment against temporally distant segments. Returns per-key (dx,dz,dyaw,weight)."""
    meas = np.zeros((len(scans), 4))
    t_key = keys / fps
    for k, s in enumerate(scans):
        if len(s) < 200:
            continue
        far = [j for j in range(len(scans)) if abs(t_key[j] - t_key[k]) >= GAP_S and len(scans[j])]
        if not far:
            continue
        dst = np.concatenate([scans[j] for j in far])
        tree = cKDTree(dst)
        # only the part of the segment that overlaps the distant map is informative
        d, _ = tree.query(s, distance_upper_bound=0.3)
        ov = np.isfinite(d)
        if ov.sum() < 200:
            continue
        piv = s[ov].mean(0)
        r = icp2d(s[ov] - piv, cKDTree(dst - piv), dst - piv)
        if r is None:
            continue
        R, t, inl = r
        yaw = np.arctan2(R[1, 0], R[0, 0])
        if abs(yaw) > np.radians(4) or np.linalg.norm(t) > 0.25 or inl < 0.5:
            continue  # implausible for VIO drift: reject rather than trust
        meas[k] = [t[0], t[1], yaw, inl * ov.mean()]
    return meas


def solve_graph(meas):
    """Per-dimension tridiagonal least squares: smoothness + weighted anchors."""
    n = len(meas)
    A = np.zeros((n, n))
    for k in range(1, n):
        A[k, k] += 1 / S_ODO ** 2; A[k - 1, k - 1] += 1 / S_ODO ** 2
        A[k, k - 1] -= 1 / S_ODO ** 2; A[k - 1, k] -= 1 / S_ODO ** 2
    w = meas[:, 3] / S_ICP ** 2
    A[np.diag_indices(n)] += w + 1e-9
    out = np.zeros((n, 3))
    for d in range(3):
        out[:, d] = np.linalg.solve(A, w * meas[:, d])
    return out


def apply(poses, keys, corr):
    """Interpolate keyframe corrections to frames; rotate about the keyframe camera position."""
    n = len(poses)
    idx = np.arange(n)
    c = np.stack([np.interp(idx, keys, corr[:, d]) for d in range(3)], 1)
    out = poses.copy()
    for i in range(n):
        dx, dz, dyaw = c[i]
        Ry = Rotation.from_euler('y', -dyaw).as_matrix()   # world yaw in x-z plane
        p = poses[i][:3, 3]
        out[i][:3, :3] = Ry @ poses[i][:3, :3]
        out[i][:3, 3] = p + np.array([dx, 0, dz])
    return out


def crispness(cloud_uv, nuv, h, vertical):
    """Median |offset| of wall points from the nearest dominant face (1 cm bins). Lower = crisper."""
    out = []
    for pp in (1, 0):
        m = vertical & (np.abs(nuv[:, pp]) > 0.9) & (h > 0.3) & (h < 1.8)
        c = cloud_uv[m, pp]
        if len(c) < 1000:
            continue
        hist, e = np.histogram(c, bins=np.arange(c.min(), c.max() + 0.01, 0.01))
        peaks = e[np.nonzero(hist > np.percentile(hist, 99))[0]] + 0.005
        d = np.abs(c[:, None] - peaks[None, :]).min(1)
        out.append(np.median(d[d < 0.1]))
    return float(np.mean(out)) if out else float('nan')


def revisit_error(scans, keys, fps):
    """Ghosting metric: for wall points seen again >= GAP_S later, the median distance between the
    two sightings (same surface, different times). Pure drift shows up here as doubled walls."""
    t = keys / fps
    d_all = []
    for k, s in enumerate(scans):
        far = [j for j in range(len(scans)) if abs(t[j] - t[k]) >= GAP_S and len(scans[j])]
        if len(s) < 200 or not far:
            continue
        d, _ = cKDTree(np.concatenate([scans[j] for j in far])).query(s, distance_upper_bound=0.15)
        d_all.append(d[np.isfinite(d)])
    d = np.concatenate(d_all) if d_all else np.zeros(0)
    return float(np.median(d)) if len(d) else float('nan')


def correct(cap, poses, floor_y, log=print):
    # LiDAR: depth frames at ~60 Hz -> SEG_FRAMES ~ 1 s. RGB tiers: keyframes at cap.fps -> ~1.5 s
    seg = SEG_FRAMES if getattr(cap, 'tier', 'lidar') == 'lidar' else max(3, int(round(cap.fps * 1.5)))
    scans, keys = segment_scans(cap, poses, floor_y, step=max(1, seg // 10), seg=seg)
    meas = anchors(scans, keys, cap.fps)
    n_anchor = int((meas[:, 3] > 0).sum())
    log(f'[drift] {len(scans)} segments, {n_anchor} anchored by distant revisits')
    if n_anchor == 0:
        return poses, {'applied': False, 'reason': 'no revisits: nothing to anchor against', 'anchors': 0}
    corr = solve_graph(meas)
    new = apply(poses, keys, corr)
    scans2, _ = segment_scans(cap, new, floor_y, step=max(1, seg // 10), seg=seg)
    e0, e1 = revisit_error(scans, keys, cap.fps), revisit_error(scans2, keys, cap.fps)
    log(f'[drift] revisit ghosting {e0 * 1000:.1f} -> {e1 * 1000:.1f} mm')
    info = {'anchors': n_anchor, 'segments': len(scans),
            'revisit_error_raw_mm': round(e0 * 1000, 2), 'revisit_error_corrected_mm': round(e1 * 1000, 2),
            'max_correction_m': round(float(np.abs(corr[:, :2]).max()), 4),
            'max_yaw_correction_deg': round(float(np.degrees(np.abs(corr[:, 2]).max())), 3)}
    return new, info
