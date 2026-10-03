"""Depth -> world point cloud with per-point normals and observation metadata."""
import numpy as np


def backproject(frame, max_depth=4.0, min_conf=2, stride=1):
    """Returns camera-frame points (N,3), normals (N,3), pixel (N,2) for valid pixels.

    Normals come from finite differences on the organised depth image, which is far cheaper
    than kNN normals on the fused cloud and keeps per-frame provenance.
    """
    d = frame.depth[::stride, ::stride]
    c = frame.confidence[::stride, ::stride]
    h, w = d.shape
    K = frame.K
    fx, fy, cx, cy = K[0, 0] / stride, K[1, 1] / stride, K[0, 2] / stride, K[1, 2] / stride
    v, u = np.indices((h, w), dtype=np.float32)
    P = np.stack([(u - cx) * d / fx, (v - cy) * d / fy, d], -1)

    du = np.zeros_like(P); dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    # orient towards camera
    flip = (n * P).sum(-1) > 0
    n[flip] *= -1

    # reject depth discontinuities (mixed pixels) - gradient too large relative to depth
    grad = np.maximum(np.linalg.norm(du, axis=-1), np.linalg.norm(dv, axis=-1))
    valid = (d > 0.15) & (d < max_depth) & (c >= min_conf) & (nn[..., 0] > 0) & (grad < 0.08 * d * stride + 0.02)
    valid[[0, -1], :] = False; valid[:, [0, -1]] = False
    return P[valid], n[valid], np.stack([u[valid], v[valid]], -1) * stride, d[valid]


def to_world(P, n, T):
    R, t = T[:3, :3], T[:3, 3]
    return P @ R.T + t, n @ R.T


def build_cloud(capture, step=4, max_depth=4.0, stride=1, poses=None, frames=None):
    """Fuse frames into a world cloud. Returns dict of arrays (float32)."""
    idx = range(0, capture.n, step) if frames is None else frames
    pts, nrm, fid, rng = [], [], [], []
    for i in idx:
        f = capture.frame(i, None if poses is None else poses[i])
        P, n, _, d = backproject(f, max_depth=max_depth, stride=stride)
        Pw, nw = to_world(P, n, f.T_wc)
        pts.append(Pw.astype(np.float32)); nrm.append(nw.astype(np.float32))
        fid.append(np.full(len(Pw), i, np.int32)); rng.append(d.astype(np.float32))
    return {'xyz': np.concatenate(pts), 'normal': np.concatenate(nrm),
            'frame': np.concatenate(fid), 'range': np.concatenate(rng)}


def voxel_reduce(cloud, voxel=0.02):
    """Keep one point per voxel (the one with the smallest sensor range = most accurate)."""
    q = np.floor(cloud['xyz'] / voxel).astype(np.int64)
    key = (q[:, 0] + (1 << 20)) * (1 << 42) + (q[:, 1] + (1 << 20)) * (1 << 21) + (q[:, 2] + (1 << 20))
    order = np.lexsort((cloud['range'], key))
    _, first = np.unique(key[order], return_index=True)
    keep = order[first]
    return {k: v[keep] for k, v in cloud.items()}
