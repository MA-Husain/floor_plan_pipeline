"""RGB-only tiers (video, photos): images -> metric depth + poses -> the same plan pipeline.

There is no LiDAR and no ARKit pose on these tiers. MapAnything (Meta, Apache-2.0 weights
`facebook/map-anything-apache`) predicts, feed-forward from a set of images, metric depth,
camera intrinsics and camera poses in one shared frame. Its output is wrapped as a capture
object with the same interface as StrayCapture, so wall extraction, segmentation, openings,
drift handling, damage and rendering are literally the same code as the LiDAR tier.

Scaling to many views on a 16 GB laptop: a *skeleton* pass reconstructs a spread of key views
over the whole walk in one shot (this fixes global layout and metric scale); every other view is
reconstructed in a small *chunk* together with a few skeleton views, and the chunk is mapped into
the skeleton frame by a robust Sim(3) fitted on the shared skeleton views (same image, same pixel
grid -> dense 3-D correspondences, no feature matching needed).

Gravity: the model's world frame is the first camera's. 'Up' is recovered from the floor and
ceiling normals (refined from the mean camera up-vector), then the world is rotated to +Y up.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .capture import Frame

ROOT = Path(__file__).resolve().parent.parent
HF_CACHE = str(ROOT / 'weights' / 'hf')
MA_ID = 'facebook/map-anything-apache'


# ---------------------------------------------------------------- model
def _offline_dinov2(torch):
    """MapAnything builds its DINOv2 encoder through torch.hub (code + 1.1 GB ImageNet weights) and
    then overwrites every weight with its own checkpoint. Load the hub code from a local clone
    (scripts/fetch_weights.py) and skip the redundant weight download, so the run is offline."""
    if getattr(torch.hub, '_brynx_patched', False):
        return
    local = ROOT / 'weights' / 'torch' / 'hub' / 'facebookresearch_dinov2_main'
    orig = torch.hub.load

    def load(repo, model, *a, **k):
        if 'dinov2' in str(repo) and local.exists():
            k = {x: v for x, v in k.items() if x not in ('force_reload', 'skip_validation', 'trust_repo', 'verbose')}
            k['pretrained'] = False
            return orig(str(local), model, *a, source='local', **k)
        return orig(repo, model, *a, **k)
    torch.hub.load = load
    torch.hub._brynx_patched = True


class Reconstructor:
    def __init__(self, device=None):
        import os
        import torch
        _offline_dinov2(torch)
        from mapanything.models import MapAnything
        self.torch = torch
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = MapAnything.from_pretrained(MA_ID, cache_dir=HF_CACHE).to(self.device).eval()

    def infer(self, images, size, intrinsics=None):
        """images: list of RGB uint8 arrays already cropped to the aspect of `size` (W, H).
        intrinsics: optional list of 3x3 at the images' own resolution."""
        from mapanything.utils.image import preprocess_inputs
        views = []
        for k, im in enumerate(images):
            v = {'img': self.torch.from_numpy(np.ascontiguousarray(im))}
            if intrinsics is not None and intrinsics[k] is not None:
                v['intrinsics'] = self.torch.from_numpy(np.asarray(intrinsics[k], np.float32))
            views.append(v)
        views = preprocess_inputs(views, resize_mode='fixed_size', size=tuple(int(s) for s in size))
        with self.torch.no_grad():
            preds = self.model.infer(views, memory_efficient_inference=True, minibatch_size=1, use_amp=True,
                                     amp_dtype='fp16' if self.device == 'mps' else 'bf16',
                                     apply_mask=True, mask_edges=True, apply_confidence_mask=False)
        out = []
        for p in preds:
            g = lambda k: p[k][0].float().cpu().numpy()
            out.append({'depth': g('depth_z')[..., 0], 'conf': g('conf'), 'mask': g('mask')[..., 0] > 0.5,
                        'K': g('intrinsics'), 'T': g('camera_poses').astype(np.float64), 'pts': g('pts3d')})
        return out


# ---------------------------------------------------------------- geometry helpers
def umeyama(src, dst, scale=True):
    """Similarity (s, R, t) minimising |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum(1).mean() if scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def robust_sim3(src, dst, iters=4, keep=0.7):
    m = np.ones(len(src), bool)
    for _ in range(iters):
        s, R, t = umeyama(src[m], dst[m])
        r = np.linalg.norm(dst - (s * src @ R.T + t), axis=1)
        m = r <= np.quantile(r, keep)
    return s, R, t, float(np.median(r[m]))


def sim3_pose(T, s, R, t):
    out = T.copy()
    out[:3, :3] = R @ T[:3, :3]
    out[:3, 3] = s * R @ T[:3, 3] + t
    return out


def crop_to_aspect(im, aspect):
    """Centre-crop to width/height = aspect. Returns crop and (x0, y0)."""
    h, w = im.shape[:2]
    if w / h > aspect:
        nw = int(round(h * aspect)); x0 = (w - nw) // 2
        return im[:, x0:x0 + nw], (x0, 0)
    nh = int(round(w / aspect)); y0 = (h - nh) // 2
    return im[y0:y0 + nh], (0, y0)


def model_size(w, h, long_side=518, patch=14):
    """(W, H) multiple of patch with the long side ~long_side, matching the image aspect."""
    s = long_side / max(w, h)
    W, H = max(patch, int(round(w * s / patch)) * patch), max(patch, int(round(h * s / patch)) * patch)
    return W, H


# ---------------------------------------------------------------- capture wrapper
@dataclass
class _View:
    depth: np.ndarray
    conf: np.ndarray
    K: np.ndarray
    T: np.ndarray


class ReconCapture:
    """Same interface as StrayCapture (n, fps, rgb_size, depth_size, poses, frame, rgb_frames)."""
    upright = True     # images are already upright: no rotation before detectors

    def __init__(self, views, image_source, rgb_size, fps, tier, error_model, names=None, groups=None):
        self.views, self._src, self.rgb_size, self.fps = views, image_source, rgb_size, fps
        self.tier, self.error_model = tier, error_model
        self.names = names or [str(i) for i in range(len(views))]
        self.groups = groups
        self.n = len(views)
        self.depth_size = (views[0].depth.shape[1], views[0].depth.shape[0])

    @property
    def positions(self):
        return np.array([v.T[:3, 3] for v in self.views])

    def poses(self):
        return np.array([v.T for v in self.views])

    def pose(self, i):
        return self.views[i].T

    def K_depth(self, i):
        return self.views[i].K

    def frame(self, i, T_wc=None):
        v = self.views[i]
        return Frame(i, v.depth, v.conf, v.K, v.T if T_wc is None else T_wc)

    def rgb_frames(self, indices):
        for i in sorted(set(int(i) for i in indices)):
            yield i, self._src(i)


def _conf_levels(conf, mask):
    """Model confidence -> Stray-like 0/1/2 levels (per view percentiles)."""
    out = np.zeros(conf.shape, np.uint8)
    c = conf[mask]
    if c.size < 100:
        return out
    lo, hi = np.percentile(c, [15, 40])
    out[mask & (conf >= lo)] = 1
    out[mask & (conf >= hi)] = 2
    return out


def gravity_align(views, iters=3):
    """Rotate the world so that the floor/ceiling normal is +Y (Stray/ARKit convention)."""
    from .cloud import backproject, to_world
    up = -np.mean([v.T[:3, 1] for v in views], 0)          # camera -y = image-up, averaged
    up /= np.linalg.norm(up)
    N = []
    for v in views[:: max(1, len(views) // 60)]:
        P, n, _, _ = backproject(Frame(0, v.depth, v.conf, v.K, v.T), max_depth=5.0, min_conf=2, stride=2)
        _, nw = to_world(P, n, v.T)
        N.append(nw)
    N = np.concatenate(N)
    for cos in (0.8, 0.9, 0.97)[:iters]:
        d = N @ up
        sel = np.abs(d) > cos
        if sel.sum() < 100:
            break
        up = (N[sel] * np.sign(d[sel])[:, None]).sum(0)
        up /= np.linalg.norm(up)
    # rotation taking `up` to +Y
    y = np.array([0.0, 1.0, 0.0])
    ax = np.cross(up, y); s = np.linalg.norm(ax); c = up @ y
    if s < 1e-9:
        R = np.eye(3)
    else:
        k = ax / s
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + s * Kx + (1 - c) * Kx @ Kx
    for v in views:
        v.T = v.T.copy()
        v.T[:3, :3] = R @ v.T[:3, :3]
        v.T[:3, 3] = R @ v.T[:3, 3]
    return R


def _to_view(p, half=True):
    d, c, m, K = p['depth'], p['conf'], p['mask'] & (p['depth'] > 0), p['K'].copy()
    lev = _conf_levels(c, m)
    if half:   # half resolution keeps the cloud size comparable to Stray's 256x192 depth
        d, lev = d[::2, ::2], lev[::2, ::2]
        K[:2] /= 2
    return _View(np.where(lev > 0, d, 0).astype(np.float32), lev, K.astype(np.float64), p['T'])


def reconstruct(load, ids, size, intrinsics=None, skeleton=None, groups=None, anchors_per_group=4,
                rec=None, log=print):
    """load(i) -> RGB (cropped to the aspect of `size`). ids: all view ids.
    skeleton: ids reconstructed jointly first; groups: list of id lists (chunks); each chunk is
    run with its `anchors_per_group` nearest skeleton views and Sim(3)-mapped onto the skeleton.
    Returns {id: _View} in the skeleton (metric) frame, plus a per-chunk alignment report."""
    rec = rec or Reconstructor()
    K_of = (lambda i: intrinsics[i]) if intrinsics is not None else (lambda i: None)
    skeleton = list(skeleton if skeleton is not None else ids)
    log(f'[rgb] skeleton pass: {len(skeleton)} views')
    sk = rec.infer([load(i) for i in skeleton], size, [K_of(i) for i in skeleton] if intrinsics is not None else None)
    out = {i: p for i, p in zip(skeleton, sk)}
    report = []
    pos = {i: k for k, i in enumerate(skeleton)}
    for g in groups or []:
        todo = [i for i in g if i not in out]
        if not todo:
            continue
        # anchors: skeleton views nearest to the chunk (by id distance = time for video)
        mid = np.median(todo)
        anchors = sorted(skeleton, key=lambda a: abs(a - mid) if isinstance(a, (int, np.integer)) else 0)[:anchors_per_group]
        anchors = [a for a in anchors if a in g] + [a for a in anchors if a not in g]
        anchors = list(dict.fromkeys(anchors))[:anchors_per_group]
        run = anchors + todo
        pr = rec.infer([load(i) for i in run], size, [K_of(i) for i in run] if intrinsics is not None else None)
        src, dst = [], []
        for a, p in zip(anchors, pr[:len(anchors)]):
            q = sk[pos[a]]
            m = p['mask'] & q['mask']
            m &= (p['conf'] >= np.percentile(p['conf'][m], 50)) if m.sum() > 100 else m
            src.append(p['pts'][m][::7]); dst.append(q['pts'][m][::7])
        src, dst = np.concatenate(src), np.concatenate(dst)
        s, R, t, res = robust_sim3(src, dst)
        report.append({'chunk': [int(todo[0]) if isinstance(todo[0], (int, np.integer)) else str(todo[0]), len(todo)],
                       'scale': round(float(s), 4), 'residual_m': round(res, 4)})
        log(f'[rgb] chunk {len(todo)} views: sim3 scale {s:.3f}, residual {res * 100:.1f} cm')
        for i, p in zip(todo, pr[len(anchors):]):
            p['T'] = sim3_pose(p['T'], s, R, t)
            p['depth'] = p['depth'] * s
            p['pts'] = s * p['pts'] @ R.T + t
            out[i] = p
    views = {i: _to_view(p) for i, p in out.items()}
    return views, report


def _corr(p, q):
    """Dense 3-D correspondences between two predictions of the same image."""
    m = p['mask'] & q['mask']
    if m.sum() > 100:
        m &= (p['conf'] >= np.percentile(p['conf'][m], 40)) & (q['conf'] >= np.percentile(q['conf'][m], 40))
    return p['pts'][m][::5], q['pts'][m][::5]


SIGMA_WINDOW_SCALE = 0.15   # a single window's metric scale is good to ~15 % (measured vs LiDAR)


def reconstruct_sequential(load, ids, size, intrinsics=None, window=28, overlap=10, rec=None, log=print):
    """Video: overlapping windows along the walk, chained through the views they share.

    Each window is metric on its own, but only to ~15 %; each overlap measures the relative scale
    of two neighbouring windows, but bad overlaps (fast turns, blur) are unreliable. So the window
    scales x_k = log(scale) are solved jointly:
        min  sum_k w_k (x_k - x_{k-1} - log rho_k)^2  +  sum_k x_k^2 / SIGMA_WINDOW_SCALE^2
    with w_k from the overlap fit residual. Then rotations/translations are chained with each
    overlap's scale fixed to the solved ratio. A bad link can no longer run the scale away."""
    rec = rec or Reconstructor()
    K_of = (lambda i: intrinsics[i]) if intrinsics is not None else (lambda i: None)
    stride = window - overlap
    starts = list(range(0, max(1, len(ids) - overlap), stride))
    wins = []
    for w, a0 in enumerate(starts):
        run = ids[a0:a0 + window]
        pr = rec.infer([load(i) for i in run], size, [K_of(i) for i in run] if intrinsics is not None else None)
        wins.append(dict(zip(run, pr)))
        log(f'[rgb] window {w + 1}/{len(starts)}: {len(run)} views')
    # relative similarity k -> k-1 from shared views
    rel = [None]
    for k in range(1, len(wins)):
        shared = [i for i in wins[k] if i in wins[k - 1]]
        src, dst = zip(*[_corr(wins[k][i], wins[k - 1][i]) for i in shared])
        src, dst = np.concatenate(src), np.concatenate(dst)
        s_, R_, t_, res = robust_sim3(src, dst)
        rel.append({'src': src, 'dst': dst, 'rho': s_, 'res': res, 'shared': len(shared)})
    # joint log-scale solve
    n = len(wins)
    A = np.eye(n) / SIGMA_WINDOW_SCALE ** 2
    bvec = np.zeros(n)
    for k in range(1, n):
        wk = 1.0 / (0.02 + 0.2 * rel[k]['res']) ** 2      # link sigma: 4 % at 10 cm residual, 12 % at 50 cm
        A[k, k] += wk; A[k - 1, k - 1] += wk; A[k, k - 1] -= wk; A[k - 1, k] -= wk
        bvec[k] += wk * np.log(rel[k]['rho']); bvec[k - 1] -= wk * np.log(rel[k]['rho'])
    x = np.linalg.solve(A, bvec)
    # chain rotations/translations with the solved scale ratios
    M = [np.eye(4)]
    M[0][:3, :3] *= np.exp(x[0])
    report = [{'window': 0, 'scale': round(float(np.exp(x[0])), 4)}]
    for k in range(1, n):
        rho = np.exp(x[k] - x[k - 1])
        _, R_, t_ = umeyama(rho * rel[k]['src'], rel[k]['dst'], scale=False)
        S = np.eye(4); S[:3, :3] = rho * R_; S[:3, 3] = t_
        M.append(M[k - 1] @ S)
        r = np.linalg.norm(rel[k]['dst'] - (rho * rel[k]['src'] @ R_.T + t_), axis=1)
        report.append({'window': k, 'shared': rel[k]['shared'], 'overlap_scale_measured': round(float(rel[k]['rho']), 4),
                       'overlap_scale_used': round(float(rho), 4), 'overlap_residual_m': round(float(np.median(r)), 4),
                       'scale': round(float(np.exp(x[k])), 4)})
        log(f"[rgb] link {k}: overlap scale measured {rel[k]['rho']:.3f}, used {rho:.3f}, residual {np.median(r) * 100:.1f} cm")
    out = {}
    for k, wv in enumerate(wins):
        sk = np.cbrt(np.linalg.det(M[k][:3, :3]))
        Rk, tk = M[k][:3, :3] / sk, M[k][:3, 3]
        for i, p in wv.items():
            if i in out:
                continue
            p['T'] = sim3_pose(p['T'], sk, Rk, tk)
            p['depth'] = p['depth'] * sk
            out[i] = p
    return {i: _to_view(p) for i, p in out.items()}, report


# ---------------------------------------------------------------- tiers
VIDEO_ERRORS = {'face_sys': 0.015, 'drift_per_m': 0.01, 'scale_rel': 0.10}   # scale: measured +-15 % per window vs LiDAR
PHOTO_ERRORS = {'face_sys': 0.03, 'drift_per_m': 0.015, 'scale_rel': 0.12}


def sharpness(im):
    g = cv2.cvtColor(cv2.resize(im, (270, 480) if im.shape[0] > im.shape[1] else (480, 270)), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def video_capture(path, every_s=0.25, rotate=None, intrinsics=None, rec=None,
                  cache=None, log=print):
    """Handheld video -> ReconCapture. Keyframes: sharpest frame in each `every_s` window."""
    vc = cv2.VideoCapture(str(path))
    fps = vc.get(cv2.CAP_PROP_FPS)
    n = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    win = max(1, int(round(fps * every_s)))
    keys, frames, best = [], {}, (None, -1, None)
    i = 0
    while True:
        ok, f = vc.read()
        if not ok:
            break
        if i % 3 == 0:   # score every 3rd frame
            rgb = cv2.cvtColor(f if rotate is None else cv2.rotate(f, rotate), cv2.COLOR_BGR2RGB)
            sc = sharpness(rgb)
            if sc > best[1]:
                best = (i, sc, rgb)
        if (i + 1) % win == 0 and best[0] is not None:
            keys.append(best[0]); frames[best[0]] = best[2]; best = (None, -1, None)
        i += 1
    vc.release()
    h, w = frames[keys[0]].shape[:2]
    size = model_size(w, h)
    aspect = size[0] / size[1]
    crop = {k: crop_to_aspect(frames[k], aspect)[0] for k in keys}
    rgb_size = (crop[keys[0]].shape[1], crop[keys[0]].shape[0])
    log(f'[rgb] video {n} frames @ {fps:.0f} fps -> {len(keys)} keyframes, model size {size}')
    idx = list(range(len(keys)))
    Ks = None
    if intrinsics is not None:
        c, (x0, y0) = crop_to_aspect(frames[keys[0]], aspect)
        K = np.array(intrinsics, float).copy(); K[0, 2] -= x0; K[1, 2] -= y0
        Ks = [K] * len(keys)
    views, report = _cached_reconstruct(cache, lambda j: crop[keys[j]], idx, size, Ks, None, None, rec, log,
                                        sequential=True)
    vlist = [views[j] for j in idx]
    gravity_align(vlist)
    cap = ReconCapture(vlist, lambda j: cv2.cvtColor(crop[keys[j]], cv2.COLOR_RGB2BGR), rgb_size,
                       fps=1.0 / every_s, tier='video', error_model=VIDEO_ERRORS,
                       names=[f'frame{k}' for k in keys])
    cap.source_frames = keys
    cap.times_s = [round(k / fps, 2) for k in keys]
    cap.alignment = report
    return cap


def _cached_reconstruct(cache, load, idx, size, Ks, sk, groups, rec, log, sequential=False):
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        log(f'[rgb] replaying cached reconstruction {cache}')
        views = {int(i): _View(z['depth'][k], z['conf'][k], z['K'][k], z['T'][k]) for k, i in enumerate(z['ids'])}
        return views, list(z['report'])
    if sequential:
        views, report = reconstruct_sequential(load, idx, size, Ks, rec=rec, log=log)
    else:
        views, report = reconstruct(load, idx, size, Ks, sk, groups, rec=rec, log=log)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(views)
        np.savez_compressed(cache, ids=np.array(ids), depth=np.stack([views[i].depth for i in ids]),
                            conf=np.stack([views[i].conf for i in ids]), K=np.stack([views[i].K for i in ids]),
                            T=np.stack([views[i].T for i in ids]), report=np.array(report, dtype=object))
    return views, report


def exif_intrinsics(img_pil):
    """Pinhole K from EXIF 35 mm-equivalent focal length (iPhone HEIC/JPEG)."""
    ex = img_pil.getexif().get_ifd(0x8769)
    f35 = ex.get(0xA405)
    if not f35:
        return None
    w, h = img_pil.size
    f = f35 / 43.2666 * np.hypot(w, h)      # 35 mm diagonal = 43.27 mm
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def photo_capture(folder, rec=None, cache=None, log=print, per_room_skeleton=2):
    """Per-room photo folders -> ReconCapture. Every photo is reconstructed jointly when they fit
    in one pass; otherwise each room folder is a chunk anchored to a skeleton of
    `per_room_skeleton` photos per room."""
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    folder = Path(folder)
    rooms = sorted(d for d in folder.iterdir() if d.is_dir())
    paths, room_of = [], []
    for d in rooms:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in ('.heic', '.jpg', '.jpeg', '.png'):
                paths.append(p); room_of.append(d.name)
    ims, Ks = [], []
    for p in paths:
        im = ImageOps.exif_transpose(Image.open(p))
        K = exif_intrinsics(im)
        s = 1600 / max(im.size)
        im = im.convert('RGB').resize((int(im.size[0] * s), int(im.size[1] * s)), Image.LANCZOS)
        if K is not None:
            K = K.copy(); K[:2] *= s
        ims.append(np.array(im)); Ks.append(K)
    # common model size from the dominant (portrait) aspect; landscape shots are rotated upright-safe by crop
    h, w = ims[0].shape[:2]
    size = model_size(w, h)
    aspect = size[0] / size[1]
    crops, Kc = [], []
    for im, K in zip(ims, Ks):
        if (im.shape[1] > im.shape[0]) != (w > h):
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)   # odd orientation: keep model size fixed
            K = None
        c, (x0, y0) = crop_to_aspect(im, aspect)
        if K is not None:
            K = K.copy(); K[0, 2] -= x0; K[1, 2] -= y0
        crops.append(c); Kc.append(K)
    log(f'[rgb] {len(paths)} photos in {len(rooms)} room folders, model size {size}')
    idx = list(range(len(paths)))
    groups = [[i for i in idx if room_of[i] == r.name] for r in rooms]
    sk = [i for g in groups for i in g[:per_room_skeleton]]
    if len(paths) <= 60:
        sk, groups = idx, []
    views, report = _cached_reconstruct(cache, lambda j: crops[j], idx, size, Kc, sk, groups, rec, log)
    vlist = [views[j] for j in idx]
    gravity_align(vlist)
    cap = ReconCapture(vlist, lambda j: cv2.cvtColor(crops[j], cv2.COLOR_RGB2BGR),
                       (crops[0].shape[1], crops[0].shape[0]), fps=1.0, tier='photo', error_model=PHOTO_ERRORS,
                       names=[f'{room_of[i]}/{paths[i].name}' for i in idx])
    cap.room_of = room_of
    cap.alignment = report
    return cap
