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
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .capture import Frame

# pycolmap and torch each ship libomp; macOS aborts on the second copy unless told it is expected
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

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

    def infer(self, images, size, intrinsics=None, poses=None, depths=None):
        """images: list of RGB uint8 arrays already cropped to the aspect of `size` (W, H).
        intrinsics: optional list of 3x3 at the images' own resolution.
        poses: optional list of known cam-to-world 4x4 (metric) or None; view 0 must be posed if any is."""
        from mapanything.utils.image import preprocess_inputs
        views = []
        for k, im in enumerate(images):
            v = {'img': self.torch.from_numpy(np.ascontiguousarray(im))}
            if intrinsics is not None and intrinsics[k] is not None:
                v['intrinsics'] = self.torch.from_numpy(np.asarray(intrinsics[k], np.float32))
            if poses is not None and poses[k] is not None:
                v['camera_poses'] = self.torch.from_numpy(np.asarray(poses[k], np.float32))
                v['is_metric_scale'] = self.torch.tensor([True])
            if depths is not None and depths[k] is not None:
                d = np.asarray(depths[k], np.float32)
                if d.shape != im.shape[:2]:
                    d = cv2.resize(d, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
                v['depth_z'] = self.torch.from_numpy(d)
                v['is_metric_scale'] = self.torch.tensor([True])
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


def reconstruct_sequential(load, ids, size, intrinsics=None, window=24, overlap=8, rec=None, log=print, depths=None):
    """Video: incremental, pose-conditioned windows along the walk.

    Window 1 is reconstructed freely. Every later window is fed its `overlap` already-solved views
    first, *with their solved poses* (and every view with one shared focal length, estimated on
    window 1 when not given), followed by the new views. MapAnything then reconstructs the new
    views consistently with the known ones instead of in an unrelated frame, so joining windows is
    a small correction (Sim(3) on the shared views' dense points) rather than a gamble on two
    independent guesses agreeing - the failure mode of independent windows on handheld video."""
    rec = rec or Reconstructor()
    out, report = {}, []
    first = ids[:window]
    D = (lambda run: [depths[i] for i in run]) if depths is not None else (lambda run: None)
    pr = rec.infer([load(i) for i in first], size, [intrinsics[i] for i in first] if intrinsics is not None else None,
                   depths=D(first))
    for i, p in zip(first, pr):
        out[i] = p
    if intrinsics is None:      # one camera, one focal: median of window 1, at the image resolution
        im0 = load(first[0]); sx = im0.shape[1] / size[0]; sy = im0.shape[0] / size[1]
        Km = np.median(np.stack([p['K'] for p in pr]), 0)
        K = np.array([[Km[0, 0] * sx, 0, im0.shape[1] / 2], [0, Km[1, 1] * sy, im0.shape[0] / 2], [0, 0, 1.0]])
        intrinsics = {i: K for i in ids}
        log(f'[rgb] focal fixed from window 1: {K[0, 0]:.0f} px ({np.degrees(2 * np.arctan(im0.shape[1] / 2 / K[0, 0])):.0f} deg h-FOV)')
    pos, stride = len(first), window - overlap
    w = 1
    while pos < len(ids):
        shared = [i for i in ids[max(0, pos - overlap):pos]]
        new = ids[pos:pos + stride]
        run = shared + new
        pr = rec.infer([load(i) for i in run], size, [intrinsics[i] for i in run],
                       poses=[out[i]['T'] for i in shared] + [None] * len(new), depths=D(run))
        pr = dict(zip(run, pr))
        src, dst = zip(*[_corr(pr[i], out[i]) for i in shared])
        src, dst = np.concatenate(src), np.concatenate(dst)
        s_, R, t, res = robust_sim3(src, dst)
        if depths is not None:      # every view carries metric depth: windows share the scale -> rigid join
            s_measured = s_
            m = np.ones(len(src), bool)
            for _ in range(4):
                _, R, t = umeyama(src[m], dst[m], scale=False)
                r = np.linalg.norm(dst - (src @ R.T + t), axis=1)
                m = r <= np.quantile(r, 0.7)
            s_, res = 1.0, float(np.median(r[m]))
        report.append({'window': w, 'shared': len(shared), 'new': len(new), 'scale': round(float(s_), 4),
                       'residual_m': round(float(res), 4)})
        log(f'[rgb] window {w + 1}: {len(new)} new views, link scale {s_:.3f}, residual {res * 100:.1f} cm')
        for i in new:
            p = pr[i]
            p['T'] = sim3_pose(p['T'], s_, R, t)
            p['depth'] = p['depth'] * s_
            p['pts'] = s_ * p['pts'] @ R.T + t
            out[i] = p
        pos += len(new); w += 1
        for i in ids[:max(0, pos - overlap)]:      # no longer shared: drop the dense point map (memory)
            out[i].pop('pts', None)
    if depths is not None:
        # geometry from the calibrated metric depth (MoGe-2); MapAnything supplies the poses. Its own
        # output depth drifts from the input (0.72x LiDAR measured) while its poses stay metric (~5 %).
        for i, p in out.items():
            d = np.asarray(depths[i], np.float32)
            h, w = p['depth'].shape
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
            p['depth'] = np.where(p['mask'], d, 0)
    return {i: _to_view(p) for i, p in out.items()}, report


# ---------------------------------------------------------------- SfM poses (COLMAP / GLOMAP)
def sfm_poses(bgr_of, ids, workdir, sequential=True, max_side=1024, log=print):
    """Camera poses from classical SfM: SIFT features, sequential (video) or exhaustive (photos)
    matching, global mapping. Returns {id: (T_wc 4x4 up to scale, K 3x3 at the input image
    resolution, sparse (u, v, z) observations in that image)}. Feature tracks across hundreds of
    frames give a globally consistent trajectory that chained network windows cannot on fast pans."""
    import pycolmap
    workdir = Path(workdir); img_dir = workdir / 'images'; img_dir.mkdir(parents=True, exist_ok=True)
    db = workdir / 'database.db'
    if db.exists():
        db.unlink()
    scale = None
    names = {}
    for i in ids:
        im = bgr_of(i)
        if scale is None:
            scale = min(1.0, max_side / max(im.shape[:2]))
        small = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else im
        names[i] = f'{int(i):05d}.jpg' if isinstance(i, (int, np.integer)) else f'{i}.jpg'
        cv2.imwrite(str(img_dir / names[i]), small, [cv2.IMWRITE_JPEG_QUALITY, 95])
    reader = pycolmap.ImageReaderOptions(); reader.camera_model = 'SIMPLE_RADIAL'
    pycolmap.extract_features(db, img_dir, image_names=list(names.values()), camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader)
    if sequential:
        po = pycolmap.SequentialPairingOptions(); po.overlap = 12; po.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=po)
    else:
        pycolmap.match_exhaustive(db)
    log(f'[sfm] features + matches for {len(names)} images; global mapping...')
    recs = pycolmap.global_mapping(db, img_dir, workdir / 'sparse')
    if not recs:
        return {}
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    log(f'[sfm] {rec.num_reg_images()}/{len(names)} images registered, {rec.num_points3D()} points'
        f'{f" ({len(recs)} models, largest kept)" if len(recs) > 1 else ""}')
    by_name = {v: k for k, v in names.items()}
    out = {}
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cw = img.cam_from_world()
        R = cw.rotation.matrix(); t = cw.translation
        T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ t
        K = img.camera.calibration_matrix().copy(); K[:2] /= scale
        obs = []
        for p2 in img.points2D:
            if p2.has_point3D():
                X = rec.points3D[p2.point3D_id].xyz
                z = (R @ X + t)[2]
                if z > 0:
                    obs.append((p2.xy[0] / scale, p2.xy[1] / scale, z))
        out[by_name[img.name]] = (T, K, np.array(obs).reshape(-1, 3))
    return out


def fuse_sfm_depth(views, sfm, rgb_size, log=print):
    """Bring network depth into the SfM frame. Per view, the network depth is rescaled to agree with
    the SfM points it sees (removing the network's per-window scale wobble); the SfM's arbitrary
    unit is converted to metres by the median network/SfM ratio over all views (the network's
    metric prior, averaged over the whole walk). Views SfM could not register are dropped."""
    ratios = {}
    for i, (T, K, obs) in sfm.items():
        v = views[i]
        if len(obs) < 15:
            continue
        h, w = v.depth.shape
        u = (obs[:, 0] * w / rgb_size[0]).astype(int); q = (obs[:, 1] * h / rgb_size[1]).astype(int)
        ok = (u >= 0) & (u < w) & (q >= 0) & (q < h)
        d = v.depth[q[ok], u[ok]]
        m = d > 0
        if m.sum() >= 10:
            ratios[i] = float(np.median(d[m] / obs[ok][m, 2]))
    if not ratios:
        return {}, None
    s = float(np.median(list(ratios.values())))       # metres per SfM unit
    spread = float(np.percentile(list(ratios.values()), 84) / np.percentile(list(ratios.values()), 16)) ** 0.5 - 1
    log(f'[sfm] metric scale {s:.4f} m/unit from {len(ratios)} views (per-view network scale spread +-{spread * 100:.0f} %)')
    out = {}
    for i, r in ratios.items():
        T, K, _ = sfm[i]
        v = views[i]
        h, w = v.depth.shape
        Kd = K.copy(); Kd[0] *= w / rgb_size[0]; Kd[1] *= h / rgb_size[1]
        T = T.copy(); T[:3, 3] *= s
        out[i] = _View((v.depth * (s / r)).astype(np.float32), v.conf, Kd, T)
    return out, {'metres_per_unit': s, 'views_scaled': len(ratios), 'per_view_scale_spread': round(spread, 4)}


# ---------------------------------------------------------------- tiers
VIDEO_ERRORS = {'face_sys': 0.02, 'drift_per_m': 0.01, 'scale_rel': 0.03}    # scale: MoGe-2 calibrated, see mono.py
PHOTO_ERRORS = {'face_sys': 0.03, 'drift_per_m': 0.015, 'scale_rel': 0.12}


def jpg_shape(buf):
    im = cv2.imdecode(buf, cv2.IMREAD_REDUCED_GRAYSCALE_2)
    return im.shape[1] * 2, im.shape[0] * 2


def sharpness(im):
    g = cv2.cvtColor(cv2.resize(im, (270, 480) if im.shape[0] > im.shape[1] else (480, 270)), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


KEY_SHIFT = 0.12   # new keyframe once the view has moved this fraction of the image width
KEY_MAX_GAP_S = 0.5


def _small(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 160 / max(g.shape)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)


def video_capture(path, every_s=None, rotate=None, sfm=False, intrinsics=None, rec=None,
                  cache=None, log=print):
    """Handheld video -> ReconCapture.

    Motion-adaptive keyframes: the image shift since the last keyframe is tracked (phase
    correlation on a 160 px thumbnail); a new keyframe is taken when the view has moved KEY_SHIFT
    of the image (fast pans get dense keyframes, the effect of slow motion without inventing
    frames) or after KEY_MAX_GAP_S, picking the sharpest of the last few frames."""
    vc = cv2.VideoCapture(str(path))
    fps = vc.get(cv2.CAP_PROP_FPS)
    n = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    max_gap = int(round(fps * (every_s or KEY_MAX_GAP_S)))
    keys, jpg, recent = [], {}, []
    aspect, crop_xy, last_small, last_i = None, None, None, -10 ** 9
    win_hann = None

    def take(i, bgr):
        nonlocal aspect, crop_xy, last_small, last_i
        if aspect is None:
            size_ = model_size(bgr.shape[1], bgr.shape[0])
            aspect = size_[0] / size_[1]
        c, crop_xy = crop_to_aspect(bgr, aspect)
        keys.append(i); jpg[i] = cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 93])[1]
        last_small, last_i = _small(bgr), i

    i = 0
    while True:
        ok, f = vc.read()
        if not ok:
            break
        if i % 2 == 0:
            bgr = f if rotate is None else cv2.rotate(f, rotate)
            sm = _small(bgr)
            recent = (recent + [(sharpness(bgr), i, bgr)])[-3:]
            if last_small is None:
                take(i, bgr)
            else:
                if win_hann is None:
                    win_hann = cv2.createHanningWindow(sm.shape[::-1], cv2.CV_32F)
                (dx, dy), resp = cv2.phaseCorrelate(last_small, sm, win_hann)
                shift = np.hypot(dx, dy) / max(sm.shape) if resp > 0.05 else 1.0
                if (shift >= KEY_SHIFT or i - last_i >= max_gap) and i - last_i >= 2:
                    _, bi, bb = max(recent, key=lambda r: r[0])
                    if bi <= last_i:
                        bi, bb = i, bgr
                    take(bi, bb)
                    recent = []
        i += 1
    vc.release()
    size = model_size(*jpg_shape(jpg[keys[0]]))
    bgr_of = lambda k: cv2.imdecode(jpg[k], cv2.IMREAD_COLOR)
    rgb_of = lambda k: cv2.cvtColor(bgr_of(k), cv2.COLOR_BGR2RGB)
    c0 = bgr_of(keys[0])
    rgb_size = (c0.shape[1], c0.shape[0])
    log(f'[rgb] video {n} frames @ {fps:.0f} fps -> {len(keys)} motion-adaptive keyframes '
        f'({len(keys) / (n / fps):.1f}/s), model size {size}')
    idx = list(range(len(keys)))
    Ks = None
    if intrinsics is not None:
        x0, y0 = crop_xy
        K = np.array(intrinsics, float).copy(); K[0, 2] -= x0; K[1, 2] -= y0
        Ks = [K] * len(keys)
    views, report = _cached_reconstruct(cache, lambda j: rgb_of(keys[j]), idx, size, Ks, None, None, rec, log,
                                        sequential=True)
    scale_info = None
    if sfm:   # poses from feature tracks; network depth rescaled into that frame
        views, scale_info = _cached_sfm(cache, views, lambda j: bgr_of(keys[j]), idx, rgb_size, True, log)
        idx = sorted(views)
    vlist = [views[j] for j in idx]
    gravity_align(vlist)
    kk = [keys[j] for j in idx]
    cap = ReconCapture(vlist, lambda m: bgr_of(kk[m]), rgb_size,
                       fps=len(keys) / (n / fps), tier='video', error_model=VIDEO_ERRORS,
                       names=[f'frame{k}' for k in kk])
    cap.source_frames = kk
    cap.times_s = [round(k / fps, 2) for k in kk]
    cap.alignment = {'windows': report, 'sfm_scale': scale_info, 'keyframes': len(keys), 'registered': len(kk)}
    return cap


def _cached_sfm(cache, views, bgr_of, idx, rgb_size, sequential, log):
    f = Path(cache).parent / 'sfm.npz' if cache else None
    if f is not None and f.exists():
        z = np.load(f, allow_pickle=True)
        sfm = {int(i): (T, K, o) for i, T, K, o in zip(z['ids'], z['T'], z['K'], z['obs'])}
        log(f'[sfm] replaying cached poses for {len(sfm)} views')
    else:
        work = (Path(cache).parent if cache else Path('/tmp')) / 'sfm_work'
        sfm = sfm_poses(bgr_of, idx, work, sequential=sequential, log=log)
        if f is not None:
            ids = sorted(sfm)
            np.savez(f, ids=np.array(ids), T=np.stack([sfm[i][0] for i in ids]), K=np.stack([sfm[i][1] for i in ids]),
                     obs=np.array([sfm[i][2] for i in ids], dtype=object))
    if len(sfm) < 0.6 * len(idx):
        log(f'[sfm] only {len(sfm)}/{len(idx)} views registered (texture-poor or fast motion): '
            f'keeping the network window chain poses')
        return views, {'used': False, 'registered': len(sfm)}
    out, info = fuse_sfm_depth(views, sfm, rgb_size, log=log)
    info['used'] = True
    return out, info


def mono_depths(load, idx, size, Ks=None, log=print, n_fov=24):
    """MoGe-2 metric depth for every view at the model resolution. Without known intrinsics, one
    focal length for the whole video is estimated first (median MoGe FOV over n_fov frames)."""
    from .mono import MetricDepth
    md = MetricDepth()
    im0 = load(idx[0]); W, H = im0.shape[1], im0.shape[0]
    if Ks is None:
        sample = [idx[k] for k in np.linspace(0, len(idx) - 1, min(n_fov, len(idx))).astype(int)]
        fovs = [md.infer(cv2.resize(load(i), size, interpolation=cv2.INTER_AREA))[1] for i in sample]
        fov = float(np.median(fovs))
        f = W / 2 / np.tan(np.radians(fov) / 2)
        K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
        Ks = [K] * len(idx)
        log(f'[rgb] focal from MoGe-2: h-FOV {fov:.1f} deg (spread {np.std(fovs):.1f}) -> {f:.0f} px')
    depths = {}
    for n, i in enumerate(idx):
        K = Ks[n] if isinstance(Ks, list) else Ks[i]
        fov = float(np.degrees(2 * np.arctan(W / 2 / K[0, 0])))
        d, _ = md.infer(cv2.resize(load(i), size, interpolation=cv2.INTER_AREA), fov_x_deg=fov)
        depths[i] = d.astype(np.float16)
    md.free()
    log(f'[rgb] MoGe-2 metric depth for {len(depths)} views (scale calibrated x1/{__import__("brynx.mono", fromlist=["x"]).SCALE_CAL})')
    return depths, Ks


def _cached_reconstruct(cache, load, idx, size, Ks, sk, groups, rec, log, sequential=False):
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        log(f'[rgb] replaying cached reconstruction {cache}')
        D, C, K, T = z['depth'], z['conf'], z['K'], z['T']     # each z[...] access decompresses: read once
        views = {int(i): _View(D[k], C[k], K[k], T[k]) for k, i in enumerate(z['ids'])}
        return views, list(z['report'])
    if sequential:
        depths, Ks = mono_depths(load, idx, size, Ks, log=log)
        views, report = reconstruct_sequential(load, idx, size, Ks, rec=rec, log=log, depths=depths)
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


def photo_capture(folder, rec=None, cache=None, log=print):
    """Per-room photo folders -> ReconCapture: each room reconstructed on its own, rooms stitched
    through doorway photos (photo_stitch). Rooms with no overlapping photo stay unplaced and are
    reported, never guessed."""
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
    views, report = _cached_photo_stitch(cache, crops, Kc, room_of, size, rec, log)
    idx = sorted(views)
    vlist = [views[j] for j in idx]
    gravity_align(vlist)
    cap = ReconCapture(vlist, lambda m: cv2.cvtColor(crops[idx[m]], cv2.COLOR_RGB2BGR),
                       (crops[0].shape[1], crops[0].shape[0]), fps=1.0, tier='photo', error_model=PHOTO_ERRORS,
                       names=[f'{room_of[i]}/{paths[i].name}' for i in idx])
    cap.room_of = [room_of[i] for i in idx]
    cap.alignment = report
    return cap


# ---------------------------------------------------------------- photo stitching
MAX_LINK_RESIDUAL_M = 0.25   # a doorway photo whose two reconstructions disagree more than this is not a link

def photo_links(crops, room_of, min_inliers=25, per_room=4, log=print):
    """Which photos of one room look into another room: SIFT + RANSAC inliers between a photo and
    every photo of the other folders. Returns {room B: [(photo id from another room, inliers, room A)]}."""
    sift = cv2.SIFT_create(nfeatures=3000)
    feats = []
    for c in crops:
        g = cv2.cvtColor(cv2.resize(c, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        feats.append(sift.detectAndCompute(g, None))
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def inliers(i, j):
        (ki, di), (kj, dj) = feats[i], feats[j]
        if di is None or dj is None or len(ki) < 20 or len(kj) < 20:
            return 0
        m = [a for a, b in bf.knnMatch(di, dj, k=2) if a.distance < 0.75 * b.distance]
        if len(m) < 15:
            return 0
        p1 = np.float32([ki[a.queryIdx].pt for a in m]); p2 = np.float32([kj[a.trainIdx].pt for a in m])
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, 2.0, 0.999)
        return int(mask.sum()) if mask is not None else 0

    rooms = sorted(set(room_of))
    links = {r: [] for r in rooms}
    for i in range(len(crops)):
        for B in rooms:
            if B == room_of[i]:
                continue
            best = max(inliers(i, j) for j in range(len(crops)) if room_of[j] == B)
            if best >= min_inliers:
                links[B].append((i, best, room_of[i]))
    for B in rooms:
        links[B] = sorted(links[B], key=lambda x: -x[1])[:per_room]
    log(f"[photo] links: {sum(len(v) for v in links.values())} photo->room views "
        f"({', '.join(f'{B}<-{sorted(set(a for _, _, a in v))}' for B, v in links.items() if v)})")
    return links


def photo_stitch(crops, Ks, room_of, size, rec=None, log=print):
    """Per-room reconstructions placed in one frame through 'guest' photos.

    Room B is reconstructed from its own photos plus a few photos taken in other rooms that look
    into B (doorway shots). A guest photo p from room A is then reconstructed twice - in A's run and
    in B's run - so its pixels give dense 3-D correspondences between the two rooms' frames. A rigid
    transform (each room keeps its own metric scale) is fitted per link; the rooms are placed along
    the maximum-confidence spanning tree from the room with most photos."""
    rec = rec or Reconstructor()
    rooms = sorted(set(room_of))
    links = photo_links(crops, room_of, log=log)
    runs = {}
    for B in rooms:
        members = [i for i in range(len(crops)) if room_of[i] == B]
        guests = [i for i, _, _ in links[B]]
        ids = members + guests
        pr = rec.infer([crops[i] for i in ids], size, [Ks[i] for i in ids])
        runs[B] = {'members': dict(zip(members, pr[:len(members)])), 'guests': dict(zip(guests, pr[len(members):]))}
        # level each room on its own (floor normal -> +Y) so unlinked rooms are still upright
        tmp = [_to_view(dict(p, T=p['T'])) for p in pr[:len(members)]]
        Rg = gravity_align(tmp)
        for p in pr:
            p['T'] = p['T'].copy(); p['T'][:3, :3] = Rg @ p['T'][:3, :3]; p['T'][:3, 3] = Rg @ p['T'][:3, 3]
            p['pts'] = p['pts'] @ Rg.T
        log(f'[photo] {B}: {len(members)} photos + {len(guests)} guest views')
    # link transforms room A -> room B from each guest photo
    edges = []
    for B in rooms:
        for p, nin, A in links[B]:
            src, dst = _corr(runs[A]['members'][p], runs[B]['guests'][p])
            if len(src) < 200:
                continue
            s_, R, t, res = robust_sim3(src, dst)
            _, R1, t1 = umeyama(src, dst, scale=False)
            r1 = np.linalg.norm(dst - (src @ R1.T + t1), axis=1)
            res1 = float(np.median(r1[r1 <= np.quantile(r1, 0.7)]))
            if res1 > MAX_LINK_RESIDUAL_M:
                log(f'[photo] reject link {A}->{B} via photo {p}: residual {res1 * 100:.0f} cm')
                continue
            edges.append({'from': A, 'to': B, 'photo': int(p), 'inliers': nin, 'R': R1, 't': t1, 'residual_m': res1,
                          'scale_disagreement': float(s_)})
    for e in edges:
        e['score'] = e['inliers'] / (1 + 10 * e['residual_m'])
    # maximum spanning tree (Prim) from the room with most photos
    root = max(rooms, key=lambda r: room_of.count(r))
    placed = {root: np.eye(4)}
    report = {'root': root, 'links': [], 'unplaced': []}
    while True:
        cand = []
        for e in edges:
            if e['from'] in placed and e['to'] not in placed:      # T_root<-A known; need T_root<-B = T_root<-A @ T_A<-B
                M = np.eye(4); M[:3, :3] = e['R']; M[:3, 3] = e['t']        # maps A -> B
                cand.append((e['score'], e['to'], placed[e['from']] @ np.linalg.inv(M), e))
            elif e['to'] in placed and e['from'] not in placed:
                M = np.eye(4); M[:3, :3] = e['R']; M[:3, 3] = e['t']
                cand.append((e['score'], e['from'], placed[e['to']] @ M, e))
        if not cand:
            break
        sc, room, T, e = max(cand, key=lambda c: c[0])
        placed[room] = T
        report['links'].append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k not in ('R', 't')})
        log(f"[photo] place {room} via {e['from']}->{e['to']} photo {e['photo']}: {e['inliers']} inliers, residual {e['residual_m'] * 100:.1f} cm")
    report['unplaced'] = [r for r in rooms if r not in placed]
    if report['unplaced']:
        log(f"[photo] not connected to the plan (no doorway overlap): {report['unplaced']}")
    # unplaced rooms: still measured, laid out in a row beside the stitched part, flagged in the plan
    xmax = max(float(placed[r][0, 3]) for r in placed) + 8.0
    for k, B in enumerate(report['unplaced']):
        T = np.eye(4); T[0, 3] = xmax + 9.0 * k
        placed[B] = T
    out = {}
    for B, T in placed.items():
        for i, p in runs[B]['members'].items():
            p['T'] = T @ p['T']
            out[i] = _to_view(p)
    return out, report


def _cached_photo_stitch(cache, crops, Ks, room_of, size, rec, log):
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=True)
        if 'stitch' in z.files:
            log(f'[rgb] replaying cached reconstruction {cache}')
            D, C, K, T = z['depth'], z['conf'], z['K'], z['T']
            return {int(i): _View(D[k], C[k], K[k], T[k]) for k, i in enumerate(z['ids'])}, z['report'].item()
    views, report = photo_stitch(crops, Ks, room_of, size, rec=rec, log=log)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        ids = sorted(views)
        np.savez_compressed(cache, ids=np.array(ids), depth=np.stack([views[i].depth for i in ids]),
                            conf=np.stack([views[i].conf for i in ids]), K=np.stack([views[i].K for i in ids]),
                            T=np.stack([views[i].T for i in ids]), report=np.array(report, dtype=object), stitch=True)
    return views, report
