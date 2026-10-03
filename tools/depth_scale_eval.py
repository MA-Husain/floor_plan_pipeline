#!/usr/bin/env python
"""Which monocular model gives the truest metric scale? Predicted depth vs iPhone LiDAR depth on
level frames of the Stray captures (RGB rotated upright). Reports per model the median ratio
pred/LiDAR and its spread across frames - the scale bias and noise the video/photo tiers inherit.
usage: python tools/depth_scale_eval.py [n_frames_per_capture]"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party' / 'MoGe'), str(ROOT / 'third_party' / 'Depth-Anything-3' / 'src')]
import cv2
import numpy as np
import torch

from brynx.capture import StrayCapture

DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
HF = str(ROOT / 'weights' / 'hf')


def frames(cap_dir, n):
    cap = StrayCapture(cap_dir)
    T = cap.poses()
    cand = [i for i in range(50, cap.n - 50, 5) if -0.3 < T[i, 1, 2] < 0.2]
    idx = [cand[k] for k in np.linspace(0, len(cand) - 1, n).astype(int)]
    K0, H0 = cap.K_rgb(0), cap.rgb_size[1]
    K = np.array([[K0[1, 1], 0, H0 - 1 - K0[1, 2]], [0, K0[0, 0], K0[0, 2]], [0, 0, 1.0]])   # upright
    out = []
    for i, im in cap.rgb_frames(idx):
        f = cap.frame(i)
        out.append((cv2.cvtColor(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB),
                    cv2.rotate(f.depth, cv2.ROTATE_90_CLOCKWISE), cv2.rotate(f.confidence, cv2.ROTATE_90_CLOCKWISE), K))
    return out


def ratio(pred, d, c):
    p = cv2.resize(pred, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
    m = (c == 2) & (d > 0.3) & (d < 4) & (p > 0) & np.isfinite(p)
    return float(np.median(p[m] / d[m])) if m.sum() > 300 else np.nan


def main(n=15):
    data = {c: frames(c, n) for c in ('c00a170fe1', 'c7d28f72c6', '1a8384c3f6')}
    res = {}
    from moge.model.v2 import MoGeModel
    from huggingface_hub import snapshot_download
    moge = MoGeModel.from_pretrained(snapshot_download('Ruicheng/moge-2-vitl-normal', cache_dir=HF) + '/model.pt').to(DEV).eval()
    for name, use_fov in (('moge2_freeFOV', False), ('moge2_trueFOV', True)):
        r = []
        for c, fr in data.items():
            for im, d, cf, K in fr:
                x = torch.from_numpy(im).permute(2, 0, 1).float().div(255).to(DEV)
                fov = float(np.degrees(2 * np.arctan(im.shape[1] / 2 / K[0, 0]))) if use_fov else None
                with torch.no_grad():
                    o = moge.infer(x, fov_x=fov, use_fp16=False)
                r.append((c, ratio(o['depth'].float().cpu().numpy(), d, cf)))
        res[name] = r
        v = np.array([x for _, x in r if np.isfinite(x)]); print(name, 'median', round(float(np.median(v)), 3), 'p16-p84', np.round(np.percentile(v, [16, 84]), 3), flush=True)
    del moge
    from depth_anything_3.api import DepthAnything3
    da3 = DepthAnything3.from_pretrained(snapshot_download('depth-anything/DA3METRIC-LARGE', cache_dir=HF)).to(DEV).eval()
    r = []
    for c, fr in data.items():
        for im, d, cf, K in fr:
            pred = da3.inference([im])
            net = pred.depth[0]
            f = K[0, 0] * net.shape[1] / im.shape[1]          # focal at the processed resolution
            r.append((c, ratio(f * net / 300.0, d, cf)))
    res['da3_metric_trueF'] = r
    summary = {}
    for name, r in res.items():
        v = np.array([x for _, x in r if np.isfinite(x)])
        per = {c: round(float(np.nanmedian([x for cc, x in r if cc == c])), 3) for c in data}
        summary[name] = {'median_ratio': round(float(np.median(v)), 3), 'p16': round(float(np.percentile(v, 16)), 3),
                         'p84': round(float(np.percentile(v, 84)), 3), 'per_capture': per, 'frames': int(len(v))}
        print(name, summary[name])
    Path('outputs/depth_scale_eval.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
