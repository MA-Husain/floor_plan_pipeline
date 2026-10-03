#!/usr/bin/env python
"""Video tier validated against LiDAR on the same walk.

A Stray Scanner capture holds an ordinary RGB video *and* LiDAR depth + ARKit poses. The video
tier is run on the RGB alone (depth and poses ignored); the result is compared with the sensor:
  - metric scale: similarity fit of the RGB camera trajectory onto the ARKit trajectory
  - depth: per-keyframe median ratio predicted / LiDAR depth (confidence 2 pixels)
Writes outputs/<capture>_video/rgb_vs_lidar.json.
usage: python tools/rgb_vs_lidar.py <stray_capture> [--intrinsics]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynz import rgb
from brynz.capture import StrayCapture


def main(cap_dir, use_k=False):
    cap = StrayCapture(cap_dir)
    out = Path(f'outputs/{Path(cap_dir).name}_video'); out.mkdir(parents=True, exist_ok=True)
    K = None
    if use_k:   # Stray RGB is stored landscape; rotated 90 CW to upright -> rotate K too
        K0, H0 = cap.K_rgb(0), cap.rgb_size[1]
        K = np.array([[K0[1, 1], 0, H0 - 1 - K0[1, 2]], [0, K0[0, 0], K0[0, 2]], [0, 0, 1.0]])
    rc = rgb.video_capture(Path(cap_dir) / 'rgb.mp4', rotate=cv2.ROTATE_90_CLOCKWISE, intrinsics=K,
                           cache=out / 'cache' / ('reconstruction_K.npz' if use_k else 'reconstruction.npz'))
    keys = [k for k in rc.source_frames if k < cap.n]
    A = cap.poses()[keys][:, :3, 3]
    B = rc.positions[:len(keys)]
    s, R, t, res = rgb.robust_sim3(B, A, keep=0.8)
    # depth ratio on keyframes (both upright)
    ratios = []
    for j, k in enumerate(keys[::5]):
        f = cap.frame(k)
        d = cv2.rotate(f.depth, cv2.ROTATE_90_CLOCKWISE); c = cv2.rotate(f.confidence, cv2.ROTATE_90_CLOCKWISE)
        pd = cv2.resize(rc.views[j * 5].depth, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (c == 2) & (d > 0.3) & (d < 4) & (pd > 0)
        if m.sum() > 200:
            ratios.append(float(np.median(pd[m] / d[m])))
    rep = {'capture': cap_dir, 'intrinsics_given': use_k, 'keyframes': len(keys),
           'trajectory_scale_lidar_over_rgb': round(float(s), 4),
           'scale_error_pct': round((1 / s - 1) * 100, 2),
           'trajectory_residual_m': round(res, 3),
           'depth_ratio_rgb_over_lidar': {'median': round(float(np.median(ratios)), 4),
                                         'p10': round(float(np.percentile(ratios, 10)), 4),
                                         'p90': round(float(np.percentile(ratios, 90)), 4)},
           'chunk_alignment': rc.alignment}
    (out / ('rgb_vs_lidar_K.json' if use_k else 'rgb_vs_lidar.json')).write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps({k: v for k, v in rep.items() if k != 'chunk_alignment'}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], '--intrinsics' in sys.argv)
