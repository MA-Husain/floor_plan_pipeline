#!/usr/bin/env python
"""Top view of wall evidence for any capture (Stray folder, or an RGB-tier reconstruction cache),
coloured by time, with the camera path. usage: python tools/topview.py <capture|video|photos> <out.png> [--cache path]"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynx.cloud import build_cloud, voxel_reduce
from brynx.geometry import floor_ceiling, manhattan_yaw, plan_coords


def load(src, cache):
    p = Path(src)
    if p.is_file():
        from brynx import rgb
        return rgb.video_capture(p, cache=cache)
    if (p / 'odometry.csv').exists():
        from brynx.capture import StrayCapture
        return StrayCapture(p)
    from brynx import rgb
    return rgb.photo_capture(p, cache=cache)


if __name__ == '__main__':
    cache = sys.argv[sys.argv.index('--cache') + 1] if '--cache' in sys.argv else None
    cap = load(sys.argv[1], cache)
    poses = cap.poses()
    cl = voxel_reduce(build_cloud(cap, step=max(2, cap.n // 300), poses=poses, stride=2), 0.03)
    fl, ce, _ = floor_ceiling(cl)
    yaw = manhattan_yaw(cl)
    uv, h = plan_coords(cl['xyz'], yaw, fl)
    w = (np.abs(cl['normal'][:, 1]) < 0.3) & (h > 0.3) & (h < 1.8)
    cam, _ = plan_coords(poses[:, :3, 3], yaw, fl)
    fig, ax = plt.subplots(figsize=(12, 12))
    sc = ax.scatter(uv[w, 0], uv[w, 1], s=0.3, c=cl['frame'][w], cmap='viridis')
    ax.plot(cam[:, 0], cam[:, 1], 'r-', lw=0.8)
    ax.plot(*cam[0], 'go', ms=10)
    plt.colorbar(sc, ax=ax, shrink=0.5, label='frame index (time)')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_title(f'{sys.argv[1]}: wall points by time, camera path red; floor-ceiling {"" if ce is None else round(ce - fl, 2)} m')
    fig.savefig(sys.argv[2], dpi=80, bbox_inches='tight')
