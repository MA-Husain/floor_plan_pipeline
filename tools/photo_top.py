#!/usr/bin/env python
"""Top view of a photo-tier reconstruction, wall points coloured by room folder.
usage: python tools/photo_top.py <photo_root> <cache.npz> <out.png>"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynz import rgb
from brynz.cloud import build_cloud, voxel_reduce
from brynz.geometry import floor_ceiling, manhattan_yaw, plan_coords

cap = rgb.photo_capture(sys.argv[1], cache=sys.argv[2], log=lambda *a: None)
cl = voxel_reduce(build_cloud(cap, step=1, poses=cap.poses(), stride=2), 0.03)
fl, ce, _ = floor_ceiling(cl)
yaw = manhattan_yaw(cl)
uv, h = plan_coords(cl['xyz'], yaw, fl)
w = (np.abs(cl['normal'][:, 1]) < 0.3) & (h > 0.3) & (h < 1.8)
rooms = sorted(set(cap.room_of)); col = {r: plt.cm.tab10(i % 10) for i, r in enumerate(rooms)}
cam, _ = plan_coords(cap.poses()[:, :3, 3], yaw, fl)
fig, ax = plt.subplots(figsize=(11, 11))
ax.scatter(uv[w, 0], uv[w, 1], s=0.3, c=[col[cap.room_of[f]] for f in cl['frame'][w]])
for r in rooms:
    m = np.array([x == r for x in cap.room_of])
    ax.text(*cam[m].mean(0), r, fontsize=10, color=col[r], weight='bold')
ax.set_aspect('equal'); ax.grid(alpha=.3)
ax.set_title(f'photo tier: wall points by room folder; floor-ceiling {None if ce is None else round(ce - fl, 2)} m')
fig.savefig(sys.argv[3], dpi=75, bbox_inches='tight')
