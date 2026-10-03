#!/usr/bin/env python
"""Ceiling evidence map: top view of every downward-facing surface above 1.9 m, coloured by
height, with the space outlines and the per-space verdict. Shows where a ceiling was actually seen.
usage: python tools/ceiling_view.py <capture> <out.png>"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynz.capture import StrayCapture
from brynz.cloud import build_cloud, voxel_reduce
from brynz.geometry import plan_coords, rotate_dirs


def main(cap_dir, out):
    plan = json.load(open(f'outputs/{Path(cap_dir).name}/plan.json'))
    yaw, floor = plan['frame']['yaw_rad'], plan['frame']['floor_y_world']
    cap = StrayCapture(cap_dir)
    cloud = voxel_reduce(build_cloud(cap, step=10, poses=cap.poses()), 0.03)
    uv, h = plan_coords(cloud['xyz'], yaw, floor)
    ny = cloud['normal'][:, 1]
    m = (ny < -0.95) & (h > 1.9) & (h < 4.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    sc = ax.scatter(uv[m, 0], uv[m, 1], c=h[m], s=1, cmap='viridis', vmin=2.0, vmax=3.3)
    plt.colorbar(sc, ax=ax, shrink=0.6, label='height of downward-facing surface (m)')
    for r in plan['rooms']:
        P = np.array(r['polygon'] + [r['polygon'][0]])
        ax.plot(P[:, 0], P[:, 1], 'k-', lw=1)
        c = r['ceiling_height']
        txt = f"{r['name']}\n" + (f"ceiling {c['value_m']:.2f} m\ncover {c['coverage']:.0%}" if c else 'ceiling not seen')
        ax.text(*P[:-1].mean(0), txt, ha='center', va='center', fontsize=8,
                bbox=dict(fc='white', alpha=0.7, lw=0))
    ax.set_aspect('equal'); ax.set_title(f'Ceiling evidence - {Path(cap_dir).name}')
    fig.savefig(out, dpi=110, bbox_inches='tight')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
