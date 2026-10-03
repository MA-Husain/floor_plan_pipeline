#!/usr/bin/env python
"""One command per capture, any tier:

  python run_capture.py <stray_scanner_folder>     LiDAR tier  (rgb.mp4 + depth/ + odometry.csv)
  python run_capture.py <walkthrough.mov|.mp4>     video tier  (any iPhone, no depth, no poses)
  python run_capture.py <folder_of_room_folders>   photo tier  (one sub-folder of stills per room)

Options: --out outputs/<name>  --no-detect  --no-damage  --live (ignore cached model outputs)
         --drift auto|on|off   --tier lidar|video|photo (override auto-detection)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault('HF_HUB_OFFLINE', '1')     # everything runs from ./weights (scripts/fetch_weights.py)
from brynx import lidar, damage
from brynx.render import draw_plan


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def detect_tier(p):
    if p.is_file():
        return 'video'
    if (p / 'odometry.csv').exists():
        return 'lidar'
    if any(d.is_dir() for d in p.iterdir()):
        return 'photo'
    raise SystemExit(f'cannot tell the tier of {p}: expected a Stray folder, a video file or a folder of room folders')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-detect', action='store_true')
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--live', action='store_true', help='ignore cached model outputs')
    ap.add_argument('--drift', default='auto', choices=['auto', 'on', 'off'])
    ap.add_argument('--no-damage', action='store_true')
    ap.add_argument('--tier', default=None, choices=['lidar', 'video', 'photo'])
    a = ap.parse_args()
    src = Path(a.capture)
    tier = a.tier or detect_tier(src)
    out = Path(a.out or f'outputs/{src.stem if src.is_file() else src.name}')
    out.mkdir(parents=True, exist_ok=True)
    cache = None if a.live else out / 'cache'
    print(f'[run] {src} -> tier: {tier}')
    if tier == 'lidar':
        capture = str(src)
    else:
        from brynx import rgb
        recon_cache = None if cache is None else cache / 'reconstruction.npz'
        capture = (rgb.video_capture(src, cache=recon_cache) if tier == 'video'
                   else rgb.photo_capture(src, cache=recon_cache))
    plan, ctx = lidar.run(capture, step=a.step, detect=not a.no_detect, cache_dir=None if a.live else out / 'cache', drift_mode=a.drift)
    if not a.no_damage:
        regions, inspected, _ = damage.scan(ctx['capture'], ctx['poses'], plan, ctx,
                                            cache=None if a.live else out / 'cache' / 'damage_cells.json')
        flags = damage.concealed_flags(regions, plan)
        plan['damage'] = {'inspected': inspected, 'regions': regions}
        plan['concealed_damage_flags'] = flags
        plan['scope'] = damage.scope_items(regions, flags, plan)
    (out / 'plan.json').write_text(json.dumps(plan, indent=1, cls=NpEncoder))
    draw_plan(plan, out / 'plan.png')
    print(f'wrote {out}/plan.json, {out}/plan.png')


if __name__ == '__main__':
    main()
