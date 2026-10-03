#!/usr/bin/env python
"""One command per capture:  python run_capture.py <capture_dir> [--out outputs/<name>] [--no-detect] [--no-damage]"""
import argparse
import json
from pathlib import Path

import numpy as np

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('capture')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-detect', action='store_true')
    ap.add_argument('--step', type=int, default=None)
    ap.add_argument('--live', action='store_true', help='ignore cached model outputs')
    ap.add_argument('--drift', default='auto', choices=['auto', 'on', 'off'])
    ap.add_argument('--no-damage', action='store_true')
    a = ap.parse_args()
    out = Path(a.out or f'outputs/{Path(a.capture).name}')
    out.mkdir(parents=True, exist_ok=True)
    plan, ctx = lidar.run(a.capture, step=a.step, detect=not a.no_detect, cache_dir=None if a.live else out / 'cache', drift_mode=a.drift)
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
