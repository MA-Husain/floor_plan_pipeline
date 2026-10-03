#!/usr/bin/env python
"""Drift ablation: the stitched footprint with the correction OFF (raw poses) and ON.

Writes <out>/drift_ablation.png (wall evidence + room outlines side by side, then overlaid) and
<out>/drift_ablation.json (wall crispness, footprint area, per-room areas, max pose correction).
usage: python tools/drift_ablation.py <capture> [out_dir]"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynx import lidar


def run(cap, mode):
    plan, ctx = lidar.run(cap, detect=False, drift_mode=mode, log=lambda *a: None)
    return plan, ctx, None


def main(cap, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    res = {m: run(cap, m) for m in ('off', 'on')}
    d = res['on'][0]['drift']     # raw vs corrected metrics, computed identically on both pose sets
    metric = {'off': (d['crispness_raw_mm'], d['revisit_error_raw_mm']),
              'on': (d['crispness_corrected_mm'], d['revisit_error_corrected_mm'])}
    # express both in the OFF frame: the correction changes yaw/floor only marginally, but the plan
    # frames are fitted independently, so align ON to OFF by the dominant-wall rotation already
    # shared (Manhattan) and a translation of the camera start point
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    cols = {'off': 'tab:red', 'on': 'tab:blue'}
    summary = {}
    for k, m in enumerate(('off', 'on')):
        plan, ctx, cr = res[m]
        uv, h, ny = ctx['uv'], ctx['h'], ctx['ny']
        w = (np.abs(ny) < 0.3) & (h > 0.3) & (h < 1.8)
        p = uv[w][::3]
        o = np.array(plan['debug']['trajectory_uv'][0])
        ax[k].scatter(p[:, 0] - o[0], p[:, 1] - o[1], s=0.05, c='k', alpha=0.3)
        ax[2].scatter(p[:, 0] - o[0], p[:, 1] - o[1], s=0.05, c=cols[m], alpha=0.15)
        for r in plan['rooms']:
            P = np.array(r['polygon'] + [r['polygon'][0]]) - o
            ax[k].plot(P[:, 0], P[:, 1], color=cols[m], lw=1.2)
        crisp, ghost = metric[m]
        ax[k].set_title(f"drift correction {m.upper()}\nrevisit ghosting {ghost:.1f} mm, wall crispness {crisp:.1f} mm, "
                        f"footprint {plan['footprint_m2']:.1f} m2, {len(plan['rooms'])} spaces")
        summary[m] = {'revisit_ghosting_mm': ghost, 'wall_crispness_mm': crisp, 'footprint_m2': plan['footprint_m2'],
                      'spaces': len(plan['rooms']), 'wall_faces': len(plan['wall_faces']),
                      'room_areas_m2': sorted(round(r['area_m2']['value'], 2) for r in plan['rooms'])}
    summary['correction'] = d
    ax[2].set_title('overlay: OFF (red) vs ON (blue) wall points')
    for a in ax:
        a.set_aspect('equal'); a.grid(alpha=0.2)
    fig.suptitle(f'Drift ablation - {Path(str(cap)).name}  (ghosting = median gap between two sightings of the same wall >= 20 s apart; '
                 f'crispness = median distance of wall points to their plane; lower is better)')
    fig.savefig(out / 'drift_ablation.png', dpi=110, bbox_inches='tight')
    (out / 'drift_ablation.json').write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps({m: summary[m] for m in ('off', 'on')}, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else f'outputs/{Path(sys.argv[1]).name}')
