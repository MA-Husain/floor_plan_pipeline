"""Debug figure: wall points, inside evidence, cell rooms, openings and detected objects."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brynx import lidar
import brynx.cells as C

cap_dir = sys.argv[1]; out = sys.argv[2]; detect = '--detect' in sys.argv
plan, d = lidar.run(cap_dir, detect=detect)
json.dump(plan, open(out.replace('.png', '.json'), 'w'), default=float)
g = d['grid']; uv, h, ny = d['uv'], d['h'], d['ny']
ext = [g.origin[0], g.origin[0] + g.shape[1] * g.res, g.origin[1], g.origin[1] + g.shape[0] * g.res]
fig, ax = plt.subplots(1, 2, figsize=(22, 10))
ax[0].imshow(d['E'], origin='lower', extent=ext, cmap='Blues', alpha=0.5)
m = (np.abs(ny) < 0.3) & (h > 0.3) & (h < 1.6)
for a in ax:
    a.scatter(uv[m, 0][::4], uv[m, 1][::4], s=0.05, c='k')
    a.set_aspect('equal')
ax[0].set_title('inside evidence (floor | carved | tops)')
rng = np.random.default_rng(3)
for r in plan['rooms']:
    p = np.array(r['polygon'] + [r['polygon'][0]])
    ax[1].fill(p[:, 0], p[:, 1], alpha=0.35, color=rng.uniform(0.2, 1, 3)); ax[1].plot(p[:, 0], p[:, 1], 'b-', lw=1)
    c = p[:-1].mean(0)
    ax[1].text(c[0], c[1], f"{r['id']} {r.get('name','')}\n{r['area_m2']['value']:.1f}m² w{r['features']['inscribed_width']:.1f}\n{r.get('type_reason','')[:28]}", fontsize=7, ha='center')
for o in plan['openings']:
    a0, a1 = o['span']; col = {'door': 'r', 'window': 'c', 'opening': 'orange'}[o['type']]
    xy = ([a0, a1], [o['wall_coord']] * 2) if o['axis'] == 'u' else ([o['wall_coord']] * 2, [a0, a1])
    ax[1].plot(*xy, col, lw=4)
for o in plan.get('objects', []):
    if o['conf'] > 0.3:
        ax[1].plot(*o['uv'], 'x', ms=4, color='m'); ax[1].text(*o['uv'], o['class'][:6], fontsize=5, color='m')
t = np.array(plan['debug']['trajectory_uv']); ax[1].plot(t[:, 0], t[:, 1], 'g:', lw=0.6)
fig.savefig(out, dpi=60, bbox_inches='tight')
