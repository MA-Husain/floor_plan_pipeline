"""Architectural rendering of a brynx plan JSON: solid walls, door gaps, windows, room labels
(name, size, ceiling height), overall dimension strings."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

INK = '#111111'
DIM = '#7a3e9d'
EXT_T = 0.20      # exterior wall drawing thickness (m)
GAP_FILL = 0.40   # gaps between rooms narrower than this are wall


def _geoms(g):
    return list(g.geoms) if hasattr(g, 'geoms') else [g]


INT_T = 0.11     # interior partition drawing thickness (m)


def wall_solid(rooms_poly):
    """Exterior shell + interior partitions. Rooms from the cell complex share edges exactly,
    so partitions are drawn as a band of INT_T centred on each shared boundary."""
    union = unary_union(rooms_poly)
    closed = union.buffer(GAP_FILL / 2, join_style='mitre').buffer(-GAP_FILL / 2, join_style='mitre')
    outer = closed.buffer(EXT_T, join_style='mitre').difference(union)
    shared = []
    for i in range(len(rooms_poly)):
        for j in range(i + 1, len(rooms_poly)):
            b = rooms_poly[i].boundary.intersection(rooms_poly[j].boundary)
            if not b.is_empty and b.length > 0.05:
                shared.append(b.buffer(INT_T / 2, cap_style='flat', join_style='mitre'))
    # thin gaps between rooms (double-faced walls) are already wall: fill them
    gaps = closed.difference(union)
    return unary_union([outer, gaps] + shared)


def opening_rect(o, depth=0.7):
    a0, a1 = o['span']
    c = o['wall_coord']
    if o['axis'] == 'u':
        return box(a0, c - depth / 2, a1, c + depth / 2)
    return box(c - depth / 2, a0, c + depth / 2, a1)


def draw_plan(plan, out_path, title='FLOOR PLAN', show_objects=False):
    rooms = [Polygon(r['polygon']).buffer(0) for r in plan['rooms']]
    walls = wall_solid(rooms)
    cuts = [opening_rect(o) for o in plan['openings']]
    walls_cut = walls.difference(unary_union(cuts)) if cuts else walls
    room_union = unary_union(rooms)

    allb = unary_union(rooms).buffer(EXT_T).bounds
    W, H = allb[2] - allb[0], allb[3] - allb[1]
    fig, ax = plt.subplots(figsize=(min(22, 3 + W * 1.25), min(22, 3.5 + H * 1.25)))
    ax.set_facecolor('white')

    for r, P in zip(plan['rooms'], rooms):
        for g in _geoms(P):
            ax.add_patch(MPoly(np.array(g.exterior.coords), closed=True, fc='#fbfaf7', ec='none', zorder=1))
    for g in _geoms(walls_cut):
        if g.is_empty:
            continue
        ax.add_patch(MPoly(np.array(g.exterior.coords), closed=True, fc=INK, ec=INK, lw=0.3, zorder=3))
        for hole in g.interiors:
            ax.add_patch(MPoly(np.array(hole.coords), closed=True, fc='#fbfaf7', ec='none', zorder=3))

    # wall thickness at each opening (for symbols): probe the uncut wall solid
    for o in plan['openings']:
        a0, a1 = o['span']; c = o['wall_coord']; w = a1 - a0
        probe = opening_rect(o, depth=0.8).intersection(walls)
        if probe.is_empty:
            t0, t1 = c - 0.06, c + 0.06
        else:
            b = probe.bounds
            t0, t1 = (b[1], b[3]) if o['axis'] == 'u' else (b[0], b[2])
        def P(al, pp):
            return (al, pp) if o['axis'] == 'u' else (pp, al)
        if o['type'] == 'window':
            for f in (0.0, 0.5, 1.0):
                pp = t0 + (t1 - t0) * f
                ax.plot(*zip(P(a0, pp), P(a1, pp)), color=INK, lw=0.9, zorder=4)
            for al in (a0, a1):
                ax.plot(*zip(P(al, t0), P(al, t1)), color=INK, lw=0.9, zorder=4)
        # doors and open passages are drawn as clean gaps in the wall (no swing symbols)

    # labels: name, size (or area for non-rectangular spaces), ceiling height when measured
    for r, Pg in zip(plan['rooms'], rooms):
        c = Pg.representative_point() if not Pg.centroid.within(Pg) else Pg.centroid
        du, dv = r['dimensions']['length_u']['value_m'], r['dimensions']['length_v']['value_m']
        small = Pg.area < 4 or min(du, dv) < 1.6
        fs = 8 if small else 11
        lines = [f"{du:.2f} × {dv:.2f} m" if Pg.area > 0.85 * du * dv else f"{Pg.area:.1f} m²"]
        if r.get('ceiling_height'):
            lines.append(f"ceiling {r['ceiling_height']['value_m']:.2f} m")
        if r.get('position_known') is False:
            lines.append('(position unknown)')   # photo tier: no doorway photo tied it to the plan
        rot = 90 if (small and dv > du * 1.8) else 0
        ax.text(c.x, c.y, r.get('name', 'Room').upper() + '\n' + '\n'.join(lines), ha='center', va='center',
                fontsize=fs, rotation=rot, linespacing=1.5, zorder=6, color=INK)

    if show_objects:
        for o in plan.get('objects', []):
            ax.plot(*o['uv'], '.', ms=2, color='tab:red', zorder=2)

    # overall dimension strings
    ub = unary_union(rooms).bounds
    off = EXT_T + 0.45
    _dim(ax, (ub[0], ub[3] + off), (ub[2], ub[3] + off), f"{ub[2] - ub[0]:.2f} m", horizontal=True)
    _dim(ax, (ub[0] - off, ub[1]), (ub[0] - off, ub[3]), f"{ub[3] - ub[1]:.2f} m", horizontal=False)

    ax.set_xlim(ub[0] - off - 0.8, ub[2] + 0.6); ax.set_ylim(ub[1] - 1.2, ub[3] + off + 0.7)
    ax.set_aspect('equal'); ax.axis('off')
    ax.text((ub[0] + ub[2]) / 2, ub[1] - 0.75, title, ha='center', va='center', fontsize=20, zorder=6)
    ax.plot([(ub[0] + ub[2]) / 2 - 1.4, (ub[0] + ub[2]) / 2 + 1.4], [ub[1] - 1.0] * 2, color=INK, lw=2)
    fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _dim(ax, p0, p1, text, horizontal):
    ax.annotate('', p1, p0, arrowprops=dict(arrowstyle='-', color=DIM, lw=1))
    tick = 0.15
    for p in (p0, p1):
        if horizontal:
            ax.plot([p[0], p[0]], [p[1] - tick, p[1] + tick], color=DIM, lw=1)
        else:
            ax.plot([p[0] - tick, p[0] + tick], [p[1], p[1]], color=DIM, lw=1)
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    ax.text(mx, my + (0.12 if horizontal else 0), text, ha='center', va='bottom' if horizontal else 'center',
            rotation=0 if horizontal else 90, color=DIM, fontsize=11,
            bbox=dict(fc='white', ec='none', pad=1) if not horizontal else None)
