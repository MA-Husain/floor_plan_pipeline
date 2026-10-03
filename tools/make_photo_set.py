#!/usr/bin/env python
"""Build a photo-tier test set from a LiDAR capture: per-room folders of stills, LiDAR plan as truth.

Follows the capture protocol (docs/CAPTURE_PROTOCOL.md, section C): per room ~6 photos looking in
different directions (sharpest frames, greedy max yaw spread) plus, for every opening of the room,
one photo taken near the opening looking into the neighbouring room. Frames come from the capture's
own RGB video, rotated upright; EXIF carries the 35 mm-equivalent focal length as an iPhone photo
would. The photo tier never sees depth or poses - those are only used here to pick the frames.
usage: python tools/make_photo_set.py <stray_capture> [out_dir] [--per-room 6]"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brynx.capture import StrayCapture
from brynx.geometry import plan_coords, rotate_dirs
from brynx.rgb import sharpness


def main(cap_dir, out_dir=None, per_room=6):
    name = Path(cap_dir).name
    out = Path(out_dir or f'data/photo_sets/{name}')
    plan = json.load(open(f'outputs/{name}/plan.json'))
    yaw, floor = plan['frame']['yaw_rad'], plan['frame']['floor_y_world']
    cap = StrayCapture(cap_dir)
    T = cap.poses()
    idx = np.arange(0, cap.n - 10, 4)
    uv, _ = plan_coords(T[idx, :3, 3], yaw, floor)
    look = rotate_dirs(T[idx][:, [0, 2], 2], yaw)            # camera optical axis in plan coords
    look /= np.linalg.norm(look, axis=1, keepdims=True) + 1e-9
    heading = np.arctan2(look[:, 1], look[:, 0])
    pitch = T[idx, 1, 2]                                       # optical axis world-y: avoid ceiling/floor sweeps
    rooms = {r['id']: r for r in plan['rooms']}
    polys = {k: Polygon(r['polygon']) for k, r in rooms.items()}
    inside = {k: np.array([P.buffer(-0.15).contains(Point(*p)) for p in uv]) for k, P in polys.items()}
    level = (pitch > -0.45) & (pitch < 0.25)        # near-level: room photos, not ceiling/floor sweeps
    chosen = {k: [] for k in rooms}
    # sharpness only for candidates
    cand = sorted({int(idx[j]) for k in rooms for j in np.nonzero(inside[k] & level)[0]})
    sharp = {}
    for i, im in cap.rgb_frames(cand[::2]):
        sharp[i] = sharpness(im)
    pos = {int(i): j for j, i in enumerate(idx)}
    for k in rooms:
        js = [pos[i] for i in sharp if inside[k][pos[i]]]
        if not js:
            continue
        js = sorted(js, key=lambda j: -sharp[int(idx[j])])[:max(per_room * 6, 12)]   # sharp pool
        pick = [js[0]]
        while len(pick) < min(per_room, len(js)):
            # spread over viewing direction and over position (metres count like ~1 rad)
            d = [min(abs(np.angle(np.exp(1j * (heading[j] - heading[q])))) + np.linalg.norm(uv[j] - uv[q])
                     for q in pick) for j in js]
            pick.append(js[int(np.argmax(d))])
        chosen[k] += [int(idx[j]) for j in pick]
    # doorway shots: near an opening, inside room A, looking towards room B
    doors = []
    for o in plan['openings']:
        if len(o['rooms']) != 2:
            continue
        a0, a1 = o['span']
        c = np.array([(a0 + a1) / 2, o['wall_coord']]) if o['axis'] == 'u' else np.array([o['wall_coord'], (a0 + a1) / 2])
        for A, B in (o['rooms'], o['rooms'][::-1]):
            toB = np.array(polys[B].centroid.coords[0]) - uv
            toB /= np.linalg.norm(toB, axis=1, keepdims=True) + 1e-9
            near = (np.linalg.norm(uv - c, axis=1) < 1.5) & np.array([polys[A].buffer(0.1).contains(Point(*p)) for p in uv])
            ok = near & level & ((look * toB).sum(1) > 0.8)
            if ok.any():
                j = np.nonzero(ok)[0]
                j = j[np.argmin(np.linalg.norm(uv[j] - c, axis=1))]
                chosen[A].append(int(idx[j])); doors.append((rooms[A]['name'], rooms[B]['name'], int(idx[j])))
    # write
    K0, H0 = cap.K_rgb(0), cap.rgb_size[1]
    fy = K0[0, 0]                                            # upright image: width = original height
    f35 = fy * 43.2666 / np.hypot(cap.rgb_size[1], cap.rgb_size[0])
    want = sorted({i for v in chosen.values() for i in v})
    frames = dict(cap.rgb_frames(want))
    manifest = {'source': cap_dir, 'per_room': {}, 'doorway_shots': doors}
    for k, ids in chosen.items():
        if not ids:
            continue
        d = out / rooms[k]['name'].replace(' ', '_').replace('/', '-')
        d.mkdir(parents=True, exist_ok=True)
        for i in sorted(set(ids)):
            im = cv2.cvtColor(cv2.rotate(frames[i], cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(im)
            ex = Image.Exif(); ifd = ex.get_ifd(0x8769); ifd[0xA405] = int(round(f35)); ex[0x8769] = ifd
            pil.save(d / f'frame_{i:05d}.jpg', quality=93, exif=ex)
        manifest['per_room'][d.name] = {'lidar_room': k, 'frames': sorted(set(ids))}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(f"{out}: {sum(len(set(v)) for v in chosen.values())} photos in {len(manifest['per_room'])} room folders, "
          f"{len(doors)} doorway shots")


if __name__ == '__main__':
    a = sys.argv
    main(a[1], a[2] if len(a) > 2 and not a[2].startswith('--') else None,
         int(a[a.index('--per-room') + 1]) if '--per-room' in a else 6)
