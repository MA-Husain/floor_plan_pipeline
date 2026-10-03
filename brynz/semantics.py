"""Room typing from open-vocabulary object detection (YOLO-World, local weights).

Each detection is lifted to 3D with the LiDAR depth under its box and dropped into the plan;
rooms are typed by weighted votes of the objects they contain. No hand-written frame ranges.
"""
from pathlib import Path
import numpy as np

CLASSES = ['bed', 'toilet', 'shower', 'bathtub', 'sink', 'refrigerator', 'stove', 'kitchen cabinet',
           'microwave', 'sofa', 'television', 'dining table', 'desk', 'wardrobe', 'staircase',
           'stair railing', 'washing machine', 'office chair']

# object -> {room type: weight}
VOTES = {
    'bed': {'Bedroom': 3}, 'wardrobe': {'Bedroom': 1},
    'toilet': {'Bathroom': 5}, 'shower': {'Bathroom': 4}, 'bathtub': {'Bathroom': 3},
    'sink': {'Bathroom': 1, 'Kitchen': 1},
    'refrigerator': {'Kitchen': 3}, 'stove': {'Kitchen': 4}, 'kitchen cabinet': {'Kitchen': 2}, 'microwave': {'Kitchen': 2},
    'washing machine': {'Utility': 3},
    'sofa': {'Living': 3}, 'television': {'Living': 1, 'Bedroom': 0.5},
    'dining table': {'Dining': 2}, 'desk': {'Study': 2}, 'office chair': {'Study': 1},
    'staircase': {'Stairs': 3}, 'stair railing': {'Stairs': 2},
}

WEIGHTS = Path(__file__).resolve().parent.parent / 'weights' / 'yolov8x-worldv2.pt'


def rot_cw_to_orig(x, y, H):
    """Detection runs on the image rotated 90deg CW (phone held portrait). Map back."""
    return y, H - 1 - x


def detect_objects(capture, poses, every_s=0.5, conf=0.3, device=None, cache=None):
    """Per-frame open-vocabulary detections lifted to 3-D. `cache` (path): replay if present,
    else run live and write it - deterministic, so cached and live runs give identical plans."""
    import json
    from pathlib import Path
    if cache and Path(cache).exists():
        return json.load(open(cache))
    out = _detect_live(capture, poses, every_s, conf, device)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(cache, 'w'))
    return out


def _detect_live(capture, poses, every_s, conf, device):
    from ultralytics import YOLOWorld
    import cv2
    import torch
    device = device or ('mps' if torch.backends.mps.is_available() else 'cpu')
    model = YOLOWorld(str(WEIGHTS))
    model.set_classes(CLASSES)
    step = max(1, int(round(capture.fps * every_s)))
    idx = list(range(0, capture.n, step))
    W, H = capture.rgb_size
    sx, sy = capture.depth_size[0] / W, capture.depth_size[1] / H
    out = []
    for i, img in capture.rgb_frames(idx):
        upright = getattr(capture, 'upright', False)
        r = model.predict(img if upright else cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), conf=conf, verbose=False, device=device)[0]
        if len(r.boxes) == 0:
            continue
        f = capture.frame(i, poses[i])
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            (ox1, oy1), (ox2, oy2) = ((x1, y1), (x2, y2)) if upright else (rot_cw_to_orig(x1, y1, H), rot_cw_to_orig(x2, y2, H))
            u0, u1 = sorted([ox1 * sx, ox2 * sx]); v0, v1 = sorted([oy1 * sy, oy2 * sy])
            cu, cv_ = (u0 + u1) / 2, (v0 + v1) / 2
            wu, wv = (u1 - u0) * 0.2, (v1 - v0) * 0.2
            patch = f.depth[int(max(cv_ - wv, 0)):int(cv_ + wv) + 1, int(max(cu - wu, 0)):int(cu + wu) + 1]
            patch = patch[patch > 0.1]
            if patch.size < 3:
                continue
            z = float(np.median(patch))
            if z > 4.5:
                continue
            K = f.K
            p = np.array([(cu - K[0, 2]) * z / K[0, 0], (cv_ - K[1, 2]) * z / K[1, 1], z])
            pw = f.T_wc[:3, :3] @ p + f.T_wc[:3, 3]
            out.append({'frame': int(i), 'class': CLASSES[int(c)], 'conf': float(s), 'xyz': pw.tolist(),
                        'bbox_rot': [float(x1), float(y1), float(x2), float(y2)], 'depth_m': z})
    return out


def object_instances(dets, radius=0.6, min_views=2):
    """Cluster per-frame detections into physical object instances (3-D, same class).
    Each instance's position is the median of its sightings, so a few sightings whose depth
    landed on a door frame or the next room are outvoted."""
    inst = []
    for d in sorted(dets, key=lambda d: -d['conf']):
        p = np.array(d['uv'])
        for o in inst:
            if o['class'] == d['class'] and np.linalg.norm(np.median(o['_p'], 0) - p) < radius:
                o['_p'].append(p); o['_c'].append(d['conf']); o['frames'].append(d['frame'])
                break
        else:
            inst.append({'class': d['class'], '_p': [p], '_c': [d['conf']], 'frames': [d['frame']]})
    out = []
    for o in inst:
        if len(o['_p']) < min_views:
            continue
        out.append({'class': o['class'], 'uv': np.median(o['_p'], 0).round(3).tolist(), 'views': len(o['_p']),
                    'conf': round(float(np.max(o['_c'])), 3), 'frames': o['frames'][:10]})
    return out
