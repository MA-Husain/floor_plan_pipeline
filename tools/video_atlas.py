"""Video atlas: frames tiled with their plan position + view direction, for building a
human-checked room map of a capture (ground-truth aid). Shows raw wall points only - never the
pipeline's rooms - so labelling is not biased by the prediction.
Usage: video_atlas.py <capture> <out.png> <n_frames>"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, cv2
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from brynz.capture import StrayCapture
from brynz.geometry import plan_coords, floor_ceiling, manhattan_yaw
from brynz.cloud import build_cloud, voxel_reduce

cap_dir, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
cap = StrayCapture(cap_dir)
cl = voxel_reduce(build_cloud(cap, step=max(4, cap.n // 600)), 0.03)
floor, _, _ = floor_ceiling(cl); yaw = manhattan_yaw(cl)
uvw, hw = plan_coords(cl['xyz'], yaw, floor)
wm = (abs(cl['normal'][:, 1]) < 0.3) & (hw > 0.3) & (hw < 1.8)
T = cap.poses(); cam, _ = plan_coords(T[:, :3, 3], yaw, floor)
fwd, _ = plan_coords(T[:, :3, 3] + T[:, :3, 2], yaw, floor); fwd -= cam
idx = np.linspace(5, cap.n - 8, n).astype(int)
imgs = dict(cap.rgb_frames(idx))
fig = plt.figure(figsize=(26, 13))
ax = fig.add_axes([0, 0, 0.42, 1])
ax.scatter(uvw[wm, 0], uvw[wm, 1], s=0.05, c='k')
ax.plot(cam[:, 0], cam[:, 1], color='0.7', lw=0.6)
for k, i in enumerate(idx):
    ax.annotate('', cam[i] + fwd[i] * 0.7, cam[i], arrowprops=dict(arrowstyle='->', color='r', lw=1))
    ax.text(*cam[i], str(k), color='r', fontsize=11, weight='bold')
ax.set_aspect('equal'); ax.axis('off')
cols = 6; rows = int(np.ceil(n / cols))
for k, i in enumerate(idx):
    r_, c_ = divmod(k, cols)
    a = fig.add_axes([0.43 + c_ * 0.095, 1 - (r_ + 1) / rows, 0.093, 1 / rows * 0.93])
    a.imshow(cv2.cvtColor(cv2.resize(cv2.rotate(imgs[i], cv2.ROTATE_90_CLOCKWISE), (240, 320)), cv2.COLOR_BGR2RGB)); a.axis('off')
    a.set_title(f'{k}: f{i}', fontsize=9, color='r', pad=1)
fig.savefig(out, dpi=55)
