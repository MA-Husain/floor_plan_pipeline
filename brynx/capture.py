"""Loader for Stray Scanner captures (App Store, LiDAR iPhones).

Layout: rgb.mp4, depth/NNNNNN.png (uint16 mm, 256x192), confidence/NNNNNN.png (0/1/2),
odometry.csv (timestamp, frame, x,y,z, qx,qy,qz,qw, fx,fy,cx,cy), camera_matrix.csv, imu.csv.

Conventions (verified empirically, see docs/diagnosis): poses are camera-to-world with the
camera in OpenCV convention (x right, y down, z forward); world is ARKit gravity-aligned, +Y up.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from scipy.spatial.transform import Rotation


@dataclass
class Frame:
    index: int
    depth: np.ndarray        # float32 metres, HxW
    confidence: np.ndarray   # uint8 0..2
    K: np.ndarray            # 3x3 intrinsics at depth resolution
    T_wc: np.ndarray         # 4x4 camera-to-world


class StrayCapture:
    def __init__(self, root):
        self.root = Path(root)
        self.odometry = pd.read_csv(self.root / 'odometry.csv', skipinitialspace=True)
        self.depth_files = sorted((self.root / 'depth').glob('*.png'))
        self.conf_files = sorted((self.root / 'confidence').glob('*.png'))
        cap = cv2.VideoCapture(str(self.root / 'rgb.mp4'))
        self.rgb_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        d0 = cv2.imread(str(self.depth_files[0]), cv2.IMREAD_ANYDEPTH)
        self.depth_size = (d0.shape[1], d0.shape[0])
        self.n = min(len(self.depth_files), len(self.odometry))

    @property
    def positions(self):
        return self.odometry[['x', 'y', 'z']].values[:self.n]

    def pose(self, i):
        r = self.odometry.iloc[i]
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([r.qx, r.qy, r.qz, r.qw]).as_matrix()
        T[:3, 3] = [r.x, r.y, r.z]
        return T

    def poses(self):
        q = self.odometry[['qx', 'qy', 'qz', 'qw']].values[:self.n]
        T = np.tile(np.eye(4), (self.n, 1, 1))
        T[:, :3, :3] = Rotation.from_quat(q).as_matrix()
        T[:, :3, 3] = self.positions
        return T

    def K_rgb(self, i):
        r = self.odometry.iloc[i]
        return np.array([[r.fx, 0, r.cx], [0, r.fy, r.cy], [0, 0, 1.0]])

    def K_depth(self, i):
        K = self.K_rgb(i).copy()
        K[0] *= self.depth_size[0] / self.rgb_size[0]
        K[1] *= self.depth_size[1] / self.rgb_size[1]
        return K

    def frame(self, i, T_wc=None):
        depth = cv2.imread(str(self.depth_files[i]), cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
        conf = cv2.imread(str(self.conf_files[i]), cv2.IMREAD_ANYDEPTH).astype(np.uint8)
        return Frame(i, depth, conf, self.K_depth(i), self.pose(i) if T_wc is None else T_wc)

    def rgb_frames(self, indices):
        """Yield (index, BGR image) for sorted frame indices, decoding sequentially."""
        cap = cv2.VideoCapture(str(self.root / 'rgb.mp4'))
        want = sorted(set(int(i) for i in indices))
        k, j = 0, 0
        while j < len(want):
            ok = cap.grab()
            if not ok:
                break
            if k == want[j]:
                _, img = cap.retrieve()
                yield k, img
                j += 1
            k += 1
        cap.release()
