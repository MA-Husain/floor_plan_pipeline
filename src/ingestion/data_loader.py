import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

class CaptureDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.depth_dir = self.data_dir / 'depth'
        self.confidence_dir = self.data_dir / 'confidence'
        
        # Load the CSV files
        self.odometry_df = self._load_csv('odometry.csv')
        self.camera_matrix_df = self._load_csv('camera_matrix.csv')
        self.imu_df = self._load_csv('imu.csv')
        
        # Cache sorted file lists
        self._depth_files = sorted(list(self.depth_dir.glob('*.png'))) if self.depth_dir.exists() else []
        self._conf_files = sorted(list(self.confidence_dir.glob('*.png'))) if self.confidence_dir.exists() else []
        
        # Determine video resolution
        self._video_width = None
        self._video_height = None
        video_path = self.data_dir / 'rgb.mp4'
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                self._video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self._video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
        
    def _load_csv(self, filename: str):
        filepath = self.data_dir / filename
        if filepath.exists():
            # Use skipinitialspace to strip spaces from column names
            return pd.read_csv(filepath, skipinitialspace=True)
        return None

    @property
    def num_frames(self):
        return len(self._depth_files)

    def get_depth_map(self, frame_index: int):
        """Loads a 16-bit depth map by frame index."""
        if frame_index < len(self._depth_files):
            depth_image = cv2.imread(str(self._depth_files[frame_index]), cv2.IMREAD_ANYDEPTH)
            return depth_image
        return None

    def get_confidence_map(self, frame_index: int):
        """Loads a confidence map by frame index (values 0, 1, or 2)."""
        if frame_index < len(self._conf_files):
            conf_image = cv2.imread(str(self._conf_files[frame_index]), cv2.IMREAD_ANYDEPTH)
            return conf_image
        return None

    def get_scaled_intrinsics(self, frame_index: int):
        """
        Returns (fx, fy, cx, cy) scaled to match the depth map resolution.
        The odometry CSV has intrinsics for the full-res video.
        Depth maps are much smaller, so we scale proportionally.
        """
        row = self.odometry_df.iloc[frame_index]
        depth_map = self.get_depth_map(frame_index)
        if depth_map is None:
            return None
        
        dh, dw = depth_map.shape
        
        # Scale from video resolution to depth resolution
        scale_x = dw / self._video_width
        scale_y = dh / self._video_height
        
        fx = row['fx'] * scale_x
        fy = row['fy'] * scale_y
        cx = row['cx'] * scale_x
        cy = row['cy'] * scale_y
        
        return fx, fy, cx, cy

    def get_pose(self, frame_index: int):
        """Returns (tx, ty, tz, qx, qy, qz, qw) for a frame."""
        row = self.odometry_df.iloc[frame_index]
        return row['x'], row['y'], row['z'], row['qx'], row['qy'], row['qz'], row['qw']
