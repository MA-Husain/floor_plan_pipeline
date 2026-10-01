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
        
    def _load_csv(self, filename: str):
        filepath = self.data_dir / filename
        if filepath.exists():
            return pd.read_csv(filepath)
        return None

    def get_depth_map(self, timestamp_index: int):
        """
        Loads a 16-bit depth map given an index (usually corresponds to frames).
        This is a basic implementation assuming files are sorted.
        """
        depth_files = sorted(list(self.depth_dir.glob('*.png')))
        if timestamp_index < len(depth_files):
            # Use cv2.IMREAD_ANYDEPTH to read the 16-bit values correctly
            depth_image = cv2.imread(str(depth_files[timestamp_index]), cv2.IMREAD_ANYDEPTH)
            return depth_image
        return None
