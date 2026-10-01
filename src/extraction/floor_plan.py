import numpy as np
import cv2

def estimate_floor_ceiling(points, up_axis=1, percentile_low=2, percentile_high=98):
    """
    Uses robust percentile estimation to find floor and ceiling heights.
    Returns (floor_height, ceiling_height_value, room_height).
    """
    y = points[:, up_axis]
    floor = np.percentile(y, percentile_low)
    ceiling = np.percentile(y, percentile_high)
    return floor, ceiling, ceiling - floor

def slice_point_cloud(points, up_axis=1, floor_height=None, slice_height_range=(0.8, 1.2)):
    """
    Takes a horizontal slice of the point cloud at wall-height.
    
    slice_height_range: (min_m, max_m) above the detected floor.
    This ensures we capture wall surfaces but not floor clutter or ceiling fixtures.
    """
    y = points[:, up_axis]
    
    if floor_height is None:
        floor_height = np.percentile(y, 2)
    
    lo = floor_height + slice_height_range[0]
    hi = floor_height + slice_height_range[1]
    
    mask = (y > lo) & (y < hi)
    return points[mask]

def project_to_2d(points, up_axis=1):
    """Projects 3D points to a 2D floor plan by dropping the vertical axis."""
    axes = [i for i in range(3) if i != up_axis]
    return points[:, axes]
