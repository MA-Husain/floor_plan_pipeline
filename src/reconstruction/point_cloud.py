import numpy as np
from scipy.spatial.transform import Rotation as R

def generate_point_cloud_numpy(depth_image, fx, fy, cx, cy, 
                                confidence_map=None, min_confidence=1,
                                depth_scale=1000.0, depth_trunc=5.0):
    """
    Generates a 3D point cloud from a depth image using pure NumPy.
    
    Key improvements:
    - Uses confidence maps to filter unreliable LiDAR measurements
    - depth_trunc increased to 5m to capture entire rooms
    - Returns (N, 3) array of 3D points in camera-local coordinates
    
    ARKit depth convention: depth values are in millimeters (uint16).
    ARKit camera convention: +X right, +Y up, -Z forward (OpenGL-style).
    The depth map stores the Z distance along the camera's optical axis.
    """
    height, width = depth_image.shape
    
    # Create meshgrid of pixel coordinates
    v, u = np.indices((height, width))
    
    # Convert depth to meters
    z = depth_image.astype(np.float64) / depth_scale
    
    # Build validity mask
    valid = (z > 0.05) & (z < depth_trunc)  # min 5cm, max 5m
    
    # Apply confidence filter if available
    if confidence_map is not None:
        valid &= (confidence_map >= min_confidence)
    
    u = u[valid].astype(np.float64)
    v = v[valid].astype(np.float64)
    z = z[valid]
    
    if len(z) == 0:
        return np.zeros((0, 3))
    
    # Pinhole camera back-projection:
    # X = (u - cx) * Z / fx
    # Y = (v - cy) * Z / fy
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into (N, 3) array
    points = np.stack((x, y, z), axis=-1)
    return points

def apply_pose_numpy(points, tx, ty, tz, qx, qy, qz, qw):
    """
    Transforms points from camera-local coordinates to world coordinates.
    
    The pose (tx,ty,tz, qx,qy,qz,qw) describes the camera's position 
    and orientation in world space. To transform a point P_camera to 
    world space: P_world = R * P_camera + T
    """
    if len(points) == 0:
        return points
        
    rotation_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()
    translation = np.array([tx, ty, tz])
    
    # P_world = R @ P_camera + T
    transformed_points = points @ rotation_matrix.T + translation
    return transformed_points

def voxel_downsample(points, voxel_size=0.01):
    """
    Voxel grid downsampling: divides space into cubes and keeps one point per cube.
    This is much better than random sampling because it preserves spatial coverage.
    """
    if len(points) == 0:
        return points
    
    # Quantize points to voxel grid
    quantized = np.floor(points / voxel_size).astype(np.int32)
    
    # Use structured array for unique finding
    _, unique_indices = np.unique(
        quantized.view(dtype=[('x', np.int32), ('y', np.int32), ('z', np.int32)]).reshape(-1),
        return_index=True
    )
    
    return points[unique_indices]

def save_ply(filepath, points):
    """
    Saves an (N, 3) numpy array to a PLY file (binary for speed).
    """
    n = len(points)
    header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
end_header
"""
    with open(filepath, 'wb') as f:
        f.write(header.encode('ascii'))
        points.astype(np.float32).tofile(f)
