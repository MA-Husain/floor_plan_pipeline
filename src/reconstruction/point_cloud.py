import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def generate_point_cloud(depth_image, fx, fy, cx, cy, depth_scale=1000.0, depth_trunc=3.0):
    """
    Generates an Open3D point cloud from a 16-bit depth image.
    depth_scale is 1000.0 because the depth is in mm.
    depth_trunc is 3.0 meters (ignore points further than 3m to reduce noise).
    """
    # Convert numpy depth image to Open3D Image
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
    
    # Create Open3D camera intrinsics
    height, width = depth_image.shape
    intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    
    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_o3d, 
        intrinsics, 
        depth_scale=depth_scale,
        depth_trunc=depth_trunc
    )
    return pcd

def apply_pose(pcd, tx, ty, tz, qx, qy, qz, qw):
    """
    Applies the global pose (translation and quaternion) to the point cloud.
    """
    # Create a 4x4 transformation matrix
    transform = np.eye(4)
    
    # Open3D / standard convention: 
    # ARKit uses a specific coordinate system. We might need to adjust it later,
    # but for now we apply it directly.
    rotation = R.from_quat([qx, qy, qz, qw]).as_matrix()
    transform[:3, :3] = rotation
    transform[:3, 3] = [tx, ty, tz]
    
    # Transform the point cloud
    pcd.transform(transform)
    return pcd
