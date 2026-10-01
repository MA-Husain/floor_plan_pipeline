import numpy as np

def ransac_fit_plane(points, n_iterations=200, distance_threshold=0.03):
    """
    RANSAC plane fitting: finds the dominant horizontal plane in a set of 3D points.
    
    Returns:
        normal: (3,) unit normal vector of the best plane
        d: distance from origin (plane eq: normal · p + d = 0)
        inlier_mask: boolean mask of points that belong to this plane
    """
    best_inliers = 0
    best_normal = None
    best_d = None
    best_mask = None
    
    n = len(points)
    if n < 3:
        return np.array([0, 1, 0]), 0, np.zeros(n, dtype=bool)
    
    for _ in range(n_iterations):
        # Pick 3 random points
        idx = np.random.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]
        
        # Compute plane normal via cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        
        # Plane equation: normal · (p - p1) = 0  =>  normal · p = normal · p1
        d = -np.dot(normal, p1)
        
        # Count inliers: points within distance_threshold of the plane
        distances = np.abs(points @ normal + d)
        inlier_mask = distances < distance_threshold
        n_inliers = np.sum(inlier_mask)
        
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_d = d
            best_mask = inlier_mask
    
    return best_normal, best_d, best_mask


def estimate_ceiling_height_ransac(points, up_axis=1):
    """
    Uses RANSAC to fit horizontal planes to floor and ceiling.
    
    Strategy:
    1. Take bottom 15% of points (by height) → fit floor plane
    2. Take top 15% of points (by height) → fit ceiling plane
    3. Compute vertical distance between the two planes
    
    This is much more robust than simple percentiles because it ignores 
    outliers and furniture.
    """
    y = points[:, up_axis]
    
    # --- Floor plane ---
    floor_cutoff = np.percentile(y, 15)
    floor_candidates = points[y <= floor_cutoff]
    
    if len(floor_candidates) > 10:
        floor_normal, floor_d, _ = ransac_fit_plane(floor_candidates)
        # Floor height = -d / normal[up_axis] (if normal is roughly vertical)
        if abs(floor_normal[up_axis]) > 0.7:  # Check it's roughly horizontal
            floor_height = -floor_d / floor_normal[up_axis]
        else:
            floor_height = np.median(floor_candidates[:, up_axis])
    else:
        floor_height = np.percentile(y, 2)
    
    # --- Ceiling plane ---
    ceil_cutoff = np.percentile(y, 85)
    ceil_candidates = points[y >= ceil_cutoff]
    
    if len(ceil_candidates) > 10:
        ceil_normal, ceil_d, _ = ransac_fit_plane(ceil_candidates)
        if abs(ceil_normal[up_axis]) > 0.7:
            ceiling_height = -ceil_d / ceil_normal[up_axis]
        else:
            ceiling_height = np.median(ceil_candidates[:, up_axis])
    else:
        ceiling_height = np.percentile(y, 98)
    
    room_height = abs(ceiling_height - floor_height)
    
    # Sanity check: typical room height is 2.3-3.5m
    # If our estimate is outside this range, the scan likely didn't capture 
    # the full floor-to-ceiling, so we note that
    confidence = "high"
    if room_height < 2.0:
        confidence = "low (scan may not reach ceiling)"
    elif room_height > 4.0:
        confidence = "low (possible outliers)"
    
    return floor_height, ceiling_height, room_height, confidence
