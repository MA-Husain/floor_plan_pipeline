"""
Drift Corrector using Loop Closure Detection.

When the camera revisits a place it has already been, we can measure how much 
the pose tracking has drifted. We then distribute that error backwards through 
the trajectory to snap everything into alignment.
"""
import numpy as np
from scipy.spatial import cKDTree

def detect_loop_closures(positions, min_travel_distance=3.0, proximity_threshold=0.3):
    """
    Detects loop closures: moments where the camera returns near a position 
    it visited much earlier in the trajectory.
    
    Args:
        positions: (N, 3) array of camera positions over time
        min_travel_distance: minimum path length between revisit to count as a loop (meters)
        proximity_threshold: how close camera must be to a past position (meters)
    
    Returns:
        List of (earlier_idx, later_idx, drift_vector) tuples
    """
    n = len(positions)
    if n < 50:
        return []
    
    # Build cumulative path length
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cum_dist = np.concatenate([[0], np.cumsum(diffs)])
    
    # Build KD-tree of positions for fast nearest-neighbor lookup
    tree = cKDTree(positions)
    
    closures = []
    # Check every 10th frame to find revisits
    for i in range(50, n, 10):
        # Find all past positions that are spatially close to current position
        nearby = tree.query_ball_point(positions[i], proximity_threshold)
        
        for j in nearby:
            # Must be much earlier in the trajectory and far apart in path distance
            if j < i - 50 and (cum_dist[i] - cum_dist[j]) > min_travel_distance:
                drift = positions[i] - positions[j]
                closures.append((j, i, drift))
                break  # One closure per query frame
    
    return closures


def correct_drift(positions, quaternions, closures):
    """
    Applies drift correction by distributing the error linearly between 
    each loop closure pair.
    
    For each closure (j, i, drift):
    - At frame j, the position was correct
    - At frame i, it has drifted by `drift` vector
    - We subtract a linearly increasing fraction of `drift` from frames j..i
    
    Returns corrected positions and quaternions.
    """
    corrected_pos = positions.copy()
    
    # Sort closures by the later index
    closures = sorted(closures, key=lambda c: c[1])
    
    for (j, i, drift) in closures:
        span = i - j
        if span <= 0:
            continue
        
        # Create linear interpolation weights from 0 (at j) to 1 (at i)
        weights = np.linspace(0, 1, span + 1).reshape(-1, 1)
        
        # Subtract increasing fraction of drift
        corrected_pos[j:i+1] -= weights * drift
    
    return corrected_pos, quaternions


def apply_drift_correction(odometry_df):
    """
    Full drift correction pipeline: detect loops, correct positions.
    Modifies the dataframe in-place and returns loop closure stats.
    """
    positions = odometry_df[['x', 'y', 'z']].values.copy()
    quaternions = odometry_df[['qx', 'qy', 'qz', 'qw']].values.copy()
    
    # Detect loop closures
    closures = detect_loop_closures(positions, 
                                     min_travel_distance=2.0, 
                                     proximity_threshold=0.5)
    
    if not closures:
        print("  No loop closures detected (short scan or no revisits)")
        return 0
    
    print(f"  Detected {len(closures)} loop closures")
    
    # Apply correction
    corrected_pos, corrected_quat = correct_drift(positions, quaternions, closures)
    
    # Write back
    odometry_df['x'] = corrected_pos[:, 0]
    odometry_df['y'] = corrected_pos[:, 1]
    odometry_df['z'] = corrected_pos[:, 2]
    
    # Report average drift magnitude
    drift_magnitudes = [np.linalg.norm(c[2]) for c in closures]
    avg_drift = np.mean(drift_magnitudes)
    max_drift = np.max(drift_magnitudes)
    print(f"  Average drift per closure: {avg_drift:.3f} m")
    print(f"  Max drift: {max_drift:.3f} m")
    
    return len(closures)
