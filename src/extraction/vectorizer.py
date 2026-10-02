import numpy as np
import cv2
import json

def points_to_grid(points_2d, resolution=0.03):
    """
    Using 3cm resolution to balance detail and speed.
    """
    if len(points_2d) == 0:
        return np.zeros((10, 10), dtype=np.uint8), (0, 0), resolution
        
    min_xy = np.min(points_2d, axis=0)
    max_xy = np.max(points_2d, axis=0)
    
    pad = 1.0 # Extra padding for contours
    min_xy -= pad
    max_xy += pad
    
    width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
    height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1
    
    grid = np.zeros((height, width), dtype=np.uint8)
    
    px = ((points_2d[:, 0] - min_xy[0]) / resolution).astype(int)
    py = ((points_2d[:, 1] - min_xy[1]) / resolution).astype(int)
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    
    grid[py, px] = 255
    return grid, min_xy, resolution


def detect_walls_skeleton(grid, offset, resolution):
    """
    V7 Pipeline: Contour-Based Polygon Extraction
    Handles SLAM drift by fusing walls, and guarantees perfect corners.
    """
    # 1. HUGE morphological closing to fuse drift/double walls
    # A 50cm kernel will bridge the gaps between ghosted walls and furniture.
    kernel_size = int(0.50 / resolution) 
    if kernel_size % 2 == 0: kernel_size += 1
    
    # We use a CIRCLE to expand equally in all directions
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    fused = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=2)
    
    # Optional: Dilate and erode to smooth boundaries
    fused = cv2.dilate(fused, k, iterations=1)
    fused = cv2.erode(fused, k, iterations=1)
    
    # 2. Find Contours
    # RETR_EXTERNAL gets the outer boundary (good for outer walls)
    # RETR_LIST gets all boundaries (outer + inner holes)
    contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_segments = []
    
    for contour in contours:
        # 3. Polygon Approximation (Douglas-Peucker)
        # Epsilon dictates how strictly the polygon fits the contour.
        # 40cm epsilon aggressively smooths bumps and enforces straight lines.
        epsilon = (0.40 / resolution) 
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Squeeze to shape (N, 2)
        approx = approx.reshape(-1, 2)
        
        # If it's too small, skip it
        if len(approx) < 3:
            continue
            
        # Extract segments from the polygon edges
        for i in range(len(approx)):
            p1 = approx[i]
            p2 = approx[(i + 1) % len(approx)]
            raw_segments.append((p1[0], p1[1], p2[0], p2[1]))
            
    # 4. Filter and snap relative to dominant angle (to enforce 90 degree corners)
    dominant_angle = _find_dominant_angle(raw_segments)
    print(f"  Dominant Polygon Angle: {dominant_angle:.1f}°")
    
    # Snap segments to 90-degree increments
    snapped = _dynamic_snap_to_axes(raw_segments, dominant_angle, angle_tolerance=20)
    
    # 5. Filter out tiny artifacts (< 60cm = likely not real walls)
    min_len_px = 0.6 / resolution
    filtered = []
    for (x1, y1, x2, y2) in snapped:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length >= min_len_px:
            filtered.append((x1, y1, x2, y2))
    
    # 6. Merge nearby parallel walls (fuse double-walls from drift)
    merged = _merge_parallel_walls(filtered, resolution, 
                                    max_perp_dist=0.30/resolution,  # 30cm
                                    max_gap=0.8/resolution)          # 80cm gap
    
    print(f"  After merge: {len(merged)} walls (from {len(filtered)} pre-merge)")
            
    # Convert back to meters
    walls = []
    for (x1, y1, x2, y2) in merged:
        mx1 = x1 * resolution + offset[0]
        my1 = y1 * resolution + offset[1]
        mx2 = x2 * resolution + offset[0]
        my2 = y2 * resolution + offset[1]
        walls.append({'start': [float(mx1), float(my1)], 'end': [float(mx2), float(my2)]})
    
    return walls


def _merge_parallel_walls(segments, resolution, max_perp_dist=15, max_gap=30):
    """
    Merge nearby parallel wall segments.
    Two walls are merged if they are:
    1. Nearly parallel (within 5 degrees)
    2. Close perpendicularly (within max_perp_dist pixels)
    3. Overlapping or close along their main axis (within max_gap pixels)
    """
    if len(segments) <= 1:
        return segments
    
    used = [False] * len(segments)
    merged = []
    
    for i in range(len(segments)):
        if used[i]:
            continue
        
        x1, y1, x2, y2 = segments[i]
        angle_i = np.arctan2(y2-y1, x2-x1)
        
        # Collect all segments parallel and close to this one
        group = [(x1, y1, x2, y2)]
        used[i] = True
        
        for j in range(i+1, len(segments)):
            if used[j]:
                continue
            
            jx1, jy1, jx2, jy2 = segments[j]
            angle_j = np.arctan2(jy2-jy1, jx2-jx1)
            
            # Check if parallel (within 10 degrees)
            angle_diff = abs(angle_i - angle_j)
            angle_diff = min(angle_diff, np.pi - angle_diff)
            if angle_diff > np.radians(10):
                continue
            
            # Check perpendicular distance
            # Project midpoint of j onto the line through i
            mid_j = np.array([(jx1+jx2)/2, (jy1+jy2)/2])
            p1 = np.array([x1, y1])
            direction = np.array([x2-x1, y2-y1])
            dir_len = np.linalg.norm(direction)
            if dir_len < 1e-6:
                continue
            direction = direction / dir_len
            normal = np.array([-direction[1], direction[0]])
            
            perp_dist = abs(np.dot(mid_j - p1, normal))
            
            if perp_dist <= max_perp_dist:
                group.append(segments[j])
                used[j] = True
        
        # Merge the group: find the bounding extent along the main axis
        if len(group) == 1:
            merged.append(group[0])
        else:
            # Project all endpoints onto the main axis
            all_pts = []
            for (gx1, gy1, gx2, gy2) in group:
                all_pts.extend([(gx1, gy1), (gx2, gy2)])
            
            # Use first segment's direction
            direction = np.array([x2-x1, y2-y1])
            dir_len = np.linalg.norm(direction)
            if dir_len < 1e-6:
                merged.append(group[0])
                continue
            direction = direction / dir_len
            
            # Project all points onto the axis
            projections = [np.dot(np.array(p) - np.array([x1, y1]), direction) for p in all_pts]
            
            # Average perpendicular position
            normal = np.array([-direction[1], direction[0]])
            perp_positions = [np.dot(np.array(p) - np.array([x1, y1]), normal) for p in all_pts]
            avg_perp = np.mean(perp_positions)
            
            # Build the merged segment
            min_proj = min(projections)
            max_proj = max(projections)
            
            start = np.array([x1, y1]) + min_proj * direction + avg_perp * normal
            end = np.array([x1, y1]) + max_proj * direction + avg_perp * normal
            
            merged.append((int(start[0]), int(start[1]), int(end[0]), int(end[1])))
    
    return merged

def _find_dominant_angle(segments):
    angles = []
    weights = []
    for (x1, y1, x2, y2) in segments:
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angle_90 = angle_deg % 90
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angles.append(angle_90)
        weights.append(length)
        
    if not angles: return 0.0
    hist, bin_edges = np.histogram(angles, bins=90, range=(0, 90), weights=weights)
    return bin_edges[np.argmax(hist)]

def _dynamic_snap_to_axes(segments, dominant_angle_deg, angle_tolerance=15):
    """
    Forces polygon edges to be perfectly straight relative to dominant angle.
    """
    snapped = []
    tol = angle_tolerance
    
    for (x1, y1, x2, y2) in segments:
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        diff1 = (angle_deg - dominant_angle_deg) % 180
        if diff1 > 90: diff1 = 180 - diff1
            
        diff2 = (angle_deg - (dominant_angle_deg + 90)) % 180
        if diff2 > 90: diff2 = 180 - diff2
            
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if diff1 <= tol:
            rad = np.radians(dominant_angle_deg)
            dx = length/2 * np.cos(rad)
            dy = length/2 * np.sin(rad)
            snapped.append((int(mid_x - dx), int(mid_y - dy), int(mid_x + dx), int(mid_y + dy)))
            
        elif diff2 <= tol:
            rad = np.radians(dominant_angle_deg + 90)
            dx = length/2 * np.cos(rad)
            dy = length/2 * np.sin(rad)
            snapped.append((int(mid_x - dx), int(mid_y - dy), int(mid_x + dx), int(mid_y + dy)))
        
        # We don't discard diagonals from the polygon approximation, we just don't snap them.
        # But to enforce strict architecture, let's just keep the snapped ones.
            
    return snapped


def generate_json_output(walls, ceiling_height, height_confidence="", output_file="output.json"):
    payload = {
        "room_name": "Scanned Space",
        "dimensions": {
            "ceiling_height_meters": round(ceiling_height, 3),
            "height_confidence": height_confidence,
            "total_walls_detected": len(walls)
        },
        "walls": []
    }
    
    for i, w in enumerate(walls):
        dx = w['end'][0] - w['start'][0]
        dy = w['end'][1] - w['start'][1]
        length = np.sqrt(dx**2 + dy**2)
        
        payload["walls"].append({
            "id": f"wall_{i+1}",
            "start": [round(w['start'][0], 3), round(w['start'][1], 3)],
            "end": [round(w['end'][0], 3), round(w['end'][1], 3)],
            "length_meters": round(length, 3)
        })
        
    with open(output_file, 'w') as f:
        json.dump(payload, f, indent=4)
        
    return payload
