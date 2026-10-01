import numpy as np
import cv2
import json

def points_to_grid(points_2d, resolution=0.01):
    """
    Converts 2D points to a binary occupancy grid at 1cm resolution.
    Higher resolution = thinner, more accurate walls.
    """
    if len(points_2d) == 0:
        return np.zeros((10, 10), dtype=np.uint8), (0, 0), resolution
        
    min_xy = np.min(points_2d, axis=0)
    max_xy = np.max(points_2d, axis=0)
    
    # Add padding
    pad = 0.2  # 20cm padding
    min_xy -= pad
    max_xy += pad
    
    width = int(np.ceil((max_xy[0] - min_xy[0]) / resolution)) + 1
    height = int(np.ceil((max_xy[1] - min_xy[1]) / resolution)) + 1
    
    grid = np.zeros((height, width), dtype=np.uint8)
    
    px = ((points_2d[:, 0] - min_xy[0]) / resolution).astype(int)
    py = ((points_2d[:, 1] - min_xy[1]) / resolution).astype(int)
    
    # Clip to bounds
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    
    grid[py, px] = 255
    
    return grid, min_xy, resolution


def detect_walls_skeleton(grid, offset, resolution):
    """
    Uses morphological skeletonization + Hough lines to find wall centerlines.
    
    Algorithm:
    1. Close gaps in the wall points (morphological closing)
    2. Skeletonize to reduce thick blobs to 1-pixel-wide lines
    3. Run Hough Transform on the skeleton for clean line segments
    4. Merge collinear, nearby segments into unified walls
    """
    # Step 1: Morphological closing to connect nearby wall points
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k_close, iterations=3)
    
    # Step 2: Skeletonize using Zhang-Suen thinning
    skeleton = cv2.ximgproc.thinning(closed) if hasattr(cv2, 'ximgproc') else _fallback_skeleton(closed)
    
    # Step 3: Hough Transform on the thin skeleton
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=int(0.3 / resolution),  # min 30cm wall
        maxLineGap=int(0.15 / resolution)       # bridge 15cm gaps
    )
    
    if lines is None:
        return []
    
    # Extract raw segments
    raw_segments = []
    for line in lines:
        pts = line[0] if line.ndim == 2 else line
        x1, y1, x2, y2 = pts
        raw_segments.append((x1, y1, x2, y2))
    
    # Step 4: Snap to dominant orientations (0° and 90°)
    # Most rooms have axis-aligned walls
    snapped = _snap_to_axes(raw_segments, angle_tolerance=15)
    
    # Step 5: Merge collinear, overlapping segments
    merged = _merge_collinear(snapped, resolution)
    
    # Convert pixels back to meters
    walls = []
    for (x1, y1, x2, y2) in merged:
        mx1 = x1 * resolution + offset[0]
        my1 = y1 * resolution + offset[1]
        mx2 = x2 * resolution + offset[0]
        my2 = y2 * resolution + offset[1]
        walls.append({'start': [float(mx1), float(my1)], 'end': [float(mx2), float(my2)]})
    
    return walls


def _fallback_skeleton(binary_img):
    """Fallback skeletonization if cv2.ximgproc is not available."""
    skeleton = np.zeros_like(binary_img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img = binary_img.copy()
    while True:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        diff = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skeleton


def _snap_to_axes(segments, angle_tolerance=15):
    """
    Snaps near-horizontal and near-vertical lines to exact 0° or 90°.
    Lines that are diagonal (>tolerance from any axis) are kept as-is.
    """
    snapped = []
    tol_rad = np.radians(angle_tolerance)
    
    for (x1, y1, x2, y2) in segments:
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        # Near horizontal (0° or 180°)
        if abs(angle) < tol_rad or abs(abs(angle) - np.pi) < tol_rad:
            mid_y = (y1 + y2) / 2
            snapped.append((x1, int(mid_y), x2, int(mid_y)))
        # Near vertical (90° or -90°)
        elif abs(abs(angle) - np.pi/2) < tol_rad:
            mid_x = (x1 + x2) / 2
            snapped.append((int(mid_x), y1, int(mid_x), y2))
        else:
            snapped.append((x1, y1, x2, y2))
    
    return snapped


def _merge_collinear(segments, resolution, dist_threshold=0.1):
    """
    Merges segments that are collinear and close together into single longer walls.
    """
    if not segments:
        return segments
    
    threshold_px = dist_threshold / resolution
    
    # Separate horizontal, vertical, and diagonal
    horiz, vert, diag = [], [], []
    for seg in segments:
        x1, y1, x2, y2 = seg
        if y1 == y2:
            horiz.append(seg)
        elif x1 == x2:
            vert.append(seg)
        else:
            diag.append(seg)
    
    merged = []
    
    # Merge horizontal segments at similar Y
    horiz.sort(key=lambda s: (s[1], min(s[0], s[2])))
    horiz_merged = _merge_1d(horiz, axis='h', threshold=threshold_px)
    merged.extend(horiz_merged)
    
    # Merge vertical segments at similar X
    vert.sort(key=lambda s: (s[0], min(s[1], s[3])))
    vert_merged = _merge_1d(vert, axis='v', threshold=threshold_px)
    merged.extend(vert_merged)
    
    merged.extend(diag)
    return merged


def _merge_1d(segments, axis, threshold):
    """Merges collinear segments along one axis."""
    if not segments:
        return []
    
    merged = []
    current = list(segments[0])
    
    for seg in segments[1:]:
        x1, y1, x2, y2 = seg
        cx1, cy1, cx2, cy2 = current
        
        if axis == 'h':
            # Same row?
            if abs(y1 - cy1) <= threshold:
                # Overlapping or close on X?
                cmin, cmax = min(cx1, cx2), max(cx1, cx2)
                smin, smax = min(x1, x2), max(x1, x2)
                if smin <= cmax + threshold:
                    # Extend
                    current = [min(cmin, smin), cy1, max(cmax, smax), cy1]
                    continue
        else:  # vertical
            if abs(x1 - cx1) <= threshold:
                cmin, cmax = min(cy1, cy2), max(cy1, cy2)
                smin, smax = min(y1, y2), max(y1, y2)
                if smin <= cmax + threshold:
                    current = [cx1, min(cmin, smin), cx1, max(cmax, smax)]
                    continue
        
        merged.append(tuple(current))
        current = list(seg)
    
    merged.append(tuple(current))
    return merged


def calculate_dimensions(global_points, up_axis=1):
    """Calculates ceiling height using robust percentile estimation."""
    y = global_points[:, up_axis]
    floor = np.percentile(y, 2)
    ceiling = np.percentile(y, 98)
    return float(ceiling - floor)


def generate_json_output(walls, ceiling_height, output_file="output.json"):
    """Formats the detected walls and dimensions into JSON."""
    payload = {
        "room_name": "Scanned Room",
        "dimensions": {
            "ceiling_height_meters": round(ceiling_height, 3),
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
