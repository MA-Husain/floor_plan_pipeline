import numpy as np
import cv2
import json

def points_to_grid(points_2d, resolution=0.02):
    if len(points_2d) == 0:
        return np.zeros((10, 10), dtype=np.uint8), (0, 0), resolution
        
    min_xy = np.min(points_2d, axis=0)
    max_xy = np.max(points_2d, axis=0)
    
    pad = 0.5
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
    V6 Pipeline: DYNAMIC Axis-aligned wall extraction.
    Finds the dominant orientation of the room, and snaps walls relative to that angle.
    """
    # 1. Close gaps
    kernel_size = int(0.10 / resolution)
    if kernel_size % 2 == 0: kernel_size += 1
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k, iterations=3)
    
    # 2. Skeletonize
    skeleton = _skeleton(closed)
    
    # 3. Hough lines
    lines = cv2.HoughLinesP(
        skeleton,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=int(0.3 / resolution),
        maxLineGap=int(0.5 / resolution)
    )
    
    if lines is None:
        return []
    
    raw = []
    for line in lines:
        pts = line[0] if line.ndim == 2 else line
        raw.append(tuple(int(v) for v in pts))
    
    # 4. Find Dominant Angle
    # We wrap angles to [0, 90) because walls are typically orthogonal
    angles = []
    weights = []
    for (x1, y1, x2, y2) in raw:
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Wrap to [0, 90)
        angle_90 = angle_deg % 90
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angles.append(angle_90)
        weights.append(length)
        
    if not angles:
        dominant_angle = 0.0
    else:
        # Use a weighted histogram to find the peak angle
        hist, bin_edges = np.histogram(angles, bins=90, range=(0, 90), weights=weights)
        dominant_angle = bin_edges[np.argmax(hist)]
    
    print(f"  Dominant Room Angle: {dominant_angle:.1f}°")
    
    # 5. Filter and snap relative to dominant angle
    snapped = _dynamic_snap_to_axes(raw, dominant_angle, angle_tolerance=15)
    
    # 6. Merge collinear/parallel segments (we rotate them to 0/90, merge, then rotate back)
    merged = _aggressive_merge_dynamic(snapped, dominant_angle, resolution)
    
    # 7. Final filter for length (> 50cm)
    min_len_px = 0.5 / resolution
    filtered = []
    for (x1, y1, x2, y2) in merged:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length >= min_len_px:
            filtered.append((x1, y1, x2, y2))
    
    # Convert to meters
    walls = []
    for (x1, y1, x2, y2) in filtered:
        mx1 = x1 * resolution + offset[0]
        my1 = y1 * resolution + offset[1]
        mx2 = x2 * resolution + offset[0]
        my2 = y2 * resolution + offset[1]
        walls.append({'start': [float(mx1), float(my1)], 'end': [float(mx2), float(my2)]})
    
    return walls


def _skeleton(binary_img):
    try:
        return cv2.ximgproc.thinning(binary_img)
    except AttributeError:
        pass
    
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


def _dynamic_snap_to_axes(segments, dominant_angle_deg, angle_tolerance=15):
    """
    Snaps lines to either dominant_angle or dominant_angle + 90.
    Discard lines that don't fit.
    """
    snapped = []
    tol = angle_tolerance
    
    for (x1, y1, x2, y2) in segments:
        angle_deg = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        
        # Calculate angular distance to dominant angle (modulo 180)
        diff1 = (angle_deg - dominant_angle_deg) % 180
        if diff1 > 90: diff1 = 180 - diff1
            
        # Calculate angular distance to orthogonal angle
        diff2 = (angle_deg - (dominant_angle_deg + 90)) % 180
        if diff2 > 90: diff2 = 180 - diff2
            
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        
        if diff1 <= tol:
            # Snap to dominant angle
            rad = np.radians(dominant_angle_deg)
            dx = length/2 * np.cos(rad)
            dy = length/2 * np.sin(rad)
            snapped.append((int(mid_x - dx), int(mid_y - dy), int(mid_x + dx), int(mid_y + dy)))
            
        elif diff2 <= tol:
            # Snap to orthogonal angle
            rad = np.radians(dominant_angle_deg + 90)
            dx = length/2 * np.cos(rad)
            dy = length/2 * np.sin(rad)
            snapped.append((int(mid_x - dx), int(mid_y - dy), int(mid_x + dx), int(mid_y + dy)))
            
    return snapped


def _rotate_points(segments, angle_deg, cx=0, cy=0):
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    rotated = []
    for (x1, y1, x2, y2) in segments:
        nx1 = cos_a * (x1 - cx) - sin_a * (y1 - cy) + cx
        ny1 = sin_a * (x1 - cx) + cos_a * (y1 - cy) + cy
        nx2 = cos_a * (x2 - cx) - sin_a * (y2 - cy) + cx
        ny2 = sin_a * (x2 - cx) + cos_a * (y2 - cy) + cy
        rotated.append((nx1, ny1, nx2, ny2))
    return rotated

def _aggressive_merge_dynamic(segments, dominant_angle, resolution):
    """
    To reuse the H/V merging logic, we rotate all segments by -dominant_angle,
    do the axis-aligned merge, and then rotate back!
    """
    if not segments: return []
    
    # 1. Rotate to 0/90
    rot_segs = _rotate_points(segments, -dominant_angle)
    
    horiz, vert = [], []
    for seg in rot_segs:
        x1, y1, x2, y2 = seg
        # Decide if horizontal or vertical based on width/height
        if abs(x2 - x1) > abs(y2 - y1):
            y_avg = (y1 + y2) / 2
            horiz.append((min(x1, x2), y_avg, max(x1, x2), y_avg))
        else:
            x_avg = (x1 + x2) / 2
            vert.append((x_avg, min(y1, y2), x_avg, max(y1, y2)))
            
    parallel_tol = 0.20 / resolution  
    gap_tol = 1.0 / resolution        
    
    for _ in range(5): 
        horiz = _merge_parallel_h(horiz, parallel_tol, gap_tol)
        vert = _merge_parallel_v(vert, parallel_tol, gap_tol)
        
    merged_rot = horiz + vert
    
    # 2. Rotate back
    final_merged = _rotate_points(merged_rot, dominant_angle)
    
    # Convert floats back to ints
    return [(int(x1), int(y1), int(x2), int(y2)) for (x1, y1, x2, y2) in final_merged]


def _merge_parallel_h(segments, parallel_tol, gap_tol):
    if len(segments) <= 1:
        return segments
    
    segments.sort(key=lambda s: (s[1], s[0]))
    merged = []
    used = [False] * len(segments)
    
    for i in range(len(segments)):
        if used[i]:
            continue
        x1, y1, x2, y2 = segments[i]
        
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            jx1, jy1, jx2, jy2 = segments[j]
            
            if abs(jy1 - y1) <= parallel_tol:
                if jx1 <= x2 + gap_tol and jx2 >= x1 - gap_tol:
                    x1 = min(x1, jx1)
                    x2 = max(x2, jx2)
                    y1 = (y1 + jy1) / 2
                    y2 = y1
                    used[j] = True
        merged.append((x1, y1, x2, y2))
    return merged


def _merge_parallel_v(segments, parallel_tol, gap_tol):
    if len(segments) <= 1:
        return segments
    
    segments.sort(key=lambda s: (s[0], s[1]))
    merged = []
    used = [False] * len(segments)
    
    for i in range(len(segments)):
        if used[i]:
            continue
        x1, y1, x2, y2 = segments[i]
        
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            jx1, jy1, jx2, jy2 = segments[j]
            
            if abs(jx1 - x1) <= parallel_tol:
                if jy1 <= y2 + gap_tol and jy2 >= y1 - gap_tol:
                    y1 = min(y1, jy1)
                    y2 = max(y2, jy2)
                    x1 = (x1 + jx1) / 2
                    x2 = x1
                    used[j] = True
        merged.append((x1, y1, x2, y2))
    return merged


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
