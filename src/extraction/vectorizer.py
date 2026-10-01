import numpy as np
import cv2
import json

def points_to_grid(points_2d, resolution=0.02):
    """
    Converts 2D points to a binary occupancy grid at a specified resolution.
    Using 2cm resolution for a good balance of detail and noise suppression.
    """
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
    V5 Pipeline: STRICT Axis-aligned wall extraction.
    Throws away all diagonal noise and only keeps true walls.
    """
    # 1. Close gaps (10cm kernel)
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
        minLineLength=int(0.3 / resolution), # min 30cm wall
        maxLineGap=int(0.5 / resolution)     # bridge 50cm gaps
    )
    
    if lines is None:
        return []
    
    raw = []
    for line in lines:
        pts = line[0] if line.ndim == 2 else line
        raw.append(tuple(int(v) for v in pts))
    
    # 4. Filter and snap STRICTLY to axes (15 degree tolerance)
    # This THROWS AWAY all diagonal lines (noise, artifacts, furniture edges)
    snapped = _strict_snap_to_axes(raw, angle_tolerance=15)
    
    # 5. Merge collinear/parallel segments
    merged = _aggressive_merge(snapped, resolution)
    
    # 6. Final filter for length (> 50cm) to remove tiny leftover stubs
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


def _strict_snap_to_axes(segments, angle_tolerance=15):
    """
    STRICTLY snaps to 0 or 90 degrees.
    If a line is diagonal (outside tolerance), it is DISCARDED entirely.
    """
    snapped = []
    tol = np.radians(angle_tolerance)
    
    for (x1, y1, x2, y2) in segments:
        angle = np.arctan2(y2 - y1, x2 - x1)
        
        # Near horizontal
        if abs(angle) < tol or abs(abs(angle) - np.pi) < tol:
            mid_y = int(round((y1 + y2) / 2))
            snapped.append((min(x1, x2), mid_y, max(x1, x2), mid_y))
        # Near vertical
        elif abs(abs(angle) - np.pi/2) < tol:
            mid_x = int(round((x1 + x2) / 2))
            snapped.append((mid_x, min(y1, y2), mid_x, max(y1, y2)))
        # Else: Discard diagonals!
    
    return snapped


def _aggressive_merge(segments, resolution):
    horiz, vert = [], []
    for seg in segments:
        x1, y1, x2, y2 = seg
        if y1 == y2:
            horiz.append((min(x1, x2), y1, max(x1, x2), y2))
        elif x1 == x2:
            vert.append((x1, min(y1, y2), x2, max(y1, y2)))
    
    # 20cm parallel tolerance, 1.0m gap tolerance
    parallel_tol = int(0.20 / resolution)  
    gap_tol = int(1.0 / resolution)        
    
    for _ in range(5): 
        horiz = _merge_parallel_h(horiz, parallel_tol, gap_tol)
        vert = _merge_parallel_v(vert, parallel_tol, gap_tol)
    
    return horiz + vert


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
                # Check overlap or close gap
                if jx1 <= x2 + gap_tol and jx2 >= x1 - gap_tol:
                    x1 = min(x1, jx1)
                    x2 = max(x2, jx2)
                    y1 = int(round((y1 + jy1) / 2))
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
                    x1 = int(round((x1 + jx1) / 2))
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
