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
    # A 30cm kernel will bridge the gaps between ghosted walls.
    kernel_size = int(0.30 / resolution) 
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
    contours, _ = cv2.findContours(fused, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    raw_segments = []
    
    for contour in contours:
        # 3. Polygon Approximation (Douglas-Peucker)
        # Epsilon dictates how strictly the polygon fits the contour.
        # 25cm epsilon smooths out small bumps and enforces straight lines.
        epsilon = (0.25 / resolution) 
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
    
    # 5. Filter out tiny artifacts (< 40cm)
    min_len_px = 0.4 / resolution
    filtered = []
    for (x1, y1, x2, y2) in snapped:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        if length >= min_len_px:
            filtered.append((x1, y1, x2, y2))
            
    # Convert back to meters
    walls = []
    for (x1, y1, x2, y2) in filtered:
        mx1 = x1 * resolution + offset[0]
        my1 = y1 * resolution + offset[1]
        mx2 = x2 * resolution + offset[0]
        my2 = y2 * resolution + offset[1]
        walls.append({'start': [float(mx1), float(my1)], 'end': [float(mx2), float(my2)]})
    
    return walls

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
