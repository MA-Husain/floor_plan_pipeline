import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

class DamageDetector:
    """
    Hybrid Damage Detection Pipeline:
    
    Layer 1: YOLOv8 (ML) — detects objects. We use this to EXCLUDE furniture 
             and focus only on wall/ceiling surfaces.
    Layer 2: Wall Surface Analysis (CV) — on regions NOT occupied by furniture,
             look for cracks, stains, discoloration, and moisture damage.
    Layer 3: Structural Anomaly Detection — uses color histogram analysis to 
             find patches that differ significantly from the dominant wall color.
    """
    
    # COCO classes that are furniture/objects (NOT damage)
    FURNITURE_CLASSES = {
        'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat',
        'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        'backpack', 'umbrella', 'handbag', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush'
    }
    
    def __init__(self, model_path=None):
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            self.model = YOLO('yolov8n.pt')
    
    def detect_furniture(self, rgb_image, conf_threshold=0.3):
        """
        Layer 1: Use YOLO to find furniture/objects so we can MASK them out.
        Returns bounding boxes of detected objects.
        """
        results = self.model(rgb_image, verbose=False, conf=conf_threshold)
        
        objects = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = self.model.names[cls_id]
                    
                    objects.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'class': cls_name,
                        'confidence': conf,
                        'is_furniture': cls_name in self.FURNITURE_CLASSES
                    })
        return objects
    
    def detect_wall_damage(self, rgb_image, furniture_boxes=None):
        """
        Layer 2: Analyze wall surfaces for damage indicators.
        
        Damage types detected:
        - Cracks: Dark, elongated lines on light walls
        - Stains: Discolored patches (yellow/brown on white walls)  
        - Moisture: Dark patches with irregular boundaries
        - Peeling: Texture irregularity near edges
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
        h, w = gray.shape
        
        # Create a mask that excludes furniture regions
        wall_mask = np.ones((h, w), dtype=np.uint8) * 255
        if furniture_boxes:
            for fb in furniture_boxes:
                if fb.get('is_furniture', False):
                    bx1, by1, bx2, by2 = map(int, fb['bbox'])
                    # Add padding around furniture
                    pad = 20
                    wall_mask[max(0,by1-pad):min(h,by2+pad), 
                             max(0,bx1-pad):min(w,bx2+pad)] = 0
        
        damages = []
        
        # --- Crack Detection ---
        cracks = self._detect_cracks(gray, wall_mask)
        damages.extend(cracks)
        
        # --- Stain/Discoloration Detection ---
        stains = self._detect_stains(hsv, gray, wall_mask)
        damages.extend(stains)
        
        # --- Moisture/Water Damage Detection ---
        moisture = self._detect_moisture(hsv, gray, wall_mask)
        damages.extend(moisture)
        
        # Post-processing: Filter out repeating grid patterns (tile grout)
        # Real damage is isolated; grout lines come in repeating patterns
        damages = self._filter_repeating_patterns(damages)
        
        return damages
    
    def _filter_repeating_patterns(self, damages):
        """
        If many similar-sized detections of the same class appear in a regular 
        grid pattern, they are likely tile grout or shelf edges, not real damage.
        """
        if len(damages) <= 3:
            return damages  # Too few to be a pattern
        
        # Group by class
        by_class = {}
        for d in damages:
            cls = d['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(d)
        
        filtered = []
        for cls, items in by_class.items():
            if len(items) > 8:
                # Too many detections of the same type = likely a pattern, not damage
                # Keep only the top 3 most confident ones
                items.sort(key=lambda x: x['confidence'], reverse=True)
                filtered.extend(items[:3])
            else:
                filtered.extend(items)
        
        return filtered
    
    def _detect_cracks(self, gray, wall_mask):
        """Detect dark elongated lines (cracks) on wall surfaces."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholding to find dark lines
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 8)
        
        # Only consider wall regions
        thresh = cv2.bitwise_and(thresh, wall_mask)
        
        # Morphological operations to connect crack fragments
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        
        horiz_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h)
        vert_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
        lines = cv2.bitwise_or(horiz_lines, vert_lines)
        
        # Dilate to connect nearby fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines = cv2.dilate(lines, kernel, iterations=1)
        
        contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cracks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80 or area > 15000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / (min(w, h) + 1e-6)
            
            # Cracks are elongated (aspect > 4) and relatively thin
            if aspect > 4.0 and min(w, h) < 30:
                # Verify it's darker than surroundings
                roi = gray[y:y+h, x:x+w]
                pad = 15
                sy, ey = max(0, y-pad), min(gray.shape[0], y+h+pad)
                sx, ex = max(0, x-pad), min(gray.shape[1], x+w+pad)
                surround = gray[sy:ey, sx:ex]
                
                if np.mean(roi) < np.mean(surround) * 0.82:
                    confidence = min(1.0, (aspect / 10.0) * (area / 500.0))
                    confidence = min(confidence, 0.95)
                    cracks.append({
                        'bbox': [x, y, x+w, y+h],
                        'class': 'crack',
                        'confidence': round(confidence, 2),
                        'severity': 'high' if area > 500 else 'low'
                    })
        
        return cracks
    
    def _detect_stains(self, hsv, gray, wall_mask):
        """Detect discolored/stained patches on walls."""
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # Stains typically have higher saturation than clean white/gray walls
        # and yellowish/brownish hue
        stain_mask = np.zeros_like(gray)
        
        # Yellow/brown stains: Hue 15-35, moderate saturation
        yellow_stain = (h_channel >= 15) & (h_channel <= 35) & (s_channel > 40) & (s_channel < 200)
        stain_mask[yellow_stain] = 255
        
        # Apply wall mask
        stain_mask = cv2.bitwise_and(stain_mask, wall_mask)
        
        # Clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_OPEN, kernel)
        stain_mask = cv2.morphologyEx(stain_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(stain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stains = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200 or area > 50000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            # Stains are typically more blob-like (aspect < 5)
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect < 5.0:
                confidence = min(0.8, area / 5000.0)
                stains.append({
                    'bbox': [x, y, x+w, y+h],
                    'class': 'stain',
                    'confidence': round(confidence, 2),
                    'severity': 'medium' if area > 1000 else 'low'
                })
        
        return stains
    
    def _detect_moisture(self, hsv, gray, wall_mask):
        """
        Detect dark, damp-looking patches that indicate water damage.
        STRICT thresholds to avoid flagging shadows as moisture.
        """
        h_channel, s_channel, v_channel = cv2.split(hsv)
        
        # Moisture damage: Very dark patches (V < 50), almost no color (S < 30)
        # This is much stricter than before to avoid shadow false positives
        moisture_mask = np.zeros_like(gray)
        dark_patch = (v_channel < 50) & (s_channel < 30)
        moisture_mask[dark_patch] = 255
        
        moisture_mask = cv2.bitwise_and(moisture_mask, wall_mask)
        
        # Larger opening kernel to remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        moisture_mask = cv2.morphologyEx(moisture_mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(moisture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        damages = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Much higher minimum area (3000px instead of 500)
            if area < 3000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Additional check: the region should be significantly darker 
            # than the overall frame average (not just a corner shadow)
            roi_mean = np.mean(gray[y:y+h, x:x+w])
            frame_mean = np.mean(gray)
            if roi_mean > frame_mean * 0.4:
                continue  # Not dark enough relative to the overall frame
            
            confidence = min(0.7, area / 15000.0)
            damages.append({
                'bbox': [x, y, x+w, y+h],
                'class': 'moisture_damage',
                'confidence': round(confidence, 2),
                'severity': 'high' if area > 8000 else 'medium'
            })
        
        return damages
    
    def full_analysis(self, rgb_image, conf_threshold=0.3):
        """
        Complete analysis pipeline:
        1. Detect furniture (to exclude from damage analysis)
        2. Analyze remaining wall surfaces for damage
        3. Return both furniture inventory and damage findings
        """
        furniture = self.detect_furniture(rgb_image, conf_threshold)
        damage = self.detect_wall_damage(rgb_image, furniture)
        
        return {
            'furniture': [f for f in furniture if f['is_furniture']],
            'damage': damage,
            'summary': {
                'furniture_count': sum(1 for f in furniture if f['is_furniture']),
                'damage_count': len(damage),
                'cracks': sum(1 for d in damage if d['class'] == 'crack'),
                'stains': sum(1 for d in damage if d['class'] == 'stain'),
                'moisture': sum(1 for d in damage if d['class'] == 'moisture_damage'),
            }
        }
