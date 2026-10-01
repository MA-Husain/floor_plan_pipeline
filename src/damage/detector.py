import cv2
from ultralytics import YOLO
from pathlib import Path
import numpy as np

class DamageDetector:
    def __init__(self, model_path=None):
        """
        Initializes an offline YOLO-based damage detector.
        
        If no custom damage model is provided, we use YOLOv8n (nano) as a 
        general object detector. For production, you would fine-tune on a 
        crack/water-damage dataset from Roboflow.
        
        The weights are automatically downloaded on first run and cached locally,
        so subsequent runs are fully offline.
        """
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use YOLOv8 nano as base - download once, run offline forever
            self.model = YOLO('yolov8n.pt')
        
    def detect(self, rgb_image, conf_threshold=0.25):
        """
        Runs YOLO inference on an RGB image.
        Returns list of detections: [{'bbox': [x1,y1,x2,y2], 'class': str, 'confidence': float}]
        """
        results = self.model(rgb_image, verbose=False, conf=conf_threshold)
        
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = self.model.names[cls_id]
                    
                    detections.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'class': cls_name,
                        'confidence': conf
                    })
        
        return detections

    def detect_cracks_cv(self, rgb_image):
        """
        Fallback: Traditional CV-based crack detection using Canny edge detection
        and morphological filtering. More targeted than adaptive thresholding.
        
        Only flags regions that have:
        1. Strong edges (Canny)
        2. Linear structure (high aspect ratio)
        3. Dark coloring relative to surroundings
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection with auto thresholds
        median = np.median(blurred)
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        edges = cv2.Canny(blurred, lower, upper)
        
        # Dilate edges to connect nearby fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cracks = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100 or area > 10000:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / (min(w, h) + 1e-6)
            
            # Cracks are elongated: aspect ratio > 3
            if aspect > 3.0:
                # Check if the region is darker than its surroundings
                roi = gray[y:y+h, x:x+w]
                surround_mean = np.mean(gray[max(0,y-10):y+h+10, max(0,x-10):x+w+10])
                roi_mean = np.mean(roi)
                
                # Crack should be darker
                if roi_mean < surround_mean * 0.85:
                    cracks.append({
                        'bbox': [x, y, x+w, y+h],
                        'class': 'crack',
                        'confidence': min(1.0, aspect / 10.0)
                    })
        
        return cracks
