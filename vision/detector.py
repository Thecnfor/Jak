import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        初始化 YOLO 检测器。
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, image):
        """
        检测图像中的物体。
        返回检测列表: [{"box": [x1, y1, x2, y2], "class": "person", "conf": 0.95}, ...]
        """
        results = self.model(image, verbose=False, conf=self.conf_threshold)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = self.model.names[cls_id]
            
            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "class": cls_name,
                "conf": conf,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2]
            })
            
        return detections
