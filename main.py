import cv2
import numpy as np
import json
import time
from vision.detector import ObjectDetector
from vision.stereo import StereoSystem
from analysis.semantic import SemanticAnalyzer
from analysis.physical import PhysicalAnalyzer
from analysis.attention import AttentionAnalyzer
import config

class VisionSystem:
    def __init__(self):
        print("正在初始化视觉系统...")
        self.detector = ObjectDetector(config.YOLO_MODEL_PATH, config.CONFIDENCE_THRESHOLD)
        self.stereo = StereoSystem(config.BASELINE, config.FOCAL_LENGTH)
        self.semantic = SemanticAnalyzer(config.CLIP_MODEL_NAME)
        self.physical = PhysicalAnalyzer()
        self.attention = AttentionAnalyzer(config.SALIENCY_METHOD)
        print("系统初始化完成。")

    def process_frame(self, frame_left, frame_right=None):
        """
        处理单帧（或立体图像对）。
        如果 frame_right 为 None，则禁用/模拟立体视觉。
        """
        start_time = time.time()
        
        # 1. 基础检测
        detections = self.detector.detect(frame_left)
        
        # 2. 立体视觉 (如果可用)
        depth_map = None
        if frame_right is not None:
            depth_map = self.stereo.compute_depth_map(frame_left, frame_right)
            # 增加 3D 坐标信息
            for det in detections:
                cx, cy = det["center"]
                point_3d = self.stereo.get_3d_point(cx, cy, depth_map)
                if point_3d:
                    det["position_3d"] = point_3d
        
        # 3. 感知分析
        # 3.1 语义偏移 (去标签化)
        semantic_scores = self.semantic.analyze_semantic_shift(frame_left, config.TARGET_LABELS)
        is_delabeled = self.semantic.check_delabeling(semantic_scores)
        
        # 3.2 物理细节 (纹理熵)
        physical_metrics = self.physical.analyze(frame_left)
        
        # 3.3 视觉注意力 (显著性)
        saliency_map = self.attention.compute_saliency_map(frame_left)
        visual_center = self.attention.compute_visual_center(saliency_map)
        focus_shift = self.attention.analyze_focus_shift(visual_center)
        
        # 4. 构建“完美信息”接口
        perfect_info = {
            "timestamp": time.time(),
            "processing_time": time.time() - start_time,
            "positioning": {
                "detected_objects": detections,
                "camera_baseline": config.BASELINE,
                "focal_length": config.FOCAL_LENGTH
            },
            "perceptual_analysis": {
                "semantic_shift": {
                    "scores": semantic_scores,
                    "is_delabeled": is_delabeled,
                    "interpretation": "纯粹观察" if is_delabeled else "功能性标签化"
                },
                "physical_metrics": physical_metrics,
                "visual_attention": {
                    "visual_center": visual_center,
                    "focus_shift_distance": focus_shift["distance_from_center"],
                    "is_aesthetic_edge_focus": focus_shift["is_edge_focus"]
                }
            }
        }
        
        return perfect_info, saliency_map

def main():
    system = VisionSystem()
    
    # 模拟输入 (替换为 cv2.VideoCapture(0) 以连接真实摄像头)
    # 创建一个用于测试的虚拟图像
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(dummy_img, (100, 100), (300, 300), (255, 255, 255), -1)  # 白色矩形
    cv2.circle(dummy_img, (600, 400), 50, (0, 0, 255), -1)  # 红色圆形
    
    # 创建一个略微偏移的右图用于立体测试
    dummy_img_right = np.roll(dummy_img, -10, axis=1)
    
    print("正在处理模拟帧...")
    info, saliency = system.process_frame(dummy_img, dummy_img_right)
    
    print(json.dumps(info, indent=2, default=str, ensure_ascii=False))
    
    # 真实环境循环示例:
    # cap = cv2.VideoCapture(0)
    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    #     info, _ = system.process_frame(frame, frame) # 如果是单目，这里模拟双目输入
    #     print(info)

if __name__ == "__main__":
    main()
