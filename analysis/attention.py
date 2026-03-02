import cv2
import numpy as np

class AttentionAnalyzer:
    def __init__(self, method="spectral_residual"):
        """
        初始化显著性检测器。
        """
        if method == "spectral_residual":
            self.saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        elif method == "fine_grained":
            self.saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        else:
            raise ValueError(f"未知的显著性方法: {method}")

    def compute_saliency_map(self, image):
        """
        计算显著性图。
        返回归一化后的图 (0-1)。
        """
        success, saliency_map = self.saliency.computeSaliency(image)
        if success:
            return (saliency_map * 255).astype("uint8")
        return None

    def compute_visual_center(self, saliency_map):
        """
        计算显著性图的加权质心。
        返回归一化坐标 (cx, cy) (0-1)。
        """
        moments = cv2.moments(saliency_map)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = saliency_map.shape[1] // 2, saliency_map.shape[0] // 2
            
        norm_cx = cx / saliency_map.shape[1]
        norm_cy = cy / saliency_map.shape[0]
        
        return (norm_cx, norm_cy)

    def analyze_focus_shift(self, visual_center):
        """
        分析焦点是否从功能中心 (0.5, 0.5) 转移到了美学边缘。
        返回距离中心的距离。
        """
        center = np.array([0.5, 0.5])
        visual_center_arr = np.array(visual_center)
        distance = np.linalg.norm(visual_center_arr - center)
        
        # 启发式规则：距离 > 0.3 意味着焦点显著偏移到边缘
        is_edge_focus = distance > 0.3
        
        return {
            "distance_from_center": distance,
            "is_edge_focus": is_edge_focus
        }
