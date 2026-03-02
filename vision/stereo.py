import cv2
import numpy as np

class StereoSystem:
    def __init__(self, baseline=0.1, focal_length=1000.0, min_disp=0, num_disp=16*5):
        """
        初始化立体视觉系统。
        :param baseline: 摄像头之间的距离（米）。
        :param focal_length: 焦距（像素）。
        :param min_disp: 最小视差。
        :param num_disp: 搜索的视差数量。
        """
        self.baseline = baseline
        self.focal_length = focal_length
        self.min_disp = min_disp
        self.num_disp = num_disp
        
        # 初始化立体匹配器 (SGBM 通常比 BM 效果更好)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

    def compute_depth_map(self, img_left, img_right):
        """
        从校正后的立体图像对计算深度图。
        返回单位为米的深度图。
        """
        # 转换为灰度图
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # 计算视差
        disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        
        # 避免除以零
        disparity[disparity <= 0] = 0.1
        
        # 深度 = (f * B) / 视差
        depth_map = (self.focal_length * self.baseline) / disparity
        
        return depth_map

    def get_3d_point(self, u, v, depth_map):
        """
        获取特定像素 (u, v) 的 3D 坐标 (X, Y, Z)。
        """
        if u >= depth_map.shape[1] or v >= depth_map.shape[0]:
            return None
            
        z = depth_map[int(v), int(u)]
        
        # 假设主点位于图像中心（为简化起见）
        cx = depth_map.shape[1] / 2
        cy = depth_map.shape[0] / 2
        
        x = (u - cx) * z / self.focal_length
        y = (v - cy) * z / self.focal_length
        
        return (x, y, z)
