from skimage.measure import shannon_entropy
from skimage.feature import canny
from skimage.color import rgb2gray
import numpy as np

class PhysicalAnalyzer:
    def analyze_texture_entropy(self, image):
        """
        计算图像纹理的香农熵。
        熵越高 = 纹理/信息越复杂。
        """
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
            
        entropy_val = shannon_entropy(gray)
        return entropy_val

    def analyze_detail_density(self, image):
        """
        计算物理细节（边缘）的密度。
        使用 Canny 边缘检测器。
        """
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
            
        edges = canny(gray, sigma=1.0)
        density = np.sum(edges) / edges.size
        return density

    def analyze(self, image):
        return {
            "texture_entropy": self.analyze_texture_entropy(image),
            "detail_density": self.analyze_detail_density(image)
        }
