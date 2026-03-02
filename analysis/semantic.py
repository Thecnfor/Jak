from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

class SemanticAnalyzer:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        初始化用于语义分析的 CLIP 模型。
        """
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        except Exception as e:
            print(f"加载 CLIP 模型时出错: {e}")
            self.model = None
            self.processor = None

    def analyze_semantic_shift(self, image, generic_labels):
        """
        计算图像与通用标签之间的余弦相似度。
        :param image: PIL 图像或 numpy 数组 (RGB)。
        :param generic_labels: 字符串列表 (例如 ["table", "chair"])。
        :return: 字典 {标签: 分数}。
        """
        if self.model is None:
            return {label: 0.0 for label in generic_labels}

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        inputs = self.processor(text=generic_labels, images=image, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)  # 也可以直接使用原始余弦相似度
        
        # 或者直接使用 logits 作为原始相似度分数（已缩放）
        # CLIP logits 是缩放后的余弦相似度 (logit_scale * cosine_similarity)
        # 为了获得纯粹的余弦相似度：
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        cosine_sim = torch.matmul(image_embeds, text_embeds.t())
        
        results = {}
        for i, label in enumerate(generic_labels):
            results[label] = cosine_sim[0][i].item()
            
        return results

    def check_delabeling(self, semantic_scores, threshold=0.2):
        """
        检查图像是否“去标签化”（即与所有通用标签的相似度都很低）。
        如果最大相似度 < 阈值，则返回 True。
        """
        max_score = max(semantic_scores.values()) if semantic_scores else 0
        return max_score < threshold
