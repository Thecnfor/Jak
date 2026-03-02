# Binocular Vision & Perceptual Shift Analysis System

这是一个基于 YOLOv8/v11 + OpenCV + CLIP 的双目视觉定位与感知偏移分析系统。

## 核心功能

1.  **双目视觉定位**: 使用双目立体匹配 (Stereo SGBM) 和 YOLO 目标检测，计算目标的 3D 坐标 (X, Y, Z)。
2.  **量化知觉偏移**:
    *   **语义偏移分析**: 利用 OpenAI CLIP 模型计算图像与通用标签的相似度，判断观察是否“去标签化”。
    *   **物理特征提取**: 计算图像的纹理熵 (Texture Entropy) 和物理细节密度 (Detail Density)。
    *   **视觉重心预测**: 利用显著性图 (Saliency Map) 分析视觉重心是否从“功能中心”向“美学边缘”迁移。

## 运行

```bash
uv run main.py
```

## 配置

在 `config.py` 中可以调整：
*   摄像头参数 (基线, 焦距)
*   YOLO 模型路径
*   CLIP 目标标签
*   显著性检测方法

## 输出接口

系统每帧输出一个 JSON 对象，包含：
*   **positioning**: 检测到的物体及其 3D 坐标。
*   **perceptual_analysis**:
    *   `semantic_shift`: 语义相似度得分及是否去标签化。
    *   `physical_metrics`: 纹理熵与细节密度。
    *   `visual_attention`: 视觉重心坐标及是否边缘聚焦。

## 注意事项

*   首次运行会自动下载 CLIP 模型和 YOLO 模型。
