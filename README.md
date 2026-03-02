# Jetson Orin Nano Binocular Vision Center

这是一个基于 Jetson Orin Nano 的双目视觉中枢与感知大脑，集成 YOLOv8/v11 + OpenCV + CLIP 实现高精度三维定位与环境感知。

## 核心功能

1.  **双目视觉定位 (3D Localization)**:
    *   作为视觉感知大脑的核心，利用双目立体匹配 (Stereo SGBM) 结合 YOLO 目标检测，实时解算目标的三维空间坐标 (X, Y, Z)，实现精准的空间定位。
2.  **视觉注意力与特征感知**:
    *   **显著性分析**: 利用显著性图 (Saliency Map) 捕捉场景中的视觉热点。
    *   **特征提取**: 辅助计算图像的纹理熵与细节密度，为分析视觉重心（如从功能区向边缘细节的迁移）提供数据支持，保留对环境美学特征的感知能力。

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
