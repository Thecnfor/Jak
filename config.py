import os

# 摄像头配置 (示例模拟值)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
FOCAL_LENGTH = 1000.0  # 像素单位
BASELINE = 0.1  # 米 (摄像头之间的距离)

# YOLO 配置
YOLO_MODEL_PATH = "yolov8n.pt"  # 如果可用，也可以是 "yolo11n.pt"
CONFIDENCE_THRESHOLD = 0.5

# CLIP 配置
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# 目标标签，用于检测是否“去标签化”
TARGET_LABELS = ["桌子", "椅子", "人", "建筑", "树", "车", "天空", "抽象", "纹理"]

# 显著性配置
SALIENCY_METHOD = "spectral_residual"  # 或 "fine_grained"

# 输出配置
OUTPUT_FPS = 30
