import os
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel

def download_models():
    print("正在下载 YOLO 模型...")
    model = YOLO("yolov8n.pt")
    # 如果不存在，这将触发下载
    print("YOLO 模型准备就绪。")

    print("正在下载 CLIP 模型...")
    model_name = "openai/clip-vit-base-patch32"
    CLIPModel.from_pretrained(model_name)
    CLIPProcessor.from_pretrained(model_name)
    print("CLIP 模型准备就绪。")

if __name__ == "__main__":
    download_models()
