# VitTransfer.py
import time

import cv2
import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel, ViTImageProcessor
from typing import List, Literal, Tuple
from PIL import Image
import numpy as np


class ViTTransfer(nn.Module):
    def __init__(self, num_classes, model_name, freeze_backbone=True):
        super(ViTTransfer, self).__init__()
        config = ViTConfig.from_pretrained(model_name)
        self.backbone = ViTModel(config)
        self.classifier = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_features = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_features)
        return logits


# ========== 新增：全局单例模式 ==========
_global_model = None
_global_processor = None
_global_device = None
_global_num_classes = None
_global_model_name = None
_global_model_path = None


def init_model(num_classes: int,
               model_name: str,
               model_path: str,
               device: Literal["cuda", "cpu"] = "cuda",
               warmup: bool = True):  # 新增 warmup 参数
    global _global_model, _global_processor, _global_device

    print(f"正在加载模型到 {device}...")
    load_start = time.time()

    # 加载模型
    model = ViTTransfer(num_classes, model_name)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    _global_model = model

    # 加载处理器
    processor = ViTImageProcessor.from_pretrained(model_name)
    _global_processor = processor
    _global_device = device

    print(f"模型加载完成，耗时: {time.time() - load_start:.2f}秒")

    # 预热（可选）
    if warmup and device == "cuda":
        print("预热模型中（消除首次推理延迟）...")
        warmup_start = time.time()

        # 创建 dummy 输入
        dummy_img = Image.new('RGB', (224, 224), color='black')
        dummy_input = processor(images=dummy_img, return_tensors="pt")
        dummy_input = dummy_input.to(device)

        # 虚拟推理
        with torch.no_grad():
            _ = model(**dummy_input)

        if device == "cuda":
            torch.cuda.synchronize()

        print(f"预热完成，耗时: {time.time() - warmup_start:.2f}秒")

    print("模型已就绪！")


def predictImage(image_path: str, classes_label: List | str) -> Tuple[str, float]:
    """快速预测（使用已加载的全局模型）"""
    global _global_model, _global_processor, _global_device

    if _global_model is None:
        raise RuntimeError("模型未初始化，请先调用 init_model()")

    # 解析标签
    if isinstance(classes_label, list):
        labels = classes_label
    else:
        with open(classes_label, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

    # 加载并预处理图像
    img = Image.open(image_path).convert('RGB')
    img_tensor = _global_processor(images=img, return_tensors="pt")
    img_tensor = img_tensor.to(_global_device)

    # 预测
    with torch.no_grad():
        output = _global_model(**img_tensor)
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output
        confidence = torch.nn.functional.softmax(logits, dim=1).max(dim=1)[0]
        index = torch.argmax(logits, dim=1).item()

    return labels[index], confidence.item()


def predictCamera(frame: np.ndarray, classes_label: List | str):

    global _global_model, _global_processor, _global_device

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = _global_processor(images=img, return_tensors="pt")
    img_tensor = img_tensor.to(_global_device)

    if isinstance(classes_label, list):
        labels = classes_label
    else:
        with open(classes_label, 'r') as f:
            labels = [line.strip() for line in f.readlines()]


    with torch.no_grad():
        output = _global_model(**img_tensor)
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output
        confidence = torch.nn.functional.softmax(logits, dim=1).max(dim=1)[0]
        index = torch.argmax(logits, dim=1).item()

    return labels[index], confidence.item(), img
