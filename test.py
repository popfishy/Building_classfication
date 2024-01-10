# -*- coding:utf-8 -*-
# @Time : 2024-01-09 19:34
# @Author: popfishy
# @File : eval.py

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import torch.utils.data as tud
import numpy as np
from PIL import Image
import os
from model.inception_resnet_v2 import Inception_ResNetv2

model_path = "Inception_ResNetv2_40.pth"
datas_path = "test_data"

target_total_pixels = 1000000

input_size = 299

labels = [
    "00_负样本",
    "01_天河大楼",
    "02_体育馆",
    "03_航院主楼",
    "04_01教学楼",
    "05_02教学楼",
    "06_03教学楼",
    "07_图书馆",
    "08_东跨线桥",
    "09_西跨线桥",
    "10_游泳馆",
    "11_博士生楼",
    "12_俱乐部",
    "13_银河大楼",
    "14_老图书馆",
    "15_三院1号楼（主楼）",
    "16_三院2号楼（老楼）",
    "17_海天楼",
    "18_四院主楼",
    "19_北斗",
    "20_校主楼",
]

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(300),  # 调整图像短边为256像素
        torchvision.transforms.CenterCrop(input_size),  # 将图像剪裁为224x224像素
        # torchvision.transforms.Resize(input_size),
        torchvision.transforms.ToTensor(),
    ]
)

# 加载模型
model = Inception_ResNetv2()
model.load_state_dict(torch.load(model_path))
model.eval()
test_pic = []
for filename in os.listdir(datas_path):
    test_pic.append(filename)
test_pic = sorted(test_pic)
for filename in test_pic:
    file_path = os.path.join(datas_path, filename)
    if os.path.isfile(file_path):
        image = Image.open(file_path)
        # 加载图像并进行预处理
        if (image.width * image.height) > target_total_pixels:
            scale_factor = (target_total_pixels / (image.width * image.height)) ** 0.5
            target_width = int(image.width * scale_factor)
            target_height = int(image.height * scale_factor)
            image = image.resize((target_width, target_height), resample=Image.BILINEAR)
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        # 进行预测
        with torch.no_grad():
            outputs = model(image_tensor)
            outputs = torch.sigmoid(outputs)
            predicted_value, predicted = torch.max(outputs, 1)
            if predicted_value.item() < 0.4:
                predicted = 0
        # 打印预测结果
        print(
            "图片名称:{}, 预测结果:{}, 预测概率值:{:.4f}, 预测标签为:{}".format(
                filename,
                predicted.item(),
                predicted_value.item(),
                labels[predicted.item()],
            )
        )
