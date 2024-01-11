# -*- coding:utf-8 -*-
# @Time : 2024-01-08 19:52
# @Author: popfishy
# @File : img_augmentation.py

import albumentations as A
import cv2
import os
from PIL import Image
import numpy as np

# 数据集根目录&输出目录
dataset_root = "/home/yjq/dataset80%/"
output_root = "/home/yjq/dataset_augmentation/"

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

max_num = 1000

transform = A.Compose(
    [
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=20, p=0.5),
        A.HueSaturationValue(),
        # A.RandomCrop(height=1500, width=3000, p=0.5),
    ]
)

category_folder_path = os.path.join(dataset_root, labels[0])
output_folder_path = os.path.join(output_root, labels[0])
os.makedirs(output_folder_path, exist_ok=True)

for category_folder in os.listdir(dataset_root):
    category_folder_path = os.path.join(dataset_root, category_folder)
    output_folder_path = os.path.join(output_root, category_folder)
    os.makedirs(output_folder_path, exist_ok=True)

    if os.path.isdir(category_folder_path):
        print(f"Processing category: {output_folder_path}")
        # 遍历类别文件夹中的图片
        num_cnt = len(os.listdir(category_folder_path))
    for filename in os.listdir(category_folder_path):
        file_path = os.path.join(category_folder_path, filename)
        filename = filename.split(".")[0]
        if os.path.isfile(file_path):
            if num_cnt > max_num:
                continue
            image = Image.open(file_path)
            image = image.convert("RGB")
            image.save(output_folder_path + "/" + filename + ".jpg")
            image = np.array(image)
            for i in range(3):
                output_filename = (
                    output_folder_path + "/" + filename + "_" + str(i) + ".jpg"
                )
                output_filename = os.path.join(
                    output_folder_path, filename + "_" + str(i) + ".jpg"
                )
                transformed = transform(image=image)["image"]
                transformed = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
                cv2.imencode(".jpg", transformed)[1].tofile(output_filename)
                num_cnt = num_cnt + 1
