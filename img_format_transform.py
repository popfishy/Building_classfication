# -*- coding:utf-8 -*-
# @Time : 2024-01-08 16:52
# @Author: popfishy
# @File : img_format_transform.py
# @Func: 将不同格式数据转换为JPEG格式，并将其等比缩放到100W像素左右，再使用改变图像质量进行压缩

from PIL import Image
import os

target_total_pixels = 1000000

# 数据集根目录
dataset_root = "/home/yjq/dataset80%"

# 设置输出目录
output_root = "/home/yjq/dataset_augmentation"

for category_folder in os.listdir(dataset_root):
    category_folder_path = os.path.join(dataset_root, category_folder)
    output_folder_path = os.path.join(output_root, category_folder)
    os.makedirs(output_folder_path, exist_ok=True)

    if os.path.isdir(category_folder_path):
        print(f"Processing category: {category_folder}")
        # if category_folder == "00_负样本":
        # 遍历类别文件夹中的图片
    for filename in os.listdir(category_folder_path):
        file_path = os.path.join(category_folder_path, filename)
        output_filename = os.path.join(output_folder_path, filename)
        if os.path.isfile(file_path):
            image = Image.open(file_path)
            output_filename = os.path.join(output_folder_path, filename)
            if os.path.exists(output_filename):
                continue
            if (image.width * image.height) < target_total_pixels:
                image.save(output_filename, format="JPEG", quality=80)
            else:
                scale_factor = (
                    target_total_pixels / (image.width * image.height)
                ) ** 0.5
                target_width = int(image.width * scale_factor)
                target_height = int(image.height * scale_factor)
                resized_image = image.resize(
                    (target_width, target_height), resample=Image.BILINEAR
                )
                resized_image.save(output_filename, format="JPEG", quality=80)
