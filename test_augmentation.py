# -*- coding:utf-8 -*-
# @Time : 2024-01-08 20:23
# @Author: popfishy
# @File : test_augmentation.py

import torchvision
import cv2
import PIL
import numpy as np
import albumentations as A

# Read image using PIL
image = cv2.imread('01_001.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
size  = image.shape
height = size[0] #高度
width = size[1] #宽度
target_width = int(width * 0.8)
target_height = int(height * 0.8)

transform = A.Compose([
    A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.RandomCrop(height=target_height, width=target_width, p=0.5)
])

# Apply transform
for i in range(50):
    transformed = transform(image=image)["image"]
    img2 = cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR)
    cv2.imwrite('pic/' + str(i) + '.jpg', img2)
