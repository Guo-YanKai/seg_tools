#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 11:47
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : dataset.py.py
# @software: PyCharm

from torch.nn import functional as F
import os
from config import args
import logging
from PIL import Image
from torch.utils.data import Dataset, SubsetRandomSampler
# torchvision中的数据增强
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomResizedCrop, CenterCrop, RandomHorizontalFlip

class BasicDataset():
    """二维CT的数据"""
    def __init__(self, data_path, scale=1):
        self.images_dirs = os.path.join(data_path, "images")
        self.masks_dirs = os.path.join(data_path, "masks")
        self.scale = scale
        assert 0<scale<=1, "Scale must between 0 and 1"

        self.image_names = os.listdir(self.images_dirs)
        logging.info(f"Creating dataset with {len(self.images_dirs)} examples.")
        self.transform  = Compose([
            Resize(0.5),
            RandomHorizontalFlip(0.5), # 以概率0.5随机水平翻转。
            ToTensor(), # 归一化为：将取值范围[0,255]的Image图像或(H,W,C)的array转换为【C,H,W】取值范围[0,1.0]的float tensor
            # Normalize(mean=, std=)  # 这里是标准化，是原始数据的均值和方差
        ])


    def __getitem__(self, item):
        name = self.image_names[item]
        image_path = os.path.join(self.images_dirs, name)
        mask_path = os.path.join(self.masks_dirs, name)
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        print("image:")
        print(type(image))
        print(image)

        print("mask:")
        print(type(mask))


        return 0

    def __len__(self):
        return len(self.image_names)


if __name__ == "__main__":
    # 测试数据
    data = BasicDataset(args.data_path)
    print(len(data))
    print(data[0])
