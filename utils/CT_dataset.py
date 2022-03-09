#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 11:47
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : dataset.py.py
# @software: PyCharm

from torch.nn import functional as F
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter
from config import args
import logging
from PIL import Image
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
# torchvision中的数据增强
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomResizedCrop, CenterCrop, \
    RandomHorizontalFlip


class BasicDataset(Dataset):
    """二维CT的数据"""

    def __init__(self, data_path, args, scale=1):
        self.images_dirs = os.path.join(data_path, "images")
        self.masks_dirs = os.path.join(data_path, "masks")
        self.scale = scale
        assert 0 < scale <= 1, "Scale must between 0 and 1"

        self.image_names = os.listdir(self.images_dirs)
        logging.info(f"Creating dataset with {len(self.images_dirs)} examples.")

        self.transform = Compose([
            # Resize(0.5),
            # RandomHorizontalFlip(0.5), # 以概率0.5随机水平翻转。
            ToTensor(),  # 归一化为：将取值范围[0,255]的Image图像或(H,W,C)的array ===>【C,H,W】取值范围[0,1.0]的float tensor
            # Normalize(mean=, std=)  # 这里是标准化，是原始数据的均值和方差
        ])

    def __getitem__(self, item):
        name = self.image_names[item]
        image_path = os.path.join(self.images_dirs, name)
        mask_path = os.path.join(self.masks_dirs, name)
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        assert image.size == mask.size, \
            f"Image and mask {name} should have the same size, but are {image.size} and {mask.size}"

        if self.transform:
            image = self.transform(image)
        mask_array = np.array(mask)
        if args.n_labels==2:
            mask_array[mask_array!=0]=1
        # mask_array = np.expand_dims(mask_array, axis=2)
        # mask_array = mask_array.transpose((2,0,1))

        return {"image": image, "mask": torch.from_numpy(mask_array),
                "name": name.split(".")[0]}

    def __len__(self):
        return len(self.image_names)


from config import args
from utils.common import split_data_val

if __name__ == "__main__":
    # 测试数据
    dataset = BasicDataset(args.data_path, args)
    train_sample, val_sample = split_data_val(dataset, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sample)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sample)

    for data in (train_loader):
        print(data["image"].shape)
        print(Counter(np.array(data["mask"].detach()).ravel()))
        break

    # print(data[214]["mask"].shape)
    # img = np.array(data[214]["mask"].detach())
    # print(Counter(img.ravel()))

    # transform = Compose([
    #     # Resize(),
    #     # RandomHorizontalFlip(0.5),  # 以概率0.5随机水平翻转。
    #     ToTensor(),  # 归一化为：将取值范围[0,255]的Image图像或(H,W,C)的array ===>【C,H,W】取值范围[0,1.0]的float tensor
    #     # Normalize(mean=0.5, std=0.5)  # 这里是标准化，是原始数据的均值和方差
    # ])
    # image  = Image.open(r"D:\code\data\cbct\label\1\images\001_215.png")
    # print(image)
    # w, h = image.size
    # pil_image = image.resize((512,512))
    # image_array = np.array(pil_image)
    # print(image_array.shape)
    # image_array = np.expand_dims(image_array, axis=2)
    # print(image_array.shape)
    # img_trans = image_array.transpose((2, 0, 1))
    # print(img_trans.shape)
    #
    # print(img_trans.dtype)
    #
    # img_trans.astype(float)
    # print(img_trans.dtype)
    #
    # img_tensor = torch.from_numpy(img_trans)
    # print(img_tensor)

    # image_array = np.array(image)
    # print(image_array)
    # print("mean:",np.mean(image_array), "std:",np.std(image_array))
    # print(image_array.shape)
    #
    #
    # image_normal = norm_img(image_array)
    # print("norm:image, mean:", np.mean(image_normal), "norm_image std:", np.std(image_normal))
    # print(Counter(image_normal.ravel()))
    #
    #
    # image_trans = transform(image)
    # print(type(image_trans))
    # print(image_trans.shape)
    # print(Counter(np.array(image_trans.detach()).ravel()))
