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
        self.args = args
        self.scale = scale

        assert 0 < scale <= 1, "Scale must between 0 and 1"

        self.image_names = os.listdir(self.images_dirs)
        logging.info(f"Creating dataset with {len(self.images_dirs)} examples.")

        self.transform = Compose([
            # Resize((128,128)),
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
        if self.args.n_labels==2:
            mask_array[mask_array!=0]=1
        # mask_array = np.expand_dims(mask_array, axis=2)
        # mask_array = mask_array.transpose((2,0,1))

        return {"image": image, "mask": torch.from_numpy(mask_array),
                "name": name.split(".")[0]}

    def __len__(self):
        return len(self.image_names)


from config import args
from utils.common import split_data_val
from utils.target_one_hot import one_hot_1,one_hot_2

if __name__ == "__main__":
    # 测试数据
    dataset = BasicDataset(args.data_path, args)
    train_sample, val_sample = split_data_val(dataset, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sample)
    val_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=val_sample)

    for i, data in enumerate(train_loader):
        print(data["mask"].shape)
        print(Counter(np.array(data["image"].detach()).ravel()))
        # label_one_hot1 = one_hot_1(data["mask"].to(dtype=torch.long), n_labels=args.n_labels)

        # label_one_hot2 = one_hot_2(data["mask"].to(dtype=torch.long), n_labels=args.n_labels)
        # print(label_one_hot1.shape)
        # print(Counter(np.array(label_one_hot1.detach()[:,8,:,:]).ravel()))
        # print(label_one_hot2.shape)
        # print(label_one_hot1.detach()==label_one_hot2.detach())
    #     print(Counter(np.array(data["mask"].detach()).ravel()))
        break


