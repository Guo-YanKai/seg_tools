#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 14:27
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : 3DCT_dataset.py
# @software: PyCharm
from collections import Counter
import os
import sys
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize,ToTensor
from config import args
import logging
import SimpleITK as sitk
import  matplotlib.pyplot as plt
import numpy as np

class Basic3D(Dataset):
    def __init__(self, data_path, args, scale=1):
        self.images_dirs = os.path.join(data_path, "data")
        self.masks_dirs = os.path.join(data_path,"label")
        self.scale = scale
        self.args = args
        self.image_names = os.listdir(self.images_dirs)
        print(f"Creating dataset with {len(self.image_names)} examples.")
        self.transforms = Compose([
            ToTensor()
        ])

    def __getitem__(self, item):
        ct_path = os.path.join(self.images_dirs, self.image_names[item])
        ct = sitk.ReadImage(ct_path, sitk.sitkInt32)
        ct_array = sitk.GetArrayFromImage(ct)
        ct_array = self.norm_img(ct_array)

        seg_path = os.path.join(self.masks_dirs, self.image_names[item])
        seg = sitk.ReadImage(seg_path, sitk.sitkUInt8)
        seg_array = sitk.GetArrayFromImage(seg)

        if self.args.n_labels == 2:
            seg_array[seg_array!=0]=1

        return {"image": torch.from_numpy(ct_array),
                "mask": torch.from_numpy(seg_array),
                "name": self.image_names[item].split(".")[0]}

    @staticmethod
    def norm_img(image):
        # 归一化像素值到（0，1）之间，且将溢出值取边界值
        # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        normal_image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # normal_image[normal_image > 1] = 1.
        # normal_image[normal_image < 0] = 0.
        return normal_image

    def __len__(self):
        return len(self.image_names)


if __name__ =="__main__":
    data_path = r"D:\code\work_code\3DUNet-Pytorch-master\slice_save\8"
    data = Basic3D(data_path, args)
    print(type(data[100]))
    print(data[100]["image"].shape)
    print(data[100]["name"])

    print(Counter(np.array(data[100]["image"][4]).ravel()))
    print(Counter(np.array(data[100]["mask"][4]).ravel()))

    plt.subplot(1,2,1)
    plt.imshow(data[100]["image"][4], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(data[100]["mask"][4], cmap="gray")
    plt.show()

