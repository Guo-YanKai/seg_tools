#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 16:46
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : unet_nested.py
# @software: PyCharm

import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self):
        super(VGGBlock, self).__init__()

    def forward(self, x):



class NestedUNet(nn.Module):
    def __init__(self, in_channels, n_labels, deepsupervision=True):
        super(NestedUNet, self).__init__()
        self.in_channels = in_channels
        self.n_labels = n_labels
        self.deepsupervision = deepsupervision


        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)





    def forward(self,x):

        return x

if __name__ =="__main__":
    x = torch.randn((1, 3, 512, 512))
    pool = nn.MaxPool2d(kernel_size=2, stride=2)
    print(pool(x).shape)