#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 13:50
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : unet.py
# @software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """结构：[convolution => [BN] => ReLU] * 2
    返回结果；原tensor大小不变，通道变为out_channels"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()  # 这句话等价于super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True，对上层传下里的tensor直接修改，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """结构：[maxpool2d] => DoubleConv
    功能：下采样"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            # 使用插值的方式上采样
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            # 使用反卷积的形式上采样
            self.up = nn.ConvTranspose2d(in_channels=out_channels // 2, out_channels=in_channels // 2, kernel_size=2,
                                         stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        return x


class unet(nn.Module):
    """实现最基本的unet结构"""

    def __int__(self, in_channels, out_labels, training=True):
        super(unet, self).__init__()
        self.in_channels = in_channels
        self.out_labels = out_labels
        self.training = training

        self.inc = DoubleConv(in_channels=self.in_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=512)

    def forward(self, x):

        return x


if __name__ == "__main__":
    x = torch.randn((1, 3, 512, 512))
    inc = DoubleConv(3, 64)
    down1 = Down(in_channels=64, out_channels=128)
    down2 = Down(in_channels=128, out_channels=256)
    down3 = Down(in_channels=256, out_channels=512)
    down4 = Down(in_channels=512, out_channels=512)
    x1 = inc(x)
    x2 = down1(x1)
    x3 = down2(x2)
    x4 = down3(x3)
    x5 = down4(x4)

    up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    print("x1:", x1.shape)
    print("x2:", x2.shape)
    print("x3:", x3.shape)
    print("x4:", x4.shape)
    print("x5:", x5.shape)

    x5 = up(x5)
    print("x5_up:", x5.shape)


    diffY = torch.tensor([x4.size()[2]-x5.size()[2]])
    diffX = torch.tensor([x4.size()[3]-x5.size()[3]])
    print(diffY, diffX)

    x6  = torch.cat([x4, x5],dim=1)
    print("x6：",x6.shape)

    # net = Down(3, 64)
    # net = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)  # 就是求卷积的逆操作
    # print(net(x).shape)
