#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/12 10:48
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : Unet3D.py
# @software: PyCharm
import torch
import torch.nn as nn


class DoubelConv3d(nn.Module):
    """结构：【conv3d => BN => ReLU】*2"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DoubelConv3d, self).__init__()
        self.doubel_conv3d = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.doubel_conv3d(x)
        return x


class Down3d(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Down3d, self).__init__()
        self.maxploo3d = nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            DoubelConv3d(in_channels, mid_channels, out_channels)
        )

    def forward(self, x):
        return self.maxploo3d(x)


class UP3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UP3d, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv = DoubelConv3d(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class Unet3D(nn.Module):

    def __init__(self, in_channels, n_labels):
        super(Unet3D, self).__init__()
        self.in_channels = in_channels
        self.n_labels = n_labels

        self.encoder1 = DoubelConv3d(self.in_channels, mid_channels=32, out_channels=64)
        self.encoder2 = Down3d(in_channels=64, mid_channels=64, out_channels=128)
        self.encoder3 = Down3d(in_channels=128, mid_channels=128, out_channels=256)
        self.encoder4 = Down3d(in_channels=256, mid_channels=256, out_channels=512)

        self.decoder1 = UP3d(in_channels=512 + 256, out_channels=256)
        self.decoder2 = UP3d(in_channels=256 + 128, out_channels=128)
        self.decoder3 = UP3d(in_channels=128 + 64, out_channels=64)

        self.decoder4 = nn.Conv3d(in_channels=64, out_channels=self.n_labels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = self.encoder1(x)

        x2 = self.encoder2(x1)

        x3 = self.encoder3(x2)

        x4 = self.encoder4(x3)

        x5 = self.decoder1(x4, x3)

        x6 = self.decoder2(x5, x2)

        x7 = self.decoder3(x6, x1)

        logits = self.decoder4(x7)

        return logits


if __name__ == "__main__":
    x = torch.randn([2, 1, 8, 512, 512])

    net = Unet3D(in_channels=1, n_labels=17)

    print(net(x).shape)
