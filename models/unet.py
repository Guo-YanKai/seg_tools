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
from torchstat import stat
from utils.common import print_models
from tensorboardX import SummaryWriter
from utils.weights_init import init_weights

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
            self.up = nn.ConvTranspose2d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=2,
                                         stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 通过torch.cat连接两个tensor
        # 这里的三行代码主要是针对使用valid卷积的模式
        # 这样,上采样后他分辨率与同一层解码器的分辨率不一致的skip cat的连接
        # diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        # diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        # x1 = F.pad(x1, pad=[diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


class Outc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Outc, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)  # 大小不变

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    """实现最基本的unet结构"""

    def __init__(self, in_channels, n_labels):
        super(Unet, self).__init__()
        self.in_channels = in_channels
        self.n_labels = n_labels

        self.inc = DoubleConv(in_channels=self.in_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=512)

        self.up1 = Up(in_channels=1024, out_channels=256)
        self.up2 = Up(in_channels=512, out_channels=128)
        self.up3 = Up(in_channels=256, out_channels=64)
        self.up4 = Up(in_channels=128, out_channels=64)

        self.outc = Outc(in_channels=64, out_channels=self.n_labels)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init_weights(m,init_type="kaiming")


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

from collections import Counter
import numpy as np
if __name__ == "__main__":
    x = torch.randn((1, 1, 512, 512))
    net = Unet(in_channels=1, n_labels=17)
    # with SummaryWriter("runs_models/unet") as w:
    #     w.add_graph(net, x)
    # print_models(net)
    # print(stat(net, (1,512,512)))
    pred = net(x)
    print(pred.shape)
    # print(torch.min(pred),torch.max(pred))
    # print(Counter(np.array(pred.detach()).ravel()))

    # print(dict(net.named_parameters())["inc.double_conv.0.weight"].shape)
    # for k,v in net.state_dict().items():
    #     print(k)
