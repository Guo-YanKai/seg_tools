#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 16:46
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : unet_nested.py
# @software: PyCharm

import torch
import torch.nn as nn
from torchstat import stat
from utils.common import print_models
from tensorboardX import SummaryWriter
from utils.weights_init import init_weights

class VGGBlock(nn.Module):
    """结构:[conv,bn,relu]*2,[B, in_channels, H, W] ==> [B, out_channels, H, W]"""

    def __init__(self, in_channels, middle_channels,
                 out_channels, act_func=nn.PReLU()):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)
        return out


class Nested_UNet(nn.Module):
    def __init__(self, in_channels, n_labels, deepsupervision=True):
        super(Nested_UNet, self).__init__()
        self.in_channels = in_channels
        self.n_labels = n_labels
        self.deepsupervision = deepsupervision

        nb_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv0_0 = VGGBlock(in_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3] + nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(2 * nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(2 * nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(2 * nb_filter[2] + nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(3 * nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(3 * nb_filter[1] + nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(4 * nb_filter[0] + nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], self.n_labels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], self.n_labels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], self.n_labels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], self.n_labels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], self.n_labels, kernel_size=1)

        #-----initialise weights--------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))  # 通道in_channels=>32,大小变为一半
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))  # 在channels出连接

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))

        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output


if __name__ == "__main__":
    x = torch.randn((3, 1, 512, 512))
    net = Nested_UNet(in_channels=1, n_labels=17, deepsupervision=True)
    # with SummaryWriter("runs_models/nested-unet") as w:
    #     w.add_graph(net, x)
    # print_models(net)
    # print(stat(net, (1, 512, 512)))
    print(net(x)[0].shape)