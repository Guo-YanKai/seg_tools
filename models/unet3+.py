#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/16 10:50
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : unet3+.py
# @software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.weights_init import init_weights


class DoubelConv(nn.Module):
    def __init__(self, in_channels, out_channels, n=2, is_batchnorm=True,
                 kernels_size=3, stride=1, padding=1):
        super(DoubelConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.kernels_size = kernels_size
        self.stride = stride
        self.padding = padding

        if is_batchnorm:
            for i in range(1, self.n + 1):
                conv = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, self.kernels_size,
                                               self.stride, self.padding),
                                     nn.BatchNorm2d(self.out_channels))
                setattr(self, "conv%d" % i, conv)
                self.in_channels = self.out_channels
        else:
            for i in range(1, self.n + 1):
                conv = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, self.kernels_size,
                                               self.stride, self.padding))
                setattr(self, "conv%d" % i, conv)

        # -------初始化子模块参数----------
        for m in self.children():
            # self.children只包括网络模块的第一代儿子模块
            # self.models()时按照深度优先遍历的方式，存储了net的所有模块
            init_weights(m, init_type="kaiming")

    def forward(self, x):
        for i in range(1, self.n + 1):
            conv = getattr(self, "conv%d" % i)
            x = conv(x)
        return x


class Unet3Plus(nn.Module):
    def __init__(self, in_channels, n_labels, bilinear=True,
                 feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Unet3Plus, self).__init__()
        self.in_channels = in_channels
        self.n_labels = n_labels

        # ------------------encoder----------------
        self.conv1 = DoubelConv(in_channels=self.in_channels, out_channels=64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = DoubelConv(in_channels=64, out_channels=128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = DoubelConv(in_channels=128, out_channels=256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = DoubelConv(in_channels=256, out_channels=512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = DoubelConv(in_channels=512, out_channels=1024)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        # -----------------decoder-------------------
        # 每一个解码器层都融合了编码器中的小尺度和同尺度的特征图，以及大尺度的特征图

        # stage 4d  ceil_model表示向上取整
        # 64*256*256 =>64*32*32
        self.h1_PT_hd4 = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(64)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # 128*128*128 => 64*32*32
        self.h2_PT_hd4 = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(64)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # 256*64*64 => 64*32*32
        self.h3_PT_hd4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(64)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # 512*32*32 => 64*32*32
        self.h4_Cat_hd4_conv = nn.Conv2d(512, 64, kernel_size=3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(64)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # 1024*16*16 => 64*32*32
        self.h5_UT_hd4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.h5_UT_hd4_conv = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.h5_UT_hd4_bn = nn.BatchNorm2d(64)
        self.h5_UT_hd4_relu = nn.ReLU(inplace=True)

        # 融合（h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, h5_UT_hd4）
        self.conv4d_1 = nn.Conv2d(320, 320, kernel_size=3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(320)
        self.relu4d_1 = nn.ReLU(inplace=True)

        # -------stage3d-----------
        # 64*256*256 = 64*64*64
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(64)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # 128*128*128 => 64**64*64
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(64)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # 256*64*64 => 64*64*64
        self.h3_Cat_hd3_conv = nn.Conv2d(256, 64, kernel_size=3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(64)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # 512*32*32 => 64*64*64
        self.h4_UT_hd3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.h4_UT_hd3_conv = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.h4_UT_hd3_bn = nn.BatchNorm2d(64)
        self.h4_UT_hd3_relu = nn.ReLU(inplace=True)

        # 1024*16*16 => 64*64*64
        self.h5_UT_hd3 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.h5_UT_hd3_conv = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.h5_UT_hd3_bn = nn.BatchNorm2d(64)
        self.h5_UT_hd3_relu = nn.ReLU(inplace=True)

        # 融合（h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, h4_UP_hd3, h5_UT_hd3）
        self.conv3d_1 = nn.Conv2d(320, 320, kernel_size=3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(320)
        self.relu3d_1 = nn.ReLU(inplace=True)

        # -------------stage2d-------------
        # 64*256*256 => 64*128*128
        self.h1_PT_hd2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(64)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # 128*128*128 => 64*128*128
        self.h2_Cat_hd2_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(64)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # 256*64*64=>64*128*128
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.hd3_UT_hd2_conv = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(64)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # 512*32*32 => 64*128*128
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.hd4_UT_hd2_conv = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(64)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # 1024*16*16 => 64*128*128
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.hd5_UT_hd2_conv = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(64)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # 融合（h1_PT_hd2,h2_Cat_hd2,hd3_UT_hd2, hd4_UT_hd2,hd5_UT_hd2）
        self.conv2d_1 = nn.Conv2d(320, 320, kernel_size=3, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(320)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # ------------stage1-------------
        # 64*256*256 => 64*256*256
        self.h1_Cat_hd1_conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(64)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # 320*128*128 => 64*256*256
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.hd2_UT_hd1_conv = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(64)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # 320*64*64 =>64*256*256
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        self.hd3_UT_hd1_conv = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(64)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # 320*32*32 => 64*256*256
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True)
        self.hd4_UT_hd1_conv = nn.Conv2d(320, 64, kernel_size=3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(64)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # 1024*16*16 => 64*256*256
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True)
        self.hd5_UT_hd1_conv = nn.Conv2d(1024, 64, kernel_size=3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(64)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # 融合(h1_Cat_hd1,hd2_UT_hd1,hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(320, 320, kernel_size=3, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(320)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # ---------output---------
        self.outUp1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.outconv1 = nn.Conv2d(320, self.n_labels, kernel_size=3, padding=1)

        # initalise weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool1(x1)

        x2 = self.conv2(x1)
        x2 = self.maxpool2(x2)

        x3 = self.conv3(x2)
        x3 = self.maxpool3(x3)

        x4 = self.conv4(x3)
        x4 = self.maxpool4(x4)

        x5 = self.conv5(x4)
        x5 = self.maxpool5(x5)

        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(x1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(x2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(x3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(x4)))
        h5_UT_hd4 = self.h5_UT_hd4_relu(self.h5_UT_hd4_bn(self.h5_UT_hd4_conv(self.h5_UT_hd4(x5))))

        hd4 = self.relu4d_1(
            self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, h5_UT_hd4), dim=1))))

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(x1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(x2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(x3)))
        h4_UP_hd3 = self.h4_UT_hd3_relu(self.h4_UT_hd3_bn(self.h4_UT_hd3_conv(self.h4_UT_hd3(hd4))))
        h5_UP_hd3 = self.h5_UT_hd3_relu(self.h5_UT_hd3_bn(self.h5_UT_hd3_conv(self.h5_UT_hd3(x5))))
        hd3 = self.relu3d_1(
            self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, h4_UP_hd3, h5_UP_hd3), dim=1)))
        )

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(x1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(x2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(x5))))
        hd2 = self.relu2d_1(
            self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), dim=1)))
        )


        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_bn(x1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(x5))))
        hd1 = self.relu1d_1(
            self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), dim=1)))
        )

        output = self.outconv1(self.outUp1(hd1))
        return output


class Unet3Plus_DeepSup(nn.Module):
    def __init__(self, in_channels, n_labels):
        super(Unet3Plus_DeepSup, self).__init__()
        self.in_channels = in_channels
        self.n_labels = n_labels
        filters = [64, 128, 256, 512, 1024]

        #-------------encoder--------------
        self.conv1 = DoubelConv(self.in_channels, filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = DoubelConv(filters[0], filters[1])
        self.maxpool2= nn.MaxPool2d(kernel_size=2)

        self.conv3 = DoubelConv(filters[1], filters[2])
        self.maxpool3= nn.MaxPool2d(kernel_size=2)

        self.conv4 = DoubelConv(filters[2], filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = DoubelConv(filters[3], filters[4])
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320, hd4->40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd4->40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd4->40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40, hd4->40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320, hd3->80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd3->80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80, hd3->80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd4->80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd4->80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear',align_corners=True)  # 14*14
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        '''stage 2d '''
        # h1->320*320, hd2->160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160, hd2->160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd2->160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd2->160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd2->160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        '''stage 1d'''
        # h1->320*320, hd1->320*320, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160, hd1->320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)  # 14*14
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80, hd1->320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 14*14
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40, hd1->320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear',align_corners=True)  # 14*14
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20, hd1->320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)  # 14*14
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        # -------------Bilinear Upsampling--------------
        self.upscore5 = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        self.upscore4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
        self.upscore3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upscore2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upscore1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, self.n_labels, kernel_size=3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, self.n_labels, kernel_size=3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, self.n_labels, kernel_size=3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, self.n_labels, kernel_size=3, padding=1)
        self.outconv5 = nn.Conv2d(filters[4], self.n_labels, kernel_size=3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, x):
        x1 = self.conv1(x)
        x1= self.maxpool1(x1)

        x2 = self.conv2(x1)
        x2 = self.maxpool2(x2)

        x3 = self.conv3(x2)
        x3 = self.maxpool3(x3)

        x4 = self.conv4(x3)
        x4 = self.maxpool4(x4)

        x5 = self.conv5(x4)
        hd5 = self.maxpool5(x5)

        ## -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(x1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(x2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(x3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(x4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))  # hd4->40*40*UpChannels

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(x1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(x2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(x3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))  # hd3->80*80*UpChannels

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(x1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(x2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(
            torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))  # hd2->160*160*UpChannels

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(x1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(
            torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))  # hd1->320*320*UpChannels

        d5 = self.upscore5(self.outconv5(hd5))
        d4 = self.upscore4(self.outconv4(hd4))
        d3 = self.upscore3(self.outconv3(hd3))
        d2 = self.upscore2(self.outconv2(hd2))
        d1 = self.upscore1(self.outconv1(hd1))

        return  d1, d2, d3, d4, d5

from torchstat import stat
from utils.common import print_models
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    x = torch.randn((2, 1, 512, 512))

    net = Unet3Plus_DeepSup(1, 17)
    print(net(x)[-1].shape)

    with SummaryWriter("runs_models/unet3+") as w:
        w.add_graph(net, x)
    print_models(net)
    print(stat(net, (1, 512, 512)))


