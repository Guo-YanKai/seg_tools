#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 11:50
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : loss.py
# @software: PyCharm
import torch
import torch.nn as nn
from utils.target_one_hot import one_hot_1
BCELoss = nn.BCELoss()
BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

CrossEntropyLoss = nn.CrossEntropyLoss()


def flatten(tensor):
    """将输入张量[N,C,D,H,W]==>[C,N*D*H*W]"""
    C = tensor.size(1)
    # axis_order = (1,0,2,3,4)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # tensor:[N,C,D,H,W] ==> [C,N,D,H,W]
    transposed = tensor.permute(axis_order)
    # tensor.view
    return transposed.contiguous().view(C, -1)


def compute_per_channel_dice(output, target, epsilon=1e-6, weight=None):
    """对于多通道的输入和目标计算戴斯系数,假设预测的输入是归一化后的概率
    参数:
    output:(torch.Tensor):[N,C,H,W]
    target:(torch.Tensor):[N,C,H,W]
    epsilon:(float) 防止被除数为0
    weight:(torch.tensor):[C,1] 每个类别的权重
        """
    target = one_hot_1(target) # 进行编码

    assert output.size() == target.size(), "output and target must have the same shape"
    output = flatten(output)
    target = flatten(target)
    target = target.float()
    # 这里用乘法求交集的前提,是模型的预测输出要非0即1,
    # 当和target对应的时候,1的位置得到1,就是交集的位置
    intersect = (output * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect
    # 分母有两种算法:(output+target).sum(-1) or (output^2+target^2).sum(-1)
    denominator = (output * output).sum(-1) + (target * target).sum(-1)

    return 2 * (intersect / denominator.clamp(min=epsilon))


class _AbstractAiceLoss(nn.Module):

    def __init__(self, weight=None, normalization="sigmoid"):
        # 对于模型的输出没有进行归一化的,可以使用sigmoid或者softmax进行归一化.
        super(_AbstractAiceLoss, self).__init__()
        self.register_buffer("weight", weight)

        assert normalization in ["sigmoid", "softmax", "None"]
        if normalization == "sigmoid":
            self.normalization = nn.Sigmoid()
        elif normalization == "softmax":
            self.normalization = nn.Softmax(dim=1)
        else:
            self.normalization = lambda x: x

    def dice(self, output, target, weight):
        """实际的戴斯分数计算,由子类实现"""

    def forward(self, output, target):
        output = self.normalization(output)

        # 计算每个channels的戴斯系数(dice coefficient)
        per_channel_dice = self.dice(output, target, weight=self.weight)

        # 计算所有通道/类别的戴斯分数的平均值
        return 1. - torch.mean(per_channel_dice)


class DiceLoss(_AbstractAiceLoss):
    """二分类的diceLoss是1-DiceCoefficient
    多分类是计算每个通道的DiceLoss,然后对这些值求平均"""

    def __init__(self, weight=None, normalization="softmax"):
        super().__init__(weight, normalization)

    def dice(self, output, target, weight):
        return compute_per_channel_dice(output, target, weight=self.weight)


class GeneralizeDiceLoss(_AbstractAiceLoss):
    def __init__(self, normalization="softmax", epsilon=1e-6):
        super(GeneralizeDiceLoss, self).__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, output, target, weight):
        assert output.size() == target.size(), "output and target must have the same shape"
        output = flatten(output)
        target = flatten(target)
        target = target.float()
        if output.size(0) == 1:
            """使用generalize dice有意义必须要大与两个类别"""
            output = torch.cat((output, 1 - output), dim=0)
            target = torch.cat((target, 1 - target), dim=0)

        # GDL的权重计算:通过其label体积的倒数进行矫正
        w_l = target.sum(-1)
        w_l = 1 / (w_l * w_l).clamp(min=self.epsilon)
        w_l.requires_grad = False
        intersect = (output * target).sum(-1)
        intersect = intersect * w_l

        demoninator = (output + target).sum(-1)
        demoninator = (demoninator * w_l).clamp(min=self.epsilon)
        return 2 * (intersect.sum() / demoninator.sum())


class BCEDiceLoss():
    """
    BCE和DiceLoss的线性组合,alpha * BCE + beta * Dice, alpha, beta
    """

    def __init__(self, alpha, beta):
        c = 0

    def forward(self, pred, target):
        return pred
