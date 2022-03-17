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
import torch.nn.functional as F

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
    target = one_hot_1(target)  # 进行编码

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


# class DiceLoss(_AbstractAiceLoss):
#     """二分类的diceLoss是1-DiceCoefficient
#     多分类是计算每个通道的DiceLoss,然后对这些值求平均"""
#
#     def __init__(self, weight=None, normalization="softmax"):
#         super().__init__(weight, normalization)
#
#     def dice(self, output, target, weight):
#         return compute_per_channel_dice(output, target, weight=self.weight)


# -----------------GeneralizeDiceLoss（待完善）------------
class GeneralizeDiceLoss(_AbstractAiceLoss):
    def __init__(self, normalization="softmax", epsilon=1e-6):
        super(GeneralizeDiceLoss, self).__init__(weight=None, normalization=normalization)
        self.epsilon = epsilon

    def dice(self, output, target, weight):
        target = one_hot_1(target)
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


# ****************DiceLoss****************
class DiceLoss(nn.Module):
    def __init__(self, args, weight=None):
        super(DiceLoss, self).__init__()
        self.args = args
        self.weight = weight

    def _dice_loss(self, pred, target):
        smooth = 1e-5
        intersect = torch.sum(pred * target)
        p_sum = torch.sum(pred * pred)
        y_sum = torch.sum(target * target)
        dice = (2 * intersect + smooth) / (p_sum + y_sum)
        loss = 1 - dice
        return loss

    def forward(self, preds, target, softmax=True):

        N, C, H, W = preds.size()
        preds = preds.reshape(N,C,H*W)
        if softmax:
            preds = torch.softmax(preds, dim=2)
        preds = preds.reshape(N,C,H,W)

        target = one_hot_1(target)
        assert preds.size() == target.size(), "output and target must have the same shape"
        class_wise_dice = []
        loss = 0.0
        for i in range(self.args.n_labels):
            dice = self._dice_loss(preds[:, i], target[:, i])
            class_wise_dice.append(dice.item())
            loss += dice * self.weight[i]
        return loss / self.args.n_labels


# ***************BCEDiceLoss**************
class BCEDiceLoss(nn.Module):
    """
    BCE和DiceLoss的线性组合,alpha * BCE + beta * Dice, alpha, beta
    """

    def __init__(self, args, alpha=1, beta=0.5, weight=None):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.args = args

        self.cre = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss(self.args, weight=weight)

    def forward(self, pred, target):
        loss1 = self.cre(pred, target)
        loss2 = self.dice(pred, target)
        return self.alpha * loss1 + self.beta * loss2


# ---------------FocalLoss------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, inputs, targets):
        print(inputs.shape, targets.shape)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


# -----------------MS-SSIM(多尺度结构相似损失函数--待完善)---------------------
class MSSSIM(nn.Module):
    def __init__(self):
        super(MSSSIM, self).__init__()

    def forward(self, pred, target):
        return 0


if __name__ == "__main__":
    device = torch.device("cuda:0")
    outputs = torch.tensor([[[2, 1., 2.3],
                            [2.5, 1, 1.2],
                            [0.3, 2., 3.4]],
                            [[2, 1., 2.3],
                             [2.5, 1, 1.2],
                             [0.3, 2., 3.4]]
                            ]).reshape(2,1,3,3).to(device)
    targets = torch.tensor([[0, 1, 0],[0, 1, 0]]).to(device)
    print("outputs:",outputs.shape, outputs)

    # N,C,H,W => N,C,H*W
    outputs2 = outputs.reshape(2, 1,-1)
    print(outputs2.shape)

    pred = F.softmax(outputs2, dim=2)
    pred  = pred.reshape(2,1,3,3)
    print("pred:", pred.shape, pred)

    # loss1 = FocalLoss()
    # print(loss1(outputs, targets))
    # loss2 = nn.CrossEntropyLoss()
    # print(loss2(outputs, targets))